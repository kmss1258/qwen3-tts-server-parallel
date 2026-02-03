# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Official Qwen3-TTS backend implementation.

This backend uses the official Qwen3-TTS Python implementation
from the qwen_tts package.
"""

import hashlib
import logging
import os
import time
from collections import OrderedDict
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from .base import TTSBackend

logger = logging.getLogger(__name__)

# Optional librosa import for speed adjustment
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class OfficialQwen3TTSBackend(TTSBackend):
    """Official Qwen3-TTS backend using the qwen_tts package."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device: Optional[str] = None,
    ):
        """
        Initialize the official backend.

        Args:
            model_name: HuggingFace model identifier
        """
        super().__init__()
        self.model_name = model_name
        self.device_override = device
        self._ready = False
        self._voice_clone_cache = OrderedDict()
        self._voice_clone_cache_size = int(
            os.getenv("TTS_VOICE_CLONE_CACHE_SIZE", "32")
        )
        self._voice_clone_cache_ttl = float(os.getenv("TTS_VOICE_CLONE_CACHE_TTL", "0"))
        self._max_new_tokens = self._read_env_int("TTS_MAX_NEW_TOKENS", 1024)

    @staticmethod
    def _read_env_int(name: str, default: int, min_value: int = 1) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError:
            logger.warning("Invalid %s=%s; using %s", name, raw, default)
            return default
        if value < min_value:
            logger.warning("Invalid %s=%s; using %s", name, raw, default)
            return default
        return value

    async def initialize(self) -> None:
        """Initialize the backend and load the model."""
        if self._ready:
            logger.info("Official backend already initialized")
            return

        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            # Determine device
            if self.device_override:
                if self.device_override == "cpu":
                    self.device = "cpu"
                    self.dtype = torch.float32
                else:
                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA is not available for requested device")
                    self.device = self.device_override
                    self.dtype = torch.bfloat16
            else:
                if torch.cuda.is_available():
                    self.device = "cuda:0"
                    self.dtype = torch.bfloat16
                else:
                    self.device = "cpu"
                    self.dtype = torch.float32

            logger.info(
                f"Loading Qwen3-TTS model '{self.model_name}' on {self.device}..."
            )

            # Try loading with Flash Attention 2, fallback to SDPA or eager if not supported
            # (e.g., RTX 5090/Blackwell GPUs don't have pre-built flash-attn wheels yet)
            attn_implementations = ["flash_attention_2", "sdpa", "eager"]
            model_loaded = False

            last_error = None
            for attn_impl in attn_implementations:
                try:
                    logger.info(f"Attempting to load model with attention: {attn_impl}")
                    self.model = Qwen3TTSModel.from_pretrained(
                        self.model_name,
                        device_map=self.device,
                        dtype=self.dtype,
                        attn_implementation=attn_impl,
                    )
                    logger.info(f"Successfully loaded model with {attn_impl} attention")
                    model_loaded = True
                    break
                except Exception as attn_error:
                    last_error = attn_error
                    logger.warning(f"Could not load with {attn_impl}: {attn_error}")
                    if attn_impl != attn_implementations[-1]:
                        logger.info("Falling back to next attention implementation...")

            if not model_loaded:
                # If GPU loading failed completely, try CPU as last resort
                if self.device != "cpu":
                    logger.warning(
                        "All GPU attention implementations failed. Falling back to CPU..."
                    )
                    self.device = "cpu"
                    self.dtype = torch.float32
                    try:
                        self.model = Qwen3TTSModel.from_pretrained(
                            self.model_name,
                            device_map=self.device,
                            dtype=self.dtype,
                            attn_implementation="eager",
                        )
                        logger.info(
                            "Successfully loaded model on CPU (GPU not compatible)"
                        )
                        model_loaded = True
                    except Exception as cpu_error:
                        raise RuntimeError(f"Failed to load model on CPU: {cpu_error}")
                else:
                    raise RuntimeError(
                        f"Failed to load model with any attention implementation. Last error: {last_error}"
                    )

            # Apply torch.compile() optimization for faster inference
            if torch.cuda.is_available() and hasattr(torch, "compile"):
                logger.info("Applying torch.compile() optimization...")
                try:
                    # Compile the model with reduce-overhead mode for faster inference
                    self.model.model = torch.compile(
                        self.model.model,
                        mode="reduce-overhead",  # Optimize for inference speed
                        fullgraph=False,  # Allow graph breaks for compatibility
                    )
                    logger.info("torch.compile() optimization applied successfully")
                except Exception as e:
                    logger.warning(f"Could not apply torch.compile(): {e}")

            # Enable cuDNN benchmarking for optimal convolution algorithms
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                logger.info("Enabled cuDNN benchmark mode")

            # Enable TF32 for faster matmul on Ampere+ GPUs (RTX 30xx/40xx)
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled TF32 precision for faster matmul")

            self._ready = True
            logger.info(
                f"Official Qwen3-TTS backend loaded successfully on {self.device}"
            )

        except Exception as e:
            logger.error(f"Failed to load official TTS backend: {e}")
            raise RuntimeError(f"Failed to initialize official TTS backend: {e}")

    async def generate_speech(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech from text using the official Qwen3-TTS model.

        Args:
            text: The text to synthesize
            voice: Voice name to use
            language: Language code
            instruct: Optional instruction for voice style
            speed: Speech speed multiplier

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if not self._ready:
            await self.initialize()

        try:
            # Generate speech
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language=language,
                speaker=voice,
                instruct=instruct,
            )

            audio = wavs[0]

            # Apply speed adjustment if needed
            if speed != 1.0 and LIBROSA_AVAILABLE:
                audio = librosa.effects.time_stretch(
                    audio.astype(np.float32), rate=speed
                )
            elif speed != 1.0:
                logger.warning("Speed adjustment requested but librosa not available")

            return audio, sr

        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            raise RuntimeError(f"Speech generation failed: {e}")

    def get_backend_name(self) -> str:
        """Return the name of this backend."""
        return "official"

    def get_model_id(self) -> str:
        """Return the model identifier."""
        return self.model_name

    def get_supported_voices(self) -> List[str]:
        """Return list of supported voice names."""
        if not self._ready or not self.model:
            # Return default voices when model is not loaded
            return ["Vivian", "Ryan", "Sophia", "Isabella", "Evan", "Lily"]

        try:
            if hasattr(self.model.model, "get_supported_speakers"):
                speakers = self.model.model.get_supported_speakers()
                if speakers:
                    return list(speakers)
        except Exception as e:
            logger.warning(f"Could not get speakers from model: {e}")

        # Fallback to default voices
        return ["Vivian", "Ryan", "Sophia", "Isabella", "Evan", "Lily"]

    def get_supported_languages(self) -> List[str]:
        """Return list of supported language names."""
        if not self._ready or not self.model:
            # Return default languages when model is not loaded
            return [
                "English",
                "Chinese",
                "Japanese",
                "Korean",
                "German",
                "French",
                "Spanish",
                "Russian",
                "Portuguese",
                "Italian",
            ]

        try:
            if hasattr(self.model.model, "get_supported_languages"):
                languages = self.model.model.get_supported_languages()
                if languages:
                    return list(languages)
        except Exception as e:
            logger.warning(f"Could not get languages from model: {e}")

        # Fallback to default languages
        return [
            "English",
            "Chinese",
            "Japanese",
            "Korean",
            "German",
            "French",
            "Spanish",
            "Russian",
            "Portuguese",
            "Italian",
        ]

    def is_ready(self) -> bool:
        """Return whether the backend is initialized and ready."""
        return self._ready

    def get_device_info(self) -> Dict[str, Any]:
        """Return device information."""
        info = {
            "device": str(self.device) if self.device else "unknown",
            "gpu_available": False,
            "gpu_name": None,
            "vram_total": None,
            "vram_used": None,
        }

        try:
            import torch

            if torch.cuda.is_available():
                info["gpu_available"] = True
                device_idx = torch.cuda.current_device()
                if isinstance(self.device, str) and self.device.startswith("cuda:"):
                    try:
                        device_idx = int(self.device.split(":", 1)[1])
                    except ValueError:
                        device_idx = torch.cuda.current_device()

                if device_idx >= 0:
                    info["gpu_name"] = torch.cuda.get_device_name(device_idx)

                    # Get VRAM info
                    props = torch.cuda.get_device_properties(device_idx)
                    info["vram_total"] = f"{props.total_memory / 1024**3:.2f} GB"

                    if self._ready:
                        allocated = torch.cuda.memory_allocated(device_idx)
                        info["vram_used"] = f"{allocated / 1024**3:.2f} GB"
        except Exception as e:
            logger.warning(f"Could not get device info: {e}")

        return info

    def supports_voice_cloning(self) -> bool:
        """
        Check if this backend supports voice cloning.

        Voice cloning requires the Base model (Qwen3-TTS-12Hz-1.7B-Base).
        The CustomVoice model does not support voice cloning.
        """
        # Check if we're using the Base model (not CustomVoice)
        return "Base" in self.model_name and "CustomVoice" not in self.model_name

    def supports_voice_design(self) -> bool:
        """
        Check if this backend supports voice design.

        Voice design requires the VoiceDesign model (Qwen3-TTS-12Hz-1.7B-VoiceDesign).
        """
        return "VoiceDesign" in self.model_name

    def get_model_type(self) -> str:
        """Return the model type (base or customvoice)."""
        if "Base" in self.model_name:
            return "base"
        elif "CustomVoice" in self.model_name:
            return "customvoice"
        return "unknown"

    def get_voice_design_model_type(self) -> str:
        """Return the model type for voice design."""
        if self.supports_voice_design():
            return "voice_design"
        return "unknown"

    async def generate_voice_clone(
        self,
        text: str,
        ref_audio: Optional[np.ndarray] = None,
        ref_audio_sr: Optional[int] = None,
        ref_text: Optional[str] = None,
        language: str = "Auto",
        x_vector_only_mode: bool = False,
        speed: float = 1.0,
        deterministic: bool = False,
        max_new_tokens: Optional[int] = None,
        voice_clone_prompt: Optional[list] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech by cloning a voice from reference audio.

        Args:
            text: The text to synthesize
            ref_audio: Reference audio as numpy array
            ref_audio_sr: Sample rate of reference audio
            ref_text: Transcript of reference audio (required for ICL mode)
            language: Language code (e.g., "English", "Chinese", "Auto")
            x_vector_only_mode: If True, use x-vector only (no ref_text needed)
            speed: Speech speed multiplier (0.25 to 4.0)
            deterministic: If True, disable sampling for deterministic output
            max_new_tokens: Maximum number of new codec tokens to generate
            voice_clone_prompt: Precomputed prompt items for voice cloning

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if not self._ready:
            await self.initialize()

        if not self.supports_voice_cloning():
            raise RuntimeError(
                "Voice cloning requires the Base model (Qwen3-TTS-12Hz-1.7B-Base). "
                "The current model does not support voice cloning."
            )

        try:
            prompt_items = voice_clone_prompt
            if prompt_items is None:
                if ref_audio is None or ref_audio_sr is None:
                    raise RuntimeError(
                        "ref_audio is required when voice_clone_prompt is not provided"
                    )
                ref_text_for_prompt = None if x_vector_only_mode else ref_text
                cache_key = self._build_voice_clone_cache_key(
                    ref_audio,
                    ref_audio_sr,
                    ref_text_for_prompt,
                    x_vector_only_mode,
                )

                if cache_key is not None:
                    prompt_items = self._get_voice_clone_cache(cache_key)

                if prompt_items is None:
                    prompt_items = self.model.create_voice_clone_prompt(
                        ref_audio=(ref_audio, ref_audio_sr),
                        ref_text=ref_text_for_prompt,
                        x_vector_only_mode=x_vector_only_mode,
                    )

                    if cache_key is not None:
                        self._put_voice_clone_cache(cache_key, prompt_items)

            generate_kwargs = {}
            resolved_max_new_tokens = (
                max_new_tokens if max_new_tokens is not None else self._max_new_tokens
            )
            if resolved_max_new_tokens is not None:
                generate_kwargs["max_new_tokens"] = resolved_max_new_tokens
            if deterministic:
                generate_kwargs.update(
                    do_sample=False,
                    top_k=1,
                    top_p=1.0,
                    temperature=1.0,
                    subtalker_dosample=False,
                    subtalker_top_k=1,
                    subtalker_top_p=1.0,
                    subtalker_temperature=1.0,
                )

            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt_items,
                **generate_kwargs,
            )

            audio = wavs[0]

            # Apply speed adjustment if needed
            if speed != 1.0 and LIBROSA_AVAILABLE:
                audio = librosa.effects.time_stretch(
                    audio.astype(np.float32), rate=speed
                )
            elif speed != 1.0:
                logger.warning("Speed adjustment requested but librosa not available")

            return audio, sr

        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            raise RuntimeError(f"Voice cloning failed: {e}")

    def voice_clone_cache_enabled(self) -> bool:
        return self._voice_clone_cache_enabled()

    def clear_voice_clone_cache(self) -> int:
        count = len(self._voice_clone_cache)
        self._voice_clone_cache.clear()
        if count:
            logger.info("Voice clone cache cleared: %s entries", count)
        return count

    def _voice_clone_cache_enabled(self) -> bool:
        return self._voice_clone_cache_size > 0

    def _build_voice_clone_cache_key(
        self,
        ref_audio: np.ndarray,
        ref_audio_sr: int,
        ref_text: Optional[str],
        x_vector_only_mode: bool,
    ) -> Optional[str]:
        if not self._voice_clone_cache_enabled():
            return None
        if not isinstance(ref_audio, np.ndarray):
            return None

        audio = np.asarray(ref_audio, dtype=np.float32)
        audio = np.ascontiguousarray(audio)
        hasher = hashlib.sha256()
        hasher.update(audio.tobytes())
        hasher.update(b"|")
        hasher.update(str(ref_audio_sr).encode("utf-8"))
        hasher.update(b"|")
        hasher.update(b"1" if x_vector_only_mode else b"0")
        if ref_text is not None:
            hasher.update(b"|")
            hasher.update(ref_text.encode("utf-8"))
        return hasher.hexdigest()

    def _get_voice_clone_cache(self, cache_key: str):
        entry = self._voice_clone_cache.get(cache_key)
        if entry is None:
            logger.info("Voice clone cache miss: %s", cache_key[:8])
            return None

        timestamp, prompt_items = entry
        if self._voice_clone_cache_ttl > 0:
            if time.time() - timestamp > self._voice_clone_cache_ttl:
                self._voice_clone_cache.pop(cache_key, None)
                logger.info("Voice clone cache expired: %s", cache_key[:8])
                return None

        self._voice_clone_cache.move_to_end(cache_key)
        logger.info("Voice clone cache hit: %s", cache_key[:8])
        return prompt_items

    def _put_voice_clone_cache(self, cache_key: str, prompt_items) -> None:
        self._voice_clone_cache[cache_key] = (time.time(), prompt_items)
        self._voice_clone_cache.move_to_end(cache_key)
        while len(self._voice_clone_cache) > self._voice_clone_cache_size:
            self._voice_clone_cache.popitem(last=False)

    async def create_voice_clone_prompt(
        self,
        ref_audio: np.ndarray,
        ref_audio_sr: int,
        ref_text: Optional[str] = None,
        x_vector_only_mode: bool = False,
    ) -> list:
        if not self._ready:
            await self.initialize()

        if not self.supports_voice_cloning():
            raise RuntimeError(
                "Voice cloning requires the Base model (Qwen3-TTS-12Hz-1.7B-Base). "
                "The current model does not support voice cloning."
            )

        ref_text_for_prompt = None if x_vector_only_mode else ref_text
        cache_key = self._build_voice_clone_cache_key(
            ref_audio,
            ref_audio_sr,
            ref_text_for_prompt,
            x_vector_only_mode,
        )

        if cache_key is not None:
            prompt_items = self._get_voice_clone_cache(cache_key)
            if prompt_items is not None:
                return prompt_items

        prompt_items = self.model.create_voice_clone_prompt(
            ref_audio=(ref_audio, ref_audio_sr),
            ref_text=ref_text_for_prompt,
            x_vector_only_mode=x_vector_only_mode,
        )

        if cache_key is not None:
            self._put_voice_clone_cache(cache_key, prompt_items)

        return prompt_items

    async def generate_voice_design(
        self,
        text: str,
        instruct: Optional[str] = None,
        language: str = "Auto",
        speed: float = 1.0,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech using the voice design model.

        Args:
            text: The text to synthesize
            instruct: Natural-language description of the desired voice/style
            language: Language code (e.g., "English", "Chinese", "Auto")
            speed: Speech speed multiplier (0.25 to 4.0)
            max_new_tokens: Maximum number of new codec tokens to generate

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if not self._ready:
            await self.initialize()

        if not self.supports_voice_design():
            raise RuntimeError(
                "Voice design requires the VoiceDesign model (Qwen3-TTS-12Hz-1.7B-VoiceDesign). "
                "The current model does not support voice design."
            )

        try:
            generate_kwargs = {}
            resolved_max_new_tokens = (
                max_new_tokens if max_new_tokens is not None else self._max_new_tokens
            )
            if resolved_max_new_tokens is not None:
                generate_kwargs["max_new_tokens"] = resolved_max_new_tokens

            wavs, sr = self.model.generate_voice_design(
                text=text,
                instruct=instruct or "",
                language=language,
                **generate_kwargs,
            )

            audio = wavs[0]

            if speed != 1.0 and LIBROSA_AVAILABLE:
                audio = librosa.effects.time_stretch(
                    audio.astype(np.float32), rate=speed
                )
            elif speed != 1.0:
                logger.warning("Speed adjustment requested but librosa not available")

            return audio, sr

        except Exception as e:
            logger.error(f"Voice design failed: {e}")
            raise RuntimeError(f"Voice design failed: {e}")
