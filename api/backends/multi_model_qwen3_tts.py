# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Multi-model backend that loads multiple official Qwen3-TTS models.

This backend enables a single server to support:
- CustomVoice (speech)
- Base (voice clone)
- VoiceDesign (voice design)
"""

import logging
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

from .base import TTSBackend
from .official_qwen3_tts import OfficialQwen3TTSBackend

logger = logging.getLogger(__name__)


class MultiModelQwen3TTSBackend(TTSBackend):
    """Multi-model backend that routes to the appropriate model."""

    def __init__(
        self,
        speech_model_name: Optional[str] = None,
        base_model_name: Optional[str] = None,
        voice_design_model_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.speech_backend = (
            OfficialQwen3TTSBackend(model_name=speech_model_name)
            if speech_model_name
            else None
        )
        self.base_backend = (
            OfficialQwen3TTSBackend(model_name=base_model_name)
            if base_model_name
            else None
        )
        self.voice_design_backend = (
            OfficialQwen3TTSBackend(model_name=voice_design_model_name)
            if voice_design_model_name
            else None
        )
        self._ready = False

    async def initialize(self) -> None:
        if self._ready:
            logger.info("Multi-model backend already initialized")
            return

        for backend in (
            self.speech_backend,
            self.base_backend,
            self.voice_design_backend,
        ):
            if backend is not None and not backend.is_ready():
                await backend.initialize()

        self._ready = True

    async def generate_speech(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        if self.speech_backend is None:
            raise RuntimeError(
                "Speech generation is not configured in multi-model mode"
            )
        return await self.speech_backend.generate_speech(
            text=text,
            voice=voice,
            language=language,
            instruct=instruct,
            speed=speed,
        )

    def get_backend_name(self) -> str:
        return "official_multi"

    def get_model_id(self) -> str:
        parts = []
        if self.speech_backend is not None:
            parts.append(f"speech={self.speech_backend.get_model_id()}")
        if self.base_backend is not None:
            parts.append(f"base={self.base_backend.get_model_id()}")
        if self.voice_design_backend is not None:
            parts.append(f"voice_design={self.voice_design_backend.get_model_id()}")
        return ";".join(parts) if parts else "unknown"

    def get_supported_voices(self) -> List[str]:
        if self.speech_backend is None:
            return []
        return self.speech_backend.get_supported_voices()

    def get_supported_languages(self) -> List[str]:
        if self.speech_backend is None:
            return []
        return self.speech_backend.get_supported_languages()

    def is_ready(self) -> bool:
        if not self._ready:
            return False
        for backend in (
            self.speech_backend,
            self.base_backend,
            self.voice_design_backend,
        ):
            if backend is not None and not backend.is_ready():
                return False
        return True

    def get_device_info(self) -> Dict[str, Any]:
        for backend in (
            self.speech_backend,
            self.base_backend,
            self.voice_design_backend,
        ):
            if backend is not None:
                return backend.get_device_info()
        return {
            "device": None,
            "gpu_available": False,
            "gpu_name": None,
            "vram_total": None,
            "vram_used": None,
        }

    def supports_voice_cloning(self) -> bool:
        if self.base_backend is None:
            return False
        return self.base_backend.supports_voice_cloning()

    def voice_clone_cache_enabled(self) -> bool:
        if self.base_backend is None:
            return False
        return self.base_backend.voice_clone_cache_enabled()

    def clear_voice_clone_cache(self) -> int:
        if self.base_backend is None:
            return 0
        return self.base_backend.clear_voice_clone_cache()

    def get_model_type(self) -> str:
        if self.base_backend is None:
            return "unknown"
        return self.base_backend.get_model_type()

    def supports_voice_design(self) -> bool:
        if self.voice_design_backend is None:
            return False
        return self.voice_design_backend.supports_voice_design()

    def get_voice_design_model_type(self) -> str:
        if self.voice_design_backend is None:
            return "unknown"
        return self.voice_design_backend.get_voice_design_model_type()

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
        voice_clone_prompt: Optional[List[Any]] = None,
    ) -> Tuple[np.ndarray, int]:
        if self.base_backend is None:
            raise RuntimeError("Voice cloning is not configured in multi-model mode")
        return await self.base_backend.generate_voice_clone(
            text=text,
            ref_audio=ref_audio,
            ref_audio_sr=ref_audio_sr,
            ref_text=ref_text,
            language=language,
            x_vector_only_mode=x_vector_only_mode,
            speed=speed,
            deterministic=deterministic,
            voice_clone_prompt=voice_clone_prompt,
        )

    async def create_voice_clone_prompt(
        self,
        ref_audio: np.ndarray,
        ref_audio_sr: int,
        ref_text: Optional[str] = None,
        x_vector_only_mode: bool = False,
    ) -> List[Any]:
        if self.base_backend is None:
            raise RuntimeError("Voice cloning is not configured in multi-model mode")
        return await self.base_backend.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_audio_sr=ref_audio_sr,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
        )

    async def generate_voice_design(
        self,
        text: str,
        instruct: Optional[str] = None,
        language: str = "Auto",
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        if self.voice_design_backend is None:
            raise RuntimeError("Voice design is not configured in multi-model mode")
        return await self.voice_design_backend.generate_voice_design(
            text=text,
            instruct=instruct,
            language=language,
            speed=speed,
        )
