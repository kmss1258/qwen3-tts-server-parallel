# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Base class for TTS backends.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Any
import numpy as np


class TTSBackend(ABC):
    """Abstract base class for TTS backends."""

    def __init__(self):
        """Initialize the backend."""
        self.model = None
        self.device = None
        self.dtype = None

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the backend and load the model.

        This method should:
        - Load the model
        - Set up device and dtype
        - Perform any necessary warmup
        """
        pass

    @abstractmethod
    async def generate_speech(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech from text.

        Args:
            text: The text to synthesize
            voice: Voice name/identifier to use
            language: Language code (e.g., "English", "Chinese", "Auto")
            instruct: Optional instruction for voice style/emotion
            speed: Speech speed multiplier (0.25 to 4.0)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """Return the name of this backend."""
        pass

    @abstractmethod
    def get_model_id(self) -> str:
        """Return the model identifier."""
        pass

    @abstractmethod
    def get_supported_voices(self) -> List[str]:
        """Return list of supported voice names."""
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Return list of supported language names."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Return whether the backend is initialized and ready."""
        pass

    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """
        Return device information.

        Returns:
            Dict with keys: device, gpu_available, gpu_name, vram_total, vram_used
        """
        pass

    def supports_voice_cloning(self) -> bool:
        """
        Return whether the backend supports voice cloning.

        Voice cloning requires the Base model (Qwen3-TTS-12Hz-1.7B-Base).
        The CustomVoice model does not support voice cloning.

        Returns:
            True if voice cloning is supported, False otherwise
        """
        return False

    def voice_clone_cache_enabled(self) -> bool:
        """Return whether voice clone prompt caching is enabled."""
        return False

    def clear_voice_clone_cache(self) -> int:
        """Clear any voice clone prompt cache and return number of entries removed."""
        return 0

    def supports_voice_design(self) -> bool:
        """
        Return whether the backend supports voice design.

        Voice design requires the VoiceDesign model (Qwen3-TTS-12Hz-1.7B-VoiceDesign).

        Returns:
            True if voice design is supported, False otherwise
        """
        return False

    def get_voice_design_model_type(self) -> str:
        """Return the model type for voice design (if supported)."""
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
        voice_clone_prompt: Optional[List[Any]] = None,
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
            voice_clone_prompt: Precomputed prompt items for voice cloning

        Returns:
            Tuple of (audio_array, sample_rate)

        Raises:
            NotImplementedError: If voice cloning is not supported by this backend
        """
        raise NotImplementedError("Voice cloning is not supported by this backend")

    async def generate_voice_design(
        self,
        text: str,
        instruct: Optional[str] = None,
        language: str = "Auto",
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech using the voice design model.

        Args:
            text: The text to synthesize
            instruct: Natural-language description of the desired voice/style
            language: Language code (e.g., "English", "Chinese", "Auto")
            speed: Speech speed multiplier (0.25 to 4.0)

        Returns:
            Tuple of (audio_array, sample_rate)

        Raises:
            NotImplementedError: If voice design is not supported by this backend
        """
        raise NotImplementedError("Voice design is not supported by this backend")

    async def create_voice_clone_prompt(
        self,
        ref_audio: np.ndarray,
        ref_audio_sr: int,
        ref_text: Optional[str] = None,
        x_vector_only_mode: bool = False,
    ) -> List[Any]:
        """
        Create a reusable voice clone prompt from reference audio.

        Args:
            ref_audio: Reference audio as numpy array
            ref_audio_sr: Sample rate of reference audio
            ref_text: Transcript of reference audio (required for ICL mode)
            x_vector_only_mode: If True, use x-vector only (no ref_text needed)

        Returns:
            List of prompt items usable as voice_clone_prompt

        Raises:
            NotImplementedError: If voice cloning is not supported by this backend
        """
        raise NotImplementedError("Voice cloning is not supported by this backend")
