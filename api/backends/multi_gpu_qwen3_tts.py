# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU backend that loads multiple official Qwen3-TTS models."""

import logging
import os
import threading
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

from .base import TTSBackend
from .official_qwen3_tts import OfficialQwen3TTSBackend

logger = logging.getLogger(__name__)


def _build_gpu_id_mapping() -> Dict[int, int]:
    """
    Build a mapping from physical GPU IDs to container-visible indices.

    When CUDA_VISIBLE_DEVICES=0,2 is set, inside the container:
    - cuda:0 = physical GPU 0
    - cuda:1 = physical GPU 2

    This function returns {0: 0, 2: 1} so we can remap user-specified IDs.
    """
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible.strip():
        cuda_visible = os.getenv("NVIDIA_VISIBLE_DEVICES", "")
    if not cuda_visible.strip():
        return {}

    mapping = {}
    for idx, gpu_id_str in enumerate(cuda_visible.split(",")):
        gpu_id_str = gpu_id_str.strip()
        if gpu_id_str:
            try:
                physical_id = int(gpu_id_str)
                mapping[physical_id] = idx
            except ValueError:
                pass
    return mapping


class MultiGpuMultiModelBackend(TTSBackend):
    """Multi-model backend that loads Base/VoiceDesign per GPU and routes by round-robin."""

    def __init__(
        self,
        device_ids: List[int],
        speech_model_name: Optional[str] = None,
        base_model_name: Optional[str] = None,
        voice_design_model_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.device_ids = device_ids
        self._ready = False
        self._rr_lock = threading.Lock()
        self._rr_index = {
            "speech": 0,
            "base": 0,
            "voice_design": 0,
        }

        self._gpu_id_mapping = _build_gpu_id_mapping()
        if self._gpu_id_mapping:
            logger.info(
                "Detected CUDA_VISIBLE_DEVICES remapping: %s", self._gpu_id_mapping
            )

        self.speech_backends = self._build_backends(speech_model_name, "speech")
        self.base_backends = self._build_backends(base_model_name, "base")
        self.voice_design_backends = self._build_backends(
            voice_design_model_name, "voice_design"
        )

    def _resolve_device(self, device_id: int) -> str:
        """
        Resolve a device ID to the actual CUDA device string.

        If CUDA_VISIBLE_DEVICES is set, remap physical IDs to container indices.
        """
        if self._gpu_id_mapping:
            if device_id in self._gpu_id_mapping:
                container_idx = self._gpu_id_mapping[device_id]
                logger.debug(
                    "Remapping physical GPU %d -> container cuda:%d",
                    device_id,
                    container_idx,
                )
                return f"cuda:{container_idx}"
            else:
                if device_id < len(self._gpu_id_mapping):
                    logger.debug(
                        "Using device_id %d as container index directly", device_id
                    )
                    return f"cuda:{device_id}"
                else:
                    logger.warning(
                        "GPU ID %d not found in CUDA_VISIBLE_DEVICES mapping %s, "
                        "using as-is (may fail)",
                        device_id,
                        self._gpu_id_mapping,
                    )
                    return f"cuda:{device_id}"
        else:
            return f"cuda:{device_id}"

    def _build_backends(self, model_name: Optional[str], role: str):
        if not model_name:
            return []
        backends = []
        for device_id in self.device_ids:
            device = self._resolve_device(device_id)
            backends.append(
                OfficialQwen3TTSBackend(model_name=model_name, device=device)
            )
        logger.info("Initialized %s backends for role=%s", len(backends), role)
        return backends

    def _next_backend(self, role: str, backends):
        if not backends:
            raise RuntimeError(f"No backends configured for role={role}")
        with self._rr_lock:
            idx = self._rr_index[role] % len(backends)
            self._rr_index[role] = idx + 1
        return backends[idx]

    async def initialize(self) -> None:
        if self._ready:
            logger.info("Multi-GPU multi-model backend already initialized")
            return

        for backend in (
            self.speech_backends + self.base_backends + self.voice_design_backends
        ):
            if not backend.is_ready():
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
        backend = self._next_backend("speech", self.speech_backends)
        return await backend.generate_speech(
            text=text,
            voice=voice,
            language=language,
            instruct=instruct,
            speed=speed,
        )

    async def generate_voice_clone(
        self,
        text: str,
        ref_audio: np.ndarray,
        ref_audio_sr: int,
        ref_text: Optional[str] = None,
        language: str = "Auto",
        x_vector_only_mode: bool = False,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        backend = self._next_backend("base", self.base_backends)
        return await backend.generate_voice_clone(
            text=text,
            ref_audio=ref_audio,
            ref_audio_sr=ref_audio_sr,
            ref_text=ref_text,
            language=language,
            x_vector_only_mode=x_vector_only_mode,
            speed=speed,
        )

    async def generate_voice_design(
        self,
        text: str,
        instruct: Optional[str] = None,
        language: str = "Auto",
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        backend = self._next_backend("voice_design", self.voice_design_backends)
        return await backend.generate_voice_design(
            text=text,
            instruct=instruct,
            language=language,
            speed=speed,
        )

    def get_backend_name(self) -> str:
        return "official_multi_gpu"

    def get_model_id(self) -> str:
        parts = []
        for role, backends in (
            ("speech", self.speech_backends),
            ("base", self.base_backends),
            ("voice_design", self.voice_design_backends),
        ):
            for backend in backends:
                parts.append(f"{role}@{backend.get_device_info().get('device')}")
        return ";".join(parts) if parts else "unknown"

    def get_supported_voices(self) -> List[str]:
        if not self.speech_backends:
            return []
        return self.speech_backends[0].get_supported_voices()

    def get_supported_languages(self) -> List[str]:
        if not self.speech_backends:
            return []
        return self.speech_backends[0].get_supported_languages()

    def is_ready(self) -> bool:
        if not self._ready:
            return False
        for backend in (
            self.speech_backends + self.base_backends + self.voice_design_backends
        ):
            if not backend.is_ready():
                return False
        return True

    def get_device_info(self) -> Dict[str, Any]:
        devices = []
        for role, backends in (
            ("speech", self.speech_backends),
            ("base", self.base_backends),
            ("voice_design", self.voice_design_backends),
        ):
            for backend in backends:
                info = backend.get_device_info().copy()
                info["role"] = role
                info["model_id"] = backend.get_model_id()
                devices.append(info)

        gpu_available = any(d.get("gpu_available") for d in devices)

        return {
            "device": "multi",
            "gpu_available": gpu_available,
            "gpu_name": None,
            "vram_total": None,
            "vram_used": None,
            "devices": devices,
        }

    def supports_voice_cloning(self) -> bool:
        if not self.base_backends:
            return False
        return self.base_backends[0].supports_voice_cloning()

    def voice_clone_cache_enabled(self) -> bool:
        if not self.base_backends:
            return False
        return self.base_backends[0].voice_clone_cache_enabled()

    def clear_voice_clone_cache(self) -> int:
        cleared = 0
        for backend in self.base_backends:
            cleared += backend.clear_voice_clone_cache()
        return cleared

    def get_model_type(self) -> str:
        if not self.base_backends:
            return "unknown"
        return self.base_backends[0].get_model_type()

    def supports_voice_design(self) -> bool:
        if not self.voice_design_backends:
            return False
        return self.voice_design_backends[0].supports_voice_design()

    def get_voice_design_model_type(self) -> str:
        if not self.voice_design_backends:
            return "unknown"
        return self.voice_design_backends[0].get_voice_design_model_type()
