# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Factory for creating TTS backend instances.
"""

import os
import logging
from typing import Optional

from .base import TTSBackend
from .official_qwen3_tts import OfficialQwen3TTSBackend
from .multi_model_qwen3_tts import MultiModelQwen3TTSBackend
from .multi_gpu_qwen3_tts import MultiGpuMultiModelBackend
from .vllm_omni_qwen3_tts import VLLMOmniQwen3TTSBackend

logger = logging.getLogger(__name__)

# Global backend instance
_backend_instance: Optional[TTSBackend] = None


def get_backend() -> TTSBackend:
    """
    Get or create the global TTS backend instance.

    The backend is selected based on the TTS_BACKEND environment variable:
    - "official" (default): Use official Qwen3-TTS implementation
    - "vllm_omni": Use vLLM-Omni for faster inference

    Returns:
        TTSBackend instance
    """
    global _backend_instance

    if _backend_instance is not None:
        return _backend_instance

    # Get backend type from environment
    backend_type = os.getenv("TTS_BACKEND", "official").lower()

    # Get model name from environment (optional override)
    model_name = os.getenv("TTS_MODEL_NAME")
    multi_model_enabled = os.getenv("TTS_MULTI_MODEL", "false").lower() == "true"
    multi_gpu_enabled = os.getenv("TTS_MULTI_GPU", "false").lower() == "true"
    gpu_device_ids = os.getenv("TTS_GPU_DEVICE_IDS", "").strip()

    logger.info(f"Initializing TTS backend: {backend_type}")

    if backend_type == "official":
        # Official backend
        if multi_model_enabled and multi_gpu_enabled:

            def resolve_visible_device_ids() -> list[int]:
                try:
                    import torch
                except Exception as exc:
                    raise RuntimeError(
                        "Torch is required to detect visible GPUs"
                    ) from exc

                count = torch.cuda.device_count()
                if count > 0:
                    return list(range(count))

                cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
                if cuda_visible.strip():
                    tokens = [
                        token.strip()
                        for token in cuda_visible.split(",")
                        if token.strip()
                    ]
                    if tokens:
                        return list(range(len(tokens)))

                raise RuntimeError("No visible CUDA devices detected")

            def resolve_device_ids(value: str) -> list[int]:
                if value.strip().lower() in ("", "all", "auto"):
                    return resolve_visible_device_ids()
                device_ids = []
                for item in value.split(","):
                    item = item.strip()
                    if not item:
                        continue
                    try:
                        device_ids.append(int(item))
                    except ValueError:
                        raise ValueError(f"Invalid GPU device id: {item}")
                if not device_ids:
                    return resolve_visible_device_ids()
                return device_ids

            def resolve_model(
                env_name: str, default_value: Optional[str]
            ) -> Optional[str]:
                value = os.getenv(env_name)
                if value is None:
                    return default_value
                if value.strip().lower() in ("", "none", "null"):
                    return None
                return value

            speech_model_name = resolve_model(
                "TTS_SPEECH_MODEL_NAME",
                model_name or "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            )
            base_model_name = resolve_model(
                "TTS_BASE_MODEL_NAME",
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            )
            voice_design_model_name = resolve_model(
                "TTS_VOICE_DESIGN_MODEL_NAME",
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            )

            resolved_device_ids = resolve_device_ids(gpu_device_ids)
            _backend_instance = MultiGpuMultiModelBackend(
                device_ids=resolved_device_ids,
                speech_model_name=speech_model_name,
                base_model_name=base_model_name,
                voice_design_model_name=voice_design_model_name,
            )
            logger.info(
                "Using multi-GPU multi-model backend with models: "
                f"speech={speech_model_name}, base={base_model_name}, voice_design={voice_design_model_name}, "
                f"devices={resolved_device_ids}"
            )
        elif multi_model_enabled:

            def resolve_model(
                env_name: str, default_value: Optional[str]
            ) -> Optional[str]:
                value = os.getenv(env_name)
                if value is None:
                    return default_value
                if value.strip().lower() in ("", "none", "null"):
                    return None
                return value

            speech_model_name = resolve_model(
                "TTS_SPEECH_MODEL_NAME",
                model_name or "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            )
            base_model_name = resolve_model(
                "TTS_BASE_MODEL_NAME",
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            )
            voice_design_model_name = resolve_model(
                "TTS_VOICE_DESIGN_MODEL_NAME",
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            )

            _backend_instance = MultiModelQwen3TTSBackend(
                speech_model_name=speech_model_name,
                base_model_name=base_model_name,
                voice_design_model_name=voice_design_model_name,
            )
            logger.info(
                "Using multi-model official backend with models: "
                f"speech={speech_model_name}, base={base_model_name}, voice_design={voice_design_model_name}"
            )
        else:
            if model_name:
                _backend_instance = OfficialQwen3TTSBackend(model_name=model_name)
            else:
                # Use default CustomVoice model
                _backend_instance = OfficialQwen3TTSBackend()

            logger.info(
                f"Using official Qwen3-TTS backend with model: {_backend_instance.get_model_id()}"
            )

    elif (
        backend_type == "vllm_omni"
        or backend_type == "vllm-omni"
        or backend_type == "vllm"
    ):
        if multi_model_enabled:
            raise ValueError(
                "TTS_MULTI_MODEL is only supported with TTS_BACKEND=official"
            )
        if multi_gpu_enabled:
            raise ValueError(
                "TTS_MULTI_GPU is only supported with TTS_BACKEND=official"
            )
        # vLLM-Omni backend
        if model_name:
            _backend_instance = VLLMOmniQwen3TTSBackend(model_name=model_name)
        else:
            # Use 1.7B model for best quality/speed tradeoff
            _backend_instance = VLLMOmniQwen3TTSBackend(
                model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
            )

        logger.info(
            f"Using vLLM-Omni backend with model: {_backend_instance.get_model_id()}"
        )

    else:
        logger.error(f"Unknown backend type: {backend_type}")
        raise ValueError(
            f"Unknown TTS_BACKEND: {backend_type}. "
            f"Supported values: 'official', 'vllm_omni'"
        )

    return _backend_instance


async def initialize_backend(warmup: bool = False) -> TTSBackend:
    """
    Initialize the backend and optionally perform warmup.

    Args:
        warmup: Whether to run a warmup inference

    Returns:
        Initialized TTSBackend instance
    """
    backend = get_backend()

    # Initialize the backend
    await backend.initialize()

    # Perform warmup if requested
    if warmup:
        warmup_enabled = os.getenv("TTS_WARMUP_ON_START", "false").lower() == "true"
        if warmup_enabled:
            logger.info("Performing backend warmup...")
            try:
                # Run a simple warmup generation
                await backend.generate_speech(
                    text="Hello, this is a warmup test.",
                    voice="Vivian",
                    language="English",
                )
                logger.info("Backend warmup completed successfully")
            except Exception as e:
                logger.warning(f"Backend warmup failed (non-critical): {e}")

    return backend


def reset_backend() -> None:
    """Reset the global backend instance (useful for testing)."""
    global _backend_instance
    _backend_instance = None
