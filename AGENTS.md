# Work Summary (2026-02)

## API + Backend Features
- Added VoiceDesign endpoints: `POST /v1/audio/voice-design` and `GET /v1/audio/voice-design/capabilities`.
- Added voice-design schemas: `VoiceDesignRequest`, `VoiceDesignCapabilities`.
- Added VoiceDesign support in `api/backends/official_qwen3_tts.py`.
- Added voice clone cache in `api/backends/official_qwen3_tts.py` with size/TTL env vars and cache-clear endpoint.
- Added multi-model backend: `api/backends/multi_model_qwen3_tts.py` (speech/base/voice_design routing).
- Added multi-GPU backend: `api/backends/multi_gpu_qwen3_tts.py` (round-robin per role).

## GPU Selection + Mapping
- Multi-GPU selection supports `TTS_GPU_DEVICE_IDS` values `all/auto` or list.
- Visible GPU detection prefers `torch.cuda.device_count()`; falls back to parsing visibility env vars.
- GPU remap handles `CUDA_VISIBLE_DEVICES` and `NVIDIA_VISIBLE_DEVICES` so physical IDs map to container ordinals.
- Docker compose now uses `NVIDIA_VISIBLE_DEVICES=0,2` with `TTS_GPU_DEVICE_IDS=all` (use only visible GPUs).

## UI + Health
- Updated demo UI (`api/static/index.html`) to include VoiceClone + VoiceDesign tests and show multi-GPU device list in status.
- `/health` includes device list when multi-GPU backend is used.

## Security + Routing
- Root `/` returns 204; admin UI moved to `/admin`.

## Tests
- Added `tests/test_voice_design_live.py` and extended `tests/test_api.py` for new endpoints and cache clear.

## Notes
- Multi-GPU backend uses round-robin per role (`speech/base/voice_design`).
- `WORKERS>1` increases concurrency but each worker loads its own model copies (VRAM impact).
- Qwen3 TTS generate now accepts `eos_token_id` as `int` or `list[int]`; suppress_tokens and truncation respect all EOS IDs.
