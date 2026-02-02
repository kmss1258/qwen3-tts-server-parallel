# ruff: noqa: F811
import io
import time
from base64 import b64encode
import json
import os
import urllib.request
from pathlib import Path

import pytest
import soundfile as sf


RUN_LIVE = os.environ.get("VOICE_CLONE_RUN_LIVE") == "1"
BASE_URL = os.environ.get("VOICE_CLONE_BASE_URL") or "http://n.kami.live:19160"
OUT_DIR = Path(
    os.environ.get(
        "VOICE_CLONE_OUT_DIR",
        str((Path(__file__).parent / "voice_clone_outputs").resolve()),
    )
)

REF_AUDIO_PATH = Path(__file__).parent / "민결희.mp3"
REF_TEXT = "그니까, 그 낭만이 약간 미쳤어요."
INPUT_TEXT = REF_TEXT
DETERMINISTIC_TEXT = "안녕."
REF_AUDIO_MAX_SECONDS = float(
    os.environ.get("VOICE_CLONE_REF_AUDIO_MAX_SECONDS", "1.0")
)


def _request_json(
    url: str,
    payload: dict,
    timeout: int = 120,
) -> tuple[int, dict, bytes]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return resp.status, headers, resp.read()
    except urllib.error.HTTPError as err:
        headers = {k.lower(): v for k, v in err.headers.items()}
        return err.code, headers, err.read()


def _get_json(url: str, timeout: int = 30, retries: int = 6, delay: int = 5) -> dict:
    last_err: Exception | None = None
    for _ in range(retries):
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            return json.loads(body)
        except Exception as err:
            last_err = err
            time.sleep(delay)
    if last_err is not None:
        raise last_err
    raise RuntimeError("Failed to fetch JSON")


def _load_ref_audio_b64() -> str:
    audio, sr = sf.read(REF_AUDIO_PATH)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    max_samples = int(sr * REF_AUDIO_MAX_SECONDS)
    if max_samples > 0:
        audio = audio[:max_samples]
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    return b64encode(buffer.getvalue()).decode("ascii")


def _create_prompt_b64(x_vector_only_mode: bool) -> str:
    ref_audio_b64 = _load_ref_audio_b64()
    payload = {
        "ref_audio": ref_audio_b64,
        "x_vector_only_mode": x_vector_only_mode,
    }
    if not x_vector_only_mode:
        payload["ref_text"] = REF_TEXT
    status, headers, body = _request_json(
        f"{BASE_URL}/v1/audio/voice-clone/prompt", payload
    )
    assert status == 200, body.decode("utf-8", errors="replace")
    assert len(body) > 0
    return b64encode(body).decode("ascii")


@pytest.fixture(scope="module")
def voice_clone_capabilities():
    if not RUN_LIVE:
        pytest.skip("Set VOICE_CLONE_RUN_LIVE=1 to run live voice clone tests.")
    capabilities = _get_json(
        f"{BASE_URL}/v1/audio/voice-clone/capabilities",
        timeout=60,
        retries=8,
        delay=10,
    )
    if not capabilities.get("supported"):
        pytest.skip(f"Voice cloning not supported: {capabilities}")
    return capabilities


def test_voice_clone_xvector_only_true(voice_clone_capabilities):
    ref_audio_b64 = _load_ref_audio_b64()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "input": INPUT_TEXT,
        "ref_audio": ref_audio_b64,
        "ref_text": None,
        "x_vector_only_mode": True,
        "language": "Korean",
        "response_format": "mp3",
        "speed": 1.0,
    }
    status, headers, body = _request_json(f"{BASE_URL}/v1/audio/voice-clone", payload)
    assert status == 200, body.decode("utf-8", errors="replace")
    assert "voice_clone" in headers.get("content-disposition", "")
    assert len(body) > 0
    (OUT_DIR / "voice_clone_xvector_true.mp3").write_bytes(body)


def test_voice_clone_xvector_only_false(voice_clone_capabilities):
    ref_audio_b64 = _load_ref_audio_b64()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "input": INPUT_TEXT,
        "ref_audio": ref_audio_b64,
        "ref_text": REF_TEXT,
        "x_vector_only_mode": False,
        "language": "Korean",
        "response_format": "mp3",
        "speed": 1.0,
    }
    status, headers, body = _request_json(f"{BASE_URL}/v1/audio/voice-clone", payload)
    assert status == 200, body.decode("utf-8", errors="replace")
    assert "voice_clone" in headers.get("content-disposition", "")
    assert len(body) > 0
    (OUT_DIR / "voice_clone_xvector_false.mp3").write_bytes(body)


def test_voice_clone_prompt_xvector_deterministic(voice_clone_capabilities):
    prompt_b64 = _create_prompt_b64(x_vector_only_mode=True)
    payload = {
        "input": DETERMINISTIC_TEXT,
        "voice_prompt_file": prompt_b64,
        "x_vector_only_mode": True,
        "deterministic": True,
        "language": "Korean",
        "response_format": "wav",
        "speed": 1.0,
    }
    status1, headers1, body1 = _request_json(
        f"{BASE_URL}/v1/audio/voice-clone", payload, timeout=300
    )
    assert status1 == 200, body1.decode("utf-8", errors="replace")
    status2, headers2, body2 = _request_json(
        f"{BASE_URL}/v1/audio/voice-clone", payload, timeout=300
    )
    assert status2 == 200, body2.decode("utf-8", errors="replace")
    assert body1 == body2


def test_voice_clone_prompt_icl_deterministic(voice_clone_capabilities):
    prompt_b64 = _create_prompt_b64(x_vector_only_mode=False)
    payload = {
        "input": DETERMINISTIC_TEXT,
        "voice_prompt_file": prompt_b64,
        "x_vector_only_mode": False,
        "deterministic": True,
        "language": "Korean",
        "response_format": "wav",
        "speed": 1.0,
    }
    status1, headers1, body1 = _request_json(
        f"{BASE_URL}/v1/audio/voice-clone", payload, timeout=300
    )
    assert status1 == 200, body1.decode("utf-8", errors="replace")
    status2, headers2, body2 = _request_json(
        f"{BASE_URL}/v1/audio/voice-clone", payload, timeout=300
    )
    assert status2 == 200, body2.decode("utf-8", errors="replace")
    assert body1 == body2
