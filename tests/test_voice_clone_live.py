import base64
import json
import os
import urllib.request
from pathlib import Path

import pytest


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


def _request_json(url: str, payload: dict) -> tuple[int, dict, bytes]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return resp.status, headers, resp.read()
    except urllib.error.HTTPError as err:
        headers = {k.lower(): v for k, v in err.headers.items()}
        return err.code, headers, err.read()


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _load_ref_audio_b64() -> str:
    return base64.b64encode(REF_AUDIO_PATH.read_bytes()).decode("ascii")


@pytest.fixture(scope="module")
def voice_clone_capabilities():
    if not RUN_LIVE:
        pytest.skip("Set VOICE_CLONE_RUN_LIVE=1 to run live voice clone tests.")
    capabilities = _get_json(f"{BASE_URL}/v1/audio/voice-clone/capabilities")
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
