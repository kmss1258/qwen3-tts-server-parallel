import json
import os
import urllib.request
from pathlib import Path

import pytest


RUN_LIVE = os.environ.get("VOICE_DESIGN_RUN_LIVE") == "1"
BASE_URL = os.environ.get("VOICE_DESIGN_BASE_URL") or "http://n.kami.live:19160"
OUT_DIR = Path(
    os.environ.get(
        "VOICE_DESIGN_OUT_DIR",
        str((Path(__file__).parent / "voice_design_outputs").resolve()),
    )
)

INPUT_TEXT = "자기야... 오늘 나 집에 혼자 있는데... 빨리 넣어 줘.. 부탁이야."
INSTRUCT = "Use a sultry and stimulating voice of a high-tone adult woman in her 20s, speaking provocatively."


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


@pytest.fixture(scope="module")
def voice_design_capabilities():
    if not RUN_LIVE:
        pytest.skip("Set VOICE_DESIGN_RUN_LIVE=1 to run live voice design tests.")
    capabilities = _get_json(f"{BASE_URL}/v1/audio/voice-design/capabilities")
    if not capabilities.get("supported"):
        pytest.skip(f"Voice design not supported: {capabilities}")
    return capabilities


def test_voice_design_basic(voice_design_capabilities):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "input": INPUT_TEXT,
        "instruct": INSTRUCT,
        "language": "Korean",
        "response_format": "mp3",
        "speed": 1.0,
    }
    status, headers, body = _request_json(f"{BASE_URL}/v1/audio/voice-design", payload)
    assert status == 200, body.decode("utf-8", errors="replace")
    assert "voice_design" in headers.get("content-disposition", "")
    assert len(body) > 0
    (OUT_DIR / "voice_design_basic.mp3").write_bytes(body)
