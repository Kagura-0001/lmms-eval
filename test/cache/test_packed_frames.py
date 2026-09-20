"""Portable media invariants, independent of candidate model availability."""

import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from lmms_eval.models.model_utils.packed_frames import FORMAT, pack_index, read_frames
from lmms_eval.protocol import ChatMessages


def pack(root: Path) -> Path:
    root.mkdir()
    (root / "frames.mjpeg").write_bytes(b"firstsecond")
    (root / "manifest.json").write_text(
        json.dumps({"format": FORMAT, "frames": {"file": "frames.mjpeg", "bytes": 11, "count": 2, "index": [{"timestamp_seconds": 0, "offset": 0, "length": 5}, {"timestamp_seconds": 0.5, "offset": 5, "length": 6}]}})
    )
    return root


def test_relocation_and_exact_timestamp(tmp_path):
    root = pack(tmp_path / "old")
    assert read_frames(root, [0.5, 0]) == [b"second", b"first"]
    new = tmp_path / "new"
    root.rename(new)
    assert read_frames(new, [0]) == [b"first"]
    with pytest.raises(ValueError, match="not cached"):
        read_frames(new, [0.25])


def test_truncated_pack_rejected(tmp_path):
    root = pack(tmp_path / "pack")
    (root / "frames.mjpeg").write_bytes(b"short")
    with pytest.raises(ValueError, match="Incomplete"):
        pack_index(str(root))


def test_data_url_preserves_mime_and_bytes():
    url = "data:image/jpeg;base64,/9j/2Q=="
    messages = ChatMessages(messages=[{"role": "user", "content": [{"type": "image", "url": url}]}])
    for convert in (messages.to_openai_messages, messages.to_qwen3_vl_openai_messages):
        assert convert()[0]["content"][0]["image_url"]["url"] == url


def test_fetch_resumes_and_does_not_recopy(tmp_path):
    path = Path(__file__).resolve().parents[2] / "tools/frame_cache.py"
    spec = importlib.util.spec_from_file_location("frame_cache", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, destination = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"complete media")
    destination.with_name("target.partial").write_bytes(b"complete ")
    assert module.fetch(str(source), destination, size=14)
    assert destination.read_bytes() == b"complete media"
    assert not module.fetch(str(source), destination, size=14)


@pytest.mark.parametrize("supports_range", [True, False])
def test_http_partial_transfer(tmp_path, supports_range):
    path = Path(__file__).resolve().parents[2] / "tools/frame_cache.py"
    spec = importlib.util.spec_from_file_location("frame_cache", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ranges = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            ranges.append(self.headers.get("Range"))
            self.send_response(206 if supports_range else 200)
            self.end_headers()
            self.wfile.write(b"media" if supports_range else b"complete media")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    target = tmp_path / "target"
    target.with_name("target.partial").write_bytes(b"complete ")
    try:
        module.fetch(f"http://127.0.0.1:{server.server_port}/file", target, size=14)
        assert target.read_bytes() == b"complete media"
        assert ranges == ["bytes=9-"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
