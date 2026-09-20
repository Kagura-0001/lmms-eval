"""Portable indexed JPEG packs shared by evaluation tasks and preparation tools.

The existing omnigauge-pyav-jpeg-sequence-v1 format is preserved. Reading a pack
does not require the original video or its original absolute filesystem path.
"""

import base64
import json
import math
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path

FORMAT = "omnigauge-pyav-jpeg-sequence-v1"


@lru_cache(maxsize=256)
def pack_index(entry: str) -> tuple[Path, dict, dict]:
    """Validate an index once per process, without hashing the large pack."""
    root = Path(entry)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["format"] != FORMAT:
        raise ValueError(f"Unsupported frame pack: {manifest['format']}")
    filename = manifest["frames"]["file"]
    if Path(filename).name != filename:
        raise ValueError("Pack filename must be relative to its manifest")
    path = root / filename
    size = path.stat().st_size
    rows = manifest["frames"]["index"]
    if size != manifest["frames"]["bytes"] or len(rows) != manifest["frames"]["count"]:
        raise ValueError(f"Incomplete frame pack: {root}")
    if any(row["offset"] < 0 or row["length"] <= 0 or row["offset"] + row["length"] > size for row in rows):
        raise ValueError("Frame index points outside pack")
    lookup = {float(row["timestamp_seconds"]): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate frame timestamps")
    return path, lookup, manifest


def read_frames(entry: str | Path, timestamps: list[float]) -> list[bytes]:
    """Read exact cached timestamps; never round towards a future frame."""
    path, lookup, _ = pack_index(str(Path(entry).resolve()))
    blobs = []
    with path.open("rb") as handle:
        for timestamp in timestamps:
            if timestamp not in lookup:
                raise ValueError(f"Timestamp {timestamp} is not cached in {entry}; explicit rebuilding is required")
            row = lookup[timestamp]
            handle.seek(row["offset"])
            blob = handle.read(row["length"])
            if len(blob) != row["length"]:
                raise EOFError("Short JPEG pack read")
            blobs.append(blob)
    return blobs


def image_content(blob: bytes) -> dict:
    """Create a native ChatMessages image without recompressing its JPEG."""
    return {"type": "image", "url": "data:image/jpeg;base64," + base64.b64encode(blob).decode("ascii")}


def build_pack(source: str | Path, destination: str | Path, *, fps: float = 2.0, jpeg_quality: int = 80, decoder_threads: int = 2) -> dict:
    """Build a full-resolution, full-video pack on a fixed causal sampling grid."""
    import av

    if not math.isfinite(fps) or fps <= 0 or not 1 <= jpeg_quality <= 100:
        raise ValueError("Invalid FPS or JPEG quality")
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(destination.name + f".building-{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    rows = []
    with av.open(str(source)) as container, (temporary / "frames.mjpeg").open("wb") as output:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = decoder_threads
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else float(container.duration / av.time_base)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Missing video duration")
        iterator = iter(container.decode(stream))
        pending, current = next(iterator, None), None
        for index in range(math.ceil(duration * fps)):
            timestamp = index / fps
            while pending is not None and float(pending.time) <= timestamp + 1e-6:
                current, pending = pending, next(iterator, None)
            frame = current if current is not None else pending if index == 0 else None
            if frame is None:
                raise EOFError(f"No frame at {timestamp}")
            buffer = BytesIO()
            frame.to_image().save(buffer, format="JPEG", quality=jpeg_quality)
            blob = buffer.getvalue()
            rows.append({"timestamp_seconds": timestamp, "offset": output.tell(), "length": len(blob)})
            output.write(blob)
        video = {"duration": duration, "width": stream.width, "height": stream.height}
    stat = source.stat()
    manifest = {
        "format": FORMAT,
        "fps": fps,
        "jpeg_quality": jpeg_quality,
        "resolution": "original",
        "max_frames": None,
        "source": {"path": str(source), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "video": video,
        "frames": {"file": "frames.mjpeg", "count": len(rows), "bytes": (temporary / "frames.mjpeg").stat().st_size, "index": rows},
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest))
    temporary.rename(destination)
    return manifest
