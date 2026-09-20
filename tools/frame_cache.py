"""Fetch frozen benchmark annotations and stage videos/indexed JPEG packs.

Sources can be mounted files, HTTP(S), hf://datasets/... or hdfs:// URIs.
No credential, Trial identity or node hostname belongs in a data manifest.
"""

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmms_eval.models.model_utils.packed_frames import build_pack, pack_index


def fetch(source: str, destination: Path, *, size: int | None = None, sha256: str | None = None) -> bool:
    """Atomically fetch an artifact, resuming mounted/HTTP partial transfers.

    Small annotation hashes are checked once here, never on the inference path.
    Existing large packs are checked by size. Returns whether bytes were copied.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and (size is None or destination.stat().st_size == size):
        if sha256 is None or hashlib.sha256(destination.read_bytes()).hexdigest() == sha256:
            return False
    partial = destination.with_name(destination.name + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if size is not None and offset > size:
        partial.unlink()
        offset = 0
    if source.startswith("hdfs://"):
        partial.unlink(missing_ok=True)
        subprocess.run(["hdfs", "dfs", "-get", source, str(partial)], check=True)
    elif source.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        kind, owner, repo, filename = source[5:].split("/", 3)
        local = hf_hub_download(repo_id=f"{owner}/{repo}", filename=filename, repo_type="dataset" if kind == "datasets" else "model")
        return fetch(local, destination, size=size, sha256=sha256)
    elif source.startswith(("http://", "https://")):
        request = urllib.request.Request(source, headers={"Range": f"bytes={offset}-"} if offset else {})
        with urllib.request.urlopen(request, timeout=120) as response:
            mode = "ab" if offset and response.status == 206 else "wb"
            with partial.open(mode) as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    else:
        path = Path(source.removeprefix("file://"))
        if size is None:
            size = path.stat().st_size
        with path.open("rb") as source_file, partial.open("ab") as output:
            source_file.seek(offset)
            shutil.copyfileobj(source_file, output, length=8 * 1024 * 1024)
    if size is not None and partial.stat().st_size != size:
        raise ValueError(f"Incomplete download: {destination}")
    if sha256 and hashlib.sha256(partial.read_bytes()).hexdigest() != sha256:
        partial.unlink()
        raise ValueError(f"Annotation checksum mismatch: {destination}")
    partial.replace(destination)
    return True


def safe_name(name: str) -> str:
    """Reject manifest paths escaping the selected destination."""
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"Invalid artifact name: {name}")
    return name


def prepare(manifest_source: str, data_root: Path, *, tasks: list[str] | None = None, limit: int | None = None, media: str = "frames", workers: int = 8) -> dict:
    """Download all selected annotations, then just the selected samples' media."""
    data_root.mkdir(parents=True, exist_ok=True)
    fetch(manifest_source, data_root / "manifest.json")
    manifest = json.loads((data_root / "manifest.json").read_text())
    selected = tasks or list(manifest["tasks"])
    keys = set()
    counts = {}
    for name in selected:
        safe_name(name)
        artifact = manifest["tasks"][name]
        target = data_root / "annotations" / f"{name}.jsonl"
        fetch(artifact["uri"], target, size=artifact["bytes"], sha256=artifact["sha256"])
        rows = [json.loads(line) for line in target.read_text().splitlines() if line]
        if len(rows) != artifact["count"]:
            raise ValueError(f"Annotation count mismatch: {name}")
        counts[name] = len(rows)
        keys.update(row["key"] for row in (rows[:limit] if limit else rows))

    def stage(key: str) -> dict:
        safe_name(key)
        item = manifest["media"][key]
        target = data_root / "frames" / key
        if media == "none":
            return {"key": key, "action": "annotations_only"}
        if media == "frames" and (target / "manifest.json").exists() and (target / "frames.mjpeg").exists():
            pack_index(str(target.resolve()))
            return {"key": key, "action": "reused"}
        if media == "frames" and item.get("pack"):
            fetch(item["pack"] + "/manifest.json", target / "manifest.json")
            index = json.loads((target / "manifest.json").read_text())
            filename = safe_name(index["frames"]["file"])
            fetch(item["pack"] + "/" + filename, target / filename, size=index["frames"]["bytes"])
            pack_index(str(target.resolve()))
            return {"key": key, "action": "copied_pack"}
        video = data_root / "videos" / f"{key}.mp4"
        fetch(item["uri"], video, size=item.get("bytes"))
        if media == "frames":
            # A previously interrupted copy may have left only an index.
            if target.exists():
                raise ValueError(f"Incomplete pack at {target}; finish its copy or remove this incomplete directory before building")
            build_pack(video, target)
        return {"key": key, "action": "built_pack" if media == "frames" else "copied_video"}

    results, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(stage, key): key for key in sorted(keys)}
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"key": futures[future], "error": f"{type(exc).__name__}: {exc}"})
    receipt = {"manifest": manifest_source, "tasks": counts, "limit": limit, "media": media, "selected_videos": len(keys), "results": results, "failures": failures}
    (data_root / "prepare_receipt.json").write_text(json.dumps(receipt, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} media failures; see {data_root / 'prepare_receipt.json'}")
    return receipt


def main() -> None:
    """CLI entry: prepare, inspect or explicitly build a frame pack."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--data-root", type=Path, required=True)
    prep.add_argument("--tasks", help="Comma-separated subtask names; default: all")
    prep.add_argument("--limit", type=int)
    prep.add_argument("--media", choices=["none", "frames", "videos"], default="frames")
    prep.add_argument("--workers", type=int, default=8)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--data-root", type=Path, required=True)
    build = commands.add_parser("build")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.workers < 1 or (args.limit is not None and args.limit < 1):
            parser.error("workers/limit must be positive")
        result = prepare(args.manifest, args.data_root, tasks=args.tasks.split(",") if args.tasks else None, limit=args.limit, media=args.media, workers=args.workers)
        print(json.dumps({key: value for key, value in result.items() if key != "results"}))
    elif args.command == "build":
        print(json.dumps(build_pack(args.source, args.destination)))
    else:
        manifest = json.loads((args.data_root / "manifest.json").read_text())
        available, missing = [], []
        for key in manifest["media"]:
            try:
                pack_index(str((args.data_root / "frames" / key).resolve()))
                available.append(key)
            except (OSError, ValueError):
                missing.append(key)
        print(json.dumps({"available": len(available), "missing": len(missing), "missing_keys": missing}))


if __name__ == "__main__":
    main()
