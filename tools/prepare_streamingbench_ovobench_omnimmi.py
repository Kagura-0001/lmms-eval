"""Publish a portable manifest from the existing frozen evaluation snapshots.

This is a one-time, local export. It neither uploads to a public Hub repository
nor changes the source annotations. Runtime evaluation only needs its output.
"""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def seconds(value: str | float) -> float:
    """Parse the source's seconds or HH:MM:SS timestamp."""
    if not isinstance(value, str) or ":" not in value:
        return float(value)
    result = 0.0
    for component in value.strip().split(":"):
        result = result * 60 + float(component)
    return result


def export(causal_cases: Path, sessions_root: Path, master_cache: Path, destination: Path) -> dict:
    """Freeze annotations and media identities without copying video bytes."""
    if destination.exists():
        raise FileExistsError(destination)
    master = json.loads((master_cache / "manifest.json").read_text())
    media = {}
    for entry in master["entries"]:
        media[entry["key"]] = {"uri": entry["source"]["path"], "bytes": entry["source"]["bytes"], "pack": str(master_cache / entry["key"]), "video": entry["video"]}
    groups = defaultdict(list)
    for row in json.loads(causal_cases.read_text()):
        groups[row["benchmark"]].append(
            {
                "uid": row["uid"],
                "benchmark": row["benchmark"],
                "task": row["task"],
                "key": row["key"],
                "protocol": "causal_suite_v1",
                "qa": [{"question": row["question"], "problem": row["problem"], "answer": str(row["target"]), "at_sec": row["at_sec"], "range_sec": row["range_sec"]}],
            }
        )
    if sum(map(len, groups.values())) != 7485:
        raise ValueError("Expected all 7,485 frozen causal questions")
    sources = {str(causal_cases): hashlib.sha256(causal_cases.read_bytes()).hexdigest()}
    for task in ("sg", "md", "si"):
        source = sessions_root / "annotations" / f"{task}.jsonl"
        sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        for line in source.read_text().splitlines():
            row = json.loads(line)
            path = Path(row["media"]["video"])
            key = hashlib.sha256(str(path).encode()).hexdigest()
            media.setdefault(key, {"uri": str(path), "bytes": path.stat().st_size, "pack": None})
            questions = []
            for qa in row["metadata"]["qa"]:
                if task == "si":
                    at = math.floor(seconds(qa["timestamp"]))
                else:
                    start, end = map(seconds, qa["timestamp"].split("--"))
                    at = math.ceil((start + end) / 2)
                questions.append({"question": qa["question"], "problem": qa["question"], "answer": str(qa["answer"]), "at_sec": at, "range_sec": [max(0, at - 30), at]})
            groups[f"omnimmi_{task}"].append({"uid": row["uid"], "benchmark": f"omnimmi_{task}", "task": task, "key": key, "protocol": "aura_omnimmi_sg_md_si_v1", "qa": questions})
    expected = {"omnimmi_sg": (300, 704), "omnimmi_md": (300, 786), "omnimmi_si": (200, 200)}
    for name, counts in expected.items():
        if (len(groups[name]), sum(len(row["qa"]) for row in groups[name])) != counts:
            raise ValueError(f"Wrong session coverage: {name}")
    (destination / "annotations").mkdir(parents=True)
    tasks = {}
    for name, rows in groups.items():
        path = destination / "annotations" / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        tasks[name] = {"uri": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "count": len(rows), "questions": sum(len(row["qa"]) for row in rows)}
    manifest = {"group": "streamingbench_ovobench_omnimmi", "version": "20260920", "tasks": tasks, "media": media, "sources": sources}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    """Export explicitly supplied, already frozen source snapshots."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("causal-cases", "sessions-root", "master-cache", "destination"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    manifest = export(args.causal_cases, args.sessions_root, args.master_cache, args.destination.resolve())
    print(json.dumps({"tasks": len(manifest["tasks"]), "questions": sum(item["questions"] for item in manifest["tasks"].values()), "videos": len(manifest["media"])}))


if __name__ == "__main__":
    main()
