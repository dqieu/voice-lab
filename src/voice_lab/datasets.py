"""Local dataset discovery and registration; audio is read in place."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import uuid

EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff", ".aif", ".opus"}


def index_dataset(kind, source, recursive=True):
    source = Path(source).expanduser().resolve()
    paths, seen, skipped = [], set(), {}

    def skip(reason):
        skipped[reason] = skipped.get(reason, 0) + 1

    def add(path):
        if path.suffix.lower() not in EXTENSIONS:
            skip("unsupported")
        elif not path.is_file():
            skip("missing")
        else:
            path = path.resolve()
            if path in seen:
                skip("duplicate")
            else:
                seen.add(path)
                paths.append(path)

    if kind == "folder":
        if not source.is_dir():
            raise ValueError("Choose an existing folder")
        # Do not follow directory symlinks or silently accept unreadable subtrees.
        def walk_error(error):
            raise ValueError(f"Cannot read folder: {error.filename}") from error

        for directory, dirs, files in os.walk(source, onerror=walk_error):
            dirs.sort()
            for name in sorted(files):
                path = Path(directory) / name
                if path.suffix.lower() in EXTENSIONS:
                    add(path)
            if not recursive:
                break
    elif kind == "csv":
        if not source.is_file() or source.suffix.lower() != ".csv":
            raise ValueError("Choose an existing CSV file")
        try:
            with source.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream, strict=True)
                headers = [h.strip() for h in (reader.fieldnames or [])]
                if headers.count("path") != 1:
                    raise ValueError("CSV needs exactly one column named path")
                reader.fieldnames = headers
                for row in reader:
                    raw = (row.get("path") or "").strip()
                    if not raw:
                        skip("empty")
                        continue
                    path = Path(raw).expanduser()
                    add(path if path.is_absolute() else source.parent / path)
        except (UnicodeError, csv.Error) as exc:
            raise ValueError("Use a UTF-8, comma-separated CSV with a path column") from exc
    else:
        raise ValueError("Dataset source must be folder or csv")
    if not paths:
        detail = "; ".join(f"{count} {reason}" for reason, count in skipped.items())
        raise ValueError(f"No supported audio files found{': ' + detail if detail else ''}")
    return paths, skipped


def read_registry(path):
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("datasets"), list):
        raise ValueError("Unsupported local dataset registry")
    return {record["id"]: record for record in data["datasets"]}


def write_registry(path, records):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps({"version": 1, "datasets": list(records.values())},
                                        indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
