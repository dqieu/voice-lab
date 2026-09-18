"""Retain dependency license notices and build provenance in the app bundle."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess


def collect_licenses(destination, ffmpeg):
    destination = Path(destination)
    # This is a generated build directory, not a user data directory.
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    packages = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        packages[name] = dist.version
        for file in dist.files or []:
            notice = any(word in file.name.lower() for word in ("license", "licence", "copying"))
            if notice and file.suffix.lower() in {"", ".txt", ".md", ".rst", ".html", ".rtf"}:
                source = Path(dist.locate_file(file))
                if source.is_file():
                    target = destination / name / file
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
    (destination / "packages.json").write_text(json.dumps(packages, indent=2), encoding="utf-8")
    license_info = subprocess.run([str(ffmpeg), "-L"], capture_output=True, text=True, check=True)
    (destination / "ffmpeg-build.txt").write_text(license_info.stdout + license_info.stderr, encoding="utf-8")
    shutil.copy2(Path(__file__).with_name("THIRD_PARTY.md"), destination / "README.md")
    return destination
