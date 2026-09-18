"""User-selected local datasets, persistence, and loopback import boundaries."""
import csv
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import soundfile as sf

from voice_lab.datasets import index_dataset
from voice_lab.player import Catalog, PlayerServer, synthetic


@pytest.fixture
def recordings(tmp_path):
    root = tmp_path / "My recordings"
    nested = root / "nested"
    nested.mkdir(parents=True)
    audio, _ = synthetic(42)
    sf.write(root / "voice, one.WAV", audio, 16000)
    sf.write(nested / "voice two.flac", audio, 16000)
    (root / "notes.txt").write_text("not audio")
    return root


def test_folder_scan_respects_recursion_and_does_not_follow_directory_symlinks(recordings):
    (recordings / "nested/loop").symlink_to(recordings, target_is_directory=True)
    direct, _ = index_dataset("folder", recordings, False)
    recursive, _ = index_dataset("folder", recordings, True)
    assert [p.name for p in direct] == ["voice, one.WAV"]
    assert len(recursive) == 2
    assert set(p.name for p in recursive) == {"voice, one.WAV", "voice two.flac"}


def test_csv_bom_relative_absolute_paths_extra_columns_and_skipped_rows(recordings, tmp_path):
    manifest = tmp_path / "recordings.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "label"])
        writer.writerows([["My recordings/voice, one.WAV", "speech"],
                          [str(recordings / "nested/voice two.flac"), "speech"],
                          [str(recordings / "voice, one.WAV"), "duplicate"],
                          ["missing.wav", "missing"], ["", "blank"], ["notes.txt", "not audio"]])
    paths, skipped = index_dataset("csv", manifest)
    assert len(paths) == 2 and all(p.is_absolute() for p in paths)
    assert skipped == {"duplicate": 1, "missing": 1, "empty": 1, "unsupported": 1}


@pytest.mark.parametrize("header", ["file", "Path", "path,path"])
def test_csv_requires_one_path_column(tmp_path, header):
    path = tmp_path / "bad.csv"
    path.write_text(header + "\n")
    with pytest.raises(ValueError, match="column named path"):
        index_dataset("csv", path)


def test_registry_restores_imports_and_keeps_unavailable_sources(recordings, tmp_path):
    registry = tmp_path / "state/datasets.json"
    catalog = Catalog(registry=registry)
    added = catalog.add("folder", str(recordings), "Personal recordings", False)
    assert added["count"] == added["indexed"] == 1
    restored = Catalog(registry=registry)
    item = next(d for d in restored.public() if d["id"] == added["id"])
    assert item["name"] == "Personal recordings" and item["available"]
    (recordings / "voice, one.WAV").unlink()
    unavailable = Catalog(registry=registry)
    item = next(d for d in unavailable.public() if d["id"] == added["id"])
    assert not item["available"] and item["import"]["unavailable"]
    assert len(json.loads(registry.read_text())["datasets"]) == 1
    with pytest.raises(ValueError, match="Dataset unavailable"):
        unavailable.sample(added["id"], 42, 1)


def test_duplicate_invalid_and_failed_persistence_do_not_publish_partial_imports(recordings, tmp_path, monkeypatch):
    catalog = Catalog(registry=tmp_path / "datasets.json")
    catalog.add("folder", str(recordings), recursive=False)
    with pytest.raises(ValueError, match="already added"):
        catalog.add("folder", str(recordings), recursive=False)
    with pytest.raises(ValueError, match="true or false"):
        catalog.add("folder", str(recordings), recursive="false")
    with pytest.raises(ValueError, match="existing folder"):
        catalog.add("folder", str(tmp_path / "absent"))

    def cannot_save(*args):
        raise OSError("disk full")

    monkeypatch.setattr("voice_lab.player.write_registry", cannot_save)
    before = catalog.public()
    with pytest.raises(OSError, match="disk full"):
        catalog.add("folder", str(recordings), recursive=True)
    assert catalog.public() == before


def test_http_import_picker_and_selected_dataset_analysis(recordings, tmp_path):
    registry = tmp_path / "datasets.json"
    selected = []

    def picker(kind):
        selected.append(kind)
        return str(recordings) if kind == "folder" else None

    server = PlayerServer(("127.0.0.1", 0), Catalog(registry=registry), tmp_path / "runs", picker=picker)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def post(path, body, headers=None):
        return json.load(urlopen(Request(base + path, data=json.dumps(body).encode(),
                                        headers={"Content-Type": "application/json", **(headers or {})})))

    try:
        assert json.load(urlopen(base + "/api/datasets"))["native_picker"]
        assert post("/api/dataset-picker", {"kind": "folder"})["path"] == str(recordings)
        assert post("/api/dataset-picker", {"kind": "csv"})["path"] is None
        assert selected == ["folder", "csv"]
        added = post("/api/datasets", {"kind": "folder", "path": str(recordings), "name": "User audio", "recursive": False})["dataset"]
        assert added["count"] == 1
        analysis = post("/api/sample", {"dataset": added["id"], "seconds": 1})
        assert analysis["source"]["dataset_name"] == "User audio"
        assert analysis["source"]["recording"] == "voice, one.WAV"
        assert analysis["duration"] == 1
        assert analysis["representation"]["full_rmse"] < 1e-6
        wav, sr = sf.read(tmp_path / "runs" / analysis["artifact"] / "original.wav")
        assert sr == 16000 and np.isfinite(wav).all()
        manifest = tmp_path / "manifest.csv"
        manifest.write_text('path,extra\n"My recordings/nested/voice two.flac",anything\n')
        csv_added = post("/api/datasets", {"kind": "csv", "path": str(manifest), "recursive": False})["dataset"]
        assert csv_added["count"] == 1
        for body, headers in [({"kind": "folder", "path": str(recordings), "recursive": "false"}, {}),
                              ({"kind": "folder", "path": str(recordings)}, {"Origin": "https://attacker.example"})]:
            with pytest.raises(HTTPError) as error:
                post("/api/datasets", body, headers)
            assert error.value.code in {400, 403}
        assert len(json.loads(registry.read_text())["datasets"]) == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
