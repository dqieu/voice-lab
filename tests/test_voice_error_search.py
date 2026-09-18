"""Reconstruction-error ranking, exact sample replay, cancellation, and API limits."""
import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import soundfile as sf

from voice_lab import player


def wait_for_search(server, identity):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = server.error_search_status(identity)
        if job["state"] != "running":
            return job
        time.sleep(.01)
    raise AssertionError("Search did not finish")


def test_unique_recordings_rank_correct_output_skip_bad_and_export_only_winner(tmp_path, monkeypatch):
    catalog = player.Catalog()
    catalog.sources["test"] = [None] * 4
    catalog.counts["test"] = 400
    seen, exported = [], []

    def analyze(catalog, request):
        index = request["source_index"]
        seen.append((index, request["seconds"], request["stress"]))
        if index == 3:
            raise ValueError("corrupt recording")
        source = {"recording": f"{index}.wav", "offset": 1., "seed": request["seed"], "source_index": index}
        # Parametric and full-packet rankings intentionally differ.
        data = {"representation": {"modes": {"parametric": {"rmse_to_minimal": [1., 8., 3.][index]},
                                                "full": {"rmse_to_minimal": [9., 1., 2.][index]}}}}
        return data, {}, {"metadata": {}}, source, {}

    def publish(output, data, waves, source, condition, packet):
        exported.append(source["recording"])
        return {**data, "source": source, "artifact": "winner"}

    monkeypatch.setattr(player, "analyze_sample", analyze)
    monkeypatch.setattr(player, "publish_result", publish)
    server = player.PlayerServer(("127.0.0.1", 0), catalog, tmp_path)
    try:
        request = {"dataset": "test", "seed": 42, "seconds": 2, "stress": "noise10", "mode": "parametric", "count": "all"}
        job = wait_for_search(server, server.start_error_search(request)["id"])
        assert job["state"] == "completed"
        assert sorted(index for index, _, _ in seen) == [0, 1, 2, 3]
        assert all(seconds == 2 and stress == "noise10" for _, seconds, stress in seen)
        assert (job["successful"], job["failed"], job["processed"]) == (3, 1, 4)
        assert job["pool_count"] == 4 and job["discovered_count"] == 400
        assert job["best_rmse"] == 8 and exported == ["1.wav"]
        summary = job["result"]["error_search"]
        assert summary["winner_request"]["source_index"] == 1
        assert summary["winner_request"]["seconds"] == 2
        assert job["best_source"]["seed"] == summary["winner_request"]["seed"]
        assert server.slots.acquire(blocking=False)
        server.slots.release()
        with pytest.raises(ValueError):
            server.start_error_search({**request, "count": True})
        with pytest.raises(ValueError):
            server.start_error_search({**request, "mode": "original"})
    finally:
        server.server_close()


def test_cancel_finishes_current_excerpt_then_returns_best_and_releases_slot(tmp_path, monkeypatch):
    catalog = player.Catalog()
    catalog.sources["test"] = [None] * 3
    catalog.counts["test"] = 3
    entered, release = threading.Event(), threading.Event()
    calls = []

    def analyze(catalog, request):
        calls.append(request["source_index"])
        entered.set()
        assert release.wait(5)
        source = {"recording": "current.wav", "offset": 0, "seed": request["seed"], "source_index": request["source_index"]}
        return {"representation": {"modes": {"parametric": {"rmse_to_minimal": .25}}}}, {}, {"metadata": {}}, source, {}

    monkeypatch.setattr(player, "analyze_sample", analyze)
    monkeypatch.setattr(player, "publish_result", lambda output, data, *args: data)
    server = player.PlayerServer(("127.0.0.1", 0), catalog, tmp_path)
    try:
        job = server.start_error_search({"dataset": "test", "count": "all"})
        assert entered.wait(5)
        server.error_search_status(job["id"], cancel=True)
        release.set()
        final = wait_for_search(server, job["id"])
        assert final["state"] == "cancelled" and final["processed"] == 1
        assert len(calls) == 1 and final["best_rmse"] == .25
        assert final["result"]["error_search"]["cancelled"]
        for _ in range(2):
            assert server.slots.acquire(blocking=False)
        for _ in range(2):
            server.slots.release()
    finally:
        release.set()
        server.server_close()


def test_source_index_replays_exact_excerpt_without_random_recording_choice(tmp_path):
    catalog = player.Catalog()
    for index in range(2):
        sf.write(tmp_path / f"{index}.wav", np.linspace(-.1, .1, 32000) * (index + 1), 16000, subtype="FLOAT")
    catalog.sources["test"] = [tmp_path / "0.wav", tmp_path / "1.wav"]
    catalog.counts["test"] = 2
    a, sr, first = catalog.sample("test", 101, 1, source_index=1)
    b, _, second = catalog.sample("test", 101, 1, source_index=1)
    assert sr == 16000 and first == second and first["recording"] == "1.wav"
    np.testing.assert_array_equal(a, b)
    for invalid in (-1, 2, True, .5):
        with pytest.raises(ValueError, match="Source index"):
            catalog.sample("test", 101, 1, source_index=invalid)


def test_http_error_search_real_analysis_download_replay_and_rejections(tmp_path):
    server = player.PlayerServer(("127.0.0.1", 0), player.Catalog(), tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def post(path, body, headers=None):
        return json.load(urlopen(Request(base + path, data=json.dumps(body).encode(),
                                        headers={"Content-Type": "application/json", **(headers or {})}), timeout=30))

    try:
        job = post("/api/error-search", {"dataset": "synthetic", "count": "all", "seconds": 1, "mode": "parametric"})
        final = wait_for_search(server, job["id"])
        remote = json.load(urlopen(base + "/api/error-search/" + job["id"]))
        assert remote["state"] == "completed" and remote["total"] == 1
        result = remote["result"]
        assert len(list(tmp_path.iterdir())) == 1
        assert final["best_rmse"] == result["representation"]["modes"]["parametric"]["rmse_to_minimal"]
        assert urlopen(base + result["downloads"]["representation"]).status == 200
        repeated = post("/api/sample", result["error_search"]["winner_request"])
        a, _ = sf.read(tmp_path / result["artifact"] / "minimal.wav")
        b, _ = sf.read(tmp_path / repeated["artifact"] / "minimal.wav")
        np.testing.assert_array_equal(a, b)
        for body, headers, code in [({"dataset": "synthetic", "mode": "original"}, {}, 400),
                                   ({"dataset": "synthetic", "count": 0}, {}, 400),
                                   ({"dataset": "absent"}, {}, 400),
                                   ({"dataset": "synthetic"}, {"Origin": "https://attacker.example"}, 403)]:
            with pytest.raises(HTTPError) as error:
                post("/api/error-search", body, headers)
            assert error.value.code == code
        for _ in range(2):
            assert server.slots.acquire(blocking=False)
        with pytest.raises(HTTPError) as error:
            post("/api/error-search", {"dataset": "synthetic"})
        assert error.value.code == 429
        for _ in range(2):
            server.slots.release()
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/api/error-search/missing")
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
