"""Known-timing signals verify meaningful activity and reconstruction invariants."""
from dataclasses import replace
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

pytest.importorskip("pysptk")
pytest.importorskip("librosa")

from voice_lab import Config, analyze
from voice_lab.player import Catalog, PlayerServer, stress, synthetic


def test_voiced_unvoiced_pause_and_packet_reconstruction():
    audio, _ = synthetic(42)
    data, waves = analyze(audio, 16000)
    assert len(data["segments"]) == 2
    assert abs(data["segments"][0]["start"] - 0.6) < 0.12
    assert abs(data["segments"][0]["end"] - 2.5) < 0.12
    assert abs(data["segments"][1]["end"] - 5.5) < 0.12
    times = np.array(data["frames"]["time"])
    pitch = np.array(data["frames"]["f0"])
    labels = np.array(data["frames"]["label"])
    assert abs(np.median(pitch[(times > 0.8) & (times < 1.8)]) - 125) < 5
    assert abs(np.median(pitch[(times > 3.5) & (times < 5.2)]) - 205) < 6
    fricative = (times > 2.15) & (times < 2.4)
    assert np.mean(labels[fricative] == "unvoiced") > 0.9
    assert np.all(labels[(times > 2.7) & (times < 3.1)] == "nonspeech")
    np.testing.assert_allclose(waves["speech"][34400:38400], audio[34400:38400], atol=1e-6)
    assert np.max(abs(waves["speech"][44000:48000])) == 0
    assert data["representation"]["full_rmse"] < 1e-6
    np.testing.assert_allclose(waves["minimal"], waves["packet_full"], atol=1e-6)
    json.dumps(data, allow_nan=False)


@pytest.mark.parametrize("audio", [np.zeros(48000), np.random.default_rng(5).normal(0, 0.015, 48000),
                                  signal.sosfilt(signal.butter(2, 1800, fs=16000, output="sos"),
                                                 np.random.default_rng(5).normal(0, 0.015, 48000))])
def test_nonspeech_negative_controls(audio):
    data, waves = analyze(audio, 16000)
    assert not data["segments"]
    assert data["stats"]["median_f0"] is None
    assert not np.any(waves["speech"])


def test_resampling_short_input_and_no_nan():
    data, waves = analyze(np.zeros((882, 2)), 44100)
    assert data["duration"] == pytest.approx(0.02)
    assert len(waves["original"]) == 320
    assert all(np.isfinite(wave).all() for wave in waves.values())
    json.dumps(data, allow_nan=False)
    with pytest.raises(ValueError):
        analyze(np.array([np.nan] * 400), 16000)
    with pytest.raises(ValueError):
        analyze(np.zeros(10), 16000)
    with pytest.raises(ValueError):
        analyze(np.zeros(400), 16000, replace(Config(), f0_min=700))
    narrow, _ = analyze(np.zeros(400), 16000, replace(Config(), f0_min=999, f0_max=1000))
    assert narrow["segments"] == []


def test_seeded_stress_and_stationary_denoising():
    audio, _ = synthetic(42)
    noisy, meta = stress(audio, 16000, "noise10", 12)
    repeated, _ = stress(audio, 16000, "noise10", 12)
    np.testing.assert_array_equal(noisy, repeated)
    noise = noisy / meta["common_gain"] - audio
    assert 10 * np.log10(np.mean(audio ** 2) / np.mean(noise ** 2)) == pytest.approx(10, abs=0.01)
    data, waves = analyze(noisy, 16000, replace(Config(), denoise=True))
    assert data["stats"]["denoise_applied"]
    assert all(len(wave) == len(audio) and np.isfinite(wave).all() for wave in waves.values())


def test_local_http_sampler_audio_range_and_access_boundaries(tmp_path):
    catalog = Catalog()
    excerpt, sr, meta = catalog.sample("synthetic", 42, 1)
    assert len(excerpt) == sr and meta["reference"][0]["end"] == 1
    server = PlayerServer(("127.0.0.1", 0), catalog, tmp_path / "runs")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert urlopen(base + "/").status == 200
        assert json.load(urlopen(base + "/api/datasets"))["datasets"][0]["id"] == "synthetic"
        request = Request(base + "/api/sample", data=json.dumps({"dataset": "synthetic", "seed": 42}).encode(), headers={"Content-Type": "application/json"})
        data = json.load(urlopen(request))
        assert len(data["segments"]) == 2
        request = Request(base + data["audio"]["original"], headers={"Range": "bytes=0-63"})
        with urlopen(request) as response:
            assert response.status == 206
            assert len(response.read()) == 64
        request = Request(base + data["audio"]["original"], headers={"Range": "bytes=-32"})
        assert len(urlopen(request).read()) == 32
        stored, sr = sf.read(tmp_path / "runs" / data["artifact"] / "speech.wav")
        assert sr == 16000 and len(stored) == 112000
        for path, headers in [("/artifacts/../../pyproject.toml", {}), ("/", {"Host": "attacker.example"}), ("/api/datasets", {"Origin": "https://attacker.example"})]:
            with pytest.raises(HTTPError) as error:
                urlopen(Request(base + path, headers=headers))
            assert error.value.code in {403, 404}
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/api/sample", data=b'{"dataset":"synthetic","seconds":5000}'))
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
