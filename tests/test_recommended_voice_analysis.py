"""Known signals test the report's branch discipline and abstention rules."""
from dataclasses import replace
import json
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
import pytest
from scipy import signal
import soundfile as sf

pytest.importorskip("pysptk")
pytest.importorskip("librosa")

from voice_lab.dsp import Config
from voice_lab.evaluation import evaluate
from voice_lab.player import Catalog, PlayerServer, synthetic, stress
from voice_lab.recommended import (
    MeasurementConfig, analyze_recommended, _burg, _consensus,
)
from voice_lab.representation import load_packet, decode_packet


@pytest.fixture(scope="module")
def controlled():
    x,reference=synthetic(42)
    data,waves=analyze_recommended(x,16000)
    return x,reference,data,waves


def test_full_stack_known_pitch_boundaries_and_residual(controlled):
    x,reference,data,waves=controlled
    t=np.array(data["frames"]["time"]);f0=np.array(data["frames"]["f0"])
    assert len(data["segments"])==2
    assert abs(np.median(f0[(t>.8)&(t<1.8)])-125)<5
    assert abs(np.median(f0[(t>3.6)&(t<5.1)])-205)<5
    assert np.mean(np.array(data["frames"]["label"])[(t>2.15)&(t<2.4)]=="unvoiced")>.9
    assert np.all(f0[(t>2.7)&(t<3.1)]==0)
    assert all(np.isfinite(w).all() and len(w)==len(x) for w in waves.values())
    assert "reconstructed" not in waves
    np.testing.assert_allclose(waves["packet_full"],waves["minimal"],atol=1e-6)
    assert data["representation"]["full_rmse"]<1e-6
    assert len(data["spectral"]["mfcc"][0])==13
    assert len(data["spectral"]["cqt_frequency_hz"])>60
    assert data["landmarks"] and data["nuclei"]
    assert data["glottal"]["frames"] and np.any(waves["glottal"])
    assert all(all(v is None for v in row) for row in np.array(data["formant_analysis"]["tracked"],dtype=object)[(t>2.7)&(t<3.1)])
    metrics=evaluate(data,{"intervals":reference,"kind":"synthetic"})
    assert metrics["activity"]["f1"]>.95 # declared 50-ms padding includes silence
    assert metrics["pitch"]["median_absolute_cents"]<30
    assert all(r["mae_hz"]<60 for r in metrics["formants"])
    json.dumps(data,allow_nan=False)


@pytest.mark.parametrize("kind",["silence","white","colored"])
def test_negative_controls_abstain(kind):
    x=np.zeros(48000) if kind=="silence" else np.random.default_rng(5).normal(0,.015,48000)
    if kind=="colored":
        x=signal.sosfilt(signal.butter(2,1800,fs=16000,output="sos"),x)
    data,waves=analyze_recommended(x,16000)
    assert not data["segments"]
    assert not data["glottal"]["frames"]
    assert all(r["status"]=="rejected" for r in data["voice_quality"])
    assert np.max(abs(waves["speech"]))==0


def test_enhancement_never_changes_measurement_waveform(controlled):
    x,*_=controlled
    noisy,_=stress(x,16000,"noise10",42)
    raw,w1=analyze_recommended(noisy,16000)
    enhanced,w2=analyze_recommended(noisy,16000,replace(Config(),denoise=True),MeasurementConfig(channel="noisy"))
    assert enhanced["stats"]["denoise_applied"]
    assert not np.array_equal(w1["enhanced"],w2["enhanced"])
    np.testing.assert_array_equal(w1["minimal"],w2["minimal"])
    assert raw["frames"]["intensity_dbfs"]==enhanced["frames"]["intensity_dbfs"]
    assert all(r["status"]=="rejected" for r in enhanced["voice_quality"])


def test_telephone_bandwidth_and_source_abstention(controlled):
    x,*_=controlled
    data,waves=analyze_recommended(x,16000,measurements=MeasurementConfig(channel="telephone"))
    assert data["formant_analysis"]["ceiling_hz"]==3400
    assert all(f<3400 for row in data["frames"]["formants"] for f in row)
    assert not np.any(waves["glottal"])
    assert data["voice_quality"][0]["status"]=="rejected"
    data,_=analyze_recommended(np.zeros((160,2)),8000)
    assert data["provenance"]["bandwidth_hz"]==3800
    assert data["duration"]==pytest.approx(.02)
    json.dumps(data,allow_nan=False)


def test_octave_disagreement_does_not_manufacture_intermediate_pitch():
    estimates=np.tile([100.,100.,200.,200.],(20,1))
    f0,_,_,status=_consensus(estimates,np.ones(20),np.ones(20),np.ones(20,dtype=bool),Config(),MeasurementConfig())
    assert np.all(f0==0) and np.all(status=="rejected")


def test_burg_recovers_known_all_pole_resonance():
    radius=np.exp(-np.pi*100/16000);angle=2*np.pi*600/16000
    expected=np.array([1.,-2*radius*np.cos(angle),radius**2])
    x=signal.lfilter([1],expected,np.random.default_rng(10).normal(size=30000))[2000:]
    a=_burg(x,2)
    np.testing.assert_allclose(a,expected,atol=.008)
    assert np.all(abs(np.roots(a))<1)


def test_clean_sustained_sine_has_low_perturbation():
    sr=16000;x=np.zeros(sr*3);t=np.arange(sr*2)/sr
    x[sr//2:sr//2+len(t)]=.2*np.sin(2*np.pi*160*t)
    data,_=analyze_recommended(x,sr)
    qualified=[r for r in data["voice_quality"] if r["status"]=="observed"]
    assert qualified
    assert qualified[0]["jitter_local"]<.002
    assert qualified[0]["shimmer_local"]<.003
    assert qualified[0]["jitter_ppq5"] is not None


def test_http_unified_analysis_exports_and_channel_selection(tmp_path):
    catalog=Catalog()
    x,_=synthetic(42)
    sf.write(tmp_path/"stereo.wav",np.c_[x,-x],16000,subtype="FLOAT")
    catalog.sources["stereo"]=[tmp_path/"stereo.wav"];catalog.counts["stereo"]=1
    server=PlayerServer(("127.0.0.1",0),catalog,tmp_path/"runs")
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f"http://127.0.0.1:{server.server_port}"
    try:
        req=Request(base+"/api/sample",data=json.dumps({"dataset":"stereo","seed":42}).encode(),headers={"Content-Type":"application/json"})
        result=json.load(urlopen(req))
        assert "comparison" not in result
        assert result["method"]=="recommended"
        assert result["source"]["selected_channel"]==0
        assert result["segments"]
        assert "reconstructed" not in result["audio"]
        original,_=sf.read(tmp_path/"runs"/result["artifact"]/"original.wav")
        np.testing.assert_array_equal(original,x)
        assert urlopen(base+result["audio"]["glottal"]).status==200
        packet_url=base+result["downloads"]["representation"]
        assert urlopen(packet_url).status==200
        packet=load_packet(tmp_path/"runs"/result["artifact"]/"representation.npz")
        minimal,_=sf.read(tmp_path/"runs"/result["artifact"]/"minimal.wav")
        np.testing.assert_allclose(decode_packet(packet),minimal,atol=1e-6)
        assert urlopen(base+result["audio"]["packet_parametric"]).status==200
        assert urlopen(base+result["downloads"]["segments"]).status==200
        for retired in ("classic","comparison"):
            with pytest.raises(HTTPError) as error:
                urlopen(Request(base+"/api/sample",data=json.dumps({"dataset":"synthetic","method":retired}).encode()))
            assert error.value.code==400
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)
