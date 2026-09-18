"""Check the exported representation, independently of the feature frontend."""
import copy
import json
import subprocess
import sys

import numpy as np
import pytest
from scipy import signal
import soundfile as sf

pytest.importorskip("pysptk")

from voice_lab.dsp import Config
from voice_lab.representation import (
    encode_streams, decode_packet, save_packet, load_packet, _lpc_from_lsf, MODES,
)


@pytest.fixture(scope="module")
def representation():
    sr = 16000; x = np.zeros(sr*2); t = np.arange(sr)/sr
    # Known vowel-like resonances followed by frication, a burst and breath noise.
    source = .05*(np.sin(2*np.pi*160*t)+.25*np.sin(2*np.pi*320*t+.4))
    vowel = signal.lfilter([1.], [1., -1.4, .8], source)
    x[2000:18000] = vowel
    rng = np.random.default_rng(7)
    x[19000:23000] = signal.sosfilt(signal.butter(2,3000,fs=sr,btype="highpass",output="sos"),rng.normal(0,.03,4000))
    x[24000:24008] = .2
    x[25000:30000] = rng.normal(0,.006,5000)
    count = (len(x)+159)//160; times = np.arange(count)*.01
    f0 = np.where((times>.125)&(times<1.125),160.,0.)
    # Deliberately mark all unvoiced sounds as nonspeech; the packet must keep them.
    labels = np.where(f0>0,"voiced","nonspeech")
    packet = encode_streams(x,f0,(f0>0).astype(float),np.ones(count),labels,Config(),{"test":"known source plus transient/noise"})
    return x,packet


def test_saved_packet_alone_preserves_all_intervals_and_component_sum(representation,tmp_path):
    x,packet=representation
    save_packet(tmp_path/"packet.npz",packet)
    exported=load_packet(tmp_path/"packet.npz")
    # The decoder derives filters from LSF; exported LPC is only an audit record.
    exported.pop("tract_lpc_audit")
    restored=decode_packet(exported)
    np.testing.assert_allclose(restored,x,atol=1e-7,rtol=1e-6)
    components=decode_packet(exported,"harmonic_only")+decode_packet(exported,"residual_only")
    np.testing.assert_allclose(components,restored,atol=1e-7)
    np.testing.assert_allclose(restored[19000:],x[19000:],atol=1e-7)
    assert np.all(exported["voice"][120:]==0)
    assert np.all(np.diff(exported["lsf"],axis=1)>0)
    # Ordered LSF interpolation also gives a stable synthesis filter.
    for a,b in zip(exported["lsf"][::20],exported["lsf"][1::20]):
        assert np.max(abs(np.roots(_lpc_from_lsf((a+b)/2))))<1


def test_ablations_and_parametric_decoder_do_not_need_dense_residual(representation):
    x,packet=representation
    outputs={mode:decode_packet(packet,mode) for mode in MODES}
    for value in outputs.values():
        assert len(value)==len(x) and np.isfinite(value).all()
        assert np.max(abs(value))<2
    assert np.sqrt(np.mean((outputs["parametric"]-x)**2))>.001
    assert np.sqrt(np.mean((outputs["flat_tract"]-x)**2))>.001
    without_residual={k:v for k,v in packet.items() if k!="residual"}
    np.testing.assert_array_equal(decode_packet(without_residual,"parametric"),outputs["parametric"])
    np.testing.assert_array_equal(decode_packet(without_residual,"harmonic_only"),outputs["harmonic_only"])
    # A tracked pitch change must affect harmonic synthesis, not just annotations.
    shifted=copy.deepcopy(packet);shifted["f0"]*=1.1
    assert not np.allclose(decode_packet(shifted,"harmonic_only"),outputs["harmonic_only"])


def test_decode_command_uses_only_exported_packet(representation,tmp_path):
    x,packet=representation
    path=tmp_path/"voice.npz";save_packet(path,packet)
    result=subprocess.run([sys.executable,"-m","voice_lab.player","decode",str(path),"--out",str(tmp_path/"decoded.wav")],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)["samples"]==len(x)
    restored,sr=sf.read(tmp_path/"decoded.wav")
    assert sr==16000
    np.testing.assert_allclose(restored,x,atol=1e-7)


@pytest.mark.parametrize("order",[2,3,17,18,30])
def test_short_silence_and_lpc_orders(order):
    x=np.zeros(320);cfg=Config(lpc_order=order);n=2
    packet=encode_streams(x,np.zeros(n),np.zeros(n),np.zeros(n),["nonspeech"]*n,cfg,{})
    np.testing.assert_array_equal(decode_packet(packet),x)
    assert packet["lsf"].shape==(n,order)


def test_invalid_lsf_or_residual_is_rejected(representation):
    _,packet=representation
    invalid=copy.deepcopy(packet);invalid["lsf"][0,1]=invalid["lsf"][0,0]
    with pytest.raises(ValueError,match="Unordered LSF"):
        decode_packet(invalid)
    invalid=copy.deepcopy(packet);invalid["residual"][0,0]=np.nan
    with pytest.raises(ValueError,match="Invalid residual"):
        decode_packet(invalid)
