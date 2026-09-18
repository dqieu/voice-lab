"""Versioned, self-contained six-stream source/filter packets and offline decoder.

RESID is the normalized LPC excitation minus HARM, not a second full excitation.
Frame-local zero-state filters and Hann overlap/add are part of the wire format.
This preserves the declared target; it is not source separation or a low-rate codec.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import signal

from .dsp import _frames

SCHEMA = "classical-source-filter-v1"
MODES = ("full", "harmonic_only", "residual_only", "parametric", "flat_tract", "constant_gain")
BANDS = np.array([0., 500., 1000., 2000., 4000., 6000., 8000.])


def _lpc_from_lsf(lsf):
    import pysptk
    # SPTK stores gain at index zero. It is always one here; AMP owns gain.
    return pysptk.lsp2lpc(np.r_[1., lsf], has_gain=True, itype=0)


def _harmonic(f0, amplitudes, phases, size, sr):
    active = np.flatnonzero(amplitudes > 0)
    if f0 <= 0 or not len(active):
        return np.zeros(size)
    angle = 2*np.pi*f0*(active[:, None]+1)*np.arange(size)[None, :]/sr
    return np.sum(amplitudes[active, None]*np.cos(angle+phases[active, None]), axis=0)


def _band_energy(x, edges, sr):
    spectrum = np.fft.rfft(x)
    power = abs(spectrum)**2/len(x)**2
    power[1:-1] *= 2
    frequency = np.fft.rfftfreq(len(x), 1/sr)
    # Include Nyquist in the final band.
    return np.array([power[(frequency >= a) & ((frequency < b) if b < sr/2 else (frequency <= b))].sum()
                     for a, b in zip(edges[:-1], edges[1:])])


def encode_streams(x, f0, periodicity, confidence, labels, cfg, provenance):
    """Fit one synthesis envelope on every frame, then factor its excitation."""
    import pysptk
    from .recommended import _burg

    sr = cfg.sample_rate
    hop = round(cfg.hop_ms*sr/1000)
    size = round(cfg.pitch_window_ms*sr/1000)
    # Even length simplifies the Nyquist convention of band-energy measurements.
    size += size % 2
    frames = _frames(x, size, hop)
    count, order, harmonics = len(frames), cfg.lpc_order, 64
    window = signal.windows.hann(size, sym=False)
    lsf = np.zeros((count, order)); lpc = np.zeros((count, order+1))
    gain = np.zeros(count); rms = np.sqrt(np.mean(frames**2, axis=1))
    cycle_rms = np.full(count, np.nan)
    amplitudes = np.zeros((count, harmonics)); phases = amplitudes.copy()
    residual = np.zeros_like(frames)
    harmonic_power = np.zeros((count, len(BANDS)-1)); remainder_power = harmonic_power.copy()
    tract_status = np.zeros(count, dtype=np.uint8)
    expansion = np.full(count, .98)
    voice = np.where(np.asarray(labels) == "voiced", 2, np.where(np.asarray(labels) == "unvoiced", 1, 0)).astype(np.uint8)
    pitch = np.asarray(f0, dtype=float).copy()
    pitch[voice != 2] = 0
    samples = np.arange(size)
    for i, frame in enumerate(frames):
        fitted = _burg((frame-frame.mean())*window, order)
        # Nearly sinusoidal frames can produce a high-order pitch-comb fit with
        # enormous coefficients. Bound both cancellation and synthesis gain.
        identity_fallback = False
        for gamma in (.98, .95, .9, .85, .8, .7, .6, .5):
            a = fitted*gamma**np.arange(order+1)
            _, response = signal.freqz([1.], a, worN=2048)
            if np.sum(abs(a)) <= 20 and np.max(abs(response)) <= 100:
                break
        else:
            a = np.r_[1., np.zeros(order)]
            identity_fallback = True
        expansion[i] = gamma
        # Conversion failures fall back to the identity envelope and are annotated.
        try:
            frequencies = pysptk.lpc2lsp(a, numsp=2048, maxiter=30, eps=1e-10)[1:]
            if len(frequencies) != order or not np.isfinite(frequencies).all() or not (np.all(np.diff(frequencies)>1e-7) and 0 < frequencies[0] < frequencies[-1] < np.pi):
                raise ValueError("Unordered LSF")
            restored = _lpc_from_lsf(frequencies)
            if not np.all(abs(np.roots(restored)) < 1):
                raise ValueError("Unstable synthesis filter")
            tract_status[i] = 2 if identity_fallback else (1 if rms[i] > 1e-12 else 0)
        except (ValueError, RuntimeError):
            frequencies = np.arange(1, order+1)*np.pi/(order+1)
            restored = _lpc_from_lsf(frequencies)
            tract_status[i] = 2
        lsf[i], lpc[i] = frequencies, restored
        excitation = signal.lfilter(restored, [1.], frame)
        gain[i] = np.sqrt(np.mean(excitation**2))
        normalized = excitation/gain[i] if gain[i] > 1e-15 else excitation.copy()
        if pitch[i] > 0:
            number = min(harmonics, int((sr/2-1)/pitch[i]))
            angle = 2*np.pi*pitch[i]*np.arange(1, number+1)[:, None]*samples[None, :]/sr
            weighted = normalized*window
            real = 2*np.einsum("kn,n->k", np.cos(angle), weighted)/window.sum()
            imaginary = -2*np.einsum("kn,n->k", np.sin(angle), weighted)/window.sum()
            amplitudes[i, :number] = np.hypot(real, imaginary)
            phases[i, :number] = np.arctan2(imaginary, real)
            period = round(sr/pitch[i]); start = max(0, size//2-period//2)
            cycles = frame[start:start+period]
            cycle_rms[i] = np.sqrt(np.mean(cycles**2))
        harmonic = _harmonic(pitch[i], amplitudes[i], phases[i], size, sr)
        # A scalar least-squares fit in output space prevents a poor source
        # projection from making component playback louder than the target.
        harmonic_output = signal.lfilter([1.], restored, harmonic*gain[i])
        model_energy = np.dot(harmonic_output, harmonic_output)
        scale = np.clip(np.dot(harmonic_output, frame)/model_energy, 0, 1) if model_energy>1e-30 else 0.
        amplitudes[i] *= scale; harmonic *= scale
        residual[i] = normalized-harmonic
        harmonic_power[i] = _band_energy(harmonic, BANDS, sr)
        remainder_power[i] = _band_energy(residual[i], BANDS, sr)
    total = harmonic_power+remainder_power
    aperiodicity = np.divide(remainder_power, total, out=np.zeros_like(total), where=total>1e-20)
    hnr = 10*np.log10((harmonic_power.sum(axis=1)+1e-20)/(remainder_power.sum(axis=1)+1e-20))
    metadata = {
        "schema": SCHEMA, "sample_rate": sr, "samples": len(x), "frame_size": size,
        "hop": hop, "lpc_order": order, "max_harmonics": harmonics, "band_edges_hz": BANDS.tolist(),
        "target": "minimal branch: selected mono channel, resampled 16 kHz, DC removal and gentle highpass",
        "provenance": provenance,
        "framing": "centers start at sample 0; zero padding; phase origin at frame start; zero-state filters per frame; periodic Hann overlap/add",
        "gain_convention": "AMP gain is unwindowed LPC-excitation RMS; HARM and RESID have this gain divided out",
        "tract": "LSF radians (0, pi), SPTK gain fixed at 1; LPC fit is a spectral envelope, not a physiological tract measurement",
        "harmonic_fit": "Hann-weighted complex projections at k*F0, capped at 64 below Nyquist; frame phases retained; nonnegative output-space least-squares scale bounded to [0,1]",
        "bandwidth_expansion": "start at 0.98; reduce as needed until coefficient L1 <=20 and sampled synthesis response <=100; recorded per frame",
        "residual": "normalized LPC excitation minus the harmonic waveform; all frames retained regardless of VAD",
        "aperiodicity": "band remainder power / (modeled harmonic + remainder power); model components need not be orthogonal",
        "hnr": "modeled harmonic / remainder energy ratio; not a calibrated physiological HNR",
        "tract_status_codes": {"0": "silence", "1": "fitted spectral envelope", "2": "identity fallback"},
        "voice_codes": {"0": "nonspeech annotation", "1": "unvoiced activity", "2": "voiced activity"},
        "precision": "float64, unquantized; dense residual frames, not a compact token stream",
        "parametric_noise_seed": 1729,
        "parametric_gain": "after tract synthesis, normalize each frame to AMP output RMS; full/component modes retain excitation gain",
        "constant_gain": "after full synthesis, normalize nonzero frames to median AMP output RMS",
    }
    return {"metadata": metadata, "time": np.arange(count)*hop/sr, "f0": pitch,
            "voice": voice, "periodicity": np.asarray(periodicity), "pitch_confidence": np.asarray(confidence),
            "lsf": lsf, "tract_lpc_audit": lpc, "tract_status": tract_status, "tract_expansion": expansion,
            "gain": gain, "rms": rms, "cycle_rms": cycle_rms,
            "harmonic_amplitudes": amplitudes, "harmonic_phases": phases,
            "harmonic_band_power": harmonic_power, "remainder_band_power": remainder_power,
            "aperiodicity": aperiodicity, "hnr_db": hnr, "residual": residual}


def save_packet(path, packet):
    arrays = {k: v for k, v in packet.items() if k != "metadata"}
    np.savez_compressed(path, metadata=np.array(json.dumps(packet["metadata"], allow_nan=False)), **arrays)


def load_packet(path):
    with np.load(Path(path), allow_pickle=False) as archive:
        packet = {k: archive[k].copy() for k in archive.files if k != "metadata"}
        packet["metadata"] = json.loads(str(archive["metadata"]))
    _validate(packet)
    return packet


def _validate(packet, needs_residual=True):
    meta = packet["metadata"]
    if meta.get("schema") != SCHEMA or meta.get("sample_rate") != 16000:
        raise ValueError("Unsupported source/filter packet schema or sample rate")
    size, hop, length, order, harmonics = (meta[k] for k in ("frame_size", "hop", "samples", "lpc_order", "max_harmonics"))
    if any(not isinstance(v, int) for v in (size, hop, length, order, harmonics)) or not (2 <= order <= 30 and 1 <= hop <= size <= 1920 and length > 0 and 1 <= harmonics <= 64):
        raise ValueError("Invalid packet dimensions")
    count = (length+hop-1)//hop
    shapes = {"f0": (count,), "voice": (count,), "lsf": (count, order), "gain": (count,), "rms": (count,),
              "harmonic_amplitudes": (count, harmonics), "harmonic_phases": (count, harmonics),
              "remainder_band_power": (count, len(BANDS)-1)}
    if needs_residual:
        shapes["residual"] = (count, size)
    for key, shape in shapes.items():
        if key not in packet or packet[key].shape != shape or not np.isfinite(packet[key]).all():
            raise ValueError(f"Invalid {key} stream")
    if not np.array_equal(meta["band_edges_hz"], BANDS):
        raise ValueError("Unsupported frequency bands")
    if np.any(packet["gain"] < 0) or np.any(packet["rms"] < 0) or np.any(packet["harmonic_amplitudes"] < 0) or np.any(packet["remainder_band_power"] < 0):
        raise ValueError("Negative gain, amplitude or noise energy")
    if np.any((packet["f0"] < 0) | (packet["f0"] > 1000)) or np.any(~np.isin(packet["voice"], [0, 1, 2])):
        raise ValueError("Invalid F0 or voicing state")
    if not (np.all(np.diff(packet["lsf"], axis=1)>0) and np.all(packet["lsf"]>0) and np.all(packet["lsf"]<np.pi)):
        raise ValueError("Unordered LSF stream")


def _noise(energy, size, sr, rng):
    spectrum = np.fft.rfft(rng.normal(size=size))
    frequency = np.fft.rfftfreq(size, 1/sr)
    power = _band_energy(np.fft.irfft(spectrum, n=size), BANDS, sr)
    for j, (a, b) in enumerate(zip(BANDS[:-1], BANDS[1:])):
        selected = (frequency >= a) & ((frequency < b) if b < sr/2 else (frequency <= b))
        spectrum[selected] *= np.sqrt(energy[j]/max(power[j], 1e-30))
    return np.fft.irfft(spectrum, n=size)


def decode_packet(packet, mode="full"):
    """Reconstruct from packet alone; no waveform, feature analyzer or audit LPC."""
    if mode not in MODES:
        raise ValueError(f"Decode mode must be one of {MODES}")
    _validate(packet, needs_residual=mode not in {"parametric", "harmonic_only"})
    meta = packet["metadata"]
    size, hop, sr, length = (meta[k] for k in ("frame_size", "hop", "sample_rate", "samples"))
    out = np.zeros(length+2*size); weight = out.copy()
    window = signal.windows.hann(size, sym=False)
    rng = np.random.default_rng(meta["parametric_noise_seed"])
    gains = packet["gain"]
    rms = packet["rms"]
    constant = float(np.median(rms[rms>1e-12])) if np.any(rms>1e-12) else 0.
    for i in range(len(gains)):
        harmonic = _harmonic(packet["f0"][i] if packet["voice"][i] == 2 else 0,
                             packet["harmonic_amplitudes"][i], packet["harmonic_phases"][i], size, sr)
        if mode == "parametric":
            excitation = harmonic+_noise(packet["remainder_band_power"][i], size, sr, rng)
        elif mode == "harmonic_only":
            excitation = harmonic
        elif mode == "residual_only":
            excitation = packet["residual"][i]
        else:
            excitation = harmonic+packet["residual"][i]
        a = np.array([1.]) if mode == "flat_tract" else _lpc_from_lsf(packet["lsf"][i])
        restored = signal.lfilter([1.], a, excitation*gains[i])
        if mode in {"parametric", "constant_gain"}:
            desired = rms[i] if mode == "parametric" else (constant if rms[i]>1e-12 else 0.)
            measured = np.sqrt(np.mean(restored**2))
            restored *= desired/max(measured, 1e-30)
        start = i*hop
        out[start:start+size] += restored*window; weight[start:start+size] += window
    offset = size//2
    return out[offset:offset+length]/np.maximum(weight[offset:offset+length], 1e-12)


def packet_summary(packet, target, decoded):
    """Small JSON inspection view; dense streams live in the downloadable NPZ."""
    gain = packet["gain"]
    valid = (packet["f0"]>0) & (gain>1e-12)
    hnr = [round(float(v), 2) if ok else None for v, ok in zip(packet["hnr_db"], valid)]
    error = decoded["full"]-target
    return {"schema": SCHEMA, "target": packet["metadata"]["target"], "metadata": packet["metadata"],
            "frames": len(gain), "frame_rate_hz": 16000/packet["metadata"]["hop"],
            "streams": {"F0": "tracked Hz + confidence", "VOICE": "activity state, periodicity and modeled HNR",
                        "TRACT": "ordered LSF radians + synthesis envelope status", "AMP": "excitation gain, output RMS and voiced cycle RMS",
                        "HARM": "harmonic amplitudes/phases and band power/aperiodicity", "RESID": "dense normalized excitation remainder for every frame"},
            "full_rmse": float(np.sqrt(np.mean(error**2))), "full_max_error": float(np.max(abs(error))),
            "snr_db": float(10*np.log10((np.mean(target**2)+1e-30)/(np.mean(error**2)+1e-30))),
            "identity_fallback_frames": int(np.sum(packet["tract_status"]==2)),
            "array_bytes": sum(v.nbytes for k,v in packet.items() if k != "metadata"),
            "display": {"time": packet["time"].tolist(), "hnr_db": hnr, "aperiodicity": packet["aperiodicity"].round(4).tolist(),
                        "gain_db": (20*np.log10(gain+1e-12)).round(2).tolist(),
                        "lsf_hz": (packet["lsf"]*16000/(2*np.pi)).round(1).tolist()},
            "modes": {key: {"rmse_to_minimal": float(np.sqrt(np.mean((value-target)**2))), "peak": float(np.max(abs(value)))}
                      for key,value in decoded.items()},
            "limits": "Full reconstruction uses the stored dense remainder. Parametric synthesis approximates it with band-shaped noise. Neither output isolates a speaker or establishes physiological validity."}
