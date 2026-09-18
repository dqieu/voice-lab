"""Shared deterministic framing, pitch candidates and activity-duration rules.

Thresholds are engineering defaults, not trained parameters. Activity means
speech-like sound; coughs, music and hum can fool these acoustic rules.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class Config:
    sample_rate: int = 16000
    frame_ms: float = 25
    hop_ms: float = 10
    pitch_window_ms: float = 50
    f0_min: float = 60
    f0_max: float = 500
    sensitivity: float = 0.5
    min_speech_ms: float = 100
    min_silence_ms: float = 140
    padding_ms: float = 50
    lpc_order: int = 18
    denoise: bool = False

    def validate(self):
        if self.sample_rate != 16000:
            raise ValueError("Analysis rate is fixed at 16 kHz; resample input first")
        if not 30 <= self.f0_min < self.f0_max <= 1000:
            raise ValueError("Pitch range must satisfy 30 <= min < max <= 1000 Hz")
        if not 0 <= self.sensitivity <= 1:
            raise ValueError("Sensitivity must be between 0 and 1")
        if not 5 <= self.hop_ms <= self.frame_ms <= 50:
            raise ValueError("Require 5 <= hop <= frame <= 50 ms")
        if not 3 * 1000 / self.f0_min <= self.pitch_window_ms <= 120:
            raise ValueError("Pitch window must cover at least three lowest-F0 periods")
        if not 2 <= self.lpc_order <= 30:
            raise ValueError("LPC order must be between 2 and 30")
        if any(not 0 <= v <= 1000 for v in
               (self.min_speech_ms, self.min_silence_ms, self.padding_ms)):
            raise ValueError("Segment durations must be between 0 and 1000 ms")


def _frames(x, size, hop):
    centers = np.arange(0, len(x), hop)
    padded = np.pad(x, (size // 2, size))
    return np.lib.stride_tricks.sliding_window_view(padded, size)[centers].copy()


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _pitch(frames, cfg):
    """Normalized lag correlation, several peaks, harmonic support, Viterbi path."""
    frames = frames - frames.mean(axis=1, keepdims=True)
    n = frames.shape[1]
    spectra = np.fft.rfft(frames, n=2 ** math.ceil(math.log2(2 * n)))
    ac = np.fft.irfft(abs(spectra) ** 2, n=(spectra.shape[1] - 1) * 2)[:, :n]
    cumulative = np.c_[np.zeros(len(frames)), np.cumsum(frames ** 2, axis=1)]
    lo, hi = int(cfg.sample_rate / cfg.f0_max), int(cfg.sample_rate / cfg.f0_min)
    lags = np.arange(lo, min(hi + 1, n - 1))
    denominator = np.sqrt(cumulative[:, n - lags] *
                          (cumulative[:, -1, None] - cumulative[:, lags]))
    corr = np.clip(ac[:, lags] / np.maximum(denominator, 1e-15), 0, 1)
    count = len(frames)
    frequencies, strengths = np.zeros((count, 5)), np.zeros((count, 5))
    magnitude = abs(spectra)
    fft_size = (spectra.shape[1] - 1) * 2
    for t in range(count):
        peaks, _ = signal.find_peaks(corr[t], height=0.25)
        if len(lags) == 1:
            peaks = np.array([0]) if corr[t, 0] >= 0.25 else np.array([], dtype=int)
        elif corr[t, 0] > corr[t, 1]:
            peaks = np.r_[0, peaks]
        if len(lags) > 1 and corr[t, -1] > corr[t, -2]:
            peaks = np.r_[peaks, len(lags) - 1]
        peaks = sorted(peaks, key=lambda j: corr[t, j], reverse=True)[:4]
        for k, j in enumerate(peaks, 1):
            lag = float(lags[j])
            if 0 < j < len(lags) - 1:
                a, b, c = corr[t, j - 1:j + 2]
                curvature = a - 2 * b + c
                if abs(curvature) > 1e-10:
                    lag += np.clip(0.5 * (a - c) / curvature, -0.5, 0.5)
            f0 = cfg.sample_rate / lag
            bins = np.rint(np.arange(1, 9) * f0 / cfg.sample_rate * fft_size).astype(int)
            bins = bins[bins < magnitude.shape[1]]
            support = np.mean(magnitude[t, bins]) / (np.max(magnitude[t]) + 1e-12)
            frequencies[t, k] = f0
            strengths[t, k] = min(1, corr[t, j] + 0.06 * support)
    periodicity = corr.max(axis=1)
    # Slight preference for the shortest true period suppresses subharmonic locking.
    emission = 1 - strengths + 0.025 * np.log2(
        cfg.f0_max / np.maximum(frequencies, cfg.f0_min))
    emission[frequencies == 0] = 10
    emission[:, 0] = 0.30 + 0.65 * np.clip(periodicity, 0, 1)
    back = np.zeros((count, 5), dtype=int)
    costs = emission[0].copy()
    for t in range(1, count):
        previous, current = frequencies[t - 1], frequencies[t]
        transition = 0.32 * abs(np.log2(
            np.maximum(previous[:, None], 1) / np.maximum(current[None, :], 1)))
        transition[(previous[:, None] == 0) | (current[None, :] == 0)] = 0.16
        transition[0, 0] = 0
        total = costs[:, None] + transition
        back[t] = total.argmin(axis=0)
        costs = emission[t] + total.min(axis=0)
    chosen = np.zeros(count, dtype=int)
    chosen[-1] = costs.argmin()
    for t in range(count - 1, 0, -1):
        chosen[t - 1] = back[t, chosen[t]]
    return frequencies[np.arange(count), chosen], periodicity


def _activity(energy, periodicity, flatness, zcr, cfg):
    low, high = np.percentile(energy, [20, 85])
    # In an all-speech excerpt there may be no valid silence/noise observation.
    structure_present = np.percentile(periodicity, 85) > 0.55
    floor = high - 14 if high - low <= 8 and structure_present else low
    margin = energy - floor
    e = np.clip((margin - 2) / 14, 0, 1)
    structure = 1 - np.clip(flatness / 0.6, 0, 1)
    score = (0.55 * e + 0.30 * periodicity + 0.15 * structure
             - 0.08 * np.clip(zcr / 0.5, 0, 1))
    score[energy < -75] = 0
    enter = 0.58 - 0.18 * cfg.sensitivity
    leave = enter - 0.12
    active, mask = False, np.zeros(len(energy), dtype=bool)
    for i, s in enumerate(score):
        active = s >= (leave if active else enter)
        mask[i] = active
    for start, end in _runs(~mask):
        if start > 0 and end < len(mask) and (end - start) * cfg.hop_ms < cfg.min_silence_ms:
            mask[start:end] = True
    for start, end in _runs(mask):
        if (end - start) * cfg.hop_ms < cfg.min_speech_ms:
            mask[start:end] = False
    pad = round(cfg.padding_ms / cfg.hop_ms)
    if pad:
        expanded = mask.copy()
        for start, end in _runs(mask):
            expanded[max(0, start - pad):min(len(mask), end + pad)] = True
        mask = expanded
        # Padding can leave tiny gaps between otherwise valid adjacent intervals.
        for start, end in _runs(~mask):
            if start > 0 and end < len(mask) and (end - start) * cfg.hop_ms < cfg.min_silence_ms:
                mask[start:end] = True
    return mask, np.clip(score, 0, 1), float(floor)


def _speaker_changes(envelope, f0, formants, speech, cfg):
    """Conservative candidate boundaries, never speaker identities/diarization."""
    step = max(1, round(100 / cfg.hop_ms))
    context = max(1, round(700 / cfg.hop_ms))
    scale = np.maximum(np.median(abs(envelope - np.median(envelope, axis=0)), axis=0), 1)

    def distance(i, width):
        left = np.arange(max(0, i - width), i)
        right = np.arange(i, min(len(speech), i + width))
        left, right = left[speech[left]], right[speech[right]]
        if min(len(left), len(right)) < width * 0.4:
            return 0
        d_env = np.mean(np.minimum(abs(np.median(envelope[left], axis=0) -
                                      np.median(envelope[right], axis=0)) / scale, 4))
        lf, rf = f0[left][f0[left] > 0], f0[right][f0[right] > 0]
        d_f0 = min(abs(np.log2(np.median(lf) / np.median(rf))) / 0.3, 4) if min(len(lf), len(rf)) >= 10 else 0
        lform, rform = formants[left], formants[right]
        differences = []
        for j in range(3):
            l = lform[:, j][lform[:, j] > 0]
            r = rform[:, j][rform[:, j] > 0]
            if min(len(l), len(r)) >= 10:
                differences.append(min(abs(np.median(l) - np.median(r)) / (200 + 200 * j), 4))
        return 0.65 * d_env + 0.20 * d_f0 + 0.15 * (np.mean(differences) if differences else 0)

    candidates = []
    for i in range(context, len(speech) - context, step):
        score = distance(i, context)
        if score > 1.5 and distance(i, context * 2) > 1.25:
            candidates.append((i, score))
    selected = []
    for i, score in sorted(candidates, key=lambda pair: pair[1], reverse=True):
        if all(abs(i - old) * cfg.hop_ms >= 1000 for old, _ in selected):
            selected.append((i, score))
    return [{"time": round(i * cfg.hop_ms / 1000, 3), "score": round(float(score), 3),
             "label": "Possible speaker change"} for i, score in sorted(selected)]
