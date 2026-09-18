"""Multi-resolution deterministic voice analysis from the supplied recommendation.

Measurements use the minimal branch. Enhancement only feeds acoustic estimators.
Missing physical measurements are null; confidence scores are not probabilities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy import signal, ndimage
from scipy.fft import dct

from .dsp import Config, _frames, _runs, _pitch, _activity, _speaker_changes


@dataclass(frozen=True)
class MeasurementConfig:
    channel: str = "broadband"
    formant_ceiling: float = 5500
    formant_order: int = 10
    agreement_cents: float = 50
    noise_method: str = "wiener"
    noise_gain_floor: float = 0.45
    kalman_process_sd_hz: float = 70
    min_quality_seconds: float = 0.30
    min_quality_snr_db: float = 20
    max_period_factor: float = 1.3
    max_amplitude_factor: float = 1.6
    iaif_tract_order: int = 18
    iaif_source_order: int = 4
    integration_leak: float = 0.99

    def validate(self):
        if self.channel not in {"broadband", "telephone", "noisy"}:
            raise ValueError("Channel must be broadband, telephone or noisy")
        if not 2000 <= self.formant_ceiling <= 7600:
            raise ValueError("Formant ceiling must be 2000–7600 Hz")
        if not 6 <= self.formant_order <= 20:
            raise ValueError("Burg order must be 6–20")
        if self.noise_method not in {"wiener", "subtraction"}:
            raise ValueError("Noise method must be wiener or subtraction")
        if not 0 < self.agreement_cents <= 100 or not 0 < self.noise_gain_floor <= 1:
            raise ValueError("Invalid agreement tolerance or noise gain floor")
        if not 0 < self.kalman_process_sd_hz <= 500 or not 0 < self.min_quality_seconds <= 5:
            raise ValueError("Invalid tracking or quality window")
        if not 0 <= self.min_quality_snr_db <= 60:
            raise ValueError("Quality SNR threshold must be 0–60 dB")
        if not 1 < self.max_period_factor <= 2 or not 1 < self.max_amplitude_factor <= 3:
            raise ValueError("Invalid cycle exclusion factors")
        if not 2 <= self.iaif_source_order <= 10 or not 6 <= self.iaif_tract_order <= 30:
            raise ValueError("Invalid IAIF orders")
        if not 0 < self.integration_leak < 1:
            raise ValueError("Integration leak must be between zero and one")


def _nullable(array, decimals=3, positive=False):
    a = np.round(np.asarray(array, dtype=float), decimals)
    valid = np.isfinite(a) & ((a > 0) if positive else True)
    out = a.astype(object)
    out[~valid] = None
    return out.tolist()


def _burg(frame, order):
    """Forward/backward error recursion; bounded reflection coefficients."""
    x = np.asarray(frame, dtype=float)
    a = np.array([1.0])
    if len(x) <= order + 1 or np.dot(x, x) < 1e-20:
        return np.r_[1., np.zeros(order)]
    forward, backward = x[1:].copy(), x[:-1].copy()
    for _ in range(order):
        k = np.clip(-2 * np.dot(backward, forward) /
                    (np.dot(forward, forward) + np.dot(backward, backward) + 1e-30), -.999, .999)
        a = np.r_[a, 0.] + k * np.r_[0., a[::-1]]
        next_forward = forward + k * backward
        backward = (backward + k * forward)[:-1]
        forward = next_forward[1:]
    return a


def _cues(x, size, hop):
    frames = _frames(x, size, hop)
    energy = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12)
    zcr = np.mean(np.diff(np.signbit(frames), axis=1), axis=1)
    power = abs(np.fft.rfft(frames * signal.windows.hann(size, sym=False), n=1024)) ** 2
    flatness = np.exp(np.mean(np.log(power + 1e-15), axis=1)) / (power.mean(axis=1) + 1e-15)
    return energy, zcr, flatness, power


def _vad(energy, periodicity, flatness, zcr, cfg):
    # A robust quiet-frame median/MAD seed, then nonspeech-only adaptation.
    quiet = energy[energy <= np.percentile(energy, 30)]
    seed = float(np.median(quiet) + np.median(abs(quiet - np.median(quiet))))
    all_voiced = np.percentile(energy, 85) - seed < 8 and np.percentile(periodicity, 85) > .7
    floor = np.full(len(energy), seed)
    estimate = seed
    for i, e in enumerate(energy):
        if periodicity[i] < .35 and e < estimate + 6:
            estimate += .015 * np.clip(e - estimate, -3, 3)
        floor[i] = estimate
    # Reuse the established duration/hysteresis policy on floor-relative energy.
    relative = energy - floor
    if all_voiced:
        relative += 14
    relative[energy < -75] = -240
    mask, score, _ = _activity(relative, periodicity, flatness, zcr, cfg)
    score[energy < -75] = 0
    return mask, score, floor


def _enhance(x, speech, cfg, mc, size, hop):
    if not cfg.denoise or np.count_nonzero(~speech) < 10:
        return x.copy(), {"applied": False, "reason": "disabled" if not cfg.denoise else "insufficient nonspeech"}
    _, _, spec = signal.stft(x, fs=16000, nperseg=size, noverlap=size-hop,
                             nfft=1024, boundary="zeros", padded=True)
    power = abs(spec) ** 2
    nonspeech = ~speech[np.minimum(np.arange(spec.shape[1]), len(speech)-1)]
    noise = np.maximum(np.median(power[:, nonspeech], axis=1), 1e-20)
    gain = np.ones_like(power)
    previous = np.zeros(len(noise))
    for t in range(spec.shape[1]):
        if nonspeech[t]:
            noise = .98 * noise + .02 * power[:, t]
        posterior = power[:, t] / noise
        prior = .85 * previous + .15 * np.maximum(posterior-1, 0)
        if mc.noise_method == "wiener":
            gain[:, t] = np.maximum(prior / (1+prior), mc.noise_gain_floor)
        else:
            gain[:, t] = np.sqrt(np.maximum(1-.7/posterior.clip(1e-10), mc.noise_gain_floor**2))
        previous = gain[:, t]**2 * posterior
    gain = ndimage.uniform_filter(gain, size=(3, 3), mode="nearest")
    _, y = signal.istft(spec * gain, fs=16000, nperseg=size, noverlap=size-hop,
                        nfft=1024, boundary=True)
    return y[:len(x)], {"applied": True, "method": mc.noise_method,
                        "gain_floor": mc.noise_gain_floor, "noise_update": .02,
                        "noise_frames": int(nonspeech.sum())}


def _pitch_estimators(x, cfg, hop):
    import pysptk
    size = round(cfg.pitch_window_ms * 16)
    filtered = signal.sosfiltfilt(signal.butter(2, min(2*cfg.f0_max, 7000), fs=16000, output="sos"), x)
    acf, periodicity = _pitch(_frames(filtered, size, hop), cfg)
    frames = _frames(x, size, hop)
    frames -= frames.mean(axis=1, keepdims=True)
    count = len(frames)
    # Exact fixed-support YIN difference via cross-correlation and prefix energies.
    maximum = min(int(16000/cfg.f0_min), size//2)
    minimum = max(2, int(16000/cfg.f0_max))
    support = size-maximum
    yin, strength, cepstrum = np.zeros(count), np.zeros(count), np.zeros(count)
    for i, f in enumerate(frames):
        cross = signal.correlate(f, f[:support], mode="valid", method="fft")[:maximum+1]
        prefix = np.r_[0., np.cumsum(f*f)]
        energy = prefix[np.arange(maximum+1)+support] - prefix[np.arange(maximum+1)]
        diff = np.maximum(prefix[support] + energy - 2*cross, 0)
        cmnd = np.ones(maximum+1)
        cmnd[1:] = diff[1:] * np.arange(1, maximum+1) / np.maximum(np.cumsum(diff[1:]), 1e-25)
        minima, _ = signal.find_peaks(-cmnd[minimum:])
        minima += minimum
        below = minima[cmnd[minima] < .2]
        if len(below):
            j = int(below[0])
            lag = float(j)
            if 0 < j < maximum:
                a,b,c = cmnd[j-1:j+2]
                if abs(a-2*b+c) > 1e-12:
                    lag += np.clip(.5*(a-c)/(a-2*b+c), -.5, .5)
            yin[i], strength[i] = 16000/lag, 1-cmnd[j]
        spec = abs(np.fft.rfft(f * signal.windows.hann(size, sym=False), n=4096))
        cep = np.fft.irfft(np.log(spec + max(spec.max()*1e-8, 1e-15)), n=4096)
        candidates, _ = signal.find_peaks(cep[minimum:maximum+1])
        if len(candidates) and np.dot(f,f) > 1e-15:
            j = minimum + candidates[np.argmax(cep[minimum+candidates])]
            if cep[j] > np.median(cep[minimum:maximum+1]) + 2*np.std(cep[minimum:maximum+1]):
                cepstrum[i] = 16000/j
    # SPTK can hang on a tiny search interval (e.g. 999–1000 Hz).
    # Give the backend an octave of support, then enforce the requested range.
    swipe = pysptk.swipe(np.ascontiguousarray(x, dtype=np.float64), 16000, hop,
                         min=min(cfg.f0_min,cfg.f0_max/2), max=cfg.f0_max, threshold=.25, otype="f0")
    swipe[(swipe<cfg.f0_min)|(swipe>cfg.f0_max)]=0
    return np.stack([acf, yin, swipe[:count], cepstrum], axis=1), periodicity, strength


def _consensus(estimates, periodicity, yin_strength, speech, cfg, mc):
    """Agreement clusters + DP; never average octave-disagreeing estimators."""
    n = len(estimates)
    frequencies, scores, agreement = np.zeros((n, 5)), np.zeros((n, 5)), np.zeros((n, 5), dtype=int)
    weights = np.c_[periodicity, yin_strength, np.full(n,.8), np.full(n,.55)]
    for t, row in enumerate(estimates):
        if not speech[t] or periodicity[t] < .45:
            continue
        for k, f in enumerate(row, 1):
            if not cfg.f0_min <= f <= cfg.f0_max:
                continue
            close = (row > 0) & (abs(1200*np.log2(np.maximum(row,1)/f)) <= mc.agreement_cents)
            # A time-domain and frequency-domain method must both support the value.
            if close.sum() < 2 or not close[:2].any() or not close[2:].any():
                continue
            frequencies[t,k] = np.exp(np.average(np.log(row[close]), weights=weights[t,close]))
            agreement[t,k] = close.sum()
            scores[t,k] = min(1., .5*periodicity[t] + .125*close.sum())
    emission = np.where(frequencies > 0, 1-scores, 20.)
    emission[:,0] = np.where(frequencies.max(axis=1)>0, .75, 0.)
    back = np.zeros((n,5),dtype=int)
    cost = emission[0]
    for t in range(1,n):
        prev, curr = frequencies[t-1], frequencies[t]
        jump = abs(np.log2(np.maximum(prev[:,None],1)/np.maximum(curr[None,:],1)))
        transition = .4*jump
        transition[jump>.5] += .5
        transition[(prev[:,None]==0)|(curr[None,:]==0)] = .18
        transition[0,0] = 0
        total = cost[:,None]+transition
        back[t] = total.argmin(axis=0)
        cost = emission[t]+total.min(axis=0)
    chosen = np.zeros(n,dtype=int)
    chosen[-1] = cost.argmin()
    for t in range(n-1,0,-1):
        chosen[t-1] = back[t,chosen[t]]
    f0 = frequencies[np.arange(n),chosen]
    status = np.where(f0>0,"observed","rejected").astype(object)
    for start,end in _runs(f0>0):
        if end-start >= 3:
            smooth = ndimage.median_filter(f0[start:end],size=3,mode="nearest")
            status[start:end][abs(smooth-f0[start:end])>.01] = "smoothed"
            f0[start:end] = smooth
    return f0, scores[np.arange(n),chosen], agreement[np.arange(n),chosen], status


def _formants(x, f0, snr, native_rate, cfg, mc, size, hop):
    ceiling = min(mc.formant_ceiling, native_rate*.475, 3400 if mc.channel=="telephone" else 7600)
    rate = round(2*ceiling)
    divisor = math.gcd(rate,16000)
    resampled = signal.resample_poly(x,rate//divisor,16000//divisor)
    pre = signal.lfilter([1,-np.exp(-2*np.pi*50/rate)],[1],resampled)
    # Build windows at the shared timeline (rounding a resampled hop would drift).
    length = round(cfg.frame_ms*rate/1000)
    padded = np.pad(pre,(length//2,length))
    centers = np.rint(np.arange(len(f0))*hop*rate/16000).astype(int)
    frames = np.lib.stride_tricks.sliding_window_view(padded,length)[centers]
    window = signal.windows.hamming(length,sym=False)
    candidates, raw, bandwidths, confidence = [], np.zeros((len(f0),3)), np.zeros((len(f0),3)), np.zeros((len(f0),3))
    coefficients = np.zeros((len(f0),mc.formant_order+1)); coefficients[:,0]=1
    for t,frame in enumerate(frames):
        valid = []
        if f0[t] > 0 and snr[t] > 6:
            a = _burg((frame-frame.mean())*window,mc.formant_order)
            coefficients[t] = a
            roots = np.roots(a)
            hz = np.angle(roots)*rate/(2*np.pi)
            bw = -rate/np.pi*np.log(np.maximum(abs(roots),1e-15))
            freq, response = signal.freqz([1],a,worN=1024,fs=rate)
            db = 20*np.log10(abs(response)+1e-15)
            peaks, _ = signal.find_peaks(db,prominence=2)
            measured=abs(np.fft.rfft((frame-frame.mean())*window,n=2048))**2
            measured=ndimage.gaussian_filter1d(measured,sigma=max(1,f0[t]*.6/(rate/2048)))
            for f,b in sorted(zip(hz,bw)):
                if 90 < f < ceiling-50 and 20 < b < 550 and len(peaks):
                    nearest = peaks[np.argmin(abs(freq[peaks]-f))]
                    index=min(len(measured)-1,round(f/rate*2048))
                    radius=max(2,round(max(b,f0[t])/rate*2048))
                    local=measured[max(0,index-radius):min(len(measured),index+radius+1)]
                    support = (abs(freq[nearest]-f) < max(80,b) and
                               measured[index] > .35*np.max(local) and measured[index] > np.median(measured))
                    q = float(np.clip((1-b/650)*min(1,snr[t]/25),0,1))
                    if support and q>.2:
                        valid.append({"hz":round(float(f),2),"bandwidth_hz":round(float(b),2),"confidence":round(q,3)})
            for j,c in enumerate(valid[:3]):
                raw[t,j],bandwidths[t,j],confidence[t,j] = c["hz"],c["bandwidth_hz"],c["confidence"]
        candidates.append(valid)
    # Ordered candidate association and scalar Kalman + RTS within observed runs.
    associated = np.zeros_like(raw); variance = np.zeros_like(raw)
    last = np.zeros(3)
    for t,cs in enumerate(candidates):
        if f0[t]==0:
            last[:]=0
            continue
        lower = 90.
        for j in range(3):
            options = [c for c in cs if c["hz"]>lower and c["hz"] < [1300,3400,ceiling][j]]
            if not options:
                continue
            c = min(options,key=lambda c: abs(c["hz"]-last[j]) if last[j]>0 else c["hz"])
            if last[j]>0 and abs(c["hz"]-last[j])>700:
                continue
            associated[t,j]=c["hz"]; last[j]=c["hz"]; lower=c["hz"]+100
            variance[t,j]=(max(40,c["bandwidth_hz"])/max(.2,c["confidence"]))**2
    tracked = np.zeros_like(raw); uncertainty=np.zeros_like(raw)
    for j in range(3):
        for start,end in _runs(associated[:,j]>0):
            obs=associated[start:end,j]; r=variance[start:end,j]
            state=obs.copy(); p=r.copy(); q=mc.kalman_process_sd_hz**2
            for k in range(1,len(obs)):
                prior=p[k-1]+q; gain=prior/(prior+r[k])
                state[k]=state[k-1]+gain*(obs[k]-state[k-1]);p[k]=(1-gain)*prior
            for k in range(len(obs)-2,-1,-1):
                gain=p[k]/(p[k]+q)
                state[k]+=gain*(state[k+1]-state[k]);p[k]+=gain**2*(p[k+1]-p[k]-q)
            tracked[start:end,j]=state;uncertainty[start:end,j]=np.sqrt(np.maximum(p,0))
    # Smoothing independent states can cross; reject the affected observations.
    crossing = (tracked[:,1]>0)&(tracked[:,0]>=tracked[:,1]) | (tracked[:,2]>0)&(tracked[:,1]>=tracked[:,2])
    tracked[crossing]=0;uncertainty[crossing]=0
    return {"candidates":candidates,"observed":_nullable(raw,1,True),
            "tracked":_nullable(tracked,1,True),"bandwidth_hz":_nullable(bandwidths,1,True),
            "confidence":confidence.round(3).tolist(),"uncertainty_hz":_nullable(uncertainty,1,True),
            "status":np.where(tracked>0,"smoothed","rejected").tolist(),
            "lpc":coefficients.round(7).tolist(),"analysis_rate":rate,"ceiling_hz":ceiling}, tracked


def _spectral(x, power, hop, count, bandwidth, pitch_floor):
    import librosa
    frequency=np.fft.rfftfreq(1024,1/16000)
    power=power.copy();power[:,frequency>bandwidth]=0
    mass=power.sum(axis=1)+1e-25
    centroid=(power*frequency).sum(axis=1)/mass
    spread=np.sqrt((power*(frequency-centroid[:,None])**2).sum(axis=1)/mass)
    rolloff=frequency[np.argmax(np.cumsum(power,axis=1)>=.85*mass[:,None],axis=1)]
    magnitude=np.sqrt(power);norm=magnitude/(np.linalg.norm(magnitude,axis=1,keepdims=True)+1e-15)
    flux=np.r_[0.,np.sqrt(np.sum(np.maximum(np.diff(norm,axis=0),0)**2,axis=1))]
    mel=librosa.filters.mel(sr=16000,n_fft=1024,n_mels=32,fmin=30,fmax=bandwidth)
    mfcc=dct(np.log(np.maximum(power@mel.T,1e-15)),type=2,norm="ortho",axis=1)[:,:13]
    logf=np.log2(np.maximum(frequency,1)); band=(frequency>=100)&(frequency<=bandwidth)
    centered=logf[band]-logf[band].mean()
    tilt=(10*np.log10(power[:,band]+1e-15)@centered)/np.dot(centered,centered)
    cqt_floor=min(60.,pitch_floor/2)
    bins=max(12,int(np.floor(12*np.log2(bandwidth/cqt_floor))))
    # Actual variable-window CQT, not relabelled STFT bins; native hop then align.
    padding=max(0,round(512*2**np.ceil(np.log2(bandwidth/cqt_floor)))-len(x))
    padded=np.pad(x,(0,padding))
    cqt=abs(librosa.cqt(padded,sr=16000,hop_length=256,fmin=cqt_floor,n_bins=bins,
                       bins_per_octave=12,pad_mode="constant"))
    source_times=np.arange(cqt.shape[1])*256/16000
    target_times=np.arange(count)*hop/16000
    cq=np.stack([np.interp(target_times,source_times,row) for row in cqt],axis=1)
    bands=np.stack([power[:,(frequency>=a)&(frequency<b)].sum(axis=1)/mass
                    for a,b in [(0,300),(300,1000),(1000,3000),(3000,8001)]],axis=1)
    return {"centroid_hz":centroid.round(2).tolist(),"spread_hz":spread.round(2).tolist(),
            "rolloff85_hz":rolloff.round(2).tolist(),"flux":flux.round(4).tolist(),
            "tilt_db_per_octave":tilt.round(3).tolist(),"band_energy_ratios":bands.round(4).tolist(),
            "mfcc":mfcc.round(4).tolist(),"delta_mfcc":np.gradient(mfcc,axis=0).round(4).tolist() if count>1 else np.zeros_like(mfcc).tolist(),
            "stft_db":(10*np.log10(power+1e-15)).round(2).tolist(),"stft_frequency_hz":frequency.tolist(),
            "cqt_db":(20*np.log10(cq+1e-12)).round(2).tolist(),
            "cqt_frequency_hz":librosa.cqt_frequencies(bins,fmin=cqt_floor).round(2).tolist(),
            "metadata":{"stft_fft":1024,"mel_bands":32,"mfcc_coefficients":13,
                        "mfcc_log":"natural","mfcc_dct":"II orthonormal",
                        "delta_mfcc_units":"coefficient/frame","cqt_native_hop":256,
                        "cqt_alignment":"linear interpolation of magnitudes","cqt_bins_per_octave":12,
                        "cqt_short_input_padding_samples":padding,"cqt_floor_hz":cqt_floor,
                        "bandwidth_hz":bandwidth,"units":{"mfcc":"log-power DCT coefficients","flux":"L2 difference of unit magnitude spectra",
                        "stft_db":"unnormalized FFT power dB","cqt_db":"magnitude dB","band_energy_ratios":"fraction of frame power"}}},flux


def _landmarks(energy,flux,f0,formants,speech,hop):
    seconds=hop/16000
    derivative=np.r_[0.,np.diff(energy)]
    voiced=f0>0; transitions=np.r_[False,voiced[1:]!=voiced[:-1]]
    formant_change=np.r_[0.,np.sum(abs(np.diff(formants[:,:2],axis=0)),axis=1)/600]
    formant_change[np.r_[True,np.any((formants[1:,:2]==0)|(formants[:-1,:2]==0),axis=1)]]=0
    burst=np.maximum(derivative,0)/10
    score=.3*np.minimum(abs(derivative)/10,3)+.3*flux+.2*np.minimum(formant_change,3)+.4*transitions+.2*burst
    peaks,_=signal.find_peaks(score,height=.45,prominence=.18,distance=max(1,round(.04/seconds)))
    events=[]
    for i in peaks:
        if not speech[max(0,i-1):min(len(speech),i+2)].any():
            continue
        kind="voicing onset" if transitions[i] and voiced[i] else "voicing offset" if transitions[i] else "burst candidate" if burst[i]>.8 else "spectral transition"
        events.append({"time":round(i*seconds,4),"label":kind,"score":round(float(score[i]),3)})
    for start,end in _runs(speech):
        events.extend([{"time":round(start*seconds,4),"label":"speech onset","score":1.},
                       {"time":round(end*seconds,4),"label":"speech offset","score":1.}])
    for start,end in _runs(speech&~voiced):
        if (end-start)*seconds>=.04:
            events.append({"time":round(start*seconds,4),"end":round(end*seconds,4),"label":"unvoiced/frication candidate","score":.5})
    sonority=ndimage.gaussian_filter1d(energy,sigma=max(1,.025/seconds))
    nuclei,_=signal.find_peaks(sonority,prominence=2,distance=max(1,round(.1/seconds)))
    nuclei=[{"time":round(i*seconds,4),"intensity_dbfs":round(float(sonority[i]),2)} for i in nuclei if speech[i] and voiced[i]]
    return sorted(events,key=lambda e:e["time"]),nuclei,score


def _pulse_measurements(x,f0,confidence,periodicity,snr,cfg,mc,hop,disable_reason=None):
    qualified=(f0>0)&(confidence>=.75)&(periodicity>.8)&(snr>=mc.min_quality_snr_db)
    records=[]; pulse_times=[]
    if disable_reason:
        return [{"status":"rejected","reason":disable_reason}],[]
    for start,end in _runs(qualified):
        a,b=start*hop,min(len(x),end*hop)
        if (b-a)/16000 < mc.min_quality_seconds:
            continue
        pitches=f0[start:end]
        reasons=[]
        if np.std(pitches)/np.mean(pitches)>.04:
            reasons.append("unstable pitch; sustained phonation required")
        frame=x[a:b]
        block_rms=np.sqrt(np.mean(_frames(frame,round(.03*16000),round(.03*16000))**2,axis=1))
        middle=block_rms[1:-1]
        if len(middle)>2 and np.std(middle)/(np.mean(middle)+1e-15)>.2:
            reasons.append("unstable amplitude envelope")
        p=16000/np.median(pitches)
        # Pick a strong pulse then follow one local maximum per predicted period.
        first=int(np.argmax(frame[:min(len(frame),round(2*p))]))
        indices=[float(first)]
        while indices[-1]+1.25*p<len(frame):
            expected=indices[-1]+p
            lo=max(1,round(expected-.22*p));hi=min(len(frame)-1,round(expected+.22*p)+1)
            k=lo+int(np.argmax(frame[lo:hi])); y0,y1,y2=frame[k-1:k+2]
            curvature=y0-2*y1+y2
            refined=k+float(np.clip(.5*(y0-y2)/curvature,-.5,.5)) if abs(curvature)>1e-20 else float(k)
            indices.append(refined)
            # Local F0 supplies the next search radius without changing the waveform.
            p=16000/f0[min(end-1,start+round(refined/hop))]
        times=np.array(indices)/16000
        periods=np.diff(times)
        amplitudes=np.array([np.ptp(frame[round(indices[i]):round(indices[i+1])+1]) for i in range(len(indices)-1)])
        valid=(periods>=1/cfg.f0_max)&(periods<=1/cfg.f0_min)&(amplitudes>1e-12)
        adjacent=valid[:-1]&valid[1:]
        adjacent &= np.maximum(periods[:-1],periods[1:])/np.maximum(np.minimum(periods[:-1],periods[1:]),1e-15)<=mc.max_period_factor
        amp_adjacent=adjacent & (np.maximum(amplitudes[:-1],amplitudes[1:])/np.maximum(np.minimum(amplitudes[:-1],amplitudes[1:]),1e-15)<=mc.max_amplitude_factor)
        if adjacent.sum()<20 or amp_adjacent.sum()<20:
            reasons.append("too few accepted adjacent cycles")
        if valid.sum()<.9*len(valid) or adjacent.sum()<.8*max(1,len(adjacent)):
            reasons.append("pulse sequence failed cycle consistency")
        record={"start":round(a/16000,4),"end":round(b/16000,4),"branch":"minimal",
                "context":"stable phonation candidate; not verified sustained vowel",
                "status":"rejected" if reasons else "observed","reason":"; ".join(reasons) or None,
                "pulse_times":(times+a/16000).round(7).tolist(),"period_seconds":periods.round(8).tolist(),
                "cycle_amplitudes":amplitudes.round(7).tolist(),"accepted_period_pairs":int(adjacent.sum()),
                "accepted_amplitude_pairs":int(amp_adjacent.sum()),"jitter_local":None,"shimmer_local":None,"shimmer_db":None}
        if not reasons:
            record.update(jitter_local=float(np.mean(abs(np.diff(periods))[adjacent])/np.mean(periods[valid])),
                          shimmer_local=float(np.mean(abs(np.diff(amplitudes))[amp_adjacent])/np.mean(amplitudes[valid])),
                          shimmer_db=float(np.mean(abs(20*np.log10(amplitudes[1:]/np.maximum(amplitudes[:-1],1e-15)))[amp_adjacent])))
            def quotient(values,width,pairs):
                deviations=[];half=width//2
                for i in range(half,len(values)-half):
                    if valid[i-half:i+half+1].all() and pairs[i-half:i+half].all():
                        deviations.append(abs(values[i]-np.mean(values[i-half:i+half+1])))
                return float(np.mean(deviations)/np.mean(values[valid])) if deviations else None
            record["jitter_rap"]=quotient(periods,3,adjacent)
            record["jitter_ppq5"]=quotient(periods,5,adjacent)
            record["jitter_ddp"]=3*record["jitter_rap"] if record["jitter_rap"] is not None else None
            for width in (3,5,11):
                record[f"shimmer_apq{width}"]=quotient(amplitudes,width,amp_adjacent)
            pulse_times.extend(record["pulse_times"])
        records.append(record)
    if not records:
        records=[{"status":"rejected","reason":"no sufficiently long, high-SNR, stable voiced interval"}]
    return records,pulse_times


def _iaif_frame(frame,mc):
    """Two-pass low-order source / higher-order tract inverse filtering."""
    x=frame-frame.mean();window=signal.windows.hann(len(x),sym=False)
    ramp=np.linspace(-x[0],x[0],mc.iaif_tract_order+1)
    extended=np.r_[ramp,x];offset=len(ramp)
    gross=_burg(x*window,1)
    tract1=_burg(signal.lfilter(gross,[1],extended)[offset:]*window,mc.iaif_tract_order)
    initial=signal.lfilter([1],[1,-mc.integration_leak],signal.lfilter(tract1,[1],extended))[offset:]
    source=_burg(initial*window,mc.iaif_source_order)
    corrected=signal.lfilter([1],[1,-mc.integration_leak],signal.lfilter(source,[1],extended))[offset:]
    tract2=_burg(corrected*window,mc.iaif_tract_order)
    derivative=signal.lfilter(tract2,[1],extended)
    flow=signal.lfilter([1],[1,-mc.integration_leak],derivative)[offset:]
    return flow,derivative[offset:],tract2,source


def _source_analysis(x,quality,cfg,mc,hop):
    size=round(max(cfg.pitch_window_ms,50)*16)
    frames=_frames(x,size,hop);window=signal.windows.hann(size,sym=False)
    flow=np.zeros(len(x)+2*size);derivative=flow.copy();weights=flow.copy();records=[]
    eligible=np.zeros(len(frames),dtype=bool)
    for r in quality:
        if r["status"]=="observed":
            # Avoid inverse-filtering frames that straddle the qualified interval.
            start=math.ceil((r["start"]*16000+size/2)/hop)
            end=math.floor((r["end"]*16000-size/2)/hop)
            eligible[start:end]=True
    for i in np.flatnonzero(eligible):
        g,dg,tract,source=_iaif_frame(frames[i],mc)
        if not np.isfinite(g).all() or not np.isfinite(dg).all():
            continue
        g-=np.mean(g)
        start=i*hop
        flow[start:start+size]+=g*window;derivative[start:start+size]+=dg*window;weights[start:start+size]+=window
        records.append({"time":round(i*hop/16000,4),"tract_lpc":tract.round(7).tolist(),"source_lpc":source.round(7).tolist(),"status":"observed"})
    offset=size//2;weight=weights[offset:offset+len(x)]
    flow=flow[offset:offset+len(x)]/np.maximum(weight,1e-12)
    derivative=derivative[offset:offset+len(x)]/np.maximum(weight,1e-12)
    metrics=[]
    for r in quality:
        if r["status"]!="observed":
            continue
        values=[];opening=[];closures=[];h1h2=[]
        for a,b in zip(r["pulse_times"][:-1],r["pulse_times"][1:]):
            start,end=round(a*16000),round(b*16000)
            if end>start and np.all(weight[start:end]>1e-8):
                amplitude=np.ptp(flow[start:end]);closing=abs(np.min(derivative[start:end]))
                if closing>1e-15:
                    values.append(float(amplitude/(closing*(b-a)*16000)))
                    cycle=flow[start:end]
                    opening.append(float(np.mean(cycle>np.min(cycle)+.1*amplitude)))
                    closures.append(float((start+np.argmin(derivative[start:end]))/16000))
                    spectrum=abs(np.fft.rfft(cycle-cycle.mean()))
                    if len(spectrum)>2 and spectrum[1]>1e-15 and spectrum[2]>1e-15:
                        h1h2.append(float(20*np.log10(spectrum[1]/spectrum[2])))
        metrics.append({"start":r["start"],"end":r["end"],"naq":float(np.median(values)) if values else None,
                        "status":"observed" if values else "rejected","cycles":len(values),
                        "open_phase_fraction_proxy":float(np.median(opening)) if opening else None,
                        "closure_time_candidates":closures,"h1_h2_db":float(np.median(h1h2)) if h1h2 else None,
                        "interpretation":"IAIF-derived proxies; acoustic pulses and derivative minima are not EGG-verified closures"})
    return {"branch":"minimal","method":"two-pass IAIF with Burg LPC","frames":records,"intervals":metrics,
            "reason":None if records else "no qualified source-analysis interval",
            "waveform_units":"uncalibrated relative flow; not physical volume velocity",
            "display":{"flow":[float(np.max(abs(c))) for c in np.array_split(flow,min(1200,len(flow)))],
                       "derivative":[float(np.max(abs(c))) for c in np.array_split(derivative,min(1200,len(derivative)))],
                       "representation":"peak magnitude envelopes; independent scales"}},flow,derivative


def analyze_recommended(audio,sample_rate,cfg=None,measurements=None,*,quality_exclusion=None,return_packet=False):
    cfg=cfg or Config();mc=measurements or MeasurementConfig();cfg.validate();mc.validate()
    original=np.asarray(audio,dtype=float)
    if original.ndim not in {1,2} or not original.size or not np.isfinite(original).all():
        raise ValueError("Audio must be a finite mono or channels-last waveform")
    if sample_rate<4000:
        raise ValueError("Sample rate must be at least 4 kHz")
    channels=original.shape[1] if original.ndim==2 else 1
    selected=int(np.argmax(np.mean(original**2,axis=0))) if original.ndim==2 else 0
    x=original[:,selected] if original.ndim==2 else original.copy()
    if sample_rate!=16000:
        divisor=math.gcd(sample_rate,16000);x=signal.resample_poly(x,16000//divisor,sample_rate//divisor)
    if len(x)<320:
        raise ValueError("Audio must contain at least 20 ms")
    size,hop=round(cfg.frame_ms*16),round(cfg.hop_ms*16)
    highpass=min(30.,cfg.f0_min*.45)
    minimal=signal.sosfiltfilt(signal.butter(2,highpass,btype="highpass",fs=16000,output="sos"),x-x.mean())
    # Physical bandwidth is limited by native rate and declared telephone channel.
    bandwidth=min(7600.,sample_rate*.475,3400. if mc.channel=="telephone" else 7600.)
    analysis_branch=signal.sosfiltfilt(signal.butter(3,bandwidth,fs=16000,output="sos"),minimal)
    energy,zcr,flatness,power=_cues(analysis_branch,size,hop)
    estimates,periodicity,yin_strength=_pitch_estimators(analysis_branch,cfg,hop)
    speech,confidence,floor=_vad(energy,periodicity,flatness,zcr,cfg)
    enhanced,enhancement=_enhance(analysis_branch,speech,cfg,mc,size,hop)
    if enhancement["applied"]:
        estimates,periodicity,yin_strength=_pitch_estimators(enhanced,cfg,hop)
        enhanced_energy,zcr,flatness,power=_cues(enhanced,size,hop)
        speech,confidence,_=_vad(enhanced_energy,periodicity,flatness,zcr,cfg)
    snr=energy-floor
    f0,pitch_confidence,agreement,pitch_status=_consensus(estimates,periodicity,yin_strength,speech,cfg,mc)
    formant_record,formants=_formants(enhanced,f0,snr,sample_rate,cfg,mc,size,hop)
    spectral,flux=_spectral(enhanced,power,hop,len(f0),bandwidth,cfg.f0_min)
    # Relative intensity is from the unenhanced branch; never claim calibrated SPL.
    intensity_size=round(max(25,3000/cfg.f0_min)*16)
    intensity=20*np.log10(np.sqrt(np.mean(_frames(minimal,intensity_size,hop)**2,axis=1))+1e-12)
    landmarks,nuclei,boundary_score=_landmarks(intensity,flux,f0,formants,speech,hop)
    exclusion=quality_exclusion or ("telephone channel: perturbation and glottal measurements withheld" if mc.channel=="telephone" else None)
    if np.max(abs(x))>=.999:
        exclusion="full-scale input: clipping may invalidate pulse/source measurements"
    quality,pulses=_pulse_measurements(minimal,f0,pitch_confidence,periodicity,snr,cfg,mc,hop,exclusion)
    glottal,flow,derivative=_source_analysis(minimal,quality,cfg,mc,hop)
    sample_indices=np.minimum(np.floor(np.arange(len(x))/hop+.5).astype(int),len(speech)-1)
    mask=speech[sample_indices];gain=mask.astype(float);segments=[]
    for a,b in _runs(mask):
        idx=np.unique(sample_indices[a:b]);fade=min(80,(b-a)//2)
        gain[a:a+fade]*=np.linspace(0,1,fade);gain[b-fade:b]*=np.linspace(1,0,fade)
        segments.append({"start":round(a/16000,4),"end":round(b/16000,4),
                         "confidence":round(float(confidence[idx].mean()),3),"voiced_fraction":round(float(np.mean(f0[idx]>0)),3)})
    labels=np.where(~speech,"nonspeech",np.where(f0>0,"voiced","unvoiced"))
    regions=[{"start":round(a/16000,4),"end":round(b/16000,4),"label":label}
             for label in ("nonspeech","voiced","unvoiced") for a,b in _runs(labels[sample_indices]==label)]
    mfcc=np.array(spectral["mfcc"])
    changes=_speaker_changes(mfcc[:,1:9],f0,formants,speech,cfg)
    speech_seconds=float(mask.sum()/16000)
    stats={"speech_seconds":speech_seconds,"voiced_fraction":float(np.mean(f0[speech]>0)) if speech.any() else 0.,
           "median_f0":float(np.median(f0[f0>0])) if np.any(f0>0) else None,
           "noise_floor_db":float(np.median(floor)),"denoise_applied":enhancement["applied"],
           "pitch_coverage":float(np.mean(f0[speech]>0)) if speech.any() else 0.,
           "formant_coverage":np.mean(formants[speech]>0,axis=0).round(4).tolist() if speech.any() else [0.,0.,0.],
           "estimated_nuclei":len(nuclei),"nuclei_per_elapsed_second":len(nuclei)/(len(x)/16000),
           "nuclei_per_speech_second":len(nuclei)/speech_seconds if speech_seconds else None,
           "qualified_quality_intervals":sum(r["status"]=="observed" for r in quality)}
    warnings=["Deterministic activity can confuse speech with coughs, music or tonal interference.",
              "Confidence values are acoustic scores, not calibrated probabilities.",
              "Landmarks and vowel-like nuclei are candidates; no phoneme or syllable identities are inferred.",
              "Stable phonation candidates are not verified sustained vowels. Perturbation and IAIF values are experimental."]
    if exclusion:
        warnings.append(exclusion)
    if sample_rate<16000 or mc.channel=="telephone":
        warnings.append("Upper spectrum is limited by native/declared channel bandwidth; upsampling restores no information.")
    data={"method":"recommended","config":asdict(cfg),"measurement_config":asdict(mc),"sample_rate":16000,"duration":len(x)/16000,
          "segments":segments,"regions":sorted(regions,key=lambda r:r["start"]),"speaker_changes":changes,"warnings":warnings,"stats":stats,
          "frames":{"time":(np.arange(len(f0))*hop/16000).round(4).tolist(),"energy_db":energy.round(2).tolist(),
                    "intensity_dbfs":intensity.round(2).tolist(),"noise_floor_db":floor.round(2).tolist(),"snr_proxy_db":snr.round(2).tolist(),
                    "zcr":zcr.round(4).tolist(),"flatness":flatness.round(4).tolist(),"periodicity":periodicity.round(4).tolist(),
                    "confidence":confidence.round(4).tolist(),"f0":f0.round(2).tolist(),"pitch_confidence":pitch_confidence.round(4).tolist(),
                    "pitch_agreement":agreement.tolist(),"pitch_status":pitch_status.tolist(),"pitch_estimators":_nullable(estimates,2,True),
                    "formants":formants.round(1).tolist(),"label":labels.tolist(),"lpc":formant_record["lpc"],
                    "boundary_score":boundary_score.round(4).tolist()},
          "formant_analysis":formant_record,"spectral":spectral,"landmarks":landmarks,"nuclei":nuclei,"voice_quality":quality,
          "pulse_times":pulses,"glottal":glottal,
          "pauses":[r for r in regions if r["label"]=="nonspeech"],
          "provenance":{"native_sample_rate":sample_rate,"input_channels":channels,"selected_channel":selected,
                        "channel_selection":"highest whole-excerpt mean-square energy","highpass_hz":highpass,"filter_phase":"offline zero phase",
                        "bandwidth_hz":bandwidth,"measurement_branch":"DC removed + gentle highpass; no enhancement",
                        "estimator_branch":"bandlimited + optional conservative enhancement","enhancement":enhancement,
                        "pitch_estimator_order":["filtered ACF","YIN","SPTK SWIPE prime","cepstrum"],
                        "swipe_backend_range_hz":[min(cfg.f0_min,cfg.f0_max/2),cfg.f0_max],
                        "swipe_range_rule":"at least one octave of backend support; candidates outside the requested range rejected",
                        "agreement_rule":"at least two estimators, including time and spectral evidence",
                        "intensity_reference":"dB relative to digital full scale; no SPL calibration",
                        "intensity_window_ms":intensity_size/16,"formant_tracker":"ordered association + scalar Kalman/RTS; no gap filling",
                        "snr_definition":"frame energy minus adaptive nonspeech floor; proxy, not measured SNR"}}
    waves={"original":x,"minimal":minimal,"speech":x*gain,"enhanced":enhanced,
           "glottal":flow,"glottal_derivative":derivative}
    from .representation import encode_streams, decode_packet, packet_summary, MODES
    packet=encode_streams(minimal,f0,periodicity,pitch_confidence,labels,cfg,data["provenance"])
    decoded={mode:decode_packet(packet,mode) for mode in MODES}
    data["representation"]=packet_summary(packet,minimal,decoded)
    waves.update({f"packet_{mode}":wave for mode,wave in decoded.items()})
    result=(data,{key:value.astype(np.float32) for key,value in waves.items()})
    return (*result,packet) if return_packet else result
