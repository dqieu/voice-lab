# Voice analysis and reconstruction

See the [main README](../README.md) for installation and [dataset imports](datasets.md) for folder and CSV use. The [supplied research report](references/deterministic-voice-features.pdf) and [implementation map](analysis.md) explain the baseline.

## Voice analysis feeding the six streams

This implements the final recommended stack in the supplied *Deterministic
Signal-Processing Methods for Extracting Human Voice Features from Audio* report.
Its measurements remain part of the six-stream pipeline, rather than a separate
method to select. LPC remains necessary for tract-envelope synthesis, formant
measurement and IAIF; a separate LPC roundtrip playback option is redundant.

| Report component | Implementation and web inspection |
|---|---|
| Decode and QA | Native rate, input channels, selected channel, DC removal, full-scale check; profile and provenance panel |
| Minimal branch | Gentle offline highpass below the pitch floor; intensity, acoustic pulse marks, perturbation and IAIF; separate playback |
| Enhanced branch | Channel-aware lowpass, adaptive nonspeech PSD, optional conservative Wiener or spectral subtraction, smoothed gains with floor; separate playback |
| Speech / voicing | Energy, normalized periodicity, ZCR and flatness; quiet-frame median/MAD seed, nonspeech noise-floor adaptation, hysteresis and duration rules; waveform and intervals |
| Multi-estimator F0 | Filtered ACF candidate/path, fixed-support YIN CMND, SPTK SWIPE prime and real cepstrum; estimator overlay checkbox |
| Pitch consensus | 50-cent agreement clusters require at least two estimators including time and spectral evidence; DP continuity with octave penalties; within-run median smoothing |
| Burg formants | Resample relative to configured/native/channel ceiling, 50-Hz pre-emphasis, Hamming window, Burg recursion, pole bandwidth and measured spectral-support checks |
| Formant continuity | Ordered candidate association, confidence-dependent observation variance, Kalman filter and offline RTS smoother; observed/tracked/null values, uncertainty and status in JSON |
| Spectrum | Actual variable-window CQT, linear-frequency STFT, 32-mel-band / 13-coefficient MFCC, delta MFCC, centroid, spread, rolloff, flux, tilt and band-energy ratios; spectral selector and full JSON |
| Prosody / segmentation | Minimal-branch relative intensity, pauses, energy/flux/formant/voicing landmarks, bursts, broad unvoiced/frication regions and estimated vowel-like nuclei; plots and bounded-playback event buttons |
| Voice perturbation | Acoustic cycle maxima, interpolated timing, peak-to-peak amplitudes, period/amplitude exclusion factors, local jitter/shimmer, shimmer dB, RAP, PPQ5, DDP and APQ3/5/11; qualified-interval results or rejection reasons |
| Source analysis | Two-pass IAIF with low-order source and higher-order tract Burg models, 0.99 leaky integration, qualified-interval overlap-add, flow/derivative WAVs, NAQ, H1-H2, closure-time and open-phase proxies; peak-envelope plot and source playback |
| Confidence / provenance | Branches, settings, windows, physical bandwidth, estimator agreement, coverage, observed/smoothed/rejected status and explicit missing values; JSON and parameters panel |
| Evaluation | Reference-bounded activity, boundary, pitch and formant metrics when references are supplied; reconstruction error and component ablations |

Defaults are 25 ms / 10 ms analysis and pitch windows of at least three periods
at the chosen floor (50 ms at 60 Hz). CQT uses 12 bins/octave starting at half
the pitch floor, a native 256-sample hop, and explicit magnitude alignment to the
shared frame timeline. Frequency-dependent windows need longer support than an
STFT; short-input padding is recorded. The formant ceiling is configurable from
2,000 to 7,600 Hz and is further limited by native bandwidth and channel profile.
Default Burg order is 10. Final JSON records analysis rates and parameters.

Pitch disagreement does not create an intermediate frequency: 100 Hz versus
200 Hz is not averaged into 150 Hz. Rejected pitch is represented by zero in
plot-compatible `frames.f0`, alongside `pitch_status: rejected`; candidate and
physical measurement arrays use nulls. Kalman smoothing never fills unobserved
formants or bridges silence. Coverage is reported alongside values.
Narrow pitch queries give the SWIPE backend at least one octave of search support
to avoid its observed narrow-range hang; estimates outside the requested range
are rejected. Backend bounds and this rule are recorded in provenance.

Pulse/source qualification requires at least 300 ms, high periodicity,
estimator agreement, a noise-floor energy margin of at least 20 dB, and stable
F0/amplitude. That margin is an **SNR proxy**, not independently measured SNR.
Telephone, detected full-scale input, synthetic reverberation and codec fallback
withhold perturbation/source outputs. Noise reduction never processes the
measurement waveform. Stable phonation candidates are not human-verified
sustained vowels; connected-speech measurements remain experimental.

The profile changes observability assumptions, not the audio's physical channel.
**Telephone** caps estimator/formant bandwidth at 3.4 kHz and withholds pulse
and glottal measurements. Select **Wiener** or **Spectral subtraction** in
**Noise reduction** to enhance the estimator branch. The redundant Noisy field
profile is absent from the UI; the API retains it for compatibility. With too
few nonspeech frames, enhancement abstains and records its reason. Changing
the decoded sampling rate never restores missing channel information.

## Samples, listening and export

The app starts with a generated synthetic source/filter control and saved local
imports. Add datasets in the UI or specify folders at launch:

```bash
voice-lab serve --dataset timit=/path/to/local/timit --dataset ptdb=/path/to/local/ptdb
```

`--out`, `--registry` and `--port` configure storage and the local server. Excerpts
are 1–45 seconds. Soundfile decodes locally, with optional ffmpeg fallback for
unsupported codecs. Seeds select the same recording and excerpt within an
unchanged catalog. White-noise conditions are 20, 10 and 0 dB **whole-excerpt**
SNR, including silence. Reverb uses a seeded nominal 0.45-second decay with a
truncated tail. Hum adds 100/200-Hz tones. One common gain scales speech and
interference if required to avoid additional clipping.

- **Original:** selected channel, resampled to 16 kHz; stress applied beforehand.
- **Speech only:** original values inside detected intervals, with 5-ms edge
  fades and silence elsewhere; original timing is retained.
- **Minimal:** DC removal plus gentle highpass, without enhancement.
- **Filtered / enhanced:** the estimator branch.
- **IAIF source estimate:** experimental flow estimate in qualified intervals;
  silence where source measurements are withheld. Its amplitude is uncalibrated.
  Playback and the source plot appear only when at least one interval qualifies;
  otherwise Advanced analysis shows a single withholding reason.

Click speech intervals or landmarks for bounded playback, use the waveform or
audio player's scrubber to navigate, and select an output row to listen at the
same position.
Analysis JSON and Segment WAVs export the current recording. Float WAV preserves
gain; each segment WAV contains original waveform values at the detected interval.
The original, enhanced, minimal and qualified glottal tracks remain separately
available in its local artifact directory.

## Find the highest reconstruction error

Below the playback output table, expand **Find sample with highest RMS error to
Minimal**. Choose a six-stream reconstruction output and test 10, 25, 50, 100,
or all indexed recordings in the selected dataset. Selecting a reconstruction
playback row also selects its search metric; otherwise Parametric is the default.

The search tests unique indexed recordings without replacement, with one seeded
excerpt per recording under the current excerpt length, listening condition,
noise reduction and analysis settings. It ranks the same unnormalized waveform
RMS error against Minimal displayed in the table. Loudness affects this metric;
it is not perceptual quality or normalized reconstruction error. Full packet
differences normally reflect numerical roundoff.

Progress reports tested/skipped counts and the best score. **Stop & show best so
far** finishes the current excerpt, then loads the best successful result.
Only the winning sample's audio, JSON and packet are saved. Its JSON/packet
metadata records search scope, counts, settings and an exact replay request.
Reanalyzing the winner with the displayed seed retains its recording index;
shuffle or choosing another dataset restores ordinary random selection.

Highest means highest among successfully tested excerpts in the current pool,
not every recording/time window in the corpus. Folder/CSV imports use their indexed pools. Changing files or reindexing a catalog can change index
based replay. Searches and exports remain local.

## Six-stream representation and standalone synthesis

The pipeline exports `representation.npz`, a versioned, self-contained
packet with six aligned streams at the configured hop (100 frames/s by default).
The **Six-stream packet** export downloads it. The playback output table selects
six decoder variants and reports waveform error against the minimal target.
Open **Advanced analysis** to inspect LSFs, gain, modeled HNR and band
aperiodicity alongside the distinct formant and intensity measurements.

| Stream | Packet fields and synthesis meaning |
|---|---|
| F0 | `f0`, `pitch_confidence`, frame times; harmonic frequencies are integer multiples of tracked pitch |
| VOICE | `voice` states, `periodicity`, `hnr_db`; voiced state enables modeled harmonics, but never removes the remainder |
| TRACT | `lsf` in radians, `tract_status`, `tract_expansion`; decoder derives its all-pole filter from LSFs. `tract_lpc_audit` is retained for inspection only |
| AMP | `gain`: LPC-excitation RMS, `rms`: output-frame RMS, `cycle_rms`: voiced central-cycle RMS proxy (NaN when unavailable) |
| HARM | `harmonic_amplitudes`, `harmonic_phases`, harmonic and remainder band power, `aperiodicity`; amplitudes refer to normalized excitation |
| RESID | `residual`: normalized LPC excitation **minus modeled harmonics**, including every frame of unvoiced detail, transients and unexplained periodicity |

The declared reconstruction target is the **minimal branch**, including detected
nonspeech. It preserves that processed mono waveform, rather than native-rate
multichannel input. Background sound is also retained. The packet is an
unquantized research representation with dense residual frames, not a compact
token stream: the parameter update rate does not imply a low waveform bitrate.

Synthesis uses a separate Burg envelope fit on every minimal-branch frame, with
no formant/phonation gate. This envelope is distinct from the ceiling-relative,
pre-emphasized F1–F3 measurement model. LSF conversion uses
[SPTK/pysptk LPC to LSP](https://pysptk.readthedocs.io/en/latest/generated/pysptk.sptk.lpc2lsp.html)
and [LSP to LPC](https://pysptk.readthedocs.io/en/latest/generated/pysptk.sptk.lsp2lpc.html),
with the embedded SPTK gain fixed at one; AMP owns excitation gain.
Bandwidth expansion starts at 0.98 and increases as needed to bound coefficient
cancellation (L1 ≤20) and sampled synthesis gain (≤100). Conversion failures
use an annotated identity envelope. Strong regularization moves more information
into the residual; it does not establish a physiological vocal-tract estimate.

Harmonic fitting uses Hann-weighted complex projections at up to 64 harmonics,
followed by a nonnegative scalar least-squares fit in output space bounded to
[0,1]. Frame-local phases are retained. Aperiodicity is a modeled excitation-band
remainder fraction, and HNR is modeled harmonic/remainder energy. Components
need not be orthogonal in excitation space; these are not calibrated physical
noise or glottal measures. Six bands span 0–500–1000–2000–4000–6000–8000 Hz.
Metadata declares padding, zero-state filters, windows, gain and phase origins.

- **Full packet:** harmonics plus complementary residual through the LSF filter
  and excitation gain. This uses only the packet, independently of the analyzer.
- **Harmonics only / Residual only:** complementary components through the
  same filter and gain. Their waveforms sum to the full reconstruction.
- **Parametric:** harmonic model plus deterministic band-shaped noise, with
  frame output RMS restored. It can decode without dense residual samples,
  although the downloadable full packet retains them for experiments.
- **Flat tract:** full excitation with its tract filter bypassed.
- **Constant gain:** full frame synthesis scaled to median nonzero output RMS.

**Full packet** reconstructs from the six exported streams against the minimal
target. The previous enhanced-branch filter roundtrip is no longer generated.

Decode an exported packet without the original recording:

```bash
voice-lab decode \
  voice-lab-runs/my-recording/representation.npz \
  --out voice-lab-runs/decoded.wav --mode full
```

`--mode` also accepts `harmonic_only`, `residual_only`, `parametric`, `flat_tract`
and `constant_gain`. Python API: `load_packet(path)` then `decode_packet(packet,
mode)`. Both `analyze(...)` and `analyze_recommended(...)` run this same pipeline.
`analyze_recommended(..., return_packet=True)` returns `(data, waves,
packet)`; its default two-value return is retained. NPZ uses numeric arrays plus
JSON metadata and loads with `allow_pickle=False`.

## File analysis and reference evaluation

```bash
voice-lab analyze /path/to/audio.wav \
  --offset 10 --seconds 30 \
  --out voice-lab-runs/my-recording
```

`--denoise`, `--channel` and `--formant-ceiling` select conditions. Analysis and
all audio/packet exports are written directly under the requested output.
The retired `--method` selector is removed. HTTP requests default to the unified
pipeline; requests explicitly selecting a retired method return an error.
`--reference /path/to/reference.json` attaches an independent
reference using **excerpt-relative** timing:

```json
{
  "kind": "human-verified segmentation and reference pitch",
  "intervals": [{"start": 0.5, "end": 1.5, "label": "voiced"}],
  "tracks": {
    "time": [0.5, 0.6, 0.7],
    "f0": [120, 121, 120],
    "formants": [[600, 1500, 2500], [610, 1490, 2500], [600, 1500, 2510]]
  }
}
```

Use zero for unvoiced F0 or absent formants in reference tracks. For complete
F0/voicing scoring, supply a reference covering the entire excerpt, including
unvoiced frames. Synthetic reference intervals carry known modulated source F0
and filter resonances; they appear automatically in the web analysis.
Those resonances are controlled filter parameters, not physiological ground truth.

Evaluation reports duration-weighted precision/recall/F1, a confusion matrix in
seconds, activity duration error, one-to-one endpoint F1 at 10/20/30/50/100-ms
tolerances and matched-boundary timing error. Declared 50-ms activity padding
can affect strict boundary scores. Pitch reports coverage, voicing error,
relative/cents error, gross error at a declared 20% relative threshold, and
octave-error rate among accepted voiced estimates. Formants report per-track
coverage and MAE/RMSE. Method disagreement alone is never called accuracy.

Corpus-level validation needs independently acquired annotations: TIMIT for
segmentation/landmarks, PTDB-TUG for synchronized laryngograph pitch, and inspected
vowel/formant references such as Hillenbrand. NTIMIT and noisy/free-field TIMIT
variants support paired channel/noise evaluation. Freeze adaptation rules before
held-out evaluation, stratify by speaker/F0/SNR/channel, and report abstention.
No licensed corpora or reference annotations are downloaded automatically.

## Verification and implementation fidelity

```bash
python -m pytest tests -q
node --check src/voice_lab/web/app.js
```

Focused checks cover known timing/F0, unvoiced speech preservation,
Burg recovery, octave-disagreement abstention, silence and white/colored noise,
short/native-bandwidth inputs, minimal-branch preservation under enhancement,
clean sustained-tone perturbation, qualified source output, packet reconstruction,
and HTTP exports including phase-inverted stereo selection and rejection of
retired method requests.
Representation checks cover saved-packet reconstruction, standalone CLI decoding,
component additivity, transient/noise preservation despite nonspeech annotations,
LSF interpolation stability, decoder ablations without residual samples, invalid
packet rejection, and short silent inputs with even/odd LPC orders.

The multi-estimator stack uses the actual [SPTK SWIPE prime implementation through
pysptk](https://pysptk.readthedocs.io/en/latest/generated/pysptk.sptk.swipe.html).
It is the published SWIPE prime variant, not a hand-labelled harmonic scorer.
The [Praat Burg documentation](https://praat.org/manual/Sound__To_Formant__burg____.html)
informs ceiling and pre-emphasis choices; our Hamming-window Burg pipeline and
scalar Kalman/RTS tracker are engineering implementations, not Praat/KARMA
reproductions. IAIF follows the two-pass source/tract algorithm described in the
report and [COVAREP's implementation](https://github.com/covarep/covarep/blob/master/glottalsource/iaif.m);
this independently implemented Burg version is not a COVAREP port or validated
physiological estimator. Decisions are logged in [implementation.md](implementation.md).

The implementation does not identify phonemes or speakers, recover overlapping
monaural sources, or measure absolute SPL without microphone calibration.
Landmarks, nuclei, acoustic pulses and glottal descriptors carry the report's
observability limits. Real-corpus accuracy and physiological source validity
remain unmeasured; synthetic/finite-waveform checks establish controlled behavior
and functioning processing, not those broader claims.
