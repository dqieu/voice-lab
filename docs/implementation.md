# Implementation decisions

This append-only log records the choices behind the current app. The baseline is
the [supplied report](references/deterministic-voice-features.pdf). The standalone
package was extracted from a locally developed toolkit on 2026-09-18. Historical
choices below are consolidated from that toolkit's implementation notes; they
are not new claims of reference-algorithm equivalence.

## 2026-09-17 - Measurement frontend

- **Owner: Codex. Adaptation:** use SPTK's actual SWIPE prime through `pysptk`
  instead of porting original SWIPE. It supplies complementary spectral F0
  evidence and its variant is identified in provenance.
- **Owner: Codex. Deviation:** filtered ACF uses an engineering Butterworth
  lowpass, not Praat's exact Gaussian filter. Candidate/path tracking and
  multi-estimator consensus implement the report's continuity recommendation.
- **Owner: Codex. Deviation:** ceiling-relative, pre-emphasized Burg analysis
  uses Hamming windows, measured-spectrum pole QA, ordered association, scalar
  Kalman filters, and offline RTS smoothing. This is not exact Praat/KARMA;
  unobserved formants stay missing and silence gaps are not interpolated.
- **Owner: Codex. Baseline-consistent:** pulse timing/amplitude, relative
  intensity, and two-pass IAIF use only the minimal branch. Noise enhancement
  affects estimators. Stable-periodicity/consensus/energy-margin gates and
  channel/codec exclusions withhold unreliable source/pulse outputs. Energy
  margin, closure timing, open phase, and NAQ are declared acoustic proxies.
- **Owner: Codex. Adaptation:** variable-window Librosa CQT uses a native
  256-sample hop; magnitudes align to the common frame timeline. Short-input
  padding and units are recorded instead of presenting an STFT as a CQT.
- **Owner: Codex. Scope choice:** select the most energetic channel before
  processing. This avoids destructive phase-inverted channel averaging and
  is not calibrated beamforming or speaker separation.
- **Owner: Codex. Baseline-consistent:** synthetic source/filter references
  and optional independent CLI JSON labels bound evaluation. Imports without
  reference annotations have no automatically asserted accuracy.

## 2026-09-18 - Six-stream packet and decoder

- **Owner: user (implementation authorized), Codex (target choice).
  Extension:** reconstruct the minimal 16-kHz selected-channel waveform and
  preserve every frame, regardless of activity decisions. This retains breaths,
  transients, frication, uncertainty, and background audio together. It does not
  claim target-speaker extraction.
- **Owner: Codex. Explicit split:** HARM models normalized excitation harmonics;
  RESID is that excitation minus HARM. The independent decoder adds them once.
  Parameter-only decoding substitutes deterministic band-shaped noise for the
  dense remainder; component waveforms support accounting and listening ablations.
- **Owner: Codex. Deviation/extension:** fit a dedicated full-band Burg synthesis
  envelope on every minimal frame, rather than reuse gated formant LPC.
  SPTK LPC/LSF conversion and decoded coefficients define inverse filtering.
  AMP owns gain. Bandwidth expansion starts at 0.98 and becomes stronger until
  coefficient L1 is at most 20 and sampled synthesis gain at most 100; failures
  use annotated identity filters. Stabilization moves unexplained detail into
  RESID and does not establish anatomical tract validity.
- **Owner: Codex. Engineering choice:** Hann-weighted complex projections retain
  phases for up to 64 harmonics, with a bounded nonnegative output-space scalar
  fit. Six excitation bands describe modeled remainder fractions and energy
  ratios, not independent physiological HNR.
- **Owner: Codex. Artifact choice:** versioned NPZ stores float64 parameters,
  dense remainder frames, audit coefficients, units, missing values, and framing
  metadata. Frame-local zero-state filters and Hann overlap/add specify decoding.
  Dense waveform residuals mean this is not a compact speech tokenizer/codec.

## 2026-09-18 - Toolkit and UI consolidation

- **Owner: user. Scope simplification:** retire the previous segmenter and
  redundant enhanced LPC roundtrip. Shared DSP, Burg/LSF synthesis, tracked
  formants, and IAIF remain in the unified pipeline. This does not replace the
  reconstruction contract with an extra method selector.
- **Owner: Codex. Backend repair:** accepted extremely narrow F0 intervals could
  hang SPTK SWIPE. Give its backend at least one octave of support below the
  requested ceiling, then reject estimates outside user bounds; record both
  ranges. Default-range behavior stays the same.
- **Owner: user (cleanup authorized), Codex (presentation).
  Presentation-only:** consolidate output selection and metrics, remove the
  duplicate seek control/profile, combine noise options, collapse details,
  and hide unqualified IAIF playback with one reason. Formants/LSFs and
  intensity/excitation gain remain distinct measurements.

## 2026-09-18 - Mac packaging and local imports

- **Owner: user. New application requirement:** native WebKit window and local
  folder or CSV datasets, with optional folder recursion and saved source refs.
- **Owner: Codex. Packaging-only:** PyInstaller freezes an isolated lightweight
  DSP/GUI environment and FFmpeg fallback. Librosa source stays outside the
  archive so Numba disk-cache locators work. Normal launch binds an available
  loopback port and stops the server on window close. Validated target is
  Apple Silicon/macOS 14+; no Intel, old-macOS, or clean-machine Gatekeeper claim.
- **Owner: Codex. Sampling choice:** imports index every supported discovered
  file rather than the original built-in health sampler's 800-file reservoir.
  Sorted folder traversal does not follow directory symlinks. CSV supports BOM,
  extra columns, relative/absolute/tilde paths, deduplication and skip counts.
- **Owner: Codex. Persistence-only:** atomically save references, not copied
  audio. Reindex at launch and retain unavailable sources as disabled options.
  Source/pulse qualifications and branch processing are unchanged.

## 2026-09-18 - Public standalone repository

- **Owner: user. Publication requirement:** extract the app into a personal
  public repository, include complete user/developer/build documentation and
  the original analysis PDF. No private audio, source registries, or generated
  research artifacts are included.
- **Owner: Codex. Packaging/scope change:** use the `voice_lab` namespace and
  its own PyPI-only project/locks. Remove implicit sibling health-corpus and
  staged LibriSpeech-shard discovery; start with generated controls and explicit
  imports. CLI folders use the same all-file index as the UI. This does not
  alter acoustic algorithms or the existing packet schema.
- **Owner: Codex. Publication choice:** publish source and recipes without
  bundling generated installers or third-party executables in Git. The supplied
  PDF remains byte-identical and separately attributed; no source redistribution
  license is newly designated on the user's behalf.
- **Owner: Codex. Packaging repair:** restrict notice collection to document
  filenames and regenerate its build directory. A PyObjC test binary named
  `copying` was previously mistaken for a license file, introducing a debug
  object without a macOS version command into the bundle. Acoustic processing
  is unchanged; verify the repaired frozen bundle's signature and self-test.
