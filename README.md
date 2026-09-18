# Voice Lab

A local web app for inspecting voice activity, pitch, formants, source estimates,
and six-stream source/filter reconstruction. It uses deterministic signal
processing, with no trained models or remote inference. Audio stays on your
computer.

Voice Lab runs in a browser or a native macOS WebKit window. Add your own audio
folder or CSV, listen to the processing branches and reconstruction variants,
inspect measurements, and export WAV, JSON, and a standalone reconstruction packet.

## Install and run

### From source

Use Python 3.11 or newer (below 3.15). The validated development environment is
Python 3.11 on Apple Silicon with macOS 14+. Installing dependencies initially
requires internet access; normal analysis and playback run locally.

```bash
git clone https://github.com/dqieu/voice-lab.git
cd voice-lab
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
voice-lab serve
```

Open **http://127.0.0.1:8765/**. The app starts with a generated synthetic example.
If the port is occupied, use `voice-lab serve --port 8766`.

With [uv](https://docs.astral.sh/uv/), the equivalent locked install is:

```bash
uv sync --locked
uv run --locked voice-lab serve
```

The first analysis can take longer while Numba compiles its numerical routines.
No recordings or licensed datasets are downloaded automatically. No GPU, Torch,
CUDA, or training environment is needed. If `pysptk` must be built from source,
install a C/C++ compiler; on macOS use `xcode-select --install`.

### Native Mac window

From the activated environment:

```bash
python -m pip install -e '.[mac]'
voice-lab-mac
```

Or use `uv sync --locked --extra mac` and `uv run --locked voice-lab-mac`.
The native window adds a Mac folder/CSV picker and chooses an available local
port. Closing its window stops its server.

### Standalone Mac installer

The [Mac packaging guide](packaging/macos/README.md) builds a drag-to-Applications
DMG containing Python, the analysis libraries, and FFmpeg. Run:

```bash
VOICE_LAB_PYTHON=python3.11 packaging/macos/build.sh
```

Open `packaging/macos/dist/Voice-Lab-macOS-arm64.dmg`, drag **Voice Lab** to
**Applications**, then launch it. Build tools and dependencies are fetched at
build time; the installed app runs offline. Apple Silicon/macOS 14+ is the tested
target. Intel builds require separate native build and validation.

This repository publishes source and build recipes. A build without a Developer
ID identity is ad-hoc signed and not notarized; it is a development build.
There is no notarized download release in this repository yet.

## Add a dataset

Choose **+ Add dataset** below the dataset menu:

- **Folder:** point to an audio folder; enable or disable scanning subfolders.
- **CSV:** point to a UTF-8 CSV with exactly one column named `path`. Other
  columns are allowed. Paths can be absolute, start with `~`, or be relative to
  the CSV's directory.

```csv
path,label
recordings/example.wav,speech
recordings/second.flac,vowel
```

The native app provides **Browse**. In a normal browser, paste the local folder
or CSV path. Imports store source references rather than copying audio. Keep
the folder or CSV accessible. Saved sources are reindexed at launch; unavailable
sources remain visible as disabled entries.

Supported extensions: WAV, FLAC, OGG, MP3, M4A, AIFF/AIF, and OPUS. Codec support
depends on libsndfile and optional FFmpeg. Missing, empty, unsupported, and
duplicate CSV entries are skipped and counted. See [dataset details](docs/datasets.md).

## Listen and compare

Select a dataset, set the excerpt length, and choose **Sample & analyze**. The output
table compares Original, Speech only, Minimal, Filtered/enhanced, a qualified
IAIF source estimate, and six reconstruction variants. Select a row to switch
audio at the same playback position. Click the waveform, intervals, or landmarks
to seek or play a region.

**Minimal** removes DC and applies a gentle highpass. It feeds waveform-sensitive
measurements and the reconstruction target. **Filtered/enhanced** adds
measurement-appropriate bandwidth limits and optional conservative Wiener or
spectral subtraction. It feeds pitch/formant estimators. Advanced settings and
plots are collapsed initially.

Expand **Find sample with highest RMS error to Minimal** below the output table
to test 10, 25, 50, 100, or all indexed recordings. Choose the reconstruction
output to rank. The search shows progress, loads the highest-error tested excerpt,
and can stop with best-so-far. Only the winner is exported. It uses one excerpt
per recording and current settings, so its maximum is bounded to tested excerpts.
Full-packet errors are normally numerical roundoff. [Search details](docs/voice_analysis.md#find-the-highest-reconstruction-error).

The six streams are **F0** (pitch), **VOICE** (voicing/periodicity), **TRACT**
(LSF filter envelope), **AMP** (gain), **HARM** (modeled harmonics), and **RESID**
(the complementary excitation remainder). Full-packet decoding retains all
minimal-branch audio, including background sound. The packet contains dense
waveform residuals: it is a research representation, not a compressed tokenizer.

The toolkit does not isolate a target speaker, separate overlapping voices,
identify phonemes, diagnose health conditions, or establish physiological
glottal-flow measurements. Speech-only output gates detected activity; it can
retain interference during those intervals. IAIF and perturbation outputs appear
only for qualified intervals and remain experimental acoustic estimates.

## Command line and Python

```bash
voice-lab analyze /path/to/audio.wav --seconds 30 --out voice-lab-runs/example
voice-lab decode voice-lab-runs/example/representation.npz \
  --out voice-lab-runs/decoded.wav --mode full
voice-lab serve --dataset recordings=/path/to/audio --port 8765
```

Analyze writes JSON, WAV variants, a segment archive, and `representation.npz`.
The CLI analyzer uses SoundFile-supported inputs; the web sampler additionally
offers FFmpeg fallback when available. `decode` needs only the packet.

```python
import soundfile as sf
from voice_lab import analyze, save_packet, load_packet, decode_packet

audio, sr = sf.read("recording.wav")
analysis, waves, packet = analyze(audio, sr, return_packet=True)
save_packet("representation.npz", packet)
reconstructed = decode_packet(load_packet("representation.npz"))
```

Use `voice-lab --help` and each subcommand's `--help` for all arguments. The
[analysis guide](docs/voice_analysis.md) documents decoder modes, measurement
units, independent reference evaluation, and confidence/abstention rules.

## Files and privacy

Browser exports and saved imports go to `./voice-lab-runs/` by default; `--out`
and `--registry` override this. The native app stores its imports, recordings,
cache, and log in `~/Library/Application Support/Voice Lab/`; `--data-dir`
overrides that directory. Exports may contain audio, recording names, and paths.

The HTTP server binds to loopback and checks Host/Origin headers. There are no
uploads, analytics, remote estimators, or model downloads. This repository
contains source, tests using generated signals, documentation, and the supplied
research PDF; it contains no local datasets or recordings.

## Documentation and development

- [Folder and CSV imports](docs/datasets.md)
- [Measurements, playback, packet format, and evaluation](docs/voice_analysis.md)
- [PDF recommendations and implementation coverage](docs/analysis.md)
- [Implementation decisions and deviations](docs/implementation.md)
- [Original analysis PDF](docs/references/deterministic-voice-features.pdf)
  and [reference provenance](docs/references/README.md)
- [Mac build and signing](packaging/macos/README.md)
- [Developer setup and tests](CONTRIBUTING.md)
- [Third-party notices](packaging/macos/THIRD_PARTY.md)
- [Changelog](CHANGELOG.md)

The application source has no designated redistribution license yet. Public
availability does not change the separate licenses of dependencies or referenced
material. The research PDF is retained verbatim and separately attributed.
