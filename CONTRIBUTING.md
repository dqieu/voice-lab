# Developing Voice Lab

Use a separate checkout and Python 3.11 environment. Only the `voice_lab`
package belongs here; the parent audio-model training project is independent.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,mac]'
python -m pytest tests -q
python -m build
```

Or install with `uv sync --locked --extra dev --extra mac`, then use `uv run` for
the commands. The `mac` extra is only needed for the native Mac window.
Tests use generated audio and temporary directories, including loopback
HTTP cases. An environment that blocks local socket binding cannot run those
cases; use a normal local terminal. Tests do not require private datasets or
network access after dependency installation.

If Node.js is installed, validate frontend syntax with:

```bash
node --check src/voice_lab/web/app.js
```

Start `voice-lab serve --port 8766 --out voice-lab-runs/development` to keep a
development session separate from another running player. Check generated
synthetic samples, audio switching at a nonzero seek position, a direct and
recursive folder import, CSV path resolution, and withheld IAIF outputs when
changing UI behavior. Do not commit recordings, saved registries, exports,
caches, installers, or credentials.

## Layout

| Path | Purpose |
|---|---|
| `src/voice_lab/dsp.py` | Shared framing, activity, candidate/path pitch, configuration |
| `src/voice_lab/recommended.py` | Two-branch measurement pipeline and qualifications |
| `src/voice_lab/representation.py` | Six-stream packet and standalone decoder |
| `src/voice_lab/evaluation.py` | Reference-bounded scoring |
| `src/voice_lab/datasets.py` | Folder/CSV indexing and atomic saved-source registry |
| `src/voice_lab/player.py` | CLI, loopback API, local sampler, exports |
| `src/voice_lab/desktop.py` | Native WebKit window, picker, release self-test |
| `src/voice_lab/web/` | HTML/CSS/JavaScript interface, included in wheels |
| `packaging/macos/` | Isolated frozen-app build and third-party notices |
| `docs/` | User/scientific guides, decision log, reference PDF |

## Dependency locks and releases

`uv.lock` covers source installation and optional development/Mac extras.
`packaging/macos/requirements.lock.txt` separately pins the validated Mac frozen
build environment, including PyInstaller and FFmpeg's provider. It is a Mac
build lock, not a portable GUI-runtime lock. Do not add CUDA or training packages.

If changing dependencies, refresh the appropriate lock in a clean environment
and run focused tests and the actual frozen executable's `--self-test`.
The [Mac build guide](packaging/macos/README.md) describes bundle/DMG verification
and Developer ID/notarization requirements. Binary releases need their bundled
third-party notices and applicable corresponding-source distribution arrangements.

Append changes that differ from the research baseline to
[docs/implementation.md](docs/implementation.md), stating the choice, rationale,
and decision owner. Check algorithm changes against independent references;
reconstruction fidelity alone does not validate pitch, formants, or physiology.
