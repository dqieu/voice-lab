# Voice Lab for macOS

After building, open `dist/Voice-Lab-macOS-arm64.dmg`, drag **Voice Lab.app** to **Applications**,
then open it. Python, the DSP libraries and the FFmpeg decoder are bundled.
The app uses the Mac's WebKit engine and a loopback-only server on an available
port. Closing the window stops its server. Internet is not required to run it.

This build targets **Apple Silicon, macOS 14+**. Intel builds must be produced
and tested with an x86_64 Python environment on an Intel Mac. No Intel or older
macOS compatibility claim follows from testing this arm64 build on this Mac.

## Add datasets

Choose **+ Add dataset** below the dataset menu. Select a folder or CSV, use
**Browse**, and optionally name it. Folder scans can include subfolders or
only files directly in that folder. Directory symlinks are not traversed.

CSV must be UTF-8 (a BOM is allowed), comma-separated, with exactly one `path`
column. Additional columns are accepted. Relative paths resolve against the
CSV's directory; absolute paths and `~` paths are supported. For example:

```csv
path,label
recordings/example.wav,speech
/Users/you/Audio/second.flac,speech
```

Imports include all discovered supported files. Empty/missing/unsupported/duplicate CSV entries are skipped
and counted in the confirmation. No readable paths, malformed CSVs and missing
`path` headers give an error instead of creating an empty dataset. Audio decoding
is deferred until sampling. Keep source folders and CSVs in place: the app
stores references, not copies. It reindexes saved sources on launch and displays
unavailable sources as disabled menu entries if a drive or folder is missing.

Supported extensions: WAV, FLAC, OGG, MP3, M4A, AIFF/AIF and OPUS. Some codecs
use the bundled FFmpeg fallback; experimental pulse/source measurements abstain
on that fallback, preserving the existing analysis convention.

Dataset registrations, exports, the Numba cache and `app.log` live under
`~/Library/Application Support/Voice Lab/`. The development browser version
also supports pasted local paths, with registrations in its output directory.

## Rebuild

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
VOICE_LAB_PYTHON=python3.11 packaging/macos/build.sh
```

The separate build environment uses `requirements.lock.txt`; it never installs
any CUDA or training dependencies. `requirements.txt` records the direct
dependency choices. PyInstaller collects Librosa as source files to make its
Numba disk-cache locators work in the frozen app. The build verifies code signing
and reconstruction from an exported/reloaded packet before creating a DMG and
SHA-256 checksum.

The executable accepts `--self-test`, `--headless`, `--data-dir` and `--port`
for isolated release checks. Normal launches need no command arguments.

## Signing

The current local development release is **ad-hoc signed, not notarized**.
For public distribution, build with `VOICE_LAB_SIGNING_IDENTITY` set to a
Developer ID Application identity, submit the DMG to Apple's notary service,
and staple the accepted ticket. Developer ID and notarization credentials are
external release requirements; the build does not disable Gatekeeper or remove
quarantine attributes. See [Apple's notarization guide](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution).
