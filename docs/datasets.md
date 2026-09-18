# Folder and CSV datasets

Voice Lab reads recordings in place. It saves a source path, kind, display name,
and folder-recursion preference rather than copying the dataset.

## Import a folder

Choose **+ Add dataset**, select **Folder**, and enter its path. In the native Mac
window, **Browse** opens the folder picker. Give the dataset an optional name.

**Scan subfolders** indexes every supported file under the folder. Disable it
to index only files directly inside the chosen folder. Traversal is sorted and
directory symlinks are not followed, preventing cycles. File references are
deduplicated by resolved path. Custom imports are not capped at 800 recordings.

## Import a CSV

Select **CSV** and enter or browse to the manifest. It must be comma-separated
UTF-8 (UTF-8 BOM is supported), with exactly one case-sensitive `path` header.
Whitespace around header names is trimmed; additional columns are ignored.

```csv
path,label,speaker
audio/first.wav,speech,001
audio/second.flac,vowel,002
```

Relative paths resolve against the CSV file's parent directory, independently
of the process's working directory. Absolute and tilde paths also work. Quote
filenames containing commas according to ordinary CSV syntax.

Empty, nonexistent, unsupported, and repeated paths are skipped and counted.
Malformed CSVs, missing or duplicate `path` headers, and imports with zero usable
files are rejected. The recursion setting applies only to folder scans.

Supported extensions are `.wav`, `.flac`, `.ogg`, `.mp3`, `.m4a`, `.aiff`, `.aif`,
and `.opus`. Indexing checks paths and extensions; decoding is deferred until
sampling. A supported extension does not guarantee that a file decodes. The
sampler tries up to 12 selections if a recording is corrupt. SoundFile is used
first; FFmpeg is a fallback when installed (bundled in the frozen Mac app).
Known lossy codecs and fallback paths withhold experimental source/pulse measures.

## Persistence and sampling

Browser registrations live in `voice-lab-runs/datasets.json` by default. The
native app uses `~/Library/Application Support/Voice Lab/datasets.json`. Registry
updates use atomic replacement. Repeated imports of the same source/kind/scan
configuration are rejected. Saved sources are reindexed at launch, so changes
to a folder or CSV become visible next launch. Missing sources remain as
disabled menu options. Reconnect the drive or restore the path and restart.

The seed chooses the recording and excerpt deterministically for an unchanged
catalog and decoder. It is not a participant-balanced sampling scheme or a
benchmark split. FFmpeg fallback begins at the requested offset, or zero when
no offset is provided; it does not promise identical codec sampling behavior.

At browser-server launch, `--dataset NAME=PATH` adds a recursively indexed
folder for that session. These launch arguments are not saved as UI registrations.
Normal launch starts with the synthetic control and any saved UI imports.
Nothing is downloaded automatically.

Exports can contain private waveform samples, recording names, and source paths.
They stay local unless you choose to share them. Keep exported artifacts and
registries out of public Git commits.
