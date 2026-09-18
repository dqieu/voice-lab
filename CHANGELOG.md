# Changelog

## 0.1.0 - 2026-09-18

- Extracted the local voice toolkit into the standalone `voice_lab` package.
- Added a browser player, native Mac WebKit window, folder/CSV dataset imports,
  saved source references, and an isolated Mac DMG build recipe.
- Included the unified deterministic two-branch voice-analysis frontend and
  F0/VOICE/TRACT/AMP/HARM/RESID packet with independent decoding and ablations.
- Included generated-signal tests, user/developer documentation, the supplied
  research PDF, and a recommendation-to-implementation map.
- Removed implicit parent-project health-corpus and LibriSpeech shard discovery
  from this standalone application; datasets are selected explicitly.
