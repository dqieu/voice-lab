# Voice Lab operating notes

Keep this application independent of any training repository. Its runtime must
remain local, deterministic, and free of trained model downloads. Read README.md
and docs/implementation.md before changing the acoustic pipeline.

- Never commit audio recordings, local registries, exports, caches, credentials,
  or machine-specific dataset defaults.
- Preserve branch-specific measurement conditions, confidence, missing values,
  and packet provenance. Do not present acoustic source proxies as validated
  physiological measures or speech activity as speaker separation.
- Log deviations from the supplied PDF baseline append-only in
  docs/implementation.md, including the decision owner and rationale.
- Run relevant focused tests. Package changes must retain the web assets in a
  built wheel; frozen Mac builds must pass their exported-packet self-test.
- Do not rewrite the reference PDF or redistribute dependency binaries without
  retaining their notices and appropriate corresponding-source arrangements.
