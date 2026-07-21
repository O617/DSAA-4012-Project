# Raw result schema

Every benchmark appends one JSON object per repetition to a JSONL file. Failed,
unsupported, and out-of-memory configurations are retained as rows rather than
silently omitted.

Each run also writes `<result>.manifest.json` containing the complete merged
configuration, software versions, Git revision, and dirty-worktree flag.

The identity fields are experiment, hardware ID, device, dtype, attention
implementation, cache mode, quantization mode, context length, batch size,
output length, and repetition. Successful rows include:

- `ttft_seconds`: prompt-forward start through availability of the first token;
- `tpot_seconds`: decode time after the first token divided by `N - 1`;
- `tps`: aggregate batch decode tokens divided by decode wall time;
- `end_to_end_seconds` and `end_to_end_tps`;
- synchronized `token_timestamps` and `token_intervals`;
- peak RSS and CUDA allocated/reserved bytes;
- prompt hash, requested/actual output length, model metadata, and quantization details.

Times exclude tokenization. The input IDs and attention mask are prepared before
the measured region. GPU timestamps synchronize the device.
