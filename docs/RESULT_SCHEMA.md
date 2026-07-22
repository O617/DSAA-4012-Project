# Raw result schema

Current successful and failure rows use schema version 2.

Every benchmark appends one JSON object per repetition to a JSONL file. Failed,
unsupported, and out-of-memory configurations are retained as rows rather than
silently omitted.

Each run also updates `<result>.manifest.json` with the latest complete merged
configuration, software versions, Git revision, and dirty-worktree flag. Its
`invocations` list preserves the same metadata for earlier resume invocations.

The identity fields are experiment, hardware ID, model ID and revision, device,
dtype, intra-op and inter-op thread counts, process CPU affinity, attention
implementation, cache and compile modes, quantization mode, context length,
batch size, output length, seed, and repetition. These fields
prevent resume mode from silently reusing observations from a materially
different configuration. Successful rows include:

- `ttft_seconds`: prompt-forward start through availability of the first token;
- `tpot_seconds`: decode time after the first token divided by `N - 1`;
- `tps`: aggregate batch decode tokens divided by decode wall time;
- `end_to_end_seconds` and `end_to_end_tps`;
- synchronized `token_timestamps` and `token_intervals`;
- peak RSS and CUDA allocated/reserved bytes;
- prompt hash, requested/actual output length, model metadata, and quantization details.

Times exclude tokenization. The input IDs and attention mask are prepared before
the measured region. GPU timestamps synchronize the device.
