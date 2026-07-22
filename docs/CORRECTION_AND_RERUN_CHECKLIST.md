# Correction and rerun checklist

This checklist is the handoff for the next CPU/RTX 6000 Ada run. Complete the
P0 items before treating any benchmark result as submission-grade.

## Why a rerun is required

The current manual decode loop calls `LlamaForCausalLM` without limiting the
number of returned logits. During prefill this materializes a
`[batch, context, vocabulary]` tensor even though greedy generation needs only
the final position. With vocabulary size 49,152 and BF16, both failed GPU
points allocate exactly 24 GiB of logits:

```text
128 * 2048 * 49152 * 2 bytes = 24 GiB
64 * 4096 * 49152 * 2 bytes = 24 GiB
```

This unnecessarily increases TTFT and peak memory. It also makes the recorded
GPU OOM boundary a full-logits boundary rather than a representative serving
boundary. Cache-off decoding repeatedly materializes full-history logits and
therefore overstates the cost of disabling the KV cache.

## P0 — correct the benchmark before rerunning

- [ ] Pass `logits_to_keep=1` on every benchmark model forward in
  `fixed_work_decode`, including prefill, cache-on decode, and cache-off decode.
- [ ] If a future model class does not support `logits_to_keep`, implement a
  documented last-hidden-state plus `lm_head` fallback rather than silently
  returning full-sequence logits.
- [ ] Add a regression test proving that every benchmark forward requests one
  logit position and that cache-off still supplies the complete token history.
- [ ] Add a smoke assertion or recorded diagnostic showing logits shape
  `[batch, 1, vocabulary]` for both cache modes.
- [ ] Run `pytest` and `ruff` in the exact target environment and save their
  complete outputs under `results/validation/`.
- [ ] Commit the correction before collecting headline data. Every final
  manifest must report `git.dirty: false` and the corrected commit hash.

## Status of existing results

Do not delete the existing records. Retain them as diagnostic provenance, but
do not mix affected rows with corrected aggregates.

| Result | Current use | Required action |
|---|---|---|
| CPU/GPU TTFT | Diagnostic only | Rerun after last-logit correction |
| CPU/GPU peak memory | Diagnostic only | Rerun after last-logit correction |
| GPU OOM boundaries | Invalid as serving boundaries | Rerun in fresh processes |
| Cache-off TPOT/TPS | Directionally useful but inflated | Rerun all cache-off points |
| Cache-on TPOT/TPS | Largely usable for comparison | Rerun for one consistent dataset |
| CPU eager versus SDPA | Directionally useful | Rerun representative points |
| CPU W8A8 performance | Decode trend is useful | Rerun performance and memory |
| WikiText-2 FP32/W8A8 perplexity | Unaffected by this bug | Retain after provenance checks |
| FP32 HellaSwag/ARC-Easy | Unaffected by this bug | Retain after provenance checks |

- [ ] Move superseded performance JSONL/CSV/figures to a clearly named
  `results/intermediate/full_logits_runs/` directory or label them with an
  explicit `full_logits_prefill` limitation.
- [ ] Generate corrected raw files under new names; never append corrected
  observations to the old JSONL files.
- [ ] Update `results/README.md` only after regenerated CSV values are checked
  against the raw observations.

## P0 — environment and model provenance

- [ ] Start from a clean environment and record the exact install command.
- [ ] Make installed versions agree with `pyproject.toml`, or update the pinned
  environment specification before the run. The previous environment used
  `accelerate 0.27.2`, below the repository requirement of `>=0.34`.
- [ ] Save `python -m pip freeze` and the output of `python -m torch.utils.collect_env`.
- [ ] Use the pinned Hugging Face revision, or hash every local model artifact
  (`config.json`, tokenizer files, generation config, and all weight shards).
- [ ] Record the model-artifact hashes in each run manifest. A local path with
  `revision: null` is not sufficient provenance.
- [ ] Run hardware inspection outside any sandbox that hides `/dev/nvidia*`.
- [ ] Record GPU name, UUID, driver, CUDA runtime, total/free memory, ECC,
  power limit, clocks, temperature, active processes, and CPU affinity.
- [ ] Bind CPU experiments to the same socket/NUMA node and thread counts used
  by the validated EPYC configuration.
- [ ] Confirm no competing GPU processes and record idle device memory before
  each experiment family.

## P0 — corrected baseline reruns

- [ ] CPU baseline: contexts `128, 512, 2048, 4096`; batches
  `1, 2, 4, 8, 16`; 64 output tokens; 2 warmups; 10 broad-grid repetitions.
- [ ] GPU baseline: the same core grid plus batches `32, 64, 128` until the
  corrected knee or a genuine memory boundary is reached.
- [ ] Run memory-boundary candidates in fresh processes so allocator residue
  or fragmentation from a previous point cannot determine the boundary.
- [ ] Compare observed GPU peak memory with analytical weight and KV-cache
  sizes. Investigate any unexplained multi-GiB difference.
- [ ] Verify every successful row has exactly 64 output tokens, 63 decode
  intervals, a stable prompt hash, and the resolved SDPA backend.
- [ ] Recompute p50/p95/p99 CSV files and all baseline figures only from
  corrected raw rows.

Suggested smoke run before the full grid:

```bash
python scripts/run_baseline_grid.py \
  --config configs/gpu_rtx6000_ada.yaml \
  --set 'workload.context_lengths=[128,2048]' \
  --set 'workload.batch_sizes=[1,64]' \
  --set workload.warmups=1 \
  --set workload.repetitions=2 \
  --set output.path=results/validation/gpu_last_logits_smoke.jsonl
```

## P1 — intervention studies still required

- [ ] CPU attention: rerun eager versus SDPA at all five representative points.
- [ ] GPU attention: compare eager and SDPA; add FlashAttention-2 only after
  verifying that the kernel executes without fallback.
- [ ] CPU cache: rerun the five repeated cache-off points; keep the 2048/8
  point single-observation-only unless enough repetitions are practical.
- [ ] GPU cache: run cache on/off at contexts `128, 512, 2048`, batches `1, 8`.
- [ ] CPU W8A8: rerun the four performance points with `lm_head` excluded and
  record the exact quantized module list.
- [ ] GPU W8A8: either produce a verified TorchAO kernel path or retain a
  structured unsupported/failure result with the exact error and versions.
- [ ] Measure W8A8 resident memory in a fresh process; do not infer model-size
  reduction from in-process peak RSS alone.

## P1 — quality evaluation gaps

- [ ] Retain the paired FP32/W8A8 WikiText-2 run after verifying the local model
  hashes and source-file SHA-256.
- [ ] Run at least one full multiple-choice task on the CPU W8A8 artifact; the
  current HellaSwag and ARC-Easy records evaluate native precision only.
- [ ] Record `lm-eval`, `datasets`, PyTorch, Transformers, and quantization
  backend versions in every quality result.
- [ ] Preserve task YAML, dataset split, sample count, source hashes, batch
  size, random seed, accuracy, normalized accuracy, and standard errors.
- [ ] Do not describe W8A8 as quality-preserving: current perplexity changed
  from 11.27 to 44.16 unless a corrected recipe produces contrary evidence.

## P1 — headline statistics and interpretation

- [ ] Select a small set of headline configurations and collect 100–200
  repetitions before emphasizing p99. Ten repetitions are acceptable for the
  broad grid but do not support a stable p99 claim.
- [ ] Randomize or rotate headline configuration order and monitor thermal and
  power drift.
- [ ] Apply the frozen batching-knee definition mechanically and state when no
  knee occurs within the feasible range.
- [ ] Analyze CPU and GPU independently; use normalized trends rather than an
  absolute cross-platform fairness claim.
- [ ] Keep negative results, OOM points, fallbacks, and single-observation
  boundaries visibly labeled.
- [ ] Do not attribute a speedup to SDPA, INT8, or caching until the executed
  backend and relevant kernels have been verified.

## Final acceptance gate

- [ ] Corrected code is committed and the worktree is clean.
- [ ] Full `pytest` and `ruff` logs are committed.
- [ ] CPU and GPU hardware reports are committed.
- [ ] Model and dataset hashes are committed.
- [ ] Corrected raw JSONL files and manifests are committed.
- [ ] Processed CSV files reproduce from raw rows without manual edits.
- [ ] Figures reproduce from the committed plotting command.
- [ ] `README.md` and `results/README.md` contain only conclusions supported by
  corrected or explicitly unaffected results.
- [ ] Report limitations distinguish model-only synchronous benchmarking from
  production request scheduling and network-level serving.

