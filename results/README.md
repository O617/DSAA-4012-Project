# Curated experiment results

The submission-grade performance records use schema version 3 and request only
the final logit position on every model forward. CPU experiments were bound to
logical CPUs `0-95` on one AMD EPYC 9654 socket with 96 intra-op threads and
one inter-op thread. GPU experiments used GPU0 of an eight-GPU RTX 6000 Ada
host.

## Corrected artifacts

- `raw/cpu_epyc9654_baseline_last_logits.jsonl`: 200 observations over the
  four contexts and five CPU batch sizes.
- `raw/gpu_rtx6000_ada_baseline_last_logits.jsonl`: 320 observations over the
  four contexts and eight GPU batch sizes. The former OOM points were run in
  fresh processes and now succeed.
- `raw/{cpu_epyc9654,gpu_rtx6000_ada}_operator_last_logits.jsonl`: eager versus
  SDPA at five representative points, ten repetitions per implementation.
- `raw/cpu_epyc9654_cache_off_last_logits.jsonl`: five cache-off points with
  ten repetitions, plus one explicitly single-observation 2048/8 boundary.
- `raw/gpu_rtx6000_ada_cache_last_logits.jsonl`: cache on/off at contexts 128,
  512, and 2048 and batches 1 and 8.
- `raw/cpu_epyc9654_quantization_w8a8_last_logits.jsonl`: four dynamic W8A8
  points with 224 Transformer Linear modules quantized and `lm_head` in FP32.
- `raw/gpu_rtx6000_ada_quantization_w8a8_torchao016_last_logits.jsonl`: a
  functional TorchAO 0.16 probe. This is a negative diagnostic, not a headline
  speed result.
- `raw/cpu_epyc9654_quality_arc_easy_w8a8_last_logits.jsonl`: full 2,376-item
  CPU W8A8 ARC-Easy evaluation with model, task-YAML, split-file, and software
  provenance.
- `raw/gpu_rtx6000_ada_headline_rotated_last_logits.jsonl`: 100 repetitions
  each for 128/1 and 4096/128, collected in ten alternating-order blocks.
- Matching benchmark manifests record the merged configuration, exact
  software, local model hashes, clean Git revision, and invocation history.
- `processed/*_last_logits.csv` and `figures/corrected_*` are regenerated
  directly from these corrected raw files.

Validation records include the isolated environment freeze, hardware report,
test and lint logs, headline telemetry, and profiler tables. The profiler
confirms both the CUDA FlashAttention kernel selected by SDPA and the Ampere
INT8 GEMM selected by TorchAO.

## Corrected findings

### Baseline surface

- CPU median TPS peaks within the tested grid at 146.96 (128/16), 119.91
  (512/16), 50.95 (2048/8), and 25.01 (4096/4). The frozen batching-knee rule
  identifies batch 16 at context 2048 and batch 8 at context 4096; no knee is
  observed through batch 16 at contexts 128 or 512.
- GPU median TPS reaches 2,973.59 (128/128), 2,947.87 (512/128), 2,700.83
  (2048/128), and 1,437.52 (4096/128). Only context 4096 reaches the frozen
  knee at batch 128; the other contexts show no knee through batch 128.
- The former 2048/128 and 4096/64 OOMs were artifacts of full-sequence logits.
  Both now use about 16.33 GiB peak allocated memory, and 4096/128 succeeds at
  31.95 GiB. The residual above weights plus KV cache scales with
  `batch * context`, consistent with transient prefill QKV/MLP activations.

### Attention and cache

- On CPU, eager/SDPA median TTFT ratios grow from 1.00 at 128/1 to 6.23 at
  4096/4; TPOT ratios grow from 1.11 to 4.09. On GPU, the corresponding ratios
  are 1.40 to 9.34 for TTFT, while TPOT remains near 1.30 across the five
  points. Profiling shows explicit eager `bmm`/softmax versus SDPA's
  `pytorch_flash::flash_fwd_kernel`.
- CPU cache-off median TPOT is 1.85x, 6.07x, 2.68x, 17.62x, and 7.45x cache-on
  at the five repeated points. The single 2048/8 boundary is 88.99x and took
  891.7 seconds; it is not a percentile estimate.
- GPU cache-off is 1-5% faster at the five smaller tested loads, within the
  regime where cache bookkeeping outweighs recomputation for this small model.
  At 2048/8 it becomes 4.22x slower. Caching is therefore not claimed as a
  universal small-load speedup.

### Quantization and quality

- CPU W8A8 produces 1.27x-2.49x the native median TPS across the four points,
  but peak RSS is 1.16x-1.66x higher. The tested eager dynamic-quantization
  path is a speed optimization, not a resident-memory reduction.
- GPU TorchAO executes real INT8 work: profiling records 225 `aten::_int_mm`
  calls and 224 Ampere INT8 GEMMs. It also performs 9,025 device-to-host scalar
  transfers in one 128/1 prefill, making the current path only about 0.21 TPS.
  No full-grid GPU W8A8 speed claim is made.
- On the frozen WikiText-2 slice, FP32/W8A8 perplexity is 11.27/44.16.
- Full ARC-Easy native/W8A8 accuracy is 56.44%/42.21%; normalized accuracy is
  49.12%/38.97%. W8A8 loses 14.23 and 10.14 percentage points respectively,
  so it is not quality-preserving for this model and recipe.

### Headline statistics

- The 100-repetition 128/1 GPU point has p50/p95/p99 TTFT of
  41.54/44.08/44.67 ms and TPOT of 41.24/44.13/44.43 ms.
- The 100-repetition 4096/128 point has p50/p95/p99 TTFT of
  6.745/6.899/6.923 s and TPOT of 88.96/89.18/89.48 ms.
- Block-boundary telemetry spans 29-70 degrees C without a progressive
  failure or memory leak. Broad-grid p99 values based on ten repetitions are
  retained in CSVs but are not emphasized as stable tail estimates.

## Legacy diagnostic records

The earlier files without `_last_logits` remain committed for provenance. All
performance JSONL/CSV/figures in that generation have the explicit limitation
`full_logits_prefill`: they materialize `[batch, context, vocabulary]` logits,
inflate TTFT and peak memory, create artificial GPU OOM boundaries, and inflate
cache-off cost. They must not be combined with schema-version-3 aggregates.
The preflight and incompatible TorchAO 0.17 probes are under
`results/intermediate/` and are not headline results.

The measurements are synchronous, model-only benchmarks. They exclude
tokenization, request scheduling, networking, and production serving queues;
CPU and GPU are analyzed as separate hardware lines rather than as a direct
fairness comparison.
