# Curated experiment results

The committed CPU results were measured on one socket of a dual-socket AMD
EPYC 9654 host. The process was bound to logical CPUs `0-95`, corresponding to
96 physical cores on NUMA node 0. PyTorch used 96 intra-op threads and one
inter-op thread.

## Committed artifacts

- `hardware_epyc9654_node0.json`: hardware/software inventory under the final affinity.
- `raw/cpu_epyc9654_baseline.jsonl`: 200 observations covering four contexts,
  five batch sizes, and ten repetitions.
- `raw/cpu_epyc9654_operator_eager.jsonl`: 50 eager-attention observations at
  five representative workload points.
- `raw/cpu_epyc9654_cache_off.jsonl`: 50 repeated observations at five points
  plus one complete 2048/8 boundary observation. The boundary row is evidence
  of extreme slowdown, not a p95/p99 estimate.
- `raw/cpu_epyc9654_quantization_w8a8.jsonl`: 40 observations at four
  representative points using dynamic INT8 transformer Linear layers with the
  vocabulary head retained in FP32.
- `raw/cpu_epyc9654_quality_wikitext2.jsonl`: paired FP32 and W8A8 perplexity
  results over the same 12,846 scored WikiText-2 tokens, including source-file
  SHA-256 provenance.
- `raw/gpu_rtx6000_ada_baseline.jsonl`: 290 successful bfloat16/SDPA
  observations over 29 feasible context/batch points, two measured OOM
  boundaries, and one adaptive skip.
- `raw/gpu_rtx6000_ada_quality_hellaswag.jsonl`: full 10,042-example
  HellaSwag validation accuracy using the ModelScope-hosted source file.
- `raw/gpu_rtx6000_ada_quality_arc_easy.jsonl`: full 2,376-example ARC-Easy
  test accuracy using the ModelScope-hosted Parquet source.
- Matching `.manifest.json` files: full configurations and invocation history.
- `processed/*.csv`: p50/p95/p99 aggregates generated from the raw JSONL files.
- `figures/*.png`: TPS heatmaps and TPOT-throughput frontiers generated from
  the curated raw files.

Probe, thread-selection, and smoke-test outputs remain under the ignored
`results/intermediate/` directory and are not submission results.

## Current findings

- Cache-off median TPOT was 2.50x--23.07x worse at the five repeated points;
  the single 2048/8 boundary observation was 75.15x worse.
- Dynamic W8A8 improved median TPS by 1.47x--2.54x, but peak RSS was
  1.09x--1.56x higher rather than lower.
- WikiText-2 perplexity increased from 11.27 (FP32) to 44.16 (W8A8). This is a
  negative quality result and rules out this recipe as a quality-preserving
  serving optimization for the tested model/backend.
- A deterministic greedy probe asking why KV caching speeds decoding produced
  a readable FP32 answer, while W8A8 entered a repeated "question of a
  question" loop. This qualitative check is consistent with the perplexity
  regression and is not treated as a standalone accuracy metric.
- GPU median throughput peaked at 2,896 TPS for context 128/batch 128, 2,862
  TPS for context 512/batch 128, 1,444 TPS for context 2048/batch 64, and 716
  TPS for context 4096/batch 32.
- GPU OOM was measured at context 2048/batch 128 and context 4096/batch 64.
- Full HellaSwag accuracy was 42.83%; length-normalized accuracy was 56.88%,
  with standard errors of 0.49 percentage points for both.
- Full ARC-Easy accuracy was 56.44%; length-normalized accuracy was 49.12%,
  with standard errors of 1.02 and 1.03 percentage points, respectively.

The default execution sandbox hides `/dev/nvidia*`; sandbox-external checks
confirmed that the host driver and all eight RTX 6000 Ada GPUs are healthy.
HellaSwag and ARC-Easy were both recovered through ModelScope while retaining
the lm-evaluation-harness scoring protocols.
