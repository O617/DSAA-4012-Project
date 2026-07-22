# DSAA4012 Final Project

Serving-optimization study of `HuggingFaceTB/SmolLM2-360M-Instruct` on a fixed
CPU platform and NVIDIA RTX 6000 Ada Generation. The benchmark separates
prefill from decode and records TTFT, TPOT, aggregate decode TPS, latency
percentiles, peak memory, and quantization quality.

The experiment design and acceptance criteria are in
[PROJECT_PLAN.md](PROJECT_PLAN.md). The raw-result fields and timing boundaries
are documented in [docs/RESULT_SCHEMA.md](docs/RESULT_SCHEMA.md).

The last-logit correction and CPU/RTX 6000 Ada rerun are complete. Earlier
performance records remain as explicitly labeled `full_logits_prefill`
diagnostics; submission-grade files contain `_last_logits` and use schema
version 3. The completed audit is in
[docs/CORRECTION_AND_RERUN_CHECKLIST.md](docs/CORRECTION_AND_RERUN_CHECKLIST.md).

## Repository status

The runnable benchmark, corrected CPU/GPU baseline surfaces, attention and
cache interventions, CPU W8A8 performance/quality study, GPU kernel probes,
and reproducible figures are complete. GPU runs were executed outside the
default device-hiding sandbox; the hardware report confirms all eight RTX 6000
Ada devices and the active driver.

Headline observations so far:

- CPU has no batching knee through batch 16 at contexts 128/512; the frozen
  rule identifies batch 16 at context 2048 and batch 8 at context 4096;
- GPU reaches 2,974, 2,948, 2,701, and 1,438 median TPS at contexts 128, 512,
  2048, and 4096 (batch 128). Only context 4096 reaches a knee at batch 128;
- the former GPU OOM points now succeed, proving they were caused by
  full-sequence logits rather than a serving-representative memory boundary;
- eager/SDPA TTFT at context 4096, batch 4 is 6.23x on CPU and 9.34x on GPU;
  profiling verifies SDPA executes the CUDA FlashAttention kernel;
- CPU cache-off median TPOT is 1.85x--17.62x cache-on at five repeated points;
  the single 2048/8 boundary is 88.99x slower. On GPU, cache-off is slightly
  faster at small loads and becomes 4.22x slower only at 2048/8;
- CPU dynamic W8A8 improves median TPS by 1.27x--2.49x, while peak RSS is
  1.16x--1.66x higher;
- on the frozen WikiText-2 slice, FP32 perplexity was 11.27 and W8A8
  perplexity was 44.16, so this quantization recipe is not quality-preserving;
- full ARC-Easy W8A8 accuracy is 42.21% versus 56.44% native (normalized:
  38.97% versus 49.12%);
- GPU TorchAO runs real Ampere INT8 GEMMs but suffers thousands of
  device-to-host scalar synchronizations and is retained as a negative probe;
- 100-repetition GPU headline runs support p99 TTFT of 44.67 ms at 128/1 and
  6.923 s at 4096/128;
- full HellaSwag validation accuracy was 42.83%, or 56.88% with the standard
  length normalization (10,042 examples; standard error about 0.49 points);
- full ARC-Easy test accuracy was 56.44%, or 49.12% with the standard length
  normalization (2,376 examples; standard error about 1.02 points).

See [results/README.md](results/README.md) for the curated artifacts.

## 1. Transfer and install

Use Python 3.10 or newer. On each target machine, copy or clone this repository,
then create a fresh environment:

```bash
cd DSAA4012_Final_Project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

For quantization and task accuracy:

```bash
python -m pip install -e '.[quantization,quality]'
```

Install PyTorch from the official selector first if the target machine requires
a CUDA-specific wheel. Record the resulting wheel, CUDA, and driver versions.

The model revision is pinned in `configs/baseline.yaml`; all derived
configurations inherit it. Model weights and datasets download into the normal
external caches and are not committed.

To use an already downloaded model directory, override the model ID and clear
the remote revision:

```bash
python scripts/smoke_test.py --config configs/cpu.yaml \
  --set model.model_id=/path/to/SmolLM2-360M \
  --set model.revision=null
```

## 2. Validate the environment

Record hardware/software metadata:

```bash
python scripts/inspect_hardware.py --output results/hardware.json
```

Run a deterministic model smoke test on CPU:

```bash
python scripts/smoke_test.py --config configs/cpu.yaml
```

Run it on the RTX 6000 Ada:

```bash
python scripts/smoke_test.py --config configs/gpu_rtx6000_ada.yaml
```

The smoke test checks the planned layer, hidden, MLP, Q-head, KV-head, and
maximum-context metadata. A mismatch exits nonzero and must be investigated.

## 3. Run benchmarks

Every script accepts repeated `--set section.key=value` overrides and `--dry-run`.
Use a tiny run first:

```bash
python scripts/run_baseline_grid.py \
  --config configs/gpu_rtx6000_ada.yaml \
  --set 'workload.context_lengths=[128]' \
  --set 'workload.batch_sizes=[1]' \
  --set workload.warmups=1 \
  --set workload.repetitions=2 \
  --set output.path=results/raw/gpu_tiny.jsonl
```

Then run the CPU and GPU baseline grids:

```bash
python scripts/run_baseline_grid.py --config configs/cpu.yaml
python scripts/run_baseline_grid.py --config configs/gpu_rtx6000_ada.yaml
```

The committed EPYC 9654 CPU run used:

```bash
taskset -c 0-95 python scripts/run_baseline_grid.py \
  --config configs/cpu_epyc9654.yaml \
  --set model.model_id=/data/main/hanxiao/SmolLM2-360M \
  --set model.revision=null
```

Run the serving interventions on each desired device by overriding the runtime
and output path:

```bash
python scripts/run_operator_study.py \
  --set runtime.device=cuda --set runtime.dtype=bfloat16 \
  --set output.path=results/raw/gpu_operator.jsonl

python scripts/run_cache_study.py \
  --set runtime.device=cuda --set runtime.dtype=bfloat16 \
  --set output.path=results/raw/gpu_cache.jsonl

python scripts/run_quantization_study.py \
  --set runtime.device=cuda --set runtime.dtype=bfloat16 \
  --set output.path=results/raw/gpu_quantization.jsonl
```

For CPU, use `runtime.device=cpu`, `runtime.dtype=float32`, and distinct output
paths. Unsupported attention/quantization configurations and OOM boundaries are
written to the JSONL file with their error type. Points skipped above an OOM
batch boundary are also recorded. Successful repetitions resume without
duplication when a command is restarted, while manifest invocation history is
preserved.

The manual greedy loop always produces the requested number of tokens even if
EOS is selected. Tokenization, prompt construction, and terminal output are
outside the measured region. CUDA is synchronized at each token boundary.

## 4. Quality evaluation

Perplexity on WikiText-2:

```bash
python scripts/evaluate_quality.py --config configs/cpu.yaml \
  --metric perplexity --quantization none --max-samples 200
python scripts/evaluate_quality.py --config configs/cpu.yaml \
  --metric perplexity --quantization dynamic_w8a8 --max-samples 200
```

If Hugging Face is unavailable, download `wiki.test.raw` once and pass
`--text-file /path/to/wiki.test.raw`. The evaluator records its path, byte
size, line count, and SHA-256 in every result. The curated run used the first
200 raw records (12,846 scored tokens) from a 1,290,590-byte, 4,358-line file
with SHA-256
`173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`.

HellaSwag and ARC-Easy through lm-evaluation-harness:

```bash
python scripts/evaluate_quality.py --config configs/cpu.yaml \
  --metric tasks --tasks hellaswag arc_easy --quantization none
```

When Hugging Face is unavailable, the included ModelScope-backed HellaSwag
configuration reads a locally downloaded validation file:

```bash
MODELSCOPE_CACHE=data/modelscope/cache modelscope download \
  --dataset modelscope/hellaswag --local_dir data/modelscope/hellaswag
wget -O data/modelscope/hellaswag/hellaswag_val.jsonl \
  https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/hellaswag/hellaswag_val.jsonl
python scripts/evaluate_quality.py --config configs/gpu_rtx6000_ada.yaml \
  --metric tasks --tasks configs/lm_eval/hellaswag_modelscope.yaml \
  --eval-batch-size 32
```

ARC-Easy is available directly as Parquet files from ModelScope. The local
configuration preserves the built-in lm-evaluation-harness prompt, target
mapping, and multiple-choice metrics:

```bash
MODELSCOPE_CACHE=data/modelscope/cache modelscope download \
  --dataset allenai/ai2_arc --local_dir data/modelscope/ai2_arc
python scripts/evaluate_quality.py --config configs/gpu_rtx6000_ada.yaml \
  --metric tasks --tasks configs/lm_eval/arc_easy_modelscope.yaml \
  --eval-batch-size 32
```

The expected HellaSwag validation and ARC-Easy test-file SHA-256 values are
recorded in their task configurations.

Use `--limit` only for a smoke test. Headline quality runs should use complete
tasks or a frozen, seeded subset whose size and uncertainty are reported.

## 5. Aggregate and plot

```bash
python scripts/aggregate_results.py \
  results/raw/gpu_rtx6000_ada_baseline_last_logits.jsonl \
  --output results/processed/gpu_rtx6000_ada_baseline_last_logits.csv

python scripts/make_plots.py \
  results/raw/cpu_epyc9654_baseline_last_logits.jsonl \
  results/raw/gpu_rtx6000_ada_baseline_last_logits.jsonl \
  --output-dir results/figures/corrected_baseline --metric tps
```

Aggregation reports p50, p95, and p99 across individual repetitions. Plotting
generates context-by-batch heatmaps and a TPOT-throughput frontier directly
from raw JSONL files. Broad-grid p99 values have only ten repetitions and are
descriptive; stable tail claims use the 100-repetition headline file.

## 6. Tests

The unit tests do not download a model:

```bash
python -m pytest
python -m ruff check src scripts tests
```

## Experimental cautions

- CPU and GPU results are independent hardware lines, not a direct fairness comparison.
- GPU commands must run in a context that exposes `/dev/nvidia*`; seeing CUDA
  libraries alone is insufficient.
- Peak CPU RSS is sampled in-process; use a fresh process for headline memory points.
- Verify FlashAttention/TorchAO execution with profiler traces before attributing speedups.
- `dynamic_w8a8` uses PyTorch dynamic INT8 Linear on CPU while preserving
  `lm_head` in FP32, and TorchAO on accelerator paths.
- Keep slowdowns, unsupported kernels, and fallback behavior as experimental results.
- Run 100–200 repetitions only for selected headline configurations after the broad grid is stable.

## Team

- Hanxiao
- Haiyang Peng

DSAA4012 Machine Learning Systems, Summer 2026.
