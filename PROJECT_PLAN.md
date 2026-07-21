# DSAA4012 Final Project Plan

## 1. Project identity

### 1.1 Team

- Hanxiao
- Haiyang Peng

### 1.2 Course archetype

Archetype 2.2: Serving-optimization study.

### 1.3 Model

`HuggingFaceTB/SmolLM2-360M-Instruct`

Relevant model properties to record and verify in the implementation:

- decoder-only `LlamaForCausalLM` architecture;
- approximately 360M parameters;
- 32 decoder layers;
- hidden size 960;
- MLP intermediate size 2560;
- 15 query-attention heads and 5 key/value heads (grouped-query attention);
- maximum context length 8192;
- RoPE positional encoding;
- RMSNorm and SiLU-based MLP;
- tied input/output embeddings;
- BF16 checkpoint;
- KV cache enabled by default.

### 1.4 Hardware scope

The main study has two independent hardware lines:

1. One fixed CPU platform.
2. One NVIDIA RTX 6000 Ada Generation GPU platform with 48GB of device memory.

All comparisons within a hardware line must run on the same machine and in a controlled environment. CPU and GPU absolute numbers are not treated as a direct fairness comparison; the project compares scaling behavior, bottleneck transitions, and normalized optimization effects on each platform.

### 1.5 Primary systems question

> How do batch size and context length reshape the TTFT-TPOT-throughput frontier of SmolLM2-360M-Instruct on CPU and RTX 6000 Ada, and when do KV caching, attention implementation, and dynamic W8A8 quantization materially shift that frontier?

### 1.6 Intended headline result

The headline result will characterize workload regions in which inference is dominated by:

- framework or kernel-launch overhead;
- weight and KV-cache memory traffic;
- attention cost from long contexts;
- or matrix-compute throughput.

It will then show which serving choices move the frontier in each region rather than claiming that one optimization is universally best.

---

## 2. Scope control

### 2.1 Core work that must be completed

- Reproducibly load and run SmolLM2-360M-Instruct on CPU and RTX 6000 Ada.
- Implement a deterministic generation benchmark with token-level timing.
- Measure context-length and batch-size scaling on both hardware lines.
- Report TTFT, TPOT, aggregate TPS, p50/p95/p99 latency, and peak memory.
- Identify a batching knee on each hardware line or state honestly that no clear knee was observed.
- Compare at least one realistic attention/operator choice.
- Compare KV cache enabled versus disabled.
- Evaluate dynamic W8A8 quantization on CPU and GPU when supported by stable kernels.
- Measure quantization quality loss with perplexity and task accuracy.
- Produce a runnable artifact with exact reproduction commands.
- Produce presentation, report, contribution statement, and AI-use acknowledgment.

### 2.2 Optional extensions

These are included only after the core result is stable and reproducible:

- W8A16 weight-only quantization as a diagnostic control.
- DynamicCache versus StaticCache plus compiled decoding.
- FlashAttention-2 if the installed stack supports it reliably.
- Context length 8192.
- Batch sizes above 128 at memory-feasible context lengths, after the required RTX 6000 Ada high-batch sweep is complete.
- A minimal streaming HTTP endpoint for end-to-end TTFT.
- Hardware-counter or profiler evidence for memory/compute-bound claims.
- A second CPU runtime such as ONNX Runtime or llama.cpp.

### 2.3 Explicit non-goals

- Training or fine-tuning SmolLM2.
- Developing a new quantization algorithm.
- Exhaustively benchmarking every runtime and precision format.
- Full Cartesian-product testing of all knobs.
- Comparing against much larger language models.
- Treating generated-text subjective quality as the primary accuracy metric.
- Claiming production-scale serving behavior from a local synchronous benchmark.

---

## 3. Experimental model

### 3.1 Inference phases

The benchmark must separate two phases.

#### Prefill

The model consumes the complete input context and produces the first output token. Prefill contributes most of TTFT and is expected to become increasingly sensitive to context length, attention implementation, and matrix-compute efficiency.

#### Decode

The model generates one new token per sequence at each step. Decode contributes TPOT and TPS and is expected to be sensitive to weight traffic, KV-cache access, batch size, runtime overhead, and quantization.

### 3.2 Bottleneck hypotheses

The project starts from the following hypotheses, which must be tested rather than assumed:

1. Small-batch GPU decode may be dominated by memory traffic and dispatch overhead because the 360M model is unlikely to saturate an RTX 6000 Ada.
2. Increasing batch size should initially improve GPU utilization and aggregate TPS, but eventually raise TTFT/TPOT and reach a batching knee.
3. Long contexts should increase TTFT strongly and may make optimized attention operators more valuable.
4. KV caching should drastically reduce decode recomputation, with its memory cost increasing with batch size and sequence length.
5. CPU decode may be bandwidth-bound at small batch sizes, so weight quantization can help even without AMX.
6. Dynamic activation quantization may lose some of its theoretical benefit to runtime scale calculation and conversion overhead.
7. W8A8 may help in some workload regions and hurt in others; the result depends on hardware, matrix shape, and kernel availability.

### 3.3 Interpretation discipline

- A faster result is not automatically attributed to lower precision without checking the executed kernels.
- A slower quantized result is retained and analyzed, not discarded.
- CPU and GPU results are interpreted separately before any cross-hardware comparison.
- Claims about compute- versus memory-bound behavior require arithmetic reasoning, profiler evidence, or a clearly stated inference from scaling behavior.
- Unsupported operators, precision fallbacks, compilation, and graph changes must be documented.

---

## 4. Metrics and definitions

### 4.1 Time to first token (TTFT)

For a request or synchronous batch:

```text
TTFT = timestamp(first generated token available) - timestamp(request/model start)
```

Two forms may be reported:

- model-only TTFT: input token IDs are prepared before timing;
- end-to-end TTFT: includes tokenization and optional serving-interface overhead.

The main experiment uses model-only TTFT. End-to-end TTFT is an optional deployment datapoint.

### 4.2 Time per output token (TPOT)

For a generation that produces `N` output tokens:

```text
TPOT = (timestamp(last token) - timestamp(first token)) / (N - 1)
```

TPOT excludes prefill and measures steady decode behavior. Report the distribution across repeated runs, including p50, p95, and p99.

### 4.3 Aggregate output-token throughput (TPS)

```text
TPS = total output tokens produced by all sequences / measured wall-clock time
```

Prompt tokens are not included in decode TPS. If prefill throughput is reported, it is labeled separately as prompt tokens per second.

### 4.4 End-to-end latency

```text
end-to-end latency = timestamp(last token) - timestamp(request start)
```

This is a supporting metric and does not replace TTFT or TPOT.

### 4.5 Peak memory

- GPU: peak allocated memory and peak reserved memory.
- CPU: peak resident set size measured in a fresh process.
- KV-cache and quantized-weight sizes should be estimated analytically and compared with observed process/device memory.

### 4.6 Quality

- Perplexity on a fixed language-modeling evaluation set.
- Accuracy on selected multiple-choice benchmarks.
- Absolute and relative differences from the BF16/FP32 baseline.
- Evaluation sample count and uncertainty must be stated.

### 4.7 Batching knee

Default operational definition:

> The first batch-size increase for which aggregate TPS improves by less than 10% over the previous tested batch size while p95 TTFT or TPOT continues to increase.

If the data does not exhibit this behavior, the report states that no knee was observed within the tested range.

---

## 5. Stage A - Repository and reproducibility foundation

### 5.1 Objectives

- Establish a clean, versioned project.
- Make every experiment configuration serializable and repeatable.
- Prevent models, datasets, and raw caches from entering Git.

### 5.2 Work items

- Create the project directory structure.
- Record Python, PyTorch, Transformers, TorchAO, CUDA, driver, and auxiliary package versions.
- Pin the Hugging Face model revision or commit hash.
- Add configuration files for hardware, workload, runtime, cache, precision, and repetitions.
- Establish deterministic seeds.
- Establish structured JSON or CSV result schemas.
- Add commands for downloading the model and evaluation datasets without redistributing them.
- Add a machine-readable environment report.

### 5.3 Planned repository structure

```text
DSAA4012_Final_Project/
├── README.md
├── PROJECT_PLAN.md
├── LICENSE
├── pyproject.toml or requirements.txt
├── configs/
│   ├── baseline.yaml
│   ├── cpu.yaml
│   ├── gpu_rtx6000_ada.yaml
│   ├── operator_study.yaml
│   ├── cache_study.yaml
│   └── quantization_study.yaml
├── src/
│   └── mlsys360/
│       ├── model.py
│       ├── prompts.py
│       ├── decode.py
│       ├── timing.py
│       ├── memory.py
│       ├── hardware.py
│       ├── quantization.py
│       └── results.py
├── scripts/
│   ├── inspect_hardware.py
│   ├── smoke_test.py
│   ├── run_baseline_grid.py
│   ├── run_operator_study.py
│   ├── run_cache_study.py
│   ├── run_quantization_study.py
│   ├── evaluate_quality.py
│   └── make_plots.py
├── tests/
├── results/
│   ├── raw/
│   ├── processed/
│   └── figures/
├── report/
├── slides/
└── docs/
```

### 5.4 Acceptance criteria

- A fresh environment can install dependencies from one documented command.
- The model revision and all software versions are recorded.
- Generated files and large model files are excluded from Git.
- Experiment results contain enough metadata to reconstruct the configuration that produced them.

---

## 6. Stage B - Model and hardware validation

### 6.1 Objectives

- Verify that the selected model runs correctly on both platforms.
- Determine available hardware features before choosing quantization backends.

### 6.2 CPU inspection

Record:

- exact CPU model;
- sockets, physical cores, and logical cores;
- RAM capacity and memory topology;
- cache sizes;
- AVX2, AVX-512, VNNI, AMX, and BF16 capabilities where applicable;
- operating system and kernel;
- thread libraries and relevant environment variables;
- NUMA topology;
- power governor or performance mode.

### 6.3 GPU inspection

Record:

- exact RTX 6000 Ada model, 48GB memory capacity, ECC state, and available device memory;
- driver version;
- CUDA runtime and compiler versions;
- PyTorch CUDA build;
- compute capability;
- BF16 and INT8 kernel availability;
- configured power limit, application clocks if available, and observed clock behavior;
- temperature and thermal/power throttling indicators during representative runs;
- idle memory usage and competing processes.

### 6.4 Model validation

- Load the exact model revision and tokenizer.
- Apply the official chat template.
- Generate deterministic output on CPU and GPU.
- Confirm vocabulary, maximum context, dtype, layer count, and cache configuration.
- Confirm CPU and GPU baseline outputs are numerically plausible.
- Record model weight memory and process/device memory.

### 6.5 Acceptance criteria

- The same smoke-test prompt runs on CPU and GPU.
- Fixed greedy decoding is deterministic within an environment.
- Hardware reports contain all features needed to interpret later results.
- Model configuration in the repository agrees with the downloaded checkpoint.

---

## 7. Stage C - Benchmark harness

### 7.1 Objectives

- Produce trustworthy token-level timing independent of terminal output and tokenizer overhead.
- Use the same workload definitions across all configurations.

### 7.2 Input construction

- Build a fixed prompt corpus.
- Apply the chat template before choosing length.
- Produce exact token-length buckets for 128, 512, 2048, and 4096 tokens.
- Reuse identical token IDs across runtime and precision comparisons.
- Use valid attention masks and deterministic padding.
- Store prompt identifiers and hashes rather than duplicating large datasets.

### 7.3 Generation protocol

- Greedy decoding with `do_sample=False`.
- Fixed output length of 64 tokens for the main benchmark.
- Prevent early EOS from changing the amount of work.
- Run under inference mode.
- Disable console printing and string decoding inside timed regions.
- Synchronize GPU execution at timing boundaries.
- Record token timestamps or equivalent first-token and decode intervals.

### 7.4 Repetition protocol

- Warm up each newly loaded configuration.
- Use broad-grid repetitions sufficient for stable p50/p95 estimates.
- Use 100-200 repetitions for headline configurations and p99.
- Randomize or rotate configuration order when thermal drift may bias results.
- Store individual observations, not only aggregates.

### 7.5 Timing outputs

Each result row should include at least:

- hardware ID;
- software versions;
- model revision;
- precision and quantization configuration;
- operator/backend;
- cache mode;
- context length;
- batch size;
- requested and actual output length;
- repetition index;
- TTFT;
- TPOT;
- TPS;
- end-to-end latency;
- peak allocated/reserved GPU memory or CPU RSS;
- warnings, fallbacks, and errors.

### 7.6 Acceptance criteria

- TTFT and TPOT can be recovered from raw observations.
- Repeated baseline measurements have explainable variance.
- Fixed work is performed for every compared configuration.
- GPU timing does not accidentally measure only asynchronous launch time.
- Tokenization is excluded from model-only measurements.

---

## 8. Stage D - Baseline workload surface

### 8.1 Purpose

Map the baseline TTFT-TPOT-TPS frontier before introducing optimizations. This is the primary experiment and must be completed before optional studies.

### 8.2 Fixed settings

- Model: SmolLM2-360M-Instruct.
- GPU precision: BF16.
- CPU precision: stable native baseline selected after hardware validation.
- Attention: SDPA or the most stable common baseline.
- Cache: DynamicCache enabled.
- Output length: 64 tokens.
- Decoding: greedy.

### 8.3 Workload grid

Core context lengths:

```text
128, 512, 2048, 4096
```

Core batch sizes:

```text
1, 2, 4, 8, 16
```

RTX 6000 Ada saturation sweep:

```text
32, 64, and 128 where memory permits
```

The GPU sweep is adaptive rather than a full Cartesian product: test increasing batch sizes at each context length until a batching knee, an out-of-memory boundary, or batch 128 is reached. Record skipped configurations and the stopping reason. This higher-batch sweep is required because a 360M model may remain substantially underutilized at batch 16 on RTX 6000 Ada. Batch sizes above 128 and context length 8192 are optional extensions. The CPU is required to run only the shared core grid; higher CPU batch sizes are optional.

### 8.4 Required analyses

- TTFT versus context length for each batch size.
- TPOT versus context length for each batch size.
- Aggregate TPS versus batch size for each context length.
- Peak memory versus batch and context.
- Normalized CPU and GPU scaling trends.
- Batching knee per hardware line.
- Evidence for underutilized, bandwidth-sensitive, and compute-sensitive regions.

### 8.5 Acceptance criteria

- The complete core grid runs on both platforms or unsupported points are documented.
- The GPU saturation sweep reaches a documented batching knee, memory boundary, or batch 128 for every core context length.
- Results include p50/p95/p99 where required.
- A batching knee is identified by the frozen rule or explicitly not observed.
- At least one headline plot can be generated directly from raw results.

---

## 9. Stage E - Attention and operator study

### 9.1 Purpose

Determine when a more efficient attention implementation changes prefill or decode performance.

### 9.2 Candidate implementations

GPU:

- eager attention;
- PyTorch SDPA;
- FlashAttention-2 if supported.

CPU:

- eager attention;
- PyTorch SDPA.

### 9.3 Representative workload points

```text
low-load:       context 128,  batch 1
medium-load:    context 512,  batch 8
long-context:   context 2048, batch 1
high-throughput context 2048, batch 16
extreme-context context 4096, batch 4
```

### 9.4 Questions

- Does optimized attention primarily improve TTFT at long contexts?
- Is TPOT improvement small when decode query length is one?
- Does operator choice change peak memory?
- At which context length does FlashAttention/SDPA justify its setup overhead?
- Are observed differences actually caused by attention, or is MLP/linear work dominant?

### 9.5 Acceptance criteria

- Backend selection is verified rather than inferred from a configuration string.
- Unsupported or silently falling-back implementations are excluded or labeled.
- Results are explained separately for prefill and decode.

---

## 10. Stage F - KV-cache study

### 10.1 Core comparison

- `use_cache=False`.
- DynamicCache enabled.

### 10.2 Core workload points

```text
contexts: 128, 512, 2048
batches:  1, 8
output:   64 tokens
```

### 10.3 Optional realistic-cache comparison

- DynamicCache.
- StaticCache plus compiled decode, if supported and clearly labeled.

### 10.4 Required analyses

- TPOT as generation position increases.
- Total decode time with and without reuse of prior key/value tensors.
- KV-cache analytical size versus observed memory growth.
- Static versus dynamic allocation trade-off.
- Whether compilation and cache layout effects can be separated.

### 10.5 Acceptance criteria

- Cache-off genuinely recomputes history and cache-on genuinely reuses it.
- Memory growth is consistent with batch, sequence length, layers, KV heads, and element size.
- StaticCache results are not attributed solely to caching if compilation is also enabled.

---

## 11. Stage G - Quantization study

### 11.1 Primary configuration

Dynamic per-token INT8 activations plus per-channel INT8 weights for Linear layers, referred to as dynamic W8A8.

### 11.2 Precision candidates

- Native baseline: BF16 on GPU and the selected native CPU precision.
- W8A8 dynamic.
- W8A16 weight-only as an optional diagnostic.

### 11.3 Backend selection

The implementation begins with a small kernel and model smoke test. A backend is selected independently for CPU and GPU based on:

- actual device support;
- successful model conversion;
- verified quantized Linear execution;
- compatibility with KV cache and generation;
- deterministic output length;
- integration with the quality-evaluation harness;
- reproducible measurements.

TorchAO is the primary candidate. ONNX Runtime or another runtime is a fallback only if it produces a stable, explainable serving path without overwhelming the project scope.

### 11.4 Representative workload points

```text
short single request: context 128,  batch 1
long single request:  context 2048, batch 1
medium batch:         context 512,  batch 8
throughput-oriented:  context 2048, batch 16
```

### 11.5 Required checks

- Identify exactly which modules are quantized.
- Record scale granularity and weight granularity.
- Record remaining native-precision operations.
- Check for device transfers or dequantization boundaries.
- Measure model size and resident memory.
- Confirm that W8A8 performs quantized matrix operations rather than only storing compressed weights.
- Separate prefill and decode effects.

### 11.6 Required analyses

- TTFT, TPOT, and TPS speedup or slowdown.
- CPU versus GPU differences.
- Dynamic activation scaling overhead.
- Weight-traffic reduction.
- Peak-memory reduction.
- Workload regions in which quantization helps or hurts.
- Quality degradation.

### 11.7 Fallback hierarchy

If dynamic W8A8 is not stable on a platform:

1. Document the exact unsupported path or failure.
2. Test W8A16 weight-only as a diagnostic.
3. Use a stable runtime-specific INT8 path if it remains within scope.
4. Preserve the failed/negative result as an implementation limitation rather than fabricating a comparison.

### 11.8 Acceptance criteria

- At least one stable quantized path is evaluated.
- Performance results are paired with quality results.
- Quantized modules and fallbacks are documented.
- No universal speedup claim is made from one workload point.

---

## 12. Stage H - Quality evaluation

### 12.1 Purpose

Measure whether lower precision changes model behavior enough to invalidate the serving improvement.

### 12.2 Primary continuous metric

Perplexity on a fixed, documented language-modeling validation set such as WikiText-2 or a controlled WikiText-103 validation subset.

Perplexity is used because it is more sensitive to small numerical changes than a small-sample accuracy benchmark.

### 12.3 Task benchmarks

Select two from:

- HellaSwag;
- ARC-Easy;
- PIQA.

The selected tasks must use the same evaluation settings across precisions. If a subset is used, its seed, sample count, and uncertainty must be reported.

### 12.4 Comparisons

- Native baseline versus W8A8 dynamic.
- Native baseline versus W8A16 if included.
- Absolute score difference.
- Relative score difference.
- Perplexity change.
- Accuracy confidence intervals or standard errors.

### 12.5 Acceptance criteria

- Quantization performance claims include a corresponding quality cost.
- The evaluation is large enough that sampling noise does not dominate the claimed difference.
- Model chat-template and scoring conventions are documented.
- Quality is evaluated on the exact quantized artifact used for performance measurement where practical.

---

## 13. Stage I - Analysis and visualization

### 13.1 Main figures

The presentation and report should be built around a small number of decisive figures:

1. CPU context-by-batch TTFT or TPS heatmap.
2. GPU context-by-batch TTFT or TPS heatmap.
3. TPOT-throughput frontier and batching knee for each platform.
4. Attention-operator speedup versus context length.
5. KV-cache effect on TPOT across output-token position.
6. Quantization speed-memory-quality trade-off.

### 13.2 Plot requirements

- Every axis has units.
- Every plot names the baseline.
- Error bars or percentile bands are shown where meaningful.
- Hardware, precision, batch, context, output length, cache, and operator are recoverable from caption or legend.
- CPU and GPU plots do not imply direct fairness when configurations differ.
- Raw result files link deterministically to generated figures.

### 13.3 Analytical calculations

Include, where relevant:

- model-weight bytes by precision;
- estimated KV-cache bytes;
- attention complexity as context grows;
- approximate work per generated token;
- measured versus theoretical memory changes;
- normalized speedup;
- marginal TPS improvement when batch doubles.

### 13.4 Acceptance criteria

- Every major conclusion is supported by a plot, table, or calculation.
- Negative results and limitations remain visible.
- No plot mixes incompatible timing definitions.
- Headline plots can be regenerated from repository scripts.

---

## 14. Stage J - Runnable system and artifacts

### 14.1 Runnable slice

The minimum end-to-end path is:

```text
obtain model and prompt data
→ load chosen hardware/runtime configuration
→ execute deterministic generation
→ collect TTFT/TPOT/TPS/memory observations
→ aggregate results
→ generate headline figures
```

### 14.2 README requirements

The final README must contain exact commands for:

- environment setup;
- model and dataset acquisition;
- hardware inspection;
- smoke test;
- baseline grid;
- operator study;
- cache study;
- quantization study;
- quality evaluation;
- result aggregation;
- headline figure generation.

### 14.3 Artifact contents

- Source code.
- Configuration files.
- Tests.
- Result-processing scripts.
- Plot scripts.
- Small processed result tables.
- Hardware and software metadata.
- Run manifests.
- Final README.
- Clear real-versus-stubbed statement.

Large public datasets and Hugging Face model weights are referenced or downloaded by script rather than submitted.

### 14.4 Acceptance criteria

- A clean run reproduces at least one headline configuration and figure.
- Commands do not depend on undocumented local paths.
- Failures for missing CUDA, unsupported kernels, or absent datasets are clear.
- Artifact size remains suitable for course submission.

---

## 15. Stage K - Presentation

### 15.1 Required structure

The 12-minute talk should cover:

1. Problem and pipeline.
2. Two or three shaping systems trade-offs.
3. Evaluation against named baselines.
4. Honest limitations.

### 15.2 Proposed narrative

1. Why a 360M decoder is useful for controlled CPU/GPU serving analysis.
2. Why prefill and decode must be measured separately.
3. How context and batch move TTFT, TPOT, and TPS.
4. Where each hardware platform reaches a batching knee.
5. Why attention, cache, and quantization help only in particular workload regions.
6. What quality and memory are traded for speed.

### 15.3 Q&A preparation

Be prepared to defend:

- why SmolLM2-360M-Instruct was selected;
- why the model remains meaningful on RTX 6000 Ada despite its small parameter count;
- why CPU quantization was included;
- metric definitions;
- batching-knee definition;
- operator/backend verification;
- compute- versus memory-bound claims;
- dynamic quantization overhead;
- quality methodology;
- scaling limits at 100 times the workload;
- trust and safety surface of an instruction model.

### 15.4 Team participation

Both members must present. Suggested ownership:

- Member A: motivation, model, workload, benchmark protocol, CPU results.
- Member B: GPU results, operator/cache/quantization, quality trade-off, limitations.
- Both: Q&A and final interpretation.

---

## 16. Stage L - Report

The report follows the six fixed course sections and remains within four main-text pages, excluding references and appendices.

### 16.1 Problem and system overview

- Model and serving pipeline.
- CPU/GPU scope.
- Primary systems question.
- TTFT/TPOT/TPS motivation.

### 16.2 Design

- Workload-shape design.
- Benchmark timing boundary.
- Hardware control.
- Operator, cache, and quantization choices.
- Alternatives considered and rejected.

### 16.3 Implementation

- What runs end to end.
- Model revision and software stack.
- Instrumentation.
- What is real, optional, incomplete, or stubbed.
- Reproduction entry points.

### 16.4 Evaluation

- Hardware and software setup.
- Baselines.
- Main workload surface.
- Operator/cache/quantization interventions.
- Quality and peak memory.
- Limitations.

### 16.5 Individual-contribution statement

- Design ownership.
- Implementation ownership.
- Experiment ownership.
- Writing ownership.
- Presentation ownership.
- Signed/agreed statement from both members.

### 16.6 AI-use acknowledgment

Record:

- tool names;
- purposes;
- relevant prompts or prompt categories;
- which outputs were used only for brainstorming, clarification, or debugging;
- confirmation that team members produced and verified the design, measurements, analysis, and report language.

---

## 17. Stage M - Individual paper analyses

Each member selects a different paper from the course menu and writes an independent approximately 800-word critical analysis.

Recommended pairing for this project:

- `vLLM / PagedAttention`: serving throughput, batching, and KV-cache management.
- `FlashAttention`: IO-aware algorithms, memory movement, and attention operators.

Each analysis must:

- explain the systems contribution in the member's own words;
- state what the paper evaluated;
- apply course concepts analytically;
- connect to the project where natural;
- assess whether claims follow from evidence;
- identify aged assumptions;
- propose a specific, motivated, falsifiable next experiment.

The individual analyses are not part of the team report PDF.

---

## 18. Risks and mitigations

### 18.1 Quantized kernels do not accelerate the target shapes

Mitigation:

- Verify kernels early.
- Compare W8A8 with W8A16.
- Retain slowdowns as valid results.
- Avoid claiming theoretical INT8 speedup without execution evidence.

### 18.2 CPU lacks expected ISA support

Mitigation:

- Inspect exact CPU features.
- Treat ISA capability as part of the explanation.
- Select a stable CPU baseline and quantization backend.

### 18.3 Full experiment matrix is too large

Mitigation:

- Complete the baseline surface first.
- Use representative workload points for interventions.
- Keep optional studies behind explicit go/no-go gates.

### 18.4 p99 estimates are unstable

Mitigation:

- Preserve individual observations.
- Use more repetitions for headline points.
- State sample count.
- Avoid overinterpreting one extreme sample.

### 18.5 Early EOS changes work per request

Mitigation:

- Enforce fixed generated-token count or use a manual decode loop.
- Record actual output length and reject mismatched runs.

### 18.6 Attention backend silently falls back

Mitigation:

- Record backend selection.
- Inspect logs/profiler traces.
- Label or exclude fallback results.

### 18.7 StaticCache confounds caching and compilation

Mitigation:

- Treat it as a combined serving configuration unless isolated experimentally.
- Keep cache-on/cache-off as the causal cache result.

### 18.8 Quality differences are smaller than benchmark noise

Mitigation:

- Use perplexity as the primary sensitive metric.
- Use sufficiently large task evaluation samples.
- Report uncertainty.

### 18.9 CPU and GPU software paths are not identical

Mitigation:

- Analyze each hardware line independently.
- Use normalized within-platform comparisons.
- Document backend and kernel differences.

### 18.10 Project direction differs from topic registration

Mitigation:

- Notify the instructor that the team remains in Archetype 2.2 but changes the model and measurement details.
- Preserve the original latency-throughput systems question.

---

## 19. Final acceptance checklist

### System

- [ ] SmolLM2-360M-Instruct runs on the selected CPU.
- [ ] SmolLM2-360M-Instruct runs on RTX 6000 Ada.
- [ ] Deterministic fixed-work generation is implemented.
- [ ] The benchmark emits raw token-level timing observations.
- [ ] Peak CPU and GPU memory are measured.

### Baseline evaluation

- [ ] Context-length scaling is measured.
- [ ] Batch-size scaling is measured.
- [ ] TTFT, TPOT, TPS, p50, p95, and p99 are reported.
- [ ] A batching knee is identified or honestly not observed.

### Serving choices

- [ ] Attention/operator comparison is verified.
- [ ] KV cache on/off comparison is complete.
- [ ] Dynamic W8A8 is evaluated or its unsupported path is documented.
- [ ] Quantized modules and fallbacks are recorded.
- [ ] Performance, memory, and quality are considered together.

### Analysis

- [ ] Prefill and decode are analyzed separately.
- [ ] CPU and GPU are analyzed separately.
- [ ] Compute/memory-bound claims have supporting evidence.
- [ ] Limitations and negative results are included.
- [ ] Figures are generated reproducibly from result files.

### Deliverables

- [ ] README reproduces headline numbers.
- [ ] Runnable slice works from submitted code.
- [ ] Report uses all six required sections.
- [ ] Main report text is no more than four pages.
- [ ] Both members present.
- [ ] Contribution statement is complete.
- [ ] AI-use acknowledgment is complete.
- [ ] Each member submits a different approximately 800-word paper analysis.
- [ ] Model, datasets, libraries, and open-source components are cited.
