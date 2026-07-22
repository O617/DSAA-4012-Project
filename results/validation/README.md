# Validation records

The corrected benchmark was validated in the isolated `dsaa4012-rerun` Conda
environment. It was created without modifying the original `baseline`
environment:

```bash
conda create -n dsaa4012-rerun --clone baseline -y
conda run -n dsaa4012-rerun python -m pip uninstall -y \
  mlflow verl opentelemetry-exporter-prometheus
conda run -n dsaa4012-rerun python -m pip install \
  'accelerate>=0.34,<2' 'datasets>=2.20,<5' 'torchao==0.16.0' 'chardet<6'
```

TorchAO 0.16.0 is pinned because it is the official C++-extension match for
PyTorch 2.10.0. TorchAO 0.17.0 supports PyTorch 2.10 only for Python APIs and
skips its compiled extensions in this combination.

The removed packages were unrelated to this repository and caused broken
requirements in the source environment. `pip_check_dsaa4012_rerun.txt` records
that the isolated target environment has no broken requirements.

`cpu_last_logits_smoke.jsonl` is a pre-commit diagnostic. Its manifest is
expected to report a dirty worktree because it verifies the correction before
the correction commit. Headline-run manifests must report a clean worktree and
the subsequent corrected commit.

`gpu_torchao016_profile.txt` verifies that the accelerator W8A8 probe executes
`aten::_int_mm` and Ampere INT8 GEMM kernels. It also records the thousands of
device-to-host scalar transfers that explain why this correct kernel path is
not a practical optimization in the tested eager TorchAO recipe.

`gpu_attention_profile.txt` verifies the requested attention backends at
context 4096/batch 4. Eager uses explicit `bmm` and softmax, while SDPA invokes
`aten::_flash_attention_forward` and `pytorch_flash::flash_fwd_kernel`.

`gpu_headline_telemetry.csv` contains one idle/pre-block sample and one sample
after each of ten alternating-order headline blocks. The 200 observations were
collected at clean revision `773adcb`; all invocation manifests report a clean
worktree.
