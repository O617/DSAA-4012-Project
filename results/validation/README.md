# Validation records

The corrected benchmark was validated in the isolated `dsaa4012-rerun` Conda
environment. It was created without modifying the original `baseline`
environment:

```bash
conda create -n dsaa4012-rerun --clone baseline -y
conda run -n dsaa4012-rerun python -m pip uninstall -y \
  mlflow verl opentelemetry-exporter-prometheus
conda run -n dsaa4012-rerun python -m pip install \
  'accelerate>=0.34,<2' 'datasets>=2.20,<5' 'torchao>=0.8,<1' 'chardet<6'
```

The removed packages were unrelated to this repository and caused broken
requirements in the source environment. `pip_check_dsaa4012_rerun.txt` records
that the isolated target environment has no broken requirements.

`cpu_last_logits_smoke.jsonl` is a pre-commit diagnostic. Its manifest is
expected to report a dirty worktree because it verifies the correction before
the correction commit. Headline-run manifests must report a clean worktree and
the subsequent corrected commit.
