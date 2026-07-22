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
- Matching `.manifest.json` files: full configurations and invocation history.
- `processed/*.csv`: p50/p95/p99 aggregates generated from the raw JSONL files.

Probe, thread-selection, and smoke-test outputs remain under the ignored
`results/intermediate/` directory and are not submission results.

The GPU line is not included yet because the current execution environment
could not communicate with the NVIDIA driver.
