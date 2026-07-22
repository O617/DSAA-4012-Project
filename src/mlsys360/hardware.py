"""Machine-readable hardware and software inventory."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _command(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def inspect_hardware() -> dict[str, Any]:
    import psutil
    import torch
    import transformers

    memory = psutil.virtual_memory()
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "hostname": platform.node(),
        "python": sys.version,
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "ram_bytes": memory.total,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
        },
        "lscpu": _command(["lscpu", "--json"]),
        "numactl": _command(["numactl", "--hardware"]),
        "nvidia_smi": _command(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,memory.free,driver_version,compute_cap,"
                "power.limit,clocks.current.sm,temperature.gpu,ecc.mode.current",
                "--format=csv,noheader,nounits",
            ]
        ),
        "nvidia_compute_apps": _command(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        ),
        "cuda_available": torch.cuda.is_available(),
        "gpus": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            report["gpus"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                }
            )
    return report


def write_hardware_report(path: str | Path) -> dict[str, Any]:
    report = inspect_hardware()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
