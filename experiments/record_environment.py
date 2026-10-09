import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch

out = Path("outputs/2026-10-09")
out.mkdir(parents=True, exist_ok=True)


def cmd(args):
    p = subprocess.run(args, capture_output=True, text=True)
    return dict(command=args, exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)


info = dict(
    python=sys.version,
    python_executable=sys.executable,
    platform=platform.platform(),
    torch=torch.__version__,
    torch_cuda=torch.version.cuda,
    cuda_available=torch.cuda.is_available(),
    gpu=torch.cuda.get_device_name(),
    capability=torch.cuda.get_device_capability(),
    gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
    disk=shutil.disk_usage(".")._asdict(),
    meminfo=Path("/proc/meminfo").read_text(),
    commands=[cmd(["nvidia-smi"]), cmd(["free", "-h"]), cmd(["df", "-h", ".", "/"])],
    packages={
        p: importlib.metadata.version(p)
        for p in [
            "torch",
            "torchvision",
            "numpy",
            "scipy",
            "open-clip-torch",
            "gdown",
            "huggingface_hub",
        ]
    },
)
for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
    if Path(p).exists():
        info[p] = Path(p).read_text().strip()
(out / "environment.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
print(json.dumps(info, indent=2), flush=True)
