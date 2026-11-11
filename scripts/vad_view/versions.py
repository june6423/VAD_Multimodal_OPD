"""Print the version and the path of each package of the VIEW-001 run, and the CUDA device (check_view.sh)."""

import importlib

import torch

for name in ("torch", "flash_attn", "fla", "verl", "transformers", "safetensors", "PIL", "numpy"):
    m = importlib.import_module(name)
    print(name, getattr(m, "__version__", "?"), getattr(m, "__file__", "?"))
print("cuda", torch.version.cuda, "devices", torch.cuda.device_count(), torch.cuda.get_device_name(0),
      torch.cuda.get_device_capability(0))
