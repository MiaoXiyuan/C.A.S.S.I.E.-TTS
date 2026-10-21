"""Check the C.A.S.S.I.E. environment for local export and inference.

Usage:
    python scripts/check_env.py
"""

from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bootstrap import configure  # noqa: E402

configure()


def pkg(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def main() -> int:
    info: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "project_root": str(ROOT),
        "proxy": {
            "HTTPS_PROXY": os.environ.get("HTTPS_PROXY"),
            "HTTP_PROXY": os.environ.get("HTTP_PROXY"),
        },
        "packages": {
            name: pkg(name)
            for name in (
                "torch",
                "torchaudio",
                "transformers",
                "accelerate",
                "peft",
                "qwen-tts",
                "huggingface-hub",
                "safetensors",
                "librosa",
                "soundfile",
                "numpy",
                "pyworld",
            )
        },
    }

    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = {
                "name": props.name,
                "total_vram_gb": round(props.total_memory / 1e9, 2),
                "capability": f"{props.major}.{props.minor}",
            }
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = repr(exc)

    try:
        import qwen_tts  # noqa: F401

        info["qwen_tts_import"] = True
    except Exception as exc:  # noqa: BLE001
        info["qwen_tts_import"] = False
        info["qwen_tts_error"] = repr(exc)

    models = ROOT / "models"
    info["models"] = {}
    for name in ("Qwen3-TTS-12Hz-1.7B-Base",):
        p = models / name
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0
        info["models"][name] = {
            "exists": p.exists(),
            "size_gb": round(total / 1e9, 2),
            "speech_tokenizer_exists": all(
                (p / "speech_tokenizer" / file).is_file()
                for file in ("config.json", "model.safetensors")
            ),
        }

    try:
        usage = shutil.disk_usage(ROOT)
        info["disk"] = {
            "total_gb": round(usage.total / 1e9, 1),
            "free_gb": round(usage.free / 1e9, 1),
        }
    except OSError:
        pass

    print(json.dumps(info, ensure_ascii=False, indent=2))

    # CUDA is recommended, but a CPU-only installation is still valid.
    ok = "torch" in info and bool(info.get("qwen_tts_import"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
