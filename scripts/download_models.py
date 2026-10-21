"""Download the Qwen3-TTS Base needed for local export and inference.

The Base includes its speech_tokenizer/; no separate training tokenizer is
needed by this inference-only release. Models are downloaded by the user and
are not included in the public package.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --repo base
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

REPOS = {
    "base": {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "local_dir": MODELS / "Qwen3-TTS-12Hz-1.7B-Base",
        "allow_patterns": [
            "config.json",
            "generation_config.json",
            "merges.txt",
            "vocab.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "model.safetensors",
            "speech_tokenizer/*",
            "README.md",
        ],
        "required": ["config.json", "model.safetensors", "speech_tokenizer/model.safetensors"],
    },
}


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", choices=["base"], default="base")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import HfApi, snapshot_download

    names = [args.repo]
    api = HfApi()
    summary: dict = {"models": {}}

    for name in names:
        spec = REPOS[name]
        local_dir: Path = spec["local_dir"]
        local_dir.mkdir(parents=True, exist_ok=True)
        info = api.model_info(spec["repo_id"])
        print(f"[{name}] {spec['repo_id']} @ {info.sha}")
        snapshot_download(
            repo_id=spec["repo_id"],
            revision=info.sha,
            local_dir=str(local_dir),
            allow_patterns=spec["allow_patterns"],
            max_workers=args.workers,
        )
        missing = [p for p in spec["required"] if not (local_dir / p).is_file()]
        if missing:
            raise RuntimeError(f"{name}: missing required files: {missing}")
        size = dir_size(local_dir)
        print(f"[{name}] done: {size / 1e9:.2f} GB -> {local_dir}")
        summary["models"][name] = {
            "repo_id": spec["repo_id"],
            "revision": info.sha,
            "local_dir": str(local_dir),
            "size_gb": round(size / 1e9, 3),
            "required": spec["required"],
        }

    (MODELS / "models.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
