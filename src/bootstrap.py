"""Project-local environment bootstrap.

Must be imported before librosa / numba / qwen_tts.

Some managed runtimes restrict writes to the project workspace. Numba's
default cache lives next to the installed library, which can fail when that
location is read-only. Redirecting NUMBA_CACHE_DIR and temporary files into
the project cache avoids that class of failure.

Importing this module is idempotent and does not require torch.
Set CASSIE_CACHE_DIR to a project-relative directory to relocate these caches.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("CASSIE_CACHE_DIR", "cache"))
if not CACHE.is_absolute():
    CACHE = ROOT / CACHE
TMP = CACHE / "tmp"


def configure() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "numba").mkdir(parents=True, exist_ok=True)
    (CACHE / "huggingface").mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    # The platform TMP/TEMP may be blocked by the sandbox.  Force Python and
    # torch/numba temp files into the project cache before torch is imported.
    os.environ["TMP"] = str(TMP)
    os.environ["TEMP"] = str(TMP)
    tempfile.tempdir = str(TMP)

    os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE / "numba"))
    os.environ.setdefault("HF_HOME", str(CACHE / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Keep any stray torch hub downloads inside the project as well.
    os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))


configure()
