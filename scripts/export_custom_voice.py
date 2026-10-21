"""Export the released LoRA adapter as a Qwen3-TTS CustomVoice checkpoint.

Input layout after downloading both GitHub Release weight assets:
    lora/adapter/adapter_config.json
    lora/adapter/adapter_model.safetensors
    lora/speaker_embedding.safetensors
    lora/best_meta.json
    lora/README.md

The lora/ directory in the source repository includes configuration, metadata
and its README. Download adapter_model.safetensors and speaker_embedding.safetensors
separately from this project's GitHub Releases and place them as shown above;
see README.md.
The base model must also be downloaded separately by the user.

The script:
  * merges the LoRA at the requested scale (default 0.3 for the published model);
  * inserts the 2048-d speaker embedding into codec_embedding[spk_id];
  * switches config.tts_model_type to "custom_voice";
  * adds spk_id / spk_is_dialect entries;
  * drops speaker_encoder.* weights;
  * writes a complete model directory with tokenizer and speech tokenizer files.

It does not modify the base model, the adapter, or any external data.

Usage:
    python scripts/export_custom_voice.py \
        --checkpoint lora \
        --output outputs/cassie_custom_voice_s03 \
        --base models/Qwen3-TTS-12Hz-1.7B-Base \
        --speaker cassie --speaker-id 3000 --lora-scale 0.3
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bootstrap import configure  # noqa: E402

configure()

from peft import PeftModel  # noqa: E402
from peft.tuners.lora import LoraLayer  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

COPY_FILES = [
    "generation_config.json",
    "merges.txt",
    "vocab.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
]


def set_lora_scale(model, scale: float) -> None:
    for module in model.modules():
        if isinstance(module, LoraLayer):
            module.set_scale("default", scale)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="lora")
    parser.add_argument("--output", default="outputs/cassie_custom_voice_s03")
    parser.add_argument("--base", default="models/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--speaker", default="cassie")
    parser.add_argument("--speaker-id", type=int, default=3000)
    parser.add_argument("--lora-scale", type=float, default=0.3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Resolve relative arguments against the project, independent of shell cwd.
    ckpt = (ROOT / args.checkpoint).resolve()
    out = (ROOT / args.output).resolve()
    base = (ROOT / args.base).resolve()
    for protected in (ckpt, base, (ROOT / "lora").resolve()):
        if out.is_relative_to(protected) or protected.is_relative_to(out):
            parser.error("output must not overlap the LoRA checkpoint or base model")
    adapter_dir = ckpt / "adapter"
    if not adapter_dir.is_dir():
        raise SystemExit(f"missing adapter dir: {adapter_dir}")
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory is not empty: {out} (use --overwrite)")
    out.mkdir(parents=True, exist_ok=True)

    free_gb = shutil.disk_usage(out).free / 1e9
    print(f"output: {out}")
    print(f"free disk: {free_gb:.1f} GB")
    if free_gb < 8:
        raise SystemExit("not enough free disk space (need ~5 GB for export)")

    best_meta = {}
    if (ckpt / "best_meta.json").is_file():
        best_meta = json.loads((ckpt / "best_meta.json").read_text(encoding="utf-8"))
    speaker_payload = load_file(str(ckpt / "speaker_embedding.safetensors"))
    speaker_embedding = speaker_payload["target_speaker_embedding"]
    if speaker_embedding.ndim != 1 or speaker_embedding.numel() != 2048:
        raise SystemExit(f"unexpected speaker embedding shape: {tuple(speaker_embedding.shape)}")

    from qwen_tts import Qwen3TTSModel  # local import after bootstrap

    print("loading base + adapter ...")
    tts = Qwen3TTSModel.from_pretrained(
        str(base),
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    peft_model = PeftModel.from_pretrained(tts.model, str(adapter_dir))
    set_lora_scale(peft_model, float(args.lora_scale))
    merged = peft_model.merge_and_unload()
    merged.eval()
    print("merged")

    state_cpu: dict[str, torch.Tensor] = {}
    skipped = 0
    for key, value in merged.state_dict().items():
        if key.startswith("speaker_encoder."):
            skipped += 1
            continue
        state_cpu[key] = value.detach().to("cpu").contiguous()

    weight_key = "talker.model.codec_embedding.weight"
    if weight_key not in state_cpu:
        raise SystemExit(f"missing key: {weight_key}")
    weight = state_cpu[weight_key]
    if args.speaker_id < 0 or args.speaker_id >= weight.shape[0]:
        raise SystemExit(f"speaker id {args.speaker_id} outside embedding table {tuple(weight.shape)}")
    weight[args.speaker_id] = speaker_embedding.to(dtype=weight.dtype).contiguous()

    config_path = base / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["tts_model_type"] = "custom_voice"
    talker = config.setdefault("talker_config", {})
    talker["spk_id"] = {args.speaker: int(args.speaker_id)}
    talker["spk_is_dialect"] = {args.speaker: False}
    (out / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for name in COPY_FILES:
        src = base / name
        if not src.is_file():
            raise SystemExit(f"missing base file: {src}")
        shutil.copy2(src, out / name)

    speech_src = base / "speech_tokenizer"
    if not speech_src.is_dir():
        raise SystemExit(f"missing speech_tokenizer: {speech_src}")
    shutil.copytree(speech_src, out / "speech_tokenizer", dirs_exist_ok=True)

    print("saving model.safetensors ...")
    save_file(state_cpu, str(out / "model.safetensors"))
    save_file(
        {"target_speaker_embedding": speaker_embedding.contiguous()},
        str(out / "speaker_embedding.safetensors"),
    )
    n_tensors = len(state_cpu)
    del state_cpu, merged, peft_model, tts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(ckpt),
        "source_adapter": str(adapter_dir),
        "base_model": str(base),
        "speaker": args.speaker,
        "speaker_id": int(args.speaker_id),
        "lora_scale": float(args.lora_scale),
        "best_meta": best_meta,
        "speaker_encoder_weights_removed": skipped,
        "state_tensors": n_tensors,
        "files": {},
    }
    # Record sizes after writing.
    for path in sorted(out.rglob("*")):
        if path.is_file():
            meta["files"][str(path.relative_to(out))] = path.stat().st_size
    (out / "custom_voice_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps({k: meta[k] for k in ("speaker", "speaker_id", "lora_scale", "source_checkpoint")}, ensure_ascii=False, indent=2))
    print(f"exported: {out}")
    print(f"model.safetensors: {(out / 'model.safetensors').stat().st_size / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
