"""C.A.S.S.I.E. CustomVoice local inference server (public release).

This server serves the final LoRA export as a single CustomVoice model.
It does not ship reference audio and does not expose ICL modes.

Runtime parameters:
    --host         Bind address (default: 127.0.0.1)
    --port         HTTP port (default: 7860)
    --model        Model id from MODEL_REGISTRY (default: v1_s03)
    --device       Torch device, e.g. cuda:0 or cpu
    --speaker      Speaker name stored in the exported CustomVoice model
    --no-browser   Do not open the WebUI automatically

Data interface:
    Model directory:  outputs/cassie_custom_voice_s03/
    Generated WAVs:   outputs/web_ui/
    WebUI document:   web/index.html

HTTP API:
    GET  /api/status    Engine and model status (JSON)
    POST /api/generate  JSON body {text, model, language, seed, instruct,
                        sampling..., output_f0}; returns WAV
    GET  /api/audio/<name>  Retained WAV download (X-Audio-URL in generation response)
    POST /api/select    Select v1_s03 (compatibility endpoint)

JSON errors: 400 invalid input, 404 unknown route/audio, 500 generation failure.
No absolute model paths or raw exceptions are returned to clients.

All paths below are resolved relative to the repository root.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import secrets
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bootstrap import configure  # noqa: E402

configure()

from qwen_tts import Qwen3TTSModel  # noqa: E402

from src.inference.f0 import process_f0  # noqa: E402

# Public release: only the final CustomVoice S0.3 export is supported.
DEFAULT_OUTPUT_F0 = {
    "pitch": 0.0,
    "mode": "off",
    "limit": 1.0,
    "step": 1.5,
}

LANGUAGES = [
    "Auto",
    "English",
    "Chinese",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]
EXPERIMENTAL_LANGUAGES = set(LANGUAGES) - {"Auto"}

MODEL_REGISTRY = [
    {
        "id": "v1_s03",
        "label": "CustomVoice · C.A.S.S.I.E. v1 · s0.3",
        "kind": "custom",
        "family": "v1",
        "family_label": "C.A.S.S.I.E. v1",
        "scale": 0.3,
        "path": ROOT / "outputs" / "cassie_custom_voice_s03",
    }
]


def available_models() -> list[dict]:
    return [m for m in MODEL_REGISTRY if (m["path"] / "model.safetensors").is_file()]


class RequestError(ValueError):
    """A client input error with a path-free, public message."""


def _as_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    raise RequestError("采样开关必须是 JSON boolean")


def _check_range(name: str, value: float, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestError(f"{name} 必须是数字")
    try:
        value = float(value)
    except OverflowError:
        raise RequestError(f"{name} 超出范围") from None
    if not (lo <= value <= hi):
        raise RequestError(f"{name} 必须在 {lo} 到 {hi} 之间")
    return value


def _check_int(name: str, value: int, lo: int, hi: int) -> int:
    value = _check_range(name, value, lo, hi)
    if not value.is_integer():
        raise RequestError(f"{name} 必须是整数")
    return int(value)


def _f0_params(
    *,
    pitch: float,
    mode: str,
    limit: float,
    step: float,
) -> dict:
    mode = str(mode or "off").strip().lower()
    if mode not in {"off", "flat", "limited", "quantized"}:
        raise RequestError("不支持的 F0 模式")
    return {
        "pitch": round(_check_range("pitch", pitch, -12.0, 6.0), 2),
        "mode": mode,
        "limit": round(_check_range("limit", limit, 0.0, 12.0), 2),
        "step": round(_check_range("step", step, 0.1, 6.0), 2),
    }


class Engine:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.Lock()
        self.tts = None
        self.output_f0 = dict(DEFAULT_OUTPUT_F0)
        self.current_id = None
        self.current_path = None
        self.current_kind = "custom"
        self.speakers = []
        self.load_seconds = 0.0
        self.generated = 0
        self.out_dir = ROOT / "outputs" / "web_ui"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.models = available_models()
        if not self.models:
            raise SystemExit("没有可用的模型目录")
        ids = [m["id"] for m in self.models]
        if args.model not in ids:
            raise SystemExit(f"--model 必须是 {ids} 之一")
        self.load(args.model)

    def _release(self) -> None:
        self.tts = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_custom(self, model: dict) -> None:
        t0 = time.time()
        print(f"[engine] loading {model['id']}", flush=True)
        tts = Qwen3TTSModel.from_pretrained(
            model["path"],
            device_map=self.args.device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        speakers = list(tts.get_supported_speakers() or [])
        if self.args.speaker.lower() not in [s.lower() for s in speakers]:
            raise ValueError(f"speaker {self.args.speaker!r} not in {speakers}")
        self.tts = tts
        self.current_id = model["id"]
        self.current_path = model["path"]
        self.current_kind = "custom"
        self.speakers = speakers
        self.load_seconds = round(time.time() - t0, 2)
        print(f"[engine] ready {model['id']} in {self.load_seconds}s speakers={speakers}", flush=True)

    def load(self, model_id: str) -> None:
        matches = [m for m in self.models if m["id"] == model_id]
        if not matches:
            raise RequestError("model 只支持 v1_s03")
        model = matches[0]
        self._release()
        self._load_custom(model)

    def sampling_defaults(self) -> dict:
        return {
            "temperature": float(self.args.temperature),
            "top_p": float(self.args.top_p),
            "top_k": int(self.args.top_k),
            "repetition_penalty": float(self.args.repetition_penalty),
            "do_sample": bool(self.args.do_sample),
            "subtalker_temperature": float(self.args.subtalker_temperature),
            "subtalker_top_p": float(self.args.subtalker_top_p),
            "subtalker_top_k": int(self.args.subtalker_top_k),
            "subtalker_dosample": bool(self.args.subtalker_dosample),
            "max_new_tokens": int(self.args.max_new_tokens),
            "seed": int(self.args.seed),
        }

    def status(self) -> dict:
        vram = 0.0
        if torch.cuda.is_available():
            vram = round(float(torch.cuda.memory_allocated() / 1e9), 2)
        return {
            "ready": self.tts is not None,
            "models": [
                {k: v for k, v in m.items() if k not in ("path", "adapter")}
                for m in self.models
            ],
            "current_model": self.current_id,
            "current_model_path": (
                Path(self.current_path).relative_to(ROOT).as_posix()
                if self.current_path and Path(self.current_path).is_relative_to(ROOT)
                else None
            ),
            "current_kind": self.current_kind,
            "speaker": self.args.speaker,
            "speakers": self.speakers,
            "language_default": "Auto",
            "languages": LANGUAGES,
            "experimental_languages": sorted(EXPERIMENTAL_LANGUAGES),
            "output_f0": self.output_f0,
            "voice_shaping_defaults": {
                "output": dict(DEFAULT_OUTPUT_F0),
            },
            "device": self.args.device,
            "vram_allocated_gb": vram,
            "load_seconds": self.load_seconds,
            "generated": self.generated,
            "sample_rate": 24000,
            "sampling_defaults": self.sampling_defaults(),
        }

    def select(self, model_id: str) -> dict:
        with self.lock:
            if model_id != self.current_id:
                self.load(model_id)
        return self.status()

    def generate(
        self,
        text: str,
        model_id: str,
        language: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        do_sample: bool,
        subtalker_temperature: float,
        subtalker_top_p: float,
        subtalker_top_k: int,
        subtalker_dosample: bool,
        seed: int,
        instruct: str | None = None,
        output_f0: dict | None = None,
    ) -> tuple[np.ndarray, int, float, str, str, int, dict]:
        text = (text or "").strip()
        if not text:
            raise RequestError("文本为空")
        if language not in LANGUAGES:
            raise RequestError("不支持的语言")
        instruct = (instruct or "").strip()
        if len(instruct) > 2000:
            raise RequestError("instruct 过长，请控制在 2000 字符以内")

        max_new_tokens = _check_int("max_new_tokens", max_new_tokens, 64, 2048)
        temperature = _check_range("temperature", temperature, 0.05, 2.0)
        top_p = _check_range("top_p", top_p, 0.05, 1.0)
        top_k = _check_int("top_k", top_k, 0, 200)
        repetition_penalty = _check_range("repetition_penalty", repetition_penalty, 1.0, 2.0)
        subtalker_temperature = _check_range(
            "subtalker_temperature", subtalker_temperature, 0.05, 2.0
        )
        subtalker_top_p = _check_range("subtalker_top_p", subtalker_top_p, 0.05, 1.0)
        subtalker_top_k = _check_int("subtalker_top_k", subtalker_top_k, 0, 200)
        seed = _check_int("seed", seed, -1, 2_147_483_647)
        if seed == -1:
            seed = secrets.randbits(31)
        text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        print(
            "[generate] "
            f"model={model_id} seed={seed} temp={temperature} top_p={top_p} top_k={top_k} "
            f"rep={repetition_penalty} do_sample={do_sample} "
            f"sub_temp={subtalker_temperature} sub_top_p={subtalker_top_p} "
            f"sub_top_k={subtalker_top_k} sub_sample={subtalker_dosample} "
            f"max_tokens={max_new_tokens} text_len={len(text)} text_sha={text_sha} "
            f"instruct_len={len(instruct)}",
            flush=True,
        )

        params = dict(DEFAULT_OUTPUT_F0)
        if output_f0 is not None:
            if not isinstance(output_f0, dict) or set(output_f0) - set(params):
                raise RequestError("output_f0 必须是仅含 pitch、mode、limit、step 的 JSON 对象")
            params.update(output_f0)
        if not isinstance(params["mode"], str):
            raise RequestError("F0 mode 必须是字符串")
        params = _f0_params(**params)
        with self.lock:
            if model_id != self.current_id:
                self.load(model_id)
            self.output_f0 = params
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            t0 = time.time()
            gen_kwargs = dict(
                text=text,
                language=language,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                subtalker_dosample=subtalker_dosample,
                subtalker_temperature=subtalker_temperature,
                subtalker_top_p=subtalker_top_p,
                subtalker_top_k=subtalker_top_k,
            )
            if instruct:
                gen_kwargs["instruct"] = instruct
            wavs, sr = self.tts.generate_custom_voice(
                speaker=self.args.speaker,
                **gen_kwargs,
            )
            elapsed = time.time() - t0
            wav = np.asarray(wavs[0], dtype=np.float32)
            peak = float(np.max(np.abs(wav))) if wav.size else 0.0
            if peak > 0:
                wav = (wav * (0.95 / peak)).astype(np.float32)
            if self.output_f0["mode"] != "off" or abs(self.output_f0["pitch"]) > 1e-6:
                t_post = time.time()
                wav = process_f0(
                    wav,
                    int(sr),
                    semitones=float(self.output_f0["pitch"]),
                    mode=str(self.output_f0["mode"]),
                    limit=float(self.output_f0["limit"]),
                    step=float(self.output_f0["step"]),
                )
                elapsed += time.time() - t_post
                print(
                    f"[output-f0] pitch={self.output_f0['pitch']:+.2f} "
                    f"mode={self.output_f0['mode']}",
                    flush=True,
                )
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = self.out_dir / f"{stamp}_{self.generated:04d}.wav"
            sf.write(str(output), wav, sr, subtype="PCM_16")
            self.generated += 1
            return wav, int(sr), elapsed, self.current_id, output.relative_to(ROOT).as_posix(), seed, params


class Handler(BaseHTTPRequestHandler):
    server_version = "CustomVoiceUI/1.0.0"
    engine: Engine

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        print(f"[http] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, path: Path) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        route = urlsplit(self.path).path
        try:
            if route in ("/", "/index.html"):
                self._html(ROOT / "web" / "index.html")
            elif route == "/api/status":
                self._json(200, self.engine.status())
            elif route.startswith("/api/audio/"):
                name = unquote(route[len("/api/audio/"):])
                path = (self.engine.out_dir / name).resolve()
                if (
                    not name or "/" in name or "\\" in name
                    or Path(name).suffix.lower() != ".wav"
                    or path.parent != self.engine.out_dir.resolve()
                    or not path.is_file()
                ):
                    self._json(404, {"error": "audio not found"})
                    return
                body = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json(404, {"error": "not found"})
        except (OSError, ValueError):
            self._json(500, {"error": "读取失败，请检查服务端文件是否完整"})

    def do_POST(self) -> None:  # noqa: N802
        route = urlsplit(self.path).path
        if route not in {"/api/select", "/api/generate"}:
            self._json(404, {"error": "not found"})
            return
        try:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise RequestError("Content-Length 必须是整数") from None
            if length <= 0 or length > 1_000_000:
                raise RequestError("请求体无效（上限 1 MB）")
            if self.headers.get_content_type() != "application/json":
                raise RequestError("Content-Type 必须是 application/json")
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeError):
                raise RequestError("请求体必须是有效 JSON 对象") from None
            if not isinstance(data, dict):
                raise RequestError("请求体必须是 JSON 对象")
            allowed = {"model"} if route == "/api/select" else {
                "text", "model", "language", "instruct", "seed", "output_f0",
                "max_new_tokens", "temperature", "top_p", "top_k",
                "repetition_penalty", "do_sample", "subtalker_temperature",
                "subtalker_top_p", "subtalker_top_k", "subtalker_dosample",
            }
            if set(data) - allowed:
                raise RequestError("请求包含不支持的字段，请查看 README 中的 API 参数")
            for field in ("text", "model", "language", "instruct"):
                if field in data and not isinstance(data[field], str):
                    raise RequestError(f"{field} 必须是字符串")
            if route == "/api/select":
                self._json(200, self.engine.select(data.get("model", "")))
                return
            wav, sr, elapsed, model_id, output, actual_seed, output_f0 = self.engine.generate(
                text=data.get("text", ""),
                model_id=data.get("model", self.engine.current_id),
                language=data.get("language", "Auto"),
                max_new_tokens=data.get("max_new_tokens", self.engine.args.max_new_tokens),
                temperature=data.get("temperature", self.engine.args.temperature),
                top_p=data.get("top_p", self.engine.args.top_p),
                top_k=data.get("top_k", self.engine.args.top_k),
                repetition_penalty=data.get("repetition_penalty", self.engine.args.repetition_penalty),
                do_sample=_as_bool(
                    data.get("do_sample", self.engine.args.do_sample),
                    self.engine.args.do_sample,
                ),
                subtalker_temperature=data.get("subtalker_temperature", self.engine.args.subtalker_temperature),
                subtalker_top_p=data.get("subtalker_top_p", self.engine.args.subtalker_top_p),
                subtalker_top_k=data.get("subtalker_top_k", self.engine.args.subtalker_top_k),
                subtalker_dosample=_as_bool(
                    data.get("subtalker_dosample", self.engine.args.subtalker_dosample),
                    self.engine.args.subtalker_dosample,
                ),
                seed=data.get("seed", self.engine.args.seed),
                instruct=str(data.get("instruct", "") or ""),
                output_f0=data.get("output_f0"),
            )
            buf = io.BytesIO()
            sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
            body = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Duration", f"{len(wav) / sr:.3f}")
            self.send_header("X-Elapsed", f"{elapsed:.3f}")
            self.send_header("X-Model", model_id)
            self.send_header("X-Mode", self.engine.current_kind)
            self.send_header("X-Output-F0", json.dumps(output_f0, ensure_ascii=False))
            self.send_header("X-Speaker", self.engine.args.speaker)
            self.send_header("X-Language", data.get("language", "Auto"))
            self.send_header("X-Sample-Rate", str(sr))
            self.send_header("X-Output", output)
            self.send_header("X-Audio-URL", f"/api/audio/{Path(output).name}")
            self.send_header("X-Seed", str(actual_seed))
            self.send_header("X-Text-Length", str(len(data.get("text", ""))))
            self.send_header(
                "X-Text-SHA256",
                hashlib.sha256(str(data.get("text", "")).encode("utf-8")).hexdigest()[:16],
            )
            self.send_header("X-Instruct-Length", str(len(str(data.get("instruct", "") or "").strip())))
            self.send_header("X-Temperature", str(data.get("temperature", self.engine.args.temperature)))
            self.send_header("X-Top-P", str(data.get("top_p", self.engine.args.top_p)))
            self.send_header("X-Top-K", str(data.get("top_k", self.engine.args.top_k)))
            self.send_header(
                "X-Repetition-Penalty",
                str(data.get("repetition_penalty", self.engine.args.repetition_penalty)),
            )
            self.send_header(
                "X-Do-Sample", "1" if _as_bool(data.get("do_sample", self.engine.args.do_sample), self.engine.args.do_sample) else "0"
            )
            self.send_header(
                "X-Subtalker-Temperature",
                str(data.get("subtalker_temperature", self.engine.args.subtalker_temperature)),
            )
            self.send_header(
                "X-Subtalker-Top-P",
                str(data.get("subtalker_top_p", self.engine.args.subtalker_top_p)),
            )
            self.send_header(
                "X-Subtalker-Top-K",
                str(data.get("subtalker_top_k", self.engine.args.subtalker_top_k)),
            )
            self.send_header(
                "X-Subtalker-Do-Sample",
                "1" if _as_bool(data.get("subtalker_dosample", self.engine.args.subtalker_dosample), self.engine.args.subtalker_dosample) else "0",
            )
            self.end_headers()
            self.wfile.write(body)
        except RequestError as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            print(f"[generate] failed: {type(exc).__name__}", flush=True)
            self._json(500, {"error": "生成失败，请检查模型完整性、设备状态或减少生成长度"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--model",
        default="v1_s03",
        choices=[m["id"] for m in MODEL_REGISTRY],
    )
    parser.add_argument("--speaker", default="cassie")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subtalker-temperature", type=float, default=0.8)
    parser.add_argument("--subtalker-top-p", type=float, default=0.9)
    parser.add_argument("--subtalker-top-k", type=int, default=30)
    parser.add_argument(
        "--subtalker-do-sample",
        dest="subtalker_dosample",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=-1, help="-1 = 每次随机；固定值可复现")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    engine = Engine(args)
    Handler.engine = engine
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print("=" * 66, flush=True)
    print(f"C.A.S.S.I.E. CustomVoice UI: {url}", flush=True)
    print(f"models: {[m['id'] for m in engine.models]}", flush=True)
    print(f"default: {engine.current_id} / speaker={args.speaker} / language=Auto", flush=True)
    print(f"sampling: {engine.sampling_defaults()}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    print("=" * 66, flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping...", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
