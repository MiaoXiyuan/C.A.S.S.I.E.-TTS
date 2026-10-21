# C.A.S.S.I.E. Voice LoRA · V1.0.0

Tips：个站文章https://www.miaoxiyuan.com/cassie-tts-voice-shaping
写文能力差，让Gemini润色结果用力过猛了…

C.A.S.S.I.E.（Central Autonomic Service System for Internal Emergencies，内部紧急情况中央自主服务系统）风格的 Qwen3-TTS LoRA，用于本地设施广播语音生成。公开包与推理 WebUI 的软件发布版本为 **V1.0.0**。模型版本为 **CustomVoice · C.A.S.S.I.E. v1 · s0.3**，服务中的模型 id 为 `v1_s03`。

本仓库仅用于推理，提供导出与推理代码、WebUI，以及 LoRA 配置和权重元数据。**最终 LoRA 权重和 CustomVoice 导出必需的 speaker embedding 通过 [GitHub Releases](https://github.com/MiaoXiyuan/C.A.S.S.I.E.-TTS/releases) 单独分发，不纳入 Git；克隆仓库或下载源码 ZIP 不包含这两份权重。** 使用者需要先下载两份 Release 权重与基座模型，再在本地导出一次 CustomVoice。

本项目**不包含训练、数据处理或评估工具，不分发基座模型、合并后的 CustomVoice 模型、训练音频／数据或游戏原始资源。**

代码、配置及项目原创文档采用 **Apache-2.0**；发布的 LoRA 权重与 speaker embedding 采用 **CC BY-SA 3.0 Unported**。具体范围和上游归属见本文末尾的[来源与许可](#来源与许可)。

## 目录结构

```text
.
├── lora/                         # 配置、元数据及下载权重的放置位置
│   ├── adapter/
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors  # 从 Releases 下载，不纳入 Git
│   ├── speaker_embedding.safetensors # 从 Releases 下载，不纳入 Git
│   ├── best_meta.json             # 权重元数据
│   └── README.md
├── scripts/
│   ├── check_env.py               # 环境检查
│   ├── download_models.py         # 下载推理所需 Base
│   ├── export_custom_voice.py     # Base + LoRA → CustomVoice
│   └── serve_custom_voice.py      # WebUI / HTTP API
├── src/
│   ├── __init__.py
│   ├── bootstrap.py
│   └── inference/
│       ├── __init__.py
│       └── f0.py                  # 可选输出 F0 后处理
├── web/index.html                # 中文单页 WebUI
├── LICENSE                       # Apache-2.0：代码及项目原创文档
├── LICENSE-WEIGHTS               # CC BY-SA 3.0：两份发布权重
├── NOTICE                        # 许可范围、来源与版权声明
├── licenses/                     # 上游许可原文
├── README.md
├── requirements.txt
└── .gitignore
```

运行时自行生成的 `models/`（下载的基座）、`outputs/`（导出模型和音频）、`test/`（本地验证产物）均已被 `.gitignore` 忽略，**不随公开包提交**。`data/`、`sources/`、`cache/` 也不发布。

## 快速开始

以下命令都在项目根目录执行。除虚拟环境激活步骤外，命令采用单行写法，可直接用于 Windows PowerShell 或 Bash。

### 1. 准备环境

- 已验证环境：Python 3.12、PyTorch 2.x + CUDA、NVIDIA GPU（16GB 显存）。
- 建议使用独立虚拟环境；依赖版本范围见 [requirements.txt](requirements.txt)。
- 导出时目标磁盘至少保留 **8GB 空闲空间**；下载基座模型还需另行预留空间。
- 默认设备为 `cuda:0`。CPU 可通过 `--device cpu` 尝试，尚未验证，速度和内存占用需要自行评估。

创建虚拟环境：

```text
python -m venv .venv
```

Windows PowerShell 激活：

```powershell
.venv/Scripts/Activate.ps1
```

Bash 激活（Linux/macOS）：

```bash
source .venv/bin/activate
```

安装依赖并检查环境：

```text
python -m pip install -r requirements.txt
python scripts/check_env.py
```

PyTorch 需与本机 CUDA 环境匹配。`pyworld` 用于可选 F0 后处理，在没有适用 wheel 的平台上可能需要本地编译。环境检查会输出本机诊断信息，请勿将原始输出作为公开示例提交。

### 2. 从 Releases 下载 LoRA 权重

进入本项目的 [GitHub Releases](https://github.com/MiaoXiyuan/C.A.S.S.I.E.-TTS/releases)，选择与代码版本对应的发布（当前为 **V1.0.0**），下载以下两个附件，并按下表放入项目目录。GitHub 自动生成的 Source code ZIP／tar.gz 不包含这些附件。

| Release 附件 | 本地放置路径（相对项目根目录） | V1.0.0 文件大小 |
|---|---|---|
| `adapter_model.safetensors` | `lora/adapter/adapter_model.safetensors` | 77,004,832 字节 |
| `speaker_embedding.safetensors` | `lora/speaker_embedding.safetensors` | 4,192 字节 |

**两份文件都必需。** speaker embedding 只有约 4KB，这是正常大小。保留仓库中的 `lora/adapter/adapter_config.json` 和 `lora/best_meta.json`，无需替换。两份权重已加入 `.gitignore`，下载后仅供本地导出使用。

### 3. 下载基座模型

```text
python scripts/download_models.py
```

脚本从 Hugging Face 下载推理所需的 Base，并记录下载 revision：

| 模型 | 本地目录 |
|---|---|
| [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | `models/Qwen3-TTS-12Hz-1.7B-Base/` |

Base 目录已包含导出与推理需要的 `speech_tokenizer/`，无需另行下载独立的 `Qwen3-TTS-Tokenizer-12Hz`。`--repo base` 保留兼容；脚本只支持下载 Base，**不会下载上述 LoRA 或 speaker embedding，二者需从 Releases 手工获取**。基座由使用者自行获取，不会打包进本项目。

### 4. 导出 CustomVoice S0.3

确认两份 Release 权重和 Base 均已下载到上述位置，再执行导出：

```text
python scripts/export_custom_voice.py --checkpoint lora --output outputs/cassie_custom_voice_s03 --base models/Qwen3-TTS-12Hz-1.7B-Base --speaker cassie --speaker-id 3000 --lora-scale 0.3 --device cuda:0
```

上述 checkpoint、output、base、speaker、speaker id、scale 和 device 也是脚本默认值，因此在默认目录布局下可直接运行：

```text
python scripts/export_custom_voice.py
```

导出会在 `outputs/cassie_custom_voice_s03/` 写入完整模型（`model.safetensors` 约 3.83GB）、tokenizer、speech tokenizer 和导出元数据。过程不修改 `lora/` 或基座目录。输出目录非空时默认拒绝覆盖；确认需要重新导出后才使用 `--overwrite`。

导出元数据 `custom_voice_meta.json` 会记录本机实际输入路径，仅供本地追踪，不作为公开文件提交。合并模型也只用于本地推理。

### 5. 启动 WebUI

```text
python scripts/serve_custom_voice.py
```

默认加载 `outputs/cassie_custom_voice_s03/`，自动打开 [http://127.0.0.1:7860](http://127.0.0.1:7860)。生成的 WAV 保存在 `outputs/web_ui/`。

指定端口并关闭自动打开浏览器：

```text
python scripts/serve_custom_voice.py --port 7861 --no-browser
```

### 6. 输出文件

以下路径均相对项目根目录：

| 内容 | 位置 |
|---|---|
| 推理生成的 WAV 音频 | `outputs/web_ui/` |
| 本地导出的 CustomVoice 模型 | `outputs/cassie_custom_voice_s03/` |
| 自行下载的 Base 模型 | `models/Qwen3-TTS-12Hz-1.7B-Base/` |

WebUI 与 HTTP API 生成的音频都保存在 `outputs/web_ui/`；网页下载的是同一份已保存的 WAV。刷新页面或清空页面历史不会删除磁盘音频，服务也不会自动清理这些文件，可按需自行删除。

`outputs/` 和 `models/` 均已忽略，不随公开包提交。

## 模型信息

| 项目 | 值 |
|---|---|
| 发布版本 | CustomVoice · C.A.S.S.I.E. v1 · s0.3 |
| 模型 id | `v1_s03`（唯一支持的模型） |
| 基座 | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
| LoRA | r=16，alpha=32，dropout=0.05 |
| target modules | `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj` |
| 最佳 epoch | 4 |
| 验证 loss | 3.2866 |
| 导出 scale | 0.3 |
| speaker | `cassie`，id 3000 |
| 音频 | 24kHz，单声道 WAV |

`lora/adapter/` 是 PEFT 格式的 LoRA 增量权重；`lora/speaker_embedding.safetensors` 是 2048 维说话人向量。导出 CustomVoice 时两者都需要。speaker id `3000` 是本项目的导出约定。

## WebUI 参数

页面仅提供单模型 CustomVoice，不需要参考音频。

主界面用于输入、生成、试听和下载，文本默认留空并实时显示字符数；“最大生成 Token”位于语言选择旁。其他参数设置默认折叠，按“采样”“风格指令”“音高处理”分组；“恢复默认”会同时关闭 Prompt 和输出 F0。

播放器每次显示选中的一条结果；“最近生成”保留最多 5 条，列表超过显示高度后独立滚动。页面预留纵向滚动条空间，展开参数设置时保持内容宽度稳定。

文本输入不能为空，本服务不设置固定字符上限，仍受模型上下文、硬件资源及 API 的 1 MB 请求体上限约束。字符数按去除首尾空白后的 Unicode 码点统计，包含中间的空格和标点。长文本可能生成更慢、占用更多显存，也可能出现韵律不稳、重复或漏读，建议按段生成。`max_new_tokens` 控制输出音频 codec token 的生成上限，不等于输入字数；上限不足时可能截断语音，可根据每段长度调整。

| 控制项 | 默认值 | 说明 |
|---|---|---|
| 模型 | `v1_s03` | 仅有最终 S0.3 |
| 语言 | `Auto` | 由模型判断；其他语言效果需自行验证 |
| Prompt / instruct | 关闭、空字符串 | 实验性风格指令；启用后才提交文本 |
| temperature / top_p / top_k | 0.5 / 0.8 / 30 | 主码本采样；更高随机性可能降低稳定性 |
| subtalker temperature / top_p / top_k | 0.8 / 0.9 / 30 | 后续子码本采样 |
| repetition_penalty | 1.05 | 重复惩罚 |
| 最大生成 Token / `max_new_tokens` | 768 | 主界面可调，范围 64–2048；生成长度上限，不足时可能截断 |
| seed | -1 | -1 每次随机；固定整数用于同环境复现 |
| 输出 F0 | 关闭，pitch=0 | 使用 WORLD 对生成结果做后处理 |

输出 F0 的 `pitch` 以半音为单位；`mode` 支持 `off`、`flat`（压平）、`limited`（围绕中位音高限幅）、`quantized`（音高量化）。`limit` 是限幅半径（默认 1 半音），`step` 是量化步长（默认 1.5 半音）。`off` 且 `pitch=0` 才完全不处理。

服务启动参数与页面采样项对应，参数名使用连字符，例如 `--temperature`、`--top-p`、`--top-k`、`--subtalker-temperature`、`--max-new-tokens`、`--seed`。其他常用参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--host` | `127.0.0.1` | 监听地址 |
| `--port` | `7860` | HTTP 端口 |
| `--model` | `v1_s03` | 仅支持此模型 id |
| `--device` | `cuda:0` | 推理设备 |
| `--speaker` | `cassie` | 导出模型中的 speaker 名称 |
| `--no-browser` | 未启用 | 不自动打开浏览器 |

完整参数见 `python scripts/serve_custom_voice.py --help`。

## HTTP API

### `GET /api/status`

返回引擎状态、唯一模型 `v1_s03` 的信息、可用语言和采样默认值。模型目录以项目相对路径返回；没有参考音频字段。

### `POST /api/generate`

请求体必须为 JSON 对象，大小不得超过 1 MB。`text` 是必填的非空字符串；其余字段可省略。默认模型为 `v1_s03`，语言为 `Auto`，`instruct` 为空，输出 F0 关闭。

```json
{
  "text": "Security alert. Chaos insurgency spotted at gate A.",
  "model": "v1_s03",
  "language": "Auto",
  "seed": 20261018,
  "instruct": "",
  "temperature": 0.5,
  "top_p": 0.8,
  "top_k": 30,
  "subtalker_temperature": 0.8,
  "subtalker_top_p": 0.9,
  "subtalker_top_k": 30,
  "output_f0": {"pitch": 0, "mode": "off", "limit": 1, "step": 1.5}
}
```

每次请求独立使用默认参数；省略 `instruct` 或 `output_f0` 不会继承上一请求的值。不支持 `reference` 或 `reference_f0`。

参数类型与范围：

| 字段 | 要求 |
|---|---|
| `text` | 字符串；去除首尾空白后不能为空，无固定字符上限 |
| `instruct` | 字符串；去除首尾空白后为 0–2000 字符 |
| `max_new_tokens` / `seed` | 整数；64–2048 / -1–2147483647 |
| `temperature` / `subtalker_temperature` | 数字，0.05–2 |
| `top_p` / `subtalker_top_p` | 数字，0.05–1 |
| `top_k` / `subtalker_top_k` | 整数，0–200 |
| `repetition_penalty` | 数字，1–2 |
| `do_sample` / `subtalker_dosample` | JSON 布尔值 `true` 或 `false` |
| `output_f0.pitch` / `limit` / `step` | 数字；-12–6 / 0–12 / 0.1–6，单位为半音 |

成功返回 `audio/wav`，并提供以下响应头：

| 响应头 | 含义 |
|---|---|
| `X-Duration` / `X-Elapsed` | 音频时长 / 生成耗时，单位为秒 |
| `X-Model` / `X-Seed` | 模型 id / 实际 seed |
| `X-Instruct-Length` | 本次风格指令长度 |
| `X-Output` | 保留 WAV 的项目相对路径 |
| `X-Audio-URL` | WAV 下载地址，例如 `/api/audio/<filename>.wav` |
| `X-Output-F0` | 本次输出 F0 参数的 JSON |

### `GET /api/audio/<filename>`

下载 `outputs/web_ui/` 中已生成并保留的 WAV。使用生成响应中的 `X-Audio-URL`，无需推断文件名。

下面的 Python 标准库示例适用于 PowerShell 和 Bash：将代码保存为 `test/api_example.py`，在项目根目录执行 `python test/api_example.py`。先创建 `test/` 目录；示例音频只写入该目录。

```python
import json
from pathlib import Path
from urllib.request import Request, urlopen

base_url = "http://127.0.0.1:7860"
output_dir = Path("test")
output_dir.mkdir(exist_ok=True)
payload = {"text": "All personnel. Please proceed to the evacuation shelter."}
request = Request(
    base_url + "/api/generate",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(request, timeout=300) as response:
    audio_url = response.headers["X-Audio-URL"]
    generated = response.read()
    (output_dir / "generated.wav").write_bytes(generated)
with urlopen(base_url + audio_url, timeout=30) as response:
    downloaded = response.read()
    (output_dir / "downloaded.wav").write_bytes(downloaded)
assert downloaded == generated
print("生成与下载成功：", len(downloaded), "bytes")
```

`POST /api/select` 作为兼容接口接受 `{"model":"v1_s03"}`。非法 JSON 或参数返回 HTTP 400，不存在的资源返回 404，内部生成错误返回 500；错误响应为 JSON，内部错误信息不包含本机路径。

## 已知限制

- 只支持最终 S0.3，不包含其他 scale、official adapter 或实验性合并模型。
- 公开 WebUI 没有 ICL 参考音频和参考处理流程。
- 当前模型未经过 instruction 微调，Prompt / instruct 效果不稳定，默认关闭。
- F0 塑形是生成后的音频处理，可能引入音质变化，默认关闭。
- 生成质量依赖基座版本和采样参数；固定 seed 仅用于相同环境下的复现。

## 来源与许可

本项目按文件内容分别授权：

| 内容 | 许可与范围 |
|---|---|
| 代码、配置、元数据及项目原创文档 | [Apache-2.0](LICENSE)，包括 `scripts/`、`src/`、`web/` 及 `lora/` 中的配置和说明；上游内容保留其原有版权和许可声明 |
| `web/index.html` 内嵌的 SCP 基金会标志图形 | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)，单独适用于该图形；不纳入代码的 Apache-2.0 授权 |
| Release 附件 `adapter_model.safetensors`（下载后放入 `lora/adapter/`） | [CC BY-SA 3.0 Unported](LICENSE-WEIGHTS) |
| Release 附件 `speaker_embedding.safetensors`（下载后放入 `lora/`） | [CC BY-SA 3.0 Unported](LICENSE-WEIGHTS) |
| 上游许可原文与归属声明 | 见 [NOTICE](NOTICE) 和 [licenses/](licenses/)，按各自上游条款保留 |

两份权重的 CC BY-SA 3.0 许可覆盖本项目有权许可的贡献。复用、修改或再分发时，应提供合理署名、许可链接及适用的修改说明；分享改编作品时遵守相同方式共享要求。该许可允许商业使用，本项目不附加“禁止商用”或“仅限研究”的条件。

本项目**不分发 Qwen3-TTS 基座、Tokenizer 权重、合并模型、训练音频／数据或游戏原始资源**。使用者自行下载的 Qwen 模型继续适用其上游 Apache-2.0 许可，本项目的权重许可不替代或修改该许可，也不重新许可上游游戏素材。生成音频应按其实际内容判断相关权利，不能仅因使用本模型便认定所有输出自动采用相同许可。

来源与归属：

- 训练所用音频来自 [Convex89524/C.A.S.S.I.E](https://github.com/Convex89524/C.A.S.S.I.E) 项目的 `data`；语句拼接工具及归档格式也来源于该项目。工具代码的 MIT 许可不等同于游戏录音的许可。
- 《SCP: Secret Laboratory》相关游戏及音频素材归属 Northwood Studios 和相关权利人；素材及配音归属说明见 [NOTICE](NOTICE)。
- SCP 社区背景与授权说明见 [SCP Wiki](https://scp-wiki.wikidot.com/) 和[社区许可指南](https://scp-wiki.wikidot.com/licensing-guide)。本项目主动按署名—相同方式共享原则发布两份权重。
- WebUI 页头与浏览器标签页图标使用 Dr_Grom 的 [`header-logo.svg`](https://github.com/Grombald/Common-Theme/blob/27f25355852ff8eaf497da8443a91844a10f6d7e/docs/svg/header-logo.svg)，来源为 SCP Wiki / Common-Theme，遵循 CC BY-SA 3.0。原图形状与颜色未修改，仅本地内嵌、添加无障碍标签并缩放显示；详见 [NOTICE](NOTICE)。
- 基座及推理实现来自 [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)；模型制作过程中的历史参考来源和保留的上游版权声明见 [NOTICE](NOTICE)。

本项目为非官方社区项目，不代表 Qwen、Northwood Studios、SCP Wiki 或上游工具作者的认可、合作或背书。
