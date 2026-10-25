# C.A.S.S.I.E. Voice LoRA · V1.5.0

基于 Qwen3-TTS 的 C.A.S.S.I.E. 风格语音 LoRA，提供本地 CustomVoice 导出、中文 WebUI 和 HTTP API。本次软件更新配套使用试听选定的 v3 模型。

- **公开包 / WebUI 版本：V1.5.0**。
- **模型版本：CustomVoice · C.A.S.S.I.E. v3 · 第4轮 · s0.8**，模型 id 为 `v3_e04_s08`。
- 本项目仅用于推理，不包含训练、数据处理或评估工具。

软件版本和模型代际分别记录。WebUI 状态栏简写为“C.A.S.S.I.E. TTS 模型版本: v3”，实际使用的 checkpoint、scale 和附件校验信息仍以本说明及 `checkpoint_meta.json` 为准。

## 下载附件

打开本项目的 [GitHub Releases](https://github.com/MiaoXiyuan/C.A.S.S.I.E.-TTS/releases)，选择注明 **V1.5.0 / `v3_e04_s08`** 的发布，下载以下两个附件，放入源码目录中的对应位置。**两份文件都必需。**

| 附件 | 文件大小 | 放置路径（相对项目根目录） |
|---|---|---|
| `adapter_model.safetensors` | 77,004,832 字节 | `lora/adapter/adapter_model.safetensors` |
| `speaker_embedding.safetensors` | 4,192 字节 | `lora/speaker_embedding.safetensors` |

speaker embedding 只有约 4KB，这是正常大小。保留本次源码仓库中的 `lora/adapter/adapter_config.json` 和 `lora/checkpoint_meta.json`；元数据已从旧版的 `best_meta.json` 改为 `checkpoint_meta.json`。

GitHub 自动生成的 **Source code ZIP / tar.gz** 和 `git clone` **不包含这两份权重**。`scripts/download_models.py` 只下载 Qwen Base，不会下载 Release 权重附件。

附件保持原始、未经预缩放的 LoRA 和本 checkpoint 的 speaker embedding。新旧 adapter 文件大小恰好相同，**不能只检查文件大小，必须核对文末 SHA-256**；旧 v1／S0.3 附件不能与本次配置混用。

## 快速使用

获取与本次 Release 匹配的源码，按[根目录 README](https://github.com/MiaoXiyuan/C.A.S.S.I.E.-TTS/blob/main/README.md) 准备 Python / PyTorch / CUDA 环境，放好上述两份权重后，在**项目根目录**执行：

```text
python -m pip install -r requirements.txt
python scripts/download_models.py
python scripts/export_custom_voice.py
python scripts/serve_custom_voice.py
```

基座为 `Qwen/Qwen3-TTS-12Hz-1.7B-Base`，需由使用者自行下载。默认以 scale **0.8** 导出到 `outputs/cassie_custom_voice_v3_e04_s08/`，随后启动 WebUI：**http://127.0.0.1:7860**。PyCharm 无参数运行导出器和服务脚本，也使用本次模型与目录。

导出器读取 `lora/checkpoint_meta.json`，校验两份原始权重的 SHA-256，加载原始 Base、adapter 和该 checkpoint 的 speaker embedding，设置 scale 0.8 后合并。**scale 只在导出时应用一次；推理直接加载合并后的 CustomVoice，不再挂载 adapter，也不再次乘 0.8。** 导出器要求命令行 scale 与本次元数据一致，避免错误标注模型 id。

导出不会修改原始权重或基座。生成的 WAV 保存在 `outputs/web_ui/`；页面提供试听和下载，刷新页面或清空历史不会删除磁盘音频。Base、合并模型、缓存和验证音频均不纳入 Git。

## 模型与选型说明

- 来源为 v3 的 `checkpoint-epoch-3`，即 **`epoch_index=3`、`epoch_number=4`**；不是 `checkpoint-epoch-4`。
- 第4轮验证 loss 为 **2.5563672079759487**。选择依据是实际试听首选，兼顾目标声音与词间连贯性。
- 本次选择**不是全程验证 best**；v3 全程最低出现在第9轮，约为 2.4813。不同版本的验证集不同，不将 loss 横比为音质提升百分比。
- LoRA 配置为 **r=16、alpha=32、dropout=0.05**，使用 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj` 七类投影。
- speaker 为 `cassie`，id 为 `3000`；该编号是本项目的导出约定。

研究进展与本次更新详情见 [CHANGELOG](https://github.com/MiaoXiyuan/C.A.S.S.I.E.-TTS/blob/main/CHANGELOG.md)。维护者本地附件准备目录为已忽略的 `releases/v3_e04_s08/`，该目录不是运行依赖。

## 使用说明与限制

- 仅提供最终 **v3、第4轮、scale 0.8** 模型，不包含其他 epoch、scale 或合并后的完整模型。
- 默认语言为 **English**，保留其他语言选择；效果需自行验证。
- WebUI 可调采样参数和 MaxTokens；MaxTokens 默认 **512**，范围 **64–2048**，上限不足时可能截断音频。
- 种子默认 **`-1`，每次随机**；可设置固定整数用于同环境复现。模型验收使用的 `20261025` 是固定验收条件，不是日常默认值。
- 文本没有固定字数上限，仍受模型上下文、硬件资源及 API 的 **1 MB 请求体上限**约束。长文本可能增加耗时和显存占用，或出现语调不稳、重复、漏读，建议分段生成。
- Prompt / instruct 效果可能不稳定，默认关闭；输出 F0 属于音频后处理，也默认关闭。
- 不分发基座模型、合并模型、训练配置、训练音频 / 数据、游戏原始资源或 `training_state.pt`。

## 许可与来源

- **两份权重：CC BY-SA 3.0 Unported**，见 [LICENSE-WEIGHTS](https://github.com/MiaoXiyuan/C.A.S.S.I.E.-TTS/blob/main/LICENSE-WEIGHTS)。复用或再分发时请保留合理署名、许可及适用的修改说明，并遵守相同方式共享要求。
- **项目原创代码、配置、元数据和文档：Apache-2.0**，见 [LICENSE](https://github.com/MiaoXiyuan/C.A.S.S.I.E.-TTS/blob/main/LICENSE)。WebUI 内嵌的 SCP 标志图形单独遵循 CC BY-SA 3.0，所用版本作者为 Dr_Grom。

权重署名：**CustomVoice · C.A.S.S.I.E. v3 · 第4轮 · s0.8 — C.A.S.S.I.E. Voice LoRA contributors，CC BY-SA 3.0**。基于 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 微调，训练音频来源为 [Convex89524/C.A.S.S.I.E](https://github.com/Convex89524/C.A.S.S.I.E) 项目提供的录音归档；相关游戏录音归属 Northwood Studios 和相关权利人，SCP 社区背景归属 SCP Wiki 及其作者。

该权重许可覆盖本项目有权许可的贡献，允许商业使用，不附加“禁止商用”或“仅限研究”条件；它不重新许可 Qwen 基座或上游游戏素材。完整来源、署名、许可范围及非官方项目声明见 [NOTICE](https://github.com/MiaoXiyuan/C.A.S.S.I.E.-TTS/blob/main/NOTICE)。本项目不代表上述组织或作者的认可、合作或背书。

## SHA-256

```text
60C6A0B40097B03E0C5B4639F88541747D112482BA948D791580DF022AE0EB33  adapter_model.safetensors
36FFE123AF88C5D6BC5C3B6A09EF28A666020227440035A1D94486A49AA21DBF  speaker_embedding.safetensors
```
