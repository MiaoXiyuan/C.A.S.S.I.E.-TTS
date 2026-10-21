# LoRA adapter

本目录用于 **CustomVoice · C.A.S.S.I.E. v1 · s0.3**，对应服务模型 id `v1_s03`。代码仓库保留适配器配置、权重元数据和本说明；两份 `.safetensors` 权重通过对应 **V1.0.0** 的 GitHub Release 单独分发，不包含在克隆或源码下载中。

## 文件

以下路径相对本目录：

| 文件 | 来源 | 作用 |
|---|---|---|
| `adapter/adapter_config.json` | 代码仓库 | PEFT LoRA 配置；基座模型字段使用公开模型 id |
| `adapter/adapter_model.safetensors` | 手工下载 Release 资产后放入 | LoRA 增量权重，约 77MB |
| `speaker_embedding.safetensors` | 手工下载 Release 资产后放入 | 2048 维说话人向量，导出 CustomVoice 时写入 `codec_embedding[3000]` |
| `best_meta.json` | 代码仓库 | 权重元数据，记录最佳 epoch 4、val loss 3.2866 |

本目录仅用于保存推理所需权重、适配器配置和权重元数据，**不包含**基座模型、合并后的完整模型、训练配置、训练音频／数据、游戏原始资源或 `training_state.pt`。训练、数据处理与评估代码不随项目发布。

## 许可

以下两份通过 GitHub Release 分发的权重采用 **CC BY-SA 3.0 Unported**，完整条款见 [LICENSE-WEIGHTS](../LICENSE-WEIGHTS)；所列路径为下载后在本目录中的放置位置：

- `adapter/adapter_model.safetensors`
- `speaker_embedding.safetensors`

该许可覆盖本项目有权许可的权重贡献。复用、修改及再分发时，应保留合理署名、许可链接和适用的修改说明，并对分享的改编作品遵守相同方式共享要求。许可允许商业使用，不附加“禁止商用”或“仅限研究”的限制。

本目录中的配置、元数据和项目原创说明采用 [Apache-2.0](../LICENSE)，并保留相关上游声明。两份权重的许可不重新许可 Qwen 基座或上游游戏素材。基座由使用者自行下载，继续遵守其上游许可；训练音频与游戏原始资源不随包分发。

训练音频来源为 [Convex89524/C.A.S.S.I.E](https://github.com/Convex89524/C.A.S.S.I.E) 项目的 `data`，相关素材归属 Northwood Studios 和相关权利人。完整来源、版权及非官方项目声明见 [NOTICE](../NOTICE) 和[根目录 README](../README.md#来源与许可)。

## 使用

首次使用时，打开本项目的 [GitHub Releases](https://github.com/MiaoXiyuan/C.A.S.S.I.E.-TTS/releases)，从对应 **V1.0.0** 的 Release 手工下载以下两份资产。**两份都必须下载**，并按项目根目录的相对路径放置：

| Release 资产文件名 | 本地目标路径 |
|---|---|
| `adapter_model.safetensors` | `lora/adapter/adapter_model.safetensors` |
| `speaker_embedding.safetensors` | `lora/speaker_embedding.safetensors` |

保留代码仓库提供的 `adapter/adapter_config.json` 和 `best_meta.json`。然后按根目录 README 下载基座模型；`scripts/download_models.py` **仅下载 Qwen Base，不下载这两份 Release 权重**。

文件就位后，在项目根目录执行，以下单行命令可直接用于 PowerShell 或 Bash：

```text
python scripts/export_custom_voice.py --checkpoint lora --output outputs/cassie_custom_voice_s03 --base models/Qwen3-TTS-12Hz-1.7B-Base --speaker cassie --speaker-id 3000 --lora-scale 0.3 --device cuda:0
```

导出后的目录可以独立加载，不需要在推理时再挂载 LoRA。导出不会修改本目录或基座；合并模型和导出元数据只保留在已忽略的 `outputs/` 中，不随公开包提交。

## 说明

- LoRA 配置为 r=16、alpha=32、dropout=0.05；
- target modules：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`；
- `--lora-scale 0.3` 是发布时确认的听感档位；其他 scale 不属于本次发布内容；
- speaker id `3000` 是导出协议中的编号，不是基座模型的官方 speaker。
- 基座模型由使用者自行下载，训练音频／数据不发布；代码与权重按上述范围分别授权。
