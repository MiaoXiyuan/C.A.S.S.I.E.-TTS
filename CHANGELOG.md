# 更新记录

软件发布版本与模型代际分别记录。当前软件 / WebUI 版本为 **V1.5.0**；模型从旧 v1／S0.3 升级到 `v3_e04_s08`，不把“v3”当作软件版本号。

## 未发布：软件 V1.5.0 / 模型 `v3_e04_s08` · 2026-10-25

### 最终选择

- 模型：**C.A.S.S.I.E. v3，第4轮，LoRA scale 0.8**。
- 来源 checkpoint：`cassie_lora_v3/checkpoint-epoch-3`，`epoch_index=3`、`epoch_number=4`；不是 `checkpoint-epoch-4`。
- 所选轮次验证 loss：**2.5563672079759487**。
- 选择依据：实际试听首选，兼顾目标声音与词间连贯性。
- 本选择不是全程验证 loss 最低的 checkpoint。v3 全程最低出现在第9轮，约为 **2.4813**；研究侧 `best` 对应第9轮，不能替代本次选择。第10轮×0.6仅是备选，不作为公开包默认模型。

不同模型版本使用的验证集不同，不能将 loss 直接横比，也不将其换算为音质提升百分比。

### 研究侧进展摘要

1. 修复了子码本 loss 的标签二次移位。后15个码本的 logits 已对应 c1…c15，通用 causal loss 又移位一次标签；现在直接对齐计算交叉熵，训练、验证和逐轮评估共用修复。
2. v2 重构词频并训练10轮，目标音色和广播语调的试听表现改善，但完整句仍有逐词朗读感。之后从 v2 第5轮进行官方整句续训，实际训练10轮，仍未达到目标。
3. v3 重新清理、核对原始音频和转写，取消前后缀碎片拼词，仅保留完整词和原生短语。语义句不额外插入词间静音；词表保留80ms间隔，并在文本中加入逗号；同时调整源音频间的响度关系。
4. v3 合成训练3000条，混入15条官方完整句各20次；最终 train / val / test 为 **3300 / 204 / 200**。
5. v3 从原始 Qwen Base 新建 LoRA，按 English 训练10轮，每轮独立保存。模型架构未变：**r=16、alpha=32、dropout=0.05**，使用 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj` 七类投影。

以上仅说明权重来源与模型选型。本公开包不包含研究侧的训练、数据处理、评估工具，或训练数据、原始音频。

### 公开推理包同步

- 当前唯一模型 id 为 `v3_e04_s08`，默认本地导出目录改为 `outputs/cassie_custom_voice_v3_e04_s08/`。导出器、服务端、WebUI 和无参数运行使用同一模型。
- 采用本次脱敏的 `lora/checkpoint_meta.json`，记录来源、正确轮次、scale 与选择依据，不沿用旧 v1 的 `best_meta.json`。
- Release 附件保持原始、未经预缩放的 adapter 和本 checkpoint 的 speaker embedding。导出时在原始 Base 上设置 scale 0.8，再合并一次；合并后的 CustomVoice 推理不再叠加 adapter 或重复应用 scale。
- 默认语言为 **English**，其他语言选项保留；Prompt 和输出 F0 默认关闭。
- 默认参数为 temperature **0.5**、top_p **0.8**、top_k **30**、repetition_penalty **1.05**；subtalker temperature **0.8**、top_p **0.9**、top_k **30**。
- 日常生成默认 **seed=-1**（每次随机）、**max_new_tokens=512**，仍允许在 WebUI / API 调整；可设置固定 seed 复现同环境结果。
- 本次模型验收实际使用 **seed=20261025、max_new_tokens=512**。该固定种子仅记录验收条件，不作为日常生成默认值。
- WebUI 状态栏简化为“C.A.S.S.I.E. TTS 模型版本: v3”；精确模型 id、轮次和 scale 仍保留在模型信息与来源元数据中。
- 继续采用源码 / 权重分离发布；新本地附件准备在已忽略的 `releases/v3_e04_s08/`，旧发布产物独立保留。基座、合并模型、运行缓存和验证音频不纳入 Git。
- 代码、配置、元数据和项目原创文档仍按 Apache-2.0 授权；两份权重仍按 CC BY-SA 3.0 Unported 授权；SCP 标志与上游声明保留原有许可范围。

### 新模型附件校验

文件名与下载后的放置路径见 [LoRA 说明](lora/README.md)。adapter 与旧模型大小相同，应使用 SHA-256 区分：

```text
60C6A0B40097B03E0C5B4639F88541747D112482BA948D791580DF022AE0EB33  adapter_model.safetensors
36FFE123AF88C5D6BC5C3B6A09EF28A666020227440035A1D94486A49AA21DBF  speaker_embedding.safetensors
```

## 初始公开包：软件 V1.0.0 / 模型 v1 · S0.3

初始公开包提供最终 v1／scale 0.3 权重、CustomVoice 导出、中文 WebUI 和 HTTP API，采用源码与 Release 权重附件分离发布。此条仅保留历史说明，不是当前默认模型或当前附件的下载指引。
