# Upstream license copies

本目录保存上游许可证原文及历史来源记录；本项目的适用范围见根目录 [NOTICE](../NOTICE)。代码仓库提供推理、导出、WebUI 相关代码及配置、元数据，不包含训练、数据处理或评估工具，也不包含两份 `.safetensors` 权重。当前模型为 **`v3_e04_s08`（v3、第4轮、scale 0.8）**，软件版本为 **V1.5.0**。权重通过注明对应模型 id 的 [GitHub Release](https://github.com/MiaoXiyuan/C.A.S.S.I.E.-TTS/releases) 单独分发，首次使用需手工下载并校验 SHA-256、放回指定路径，见 [lora/README.md](../lora/README.md#使用)。保留这些许可副本用于说明权重的来源，不表示对应训练实现或第三方工具随包分发。

| 文件 | 上游及原文来源 | 范围 |
|---|---|---|
| [Qwen3-TTS-Apache-2.0.txt](Qwen3-TTS-Apache-2.0.txt) | [Qwen3-TTS，参考版本 022e286b](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/LICENSE) | 基座／推理实现及历史训练来源的许可与原有声明 |
| [Instavar-Apache-2.0.txt](Instavar-Apache-2.0.txt) | [Instavar 社区训练补丁](https://github.com/instavar/qwen3-tts-lora-finetuning/blob/main/LICENSE) | 权重训练所参考的社区修正；相关训练代码不随包发布 |
| [CASSIE-MIT.txt](CASSIE-MIT.txt) | [Convex89524/C.A.S.S.I.E 工具](https://github.com/Convex89524/C.A.S.S.I.E/blob/master/LICENSE.txt) | 上游工具代码的许可记录；不适用于游戏录音 |

本项目代码的 Apache-2.0 全文为 [LICENSE](../LICENSE)，由 [Apache 官方原文](https://www.apache.org/licenses/LICENSE-2.0.txt)取得。

本项目通过 GitHub Release 分发的 `adapter_model.safetensors` 和 `speaker_embedding.safetensors` 仍采用 CC BY-SA 3.0 Unported，全文为 [LICENSE-WEIGHTS](../LICENSE-WEIGHTS)，由 [Creative Commons 官方原文](https://creativecommons.org/licenses/by-sa/3.0/legalcode.txt)取得。它仅覆盖本项目有权许可的权重贡献，不替代任何上游素材许可；代码、配置及元数据的 Apache-2.0 许可范围不变。

许可原文按下载内容保留，包括上游版权行和年份。这里没有包含第三方工具程序、依赖包、基座模型或录音资源。

## WebUI 标志

`web/index.html` 本地内嵌的 SCP 基金会标志用于页头和浏览器标签页图标（favicon），来自 [SCP Wiki](https://scp-wiki.wikidot.com/) 使用的 [`header-logo.svg`](https://github.com/Grombald/Common-Theme/blob/27f25355852ff8eaf497da8443a91844a10f6d7e/docs/svg/header-logo.svg)，该版本作者为 **Dr_Grom**，许可为 [CC BY-SA 3.0 Unported](https://creativecommons.org/licenses/by-sa/3.0/)。图形形状和颜色未修改，仅添加无障碍标签并缩放显示。

此许可单独覆盖内嵌标志图形，图形不在本项目 Apache-2.0 代码授权范围内；独立的项目代码仍按 Apache-2.0 授权。完整署名和来源见 [NOTICE](../NOTICE)。
