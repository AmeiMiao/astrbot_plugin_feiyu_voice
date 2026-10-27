# 🐋 FeiyuVoice · Google Colab 推理

两份 Notebook 相互独立，只使用已经发布的模型，不需要 Google Drive、本机 Python、WSL 或 CUDA。

| Notebook | 打开 |
|---|---|
| [RVC 人声转换](FeiyuVoice_RVC.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Esmerubphire/astrbot_plugin_feiyu_voice/blob/main/colab/FeiyuVoice_RVC.ipynb) |
| [CosyVoice3 TTS](FeiyuVoice_TTS.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Esmerubphire/astrbot_plugin_feiyu_voice/blob/main/colab/FeiyuVoice_TTS.ipynb) |

**当前为待云端验收的实现。** 两份 Notebook 与同版本辅助文件一起随本仓库 `main` 分发，可使用上方入口打开。已有静态检查和本地 CUDA 推理证据，真实 Colab GPU 完整验收仍为 `NOT RUN`，详见 [验证记录](VALIDATION.md)。开发者测试尚未合并的更新时，可上传 `.ipynb` 并按下方步骤预放同版本辅助文件。

1. 打开 Notebook，选择 Runtime → Change runtime type → GPU。
2. 顺序运行 GPU 检查、独立环境安装和模型下载。
3. RVC 上传干净人声并设置参数；TTS 选择 Cute / Gentle 并输入短句。
4. 执行推理、试听，再运行下载单元格保存 WAV。

没有可用 NVIDIA GPU 时会提前停止。GPU 型号、显存、CUDA 状态在初始化时显示，随后检查独立环境实际 CUDA 运算。免费资源、具体 GPU 和会话寿命不保证；本项目未认证最低显存、T4、L4、A100 等每一种型号。首次 TTS 模型下载约 **6.94 GiB**，RVC 发布包约 **83 MiB**，此外还需 ContentVec / RMVPE、Python、PyTorch 与其他依赖；安装阶段预检 12 GiB 余量，模型下载阶段另按缺失文件体积加 2 GiB 预检，均是操作余量，不是严格峰值磁盘保证。使用后及时下载，运行时被回收后临时文件会丢失。

## RVC 的行为

使用 FeiyuVoice-RVC-v1（RVC v2 / 40 kHz）及匹配索引，RMVPE，Pitch 0，Index Rate 0.3，Protect 0.33，Volume Envelope 1.0，Speaker 0。表单仅开放 Pitch（-24…24 整数半音）、Index（0…1）、Protect（0…0.5）；0.5 关闭清辅音保护。

WAV 优先，也支持 MP3 / FLAC。上传大小上限 128 MiB，时长 0.5–600 秒；先用 5–15 秒有授权的干净人声试验。FFprobe 检查音轨、长度，FFmpeg 另存单声道浮点 WAV，采样率保留给 Applio 的既有重采样流程。原上传文件和既有结果不覆盖，每次任务有独立目录。没有默认人工换气衰减、变速、自动调音、伴奏分离或混音。

辅助层直接调用与本地版本相同的 `VoiceConverter.convert_audio`；不另写内容编码、F0、检索或合成算法。加载后校验实际网络 tensor digest，种子沿用发布版 seed 42＋分段序号。输出检查 40 kHz、单声道、有限值、音量和时长，并记录参数及 SHA-256。

## TTS 的行为

使用 FeiyuVoice v1.0.1 原有资源：

| 声线 | LLM | 参考 |
|---|---|---|
| Cute（默认） | F6 微调 `feiyu_cute_F6_0040.pt` | N01 |
| Gentle | 原始 C3 `llm.pt` | N10 |

Gentle 不是另一份微调模型。每次任务使用独立推理进程，按发布 `manifest.json` 选择真实 checkpoint / reference。调用公开服务仓库中已验证的 `feiyu_runtime.tts`，沿用原文本前端、prompt prefix、seed 42、speed 1.0、FP32、非流式及 CPU ONNX 前端；禁用 TensorRT / vLLM。输出 `native.wav`（FLOAT）和与原版相同峰值保护的 `speech.wav`（PCM16），均保持原生 24 kHz；Notebook 试听与下载 PCM16 文件。未新增音色处理。

输入 1–2000 字符，初次用中文短句；可输入英文，其他语言和长文本没有充分验证。每次加载模型有等待时间，不驻留模型或新增后台服务。

文本处理以已经验证的当前发布环境为准：调用原 `text_frontend=True` 路径，但保持未安装可选 WeText / ttsfrd 时的内建处理。通用服务预设中有 `wetext==0.0.4`；Colab 的有效依赖清单仅排除这一项，因为它会在初始化时另从 ModelScope 下载未固定的 TN 资源。这样与已验证本地发布环境一致，也避免推理时出现额外的未固定资源来源。没有自行编写替代的规范化算法；数字、日期、英文等读法仍需样例验收。若安装了额外前端，Notebook 会要求使用新的独立环境，避免静默改变行为。

## 版本固定与下载

完整机器可读记录见 [releases.json](releases.json)：

| 内容 | 固定版本 |
|---|---|
| 已发布服务端适配器、依赖预设及 Applio overlay | `7baff58fdb4c4472a392be94da6dfa9690d96a8b` |
| [RVC](https://huggingface.co/Esmerubphire/FeiyuVoice-RVC/tree/04e5011b72341a742ef17a718d5da0435343f893) | `04e5011b72341a742ef17a718d5da0435343f893` |
| [TTS](https://huggingface.co/Esmerubphire/FeiyuVoice-CosyVoice3/tree/f37e9d1dda26e8b45b4e9f1b4c347c0a07b797e2) | `f37e9d1dda26e8b45b4e9f1b4c347c0a07b797e2` |
| Applio | `67a38f6abab3e90a55efa534ea3185a222c102f0` ＋已发布且校验的 17 文件 overlay |
| CosyVoice | `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`，递归初始化固定 submodule |
| 独立 Python / uv | `3.11.17` / `0.13.5` |
| torch / torchaudio | `2.11.0+cu128`（复用当前 CUDA 预设） |
| torchcodec | `0.11.0+cpu` |
| HF Hub | RVC `1.16.1`；TTS `0.36.2`（与各自 Transformers 约束匹配） |
| pip / setuptools / wheel | `25.2` / `79.0.1` / `0.48.0`，避免最新 setuptools 移除旧组件需要的接口 |

RVC 的 `v1` 标签早于最新署名文档；这里固定包含最新 COPYRIGHT / NOTICE 的 commit，权重与 v1 标签相同。TTS 没有发布标签，固定已发布的 v1.0.1 commit。没有修改或重新发布模型。

`huggingface_hub.hf_hub_download(..., token=False)` 下载公开资源，不需要登录。先校验固定 `files_sha256.json` 自身的 SHA-256，再按其逐文件校验，已有且正确的文件复用，损坏文件明确重下。ContentVec / RMVPE 也用 Hub API，沿用公开 `runtime_compat/upstream_assets.json` 的 revision 与 SHA-256。训练集、未公开参考、私人实验素材均不使用。

Notebook 辅助文件从本仓 `main/colab/` 获取，并在执行前验证 Notebook 内固定的 SHA-256；上游推理代码另用不可变 commit。辅助文件更新后，必须运行 `python colab/build_notebooks.py` 同步 Notebook 内的 hash，并一起发布；不同版本混用会拒绝运行。维护更新时须同步发布 Notebook 和辅助文件，避免版本不匹配。

独立环境放在 `/content/feiyuvoice/{rvc,tts}/venv`，不升级内核或系统全部依赖。只安装缺失的系统包。安装分阶段可重复，成功后重跑不会重装；每次仍校验实际 CUDA kernel、引擎导入和源文件。完整环境解析结果保存在 `requirements-resolved.txt`，关键直接依赖沿用已发布版本锁；传递依赖不是跨时间完全固定的 lock，需要实测。若上游包下架、驱动不兼容、GitHub/HF/PyPI/PyTorch 不可达，会明确失败，不降级为 CPU 或更换未经验证的模型。

## 授权与 Colab 政策

原声素材来自 B 站 UP 主 [星落落_oi](https://space.bilibili.com/30584746/)，训练素材已获授权。见 [COPYRIGHT](../COPYRIGHT.md)、[NOTICE](../NOTICE.md) 和下载包内各许可证；上游代码 Apache-2.0 / MIT 不自动授予角色资产、声线、歌曲或原始素材的全部权利。本集成不更改 Hugging Face 模型的许可声明。

只处理有权使用的内容，不用于冒充或制造误导。Google Colab [官方 FAQ](https://research.google.com/colaboratory/faq.html) 当前列明禁止 creating deepfakes，且对非交互式服务有额外限制；授权不代表平台批准。请核对用途，受限时停止并使用获准的本地环境，不绕过限制。本实现使用 Notebook 原生表单、上传、试听和下载，无 Gradio、WebUI、FastAPI、公网端口或隧道。

## 开发与验证

`python -m pip install nbformat==5.10.4` 后运行 `python colab/test_colab.py`。修改辅助文件后先重新生成 Notebook，再执行测试。不要把音频、输出、凭据、完整执行日志或权重写回 `.ipynb` 或提交 Git。

**未推送前在真实 Colab 验收**：先上传 `.ipynb`，临时新增一个初始化单元格运行下面代码，只上传本目录同版本的 `feiyu_colab.py` 与 `releases.json`（不要上传模型、个人配置或音频）。然后正常运行 Notebook 的第 1 步；已有辅助文件仍会经过 Notebook 中固定的 SHA-256 检查，匹配后直接复用，不依赖未发布的 GitHub raw 地址。两份 Notebook 分别在新会话验收。该临时单元格不需要写回发布 Notebook。

```python
from google.colab import files
from pathlib import Path
helpers = Path("/content/feiyuvoice/notebook_helpers")
helpers.mkdir(parents=True, exist_ok=True)
for name, data in files.upload().items():
    if name not in {"feiyu_colab.py", "releases.json"}:
        raise ValueError("只上传这两个同版本辅助文件")
    (helpers / name).write_bytes(data)
```

真实云端验收必须分别从全新 Colab GPU 会话执行，Cute / Gentle 均需测试；本地通过不能记为云端 PASS。具体结果与待办见 [VALIDATION.md](VALIDATION.md)。
