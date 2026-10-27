# Colab 集成交付与验证记录

日期：2026-10-26。范围仅为现有正式模型推理，无训练、微调、权重修改、模型迁移或重新发布。

## 2026-10-27：Colab 反馈修复

用户反馈：Colab CUDA 13.0 环境手动调整 CUDA 配置并建立 `feiyu_runtime` symlink 后可使用。反馈未附完整环境/运行日志，不能据此把本次自动修复标记为完整云端 PASS。

- 新增 `auto / cu128 / cu130` 安装表单。auto 优先看 kernel 的 PyTorch CUDA runtime，缺失时再看驱动 CUDA 能力；只选择 `releases.json` 中固定的官方 2.11.0 wheel。包索引、torch、torchaudio、constraints、生成 profile 和就绪标记同步，不修改共享/本机 `cuda.json`。
- 安装和 RVC/TTS 推理入口显式加入 `publication/server` 路径，不依赖启动目录或外部 PYTHONPATH，不创建 symlink。子进程仍显式设置 PYTHONPATH。
- 新增 CUDA 12.8/13.0 自动选择、驱动信息回退、显式选择、错误版本、共享预设保护、安装/profile 一致性、禁止覆盖已选环境，以及无 PYTHONPATH、无 symlink、不同工作目录下的实际 CLI 子进程导入回归。
- 保留模型 revision、权重、索引、参考与推理算法。更新两个 Notebook 内的辅助文件 SHA；旧会话应先保存输出，再用新版 Notebook 与新 GPU 会话。

2026-10-26 的下列音频证据属于 cu128；不能外推为 cu130 的音质、速度或最低显存验证。真实 Colab GPU 自动修复路径仍待验收。CUDA 13.0 也不意味着所有 cu128 构建必然不兼容；驱动与 wheel runtime 的含义已在 README 区分。

本轮验证结果：18 项静态/回归测试全部通过；真实子进程覆盖 RVC 与 TTS 的模块导入（模型前主动停止的测试夹具）；PyTorch 官方源上 `torch==2.11.0+cu130`、`torchaudio==2.11.0+cu130` 的 CPython 3.11 Linux x86_64 wheel 均返回 HTTP 200，仅验证可获取性，未安装或运行 cu130。另从 `/tmp` 移除 PYTHONPATH、不建立 symlink，执行真实本地 cu128 RVC 推理成功：40 kHz、2.72 秒、WAV 有效，SHA-256 与上次同输入默认参数输出一致（`c83ddbbb8405a744f6a199e74e87a16e1242a310fddc7326abb28c80d1e9d4bb`）。完整新会话安装及 Colab GPU 自动修复验收仍为 NOT RUN。

## 实现与文件清单

| 文件 | 作用 |
|---|---|
| `colab/FeiyuVoice_RVC.ipynb` | 独立 RVC Notebook：GPU 检查、安装、下载、上传、参数表单、转换、试听、下载 |
| `colab/FeiyuVoice_TTS.ipynb` | 独立 TTS Notebook：GPU 检查、安装、下载、Cute / Gentle 表单、生成、试听、下载 |
| `colab/feiyu_colab.py` | 标准库引导及独立子进程编排；复用已发布 TTS 适配器和 Applio 转换入口 |
| `colab/releases.json` | 固定公开源码与模型 revision、校验清单的可信 SHA-256、Python / Hub / uv 版本 |
| `colab/build_notebooks.py` | 重建无输出 Notebook，同步辅助文件 SHA-256 |
| `colab/test_colab.py` | Notebook 格式、引导下载、缓存、参数、环境重复执行和输入保护测试 |
| `colab/README.md` | 普通用户运行方法、版本、授权及平台限制、维护方法 |
| `colab/VALIDATION.md` | 本记录：实际测试、未验证项及交付范围 |
| `README.md` | 现有公开仓库主页新增两个真实仓库路径的 Colab badge 和运行步骤 |
| `.gitattributes` | 固定 Colab 源文件 LF，避免跨系统换行改变辅助文件 SHA-256 |

研究工作区另同步维护 `docs/FEIYU_README.md`（公开主页源文稿）、研究主页入口，以及 `scripts/sync_voice_runtime_source.py` 中的 Colab 源码白名单复制。这些维护文件不需要打包进推理环境。本次没有同步之前未发布的桌面 GUI 改动，也没有更改现有服务、插件、模型包或推理协议。

RVC 保持 RMVPE、40 kHz、Speaker 0、seed 42、Volume Envelope 1.0，默认 Pitch 0 / Index 0.3 / Protect 0.33，可调整三项表单参数。无默认后处理。上传的原文件和已有输出不覆盖。TTS 按发布 manifest 选择 F6＋N01 或 C3＋N10，使用原适配器生成原生 24 kHz FLOAT 与 PCM16 播放文件。两份 Notebook 各自安装自己的环境，互不依赖，不启动服务。

## 仓库和发布配置核对

公开仓库为 `https://github.com/Esmerubphire/astrbot_plugin_feiyu_voice`，GitHub API 确认默认分支 `main`；现有本地发布检出基线为 `7baff58fdb4c4472a392be94da6dfa9690d96a8b`。外层研究仓库是无远程的 `master`，有大量既有改动，本次未对其进行提交或整体复制。

两个 HF 固定 revision 可匿名读取，RVC 23 个远端文件、TTS 32 个远端文件。固定 checksum 清单分别覆盖 21 / 31 个文件，另独立验证清单自身。TTS 包约 6.94 GiB；RVC 包约 83 MiB，不含上游通用模型。所有引用与摘要见 [releases.json](releases.json)。RVC `v1` 标签较早，因此使用包含已发布署名更新的固定 commit；其权重与 v1 一致。TTS 无标签，固定已发布 v1.0.1 的 commit。

下载保留 COPYRIGHT、NOTICE 和上游 LICENSE。测试资源使用公开发布包的合成参考；没有上传原始人声、训练集或私人实验素材。没有更改 HF 许可。

## 本地测试环境和方法

下面的“本地通过”**均不代表真实 Google Colab GPU 的 PASS**。

| 项目 | 实际环境 |
|---|---|
| 静态/单元检查 | Windows Python 3.11，nbformat 5.10.4 |
| 实际推理 | Ubuntu 24.04 / WSL2 Linux x86_64，Python 3.11.17 |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU，24,463 MiB，驱动 621.01 |
| torch / torchaudio | 2.11.0+cu128 / 2.11.0+cu128 |
| RVC Transformers / Hub | 5.13.1 / 1.16.1 |
| TTS Transformers / Hub / ONNX Runtime | 4.51.3 / 0.36.2 / 1.22.0，CPUExecutionProvider |
| 新 Python 引导 | uv 0.13.5，在新目录下载 CPython 3.11.17 并创建隔离环境，非复用原 venv |

模型测试夹具逐文件比对固定 HF checksum，已有大权重在私有测试目录建立硬链接复用，未重新下载约 7 GiB 大文件；只有缺少的小型公开元数据进行了真实网络下载。实际推理使用已有已验证模型环境，**没有把完整的新环境安装、全量权重下载称作本轮已验收**。

| 测试 | 本地结果与证据范围 |
|---|---|
| Notebook JSON / nbformat / Python cell 编译 | 本地通过；共 12 项自动化测试通过，两份合法，无执行输出、缓存音频、个人路径或凭据 |
| 参数、无 GPU、上传时长、覆盖保护、路径越界、磁盘不足 | 本地通过；自动化边界测试 |
| 引导辅助文件 SHA 校验与重复执行 | 本地通过；执行真实 bootstrap 逻辑，HTTP 来源用测试替身，不冒充线上已发布 |
| 安装完成标记的重复初始化 | 本地通过；重跑不重装、不 reset；仍调用诊断。中断阶段按代码可重入设计，完整云端重跑未测 |
| HF 匿名公开资源与损坏修复 | 本地通过；两个 Hub 版本均用 `token=False` 真实修复一份测试包署名文档副本，权重和校验正确的文件复用 |
| 固定 GitHub 源码获取 | 本地通过；新目录浅检出固定已发布 commit |
| uv / 独立 Python 创建 | 本地通过；新工具目录、新 Python 目录与新 venv；没有覆盖已有模型环境 |
| RVC 默认参数 | 本地通过；40 kHz、单声道、2.72 秒，有限值、非静音、时长和实际网络 tensor digest 均检查 |
| RVC 调整参数与 Notebook submit 链路 | 本地通过；Pitch +2 / Index 0.2 / Protect 0.4，40 kHz、2.72 秒，输出 SHA 与默认不同 |
| Cute 中文 | 本地通过；24 kHz、4.36 秒，F6 / N01 摘要与发布 manifest 一致 |
| Gentle 中文 | 首次本地 FAIL（ONNX 解析），文件 SHA 正确；串行重试本地通过，24 kHz、7.60 秒，C3 / N10 摘要一致 |
| Cute 英文 | 本地通过；24 kHz、4.64 秒，经 Notebook submit→独立子进程链路生成；不等于所有语言已验证 |
| 播放/下载控件 | 静态确认使用 IPython Audio 和 Colab files.download；真实浏览器交互未执行 |
| 与本地版听感对比 | NOT RUN；没有把采样率/波形检查当作听感验收 |

Gentle 首次失败为 ONNX Runtime `INVALID_PROTOBUF`，当时 tokenizer 文件 SHA-256 为发布值 `23236a74175dbdda47afc66dbadd5bcb41303c467a57c261cb8539ad9db9208d`；未改文件，后续串行重试成功。根因未确定，不声称已修复所有资源或加载问题。没有增加隐藏重试或更改模型。

英文首轮测试曾与故意损坏测试元数据的缓存修复检查重叠，被完整性检查正确拒绝（COPYRIGHT.md SHA 不符）；恢复完成后串行重跑成功。这是测试编排冲突，不记为已发现的模型缺陷。所有自动化测试最终通过。

TTS 本机实际 `actual_text_frontend` 为空，表示原环境没有 WeText / ttsfrd、使用 CosyVoice 内建处理；两个 ONNX 会话实际使用 CPUExecutionProvider。通用服务预设的 WeText 0.0.4 会额外从 ModelScope 下载未固定 TN 资源，故 Colab 仅排除该可选项，保持已验证发布行为；仍调用原 `text_frontend=True` 代码。安装额外前端时明确拒绝，不静默变换读法。完整新环境的解析与真实云端文本行为仍需验收。

## 真实 Google Colab GPU 验收

状态含义：PASS = 已在真实 Colab GPU 执行成功；FAIL = 已在对应环境执行失败；BLOCKED = 明确外部条件阻止；NOT RUN = 尚未执行。下面没有任何云端 PASS。

| 验收项目 | 状态 | 原因 |
|---|---|---|
| 全新 RVC Colab GPU 会话安装与下载 | NOT RUN | 本轮没有已连接的真实 Colab GPU 会话 |
| 上传、转换、内嵌播放、下载 WAV | NOT RUN | 尚未在 Colab 执行 |
| 另一全新 TTS Colab GPU 会话安装与下载 | NOT RUN | 尚未在 Colab 执行 |
| Cute 中文、Gentle 中文及参考路由 | NOT RUN | 已有本地证据，仍需云端重验 |
| 可选英文、音质对比 | NOT RUN | 尚未在 Colab 执行 |
| 不同 GPU 的显存、驱动兼容 | NOT RUN | 本机 5090 不等于 T4 / L4 / A100 验证 |

## 限制、Git 状态和实质性待办

免费 GPU、VRAM 和会话时长无保证；固定 CUDA 12.8 wheel 需要兼容驱动和 GPU。未承诺最低显存或特定 Colab 型号。安装和模型下载依赖 GitHub、Astral Python 构建、PyPI、PyTorch wheel 源和 HF；任一不可达可能阻止初始化。顶层依赖固定，传递依赖保存本次 freeze，不等于跨时间完全锁定。TTS 初始化和全包校验较慢，每次生成重新加载模型。

Colab [官方 FAQ](https://research.google.com/colaboratory/faq.html) 明确禁止 creating deepfakes；当前用途仍需符合平台政策和相关授权，素材授权不等于平台批准。没有绕过限制的措施，也没有公网服务、隧道或网页替代 Notebook 交互。

开发分支：`codex/colab-inference`，基于原 `main`。经用户确认，本次将 `.gitattributes`、`README.md` 与 `colab/` 共 10 个文件作为独立提交快进推送到现有仓库 `main`，使 README badge 和匹配的 raw helper 可一并访问。未打 tag、创建 Release 或修改模型仓库。GitHub 提交 diff 可用于审查；发布入口可访问不等于真实 Colab GPU 验收通过。

剩余实质性工作：在允许该用途的实际 Colab GPU 上分别完成两个全新会话验收（含全部安装、全量下载、两声线、上传/播放/下载、听感及不同 GPU 资源表现），调查若再现的 ONNX 加载失败。本次发布保留上述未验证状态，不扩展训练、WebUI 或 Spaces。
