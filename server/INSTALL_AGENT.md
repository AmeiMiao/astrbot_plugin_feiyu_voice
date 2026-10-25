# 给 AI agent 的安装说明：CUDA 与 WSL ROCm

服务端支持显式选择 CUDA 或 ROCm 推理环境。两者使用同一份肥鱼模型权重和 RVC 索引，AstrBot 的 HTTP 请求不变。AMD 路径目前没有实体显卡验证，应作为可安装、可诊断的实验配置交付；不得把安装成功写成音频或硬件验收通过。

以下命令以公开仓库的 `server/` 为工作目录；本地开发仓库可在项目根目录执行。所有 `models/`、`configs/` 路径均相对于该目录，不相对于配置文件所在目录。

## 1. 选择后端

先在 Windows PowerShell 获取 GPU 名称和系统信息：

```powershell
Get-CimInstance Win32_VideoController | Select-Object Name, PNPDeviceID
wsl --list --verbose
wsl -d Ubuntu-24.04 -- uname -a
```

| 检测结果 | 选择 |
|---|---|
| 有可用 NVIDIA 独显 | `cuda`，先在 WSL 确认 `nvidia-smi` 可用 |
| 只有 AMD，且该型号/驱动/WSL 组合在 AMD 官方支持范围内 | `rocm`，装对应 Windows WSL 驱动和 ROCm 运行时 |
| 混合显卡 | 默认用已可用的 NVIDIA；用户要求 AMD 时明确选择 ROCm |
| AMD 未列入支持组合、型号未知或 GPU 在 WSL 不可见 | 说明缺少哪项兼容条件；不能宣称 GPU 服务已装好 |

ROCm 采用 PyTorch 的 `cuda:0` 设备名，实际后端用 `torch.version.hip` 判别。不要把代码中的所有 `cuda` 替换成 `hip`，不要在同一个虚拟环境混装两个后端。参考 [AMD 当前兼容矩阵](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) 和 [PyTorch HIP 说明](https://docs.pytorch.org/docs/main/notes/hip.html)。

仓库提供固定的依赖预设：

| 预设 | RVC / TTS | 可选 SVS | 初始精度/前端 |
|---|---|---|---|
| `runtime_profiles/cuda.json` | torch/torchaudio 2.11.0+cu128 | 2.8.0+cu128 | RVC/TTS FP32，SVS FP16，ONNX CPU |
| `runtime_profiles/rocm.json` | torch/torchaudio 2.11.0+rocm7.2 | 2.11.0+rocm7.2 | FP32，RMVPE CPU，ONNX CPU |

ROCm 预设的 wheel 来自 PyTorch 官方 `rocm7.2` 索引；wheel 存在不等于每张 Radeon 都获支持。先核对目标硬件与所选版本的官方条件。需要另一 ROCm 版本时新增独立预设、明确版本并重新验证，不静默改动这些预设。AMD FP32 SVS 显存和速度尚无实测，显存不足时先缩短分段，任何 FP16 调整都应记录条件。

## 2. 安装独立推理环境

在 WSL Ubuntu 安装基础组件（Ubuntu 镜像中包名可能略有差异）：

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg libsndfile1 libsox-dev build-essential python3-dev
```

使用独立 Python 3.11。Ubuntu 24.04 默认 Python 不一定是 3.11，可以通过已安装的 uv 执行 `uv python install 3.11.17`，再用 `uv python find 3.11.17` 获取路径。不要替换系统 Python。以下命令都在 WSL 内、仓库 `server/` 目录执行：

将推理环境与上游源码放在 WSL 本地文件系统，例如 `$HOME/.local/share/feiyu-voice/`。避免把几套大型 Python 环境安装在 `/mnt/c/` 等 Windows 挂载盘上；本轮空目录安装测试实际观察到该位置的文件写入显著变慢。API 代码和模型包可以保留在 Windows 项目目录，通过 WSL 路径访问。

```bash
feiyu_python="$(uv python find 3.11.17)"
feiyu_backend=cuda  # AMD 改为 rocm
feiyu_env_root="$HOME/.local/share/feiyu-voice/$feiyu_backend"

"$feiyu_python" scripts/setup_runtime.py --backend "$feiyu_backend" --engine rvc \
  --python "$feiyu_python" --destination "$feiyu_env_root/rvc"
"$feiyu_python" scripts/setup_runtime.py --backend "$feiyu_backend" --engine tts \
  --python "$feiyu_python" --destination "$feiyu_env_root/tts"
```

只需要换声就只安装 RVC；只需要说话就只安装 TTS。安装器克隆固定 commit、复制 RVC 兼容源码、下载并校验通用 ContentVec/RMVPE 资产、安装选定后端、运行 `pip check`，输出依赖快照、源码 SHA 清单和诊断 JSON。每个引擎一个环境；现有目录会被拒绝，避免覆盖正在使用的安装。失败日志和目录保留排查，修复后用新的目标目录重试。

`--dry-run` 可先打印依赖安装计划。诊断包括实际 GPU 矩阵运算、卷积、FFT 和推理入口导入；它不会下载肥鱼权重，也不会替代实际音频验证。

请保留仓库 `.gitattributes`。RVC 兼容源码与模型包绑定的是原始字节 SHA，包括换行符；不要让编辑器或格式化器批量改写 `runtime_compat/Applio/`。安装器会先检查兼容源码 SHA，再复制到独立环境。

若复用现有兼容环境，可使用 `scripts/register_runtime.py --backend cuda --engine rvc --repo /path/to/Applio --python /path/to/env/bin/python --output /path/to/new/runtime.local.json`。这只读取环境、生成配置并运行诊断，不修改环境；版本和源码 commit 不匹配时应建立新环境。自建 Python 若需要额外共享库路径，在生成配置中填写 `python_library_dir`，并让安装/诊断命令使用同一个 `LD_LIBRARY_PATH`。

## 3. 下载模型、连接配置

下载以下仓库到服务端 `models/` 中，保持包内文件结构：

- TTS → `models/FeiyuVoice-CosyVoice3`：[Hugging Face](https://huggingface.co/Esmerubphire/FeiyuVoice-CosyVoice3) / [ModelScope](https://modelscope.cn/models/Esmerubphire/FeiyuVoice-CosyVoice3)。
- RVC → `models/FeiyuVoice-RVC`：[Hugging Face](https://huggingface.co/Esmerubphire/FeiyuVoice-RVC) / [ModelScope](https://modelscope.cn/models/Esmerubphire/FeiyuVoice-RVC)。

将安装器输出的两个 `runtime.local.json` 复制到服务端 `configs/runtime.rvc.cuda.local.json`、`configs/runtime.tts.cuda.local.json`；AMD 使用 `.rocm.local.json` 文件名。文件内容中的 Python 和源码路径必须是实际 WSL 路径，保留安装器生成的版本、commit、源码 SHA。若环境安装在 WSL ext4 中，复制到 Windows 项目时用 `/mnt/c/.../server/configs/` 目标路径。

把 `configs/voice_service.cuda.example.json`（或 `.rocm.example.json`）复制为 `configs/voice_service.json`。关键字段如下：

```json
{
  "engine": "cosyvoice3_release",
  "package": "models/FeiyuVoice-CosyVoice3",
  "runtime_profile": "configs/runtime.tts.rocm.local.json"
}
```

RVC 对应 `engine: feiyu_rvc_release` 和 RVC 的 profile。未安装的能力从服务配置中移除或置空对象。`runtime_profile` 指定时，采用本仓库的独立适配器，无需修改模型包的 `infer.py`、`run_rvc.py`、权重或包内校验文件，也不需要向包内写 `runtime.local.json`。没有该字段的旧配置继续使用包内启动器。

如果更改本地环境配置或源码，重新生成清单并重新验证，然后重载服务；不要手填虚假 SHA 来绕过校验。

## 4. 安装 API 与 AstrBot

API 是 CPU 环境，与三个推理环境分开。在 Windows 的 `server/` 目录：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-service.txt
.\scripts\Start-VoiceService.ps1 -Config .\configs\voice_service.json
```

也可以在 Linux/WSL 创建独立 `.venv` 并运行 `python -m feiyu_voice_service --config configs/voice_service.json`。服务在 WSL 内时，包体路径也必须是 Linux 可访问路径。

使用服务生成的本地 API Key 配置 AstrBot 插件，默认地址 `http://127.0.0.1:9880`。遵循现有服务说明的鉴权与监听地址设置。不要把 API Key、用户上传内容或机器路径提交到 GitHub。

`GET /v1/capabilities` 中 `available` 仍表示包体/配置可用；新增 `runtime_backend` 和 `runtime_verification` 表示选择的后端及配置检查层级。实际推理结果记录本次 GPU、后端和精度。现有上传、说话、换声、唱歌、任务查询和下载请求格式不变。

## 5. 可选 MIDI/ABC 唱歌

额外安装 SVS：

```bash
"$feiyu_python" scripts/setup_runtime.py --backend "$feiyu_backend" --engine svs \
  --python "$feiyu_python" --destination "$feiyu_env_root/svs"
```

将 `configs/singing.cuda.example.json`（或 `.rocm.example.json`）复制为 `configs/singing.json`，下载指定 SVS 权重并提供自有已对齐 prompt WAV/JSON，填写其真实 SHA256。将 SVS profile 复制到 `configs/runtime.svs.cuda.local.json`（或 `.rocm.local.json`）；示例已设置 `svs_runtime_profile`，该字段优先于旧 `svs_runtime`。独立命令运行唱歌时，`rvc_runtime_profile` 指向选定 RVC profile；通过 API 调用时，服务会自动继承当前声音的 RVC profile。

开启 `voice_service.json` 的 `singing` 配置。SVS 与 RVC 分进程串行运行，不同时常驻显存。谱子和歌词解析在 CPU。精度来自 SVS profile，不用修改 HTTP 请求。用户未提供对齐参考时，只交付可用的 TTS/RVC，不把唱歌显示为可用。

## 6. 安装验收与故障处理

服务启动后可用 API 环境运行实际验收脚本（只验说话时省略 `--vocal`）：

```powershell
.venv\Scripts\python.exe scripts\validate_runtime_service.py --tts `
  --key-file .runtime\voice-service\api_key.txt --vocal C:\path\to\clean-vocal.wav `
  --output-dir .runtime\acceptance-new-install
```

可选唱歌再添加 `--score examples/singing/feiyu_demo.abc --lyrics "晨风轻轻吹过山岗云光落在我的身旁"`。输出目录须是新目录，验收结果不包含 API Key。

1. 保存各环境的 `diagnostic.json`；应为 success，后端与预期一致。GPU 不可用或装错 wheel 会报错，不自动退化成全 CPU 服务。
2. 通过原 HTTP 接口实际生成至少一条可爱 TTS、一条温柔 TTS和一条带索引的 RVC WAV；确认下载、时长、采样率、有限非静音，并试听。可选 SVS 再测试一份短谱子和连音/休止。
3. 记录 `worker_result.json` / 引擎 receipt；RVC 还应检查 `f0_devices`，确认 CPU 前处理是预设选择。不能把“GPU 名称可见”当作推理已在 GPU 上运行。
4. 相同权重跨 GPU 的 WAV 不要求逐字节一致。权重/索引身份必须一致，音质、音高和长音频稳定性要单独比较。
5. AMD 尚无实卡验收记录。完成某台机器测试后报告准确显卡、驱动、WSL、torch/ROCm 版本及通过项目，避免宣传“所有 A 卡兼容”。

常见问题：ONNX 报 NVIDIA 库缺失时检查是否误装 `onnxruntime-gpu` 或误用旧包内 TTS 启动器；新适配器明确使用 CPU ONNX。torch 被依赖替换时检查约束文件和 `pip check`，不要混装后端。缺失 `libpython` 时配置 Python 共享库路径；缺失 FFmpeg/音频库时补齐系统组件。AMD 算子问题先定位实际阶段，RMVPE 默认 CPU，主网络失败必须如实报错并保留日志，不自动换模型。
