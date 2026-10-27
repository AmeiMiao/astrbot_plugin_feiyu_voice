"""Generate clean, deterministic Colab notebooks and their helper integrity pins."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLAB = ROOT / "colab"
REPO = "Esmerubphire/astrbot_plugin_feiyu_voice"


def cell(kind, source, ident):
    result = dict(cell_type=kind, id=ident, metadata={}, source=source.strip() + "\n")
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(engine):
    rvc = engine == "rvc"
    title = "RVC v1 人声换色" if rvc else "CosyVoice3 v1.0.1 文字转语音"
    # GitHub serves LF blobs. Normalize before hashing even on Windows.
    for name in ("feiyu_colab.py", "releases.json"):
        path = COLAB / name
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    pins = {name: hashlib.sha256((COLAB / name).read_bytes()).hexdigest()
            for name in ("feiyu_colab.py", "releases.json")}
    bootstrap = f'''
#@title 1 · 检查 GPU，准备辅助脚本
import hashlib, importlib.util, pathlib, sys, urllib.request, urllib.error
ROOT = pathlib.Path("/content/feiyuvoice")
HELPERS = ROOT / "notebook_helpers"
HELPERS.mkdir(parents=True, exist_ok=True)
# Helper bytes are SHA-256 pinned. Inference source uses a separate immutable commit.
HELPER_REF = "main"
PINS = {pins!r}
for name, expected in PINS.items():
    target = HELPERS / name
    if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        url = f"https://raw.githubusercontent.com/{REPO}/{{HELPER_REF}}/colab/{{name}}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
        except urllib.error.URLError as exc:
            raise RuntimeError("辅助文件尚未发布或网络不可达；未推送的验收可按 colab/README.md 开发者步骤上传同版本辅助文件。") from exc
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError("辅助脚本版本与 Notebook 不匹配，请重新打开同版本 Notebook。")
        temporary = target.with_suffix(target.suffix + ".download")
        temporary.write_bytes(data)
        temporary.replace(target)
spec = importlib.util.spec_from_file_location("feiyu_colab", HELPERS / "feiyu_colab.py")
fv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fv)
fv.check_gpu()
'''
    cells = [cell("markdown", f'''
# 🐋 FeiyuVoice · {title}

这是**推理**入口，使用已发布的肥鱼声库。与另一份 Notebook 相互独立，不训练或修改权重，不启动公网服务，不需要挂载 Drive。

先在 **Runtime → Change runtime type → GPU** 选择可用 NVIDIA GPU，再按顺序执行。首次安装和下载需要一些时间；免费 GPU、具体型号、显存和会话时长均不保证。请及时下载结果，断开并删除运行时会丢失临时文件。

**授权和平台边界**：只处理你有权使用的文本/干净人声，不用于冒充真人或制造误导性内容。Google Colab [当前政策](https://research.google.com/colaboratory/faq.html)禁止 creating deepfakes；素材授权不自动等于平台批准。若你的用途受限，请停止并使用获准的本地环境，不绕过限制。原声素材作者为 [星落落_oi](https://space.bilibili.com/30584746/)，训练素材已获授权；[COPYRIGHT](https://github.com/{REPO}/blob/main/COPYRIGHT.md)、模型 NOTICE 和上游许可分别适用。

**验证状态：真实 Colab GPU 验收尚未完成。** 本地 Linux/CUDA 结果见仓库 [colab/VALIDATION.md](https://github.com/{REPO}/blob/main/colab/VALIDATION.md)，不能视为云端 PASS。
''', "intro"), cell("code", bootstrap, "bootstrap"),
        cell("code", f'''
#@title 2 · 安装独立环境（可重复运行）
cuda_build = "auto" #@param ["auto", "cu128", "cu130"]
# auto 优先按 Colab kernel 的 PyTorch CUDA build 选择；没有该信息时参考驱动能力。
fv.setup("{engine}", ROOT, cuda_build=cuda_build)
''', "setup"),
        cell("code", f'''
#@title 3 · 下载并校验正式发布模型
# 使用公开 Hugging Face 仓库，不需要 Token。TTS 全包约 6.94 GiB。
fv.worker("{engine}", ROOT, "download")
''', "models")]
    if rvc:
        cells += [cell("markdown", '''
## 上传干净人声并转换

支持 WAV / MP3 / FLAC，一次一个文件，0.5–600 秒、最大 128 MiB。建议先用 5–15 秒有授权的干净清唱/语音；伴奏不会在这里自动分离。原文件会保存到唯一上传目录，预处理另存单声道 WAV。

默认：RVC v2 / 40 kHz，RMVPE，Pitch 0，Index 0.3，Protect 0.33，Volume Envelope 1.0，Speaker 0。Pitch 单位为半音；Index 越高检索影响越大；Protect 保护清辅音/气声，0.5 表示关闭该保护。没有默认换气衰减或额外音效。
''', "input-help"), cell("code", '''
#@title 4 · 上传人声（选择一个文件）
from google.colab import files
import os, uuid
upload_dir = ROOT / "rvc/uploads" / uuid.uuid4().hex
upload_dir.mkdir(parents=True)
previous_dir = pathlib.Path.cwd()
try:
    os.chdir(upload_dir)
    uploaded = files.upload()
finally:
    os.chdir(previous_dir)
if len(uploaded) != 1:
    raise ValueError("请重新运行本单元格，只选择一个音频文件。")
source_audio = upload_dir / next(iter(uploaded))
source_audio = source_audio.resolve(strict=True)
if not source_audio.is_relative_to(upload_dir.resolve()):
    raise ValueError("上传路径异常。")
print("已上传：", source_audio.name, "；转换前会检查格式、时长并完整解码。")
del uploaded  # release kernel's extra in-memory copy
''', "upload"), cell("code", '''
#@title 5 · 设置参数并转换
pitch_shift = 0 #@param {type:"slider", min:-24, max:24, step:1}
index_rate = 0.3 #@param {type:"slider", min:0, max:1, step:0.01}
protect = 0.33 #@param {type:"slider", min:0, max:0.5, step:0.01}
fv.validate_parameters(pitch_shift, index_rate, protect)
result_wav = fv.submit("rvc", ROOT, input=str(source_audio), pitch_shift=pitch_shift,
                      index_rate=index_rate, protect=protect)
print("输出：", result_wav)
''', "infer")]
    else:
        cells += [cell("markdown", '''
## 选择声线并输入文本

**Cute（默认）**：F6 微调 LLM ＋发布包 N01 参考。**Gentle**：原始 C3 LLM ＋ N10 参考，并非另一份微调模型，也不是只换提示词。参考音频使用发布包内的合成参考，不上传未公开素材。

沿用发布参数：seed 42、FP32、speed 1.0、非流式、原有文本前端；原生 24 kHz 输出，另保存与发布版相同的峰值保护 PCM16 WAV。没有新增降噪、变速或音高处理。支持输入中文/英文，但未充分验证所有语言和长文本。每次生成在独立进程加载模型，切换声线不会残留上一份 LLM 权重。
''', "style-help"), cell("code", '''
#@title 4 · 选择声线，输入文本并生成
style = "Cute" #@param ["Cute", "Gentle"]
text = "你好呀，我是肥鱼。今天想听我说点什么？" #@param {type:"string"}
if not 1 <= len(text.strip()) <= 2000:
    raise ValueError("请输入 1–2000 字符；初次测试建议用一句短句。")
result_wav = fv.submit("tts", ROOT, style=style.lower(), text=text)
print("输出：", result_wav)
''', "infer")]
    cells += [cell("code", '''
#@title 试听结果
from IPython.display import Audio, display
display(Audio(filename=str(result_wav)))
print((result_wav.parent / "result.json").read_text(encoding="utf-8"))
''', "play"), cell("code", '''
#@title 下载 WAV（每次生成后及时保存）
from google.colab import files
files.download(str(result_wav))
''', "download"), cell("markdown", '''
## 遇到问题

- 无 GPU / CUDA kernel 失败：重新选择可用、兼容的 GPU；初始化会检查实际 CUDA 运算，不能仅靠 GPU 名称判断支持。
- 显存不足：使用更短的音频/文本，或切换显存更充足的兼容 GPU。本项目没有认证最低显存或所有 Colab GPU。
- 下载失败：检查 Hugging Face / GitHub / PyPI / PyTorch 的访问情况，然后重跑对应单元格；校验通过的模型会复用。完整初始化失败后可重跑安装；不会改动 Notebook kernel 的包。若修改了版本或源代码，使用新会话。
- 音频错误：RVC 检查上传格式、时长和解码；TTS 检查文本与所选声线。先运行最短样例，再增大输入。
- 推理日志保留明确异常；独立环境 `requirements-resolved.txt`、`diagnostic.json`、任务 `result.json` 可用于排查。分享前删除个人文本、路径或上传文件信息。

这里只提供交互式推理；没有训练、任意第三方声线、WebUI、隧道、分离伴奏或自动混音。
''', "troubleshooting")]
    nb = dict(cells=cells, nbformat=4, nbformat_minor=5,
              metadata=dict(accelerator="GPU", colab=dict(name=f"FeiyuVoice_{'RVC' if rvc else 'TTS'}.ipynb",
                                                          provenance=[], private_outputs=True),
                            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                            language_info=dict(name="python")))
    (COLAB / f"FeiyuVoice_{'RVC' if rvc else 'TTS'}.ipynb").write_bytes(
        (json.dumps(nb, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))


if __name__ == "__main__":
    for engine in ("rvc", "tts"):
        build(engine)
