"""Small Colab orchestration layer; inference stays in the published engines.

Only this module's standard-library functions run in the notebook kernel.
Model imports and inference run in each engine's isolated Python subprocess.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE / "releases.json").read_text(encoding="utf-8"))
GIB = 1024**3


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(command, **kwargs):
    return subprocess.run([str(x) for x in command], check=True, **kwargs)


def safe_path(root, name):
    target = (Path(root) / name).resolve()
    if not target.is_relative_to(Path(root).resolve()) or target == Path(root).resolve():
        raise ValueError("Unsafe package path: " + name)
    return target


def need_disk(root, required):
    free = shutil.disk_usage(root).free
    if free < required:
        raise RuntimeError(f"磁盘空间不足：需要约 {required / GIB:.1f} GiB，当前可用 {free / GIB:.1f} GiB。")
    print(f"磁盘可用 {free / GIB:.1f} GiB；本步骤预留 {required / GIB:.1f} GiB。")


def check_gpu():
    print("Notebook kernel:", sys.version.split()[0], platform.platform())
    try:
        import torch
        print("Kernel PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda,
              "CUDA available:", torch.cuda.is_available())
    except ImportError:
        print("Kernel 未安装 PyTorch；稍后使用独立环境，不修改 kernel。")
    try:
        info = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"
        ], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("没有可用 NVIDIA GPU。请先在 Colab 的 Runtime / Change runtime type 中选择 GPU。") from exc
    if not info:
        raise RuntimeError("NVIDIA GPU 列表为空，请切换到可用 GPU 会话。")
    print("GPU / total VRAM / free VRAM / driver:\n" + info)
    print("具体 GPU、驱动和显存是否兼容，还需以下独立环境的 CUDA kernel 检查与实际合成确认。")
    return info


def clone_pinned(url, commit, target, submodules=False):
    target = Path(target)
    if not (target / ".git").exists():
        target.mkdir(parents=True, exist_ok=True)
        run(["git", "init", target], stdout=subprocess.DEVNULL)
        run(["git", "-C", target, "remote", "add", "origin", url])
    actual = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"], capture_output=True, text=True)
    if actual.returncode or actual.stdout.strip() != commit:
        # Never reset a checkout with a different existing HEAD.
        if actual.returncode == 0:
            raise RuntimeError("Source checkout has a different HEAD: " + str(target))
        run(["git", "-C", target, "fetch", "--depth", "1", "origin", commit])
        run(["git", "-C", target, "-c", "core.autocrlf=false", "checkout", "--detach", "FETCH_HEAD"])
    if submodules:
        run(["git", "-C", target, "submodule", "update", "--init", "--recursive", "--depth", "1"])


def runtime_env(profile, server):
    env = dict(os.environ, PYTHONPATH=str(server), PYTHONDONTWRITEBYTECODE="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    if profile.get("python_library_dir"):
        env["LD_LIBRARY_PATH"] = profile["python_library_dir"] + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def inference_requirements(engine, source, destination):
    text = Path(source).read_text(encoding="utf-8")
    if engine == "tts":
        # The released, audio-validated environment has neither WeText nor ttsfrd.
        # WeText 0.0.4 otherwise fetches an unpinned ModelScope TN model on init.
        text = "\n".join(line for line in text.splitlines() if not line.startswith("wetext==")) + "\n"
    Path(destination).write_text(text, encoding="utf-8")
    return Path(destination)


def setup(engine, root):
    """Restartable stages, with the existing installer pins/profile/diagnostics."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("此安装入口仅用于 Colab/Linux x86_64。")
    check_gpu()
    state = root / engine
    state.mkdir(exist_ok=True)
    ready = state / "ready.json"
    if not ready.exists():
        need_disk(root, 12 * GIB)  # environment build reserve, separate from model download
        # Only missing OS packages are installed; no apt upgrade or kernel pip upgrade.
        required = ["git", "ffmpeg", "libsndfile1", "libsox-dev", "libportaudio2", "build-essential"]
        missing = [p for p in required if subprocess.run(
            ["dpkg-query", "-W", "-f=${Status}", p], capture_output=True, text=True
        ).stdout.strip() != "install ok installed"]
        if missing:
            run(["apt-get", "update", "-qq"])
            run(["apt-get", "install", "-y", "--no-install-recommends", *missing])
        tools = root / "tools"
        tools.mkdir(exist_ok=True)
        tool_env = dict(os.environ, PYTHONPATH=str(tools), UV_PYTHON_INSTALL_DIR=str(root / "python"),
                        UV_NO_PROGRESS="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
        if not (tools / "uv").exists():
            run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--target", tools,
                 "uv==" + LOCK["uv"]])
        python = state / "venv/bin/python"
        if not python.exists():
            run([sys.executable, "-m", "uv", "venv", "--seed", "--managed-python", "--python",
                 LOCK["python"], state / "venv"], env=tool_env)
        publication = root / "publication"
        clone_pinned(LOCK["source_repo"], LOCK["source_commit"], publication)
        server = publication / "server"
        sys.path.insert(0, str(server / "scripts"))
        from setup_runtime import SOURCES, build_profile
        preset = read(server / "runtime_profiles/cuda.json")
        stack = preset["engines"][engine]
        source = state / "source"
        clone_pinned(*SOURCES[engine], source, submodules=(engine == "tts"))
        if engine == "rvc":
            overlay = server / "runtime_compat/Applio"
            for name, digest in read(server / "runtime_compat/source_sha256.json").items():
                if sha(safe_path(overlay, name)) != digest:
                    raise ValueError("Applio overlay checksum mismatch: " + name)
            shutil.copytree(overlay, source, dirs_exist_ok=True)
        constraints = state / "torch-constraints.txt"
        constraints.write_text(f"torch=={stack['torch']}\ntorchaudio=={stack['torchaudio']}\n")
        pip_env = dict(os.environ, PIP_NO_CACHE_DIR="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
        run([python, "-m", "pip", "install", "pip==25.2", "setuptools==79.0.1", "wheel==0.48.0"], env=pip_env)
        run([python, "-m", "pip", "install", "--index-url", preset["torch_index"],
             "torch==" + stack["torch"], "torchaudio==" + stack["torchaudio"]], env=pip_env)
        requirements = inference_requirements(engine, server / f"runtime_profiles/requirements-{engine}.txt",
                                              state / "requirements-colab.txt")
        run([python, "-m", "pip", "install", "-c", constraints, "-r", requirements], env=pip_env)
        run([python, "-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cpu",
             "torchcodec==" + stack["torchcodec"], "-c", constraints], env=pip_env)
        # Explicit Hub pin; public files are always downloaded with token=False.
        run([python, "-m", "pip", "install", "huggingface_hub==" + LOCK[engine]["hub_version"]], env=pip_env)
        run([python, "-m", "pip", "check"])
        with (state / "requirements-resolved.txt").open("w") as stream:
            run([python, "-m", "pip", "freeze"], stdout=stream)
        profile = build_profile(preset, stack, engine, python, source, "Google-Colab/Linux")
        write(state / "runtime.local.json", profile)
        write(ready, {"lock_sha256": sha(HERE / "releases.json")})
    elif read(ready)["lock_sha256"] != sha(HERE / "releases.json"):
        raise RuntimeError("配置版本已经变化，请在新的 Colab 会话运行；保留现有结果。")
    server = root / "publication/server"
    profile_path = state / "runtime.local.json"
    profile = read(profile_path)
    env = runtime_env(profile, server)
    run([profile["python"], server / "scripts/diagnose_runtime.py", "--profile", profile_path,
         "--output", state / "diagnostic.json"], env=env)
    print(json.dumps(read(state / "diagnostic.json"), ensure_ascii=False, indent=2))
    return state


def download(engine, root):
    from huggingface_hub import hf_hub_download
    root = Path(root).resolve()
    package = root / engine / "package"
    package.mkdir(parents=True, exist_ok=True)
    release = LOCK[engine]

    def fetch(name, digest):
        target = safe_path(package, name)
        if target.is_file() and sha(target) == digest:
            return target
        # A corrupt local file is explicitly redownloaded, never trusted as a cache hit.
        downloaded = Path(hf_hub_download(release["repo"], name, revision=release["revision"],
                          local_dir=package, token=False, force_download=target.exists()))
        if sha(downloaded) != digest:
            raise ValueError("Hugging Face checksum mismatch: " + name)
        return downloaded

    checksum = fetch("files_sha256.json", release["checksums_sha256"])
    records = read(checksum)
    missing_bytes = sum(r["bytes"] for n, r in records.items()
                        if not safe_path(package, n).is_file() or sha(safe_path(package, n)) != r["sha256"])
    need_disk(root, missing_bytes + 2 * GIB)
    for name, record in records.items():
        print("检查/下载", name, flush=True)
        fetch(name, record["sha256"])
    if sha(package / "manifest.json") != release["manifest_sha256"]:
        raise ValueError("Release manifest mismatch")
    if engine == "rvc":
        source = Path(read(root / engine / "runtime.local.json")["repo"])
        assets = read(root / "publication/server/runtime_compat/upstream_assets.json")
        for asset in assets:
            # Use the official Hub API for pinned public ContentVec/RMVPE too.
            tail = asset["url"].split("huggingface.co/", 1)[1]
            owner, repo, _, revision, name = tail.split("/", 4)
            target = safe_path(source, asset["destination"])
            if target.is_file() and sha(target) == asset["sha256"]:
                continue
            cached = hf_hub_download(owner + "/" + repo, name, revision=revision, token=False)
            if sha(cached) != asset["sha256"]:
                raise ValueError("Upstream asset checksum mismatch: " + name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, target)
    print("公开发布包校验完成：", release["repo"], release["revision"])


def worker(engine, root, action, request=None):
    root = Path(root).resolve()
    profile = read(root / engine / "runtime.local.json")
    command = [profile["python"], HERE / "feiyu_colab.py", action, "--engine", engine, "--root", root]
    if request is not None:
        command += ["--request", request]
    run(command, env=runtime_env(profile, root / "publication/server"))


def validate_parameters(pitch, index_rate, protect):
    if isinstance(pitch, bool) or int(pitch) != pitch or not -24 <= pitch <= 24:
        raise ValueError("Pitch Shift 必须是 -24 到 24 的整数半音。")
    if not math.isfinite(index_rate) or not 0 <= index_rate <= 1:
        raise ValueError("Index Rate 必须在 0 到 1 之间。")
    if not math.isfinite(protect) or not 0 <= protect <= 0.5:
        raise ValueError("Protect 必须在 0 到 0.5 之间。")


def prepare_audio(source, destination):
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if source.suffix.lower() not in (".wav", ".mp3", ".flac"):
        raise ValueError("请上传 WAV、MP3 或 FLAC。")
    if source.stat().st_size > 128 * 1024**2:
        raise ValueError("输入文件超过 128 MiB；请先截取较短片段。")
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
        "stream=codec_type:format=duration", "-of", "json", str(source)
    ], text=True))
    duration = float(probe.get("format", {}).get("duration", "nan"))
    if not probe.get("streams") or not math.isfinite(duration) or not 0.5 <= duration <= 600:
        raise ValueError("音频必须可解码且时长为 0.5–600 秒；建议先用 5–15 秒干净人声。")
    if destination.exists() or source == destination:
        raise ValueError("预处理文件已存在；请使用新的任务目录。")
    # Decode only: keep sample rate, mix to mono; upstream RVC does its normal resampling.
    run(["ffmpeg", "-v", "error", "-nostdin", "-n", "-i", source, "-map", "0:a:0",
         "-ac", "1", "-c:a", "pcm_f32le", destination])
    return duration


def audio_info(path, rate):
    import numpy as np
    import soundfile as sf
    wave, actual = sf.read(path)
    if actual != rate or wave.ndim != 1 or len(wave) < rate / 10 or not np.isfinite(wave).all() or abs(wave).max() < 1e-7:
        raise ValueError("输出音频无效（采样率、声道、有限值或音量检查失败）。")
    return dict(sample_rate=actual, seconds=len(wave) / actual, sha256=sha(path))


def convert(root, request):
    """Call the same Applio VoiceConverter, with release identity/seed checks."""
    root = Path(root).resolve()
    profile = read(root / "rvc/runtime.local.json")
    package = root / "rvc/package"
    from feiyu_runtime.common import prepare, verify_audio_library, verify_package, verify_repo
    manifest = verify_package(package)
    source = verify_repo(profile)
    for name, digest in read(package / "runtime_pins.json").items():
        if sha(safe_path(source, name)) != digest:
            raise ValueError("Applio package source pin mismatch: " + name)
    pitch, index, protect = request["pitch_shift"], request["index_rate"], request["protect"]
    validate_parameters(pitch, index, protect)
    out = Path(request["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    decoded = out / "decoded_input.wav"
    duration = prepare_audio(request["input"], decoded)
    settings = out / "settings.json"
    write(settings, profile["settings"])
    os.environ.update(APPLIO_CONFIG_FILE=str(settings), APPLIO_LOGS_DIR=str(out / "logs"),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4")
    os.chdir(source)
    sys.path.insert(0, str(source))
    import random
    import numpy as np
    import torch
    from rvc.infer.infer import VoiceConverter
    from rvc.lib.algorithm.synthesizers import Synthesizer
    info = prepare(torch, profile)
    verify_audio_library(profile)
    converter = VoiceConverter()
    model, index_path = package / manifest["model"]["pth"], package / manifest["model"]["index"]
    converter.get_vc(str(model), manifest["speaker_id"])
    if next(converter.net_g.parameters()).device.type != "cuda":
        raise RuntimeError("RVC 主网络没有加载到 CUDA GPU。")
    digest = hashlib.sha256()
    for key, tensor in sorted(converter.net_g.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    if digest.hexdigest() != manifest["model_tensor_digest"]:
        raise ValueError("Loaded network differs from released FeiyuVoice-RVC v1")
    calls = []
    original = Synthesizer.infer

    def seeded(self, *args, **kwargs):
        seed = manifest["seed"] + len(calls)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        calls.append(seed)
        return original(self, *args, **kwargs)

    Synthesizer.infer = seeded
    output = out / "feiyu_rvc.wav"
    try:
        converter.convert_audio(str(decoded), str(output), str(model), str(index_path), pitch=int(pitch),
            f0_method=manifest["inference"]["f0_method"], index_rate=index,
            volume_envelope=manifest["inference"]["volume_envelope"], protect=protect,
            sid=manifest["speaker_id"], embedder_model="contentvec", export_format="WAV")
    finally:
        Synthesizer.infer = original
    result = audio_info(output, 40000)
    if not calls or abs(result["seconds"] - duration) > 0.2:
        raise ValueError("RVC 未执行合成或输出时长异常。")
    write(out / "result.json", dict(status="success", **result, runtime=info,
        model_sha256=sha(model), index_sha256=sha(index_path), model_tensor_digest=digest.hexdigest(),
        seed_calls=calls, pitch_shift=pitch, index_rate=index, protect=protect,
        speaker_id=0, f0_method="rmvpe", volume_envelope=1, breath_attenuation_applied=False))


def infer_tts(root, request_path):
    root = Path(root).resolve()
    request = read(request_path)
    if request.get("style") not in ("cute", "gentle") or not 1 <= len(request.get("text", "").strip()) <= 2000:
        raise ValueError("请选择 Cute/Gentle，并输入 1–2000 字符文本。")
    if any(importlib.util.find_spec(name) is not None for name in ("wetext", "ttsfrd")):
        raise RuntimeError("此 Notebook 固定使用原发布环境的内建文本处理。请使用新的独立环境，不安装额外 WeText/ttsfrd 资源。")
    from feiyu_runtime.tts import main
    main(root / "tts/package", read(root / "tts/runtime.local.json"), ["--request", str(request_path)])
    audio_info(Path(request["output_dir"]) / "speech.wav", 24000)


def submit(engine, root, **parameters):
    root = Path(root).resolve()
    jobs = root / engine / "outputs"
    jobs.mkdir(parents=True, exist_ok=True)
    job = uuid.uuid4().hex
    request = jobs / (job + ".json")
    out = jobs / job
    write(request, dict(output_dir=str(out), **parameters))
    worker(engine, root, engine, request)
    return out / ("speech.wav" if engine == "tts" else "feiyu_rvc.wav")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("download", "rvc", "tts"))
    parser.add_argument("--engine", choices=("rvc", "tts"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "download":
            download(args.engine, args.root)
        elif args.action == "rvc" and args.engine == "rvc":
            convert(args.root, read(args.request))
        elif args.action == "tts" and args.engine == "tts":
            infer_tts(args.root, args.request.resolve())
        else:
            raise ValueError("Action/engine mismatch")
    except Exception as exc:
        if "out of memory" in str(exc).lower():
            print("GPU 显存不足。请缩短输入或改用显存更充足的兼容 GPU；本项目未认证低显存配置。", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
