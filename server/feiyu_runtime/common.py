"""Runtime validation shared by isolated GPU workers; no torch import in the API."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_path(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Asset path escapes its package: " + name)
    return path


def verify_package(package):
    package = Path(package).resolve()
    for name, record in load(package / "files_sha256.json").items():
        path = checked_path(package, name)
        if sha(path) != record["sha256"]:
            raise ValueError("Package integrity check failed: " + name)
    return load(package / "manifest.json")


def validate_profile(profile):
    if profile.get("schema") != "feiyu-runtime-v1":
        raise ValueError("Unsupported runtime profile schema")
    if profile.get("backend") not in ("cuda", "hip", "xpu", "cpu"):
        raise ValueError("Runtime backend must be cuda, hip, xpu or cpu")
    if profile.get("host", "wsl") not in ("wsl", "windows"):
        raise ValueError("Runtime host must be wsl or windows")
    if profile.get("engine") not in ("rvc", "tts", "svs"):
        raise ValueError("Runtime engine must be rvc, tts or svs")
    for key in ("python", "repo", "commit", "torch_version", "torchaudio_version"):
        if not profile.get(key):
            raise ValueError("Runtime profile missing " + key)
    if not profile.get("source_sha256"):
        raise ValueError("Runtime needs the source manifest produced by setup_runtime.py")
    if profile.get("precision") not in ("float32", "float16"):
        raise ValueError("Unsupported runtime precision")
    if profile["backend"] == "xpu" and profile["precision"] != "float32":
        raise ValueError("XPU profiles run FP32 inference")
    if profile["engine"] in ("tts", "rvc") and profile["precision"] != "float32":
        raise ValueError("TTS/RVC release adapters preserve FP32 inference")
    if profile.get("f0_device", "cpu") not in ("cpu", "model"):
        raise ValueError("f0_device must be cpu or model")
    if profile.get("onnx_provider", "CPUExecutionProvider") != "CPUExecutionProvider":
        raise ValueError("These profiles use the explicit CPU ONNX frontend")
    return profile


def verify_repo(profile):
    validate_profile(profile)
    repo = Path(profile["repo"]).resolve()
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != profile["commit"]:
        raise ValueError("Runtime source commit changed")
    for name, digest in profile["source_sha256"].items():
        if sha(checked_path(repo, name)) != digest:
            raise ValueError("Runtime source changed: " + name)
    return repo


def backend_name(torch):
    if getattr(torch.version, "hip", None):
        return "hip"
    if getattr(torch.version, "cuda", None):
        return "cuda"
    if getattr(torch.version, "xpu", None):
        return "xpu"
    return "cpu"


def apply_backend_env(profile):
    """Set backend environment before torch is imported (it is read at library load)."""
    if profile.get("backend") == "xpu":
        # Unsupported XPU ops run on the CPU instead of aborting a whole job.
        os.environ.setdefault("PYTORCH_ENABLE_XPU_FALLBACK", "1")


def accelerator(torch, backend):
    """The torch namespace holding device APIs: torch.xpu for Intel, torch.cuda for CUDA and HIP."""
    return torch.xpu if backend == "xpu" else torch.cuda


def device_type(backend):
    return {"cpu": "cpu", "xpu": "xpu"}.get(backend, "cuda")


def seed_all(torch, backend, seed):
    torch.manual_seed(seed)
    if backend != "cpu":
        accelerator(torch, backend).manual_seed_all(seed)


def prepare(torch, profile):
    """Fail on the wrong wheel or missing GPU instead of silently using CPU."""
    validate_profile(profile)
    if str(torch.__version__) != profile["torch_version"]:
        raise ValueError("PyTorch version differs from runtime profile")
    requested = profile["backend"]
    built = backend_name(torch)
    if requested != "cpu" and (built != requested or not accelerator(torch, requested).is_available()):
        raise RuntimeError(f"Requested {requested} GPU is unavailable (PyTorch build: {built}); no automatic CPU substitution")
    device = "cpu" if requested == "cpu" else device_type(requested) + ":0"
    if requested == "hip":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.set_num_threads(4)
    return dict(backend=requested, device=device, torch=str(torch.__version__),
                hip=getattr(torch.version, "hip", None), cuda=getattr(torch.version, "cuda", None),
                xpu=getattr(torch.version, "xpu", None),
                precision=profile["precision"],
                gpu=None if device == "cpu" else accelerator(torch, requested).get_device_name(0),
                validation="actual_worker_runtime; not a hardware certification")


def verify_audio_library(profile):
    import torchaudio
    if str(torchaudio.__version__) != profile["torchaudio_version"]:
        raise ValueError("torchaudio version differs from runtime profile")
    if profile.get("host") == "windows":
        use_soundfile_loader(torchaudio)


def use_soundfile_loader(torchaudio):
    """torchaudio >= 2.9 decodes through torchcodec, which needs shared FFmpeg DLLs that
    a plain Windows install does not have. Reference audio is WAV, so read it with
    soundfile, which is what the upstream `backend='soundfile'` argument asked for."""
    import soundfile
    import torch

    def load(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None,
             buffer_size=4096, backend=None):
        if not normalize:
            raise ValueError("The soundfile loader only returns normalized float32 audio")
        data, rate = soundfile.read(str(uri), start=frame_offset, frames=num_frames, dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy() if channels_first else data), rate

    torchaudio.load = load


@contextmanager
def cpu_onnx_sessions(ort, sessions):
    """Confine the upstream provider override to initialization in one worker."""
    original = ort.InferenceSession

    def create(*args, **kwargs):
        # Upstream passes providers by keyword; reject ambiguous positional use.
        if len(args) > 2:
            raise ValueError("Unexpected positional ONNX provider arguments")
        kwargs["providers"] = ["CPUExecutionProvider"]
        session = original(*args, **kwargs)
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("ONNX frontend did not use the requested CPU provider")
        sessions.append(dict(providers=session.get_providers()))
        return session

    ort.InferenceSession = create
    try:
        yield
    finally:
        ort.InferenceSession = original
