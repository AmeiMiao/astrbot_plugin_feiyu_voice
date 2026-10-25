"""Runtime validation shared by isolated GPU workers; no torch import in the API."""

from contextlib import contextmanager
import hashlib
import json
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
    if profile.get("backend") not in ("cuda", "hip", "cpu"):
        raise ValueError("Runtime backend must be cuda, hip or cpu")
    if profile.get("engine") not in ("rvc", "tts", "svs"):
        raise ValueError("Runtime engine must be rvc, tts or svs")
    for key in ("python", "repo", "commit", "torch_version", "torchaudio_version"):
        if not profile.get(key):
            raise ValueError("Runtime profile missing " + key)
    if not profile.get("source_sha256"):
        raise ValueError("Runtime needs the source manifest produced by setup_runtime.py")
    if profile.get("precision") not in ("float32", "float16"):
        raise ValueError("Unsupported runtime precision")
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
    return "cpu"


def prepare(torch, profile):
    """Fail on the wrong wheel or missing GPU instead of silently using CPU."""
    validate_profile(profile)
    if str(torch.__version__) != profile["torch_version"]:
        raise ValueError("PyTorch version differs from runtime profile")
    requested = profile["backend"]
    built = backend_name(torch)
    if requested != "cpu" and (built != requested or not torch.cuda.is_available()):
        raise RuntimeError(f"Requested {requested} GPU is unavailable (PyTorch build: {built}); no automatic CPU substitution")
    device = "cpu" if requested == "cpu" else "cuda:0"
    if requested == "hip":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.set_num_threads(4)
    return dict(backend=requested, device=device, torch=str(torch.__version__),
                hip=getattr(torch.version, "hip", None), cuda=getattr(torch.version, "cuda", None),
                precision=profile["precision"], gpu=None if device == "cpu" else torch.cuda.get_device_name(0),
                validation="actual_worker_runtime; not a hardware certification")


def verify_audio_library(profile):
    import torchaudio
    if str(torchaudio.__version__) != profile["torchaudio_version"]:
        raise ValueError("torchaudio version differs from runtime profile")


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
