"""Create one isolated Linux inference environment and a machine-local profile."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from feiyu_runtime.common import save, sha

SOURCES = {
    "rvc": ("https://github.com/IAHispano/Applio.git", "67a38f6abab3e90a55efa534ea3185a222c102f0"),
    "tts": ("https://github.com/FunAudioLLM/CosyVoice.git", "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"),
    "svs": ("https://github.com/Soul-AILab/SoulX-Singer.git", "81aeb3ae772c70093c3de74dc23c92d983801ae4"),
}


def run(command, **kwargs):
    subprocess.run([str(x) for x in command], check=True, **kwargs)


def native_windows():
    return sys.platform == "win32"


def venv_python(destination):
    return destination / ("venv/Scripts/python.exe" if native_windows() else "venv/bin/python")


def plan(backend, engine, destination, python):
    preset = json.loads((ROOT / "runtime_profiles" / (backend + ".json")).read_text())
    stack = preset["engines"][engine]
    venv, repo = destination / "venv", destination / "source"
    executable = venv_python(destination)
    constraint = destination / "torch-constraints.txt"
    # Native Windows has no system Python 3.11 to rely on: uv provisions it.
    create = (["uv", "venv", "--seed", "--python", preset["python"], str(venv)] if native_windows()
              else [python, "-m", "venv", str(venv)])
    return preset, stack, [
        create,
        [str(executable), "-m", "pip", "install", "--upgrade", "pip"],
        [str(executable), "-m", "pip", "install", "--index-url", preset["torch_index"],
         "torch==" + stack["torch"], "torchaudio==" + stack["torchaudio"]],
        [str(executable), "-m", "pip", "install", "-c", str(constraint), "-r",
         str(ROOT / "runtime_profiles" / ("requirements-" + engine + ".txt"))],
    ]


def source_manifest(repo):
    # Hash source/configs, including overlays and submodules, not model binaries.
    excluded = {".git", "__pycache__", ".cache", ".venv", "venv", "logs", "outputs"}
    suffixes = {".py", ".json", ".yaml", ".yml", ".toml"}
    return {p.relative_to(repo).as_posix(): sha(p) for p in sorted(repo.rglob("*"))
            if p.is_file() and p.suffix in suffixes and not p.name.endswith(".local.json")
            and not any(part in excluded for part in p.relative_to(repo).parts)}


def build_profile(preset, stack, engine, python, repo, distribution):
    profile = dict(schema="feiyu-runtime-v1", engine=engine, backend=preset["backend"],
                   distribution=distribution, python=str(python), repo=str(repo), commit=SOURCES[engine][1],
                   torch_version=stack["torch"], torchaudio_version=stack["torchaudio"], precision=stack["precision"],
                   host="windows" if native_windows() else "wsl",
                   f0_device=stack.get("f0_device", "cpu"), onnx_provider="CPUExecutionProvider",
                   source_sha256=source_manifest(repo), hardware_validation=preset["hardware_validation"])
    library = Path(subprocess.check_output([str(python), "-c", "import sys; print(sys.base_prefix + '/lib')"], text=True).strip())
    if list(library.glob("libpython3.11.so*")):
        profile["python_library_dir"] = str(library)
    if engine == "rvc":
        settings = ROOT / "configs/applio_settings.json"
        profile["settings"] = json.loads(settings.read_text())
    return profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=("cuda", "rocm", "xpu"))
    parser.add_argument("--engine", required=True, choices=tuple(SOURCES))
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--python", default="python3.11")
    parser.add_argument("--distribution", default="Ubuntu-24.04")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dest = args.destination.expanduser().resolve()
    preset, stack, commands = plan(args.backend, args.engine, dest, args.python)
    if args.dry_run:
        print(json.dumps(dict(commands=commands, source=SOURCES[args.engine], profile=str(dest / "runtime.local.json"),
                             hardware_validation=preset["hardware_validation"]), indent=2))
        return 0
    windows = native_windows()
    if windows and args.backend != "xpu":
        parser.error("Native Windows installs are only provided for Intel XPU; run CUDA/ROCm inside WSL")
    if (sys.platform not in ("linux", "win32")) or platform.machine().lower() not in ("x86_64", "amd64"):
        parser.error("Run this installer on x86_64 Linux/WSL, or on Windows for Intel XPU")
    if dest.exists():
        parser.error("Destination already exists. Use a new directory; never modify an active environment")
    if windows:
        if not shutil.which("uv"):
            parser.error("Install uv first (it provisions Python " + preset["python"] + "): https://docs.astral.sh/uv/")
    else:
        version = subprocess.check_output([args.python, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], text=True).strip()
        if version != preset["python"]:
            parser.error("This preset requires Python " + preset["python"])
    if not shutil.which("git") or not shutil.which("ffmpeg"):
        parser.error("Install git and ffmpeg first (see INSTALL_AGENT.md system packages)")
    dest.mkdir(parents=True)
    constraint = dest / "torch-constraints.txt"
    constraint.write_text("torch==" + stack["torch"] + "\ntorchaudio==" + stack["torchaudio"] + "\n")
    repo = dest / "source"
    url, commit = SOURCES[args.engine]
    run(["git", "-c", "core.autocrlf=false", "clone", "--no-checkout", url, repo])
    run(["git", "-C", repo, "checkout", "--detach", commit])
    if args.engine == "tts":
        run(["git", "-C", repo, "submodule", "update", "--init", "--recursive"])
    if args.engine == "rvc":
        overlay = ROOT / "runtime_compat/Applio"
        if not overlay.is_dir():
            raise FileNotFoundError("Use the published server/ directory, including runtime_compat/Applio")
        for name, digest in json.loads((ROOT / "runtime_compat/source_sha256.json").read_text()).items():
            if sha(overlay / name) != digest:
                raise ValueError("Runtime overlay checksum mismatch (preserve Git attributes/line endings): " + name)
        shutil.copytree(overlay, repo, dirs_exist_ok=True)
        assets = json.loads((ROOT / "runtime_compat/upstream_assets.json").read_text())
        for asset in assets:
            target = repo / asset["destination"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and sha(target) == asset["sha256"]:
                continue
            temporary = target.with_name(target.name + ".download")
            with urllib.request.urlopen(asset["url"], timeout=120) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            if sha(temporary) != asset["sha256"]:
                raise ValueError("Upstream asset checksum mismatch: " + asset["destination"])
            temporary.replace(target)
    for command in commands:
        run(command)
    executable = venv_python(dest)
    if stack.get("torchcodec") and not windows:
        run([executable, "-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cpu",
             "torchcodec==" + stack["torchcodec"], "-c", constraint])
    # Freeze the fully resolved environment and reject inconsistent dependencies.
    run([executable, "-m", "pip", "check"])
    freeze = subprocess.check_output([str(executable), "-m", "pip", "freeze"], text=True)
    (dest / "requirements-resolved.txt").write_text(freeze)
    profile = build_profile(preset, stack, args.engine, executable, repo, args.distribution)
    save(dest / "runtime.local.json", profile)
    # Diagnose the actual backend; installer success never implies audio validation.
    diagnostic_env = dict(os.environ)
    if profile.get("python_library_dir"):
        diagnostic_env["LD_LIBRARY_PATH"] = profile["python_library_dir"] + ":" + diagnostic_env.get("LD_LIBRARY_PATH", "")
    run([executable, ROOT / "scripts/diagnose_runtime.py", "--profile", dest / "runtime.local.json",
         "--output", dest / "diagnostic.json"], env=diagnostic_env)
    print("Profile ready: " + str(dest / "runtime.local.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
