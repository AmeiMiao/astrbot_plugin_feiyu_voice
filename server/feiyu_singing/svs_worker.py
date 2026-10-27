"""GPU worker for the pinned SoulX-Singer; no training or upstream code changes."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feiyu_runtime.common import accelerator, apply_backend_env, prepare, seed_all, verify_audio_library, verify_repo


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, type=Path)
    job = load(parser.parse_args().job)
    out = Path(job["output_dir"])
    if (out / "result.json").exists() or (out / "generated.wav").exists():
        raise FileExistsError("SVS output already exists")
    result, started = dict(status="failed"), time.perf_counter()
    try:
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
                          CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="8")
        runtime = job["runtime"]
        managed = runtime.get("schema") == "feiyu-runtime-v1"
        if managed:
            verify_repo(runtime)
        repo = Path(runtime["repo"])
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if commit != runtime["commit"]:
            raise ValueError("SoulX-Singer commit changed")
        for name, digest in runtime.get("code_sha256", {}).items():
            if sha(repo / name) != digest:
                raise ValueError("SoulX-Singer source changed: " + name)
        for item in [job["weight"], job["phoneset"], job["target"], *job["prompt"].values()]:
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("Input/weight checksum differs: " + item["path"])
        if managed:
            apply_backend_env(runtime)
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        import numpy as np
        import soundfile as sf
        import torch
        if managed:
            runtime_info = prepare(torch, runtime)
            verify_audio_library(runtime)
            device = runtime_info["device"]
            backend = runtime_info["backend"]
        else:
            device = "cuda"
            backend = "hip" if torch.version.hip else "cuda"
            runtime_info = dict(backend=backend, device=device,
                                torch=str(torch.__version__))
        import cli.inference as official
        from soulxsinger.utils.data_processor import DataProcessor
        from soulxsinger.utils.file_utils import load_config
        torch.set_num_threads(8)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        config = load_config(str(repo / "soulxsinger/config/soulxsinger.yaml"))
        params = job["parameters"]
        config.infer.cfg, config.infer.n_steps = params["cfg"], params["n_steps"]
        dp = DataProcessor(config.audio.hop_size, config.audio.sample_rate, job["phoneset"]["path"], device="cpu")
        segments = load(job["target"]["path"])
        checks = []
        for meta in segments:
            tensors = dp.process(copy.deepcopy(meta), None)
            expected = round((meta["time"][1] - meta["time"][0]) / 20)
            frames = tensors["mel2note"].shape[1]
            pads = int((tensors["mel2note"] == 0).sum())
            if frames != expected or pads:
                raise ValueError(f"Invalid score tensorization: {meta['index']}: {frames=} {expected=} {pads=}")
            checks.append(dict(index=meta["index"], frames=frames, pad_frames=pads))
        result["score_tensorization"] = checks
        (out / "preflight.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
        if device == "cpu" and params["fp16"]:
            raise ValueError("CPU SVS requires float32")
        if backend == "xpu" and params["fp16"]:
            raise ValueError("XPU SVS requires float32 (upstream FP16 autocast is CUDA-only)")
        model = official.build_model(job["weight"]["path"], config, device=device, use_fp16=params["fp16"])
        random.seed(params["seed"])
        np.random.seed(params["seed"])
        seed_all(torch, backend, params["seed"])
        call = SimpleNamespace(device=device, prompt_wav_path=job["prompt"]["audio"]["path"],
                               prompt_metadata_path=job["prompt"]["metadata"]["path"],
                               target_metadata_path=job["target"]["path"], phoneset_path=job["phoneset"]["path"],
                               save_dir=str(out), control="score", auto_shift=False, pitch_shift=0, use_fp16=params["fp16"])
        if device != "cpu":
            accelerator(torch, backend).reset_peak_memory_stats()
        official.process(call, config, model)
        wave, rate = sf.read(out / "generated.wav", dtype="float32")
        expected_samples = job["frames"] * 480
        if rate != 24000 or wave.ndim != 1 or len(wave) != expected_samples:
            raise ValueError(f"Unexpected SVS output shape: {rate=}, {wave.shape=}, {expected_samples=}")
        if not np.isfinite(wave).all() or np.sqrt(np.mean(wave ** 2)) < 1e-7:
            raise ValueError("SVS output is non-finite or silent")
        result.update(status="success", output_sha256=sha(out / "generated.wav"), frames=len(wave), sample_rate=rate,
                      duration_seconds=len(wave) / rate, peak=float(abs(wave).max()),
                      near_full_scale_samples=int((abs(wave) >= .999).sum()), parameters=params,
                      runtime=runtime_info,
                      gpu=accelerator(torch, backend).get_device_name() if device != "cpu" else None,
                      gpu_peak_allocated_bytes=accelerator(torch, backend).max_memory_allocated() if device != "cpu" else None,
                      model_sha256=job["weight"]["sha256"], model_commit=commit,
                      determinism="Seed recorded; warn-only kernels; bitwise reproducibility not assumed")
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    result["elapsed_seconds"] = time.perf_counter() - started
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return int(result["status"] != "success")


if __name__ == "__main__":
    raise SystemExit(main())
