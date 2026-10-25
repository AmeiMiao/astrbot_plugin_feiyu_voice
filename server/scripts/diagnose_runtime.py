"""Check actual backend kernels and inference imports in the selected environment."""
import argparse
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feiyu_runtime.common import load, prepare, save, verify_audio_library, verify_repo


def diagnose(profile):
    repo = verify_repo(profile)
    os.chdir(repo)
    sys.path[:0] = [str(repo), str(repo / "third_party/Matcha-TTS")]
    import torch
    info = prepare(torch, profile)
    verify_audio_library(profile)
    device = info["device"]
    with torch.inference_mode():
        a = torch.randn(128, 128, device=device)
        b = a @ a.T
        c = torch.nn.Conv1d(2, 4, 3).to(device)(torch.randn(1, 2, 2048, device=device))
        f = torch.fft.rfft(torch.randn(4096, device=device))
        if not all(bool(torch.isfinite(x).all()) for x in (b, c, f)):
            raise ValueError("Non-finite backend kernel result")
        if device != "cpu":
            torch.cuda.synchronize()
    if profile["engine"] == "rvc":
        from rvc.infer.infer import VoiceConverter
        from rvc.lib.predictors.f0 import RMVPE
        import faiss
    elif profile["engine"] == "tts":
        from cosyvoice.cli.cosyvoice import AutoModel
        import onnxruntime
        if "CPUExecutionProvider" not in onnxruntime.get_available_providers():
            raise RuntimeError("CPU ONNX provider is unavailable")
    else:
        import cli.inference
    return dict(status="success", runtime=info, checks=["source", "versions", "matmul", "conv1d", "fft", "inference_imports"],
                audio_validation="not_run", hardware_certification=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    # Resolve before diagnose() changes the working directory.
    output = args.output.resolve()
    try:
        result = diagnose(load(args.profile))
    except Exception as exc:
        result = dict(status="failed", error=str(exc), traceback=traceback.format_exc(), hardware_certification=False)
    save(output, result)
    print(result["status"] + ": " + str(output))
    return int(result["status"] != "success")


if __name__ == "__main__":
    raise SystemExit(main())
