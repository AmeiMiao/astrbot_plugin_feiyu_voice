"""Managed RVC adapter; model-package checks and network identity remain enforced."""

import argparse
import hashlib
import inspect
import json
import os
import random
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path

from feiyu_runtime.common import apply_backend_env, device_type, prepare, seed_all, verify_repo, verify_audio_library

BUNDLE = None


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def verify():
    for name, record in load(BUNDLE / "files_sha256.json").items():
        path = (BUNDLE / name).resolve()
        if (
            not path.is_relative_to(BUNDLE)
            or not path.is_file()
            or sha(path) != record["sha256"]
        ):
            raise ValueError("Package integrity check failed: " + name)
    manifest = load(BUNDLE / "manifest.json")
    if manifest["schema"] != "feiyu-rvc-release-v1":
        raise ValueError("Unexpected package schema")
    return manifest


def main(package, profile, argv=None):
    global BUNDLE
    BUNDLE = Path(package).resolve()
    parser = argparse.ArgumentParser(
        description="FeiyuVoice-RVC v1: convert clean vocals, optionally attenuate manually marked breaths"
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--regions", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    manifest = verify()
    if args.verify_only:
        print(
            json.dumps(
                dict(
                    status="verified",
                    package=manifest["id"],
                    version=manifest["version"],
                )
            )
        )
        return 0
    if not args.input:
        parser.error("--input is required")
    runtime = profile
    repo = verify_repo(profile)
    for name, digest in load(BUNDLE / "runtime_pins.json").items():
        if sha(repo / name) != digest:
            raise ValueError(
                "Configured Applio runtime differs from the validated files: " + name
            )
    source = args.input.resolve(strict=True)
    out = (
        args.output_dir.resolve()
        if args.output_dir
        else BUNDLE
        / "outputs"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    if out.exists():
        raise ValueError(
            "Choose a new output directory; existing results are never overwritten"
        )
    out.mkdir(parents=True)
    settings = out / "applio_settings.json"
    save(settings, runtime["settings"])
    os.environ.update(
        APPLIO_CONFIG_FILE=str(settings),
        APPLIO_LOGS_DIR=str(out / "runtime_logs"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="4",
        PYTHONDONTWRITEBYTECODE="1",
    )
    apply_backend_env(profile)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import numpy as np
    import soundfile as sf
    import torch
    from rvc.infer.infer import VoiceConverter
    from rvc.infer.pipeline import Pipeline
    from rvc.lib.algorithm.synthesizers import Synthesizer

    runtime_info = prepare(torch, profile)
    verify_audio_library(profile)
    if runtime_info["backend"] in ("cpu", "xpu"):
        # Upstream only auto-detects cuda/mps; the accelerator is chosen explicitly.
        from rvc.configs.config import Config
        Config().device = runtime_info["device"]
    f0_loads = []
    import rvc.infer.pipeline as pipeline_module
    original_rmvpe = pipeline_module.RMVPE

    def configured_rmvpe(*args, **kwargs):
        if profile.get("f0_device", "cpu") == "cpu":
            kwargs["device"] = "cpu"
        predictor = original_rmvpe(*args, **kwargs)
        f0_loads.append(predictor)
        return predictor

    pipeline_module.RMVPE = configured_rmvpe
    model = manifest["model"]
    model_path = (BUNDLE / model["pth"]).resolve()
    index_path = (BUNDLE / model["index"]).resolve()
    params = manifest["inference"]
    loads, calls = [], []
    original_load = VoiceConverter.get_vc

    def observe_load(self, *positional, **keywords):
        value = original_load(self, *positional, **keywords)
        expected_device = device_type(runtime_info["backend"])
        if next(self.net_g.parameters()).device.type != expected_device:
            raise RuntimeError("RVC network is not on the configured device")
        if (
            Path(self.model_path).resolve() != model_path
            or sha(self.model_path) != model["pth_sha256"]
        ):
            raise ValueError("Unexpected RVC model was loaded")
        digest = hashlib.sha256()
        for key, tensor in sorted(self.net_g.state_dict().items()):
            digest.update(key.encode())
            digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
        if digest.hexdigest() != manifest["model_tensor_digest"]:
            raise ValueError("Loaded network differs from the selected release")
        loads.append(
            dict(
                model_sha256=sha(self.model_path),
                network_tensor_digest=digest.hexdigest(),
            )
        )
        return value

    VoiceConverter.get_vc = observe_load
    original_infer = Synthesizer.infer

    def seeded_infer(self, *positional, **keywords):
        seed = manifest["seed"] + len(calls)
        random.seed(seed)
        np.random.seed(seed)
        seed_all(torch, runtime_info["backend"], seed)
        calls.append(dict(seed=seed))
        return original_infer(self, *positional, **keywords)

    Synthesizer.infer = seeded_infer
    original_convert = Pipeline.voice_conversion

    def observe_convert(self, *positional, **keywords):
        values = (
            inspect.signature(original_convert)
            .bind(self, *positional, **keywords)
            .arguments
        )
        if (
            float(values["protect"]) != params["protect"]
            or float(values["index_rate"]) != params["index_rate"]
        ):
            raise ValueError("Conversion parameters differ from the release")
        return original_convert(self, *positional, **keywords)

    Pipeline.voice_conversion = observe_convert
    native = out / "rvc_native.wav"
    save(
        out / "job.json",
        dict(
            package=manifest["id"],
            input=str(source),
            input_sha256=sha(source),
            manifest_sha256=sha(BUNDLE / "manifest.json"),
            parameters=params,
        ),
    )
    sys.argv = [
        str(repo / "core.py"),
        "infer",
        "--input-path",
        str(source),
        "--output-path",
        str(native),
        "--pth-path",
        str(model_path),
        "--index-path",
        str(index_path),
        "--pitch",
        str(params["pitch_shift"]),
        "--index-rate",
        str(params["index_rate"]),
        "--protect",
        str(params["protect"]),
        "--volume-envelope",
        str(params["volume_envelope"]),
        "--f0-method",
        params["f0_method"],
        "--embedder-model",
        "contentvec",
        "--export-format",
        "WAV",
    ]
    try:
        runpy.run_path(str(repo / "core.py"), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
    if not native.is_file() or not loads or not calls:
        raise ValueError("The runtime did not produce a verified conversion")
    wave, rate = sf.read(native)
    if (
        rate != 40000
        or wave.ndim != 1
        or not len(wave)
        or not np.isfinite(wave).all()
        or np.sqrt(np.mean(wave**2)) < 1e-7
    ):
        raise ValueError("Invalid converted audio")
    patch = None
    if args.regions:
        sys.path.insert(0, str(BUNDLE))
        from apply_breath import process

        patch = process(native, args.regions, out / "breath")
    receipt = dict(
        status="success",
        package=manifest["id"],
        version=manifest["version"],
        model_sha256=model["pth_sha256"],
        index_sha256=model["index_sha256"],
        inference=params,
        native_sha256=sha(native),
        actual_model_loads=loads,
        actual_synth_calls=calls,
        breath_attenuation_applied=patch is not None,
        breath_attenuation=patch,
        sample_rate=rate,
        seconds=len(wave) / rate,
        new_training_updates=0,
        runtime=runtime_info,
        f0_devices=[str(p.model.device) for p in f0_loads],
    )
    save(out / "cover_result.json", receipt)
    print(out / "breath/final.wav" if patch else native)
    return 0
