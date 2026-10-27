"""Adapters for existing immutable release packages. Invoked by the queue in isolation."""

import argparse
import shutil
from pathlib import Path

from feiyu_singing.pipeline import (
    ROOT,
    linux_path,
    load,
    run_logged,
    runtime_command,
    save,
    sha,
    synthesize,
)


def execute(job):
    import numpy as np
    import soundfile as sf

    folder = Path(job["job_directory"])
    backend, request, kind = job["backend"], job["request"], job["kind"]
    package = Path(backend["package"])
    managed = backend.get("adapter") == "managed"
    runtime = backend["runtime"]
    runtime_path = Path(backend.get("runtime_path", package / "runtime.local.json"))
    if (
        sha(package / "manifest.json") != backend["manifest_sha256"]
        or sha(runtime_path) != backend["runtime_sha256"]
    ):
        raise ValueError(
            "Package configuration changed after submission; reload service and resubmit"
        )
    public_rvc = (
        kind != "speech"
        and load(package / "manifest.json")["schema"] == "feiyu-rvc-release-v1"
    )
    script = package / (
        "infer.py"
        if kind == "speech"
        else "run_rvc.py"
        if public_rvc
        else "run_cover.py"
    )
    if sha(script) != backend["script_sha256"]:
        raise ValueError("Package adapter changed after submission")
    if managed:
        from feiyu_voice_service.registry import adapter_digest
        if adapter_digest() != backend["adapter_sha256"]:
            raise ValueError("Managed adapter changed after submission")
        profile_snapshot = folder / "runtime_profile.json"
        save(profile_snapshot, backend["runtime"])
        script = ROOT / "feiyu_runtime/worker.py"
        prefix = ["--kind", "tts" if kind == "speech" else "rvc", "--package", linux_path(package, runtime),
                  "--profile", linux_path(profile_snapshot, runtime)]
    else:
        prefix = []
    if kind == "speech":
        runtime = backend["runtime"]
        task = folder / "tts_request.json"
        save(
            task,
            dict(
                text=request["text"],
                style=request["style"],
                output_dir=linux_path(folder / "output", runtime),
            ),
        )
        # Some standalone Python builds require an explicit library search path.
        rt = dict(distribution=runtime.get("distribution"), python=runtime["python"],
                  python_library_dir=runtime.get("python_library_dir"), host=runtime.get("host"))
        command = runtime_command(rt, script, *prefix, "--request", linux_path(task, runtime))
        run_logged(command, folder / "inference.log")
        receipt = load(folder / "output/result.json")
        source = folder / "output/speech.wav"
        if receipt["status"] != "success" or sha(source) != receipt["playback_sha256"]:
            raise ValueError("TTS package did not return a verified playback WAV")
        metadata = dict(
            style=request["style"],
            sample_rate=receipt["sample_rate"],
            runtime_rvc=False,
            runtime=receipt.get("runtime"),
        )
    elif kind == "conversion":
        runtime = backend["runtime"]
        rt = dict(
            distribution=runtime.get("distribution") if managed else runtime["wsl_distribution"],
            python=runtime["python"] if managed else runtime["wsl_python"]
        )
        if managed:
            rt["python_library_dir"] = runtime.get("python_library_dir")
            rt["host"] = runtime.get("host")
        command = runtime_command(
            rt,
            script,
            *prefix,
            "--input",
            linux_path(job["audio_path"], runtime),
            "--output-dir",
            linux_path(folder / "output", runtime),
        )
        if not public_rvc:
            command += ["--project-root", linux_path(ROOT, runtime)]
        if job.get("regions_path"):
            command += ["--regions", linux_path(job["regions_path"], runtime)]
        run_logged(command, folder / "inference.log")
        receipt = load(folder / "output/cover_result.json")
        applied = receipt.get(
            "breath_attenuation_applied", receipt.get("D76_applied", False)
        )
        source = (
            folder
            / "output"
            / (
                ("breath/final.wav" if public_rvc else "D76/final.wav")
                if applied
                else "rvc_native.wav"
            )
        )
        if (
            receipt["status"] != "success"
            or receipt["package"] != backend["package_id"]
        ):
            raise ValueError("RVC package returned an invalid result")
        metadata = dict(breath_attenuation_applied=applied, runtime=receipt.get("runtime"),
                        f0_devices=receipt.get("f0_devices"))
    else:
        config = backend["singing"]["config"]
        # Voice routing also applies to score singing, so replacing the configured
        # RVC package changes the last stage without changing AstrBot's API calls.
        config["rvc_package"], config["rvc_package_id"] = (
            str(package),
            backend["package_id"],
        )
        config["rvc_runtime"] = dict(
            distribution=backend["runtime"].get("distribution") if managed else backend["runtime"]["wsl_distribution"],
            python=backend["runtime"]["python"] if managed else backend["runtime"]["wsl_python"],
        )
        if managed:
            config["rvc_runtime_profile"] = str(profile_snapshot)
        config_path = folder / "singing_config.json"
        save(config_path, config)
        result = synthesize(
            job["score_path"],
            lyrics=request.get("lyrics"),
            lyrics_file=job.get("lyrics_path"),
            pinyin_file=job.get("pinyin_path"),
            track=request.get("track"),
            channel=request.get("channel"),
            bpm=request.get("bpm"),
            midi_encoding=request["midi_encoding"],
            transpose=request["transpose"],
            segment_seconds=request["segment_seconds"],
            output_dir=folder / "output",
            config=config_path,
            regions=job.get("regions_path"),
        )
        source = Path(result["output_file"])
        metadata = dict(
            breath_attenuation_applied=result["rvc"].get(
                "breath_attenuation_applied", result["rvc"].get("D76_applied", False)
            ),
            source="score_to_svs_to_rvc",
        )
    wave, rate = sf.read(source)
    if not wave.size or not np.isfinite(wave).all() or np.sqrt(np.mean(wave**2)) < 1e-7:
        raise ValueError("Generated audio is empty, silent or non-finite")
    shutil.copyfile(source, folder / "audio.wav")
    audio = dict(
        metadata,
        sample_rate=rate,
        seconds=len(wave) / rate,
        channels=1 if wave.ndim == 1 else wave.shape[1],
    )
    save(
        folder / "worker_result.json",
        dict(status="success", sha256=sha(folder / "audio.wav"), audio=audio),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    execute(load(parser.parse_args().job))


if __name__ == "__main__":
    main()
