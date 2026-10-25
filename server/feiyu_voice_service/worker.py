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
    if (
        sha(package / "manifest.json") != backend["manifest_sha256"]
        or sha(package / "runtime.local.json") != backend["runtime_sha256"]
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
    if kind == "speech":
        runtime = backend["runtime"]
        task = folder / "tts_request.json"
        save(
            task,
            dict(
                text=request["text"],
                style=request["style"],
                output_dir=linux_path(folder / "output"),
            ),
        )
        # Some standalone Python builds require an explicit library search path.
        rt = dict(distribution=runtime["distribution"], python=runtime["python"])
        command = runtime_command(rt, script, "--request", linux_path(task))
        if runtime.get("python_library_dir"):
            position = command.index(runtime["python"])
            command[position:position] = [
                "env",
                "LD_LIBRARY_PATH=" + runtime["python_library_dir"],
            ]
        run_logged(command, folder / "inference.log")
        receipt = load(folder / "output/result.json")
        source = folder / "output/speech.wav"
        if receipt["status"] != "success" or sha(source) != receipt["playback_sha256"]:
            raise ValueError("TTS package did not return a verified playback WAV")
        metadata = dict(
            style=request["style"],
            sample_rate=receipt["sample_rate"],
            runtime_rvc=False,
        )
    elif kind == "conversion":
        runtime = backend["runtime"]
        rt = dict(
            distribution=runtime["wsl_distribution"], python=runtime["wsl_python"]
        )
        command = runtime_command(
            rt,
            script,
            "--input",
            linux_path(job["audio_path"]),
            "--output-dir",
            linux_path(folder / "output"),
        )
        if not public_rvc:
            command += ["--project-root", linux_path(ROOT)]
        if job.get("regions_path"):
            command += ["--regions", linux_path(job["regions_path"])]
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
        metadata = dict(breath_attenuation_applied=applied)
    else:
        config = backend["singing"]["config"]
        # Voice routing also applies to score singing, so replacing the configured
        # RVC package changes the last stage without changing AstrBot's API calls.
        config["rvc_package"], config["rvc_package_id"] = (
            str(package),
            backend["package_id"],
        )
        config["rvc_runtime"] = dict(
            distribution=backend["runtime"]["wsl_distribution"],
            python=backend["runtime"]["wsl_python"],
        )
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
