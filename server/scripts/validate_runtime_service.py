"""Actual HTTP-to-model installation acceptance; never infer success from config alone."""
import argparse
import hashlib
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feiyu_runtime.common import save


def validate(args):
    import httpx
    import numpy as np
    import soundfile as sf
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    token = args.key_file.read_text().strip()
    results = []
    with httpx.Client(base_url=args.base_url.rstrip("/"), headers={"Authorization": "Bearer " + token}, timeout=120) as client:
        def request(method, route, **kwargs):
            response = client.request(method, route, **kwargs)
            response.raise_for_status()
            return response
        cap = request("GET", "/v1/capabilities").json()
        tasks = []
        if args.tts:
            for style in ("cute", "gentle"):
                tasks.append(("speech", dict(voice=args.voice, style=style,
                    text="你好，我是肥鱼。今天我们来检查语音合成和声音服务是否正常。")))
        if args.vocal:
            asset = request("POST", "/v1/assets", files={"file": (args.vocal.name, args.vocal.read_bytes())}).json()
            tasks.append(("voice-conversion", dict(voice=args.voice, audio_id=asset["id"])))
        if args.score:
            asset = request("POST", "/v1/assets", files={"file": (args.score.name, args.score.read_bytes())}).json()
            tasks.append(("singing", dict(voice=args.voice, score_id=asset["id"], lyrics=args.lyrics)))
        if not tasks:
            raise ValueError("Select --tts, --vocal and/or --score")
        for route, payload in tasks:
            job = request("POST", "/v1/" + route, json=payload).json()
            deadline = time.monotonic() + 1800
            while True:
                row = request("GET", "/v1/jobs/" + job["id"]).json()
                if row["status"] in ("failed", "interrupted"):
                    raise RuntimeError("Inference failed: " + str(row))
                if row["status"] == "success":
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Inference acceptance timed out")
                time.sleep(1)
            expected_path = "/v1/jobs/" + job["id"] + "/audio"
            if row["audio_url"] != expected_path:
                raise ValueError("Unexpected audio download path")
            data = request("GET", expected_path).content
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("Downloaded WAV hash mismatch")
            path = output / (route + "_" + payload.get("style", "default") + ".wav")
            path.write_bytes(data)
            wave, rate = sf.read(path)
            if not wave.size or not np.isfinite(wave).all() or np.sqrt(np.mean(wave**2)) < 1e-7:
                raise ValueError("Invalid or silent output")
            results.append(dict(route=route, style=payload.get("style"), job=row, sample_rate=rate,
                                seconds=len(wave) / rate, downloaded_sha256=row["sha256"]))
            save(output / "validation.json", dict(status="in_progress", capabilities=cap, results=results))
            print(route + " " + payload.get("style", "") + " passed", flush=True)
        save(output / "validation.json", dict(status="passed", capabilities=cap, results=results,
             subjective_listening="pending", hardware_certification=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:9880")
    parser.add_argument("--key-file", required=True, type=Path)
    parser.add_argument("--voice", default="feiyu")
    parser.add_argument("--tts", action="store_true")
    parser.add_argument("--vocal", type=Path)
    parser.add_argument("--score", type=Path)
    parser.add_argument("--lyrics")
    parser.add_argument("--output-dir", required=True, type=Path)
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
