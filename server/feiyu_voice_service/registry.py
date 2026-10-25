"""Validate configured package formats; do not import GPU runtimes into the server."""

import copy
from pathlib import Path

from feiyu_singing.pipeline import load, resolve, sha


class Registry:
    def __init__(self, config_path):
        self.path = Path(config_path).resolve()
        self.reload()

    def reload(self):
        config = load(self.path)
        if config.get("schema") != "feiyu-voice-service-v1":
            raise ValueError("Unknown service configuration schema")
        entries = {}
        for kind, engine in (
            ("tts", "cosyvoice3_release"),
            ("rvc", "a60_cover_release"),
        ):
            entries[kind] = {}
            for voice, item in config.get(kind, {}).items():
                package = resolve(item["package"])
                entry = dict(voice=voice, engine=item["engine"], available=False)
                try:
                    if item["engine"] != engine and not (
                        kind == "rvc" and item["engine"] == "feiyu_rvc_release"
                    ):
                        raise ValueError("No adapter installed for this engine")
                    manifest = load(package / "manifest.json")
                    if kind == "tts":
                        if manifest["schema"] != "feiyu-voice-local-release-v1":
                            raise ValueError("Unsupported TTS package format")
                        styles = list(manifest["styles"])
                        script = package / "infer.py"
                        version = manifest["version"]
                        identity = "FeiyuVoice-" + version
                    else:
                        if manifest["schema"] not in (
                            "feiyu-a60-d76-cover-package-v1",
                            "feiyu-rvc-release-v1",
                        ):
                            raise ValueError("Unsupported conversion package format")
                        styles = []
                        script = package / (
                            "run_rvc.py"
                            if manifest["schema"] == "feiyu-rvc-release-v1"
                            else "run_cover.py"
                        )
                        version, identity = manifest["version"], manifest["id"]
                    runtime = package / "runtime.local.json"
                    if (
                        not script.is_file()
                        or not runtime.is_file()
                        or not (package / "files_sha256.json").is_file()
                    ):
                        raise ValueError(
                            "Package scripts or local runtime configuration missing"
                        )
                    entry.update(
                        available=True,
                        version=version,
                        package_id=identity,
                        styles=styles,
                        package=str(package),
                        manifest_sha256=sha(package / "manifest.json"),
                        runtime=load(runtime),
                        runtime_sha256=sha(runtime),
                        script_sha256=sha(script),
                    )
                except (OSError, ValueError, KeyError) as exc:
                    entry["reason"] = type(exc).__name__ + ": " + str(exc)
                entries[kind][voice] = entry
        singing = dict(available=False, reason="Score synthesis is not configured")
        if config.get("singing"):
            try:
                path = resolve(config["singing"]["config"])
                settings = load(path)
                voice = config["singing"]["rvc_voice"]
                selected = entries["rvc"].get(voice, {})
                if not selected.get("available"):
                    raise ValueError("Selected conversion package is unavailable")
                if not resolve(settings["svs_weight"]["path"]).is_file():
                    raise ValueError("SVS weights are not installed")
                for role in ("audio", "metadata"):
                    if not resolve(settings["prompt"][role]["path"]).is_file():
                        raise ValueError("SVS reference is not installed")
                singing = dict(
                    available=True,
                    rvc_voice=voice,
                    config=settings,
                    config_sha256=sha(path),
                )
            except (OSError, ValueError, KeyError) as exc:
                singing["reason"] = str(exc)
        self.config, self.entries, self.singing = config, entries, singing

    def public(self):
        return dict(
            tts=[
                {
                    k: e[k]
                    for k in (
                        "voice",
                        "engine",
                        "available",
                        "version",
                        "package_id",
                        "styles",
                    )
                    if k in e
                }
                for e in self.entries["tts"].values()
            ],
            rvc=[
                {
                    k: e[k]
                    for k in ("voice", "engine", "available", "version", "package_id")
                    if k in e
                }
                for e in self.entries["rvc"].values()
            ],
            singing=dict(
                available=self.singing["available"],
                formats=["midi", "abc"],
                languages=["zh"],
                rvc_voice=self.singing.get("rvc_voice"),
            ),
            verification="Package format/config presence checked; workers verify assets and runtime before inference",
            gpu_concurrency=1,
        )

    def select(self, kind, voice):
        entry = self.entries["tts" if kind == "speech" else "rvc"].get(voice)
        if not entry or not entry["available"]:
            raise ValueError("Requested voice capability is not installed/configured")
        if kind == "singing" and not self.singing["available"]:
            raise ValueError(
                "Score singing requires an installed SVS engine and reference, in addition to RVC"
            )
        result = copy.deepcopy(entry)
        if kind == "singing":
            result["singing"] = copy.deepcopy(self.singing)
        return result
