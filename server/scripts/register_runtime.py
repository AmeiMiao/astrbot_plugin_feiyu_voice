"""Register an existing compatible environment without installing or editing it."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from setup_runtime import ROOT, SOURCES, build_profile, run, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=("cuda", "rocm"))
    parser.add_argument("--engine", required=True, choices=tuple(SOURCES))
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--distribution", default="Ubuntu-24.04")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Profile exists; choose a new output file")
    preset = json.loads((ROOT / "runtime_profiles" / (args.backend + ".json")).read_text())
    commit = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCES[args.engine][1]:
        parser.error("Source commit does not match the selected engine")
    profile = build_profile(preset, preset["engines"][args.engine], args.engine,
                            args.python.absolute(), args.repo.resolve(), args.distribution)
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, profile)
    diagnostic_env = dict(os.environ)
    if profile.get("python_library_dir"):
        diagnostic_env["LD_LIBRARY_PATH"] = profile["python_library_dir"] + ":" + diagnostic_env.get("LD_LIBRARY_PATH", "")
    run([args.python, ROOT / "scripts/diagnose_runtime.py", "--profile", output,
         "--output", output.with_suffix(".diagnostic.json")], env=diagnostic_env)
    print("Registered: " + str(output))


if __name__ == "__main__":
    main()
