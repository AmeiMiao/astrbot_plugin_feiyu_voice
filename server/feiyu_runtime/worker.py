"""Process entry point for managed CUDA/HIP adapters."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feiyu_runtime.common import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=("tts", "rvc"))
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    profile = load(args.profile)
    if profile["engine"] != args.kind:
        raise ValueError("Runtime engine does not match worker kind")
    if args.kind == "tts":
        from feiyu_runtime.tts import main as run
    else:
        from feiyu_runtime.rvc import main as run
    return run(args.package.resolve(), profile, remaining)


if __name__ == "__main__":
    raise SystemExit(main())
