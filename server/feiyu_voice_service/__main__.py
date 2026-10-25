import argparse
from pathlib import Path

import uvicorn

from feiyu_singing.pipeline import ROOT

from .app import create_app


def main():
    parser = argparse.ArgumentParser(
        description="Feiyu local voice API; run a single process for one shared GPU queue"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/voice_service.json"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9880)
    args = parser.parse_args()
    app = create_app(args.config)
    print(f"API key file (keep private): {app.state.key_file}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
