"""Serve the real model experience on a local HTTP endpoint."""

import argparse
import logging

import uvicorn

from .app import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8032)
    parser.add_argument("--model-server", default="http://127.0.0.1:8031")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-type", choices=("audio", "omni"), default="omni")
    parser.add_argument("--demo-config", help="Demo deployment JSON for frontend settings")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        create_app(
            settings_path=args.config,
            demo_path=args.demo_config,
            model_type=args.model_type,
            tokenizer_path=args.tokenizer_path,
            model_server_url=args.model_server,
        ),
        host=args.host,
        port=args.port,
        ws_max_size=512 * 1024,
    )


if __name__ == "__main__":
    main()
