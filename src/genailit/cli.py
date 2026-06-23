from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence
from uuid import uuid4

from .core import GenAILitApp


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="genailit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a GenAILit app from a file")
    run_parser.add_argument("app_path", type=Path)
    run_parser.add_argument("--host", default="0.0.0.0")
    run_parser.add_argument("--port", type=int, default=8501)

    return parser.parse_args(argv)


def load_app_from_file(app_path: Path) -> GenAILitApp:
    resolved_path = app_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise SystemExit(f"App file not found: {resolved_path}")

    if str(resolved_path.parent) not in sys.path:
        sys.path.insert(0, str(resolved_path.parent))

    module_name = f"genailit_cli_{resolved_path.stem}_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, resolved_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load app file: {resolved_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return _extract_app(module, resolved_path)


def _extract_app(module: ModuleType, source_path: Path) -> GenAILitApp:
    if not hasattr(module, "app"):
        raise SystemExit(
            f"{source_path} must define an app variable that is a GenAILitApp instance"
        )

    app = getattr(module, "app")
    if not isinstance(app, GenAILitApp):
        raise SystemExit(
            f"{source_path} app must be an instance of GenAILitApp"
        )

    return app


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.command == "run":
        app = load_app_from_file(args.app_path)
        app.run(host=args.host, port=args.port)
        return 0

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
