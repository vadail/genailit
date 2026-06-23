from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_import_genailit_does_not_require_langgraph() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    code = """
import builtins

original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and (name == "langgraph" or name.startswith("langgraph.")):
        raise ImportError("LangGraph should stay optional")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

import genailit

assert genailit.GenAILitApp
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src)

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
