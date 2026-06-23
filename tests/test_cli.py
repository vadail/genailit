from __future__ import annotations

from textwrap import dedent

import pytest

from genailit import GenAILitApp
from genailit.cli import load_app_from_file, main, parse_args


def test_cli_parses_basic_arguments() -> None:
    args = parse_args(["run", "demo.py", "--host", "0.0.0.0", "--port", "8501"])

    assert args.command == "run"
    assert str(args.app_path) == "demo.py"
    assert args.host == "0.0.0.0"
    assert args.port == 8501


def test_cli_loads_app_from_file(tmp_path) -> None:
    app_file = tmp_path / "app.py"
    app_file.write_text(
        dedent(
            """
            from genailit import GenAILitApp

            app = GenAILitApp()

            @app.agent
            async def demo(input_data, context):
                yield "Hola desde CLI"
            """
        )
    )

    app = load_app_from_file(app_file)

    assert isinstance(app, GenAILitApp)
    assert app.adapter is not None


def test_cli_errors_when_app_is_missing(tmp_path) -> None:
    app_file = tmp_path / "app.py"
    app_file.write_text("value = 1\n")

    with pytest.raises(SystemExit) as exc:
        load_app_from_file(app_file)

    assert "must define an app variable" in str(exc.value)


def test_cli_errors_when_app_is_not_genailit_app(tmp_path) -> None:
    app_file = tmp_path / "app.py"
    app_file.write_text("app = object()\n")

    with pytest.raises(SystemExit) as exc:
        load_app_from_file(app_file)

    assert "must be an instance of GenAILitApp" in str(exc.value)


def test_cli_runs_loaded_app_with_requested_host_and_port(tmp_path, monkeypatch) -> None:
    app_file = tmp_path / "app.py"
    app_file.write_text(
        dedent(
            """
            from genailit import GenAILitApp

            app = GenAILitApp()
            """
        )
    )

    recorded: dict[str, object] = {}

    def fake_run(self, host: str = "0.0.0.0", port: int = 8501, reload: bool = False) -> None:
        recorded["self"] = self
        recorded["host"] = host
        recorded["port"] = port
        recorded["reload"] = reload

    monkeypatch.setattr(GenAILitApp, "run", fake_run)

    exit_code = main(["run", str(app_file), "--host", "0.0.0.0", "--port", "8501"])

    assert exit_code == 0
    assert isinstance(recorded["self"], GenAILitApp)
    assert recorded["host"] == "0.0.0.0"
    assert recorded["port"] == 8501
    assert recorded["reload"] is False
