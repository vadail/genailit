from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genailit import GenAILitApp


app = GenAILitApp()


@app.agent
async def sagemaker_demo(input_data, context):
    yield "Hola desde SageMaker "
    yield "con GenAILit"


def main() -> None:
    app.run(host="0.0.0.0", port=8501)


if __name__ == "__main__":
    main()
