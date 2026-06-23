from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GenAILitEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    payload: dict[str, Any] = Field(default_factory=dict)

