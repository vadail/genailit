import pytest

from genailit import GenAILitEvent


def test_genailit_event_models_name_and_payload() -> None:
    event = GenAILitEvent(name="step_started", payload={"step": 1})

    assert event.name == "step_started"
    assert event.payload == {"step": 1}


def test_genailit_event_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        GenAILitEvent(name="x", unknown=True)  # type: ignore[call-arg]

