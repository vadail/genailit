from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """
    Pricing expresado en USD por millón de tokens.

    Ejemplo:
        input_per_million_tokens = 0.30

    significa:

        0.30 USD / 1_000_000 tokens de entrada
    """

    input_per_million_tokens: float
    output_per_million_tokens: float


PRICING: dict[str, dict[str, ModelPricing]] = {
    "google_genai": {
        # Revisar periódicamente precios oficiales
        "gemini-2.5-flash": ModelPricing(
            input_per_million_tokens=0.30,
            output_per_million_tokens=2.50,
        ),
    },
    "bedrock_converse": {
        # Ajustar con precios oficiales de tu organización/región
        "eu.amazon.nova-micro-v1:0": ModelPricing(
            input_per_million_tokens=0.035,
            output_per_million_tokens=0.14,
        ),
    },
}


def get_model_pricing(
    provider: str | None,
    model: str | None,
) -> ModelPricing | None:
    if not provider or not model:
        return None

    provider_catalog = PRICING.get(provider)
    if provider_catalog is None:
        return None

    return provider_catalog.get(model)


def estimate_cost_usd(
    provider: str | None,
    model: str | None,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    pricing = get_model_pricing(provider, model)

    if pricing is None:
        return None

    input_cost = (
        input_tokens / 1_000_000
    ) * pricing.input_per_million_tokens

    output_cost = (
        output_tokens / 1_000_000
    ) * pricing.output_per_million_tokens

    return round(input_cost + output_cost, 8)