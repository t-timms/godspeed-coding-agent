"""Cost estimation for LLM API calls.

Maps model prefixes to per-token pricing. Prices are approximate and updated
periodically — they're for budget awareness, not billing-accurate accounting.
Ollama and local models are always $0.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache

logger = logging.getLogger(__name__)

# Pricing: (input_per_million_tokens, output_per_million_tokens)
# Source: provider pricing pages as of April 2026.
# Local/free models are omitted — they default to (0, 0).
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic Claude
    "claude-opus": (15.0, 75.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku": (0.80, 4.00),
    # OpenAI
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    "o3": (10.0, 40.0),
    "o3-mini": (1.10, 4.40),
    "codex-1": (10.0, 40.0),
    # Google Gemini
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.0-flash": (0.10, 0.40),
    # DeepSeek
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    "deepseek-v4-pro": (1.74, 3.48),
    "deepseek-v4-flash": (0.14, 0.28),
    # Mistral
    "mistral-large": (2.0, 6.0),
    "mistral-small": (0.10, 0.30),
}

# Provider prefixes that are always free (local inference)
_FREE_PREFIXES = ("ollama/", "ollama_chat/", "lm_studio/", "llamacpp/")


def _is_free_provider(model_lower: str) -> bool:
    """Check if a model is from a free/local provider."""
    return any(model_lower.startswith(prefix) for prefix in _FREE_PREFIXES)


def _strip_provider_prefix(model_lower: str) -> str:
    """Strip the provider prefix from a model name for pricing matching."""
    return model_lower.split("/")[-1] if "/" in model_lower else model_lower


@lru_cache(maxsize=64)
def _get_model_pricing(name: str) -> tuple[float, float]:
    """Get cached (input_price, output_price) per million tokens for a model name.

    Returns (0.0, 0.0) for unknown models.
    """
    best_match = ""
    best_pricing = (0.0, 0.0)
    for prefix, pricing in _MODEL_PRICING.items():
        if name.startswith(prefix) and len(prefix) > len(best_match):
            best_match = prefix
            best_pricing = pricing
    return best_pricing


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    """Estimate the cost of an LLM call in USD.

    Returns 0.0 for local/free models or unknown pricing.
    Negative token counts are clamped to 0.
    ``cached_input_tokens`` are charged at 10% of the standard input rate
    (Anthropic/DeepSeek prompt-caching discount).
    """
    if input_tokens < 0:
        input_tokens = 0
    if output_tokens < 0:
        output_tokens = 0
    if cached_input_tokens < 0:
        cached_input_tokens = 0
    cached_input_tokens = min(cached_input_tokens, input_tokens)

    model_lower = model.lower()

    # Free local models
    if _is_free_provider(model_lower):
        return 0.0

    # Strip provider prefix for matching (e.g., "anthropic/claude-sonnet..." → "claude-sonnet...")
    name = _strip_provider_prefix(model_lower)

    inp_price, out_price = _get_model_pricing(name)
    if inp_price == 0.0 and out_price == 0.0 and name != "":
        logger.debug("Unknown model pricing for %r — cost estimate is 0.0", model)

    non_cached = max(0, input_tokens - cached_input_tokens)
    input_cost = (non_cached / 1_000_000) * inp_price
    cached_cost = (cached_input_tokens / 1_000_000) * inp_price * 0.10
    output_cost = (output_tokens / 1_000_000) * out_price
    return input_cost + cached_cost + output_cost


def get_cheapest_model(models: list[str]) -> str:
    """Return the model with the lowest per-token cost from a list.

    Used for compaction: pick the cheapest model to summarize history.
    Returns the first model if all are free (Ollama) or unknown.
    """
    if not models:
        return ""

    best_model = models[0]
    best_cost = float("inf")

    for model_name in models:
        model_lower = model_name.lower()
        # Free models get cost 0
        if _is_free_provider(model_lower):
            return model_name  # Can't beat free

        name = _strip_provider_prefix(model_lower)
        inp_price, out_price = _get_model_pricing(name)
        total_cost = inp_price + out_price

        if total_cost == 0.0 and inp_price == 0.0 and out_price == 0.0:
            # Unknown pricing — skip, don't treat as free
            continue

        if total_cost < best_cost:
            best_cost = total_cost
            best_model = model_name

    return best_model


def format_cost(cost: float) -> str:
    """Format a cost value for display.

    Returns "free" for zero cost, otherwise "$X.XXXX".
    """
    if math.isclose(cost, 0.0, abs_tol=1e-10):
        return "free"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"
