"""Estimated API cost from token usage.

PRICES are Anthropic list prices in USD per million tokens. They are NOT fetched live:
update them by hand when prices change (https://platform.claude.com/docs/en/about-claude/pricing).
Last checked: 2026-11-11.
"""
from __future__ import annotations

USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")

# model -> (input, output, cache read) in USD per 1M tokens.
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-opus-5": (5.00, 25.00, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-sonnet-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}
# Writing the 5-minute prompt cache (what agent.py uses) costs 1.25x the input price.
CACHE_WRITE_MULTIPLIER = 1.25


def empty_usage() -> dict[str, int]:
    return dict.fromkeys(USAGE_FIELDS, 0)


def add_usage(total: dict[str, int], usage) -> None:
    """Add one response's usage (an SDK object or a dict) into total. Missing or None fields count as 0."""
    if usage is None:
        return
    for f in USAGE_FIELDS:
        v = usage.get(f) if isinstance(usage, dict) else getattr(usage, f, None)
        total[f] += int(v or 0)


def estimate_cost_usd(usage: dict, model: str) -> float | None:
    """Estimated USD for the given usage, or None when the model is not in PRICES."""
    price = PRICES.get(model)
    if price is None:
        return None
    inp, out, cache_read = price
    cost = (usage.get("input_tokens", 0) * inp
            + usage.get("output_tokens", 0) * out
            + usage.get("cache_read_input_tokens", 0) * cache_read
            + usage.get("cache_creation_input_tokens", 0) * inp * CACHE_WRITE_MULTIPLIER)
    return cost / 1_000_000
