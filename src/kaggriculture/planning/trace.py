"""Structured chain-of-causation traces for research replays.

Traces never appear in the official action dict sent to the engine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union


SCHEMA = "coc_v1"


@dataclass
class Alternative:
    macro: str
    score: float


@dataclass
class DecisionTrace:
    source: str
    headline: str
    schema: str = SCHEMA
    unit: Optional[str] = None
    macro: Optional[str] = None
    target: Optional[list[int]] = None
    plan: list[str] = field(default_factory=list)
    base_value: Optional[float] = None
    score: Optional[float] = None
    breakdown: Optional[dict[str, float]] = None
    causes: list[str] = field(default_factory=list)
    alternatives: list[Alternative] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["alternatives"] = [
            {"macro": alt.macro, "score": alt.score} for alt in self.alternatives
        ]
        return payload


TraceLike = Union[DecisionTrace, Mapping[str, Any], Sequence[Any], None]


def missing_trace(source: str) -> DecisionTrace:
    return DecisionTrace(
        source=source,
        headline="missing trace",
        causes=["missing_trace"],
    )


def dump_trace(trace: TraceLike) -> Any:
    if trace is None:
        return None
    if isinstance(trace, DecisionTrace):
        return trace.to_dict()
    if isinstance(trace, (list, tuple)):
        return [dump_trace(item) for item in trace]
    if isinstance(trace, Mapping):
        return dict(trace)
    return None


def trace_headline(trace: TraceLike) -> str:
    """One-line headline for UI; crew lists use the seat summary first."""
    if trace is None:
        return ""
    if isinstance(trace, DecisionTrace):
        return trace.headline
    if isinstance(trace, Mapping):
        return str(trace.get("headline") or "")
    if isinstance(trace, (list, tuple)) and trace:
        first = trace[0]
        extra = f" (+{len(trace) - 1} units)" if len(trace) > 1 else ""
        return f"{trace_headline(first)}{extra}"
    return ""


def trace_target(trace: TraceLike) -> Optional[tuple[int, int]]:
    if trace is None:
        return None
    if isinstance(trace, (list, tuple)):
        for item in trace:
            found = trace_target(item)
            if found is not None:
                return found
        return None
    payload = trace.to_dict() if isinstance(trace, DecisionTrace) else dict(trace)
    target = payload.get("target")
    if isinstance(target, (list, tuple)) and len(target) >= 2:
        return int(target[0]), int(target[1])
    return None


def describe_action(action: Mapping[str, Any] | None) -> str:
    if not action:
        return "PASS"
    farmer = action.get("farmer") or ["PASS"]
    op = farmer[0] if farmer else "PASS"
    market = action.get("market") or []
    market_ops = ",".join(str(order[0]) for order in market if order)
    if market_ops:
        return f"{op} / market {market_ops}"
    return str(op)
