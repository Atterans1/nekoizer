"""Single source of truth for engine parameters: types, ranges, validation."""
from __future__ import annotations

# ---------------------------------------------------------------- PUBLIC
ParamSpec: dict[str, dict] = {
    "pitch_semitones": {"type": "int", "min": -12, "max": 12, "default": 0},
    "f0_preserve": {"type": "bool", "default": False},
    "timbre_blend": {"type": "float", "min": 0.0, "max": 1.0, "default": 1.0},
    "cfm_steps": {"type": "int", "min": 4, "max": 15, "default": 4},
    "chunk_ms": {"type": "int", "min": 80, "max": 320, "default": 320},
    "crossfade_ms": {"type": "int", "min": 5, "max": 160, "default": 20},
    "vad_threshold": {"type": "float", "min": 0.05, "max": 0.95, "default": 0.5},
    "output_gain": {"type": "float", "min": 0.0, "max": 2.0, "default": 1.0},
}


def Defaults() -> dict:
    return {k: v["default"] for k, v in ParamSpec.items()}


def Ranges() -> dict:
    return {
        k: {"type": v["type"], **({"min": v["min"], "max": v["max"]} if "min" in v else {})}
        for k, v in ParamSpec.items()
    }


def Validate(key: str, value, current: dict | None = None):
    """Return the coerced value or raise ValueError. crossfade_ms is also bounded by chunk_ms // 2."""
    spec = ParamSpec.get(key)
    if spec is None:
        raise ValueError(f"Unknown parameter '{key}'")
    kind = spec["type"]
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be true or false")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    if kind == "int":
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{key} must be a whole number")
        value = int(value)
    else:
        value = float(value)
        if value != value:
            raise ValueError(f"{key} must not be NaN")
    if not spec["min"] <= value <= spec["max"]:
        raise ValueError(f"{key} must be between {spec['min']} and {spec['max']} (got {value})")
    if key == "crossfade_ms" and current is not None and value > current["chunk_ms"] // 2:
        raise ValueError(f"crossfade_ms must be at most half of chunk_ms ({current['chunk_ms'] // 2})")
    if key == "chunk_ms" and current is not None and current["crossfade_ms"] > value // 2:
        raise ValueError(f"chunk_ms {value} is too small for crossfade_ms {current['crossfade_ms']}")
    return value
