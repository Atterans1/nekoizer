"""Single source of truth for the Seed-VC voice changer: parameter names, ranges, defaults, presets, fixed constants.

Every module (audio/input_chain.py, engine/seedvc_expressive.py, audio/output_chain.py, ui/) reads parameters by these
exact names through a ParamStore. Defaults come from docs/seedvc/SPEC.md; resolutions in docs/seedvc/SPEC_RESOLUTIONS.md.
"""
from __future__ import annotations

import threading
from typing import Any, Callable

# ---------------------------------------------------------------- PUBLIC
SLOT_NAMES = tuple("slot " + str(i + 1) for i in range(10))  # generic names: one clip/model per slot, no emotion mapping

# Tunable controls. type: int | float | bool | enum. "values" is for enum.
ParamSpec: dict[str, dict[str, Any]] = {
    "block_ms": {"type": "int", "min": 100, "max": 500, "default": 220},
    "overlap_ms": {"type": "float", "min": 40.0, "max": 50.0, "default": 45.0},
    "diffusion_steps": {"type": "int", "min": 6, "max": 16, "default": 10},
    "cfg_rate": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.65},
    "left_context_s": {"type": "float", "min": 1.5, "max": 2.5, "default": 2.0},
    "pitch_shift_st": {"type": "float", "min": -12.0, "max": 12.0, "default": 0.0},
    "formant_shift": {"type": "float", "min": -2.0, "max": 2.0, "default": 0.0},
    "f0_expander": {"type": "float", "min": 0.8, "max": 1.4, "default": 1.0},
    "prosody_follow": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
    "energy_follow": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
    "timbre_mix": {"type": "float", "min": 0.0, "max": 1.0, "default": 1.0},
    "emotion_mode": {"type": "enum", "values": ("auto", "manual"), "default": "auto"},
    "manual_slot": {"type": "int", "min": 0, "max": 9, "default": 0},
    "monitor_enabled": {"type": "bool", "default": False},
    "monitor_volume": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
    "auto_balancer": {"type": "bool", "default": True},
}

# Fixed by the spec or by a documented choice (SPEC_RESOLUTIONS.md). Not sliders.
HPF_HZ = 80.0                  # spec: 80 Hz high-pass; R8: 2nd-order Butterworth
COMP_RATIO = 2.5               # spec: soft 2.5:1
COMP_THRESHOLD_DB = -20.0      # R9 (choice)
COMP_KNEE_DB = 6.0             # R9 (choice)
COMP_ATTACK_MS = 8.0           # R9 (choice)
COMP_RELEASE_MS = 120.0        # R9 (choice)
COMP_MAKEUP_MAX_DB = 9.0       # R9 (choice)
VAD_HANG_MS = 120.0            # spec
VAD_FRAME_SAMPLES = 512        # Silero ONNX at 16 kHz = 32 ms (R10)
VAD_THRESHOLD = 0.5            # R10 (choice)
BALANCER_THRESHOLD = 0.85      # spec: approaches 85 percent of block time
BALANCER_STEP_DROP = 2         # spec
BALANCER_HOLD_S = 3.0          # spec
BALANCER_EMA_ALPHA = 0.3       # R17 (choice): EMA of inference_time / block_time
PROMPT_MAX_S = 3.0             # R18 (choice): Seed-VC documented real-time prompt length
MODEL_SR = 22050               # seed-uvit-tat-xlsr-tiny output rate
SOLA_SEARCH_MS = 20.0          # Seed-VC real-time-gui.py: sola_search_frame = samplerate // 50 (choice: same value)
EXTRA_CONTEXT_RIGHT_S = 0.02   # Seed-VC documented real-time config: right context 0.02 s (adds 20 ms lookahead)
INPUT_SR = 16000               # XLSR content encoder input rate
EMOTION_DWELL_MS = 400.0       # R11
EMOTION_RMS_WINDOW_S = 30.0    # R11
EMOTION_PITCH_WINDOW_S = 1.0   # R11
EMOTION_WHISPER_DB_BELOW_MEDIAN = 8.0   # R11
EMOTION_ANGRY_DB_ABOVE_MEDIAN = 6.0     # R11
EMOTION_ANGRY_PITCH_STD_ST = 2.5        # R11
EMOTION_HYPED_PITCH_STD_ST = 3.5        # R11
EMOTION_EMBED_TAU_S = 0.2      # R12
F0_MEDIAN_WINDOW_S = 10.0      # R15
RECORDER_MAX_S = 10.0          # spec: built-in 10 s recorder
HOTKEY_MODIFIERS = "Ctrl+Alt"  # R21: Ctrl+Alt+1..9 selects slots 0..8; Ctrl+Alt+0 (VK_0) selects slot 9
CABLE_OUTPUT_NAME = "CABLE Input"

# Presets (R19). Values are choices; only listed controls change. The fourth preset is cut off in the source (R20).
PRESETS: dict[str, dict[str, Any] | None] = {
    "Natural Conversational": {"emotion_mode": "auto", "f0_expander": 1.0},
    "High-Energy Gaming / Ragebait": {"emotion_mode": "auto", "f0_expander": 1.3},
    "Soft / Whisper": {"emotion_mode": "manual", "manual_slot": 2, "f0_expander": 0.8},
    "Ultra-Low- (cut off in source)": None,
    "Bright Announcer": {"emotion_mode": "manual", "manual_slot": 1, "pitch_shift_st": 2.0, "formant_shift": 0.5},
    "Deep Narrator": {"emotion_mode": "manual", "manual_slot": 0, "pitch_shift_st": -3.0, "formant_shift": -0.5, "f0_expander": 0.9},
    "Gruff Villain": {"emotion_mode": "manual", "manual_slot": 3, "pitch_shift_st": -2.0, "formant_shift": -1.0, "f0_expander": 1.2},
    "Chipmunk Hype": {"emotion_mode": "manual", "manual_slot": 1, "pitch_shift_st": 5.0, "formant_shift": 1.0, "f0_expander": 1.3},
    "Calm Dialog": {"emotion_mode": "auto", "prosody_follow": 0.5, "energy_follow": 0.5, "timbre_mix": 0.9, "f0_expander": 0.9},
    "Flat Robot": {"emotion_mode": "manual", "manual_slot": 0, "pitch_shift_st": 0.0, "prosody_follow": 0.0, "energy_follow": 0.0, "timbre_mix": 1.0, "f0_expander": 0.8},
}
TRUNCATED_PRESET_LABEL = "Ultra-Low-"   # the source text stops here; no name or values are invented
FOURTH_PRESET_VALUES = None


def Defaults() -> dict[str, Any]:
    return {k: v["default"] for k, v in ParamSpec.items()}


def Validate(name: str, value: Any) -> Any:
    """Return the coerced value or raise ValueError. Unknown names raise KeyError."""
    spec = ParamSpec[name]
    kind = spec["type"]
    if kind == "bool":
        if isinstance(value, bool):
            return value
        raise ValueError(f"{name} must be a bool")
    if kind == "enum":
        if value in spec["values"]:
            return value
        raise ValueError(f"{name} must be one of {spec['values']}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    v = int(round(value)) if kind == "int" else float(value)
    if v != v or not (spec["min"] <= v <= spec["max"]):
        raise ValueError(f"{name}={value} is outside {spec['min']}..{spec['max']}")
    return v


class ParamStore:
    """Thread-safe parameter dictionary shared by every module. Subscribers are called after a change."""

    # ---- PUBLIC
    def __init__(self, initial: dict[str, Any] | None = None):
        self._lock = threading.Lock()
        self._values = Defaults()
        self._subscribers: list[Callable[[str, Any], None]] = []
        for k, v in (initial or {}).items():
            self._values[k] = Validate(k, v)

    def Get(self, name: str) -> Any:
        with self._lock:
            return self._values[name]

    def Set(self, name: str, value: Any) -> Any:
        coerced = Validate(name, value)
        with self._lock:
            if name == "overlap_ms" or name == "block_ms":
                block = coerced if name == "block_ms" else self._values["block_ms"]
                overlap = coerced if name == "overlap_ms" else self._values["overlap_ms"]
                if overlap > block / 2:
                    raise ValueError("overlap_ms must be at most half of block_ms")
            self._values[name] = coerced
            subs = list(self._subscribers)
        for fn in subs:
            fn(name, coerced)
        return coerced

    def Snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._values)

    def ApplyPreset(self, name: str) -> dict[str, Any]:
        values = PRESETS[name]  # KeyError for unknown names
        if values is None:
            raise ValueError("this preset is cut off in the spec and has no values")
        for k, v in values.items():
            self.Set(k, v)
        return self.Snapshot()

    def Subscribe(self, fn: Callable[[str, Any], None]) -> None:
        with self._lock:
            self._subscribers.append(fn)


class Janitor:
    """One cleanup point: tasks run newest-first, once, and a failing task does not stop the others."""

    # ---- PUBLIC
    def __init__(self):
        self._tasks: list[Callable[[], None]] = []
        self._lock = threading.Lock()
        self.failures = 0

    def Add(self, task: Callable[[], None]) -> None:
        with self._lock:
            self._tasks.append(task)

    def Cleanup(self) -> None:
        with self._lock:
            pending, self._tasks = self._tasks[::-1], []
        for task in pending:
            try:
                task()
            except Exception:  # noqa: BLE001 - cleanup must continue
                self.failures += 1
