"""Live monitoring for the Seed-VC pipeline: rolling telemetry, system probes and health rules.

monitoring.recorder.TelemetryRecorder is the entry point; monitoring.metrics holds the pure maths and the health thresholds;
monitoring.probes reads this process's CPU and RAM and the GPU's load and memory.
"""
# ---- PUBLIC
from monitoring.metrics import EvaluateHealth, RULE_NAMES  # noqa: F401
from monitoring.recorder import TelemetryRecorder  # noqa: F401
