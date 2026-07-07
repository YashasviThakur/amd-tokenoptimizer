"""GPU telemetry for the cockpit.

In `live` mode it reads a real device via `rocm-smi` (AMD) or `nvidia-smi`.
In `mock` mode — or when no GPU tool is present — it synthesizes a believable
AMD Instinct reading whose utilization *spikes* every time a query is answered
on-device. That spike, on screen next to the savings counter, is the demo's
money shot: the AMD GPU visibly lighting up for a zero-remote-token answer.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass


@dataclass
class GpuMonitor:
    mode: str = "mock"
    device_name: str = "AMD Instinct MI300X"
    _activity: float = 0.0  # 0..1, decays over ~3s, spikes on local inference
    _last: float = 0.0

    def mark_local_inference(self, tokens: int = 0) -> None:
        self._activity = min(1.0, self._activity + 0.6 + min(0.35, tokens / 400.0))

    def _decay(self, now: float) -> None:
        if self._last:
            self._activity = max(0.0, self._activity - (now - self._last) / 3.0)
        self._last = now

    def read(self, now: float | None = None) -> dict:
        now = now if now is not None else time.time()
        self._decay(now)

        if self.mode == "live":
            real = _read_rocm() or _read_nvidia()
            if real:
                real["activity"] = round(self._activity, 3)
                return real

        # mock / fallback — synthesize an MI300X that reacts to local inference
        util = min(100.0, 6.0 + self._activity * 88.0)
        mem_total = 192000.0  # MI300X = 192 GB HBM3
        mem_used = 14000.0 + self._activity * 26000.0
        return {
            "name": self.device_name,
            "util_percent": round(util, 1),
            "mem_used_mb": round(mem_used, 1),
            "mem_total_mb": mem_total,
            "temp_c": round(38.0 + self._activity * 24.0, 1),
            "power_w": round(140.0 + self._activity * 500.0, 1),
            "source": "mock" if self.mode != "live" else "fallback",
            "activity": round(self._activity, 3),
        }


def _num(value) -> float | None:
    try:
        return float(str(value).split()[0].replace("%", ""))
    except (TypeError, ValueError, IndexError):
        return None


def _read_rocm() -> dict | None:
    exe = shutil.which("rocm-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--showuse", "--showmemuse", "--showtemp", "--showpower", "--json"],
            capture_output=True, text=True, timeout=3,
        )
        data = json.loads(out.stdout)
        card = next(iter(data.values()))

        def find(*needles):
            for needle in needles:
                for k, v in card.items():
                    if needle.lower() in k.lower():
                        n = _num(v)
                        if n is not None:
                            return n
            return None

        used = find("VRAM Total Used Memory")
        total = find("VRAM Total Memory")
        return {
            "name": card.get("Card series") or card.get("Card model") or "AMD GPU",
            "util_percent": find("GPU use (%)", "GPU use") or 0.0,
            "mem_used_mb": (used / (1024 * 1024)) if used else (find("GPU Memory Allocated (VRAM%)") or 0.0),
            "mem_total_mb": (total / (1024 * 1024)) if total else 0.0,
            "temp_c": find("Temperature (Sensor edge)", "Temperature", "edge") or 0.0,
            "power_w": find("Average Graphics Package Power", "Current Socket Graphics Package Power", "Power") or 0.0,
            "source": "rocm-smi",
        }
    except Exception:
        return None


def _read_nvidia() -> dict | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe,
             "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        parts = out.stdout.strip().splitlines()[0].split(",")
        return {
            "name": parts[0].strip(),
            "util_percent": float(parts[1]),
            "mem_used_mb": float(parts[2]),
            "mem_total_mb": float(parts[3]),
            "temp_c": float(parts[4]),
            "power_w": float(parts[5]),
            "source": "nvidia-smi",
        }
    except Exception:
        return None
