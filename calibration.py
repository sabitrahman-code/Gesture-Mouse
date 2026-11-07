"""
calibration.py  -  hand-to-screen mapping, sensitivity and the calibration wizard
================================================================================
You never move your hand across the WHOLE camera image.  Instead a smaller rectangle
(the "active region", drawn in blue in the preview) is stretched over the whole screen.

    Calibration.region       (x0, y0, x1, y1) in 0..1 camera coordinates
    Calibration.sensitivity  >1 shrinks the region around its centre => less hand travel

CalibrationWizard walks you through the four corners of YOUR comfortable reach
(index fingertip, hold still ~1.5 s each) and stores the result in calibration.json.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np

import config as cfg

Region = Tuple[float, float, float, float]


def _path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.CALIBRATION_FILE)


class Calibration:
    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w, self.screen_h = screen_w, screen_h
        self.region: Region = tuple(cfg.DEFAULT_ACTIVE_REGION)      # type: ignore[assignment]
        self.sensitivity = float(cfg.SENSITIVITY)
        self.calibrated = False
        self.swap_handedness = None        # saved Left/Right fix (None = not decided yet)

    # ---------------------------------------------------------------- persistence
    @classmethod
    def load(cls, screen_w: int, screen_h: int) -> "Calibration":
        cal = cls(screen_w, screen_h)
        try:
            with open(_path(), "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return cal
        try:
            if "region" in data:                          # region only exists after a real calibration
                cal.region = tuple(float(v) for v in data["region"])      # type: ignore[assignment]
                cal.calibrated = True
            cal.sensitivity = float(data.get("sensitivity", cfg.SENSITIVITY))
        except (TypeError, ValueError):
            pass
        swap = data.get("swap_handedness")
        cal.swap_handedness = swap if isinstance(swap, bool) else None
        return cal

    def save(self) -> None:
        data = {"sensitivity": self.sensitivity, "swap_handedness": self.swap_handedness}
        if self.calibrated:
            data["region"] = list(self.region)
        try:
            with open(_path(), "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except OSError as exc:
            print(f"[calibration] could not save: {exc}")

    # ---------------------------------------------------------------- mapping
    def set_region(self, region: Region) -> None:
        self.region = region
        self.calibrated = True

    def adjust_sensitivity(self, delta: float) -> None:
        self.sensitivity = float(np.clip(self.sensitivity + delta, cfg.SENSITIVITY_MIN, cfg.SENSITIVITY_MAX))

    def delta_to_screen(self, dx: float, dy: float) -> Tuple[float, float]:
        """Hand movement (camera-normalised) -> cursor movement in pixels, same scale as to_screen()."""
        x0, y0, x1, y1 = self.effective_region()
        return dx / (x1 - x0) * (self.screen_w - 1), dy / (y1 - y0) * (self.screen_h - 1)

    def effective_region(self) -> Region:
        x0, y0, x1, y1 = self.region
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        hw = (x1 - x0) / 2.0 / self.sensitivity
        hh = (y1 - y0) / 2.0 / self.sensitivity
        return cx - hw, cy - hh, cx + hw, cy + hh

    def to_screen(self, nx: float, ny: float) -> Tuple[float, float]:
        """Camera-normalised (0..1) -> screen pixels, clamped to the screen."""
        x0, y0, x1, y1 = self.effective_region()
        u = min(max((nx - x0) / (x1 - x0), 0.0), 1.0)
        v = min(max((ny - y0) / (y1 - y0), 0.0), 1.0)
        return u * (self.screen_w - 1), v * (self.screen_h - 1)


class CalibrationWizard:
    """Feed it the index fingertip every frame; it returns the new region when finished."""

    STEPS = ("TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT")

    def __init__(self):
        self.active = False
        self.message = ""
        self.message_t = -1e9
        self._i = 0
        self._corners: list = []
        self._phase = "ready"
        self._t0 = 0.0
        self._last_t = 0.0
        self._samples: list = []

    # ---------------------------------------------------------------- control
    def start(self, now: float) -> None:
        self.active = True
        self._i = 0
        self._corners = []
        self.message = ""
        self._reset_step(now)

    def cancel(self) -> None:
        self.active = False

    def _reset_step(self, now: float) -> None:
        self._phase, self._t0, self._samples = "ready", now, []

    # ---------------------------------------------------------------- for the UI
    @property
    def prompt(self) -> str:
        if not self.active:
            return ""
        corner = self.STEPS[self._i]
        if self._phase == "ready":
            return f"CALIBRATION {self._i + 1}/4: move index finger to your {corner} reach"
        return f"CALIBRATION {self._i + 1}/4: hold still at {corner}..."

    @property
    def progress(self) -> float:
        if not self.active or self._phase != "hold":
            return 0.0
        return min(1.0, max(0.0, (self._last_t - self._t0) / cfg.CALIBRATION_HOLD_S))

    # ---------------------------------------------------------------- per frame
    def update(self, tip: Optional[np.ndarray], now: float) -> Optional[Region]:
        if not self.active:
            return None
        self._last_t = now
        if tip is None:                                   # hand lost: restart this corner
            self._reset_step(now)
            return None
        if self._phase == "ready":
            if now - self._t0 >= cfg.CALIBRATION_READY_S:
                self._phase, self._t0, self._samples = "hold", now, []
            return None

        self._samples.append(np.asarray(tip, dtype=float))
        median = np.median(self._samples, axis=0)
        if np.linalg.norm(np.asarray(tip) - median) > cfg.CALIBRATION_STABILITY:
            self._reset_step(now)                         # moved too much: try this corner again
            return None
        if now - self._t0 < cfg.CALIBRATION_HOLD_S:
            return None

        self._corners.append(median)
        self._i += 1
        if self._i < len(self.STEPS):
            self._reset_step(now)
            return None
        return self._finish(now)

    def _finish(self, now: float) -> Optional[Region]:
        tl, tr, br, bl = self._corners
        x0, x1 = (tl[0] + bl[0]) / 2.0, (tr[0] + br[0]) / 2.0
        y0, y1 = (tl[1] + tr[1]) / 2.0, (bl[1] + br[1]) / 2.0
        if x1 - x0 < cfg.CALIBRATION_MIN_SIZE[0] or y1 - y0 < cfg.CALIBRATION_MIN_SIZE[1]:
            self.start(now)                               # start() clears the message, so set it after
            self.message = "Area too small - spread the corners further apart and retry"
            self.message_t = now
            return None
        self.active = False
        self.message, self.message_t = "Calibration saved", now
        return (float(x0), float(y0), float(x1), float(y1))
