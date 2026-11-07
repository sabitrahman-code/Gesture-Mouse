"""
filters.py  -  cursor anti-jitter
=================================
The cursor goes through up to three stages (all tunable in config.py):

  1. main filter  (config.SMOOTHING_MODE)
        "one_euro"        adaptive low-pass: heavy smoothing when slow (kills jitter), light when fast (no lag)
        "ema"             plain exponential smoothing, weight config.CURSOR_EMA_ALPHA
        "moving_average"  mean of the last config.MOVING_AVERAGE_WINDOW points
        "none"
  2. short moving average (config.CURSOR_AVERAGE_FRAMES, 1 = off)
  3. dead-zone in the controller (config.CURSOR_DEADZONE_PX)

CursorSmoother also resets itself after a tracking gap so the cursor never "swims" in from stale data.
"""
import math
from collections import deque

import config as cfg


class OneEuroFilter:
    """1-D One Euro filter (Casiez et al., CHI 2012)."""

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self._t = None
        self._x = None
        self._dx = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self._t is None:
            self._t, self._x, self._dx = t, x, 0.0
            return x
        dt = max(t - self._t, 1e-4)
        self._t = t
        raw_dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * raw_dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x


class CursorSmoother:
    """Smooths an (x, y) stream.  Call it like a function: smoother(x, y, timestamp)."""

    GAP_RESET_S = 0.4

    def __init__(self, mode=None):
        self.mode = mode or cfg.SMOOTHING_MODE
        self._fx = OneEuroFilter(cfg.ONE_EURO_MIN_CUTOFF, cfg.ONE_EURO_BETA, cfg.ONE_EURO_D_CUTOFF)
        self._fy = OneEuroFilter(cfg.ONE_EURO_MIN_CUTOFF, cfg.ONE_EURO_BETA, cfg.ONE_EURO_D_CUTOFF)
        self._ema = None
        self._buf = deque(maxlen=max(1, cfg.MOVING_AVERAGE_WINDOW))
        self._post = deque(maxlen=max(1, cfg.CURSOR_AVERAGE_FRAMES))
        self._last_t = None

    def reset(self):
        self._fx.reset()
        self._fy.reset()
        self._ema = None
        self._buf.clear()
        self._post.clear()
        self._last_t = None

    def __call__(self, x, y, t):
        if self._last_t is not None and t - self._last_t > self.GAP_RESET_S:
            self.reset()
        self._last_t = t

        if self.mode == "one_euro":
            x, y = self._fx(x, t), self._fy(y, t)
        elif self.mode == "ema":
            a = cfg.CURSOR_EMA_ALPHA
            self._ema = (x, y) if self._ema is None else (a * x + (1 - a) * self._ema[0],
                                                          a * y + (1 - a) * self._ema[1])
            x, y = self._ema
        elif self.mode == "moving_average":
            self._buf.append((x, y))
            n = len(self._buf)
            x, y = sum(p[0] for p in self._buf) / n, sum(p[1] for p in self._buf) / n

        if self._post.maxlen > 1:                          # stage 2: short moving average
            self._post.append((x, y))
            n = len(self._post)
            x, y = sum(p[0] for p in self._post) / n, sum(p[1] for p in self._post) / n
        return x, y
