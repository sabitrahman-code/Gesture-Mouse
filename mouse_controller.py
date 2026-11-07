"""
mouse_controller.py  -  executes Events on the operating system
===============================================================
This is the ONLY module that touches the OS (pyautogui, volume, brightness).
To use pynput instead of pyautogui, change the `_pg()` method.

* Cursor:  PINCH-TO-MOVE (relative).  When a pinch (or a fist drag) starts the controller syncs to the
           REAL cursor position; every `move_rel` then adds the hand movement (calibration scale,
           same as before) -> smoothing -> dead-zone -> pyautogui.moveTo.  Without events the cursor
           never moves, so it is stationary whenever you are not pinching.
* Double click: the first tap already clicked, so the second tap sends ONE more click at the
  same position - the OS pairs the two clicks into a double click.
* Alt+Tab: Alt is pressed lazily on the first swipe and ALWAYS released when the mode ends.
* Volume / brightness run on a worker thread so a slow OS call never stalls the camera loop.
* dry_run=True: nothing is sent to the OS; actions are printed and shown in the preview.
"""
from __future__ import annotations

import math
import platform
import queue
import subprocess
import threading
import time
from typing import List, Optional, Tuple

import config as cfg
from calibration import Calibration
from filters import CursorSmoother
from gesture_recognizer import Event

_SYSTEM = platform.system()


def get_screen_size() -> Tuple[int, int]:
    try:
        import pyautogui
        size = pyautogui.size()
        return int(size.width), int(size.height)
    except Exception:
        return 1920, 1080


# =============================================================== volume / brightness
class SystemControls:
    """Relative volume / brightness changes (percent), executed on a background thread."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self._q: "queue.Queue[Tuple[str, float]]" = queue.Queue()
        self._mac_volume: Optional[int] = None
        self._brightness: Optional[int] = None
        self._warned = set()
        if not dry_run:
            threading.Thread(target=self._worker, daemon=True).start()

    def volume(self, pct: float) -> None:
        if not self.dry_run:
            self._q.put(("volume", pct))

    def brightness(self, pct: float) -> None:
        if not self.dry_run:
            self._q.put(("brightness", pct))

    def _worker(self) -> None:
        while True:
            kind, pct = self._q.get()
            totals = {kind: pct}
            time.sleep(0.02)                              # merge a burst of events
            while True:
                try:
                    k, p = self._q.get_nowait()
                except queue.Empty:
                    break
                totals[k] = totals.get(k, 0.0) + p
            for k, p in totals.items():
                try:
                    (self._set_volume if k == "volume" else self._set_brightness)(int(p))
                except Exception as exc:
                    self._warn(k, f"{k} control failed: {exc}")

    def _warn(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            print(f"[system] {msg}")

    def _set_volume(self, pct: int) -> None:
        if pct == 0:
            return
        if _SYSTEM == "Windows":
            import pyautogui                              # each media-key press = 2 %
            pyautogui.press("volumeup" if pct > 0 else "volumedown",
                            presses=max(1, round(abs(pct) / 2)), interval=0)
        elif _SYSTEM == "Darwin":
            if self._mac_volume is None:
                out = subprocess.check_output(["osascript", "-e", "output volume of (get volume settings)"])
                self._mac_volume = int(out.strip())
            self._mac_volume = max(0, min(100, self._mac_volume + pct))
            subprocess.run(["osascript", "-e", f"set volume output volume {self._mac_volume}"], check=False)
        else:
            try:
                subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{pct:+d}%"], check=True)
            except (OSError, subprocess.CalledProcessError):
                sign = "+" if pct > 0 else "-"
                subprocess.run(["amixer", "-q", "sset", "Master", f"{abs(pct)}%{sign}"], check=False)

    def _set_brightness(self, pct: int) -> None:
        if pct == 0:
            return
        try:
            import screen_brightness_control as sbc
        except ImportError:
            self._warn("sbc", "brightness needs:  pip install screen-brightness-control")
            return
        if self._brightness is None:
            self._brightness = int(sbc.get_brightness()[0])
        self._brightness = max(0, min(100, self._brightness + pct))
        sbc.set_brightness(self._brightness)


# =============================================================== mouse controller
class MouseController:
    def __init__(self, calibration: Calibration, dry_run: bool = False):
        self.calib = calibration
        self.dry_run = dry_run
        self.paused = False
        self.cursor_xy: Tuple[int, int] = (calibration.screen_w // 2, calibration.screen_h // 2)
        self.last_action: Tuple[str, float] = ("", 0.0)

        self._smoother = CursorSmoother()
        self._target: Tuple[float, float] = (float(self.cursor_xy[0]), float(self.cursor_xy[1]))
        self._last_sent: Optional[Tuple[float, float]] = None
        self._last_click_xy: Optional[Tuple[int, int]] = None
        self._dragging = False
        self._alt_down = False
        self._scroll_acc = 0.0
        self._warned = set()
        self._units = cfg.SCROLL_UNITS_PER_NOTCH.get(_SYSTEM, 1)
        self._alt_key = "command" if _SYSTEM == "Darwin" else "alt"
        self._zoom_mod = "command" if _SYSTEM == "Darwin" else "ctrl"
        self._sys = SystemControls(dry_run)
        self._pyautogui = None
        if not dry_run:
            import pyautogui
            pyautogui.FAILSAFE = cfg.PYAUTOGUI_FAILSAFE
            pyautogui.PAUSE = 0                           # no artificial delay after each call
            pyautogui.MINIMUM_DURATION = 0
            self._pyautogui = pyautogui

        self._handlers = {
            "track_start": self._on_track_start, "move_rel": self._on_move_rel,
            "left_click": self._on_left_click,
            "double_click": self._on_double_click, "right_click": self._on_right_click,
            "drag_start": self._on_drag_start, "drag_end": self._on_drag_end,
            "scroll": self._on_scroll, "zoom": self._on_zoom,
            "window_next": self._on_window_next, "window_prev": self._on_window_prev,
            "window_end": self._on_window_end, "level": self._on_level, "macro": self._on_macro,
            "media_play_pause": self._on_media,
        }

    # ---------------------------------------------------------------- plumbing
    def _pg(self, name: str, *args) -> None:
        if self._pyautogui is None:
            return
        try:
            getattr(self._pyautogui, name)(*args)
        except Exception as exc:
            if name not in self._warned:
                self._warned.add(name)
                print(f"[mouse] pyautogui.{name} failed: {exc}")

    def _act(self, label: str) -> None:
        self.last_action = (label, time.perf_counter())
        if self.dry_run:
            print(f"[dry-run] {label}")

    def handle(self, events: List[Event], now: float) -> None:
        for e in events:
            if e.kind == "toggle_pause":
                self.toggle_pause()
                continue
            if self.paused:
                continue
            fn = self._handlers.get(e.kind)
            if fn is not None:
                fn(e, now)

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        if self.paused:
            self.release_all()
        self._act("PAUSED" if self.paused else "RESUMED")

    def release_all(self) -> None:
        """Release every button / key we might be holding. Always call on exit."""
        if self._dragging:
            self._pg("mouseUp")
            self._dragging = False
        if self._alt_down:
            self._pg("keyUp", self._alt_key)
            self._alt_down = False
        self._pg("keyUp", "shift")
        self._smoother.reset()

    # ---------------------------------------------------------------- handlers
    def _real_cursor(self) -> Tuple[int, int]:
        """The actual OS cursor position (so mouse use between pinches is respected)."""
        if self._pyautogui is not None:
            try:
                pos = self._pyautogui.position()
                return int(pos[0]), int(pos[1])
            except Exception:
                pass
        return self.cursor_xy

    def _on_track_start(self, e: Event, now: float) -> None:
        """A pinch-move or a drag begins: continue from where the cursor really is."""
        self.cursor_xy = self._real_cursor()
        self._target = (float(self.cursor_xy[0]), float(self.cursor_xy[1]))
        self._smoother.reset()
        self._last_sent = None

    def _on_move_rel(self, e: Event, now: float) -> None:
        dx, dy = self.calib.delta_to_screen(e.x, e.y)          # same hand->screen scale as calibration
        tx = min(max(self._target[0] + dx * cfg.PINCH_CURSOR_GAIN, 0.0), self.calib.screen_w - 1.0)
        ty = min(max(self._target[1] + dy * cfg.PINCH_CURSOR_GAIN, 0.0), self.calib.screen_h - 1.0)
        self._target = (tx, ty)
        fx, fy = self._smoother(tx, ty, now)                   # smoothing settings unchanged
        if (self._last_sent is not None
                and math.hypot(fx - self._last_sent[0], fy - self._last_sent[1]) < cfg.CURSOR_DEADZONE_PX):
            return
        self._last_sent = (fx, fy)
        self.cursor_xy = (int(round(fx)), int(round(fy)))
        self._pg("moveTo", *self.cursor_xy)

    def _on_media(self, e: Event, now: float) -> None:
        self._act("PLAY / PAUSE")
        self._pg("press", cfg.MEDIA_KEY)

    def _on_left_click(self, e: Event, now: float) -> None:
        self._act("LEFT CLICK")
        self._last_click_xy = self.cursor_xy
        self._pg("click")

    def _on_double_click(self, e: Event, now: float) -> None:
        self._act("DOUBLE CLICK")
        if self._last_click_xy is not None:               # same spot as the first click
            self.cursor_xy = self._last_click_xy
            self._pg("moveTo", *self._last_click_xy)
        self._pg("click")

    def _on_right_click(self, e: Event, now: float) -> None:
        self._act("RIGHT CLICK")
        self._pg("rightClick")

    def _on_drag_start(self, e: Event, now: float) -> None:
        self._act("DRAG START")
        self._dragging = True
        self._pg("mouseDown")

    def _on_drag_end(self, e: Event, now: float) -> None:
        if self._dragging:
            self._act("DROP")
            self._dragging = False
            self._pg("mouseUp")

    def _on_scroll(self, e: Event, now: float) -> None:
        self._scroll_acc += e.value * self._units
        n = int(self._scroll_acc)
        if n:
            self._scroll_acc -= n
            self._act("SCROLL UP" if n > 0 else "SCROLL DOWN")
            self._pg("scroll", n)

    def _on_zoom(self, e: Event, now: float) -> None:
        steps = int(e.value)
        if steps == 0:
            return
        self._act(f"ZOOM {'IN' if steps > 0 else 'OUT'} x{abs(steps)}")
        if cfg.ZOOM_METHOD == "scroll":
            amount = int(round(self._units * cfg.ZOOM_SCROLL_NOTCHES)) * steps
            self._pg("keyDown", self._zoom_mod)
            self._pg("scroll", amount)
            self._pg("keyUp", self._zoom_mod)
        else:
            for _ in range(abs(steps)):
                self._pg("hotkey", self._zoom_mod, "=" if steps > 0 else "-")

    def _ensure_alt(self) -> None:
        if not self._alt_down:
            self._pg("keyDown", self._alt_key)
            self._alt_down = True

    def _on_window_next(self, e: Event, now: float) -> None:
        self._act("NEXT WINDOW")
        self._ensure_alt()
        self._pg("press", "tab")

    def _on_window_prev(self, e: Event, now: float) -> None:
        self._act("PREVIOUS WINDOW")
        self._ensure_alt()
        self._pg("keyDown", "shift")
        self._pg("press", "tab")
        self._pg("keyUp", "shift")

    def _on_window_end(self, e: Event, now: float) -> None:
        if self._alt_down:
            self._act("WINDOW SELECTED")
            self._pg("keyUp", self._alt_key)
            self._alt_down = False

    def _on_level(self, e: Event, now: float) -> None:
        kind = str(e.data)
        self._act(f"{kind.upper()} {e.value:+.0f}%")
        (self._sys.volume if kind == "volume" else self._sys.brightness)(e.value)

    def _on_macro(self, e: Event, now: float) -> None:
        label, keys = e.data
        self._act(f"MACRO: {label}")
        self._pg("hotkey", *keys)
