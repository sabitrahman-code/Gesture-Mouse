"""
gesture_recognizer.py  -  turns tracked hands into high-level Events
====================================================================
Pipeline per frame:

    Hand (landmarks)  --HandAnalyzer-->  HandFeatures (finger states, confirmed pinches, debounced pose)
                      --OneHandGestures / TwoHandGestures-->  [Event, Event, ...]

This module knows NOTHING about the screen or the OS; it only emits Events:

    track_start                  a cursor-moving gesture begins (the controller syncs to the real cursor)
    move_rel(x, y)               hand movement since the last frame, normalised camera units
    left_click, double_click, right_click, drag_start, drag_end
    scroll(value)                wheel notches, + = up
    zoom(value)                  signed integer steps, + = zoom in
    window_next / window_prev / window_end
    level(value, data)           data = "volume" | "brightness", value = signed percent
    macro(data=(label, keys)), media_play_pause, toggle_pause

STABILITY RULES used everywhere
    * pinch must be seen PINCH_CONFIRM_FRAMES in a row to start and PINCH_RELEASE_FRAMES to end
      (clicks / cursor use a stricter "click-grade" pinch, see HandAnalyzer)
    * poses must persist POSE_STABLE_FRAMES before they count
    * every discrete gesture has a hold time and/or cooldown, and one-shot gestures must be
      released before they can fire again
Poses: OPEN FIST POINT PEACE THREE PINCH MIDDLE_PINCH UNKNOWN (NONE before any is stable).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

import config as cfg
from hand_tracker import (Hand, PALM_POINTS, WRIST, THUMB_MCP, THUMB_TIP, INDEX_MCP, INDEX_PIP, INDEX_TIP,
                          MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP, RING_MCP, RING_PIP, RING_TIP,
                          PINKY_MCP, PINKY_PIP, PINKY_TIP)


@dataclass
class Event:
    kind: str
    x: float = 0.0
    y: float = 0.0
    value: float = 0.0
    data: object = None


# =============================================================== per-hand features
@dataclass
class HandFeatures:
    label: str
    fresh: bool                 # False = hand is momentarily lost, features are the last known ones
    hand: Hand
    size: float                 # wrist -> middle knuckle, in pixels (unit for all ratios)
    fingers: Tuple[bool, bool, bool, bool, bool]    # thumb, index, middle, ring, pinky extended?
    ext_count: int              # how many of index..pinky are extended (0-4)
    thumb_ratio: float          # thumb "stick-out" measure (tucked thumb < ~0.8)
    idx_ratio: float            # thumb tip <-> index tip / size
    mid_ratio: float            # thumb tip <-> middle tip / size
    index_pinch: bool           # CONFIRMED pinch state (hysteresis + confirmation frames)
    middle_pinch: bool
    raw_pose: str
    pose: str                   # debounced pose
    tip: Tuple[float, float]    # index fingertip (normalised)
    pinch_mid: Tuple[float, float]   # midpoint thumb tip / index tip (normalised)
    palm: Tuple[float, float]   # palm centre (normalised)
    scroll_y: float             # mean y of index + middle fingertips (normalised)
    index_vec: Tuple[float, float]   # index MCP -> tip vector in pixels
    click_pinch: bool = False   # strict, confirmed index+thumb pinch (cursor movement and left click)
    right_pinch: bool = False   # strict, confirmed middle+thumb pinch (right click)
    speed: float = 0.0          # palm speed in frame widths per second (smoothed)
    pinch_now: bool = False     # instantaneous (unconfirmed) "fingers are touching" state of THIS frame


class HandAnalyzer:
    """Stateful per-hand feature extractor (pinch hysteresis + confirmation + pose debouncing)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._idx_pinch = self._mid_pinch = False
        self._idx_on = self._idx_off = self._mid_on = self._mid_off = 0
        self._pose = "NONE"
        self._cand: Optional[str] = None
        self._cand_n = 0
        self.last: Optional[HandFeatures] = None
        # click-grade pinch state (cursor + clicks only)
        self._c_idx = self._c_mid = False
        self._c_idx_on = self._c_idx_off = self._c_mid_on = self._c_mid_off = 0
        self._c_idx_armed = self._c_mid_armed = False
        self._age = 0
        self._speed = 0.0
        self._prev_palm_px = None
        self._prev_t = None

    @staticmethod
    def _confirm(state: bool, raw: bool, on_n: int, off_n: int,
                 need_on: Optional[int] = None, need_off: Optional[int] = None):
        """Flip `state` only after the opposite condition was seen several frames in a row."""
        need_on = cfg.PINCH_CONFIRM_FRAMES if need_on is None else need_on
        need_off = cfg.PINCH_RELEASE_FRAMES if need_off is None else need_off
        if raw:
            on_n, off_n = on_n + 1, 0
        else:
            on_n, off_n = 0, off_n + 1
        if not state and on_n >= need_on:
            state = True
        elif state and off_n >= need_off:
            state = False
        return state, on_n, off_n

    @staticmethod
    def _raw_pose(fingers, index_pinch, middle_pinch) -> str:
        _, ix, md, rg, pk = fingers
        if index_pinch:
            return "PINCH"
        if middle_pinch:
            return "MIDDLE_PINCH"
        n = ix + md + rg + pk
        if n == 4:
            return "OPEN"
        if n == 0:
            return "FIST"
        if ix and not md and not rg and not pk:
            return "POINT"
        if ix and md and not rg and not pk:
            return "PEACE"
        if ix and md and rg and not pk:
            return "THREE"
        return "UNKNOWN"

    def update(self, hand: Hand) -> HandFeatures:
        if hand.stale and self.last is not None:
            self.last = replace(self.last, fresh=False, hand=hand)
            return self.last

        p, n = hand.pts, hand.npts

        def d(a, b):
            return float(np.hypot(p[a][0] - p[b][0], p[a][1] - p[b][1]))

        size = max(d(WRIST, MIDDLE_MCP), 1e-6)

        def reach(tip, pip):                       # > ~1.2 means the finger is straight
            return d(tip, WRIST) / max(d(pip, WRIST), 1e-6)

        thumb_ratio = d(THUMB_TIP, PINKY_MCP) / max(d(THUMB_MCP, PINKY_MCP), 1e-6)
        fingers = (
            thumb_ratio > cfg.THUMB_EXTENDED_RATIO,
            reach(INDEX_TIP, INDEX_PIP) > cfg.FINGER_EXTENDED_RATIO,
            reach(MIDDLE_TIP, MIDDLE_PIP) > cfg.FINGER_EXTENDED_RATIO,
            reach(RING_TIP, RING_PIP) > cfg.FINGER_EXTENDED_RATIO,
            reach(PINKY_TIP, PINKY_PIP) > cfg.FINGER_EXTENDED_RATIO,
        )

        idx_ratio = d(THUMB_TIP, INDEX_TIP) / size
        mid_ratio = d(THUMB_TIP, MIDDLE_TIP) / size
        thumb_out = thumb_ratio >= cfg.PINCH_MIN_THUMB_RATIO      # a tucked thumb (fist) is never a pinch
        idx_raw = thumb_out and idx_ratio < (cfg.PINCH_OFF if self._idx_pinch else cfg.PINCH_ON)
        mid_raw = thumb_out and mid_ratio < (cfg.PINCH_OFF if self._mid_pinch else cfg.PINCH_ON)
        if idx_raw and mid_raw:                    # both close: the tighter one wins
            if idx_ratio <= mid_ratio:
                mid_raw = False
            else:
                idx_raw = False
        self._idx_pinch, self._idx_on, self._idx_off = self._confirm(
            self._idx_pinch, idx_raw, self._idx_on, self._idx_off)
        self._mid_pinch, self._mid_on, self._mid_off = self._confirm(
            self._mid_pinch, mid_raw, self._mid_on, self._mid_off)
        if self._idx_pinch and self._mid_pinch:
            if idx_ratio <= mid_ratio:
                self._mid_pinch = False
            else:
                self._idx_pinch = False

        raw = self._raw_pose(fingers, self._idx_pinch, self._mid_pinch)
        if raw == self._pose:
            self._cand, self._cand_n = None, 0
        else:
            if raw == self._cand:
                self._cand_n += 1
            else:
                self._cand, self._cand_n = raw, 1
            if self._cand_n >= cfg.POSE_STABLE_FRAMES:
                self._pose, self._cand, self._cand_n = raw, None, 0

        # ---- click-grade pinch: stricter, used ONLY for cursor movement and clicks --------------
        palm_px = p[PALM_POINTS].mean(axis=0)
        if (self._prev_palm_px is not None and hand.last_seen > self._prev_t
                and hand.last_seen - self._prev_t < 0.25):
            v = float(np.linalg.norm(palm_px - self._prev_palm_px)) / hand.frame_w / (hand.last_seen - self._prev_t)
            self._speed = 0.6 * self._speed + 0.4 * v
        self._prev_palm_px, self._prev_t = palm_px, hand.last_seen
        self._age += 1
        if idx_ratio >= cfg.CLICK_PINCH_OFF:           # fingers seen clearly open: a pinch may start
            self._c_idx_armed = True
        if mid_ratio >= cfg.RIGHT_CLICK_PINCH_OFF:
            self._c_mid_armed = True
        settled = self._age >= cfg.MIN_TRACK_FRAMES and self._speed <= cfg.CLICK_START_MAX_SPEED
        if self._c_idx:
            idx_c = idx_ratio < cfg.CLICK_PINCH_OFF
        else:
            idx_c = (settled and self._c_idx_armed and thumb_out and idx_ratio < cfg.CLICK_PINCH_ON
                     and (mid_ratio - idx_ratio) >= cfg.CLICK_AMBIGUITY_MARGIN)
        if self._c_mid:
            mid_c = mid_ratio < cfg.RIGHT_CLICK_PINCH_OFF
        else:
            mid_c = (settled and self._c_mid_armed and thumb_out and mid_ratio < cfg.RIGHT_CLICK_PINCH_ON
                     and idx_ratio >= cfg.RIGHT_CLICK_MIN_INDEX_GAP)
        self._c_idx, self._c_idx_on, self._c_idx_off = self._confirm(
            self._c_idx, idx_c, self._c_idx_on, self._c_idx_off,
            cfg.CLICK_CONFIRM_FRAMES, cfg.CLICK_RELEASE_FRAMES)
        self._c_mid, self._c_mid_on, self._c_mid_off = self._confirm(
            self._c_mid, mid_c, self._c_mid_on, self._c_mid_off,
            cfg.RIGHT_CLICK_CONFIRM_FRAMES, cfg.CLICK_RELEASE_FRAMES)
        if self._c_idx and self._c_mid:                 # never both: the tighter one wins
            if idx_ratio <= mid_ratio:
                self._c_mid = False
            else:
                self._c_idx = False

        mid = (n[THUMB_TIP] + n[INDEX_TIP]) / 2.0
        palm = n[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]].mean(axis=0)
        self.last = HandFeatures(
            label=hand.label, fresh=True, hand=hand, size=size, fingers=fingers,
            ext_count=int(sum(fingers[1:])), thumb_ratio=thumb_ratio,
            idx_ratio=idx_ratio, mid_ratio=mid_ratio,
            index_pinch=self._idx_pinch, middle_pinch=self._mid_pinch,
            raw_pose=raw, pose=self._pose,
            tip=(float(n[INDEX_TIP][0]), float(n[INDEX_TIP][1])),
            pinch_mid=(float(mid[0]), float(mid[1])), palm=(float(palm[0]), float(palm[1])),
            scroll_y=float((n[INDEX_TIP][1] + n[MIDDLE_TIP][1]) / 2.0),
            index_vec=(float(p[INDEX_TIP][0] - p[INDEX_MCP][0]), float(p[INDEX_TIP][1] - p[INDEX_MCP][1])),
            click_pinch=self._c_idx, right_pinch=self._c_mid, speed=self._speed,
            pinch_now=idx_ratio < cfg.CLICK_PINCH_OFF)
        return self.last


# =============================================================== one-hand gestures
class OneHandGestures:
    """
    CURSOR (Apple-Vision-Pro style): the cursor moves ONLY while index + thumb are pinched.
        pinch + move the hand   cursor moves (relative: continues from where it stopped, no jump)
        release the pinch       cursor stays exactly where it is
    CLICKS (strict "click-grade" pinch, see HandAnalyzer)
        quick pinch tap, hand still        left click   (second tap within DOUBLE_CLICK_WINDOW_S = double click)
        middle finger + thumb tap          right click
    OTHER
        scroll        PEACE pose (index+middle up), move the hand up / down
        drag & drop   close a FIST to grab (mouse down) ... open the hand to drop (mouse up)
    """

    IDLE, PINCH_IDX, PINCH_MID, DRAG, SCROLL = range(5)

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = self.IDLE
        self.cursor_active = False           # True while this hand is driving the cursor
        self._pinch_t0 = 0.0
        self._pinch_start_px = None
        self._prev_pt: Optional[Tuple[float, float]] = None
        self._moved = False                  # pinch has travelled far enough to be a cursor move, not a tap
        self._last_click_t = -1e9            # last single click (double-click window)
        self._last_click_emit = -1e9         # last click of any kind (anti double-fire)
        self._last_release_t = -1e9
        self._last_rc_t = -1e9
        self._settle_until = 0.0
        self._scroll_y: Optional[float] = None
        self._prev_idx = self._prev_mid = False
        self._fist_t0: Optional[float] = None
        self._fist_ok = False
        self._last_active_t = -1e9           # last time the hand was NOT a fist (grab precondition)
        self._open_n = 0
        self._prev_palm: Optional[Tuple[float, float]] = None

    @property
    def name(self) -> str:
        return {self.IDLE: "READY", self.PINCH_IDX: "MOVE" if self._moved else "PINCH",
                self.PINCH_MID: "RIGHT-CLICK", self.DRAG: "DRAG", self.SCROLL: "SCROLL"}[self.state]

    def abort(self) -> List[Event]:
        """Hand lost or a two-hand mode took over: release anything held."""
        ev = [Event("drag_end")] if self.state == self.DRAG else []
        self.reset()
        return ev

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _palm_px(f: HandFeatures) -> np.ndarray:
        """Palm centre in pixels.  The cursor follows the PALM, not the fingertips: opening or closing
        the pinch changes the fingertip positions but leaves the palm still, so a tap can never move
        the cursor."""
        return f.hand.pts[PALM_POINTS].mean(axis=0)

    def _click_event(self, now: float) -> Optional[Event]:
        if now - self._last_click_emit < cfg.CLICK_MIN_INTERVAL_S:        # anti double-fire
            return None
        self._last_click_emit = now
        if now - self._last_click_t <= cfg.DOUBLE_CLICK_WINDOW_S:
            self._last_click_t = -1e9
            return Event("double_click")
        self._last_click_t = now
        return Event("left_click")

    # ------------------------------------------------------------ per frame
    def update(self, f: HandFeatures, now: float, quiet: bool = False) -> List[Event]:
        """quiet=True while a two-hand gesture is being confirmed: nothing new may start."""
        ev: List[Event] = []
        if not f.fresh:                      # occluded: freeze state (an active drag stays held)
            return ev
        self.cursor_active = False
        if f.raw_pose != "FIST":
            self._last_active_t = now
        idx_edge = f.click_pinch and not self._prev_idx
        mid_edge = f.right_pinch and not self._prev_mid
        self._prev_idx, self._prev_mid = f.click_pinch, f.right_pinch

        if self.state == self.DRAG:
            ev = self._drag(f, now)
        elif self.state == self.PINCH_IDX:
            ev = self._pinch_idx(f, now, quiet)
        elif self.state == self.PINCH_MID:
            ev = self._pinch_mid(f, now)
        elif self.state == self.SCROLL:
            ev = self._scroll(f)
        else:
            if quiet:
                self._fist_t0 = None
                return ev
            ev = self._idle(f, now, idx_edge, mid_edge)
        if quiet:                            # swallow everything except releasing a drag
            ev = [e for e in ev if e.kind == "drag_end"]
        return ev

    # ---- idle: start something
    def _idle(self, f: HandFeatures, now: float, idx_edge: bool, mid_edge: bool) -> List[Event]:
        rearmed = now - self._last_release_t >= cfg.PINCH_REARM_S
        if idx_edge and rearmed:                                   # pinch begins: tap or cursor move?
            self.state, self._pinch_t0, self._moved = self.PINCH_IDX, now, False
            self._pinch_start_px, self._prev_pt = self._palm_px(f), f.palm
            self._fist_t0 = None
            return []
        if mid_edge and rearmed and not f.click_pinch:             # middle pinch: right-click tap
            self.state, self._pinch_t0 = self.PINCH_MID, now
            self._fist_t0 = None
            return []
        if f.pose == "PEACE":
            self.state, self._scroll_y = self.SCROLL, f.scroll_y
            self._fist_t0 = None
            return []
        if f.pose == "FIST" and now >= self._settle_until:         # fist held long enough = grab
            if self._fist_t0 is None:
                self._fist_t0 = now
                self._fist_ok = (now - self._last_active_t) <= cfg.GRAB_REQUIRES_OPEN_WITHIN_S
            elif self._fist_ok and now - self._fist_t0 >= cfg.GRAB_HOLD_S:
                return self._start_drag(f)
            return []
        self._fist_t0 = None
        return []

    # ---- pinch with the index finger: still = click, moving = cursor
    def _pinch_idx(self, f: HandFeatures, now: float, quiet: bool) -> List[Event]:
        ev: List[Event] = []
        held = now - self._pinch_t0
        if f.click_pinch:
            if not self._moved:
                travel = float(np.linalg.norm(self._palm_px(f) - self._pinch_start_px)) / f.hand.frame_w
                if travel >= cfg.PINCH_MOVE_START_DIST and not quiet:
                    self._moved = True                       # breakaway: from here on the cursor follows
                    self._prev_pt = f.palm
                    ev.append(Event("track_start"))
            else:
                if f.pinch_now:          # the instant the fingers open the cursor stops (no confirm lag)
                    ev.append(Event("move_rel", f.palm[0] - self._prev_pt[0], f.palm[1] - self._prev_pt[1]))
                self._prev_pt = f.palm
            self.cursor_active = self._moved and f.pinch_now
            return ev
        self.state, self._last_release_t = self.IDLE, now          # pinch released: cursor stops here
        self._settle_until = now + cfg.POST_CLICK_FREEZE_S
        if not self._moved and cfg.TAP_MIN_S <= held <= cfg.TAP_MAX_S:
            click = self._click_event(now)
            if click is not None:
                ev.append(click)
        return ev

    # ---- pinch with the middle finger: right-click tap (the cursor never moves)
    def _pinch_mid(self, f: HandFeatures, now: float) -> List[Event]:
        ev: List[Event] = []
        held = now - self._pinch_t0
        if f.click_pinch:                                          # index took over: cancel
            self.state, self._last_release_t = self.IDLE, now
            return ev
        if f.right_pinch:
            return ev
        self.state, self._last_release_t = self.IDLE, now
        self._settle_until = now + cfg.POST_CLICK_FREEZE_S
        if (cfg.TAP_MIN_S <= held <= cfg.RIGHT_CLICK_MAX_S
                and now - self._last_rc_t >= cfg.RIGHT_CLICK_COOLDOWN_S
                and now - self._last_click_emit >= cfg.CLICK_MIN_INTERVAL_S):
            self._last_rc_t = self._last_click_emit = now
            ev.append(Event("right_click"))
        return ev

    # ---- scroll
    def _scroll(self, f: HandFeatures) -> List[Event]:
        ev: List[Event] = []
        if f.pose != "PEACE":
            self.state, self._scroll_y = self.IDLE, None
            return ev
        if self._scroll_y is not None:
            dy = f.scroll_y - self._scroll_y
            if cfg.SCROLL_DEADZONE < abs(dy) < cfg.SCROLL_MAX_JUMP:
                notches = -dy * cfg.SCROLL_GAIN
                ev.append(Event("scroll", value=-notches if cfg.SCROLL_INVERT else notches))
        self._scroll_y = f.scroll_y
        return ev

    # ---- drag & drop: fist = grab at the current cursor, open hand = drop
    def _start_drag(self, f: HandFeatures) -> List[Event]:
        self.state, self._open_n, self._fist_t0 = self.DRAG, 0, None
        self._prev_palm = f.palm
        return [Event("track_start"), Event("drag_start")]

    def _drag(self, f: HandFeatures, now: float) -> List[Event]:
        ev: List[Event] = []
        self._open_n = self._open_n + 1 if f.ext_count >= cfg.DROP_MIN_FINGERS else 0
        if self._open_n >= cfg.DROP_CONFIRM_FRAMES:                # hand opened: drop
            ev.append(Event("drag_end"))
            self.state, self._last_release_t = self.IDLE, now
            self._settle_until = now + cfg.POST_CLICK_FREEZE_S
            return ev
        ev.append(Event("move_rel", f.palm[0] - self._prev_palm[0], f.palm[1] - self._prev_palm[1]))
        self._prev_palm = f.palm
        self.cursor_active = True
        return ev


# =============================================================== two-hand gestures
class TwoHandGestures:
    """
    Continuous modes (hold the pose to engage, Left = modifier hand, Right = primary by default):
        ZOOM        both hands PINCH, move them apart / together
        WINDOW      modifier FIST (holds Alt) + primary OPEN palm: push right = next, left = previous
        VOLUME      modifier OPEN  + primary PINCH, move up / down
        BRIGHTNESS  modifier PEACE + primary PINCH, move up / down
    One-shot gestures (hold, fire once, must be released before they can fire again):
        PAUSE       both FISTS held close together           -> toggle_pause
        MEDIA       both index fingers up, side by side "||"  -> media_play_pause
        MACRO       pose pairs from config.MACROS            -> macro
    """

    ONE_SHOT = ("PAUSE", "MEDIA")
    MODE_POSE = {"WINDOW": "FIST", "VOLUME": "OPEN", "BRIGHTNESS": "PEACE"}

    def __init__(self):
        self.reset()

    def reset(self):
        self.mode: Optional[str] = None
        self.progress: Optional[Tuple[str, float]] = None      # (label, 0..1) of the pose being held
        self.level_total = 0.0
        self._pending: Optional[str] = None
        self._pending_t = 0.0
        self._release_t: Optional[float] = None
        self._fired: Optional[str] = None
        self._last_fire_t: Dict[str, float] = {}
        self._zoom_last: Optional[float] = None
        self._zoom_acc = 0.0
        self._win_anchor: Optional[float] = None
        self._win_armed = True
        self._win_next_t = 0.0
        self._level_val: Optional[float] = None
        self._level_anchor: Optional[float] = None

    @property
    def active(self) -> bool:
        return self.mode is not None

    @property
    def pending(self) -> bool:
        return self._pending is not None

    @property
    def holding(self) -> bool:
        """True while a two-hand pose is being confirmed OR a one-shot one is still held after firing."""
        return self._pending is not None or self._fired is not None

    def force_exit(self) -> List[Event]:
        ev = [Event("window_end")] if self.mode == "WINDOW" else []
        self.reset()
        return ev

    # ------------------------------------------------------------ geometry helpers
    @staticmethod
    def _close(a, b, gap) -> bool:
        return abs(a[0] - b[0]) <= gap[0] and abs(a[1] - b[1]) <= gap[1]

    @staticmethod
    def _vertical(f: HandFeatures) -> bool:
        vx, vy = f.index_vec
        return vy < 0 and abs(vx) <= abs(vy) * math.tan(math.radians(cfg.MEDIA_MAX_TILT_DEG))

    @staticmethod
    def _dist(a: HandFeatures, b: HandFeatures) -> float:
        pa = (a.hand.pts[THUMB_TIP] + a.hand.pts[INDEX_TIP]) / 2.0
        pb = (b.hand.pts[THUMB_TIP] + b.hand.pts[INDEX_TIP]) / 2.0
        return float(np.linalg.norm(pa - pb)) / a.hand.frame_w

    def _candidate(self, left, right, mod, ctl, paused) -> Optional[Tuple[str, str, float]]:
        """-> (key, label, hold seconds) of the pose pair currently shown, if any."""
        if (cfg.ENABLE_PAUSE_GESTURE and left.pose == "FIST" and right.pose == "FIST"
                and self._close(left.palm, right.palm, cfg.PAUSE_MAX_GAP)):
            return "PAUSE", "PAUSE / RESUME", cfg.PAUSE_HOLD_S
        if paused:                                       # while paused only the resume gesture works
            return None
        if (cfg.ENABLE_MEDIA_GESTURE and left.pose == "POINT" and right.pose == "POINT"
                and self._close(left.tip, right.tip, cfg.MEDIA_MAX_GAP)
                and self._vertical(left) and self._vertical(right)):
            return "MEDIA", "PLAY / PAUSE", cfg.MEDIA_HOLD_S
        if cfg.ENABLE_MACROS and (left.pose, right.pose) in cfg.MACROS:
            return (f"MACRO:{left.pose}+{right.pose}", cfg.MACROS[(left.pose, right.pose)][0].upper(),
                    cfg.MACRO_HOLD_S)
        if cfg.ENABLE_ZOOM and left.pose == "PINCH" and right.pose == "PINCH":
            return "ZOOM", "ZOOM", cfg.TWO_HAND_ENGAGE_S
        if cfg.ENABLE_WINDOW_SWITCH and mod.pose == "FIST" and ctl.pose == "OPEN":
            return "WINDOW", "WINDOW SWITCH", cfg.WINDOW_ENGAGE_S
        if cfg.ENABLE_VOLUME and mod.pose == "OPEN" and ctl.pose in cfg.LEVEL_CONTROLLER_POSES:
            return "VOLUME", "VOLUME", cfg.TWO_HAND_ENGAGE_S
        if cfg.ENABLE_BRIGHTNESS and mod.pose == "PEACE" and ctl.pose in cfg.LEVEL_CONTROLLER_POSES:
            return "BRIGHTNESS", "BRIGHTNESS", cfg.TWO_HAND_ENGAGE_S
        return None

    def _still_valid(self, left, right, mod) -> bool:
        if self.mode == "ZOOM":
            return left is not None and right is not None and left.index_pinch and right.index_pinch
        return mod is not None and mod.pose == self.MODE_POSE[self.mode]

    def _engage(self, key: str) -> None:
        self.mode, self._release_t = key, None
        self._zoom_last, self._zoom_acc = None, 0.0
        self._win_anchor, self._win_armed, self._win_next_t = None, True, 0.0
        self._level_val = self._level_anchor = None
        self.level_total = 0.0

    # ------------------------------------------------------------ per-frame update
    def update(self, feats: Dict[str, HandFeatures], now: float, paused: bool = False) -> List[Event]:
        ev: List[Event] = []
        self.progress = None
        left, right = feats.get("Left"), feats.get("Right")
        mod, ctl = feats.get(cfg.MODIFIER_HAND), feats.get(cfg.PRIMARY_HAND)

        # ---------- not in a mode: look for a pose pair to hold
        if self.mode is None:
            cand = None
            if left is not None and right is not None and left.fresh and right.fresh:
                cand = self._candidate(left, right, mod, ctl, paused)
            key = cand[0] if cand else None
            if key != self._fired:
                self._fired = None
            if cand is None or key == self._fired:
                self._pending = None
                return ev
            if key != self._pending:
                self._pending, self._pending_t = key, now
            frac = (now - self._pending_t) / max(cand[2], 1e-6)
            self.progress = (cand[1], min(1.0, frac))
            if frac < 1.0:
                return ev
            self._pending = None
            if key in self.ONE_SHOT or key.startswith("MACRO:"):
                self._fired = key                         # fire once per hold, re-arm after release
                if now - self._last_fire_t.get(key, -1e9) >= cfg.ONE_SHOT_COOLDOWN_S:
                    self._last_fire_t[key] = now
                    if key == "PAUSE":
                        ev.append(Event("toggle_pause"))
                    elif key == "MEDIA":
                        ev.append(Event("media_play_pause"))
                    else:
                        label, keys = cfg.MACROS[(left.pose, right.pose)]
                        ev.append(Event("macro", data=(label, list(keys))))
            else:
                self._engage(key)
            return ev

        # ---------- in a mode: stay while the pose holds (with a flicker grace period)
        if paused or not self._still_valid(left, right, mod):
            if self._release_t is None:
                self._release_t = now
            if paused or now - self._release_t > cfg.TWO_HAND_RELEASE_GRACE_S:
                if self.mode == "WINDOW":
                    ev.append(Event("window_end"))
                self.mode, self._pending, self._release_t = None, None, None
            return ev
        self._release_t = None

        if self.mode == "ZOOM":
            if left.fresh and right.fresh:
                dist = self._dist(left, right)
                if self._zoom_last is not None:
                    self._zoom_acc += dist - self._zoom_last
                self._zoom_last = dist
                steps = int(self._zoom_acc / cfg.ZOOM_STEP_NORM)
                if steps:
                    self._zoom_acc -= steps * cfg.ZOOM_STEP_NORM
                    ev.append(Event("zoom", value=-steps if cfg.ZOOM_INVERT else steps))
        elif self.mode == "WINDOW":
            if ctl is not None and ctl.fresh:
                ev += self._window_step(ctl.palm[0], now)
            else:
                self._win_anchor = None
        else:                                            # VOLUME / BRIGHTNESS
            ev += self._level(ctl)
        return ev

    # ---- joystick-style window stepping
    def _window_step(self, x: float, now: float) -> List[Event]:
        if self._win_anchor is None:
            self._win_anchor, self._win_armed = x, True
            return []
        dx = x - self._win_anchor
        if cfg.WINDOW_INVERT:
            dx = -dx
        kind = "window_next" if dx > 0 else "window_prev"
        if self._win_armed:
            if abs(dx) >= cfg.WINDOW_PUSH_DX:
                self._win_armed = False
                self._win_next_t = now + (cfg.WINDOW_REPEAT_DELAY_S if cfg.WINDOW_REPEAT_DELAY_S > 0 else 1e9)
                return [Event(kind)]
            return []
        if abs(dx) < cfg.WINDOW_REARM_DX:                 # back near the centre: ready for the next step
            self._win_armed, self._win_anchor = True, x
        elif abs(dx) >= cfg.WINDOW_PUSH_DX and now >= self._win_next_t:
            self._win_next_t = now + cfg.WINDOW_REPEAT_S  # still pushing: auto-repeat
            return [Event(kind)]
        return []

    # ---- relative volume / brightness
    def _level(self, ctl: Optional[HandFeatures]) -> List[Event]:
        if ctl is None or not ctl.fresh:
            self._level_anchor = None
            return []
        if cfg.LEVEL_SOURCE == "pinch":
            value = float(np.clip((ctl.idx_ratio - 0.15) / 0.85, 0.0, 1.0))
        else:
            value = 1.0 - ctl.palm[1]                    # higher hand = larger value
        self._level_val = value if self._level_val is None else 0.5 * value + 0.5 * self._level_val
        if self._level_anchor is None:
            self._level_anchor = self._level_val
            return []
        delta = self._level_val - self._level_anchor
        if abs(delta) < cfg.LEVEL_DEADZONE:
            return []
        effective = delta - math.copysign(cfg.LEVEL_DEADZONE, delta)
        steps = int(effective * 100.0 * cfg.LEVEL_GAIN / cfg.LEVEL_STEP_PERCENT)
        if steps == 0:
            return []
        amount = steps * cfg.LEVEL_STEP_PERCENT
        self._level_anchor += amount / (100.0 * cfg.LEVEL_GAIN)
        self.level_total += amount
        return [Event("level", value=float(amount), data=self.mode.lower())]


# =============================================================== engine
class GestureEngine:
    """Entry point: engine.update(hands, now, paused) -> list of Events."""

    def __init__(self):
        self.analyzers = {"Left": HandAnalyzer(), "Right": HandAnalyzer()}
        self.one = OneHandGestures()
        self.two = TwoHandGestures()
        self.features: Dict[str, HandFeatures] = {}
        self._primary_label: Optional[str] = None

    def reset(self):
        for a in self.analyzers.values():
            a.reset()
        self.one.reset()
        self.two.reset()
        self.features = {}

    @property
    def mode_name(self) -> str:
        return self.two.mode if self.two.active else self.one.name

    @property
    def progress(self) -> Optional[Tuple[str, float]]:
        """(label, 0..1) of the two-hand pose being held, for the on-screen progress bar."""
        return self.two.progress

    def primary(self) -> Optional[HandFeatures]:
        p = self.features.get(cfg.PRIMARY_HAND)
        if p is not None:
            return p
        return next(iter(self.features.values())) if len(self.features) == 1 else None

    def update(self, hands: Dict[str, Hand], now: float, paused: bool = False) -> List[Event]:
        feats: Dict[str, HandFeatures] = {}
        for label in ("Left", "Right"):
            hand = hands.get(label)
            if hand is None:
                self.analyzers[label].reset()
            else:
                feats[label] = self.analyzers[label].update(hand)
        self.features = feats
        events: List[Event] = []

        if paused:                                        # only the resume gesture is alive
            events += self.two.force_exit() if self.two.active else []
            events += self.one.abort()
            if cfg.ENABLE_TWO_HAND:
                events += self.two.update(feats, now, paused=True)
            return events

        if cfg.ENABLE_TWO_HAND:
            events += self.two.update(feats, now)

        if self.two.active:                               # a two-hand mode owns both hands
            events += self.one.abort()
            self._primary_label = None
            return events

        prim = self.primary()
        if prim is None:
            events += self.one.abort()
            self._primary_label = None
            return events
        if self._primary_label not in (None, prim.label):
            events += self.one.abort()                    # the controlling hand changed
        self._primary_label = prim.label
        events += self.one.update(prim, now, quiet=self.two.holding)
        return events
