"""
selftest.py  -  webcam-free sanity checks.   Run:  python selftest.py

Builds synthetic hands (landmark arrays), feeds them through the real tracker / gesture engine /
smoother / calibration / controller (dry-run) and asserts the expected events come out.
No camera, no MediaPipe model and no mouse control are involved.
"""
import os
import sys
import tempfile

import numpy as np

import config as cfg
from calibration import Calibration, CalibrationWizard
from filters import CursorSmoother
from gesture_recognizer import Event, GestureEngine
from hand_tracker import Hand, HandTracker
from mouse_controller import MouseController

W, H, DT = 640, 480, 1.0 / 30.0

# open right hand, fingers up, pixel coordinates (wrist = 0, thumb 1-4, index 5-8, ... pinky 17-20)
OPEN = np.array([
    (320, 420), (290, 400), (265, 370), (250, 340), (235, 310),
    (295, 330), (292, 290), (289, 260), (287, 230),
    (320, 320), (320, 275), (320, 242), (320, 210),
    (345, 330), (348, 290), (350, 262), (352, 235),
    (368, 345), (376, 315), (381, 292), (385, 270)], dtype=float)
FINGERS = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
POSES = {"POINT": (1, 0, 0, 0), "PEACE": (1, 1, 0, 0), "OPEN": (1, 1, 1, 1), "FIST": (0, 0, 0, 0),
         "PINCH": (1, 0, 0, 0), "MIDDLE_PINCH": (0, 1, 0, 0)}
BASE_CENTER = OPEN[[0, 5, 9, 13, 17]].mean(axis=0)


def make_hand(label, pose, pos=(320, 300), stale=False, t=0.0):
    pts = OPEN.copy()
    for ext, (mcp, pip, dip, tip) in zip(POSES[pose], FINGERS):
        if not ext:
            mx, my = pts[mcp]
            pts[pip], pts[dip], pts[tip] = (mx, my - 35), (mx, my - 15), (mx, my + 5)
    if pose != "OPEN":
        pts[3], pts[4] = (300, 375), (335, 360)            # tuck the thumb
    if pose == "PINCH":
        pts[4] = pts[8] + (3, 3)
    if pose == "MIDDLE_PINCH":
        pts[4] = pts[12] + (3, 3)
    pts += np.array(pos, dtype=float) - BASE_CENTER
    return Hand(label, 0.95, pts, pts / [W, H], W, H, t, stale)


class Sim:
    """Feeds frames to a GestureEngine at 30 FPS and collects events."""

    def __init__(self):
        self.engine, self.t, self.events = GestureEngine(), 0.0, []

    def run(self, frames, make, paused=False):
        for i in range(frames):
            self.events += self.engine.update(make(i, self.t), self.t, paused=paused)
            self.t += DT

    def kinds(self):
        return [e.kind for e in self.events if e.kind != "move_rel"]

    def moves(self):
        return [e for e in self.events if e.kind == "move_rel"]


def noisy(h, amp, rng):
    h.pts = h.pts + rng.normal(0, amp, h.pts.shape)
    h.npts = h.pts / [W, H]
    return h


def R(pose, pos=(320, 300), **kw):
    return lambda i, t: {"Right": make_hand("Right", pose, pos if not callable(pos) else pos(i), t=t, **kw)}


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# ------------------------------------------------------------------ one hand
@check
def cursor_is_stationary_without_pinch():
    s = Sim()
    for pose in ("POINT", "OPEN", "PEACE" if False else "POINT"):
        s.run(30, lambda i, t: {"Right": make_hand("Right", pose, (150 + 8 * i, 240 + 3 * i), t=t)})
    assert s.events == [], s.events                                   # pointing / waving never moves it


@check
def cursor_moves_only_while_pinched():
    s = Sim()
    s.run(10, R("POINT"))                                             # fingers open first (arms the pinch)
    s.run(24, lambda i, t: {"Right": make_hand("Right", "PINCH", (300 + 5 * i, 300), t=t)})
    assert s.kinds() == ["track_start"], s.kinds()
    total = sum(e.x for e in s.moves())
    assert 0.10 < total < 0.17, total                                 # follows the hand (minus the break-away dead travel)
    assert all(e.x > 0 and abs(e.y) < 1e-6 for e in s.moves())
    n = len(s.moves())
    s.run(20, lambda i, t: {"Right": make_hand("Right", "POINT", (420 + 6 * i, 300), t=t)})
    assert len(s.moves()) == n, "cursor must stay still after the pinch is released"
    assert s.kinds() == ["track_start"], s.kinds()                    # moving is never a click


@check
def tap_clicks_without_moving_cursor():
    s = Sim()
    s.run(10, R("POINT"))
    s.run(6, R("PINCH"))
    s.run(6, R("POINT"))
    assert s.kinds() == ["left_click"] and s.moves() == [], (s.kinds(), s.moves())


@check
def hand_tremor_does_not_move_cursor_or_spoil_click():
    rng = np.random.default_rng(3)
    s = Sim()
    s.run(10, lambda i, t: {"Right": noisy(make_hand("Right", "POINT", t=t), 1.2, rng)})
    s.run(7, lambda i, t: {"Right": noisy(make_hand("Right", "PINCH", t=t), 1.2, rng)})
    s.run(6, lambda i, t: {"Right": noisy(make_hand("Right", "POINT", t=t), 1.2, rng)})
    assert s.kinds() == ["left_click"] and s.moves() == [], (s.kinds(), len(s.moves()))


@check
def long_still_pinch_is_neither_click_nor_toggle():
    s = Sim()
    s.run(10, R("POINT"))
    s.run(24, R("PINCH"))                                             # 0.8 s hold, no movement
    s.run(6, R("POINT"))
    assert s.events == [], s.kinds()                                  # no click, no pointer on/off any more


@check
def double_click():
    s = Sim()
    s.run(10, R("POINT"))
    for _ in range(2):
        s.run(6, R("PINCH"))
        s.run(4, R("POINT"))
    s.run(6, R("POINT"))
    assert s.kinds() == ["left_click", "double_click"] and s.moves() == [], s.kinds()


@check
def right_click():
    s = Sim()
    s.run(10, R("POINT"))
    s.run(8, R("MIDDLE_PINCH"))
    s.run(6, R("POINT"))
    assert s.kinds() == ["right_click"] and s.moves() == [], s.kinds()


@check
def right_click_needs_a_deliberate_pinch():
    s = Sim()
    s.run(10, R("POINT"))
    s.run(3, R("MIDDLE_PINCH"))                                       # too brief
    s.run(8, R("POINT"))
    s.run(22, R("MIDDLE_PINCH"))                                      # too long (> 0.5 s)
    s.run(8, R("POINT"))
    assert s.kinds() == [], s.kinds()


def _both_tips_near_thumb(t):
    h = make_hand("Right", "PINCH", t=t)
    h.pts[12] = h.pts[4] + (8, 4)                                     # middle tip touches the thumb too
    h.npts = h.pts / [W, H]
    return h


@check
def ambiguous_pinch_triggers_nothing():
    s = Sim()
    s.run(10, R("POINT"))
    s.run(8, lambda i, t: {"Right": _both_tips_near_thumb(t)})
    s.run(6, R("POINT"))
    assert s.kinds() == [] and s.moves() == [], s.kinds()             # neither left nor right click


@check
def no_click_while_hand_moves_fast():
    s = Sim()
    s.run(8, lambda i, t: {"Right": make_hand("Right", "POINT", (100 + 40 * i, 300), t=t)})
    s.run(6, lambda i, t: {"Right": make_hand("Right", "PINCH", (420 + 40 * i, 300), t=t)})
    s.run(6, lambda i, t: {"Right": make_hand("Right", "POINT", (660 + 40 * i, 300), t=t)})
    assert s.kinds() == [] and s.moves() == [], s.kinds()


@check
def hand_that_appears_pinching_cannot_click():
    s = Sim()
    s.run(10, R("PINCH"))                                             # fingers never seen open first
    s.run(6, R("POINT"))
    assert s.kinds() == [] and s.moves() == [], s.kinds()


@check
def flicker_does_not_misfire():
    s = Sim()
    s.run(10, R("POINT"))
    s.run(1, R("PINCH"))                                              # one-frame glitch
    s.run(8, R("POINT"))
    assert s.kinds() == [], s.kinds()
    s.run(4, R("PINCH"))                                              # one-frame release glitch inside a pinch
    s.run(1, R("POINT"))
    s.run(4, R("PINCH"))
    s.run(8, R("POINT"))
    assert s.kinds() == ["left_click"], s.kinds()


@check
def fist_grabs_and_open_hand_drops():
    s = Sim()
    s.run(10, R("POINT", (300, 300)))
    s.run(20, lambda i, t: {"Right": make_hand("Right", "FIST", (300 + 4 * i, 300), t=t)})
    assert s.kinds() == ["track_start", "drag_start"], s.kinds()      # grabs where the cursor is: no jump
    assert sum(e.x for e in s.moves()) > 0.05                         # cursor follows the fist (relative)
    s.run(6, lambda i, t: {"Right": make_hand("Right", "OPEN", (376, 300), t=t)})
    assert s.kinds() == ["track_start", "drag_start", "drag_end"], s.kinds()
    n = len(s.moves())
    s.run(10, lambda i, t: {"Right": make_hand("Right", "OPEN", (376 + 8 * i, 300), t=t)})
    assert len(s.moves()) == n, "nothing moves after the drop"


@check
def fist_that_was_never_open_cannot_grab():
    s = Sim()
    s.run(40, R("FIST"))                                              # resting fist / fist entering the picture
    assert s.events == [], s.kinds()


@check
def scroll_up():
    s = Sim()
    s.run(8, R("PEACE"))
    s.run(15, lambda i, t: {"Right": make_hand("Right", "PEACE", (320, 300 - 4 * i), t=t)})
    vals = [e.value for e in s.events if e.kind == "scroll"]
    assert vals and all(v > 0 for v in vals), vals


@check
def occlusion_keeps_drag_then_releases():
    s = Sim()
    s.run(10, R("POINT"))
    s.run(14, R("FIST"))
    assert s.kinds() == ["track_start", "drag_start"], s.kinds()
    s.run(5, R("FIST", stale=True))                                   # tracking blip
    assert s.kinds() == ["track_start", "drag_start"], s.kinds()
    s.run(2, lambda i, t: {})                                         # hand gone
    assert s.kinds() == ["track_start", "drag_start", "drag_end"], s.kinds()


# ------------------------------------------------------------------ two hands
@check
def zoom_in_without_stray_events():
    s = Sim()
    s.run(14, lambda i, t: {"Left": make_hand("Left", "PINCH", (200, 300), t=t),
                            "Right": make_hand("Right", "PINCH", (440, 300), t=t)})
    s.run(15, lambda i, t: {"Left": make_hand("Left", "PINCH", (200 - 6 * i, 300), t=t),
                            "Right": make_hand("Right", "PINCH", (440 + 6 * i, 300), t=t)})
    zooms = [e.value for e in s.events if e.kind == "zoom"]
    assert zooms and sum(zooms) >= 4, zooms
    assert set(s.kinds()) == {"zoom"}, s.kinds()                      # no click, no cursor movement


@check
def zoom_after_open_hands_has_no_stray_cursor_events():
    s = Sim()
    s.run(8, lambda i, t: {"Left": make_hand("Left", "POINT", (200, 300), t=t),
                           "Right": make_hand("Right", "POINT", (440, 300), t=t)})
    s.run(18, lambda i, t: {"Left": make_hand("Left", "PINCH", (200, 300), t=t),
                            "Right": make_hand("Right", "PINCH", (440, 300), t=t)})
    s.run(15, lambda i, t: {"Left": make_hand("Left", "PINCH", (200 - 6 * i, 300), t=t),
                            "Right": make_hand("Right", "PINCH", (440 + 6 * i, 300), t=t)})
    assert set(s.kinds()) == {"zoom"} and s.moves() == [], s.kinds()


@check
def cursor_stays_still_while_two_hand_pose_is_confirmed():
    s = Sim()
    s.run(8, lambda i, t: {"Left": make_hand("Left", "POINT", (150, 300), t=t),
                           "Right": make_hand("Right", "POINT", (480, 300), t=t)})
    s.run(30, lambda i, t: {"Left": make_hand("Left", "OPEN", (150, 300), t=t),
                            "Right": make_hand("Right", "PINCH", (480 + 3 * i, 300), t=t)})
    assert s.moves() == [] and "track_start" not in s.kinds(), s.kinds()   # volume gesture owns the hands


@check
def window_switch_joystick():
    s = Sim()
    left = lambda t: make_hand("Left", "FIST", (150, 300), t=t)
    s.run(18, lambda i, t: {"Left": left(t), "Right": make_hand("Right", "OPEN", (400, 240), t=t)})
    s.run(10, lambda i, t: {"Left": left(t), "Right": make_hand("Right", "OPEN", (400 + 14 * (i + 1), 240), t=t)})
    s.run(6, lambda i, t: {"Left": left(t), "Right": make_hand("Right", "OPEN", (540 - 25 * (i + 1), 240), t=t)})
    s.run(10, lambda i, t: {"Left": left(t), "Right": make_hand("Right", "OPEN", (400 - 14 * (i + 1), 240), t=t)})
    assert s.kinds() == ["window_next", "window_prev"], s.kinds()
    s.run(14, lambda i, t: {"Right": make_hand("Right", "OPEN", (300, 240), t=t)})   # fist hand leaves
    assert s.kinds() == ["window_next", "window_prev", "window_end"], s.kinds()


@check
def volume_and_brightness_with_pinch_knob():
    for mod_pose, name in (("OPEN", "volume"), ("PEACE", "brightness")):
        s = Sim()
        s.run(18, lambda i, t: {"Left": make_hand("Left", mod_pose, (150, 300), t=t),
                                "Right": make_hand("Right", "PINCH", (480, 300), t=t)})
        s.run(25, lambda i, t: {"Left": make_hand("Left", mod_pose, (150, 300), t=t),
                                "Right": make_hand("Right", "PINCH", (480, 300 - 4 * i), t=t)})
        levels = [e for e in s.events if e.kind == "level"]
        assert levels and levels[0].data == name and sum(e.value for e in levels) >= 20, (name, levels)
        assert s.engine.two.level_total == sum(e.value for e in levels)


@check
def volume_ignores_small_drift():
    s = Sim()
    s.run(18, lambda i, t: {"Left": make_hand("Left", "OPEN", (150, 300), t=t),
                            "Right": make_hand("Right", "PINCH", (480, 300), t=t)})
    s.run(20, lambda i, t: {"Left": make_hand("Left", "OPEN", (150, 300), t=t),
                            "Right": make_hand("Right", "PINCH", (480, 300 - 0.15 * i), t=t)})
    assert [e for e in s.events if e.kind == "level"] == []


@check
def macros_fire_once():
    for pose, label in (("OPEN", "Show desktop"), ("PEACE", "Screenshot tool")):
        s = Sim()
        s.run(40, lambda i, t: {"Left": make_hand("Left", pose, (150, 300), t=t),
                                "Right": make_hand("Right", pose, (480, 300), t=t)})
        assert s.kinds() == ["macro"] and s.events[-1].data[0] == label, s.kinds()


@check
def pause_needs_two_close_fists():
    far = Sim()
    far.run(60, lambda i, t: {"Left": make_hand("Left", "FIST", (120, 300), t=t),
                              "Right": make_hand("Right", "FIST", (520, 300), t=t)})
    assert "toggle_pause" not in far.kinds(), far.kinds()             # fists far apart: no pause
    near = Sim()
    near.run(6, lambda i, t: {"Left": make_hand("Left", "OPEN", (250, 300), t=t),
                              "Right": make_hand("Right", "OPEN", (390, 300), t=t)})
    near.run(45, lambda i, t: {"Left": make_hand("Left", "FIST", (250, 300), t=t),
                               "Right": make_hand("Right", "FIST", (390, 300), t=t)})
    assert near.kinds() == ["toggle_pause"], near.kinds()             # and no accidental grab meanwhile


@check
def paused_engine_only_listens_for_resume():
    s = Sim()
    s.run(40, lambda i, t: {"Left": make_hand("Left", "OPEN", (150, 300), t=t),
                            "Right": make_hand("Right", "OPEN", (480, 300), t=t)}, paused=True)
    assert s.kinds() == [], s.kinds()                                 # macro ignored while paused
    s.run(45, lambda i, t: {"Left": make_hand("Left", "FIST", (250, 300), t=t),
                            "Right": make_hand("Right", "FIST", (390, 300), t=t)}, paused=True)
    assert s.kinds() == ["toggle_pause"], s.kinds()


@check
def media_play_pause_needs_parallel_fingers():
    s = Sim()
    s.run(36, lambda i, t: {"Left": make_hand("Left", "POINT", (250, 300), t=t),
                            "Right": make_hand("Right", "POINT", (390, 300), t=t)})
    assert s.kinds() == ["media_play_pause"], s.kinds()
    far = Sim()
    far.run(45, lambda i, t: {"Left": make_hand("Left", "POINT", (120, 300), t=t),
                              "Right": make_hand("Right", "POINT", (520, 300), t=t)})
    assert "media_play_pause" not in far.kinds()


# ------------------------------------------------------------------ tracker
class FakeBackend:
    def __init__(self, default_swap=False):
        self.script, self.default_swap = [], default_swap

    def detect(self, rgb, ts):
        return self.script.pop(0)

    def close(self):
        pass


def lm(cx, noise=0.0, rng=None):
    n = OPEN / [W, H]
    n[:, 0] += cx - n[[0, 5, 9, 13, 17], 0].mean()
    if noise:
        n = n + rng.normal(0, noise, n.shape)
    return n


FRAME = np.zeros((H, W, 3), np.uint8)


@check
def tracker_labels_and_occlusion():
    fb = FakeBackend()
    tr = HandTracker(backend=fb)
    fb.script = [
        [("Left", .9, lm(.3)), ("Right", .9, lm(.7))],
        [("Left", .9, lm(.7))],                              # right hand mislabelled as Left
        [("Right", .9, lm(.7)), ("Right", .8, lm(.3))],      # both labelled Right
        [], [],
    ]
    out = tr.process(FRAME, 0.0)
    assert set(out) == {"Left", "Right"}
    out = tr.process(FRAME, 0.03)
    assert not out["Right"].stale and out["Left"].stale, {k: v.stale for k, v in out.items()}
    out = tr.process(FRAME, 0.06)
    assert out["Left"].center[0] < out["Right"].center[0] and not out["Left"].stale
    out = tr.process(FRAME, 0.10)
    assert set(out) == {"Left", "Right"} and all(h.stale for h in out.values())
    out = tr.process(FRAME, 0.06 + cfg.LOST_GRACE_S + 0.1)
    assert out == {}, out


@check
def handedness_inversion_is_fixed():
    """Backend labels are reversed (the Tasks-API behaviour): position wins, and the swap flag self-corrects."""
    fb = FakeBackend(default_swap=False)
    tr = HandTracker(backend=fb)
    assert tr.swap is False
    fb.script = [[("Right", .9, lm(.3)), ("Left", .9, lm(.7))] for _ in range(cfg.HANDEDNESS_FIX_FRAMES + 3)]
    for i in range(len(fb.script)):
        out = tr.process(FRAME, i * DT)
        assert out["Left"].center[0] < out["Right"].center[0], "labels must follow screen position"
    assert tr.swap is True and tr.consume_swap_change() and not tr.consume_swap_change()
    fb.script = [[("Left", .9, lm(.75))]]                    # single hand on the right, reversed label
    out = tr.process(FRAME, 1.0)
    assert set(out) == {"Right"}, set(out)                   # swap now applied to lone hands too
    # correct labels never trigger a swap
    tr2 = HandTracker(backend=FakeBackend())
    tr2.backend.script = [[("Left", .9, lm(.3)), ("Right", .9, lm(.7))] for _ in range(30)]
    for i in range(30):
        tr2.process(FRAME, i * DT)
    assert tr2.swap is False
    # explicit start value and manual toggle ([H] key)
    tr3 = HandTracker(backend=FakeBackend(), swap=True)
    assert tr3.swap is True and tr3.toggle_swap() is False and tr3.consume_swap_change()


@check
def tracker_default_swap_per_backend():
    from hand_tracker import _LegacyBackend, _TasksBackend
    assert _TasksBackend.default_swap is True and _LegacyBackend.default_swap is False


@check
def landmark_smoothing_reduces_jitter():
    rng = np.random.default_rng(1)
    fb = FakeBackend()
    tr = HandTracker(backend=fb)
    fb.script = [[("Right", .9, lm(.5, noise=0.004, rng=rng))] for _ in range(120)]
    raw, smooth = [], []
    for i in range(120):
        raw.append(fb.script[0][0][2][8, 0])
        smooth.append(tr.process(FRAME, i * DT)["Right"].npts[8, 0])
    assert np.std(smooth[20:]) < 0.85 * np.std(raw[20:])


# ------------------------------------------------------------------ smoothing / calibration / controller
@check
def smoothing_reduces_jitter_but_follows_steps():
    for mode in ("one_euro", "ema", "moving_average"):
        rng = np.random.default_rng(0)
        sm = CursorSmoother(mode)
        raw, out = [], []
        for i in range(90):
            x = 500 + rng.normal(0, 2.0)
            raw.append(x)
            out.append(sm(x, 300, i * DT)[0])
        assert np.std(out[30:]) < 0.7 * np.std(raw[30:]), mode
        for i in range(90, 120):
            v = sm(900, 300, i * DT)[0]
        assert abs(v - 900) < 25, (mode, v)


@check
def calibration_mapping():
    cal = Calibration(1920, 1080)
    cal.region = (0.2, 0.2, 0.8, 0.8)
    cal.sensitivity = 1.0
    sx, sy = cal.to_screen(0.5, 0.5)
    assert abs(sx - 959.5) < 1 and abs(sy - 539.5) < 1
    assert cal.to_screen(0.0, 0.0) == (0.0, 0.0)
    cal.sensitivity = 2.0
    assert cal.effective_region() == (0.35, 0.35, 0.65, 0.65)


@check
def hand_movement_maps_to_cursor_movement():
    cal = Calibration(1920, 1080)
    cal.region, cal.sensitivity = (0.2, 0.2, 0.8, 0.8), 1.0
    dx, dy = cal.delta_to_screen(0.3, 0.15)                   # half the region width / quarter of its height
    assert abs(dx - 959.5) < 1 and abs(dy - 269.75) < 1
    cal.sensitivity = 2.0                                     # sensitivity doubles the cursor travel
    assert abs(cal.delta_to_screen(0.3, 0.0)[0] - 1919) < 1


def _run_wizard(corners):
    wiz, t, region = CalibrationWizard(), 0.0, None
    wiz.start(t)
    for corner in corners:
        for _ in range(int((cfg.CALIBRATION_READY_S + cfg.CALIBRATION_HOLD_S + 0.3) / DT)):
            region = wiz.update(np.array(corner), t) or region
            t += DT
    return region, wiz


@check
def calibration_wizard():
    region, wiz = _run_wizard([(0.25, 0.2), (0.75, 0.2), (0.75, 0.7), (0.25, 0.7)])
    assert region is not None and not wiz.active
    assert np.allclose(region, (0.25, 0.2, 0.75, 0.7), atol=0.01), region
    region, wiz = _run_wizard([(0.45, 0.4), (0.5, 0.4), (0.5, 0.45), (0.45, 0.45)])   # tiny area
    assert region is None and wiz.active and "too small" in wiz.message, wiz.message


@check
def calibration_file_roundtrip_keeps_handedness():
    old = cfg.CALIBRATION_FILE
    cfg.CALIBRATION_FILE = os.path.join(tempfile.mkdtemp(), "cal.json")
    try:
        cal = Calibration(1920, 1080)
        cal.swap_handedness = True
        cal.save()                                           # not calibrated yet: must not invent a region
        again = Calibration.load(1920, 1080)
        assert again.swap_handedness is True and not again.calibrated
        again.set_region((0.1, 0.1, 0.9, 0.9))
        again.save()
        final = Calibration.load(1920, 1080)
        assert final.calibrated and final.swap_handedness is True and final.region == (0.1, 0.1, 0.9, 0.9)
    finally:
        cfg.CALIBRATION_FILE = old


@check
def controller_moves_relatively_and_stays_put():
    cal = Calibration(1920, 1080)
    cal.region, cal.sensitivity = (0.2, 0.2, 0.8, 0.8), 1.0
    mc = MouseController(cal, dry_run=True)
    start = mc.cursor_xy
    mc.handle([Event("move_rel", 0.1, 0.0)], 0.0)             # no pinch-start event yet: still moves from the sync point
    mc.handle([Event("track_start")], 1.0)
    assert mc.cursor_xy[0] >= start[0]
    x0 = mc.cursor_xy
    mc.handle([Event("move_rel", 0.06, 0.0)], 1.03)
    assert abs(mc.cursor_xy[0] - (x0[0] + 0.06 / 0.6 * 1919)) < 3 and mc.cursor_xy[1] == x0[1]
    frozen = mc.cursor_xy
    mc.handle([], 1.5)                                         # nothing happens without events
    assert mc.cursor_xy == frozen
    mc.handle([Event("track_start"), Event("move_rel", 5.0, -5.0)], 3.0)   # clamped to the screen
    assert mc.cursor_xy == (1919, 0), mc.cursor_xy


@check
def controller_dry_run():
    cal = Calibration(1920, 1080)
    mc = MouseController(cal, dry_run=True)
    mc.handle([Event("track_start"), Event("move_rel", .01, .01), Event("left_click"),
               Event("double_click"), Event("drag_start"), Event("drag_end"), Event("scroll", value=1.5),
               Event("zoom", value=2), Event("window_next"), Event("window_end"),
               Event("level", value=4, data="volume"), Event("media_play_pause"),
               Event("macro", data=("x", ["win", "d"]))], 1.0)
    mc.handle([Event("toggle_pause")], 1.1)
    assert mc.paused
    mc.last_action = ("", 0.0)
    mc.handle([Event("left_click"), Event("move_rel", .2, .2)], 1.2)
    assert mc.last_action[0] == ""                           # ignored while paused


if __name__ == "__main__":
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:                              # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    sys.exit(1 if failed else 0)
