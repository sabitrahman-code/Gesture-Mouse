"""
main.py  -  Gesture Mouse application
=====================================
    python main.py                      normal run (first run starts the calibration wizard)
    python main.py --dry-run --debug    SAFE tuning mode: nothing is clicked, numbers are shown
    python main.py --help               all options

Window hotkeys:  q / Esc quit | c calibrate | p pause | + / - sensitivity | d debug | h swap Left/Right
Cursor: PINCH index + thumb and move your hand (like Apple Vision Pro); release = the cursor stays put.
        Quick pinch tap = left click, middle finger + thumb tap = right click.
Safety: hold BOTH fists side by side for ~1.2 s to pause / resume from the camera.
"""
from __future__ import annotations

import argparse
import platform
import threading
import time

import cv2

import config as cfg
from calibration import Calibration, CalibrationWizard
from gesture_recognizer import GestureEngine
from hand_tracker import INDEX_TIP, HandTracker, draw_hand
from mouse_controller import MouseController, get_screen_size

FONT = cv2.FONT_HERSHEY_SIMPLEX
HAND_COLORS = {"Left": (255, 120, 0), "Right": (0, 220, 0)}


# =============================================================== camera thread
class ThreadedCamera:
    """Grabs frames on a background thread so processing always sees the NEWEST frame (low latency)."""

    def __init__(self, index, width, height, fps, fourcc=None):
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else None
        self.cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
        if not self.cap.isOpened() and backend:
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index}. Try --camera 1, close other apps using it, "
                               "and check the OS camera permission.")
        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._cond:
                self._frame, self._seq = frame, self._seq + 1
                self._cond.notify_all()

    def read(self, last_seq, timeout=1.0):
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq, timeout=timeout)
            if self._seq == last_seq:
                return None, last_seq
            return self._frame, self._seq

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


# =============================================================== overlay
def put(img, text, org, color=(255, 255, 255), scale=0.5, thick=1):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_overlay(frame, calib, controller, engine, hands, wizard, fps, debug, dry_run, now):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = calib.effective_region()
    cv2.rectangle(frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (255, 160, 0), 1)

    for label, hand in hands.items():
        draw_hand(frame, hand, HAND_COLORS[label])
        f = engine.features.get(label)
        wx, wy = int(hand.pts[0][0]), int(hand.pts[0][1])
        put(frame, f"{label} {f.pose if f else ''}{' (lost)' if hand.stale else ''}",
            (max(wx - 40, 2), min(wy + 22, h - 4)), HAND_COLORS[label], 0.5)
        if debug and f:
            bits = "".join("1" if b else "0" for b in f.fingers)
            put(frame, f"idx {f.idx_ratio:.2f} mid {f.mid_ratio:.2f} fing {bits} "
                       f"L{int(f.click_pinch)} R{int(f.right_pinch)} v{f.speed:.1f}",
                (max(wx - 40, 2), min(wy + 40, h - 4)), (0, 255, 255), 0.45)

    status = "PAUSED" if controller.paused else engine.mode_name
    put(frame, f"FPS {fps:4.1f}   MODE {status}   Sens x{calib.sensitivity:.1f}"
               f"{'   [DRY-RUN]' if dry_run else ''}", (8, 18), (0, 255, 255), 0.55)
    put(frame, "q quit | c calibrate | p pause | +/- sens | d debug | h swap L/R", (8, h - 8),
        (200, 200, 200), 0.42)

    # pinch indicator: yellow ring = pinch seen (tap = click), green ring = pinch is driving the cursor
    if controller.paused:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 165, 255), 4)
    elif not wizard.active:
        prim = engine.primary()
        if prim is not None and prim.fresh:
            px, py = int(prim.pinch_mid[0] * w), int(prim.pinch_mid[1] * h)
            if engine.one.cursor_active:
                cv2.circle(frame, (px, py), 14, (0, 220, 0), 3, cv2.LINE_AA)
            elif engine.one.state == engine.one.PINCH_IDX:
                cv2.circle(frame, (px, py), 14, (0, 220, 255), 2, cv2.LINE_AA)
            elif engine.one.state == engine.one.PINCH_MID:
                cv2.circle(frame, (px, py), 14, (255, 120, 0), 2, cv2.LINE_AA)
        if engine.one.state == engine.one.IDLE and not engine.two.active:
            put(frame, "pinch + move = cursor | tap = click | middle+thumb = right click",
                (8, 40), (200, 200, 200), 0.45)

    # hold-to-confirm progress bar (pause, play/pause, macros, two-hand modes)
    prog = engine.progress
    if prog is not None and not wizard.active:
        label, frac = prog
        bx, by, bw = w // 2 - 100, h - 70, 200
        put(frame, label, (bx, by - 6), (255, 255, 255), 0.5)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + 10), (255, 255, 255), 1)
        cv2.rectangle(frame, (bx, by), (bx + int(bw * frac), by + 10), (0, 255, 255), -1)

    # live level gauge while adjusting volume / brightness
    if engine.two.mode in ("VOLUME", "BRIGHTNESS"):
        put(frame, f"{engine.two.mode} {engine.two.level_total:+.0f}%  (up = more, down = less)",
            (8, 40), (0, 255, 255), 0.55, 2)
    elif engine.two.mode == "WINDOW":
        put(frame, "ALT+TAB: push palm RIGHT = next, LEFT = previous, open fist = select",
            (8, 40), (0, 255, 255), 0.5, 2)

    if not calib.calibrated and not wizard.active:
        put(frame, "Using default area - press C to calibrate", (8, 58), (0, 200, 255), 0.5)

    text, t = controller.last_action
    if text and now - t < 0.8:
        put(frame, text, (w // 2 - 6 * len(text), h - 40), (0, 255, 0), 0.8, 2)

    if dry_run:                                           # virtual cursor
        cx = int(controller.cursor_xy[0] / calib.screen_w * w)
        cy = int(controller.cursor_xy[1] / calib.screen_h * h)
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), 2, cv2.LINE_AA)

    if wizard.active:
        put(frame, wizard.prompt, (8, 62), (0, 200, 255), 0.55, 2)
        cv2.rectangle(frame, (8, 72), (8 + int(200 * wizard.progress), 80), (0, 255, 0), -1)
        cv2.rectangle(frame, (8, 72), (208, 80), (255, 255, 255), 1)
    if wizard.message and now - wizard.message_t < 4.0:       # result of the last calibration, shown briefly
        put(frame, wizard.message, (8, 100 if wizard.active else 62), (0, 255, 0), 0.55, 2)


# =============================================================== CLI
def parse_args():
    ap = argparse.ArgumentParser(description="Gesture-controlled virtual mouse")
    ap.add_argument("--camera", type=int, default=cfg.CAMERA_INDEX, help="camera index")
    ap.add_argument("--width", type=int, default=cfg.CAMERA_WIDTH)
    ap.add_argument("--height", type=int, default=cfg.CAMERA_HEIGHT)
    ap.add_argument("--dry-run", action="store_true", help="do not control the OS, just show actions")
    ap.add_argument("--debug", action="store_true", help="show pinch ratios and finger states")
    ap.add_argument("--no-preview", action="store_true", help="run without a window (Ctrl+C to quit)")
    ap.add_argument("--calibrate", action="store_true", help="start the calibration wizard now")
    ap.add_argument("--skip-calibration", action="store_true", help="do not auto-start calibration")
    ap.add_argument("--primary-hand", choices=["Left", "Right"], default=cfg.PRIMARY_HAND,
                    help="hand that moves the cursor (the other is the modifier hand)")
    ap.add_argument("--sensitivity", type=float, default=None)
    return ap.parse_args()


# =============================================================== main loop
def main():
    args = parse_args()
    cfg.PRIMARY_HAND = args.primary_hand
    cfg.MODIFIER_HAND = "Left" if args.primary_hand == "Right" else "Right"

    screen_w, screen_h = get_screen_size()
    calib = Calibration.load(screen_w, screen_h)
    if args.sensitivity:
        calib.sensitivity = args.sensitivity
    controller = MouseController(calib, dry_run=args.dry_run)
    engine = GestureEngine()
    wizard = CalibrationWizard()
    tracker = HandTracker(swap=calib.swap_handedness)
    cam = ThreadedCamera(args.camera, args.width, args.height, cfg.CAMERA_FPS, cfg.CAMERA_FOURCC)

    debug = args.debug
    preview = not args.no_preview
    seq, fps, t_prev = 0, 0.0, time.perf_counter()
    if args.calibrate or (not calib.calibrated and not args.skip_calibration):
        wizard.start(t_prev)

    print(f"Gesture Mouse running on a {screen_w}x{screen_h} screen"
          f"{' (DRY-RUN)' if args.dry_run else ''}.  Press q in the window to quit.")
    print("Pinch index + thumb and move your hand to move the cursor; a quick tap clicks.  "
          "If Left/Right look reversed, press H in the window.")
    try:
        while True:
            frame, seq = cam.read(seq)
            if frame is None:
                continue
            now = time.perf_counter()
            frame = cv2.flip(frame, 1)                    # mirror: makes Left/Right labels correct
            hands = tracker.process(frame, now)

            if wizard.active:
                prim = hands.get(cfg.PRIMARY_HAND) or (next(iter(hands.values())) if len(hands) == 1 else None)
                tip = prim.npts[INDEX_TIP] if prim is not None and not prim.stale else None
                region = wizard.update(tip, now)
                if region is not None:
                    calib.set_region(region)
                    calib.save()
            else:
                controller.handle(engine.update(hands, now, paused=controller.paused), now)
            if tracker.consume_swap_change():             # Left/Right fix: remember it
                calib.swap_handedness = tracker.swap
                calib.save()

            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt

            if not preview:
                continue
            draw_overlay(frame, calib, controller, engine, hands, wizard, fps, debug, args.dry_run, now)
            cv2.imshow(cfg.WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                controller.release_all()
                engine.reset()
                wizard.start(now)
            elif key == ord("p"):
                controller.toggle_pause()
            elif key in (ord("+"), ord("=")):
                calib.adjust_sensitivity(cfg.SENSITIVITY_STEP)
                calib.save()
            elif key in (ord("-"), ord("_")):
                calib.adjust_sensitivity(-cfg.SENSITIVITY_STEP)
                calib.save()
            elif key == ord("d"):
                debug = not debug
            elif key == ord("h"):
                tracker.toggle_swap()
                print(f"[main] Left/Right labels swapped -> swap={tracker.swap}")
    except KeyboardInterrupt:
        pass
    finally:
        controller.release_all()                          # never leave a button / Alt key stuck down
        cam.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
