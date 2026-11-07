"""
hand_tracker.py  -  hand landmark detection + independent per-hand tracking
===========================================================================
* Wraps MediaPipe.  Two backends are supported and chosen automatically:
    - "legacy": mp.solutions.hands   (older mediapipe releases)
    - "tasks" : HandLandmarker       (current releases; model file is auto-downloaded)
* Detects up to two hands.  The frame is MIRRORED (selfie view) before detection.
* LEFT / RIGHT FIX.  MediaPipe's two APIs label mirrored frames differently (the Tasks API comes
  out reversed), so the tracker keeps a `swap` flag:
    - initial value: calibration.json  >  config.SWAP_HANDEDNESS  >  backend default
    - whenever two hands are clearly apart, labels are taken from their screen position
      (user's left hand = left side of the mirrored image) and compared with the model's labels;
      if they are consistently reversed the flag flips automatically (and is saved by main.py);
    - [H] in the window flips it manually.
* One track per label ("Left", "Right"): a lone mislabelled hand is re-assigned to the track it
  continues, and when detection drops the last pose is kept (`stale=True`) for LOST_GRACE_S.
* Landmarks are lightly smoothed (EMA) per track to remove sensor jitter.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config as cfg

# --------------------------------------------------------------- landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
PALM_POINTS = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


@dataclass
class Hand:
    """One tracked hand.  `pts` are pixels, `npts` are normalised 0..1 (x right, y down)."""
    label: str                 # "Left" / "Right" as the USER sees their hands
    score: float
    pts: np.ndarray            # (21, 2) pixel coordinates
    npts: np.ndarray           # (21, 2) normalised coordinates
    frame_w: int
    frame_h: int
    last_seen: float
    stale: bool = False        # True = not detected this frame, showing last known pose

    @property
    def center(self) -> np.ndarray:
        return self.npts[PALM_POINTS].mean(axis=0)


# =============================================================== backends
class _LegacyBackend:
    """mediapipe.solutions.hands (removed from the newest mediapipe releases)."""
    default_swap = False       # labels are correct on a mirrored frame

    def __init__(self, mp):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=cfg.MAX_HANDS,
            model_complexity=cfg.MODEL_COMPLEXITY,
            min_detection_confidence=cfg.MIN_DETECTION_CONF,
            min_tracking_confidence=cfg.MIN_TRACKING_CONF)

    def detect(self, rgb, ts_ms):
        rgb.flags.writeable = False
        res = self._hands.process(rgb)
        rgb.flags.writeable = True
        out = []
        if res.multi_hand_landmarks:
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                cls = hd.classification[0]
                out.append((cls.label, cls.score, np.array([(p.x, p.y) for p in lm.landmark])))
        return out

    def close(self):
        self._hands.close()


class _TasksBackend:
    """mediapipe.tasks HandLandmarker in VIDEO mode (needs hand_landmarker.task)."""
    default_swap = True        # on a mirrored frame the Tasks labels come out reversed

    def __init__(self, mp):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self._mp = mp
        path = _ensure_model(cfg.HAND_MODEL_PATH, cfg.HAND_MODEL_URL)
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg.MAX_HANDS,
            min_hand_detection_confidence=cfg.MIN_DETECTION_CONF,
            min_hand_presence_confidence=cfg.MIN_DETECTION_CONF,
            min_tracking_confidence=cfg.MIN_TRACKING_CONF)
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb, ts_ms):
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._landmarker.detect_for_video(image, ts_ms)
        out = []
        for lms, handed in zip(res.hand_landmarks, res.handedness):
            cat = handed[0]
            out.append((cat.category_name, cat.score, np.array([(p.x, p.y) for p in lms])))
        return out

    def close(self):
        self._landmarker.close()


def _ensure_model(path: str, url: str) -> str:
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(path):
        print(f"[hand_tracker] downloading hand model -> {path}")
        urllib.request.urlretrieve(url, path)
    return path


def _make_backend():
    import mediapipe as mp
    try:
        has_legacy = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
    except Exception:
        has_legacy = False
    choice = cfg.HAND_BACKEND
    if choice == "legacy" or (choice == "auto" and has_legacy):
        if not has_legacy:
            raise RuntimeError("HAND_BACKEND='legacy' but this mediapipe has no mp.solutions.hands")
        return _LegacyBackend(mp)
    return _TasksBackend(mp)


def _flip(label: str) -> str:
    return "Left" if label == "Right" else "Right"


# =============================================================== tracker
class HandTracker:
    """process(frame, now) -> {"Left": Hand, "Right": Hand} (only the hands currently tracked)."""

    def __init__(self, backend=None, swap: Optional[bool] = None):
        self.backend = backend if backend is not None else _make_backend()
        if swap is None:
            swap = cfg.SWAP_HANDEDNESS
        if swap is None:
            swap = bool(getattr(self.backend, "default_swap", False))
        self.swap: bool = bool(swap)
        self.swap_changed = False          # set when the swap flag changed (main.py saves it)
        self._vote = 0
        self._tracks: Dict[str, Hand] = {}
        self._last_ts = 0

    # ------------------------------------------------------------ public API
    def process(self, frame_bgr: np.ndarray, now: float) -> Dict[str, Hand]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ts = max(int(now * 1000), self._last_ts + 1)      # timestamps must strictly increase
        self._last_ts = ts
        return self._update_tracks(self.backend.detect(rgb, ts), w, h, now)

    def toggle_swap(self) -> bool:
        """Manual fix ([H] key): flip Left/Right interpretation."""
        self.swap = not self.swap
        self.swap_changed = True
        self._vote = 0
        self._tracks.clear()
        return self.swap

    def consume_swap_change(self) -> bool:
        changed, self.swap_changed = self.swap_changed, False
        return changed

    def close(self):
        try:
            self.backend.close()
        except Exception:
            pass

    # ------------------------------------------------------------ handedness self-check
    def _vote_handedness(self, reversed_now: bool) -> None:
        if reversed_now:
            self._vote -= 1
        else:
            self._vote = min(self._vote + 1, 6)
        if self._vote <= -cfg.HANDEDNESS_FIX_FRAMES:
            self.swap = not self.swap
            self.swap_changed = True
            self._vote = 0
            self._tracks.clear()
            print("[hand_tracker] Left/Right labels were reversed -> corrected automatically "
                  f"(swap={self.swap})")

    # ------------------------------------------------------------ track management
    def _dist_to_track(self, label: str, center: np.ndarray, now: float) -> float:
        tr = self._tracks.get(label)
        if tr is None or now - tr.last_seen > cfg.LOST_GRACE_S:
            return float("inf")
        return float(np.linalg.norm(center - tr.center))

    def _update_tracks(self, dets: List[Tuple[str, float, np.ndarray]], w: int, h: int,
                       now: float) -> Dict[str, Hand]:
        cands = []
        for label, score, lm in dets:
            npts = np.asarray(lm, dtype=float)[:, :2]
            if self.swap:
                label = _flip(label)
            cands.append({"label": label, "score": float(score), "npts": npts,
                          "center": npts[PALM_POINTS].mean(axis=0)})
        cands.sort(key=lambda c: -c["score"])
        cands = cands[: cfg.MAX_HANDS]

        # ---- resolve labels
        if len(cands) == 2:
            a, b = sorted(cands, key=lambda c: c["center"][0])   # a = left side of the mirrored image
            apart = (b["center"][0] - a["center"][0]) >= cfg.HAND_SEPARATION_MIN
            if apart:
                # your left hand is on the left of a mirrored image: trust POSITION over the model label
                if cfg.AUTO_FIX_HANDEDNESS and a["label"] != b["label"]:
                    self._vote_handedness(reversed_now=(a["label"] == "Right"))
                a["label"], b["label"] = "Left", "Right"
            elif a["label"] == b["label"]:
                a["label"], b["label"] = "Left", "Right"
        elif len(cands) == 1:
            c = cands[0]
            own, other = c["label"], _flip(c["label"])
            d_own = self._dist_to_track(own, c["center"], now)
            d_other = self._dist_to_track(other, c["center"], now)
            if d_other < d_own and d_other < cfg.LABEL_SWAP_MAX_DIST:
                c["label"] = other

        # ---- light landmark smoothing (continuity-checked so fast jumps are not smeared)
        alpha = cfg.LANDMARK_SMOOTHING
        fresh: Dict[str, Hand] = {}
        for c in cands:
            npts = c["npts"]
            prev = self._tracks.get(c["label"])
            if (alpha < 1.0 and prev is not None and not prev.stale and now - prev.last_seen <= 0.15
                    and float(np.linalg.norm(c["center"] - prev.center)) < 0.15):
                npts = alpha * npts + (1.0 - alpha) * prev.npts
            fresh[c["label"]] = Hand(
                label=c["label"], score=c["score"], npts=npts,
                pts=npts * np.array([w, h], dtype=float),
                frame_w=w, frame_h=h, last_seen=now)

        # ---- update tracks, carry lost hands for a short grace period
        out: Dict[str, Hand] = {}
        for label in ("Left", "Right"):
            if label in fresh:
                self._tracks[label] = fresh[label]
                out[label] = fresh[label]
            elif label in self._tracks and now - self._tracks[label].last_seen <= cfg.LOST_GRACE_S:
                out[label] = replace(self._tracks[label], stale=True)
            else:
                self._tracks.pop(label, None)
        return out


# =============================================================== drawing helper
def draw_hand(frame: np.ndarray, hand: Hand, color=(0, 255, 0)) -> None:
    """Draw the skeleton (red when the hand is stale)."""
    if hand.stale:
        color = (0, 0, 255)
    pts = [(int(x), int(y)) for x, y in hand.pts]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
