"""
config.py  -  every tunable number of the Gesture Mouse lives in this file
==========================================================================

HOW TO TUNE (calibration & threshold guide)
-------------------------------------------
1. Run   python main.py --debug --dry-run
   (dry-run = nothing is clicked, a green virtual cursor is drawn instead).
   For every hand the overlay shows:
       pose      OPEN / FIST / POINT / PEACE / THREE / PINCH / MIDDLE_PINCH / UNKNOWN
       idx       thumb-tip <-> index-tip distance  (1.00 = one palm length)
       mid       thumb-tip <-> middle-tip distance (same unit)
       fing      five digits thumb..pinky, 1 = extended
   Perform a gesture, read the numbers, and pick a threshold that sits
   BETWEEN the "gesture" value and the "no gesture" value.

2. Quick symptom -> knob table

   SYMPTOM                                        KNOB (what to do)
   ---------------------------------------------  --------------------------------------
   Left / Right labels are reversed               press [H] in the window (saved), or SWAP_HANDEDNESS
   Cursor shakes while the hand is still          ONE_EURO_MIN_CUTOFF down, CURSOR_AVERAGE_FRAMES up,
                                                  CURSOR_DEADZONE_PX up, LANDMARK_SMOOTHING down
   Cursor lags behind fast hand movement          ONE_EURO_BETA up, CURSOR_AVERAGE_FRAMES = 1
   Have to stretch the arm to reach the corners   press [+] in the window, or recalibrate with [c]
   Cursor too fast / too twitchy                  press [-] in the window
   Cursor does not move                           you must be PINCHING (index + thumb) while moving the hand
   Cursor moves a little during a click tap       PINCH_MOVE_START_DIST up
   Cursor needs too much / little hand travel     press [+] / [-] in the window, PINCH_CURSOR_GAIN
   Left click fires by accident                   CLICK_PINCH_ON down (e.g. 0.20), CLICK_CONFIRM_FRAMES up
   Left click is hard to trigger                  CLICK_PINCH_ON up (e.g. 0.30), TAP_MAX_S up
   Right click fires by accident                  RIGHT_CLICK_PINCH_ON down, RIGHT_CLICK_CONFIRM_FRAMES up
   Right click is hard to trigger                 RIGHT_CLICK_PINCH_ON up (e.g. 0.27), RIGHT_CLICK_MIN_INDEX_GAP down
   Left / right click mixed up                    CLICK_AMBIGUITY_MARGIN up, RIGHT_CLICK_MIN_INDEX_GAP up
   Clicks fire while the hand is moving fast      CLICK_START_MAX_SPEED down
   A fist is mistaken for a pinch                 PINCH_MIN_THUMB_RATIO up
   Click fires twice                              CLICK_MIN_INTERVAL_S up, PINCH_REARM_S up
   Double click is not recognised                 DOUBLE_CLICK_WINDOW_S up
   Fist does not grab                             GRAB_HOLD_S down; check the 'fing' bits in --debug
   Fist grabs by accident                         GRAB_HOLD_S up, GRAB_REQUIRES_OPEN_WITHIN_S down
   Dropped by accident while dragging             DROP_CONFIRM_FRAMES up, DROP_MIN_FINGERS up
   Does not drop when the hand opens              DROP_CONFIRM_FRAMES down, DROP_MIN_FINGERS down
   Finger wrongly seen as bent / straight         FINGER_EXTENDED_RATIO (down = more "straight")
   Poses flicker between two names                POSE_STABLE_FRAMES up
   Scrolling too fast / too slow                  SCROLL_GAIN down / up
   Scroll creeps when hand is still               SCROLL_DEADZONE up
   Window switcher steps too easily / too rarely  WINDOW_PUSH_DX up / down
   Two-hand mode triggers by accident             TWO_HAND_ENGAGE_S / MACRO_HOLD_S up
   Zoom too fast / slow                           ZOOM_STEP_NORM up / down
   Volume or brightness too fast                  LEVEL_GAIN down or LEVEL_STEP_PERCENT up
   Volume drifts when hand is almost still        LEVEL_DEADZONE up
   Lower FPS than 30                              lower CAMERA_WIDTH/HEIGHT, better lighting

All distances are divided by "hand size" (wrist -> middle-finger knuckle), so
the thresholds work at any distance from the camera.
"""
import platform

_SYSTEM = platform.system()          # "Windows" | "Darwin" (macOS) | "Linux"
IS_MAC = _SYSTEM == "Darwin"

# ============================================================== 1. CAMERA
CAMERA_INDEX = 0
CAMERA_WIDTH = 640           # 640x480 is the sweet spot for speed
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_FOURCC = "MJPG"       # MJPG lets many USB webcams reach 30 FPS; None = leave default

# ============================================================== 2. MEDIAPIPE
HAND_BACKEND = "auto"        # "auto" | "legacy" (mp.solutions) | "tasks" (HandLandmarker)
MAX_HANDS = 2
MODEL_COMPLEXITY = 0         # legacy backend only: 0 = fastest, 1 = more accurate
MIN_DETECTION_CONF = 0.6
MIN_TRACKING_CONF = 0.5
# Tasks backend needs a model file; it is downloaded automatically on first run.
HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")

# ---- Left / Right handedness -------------------------------------------------------------
# The frame is mirrored (selfie view) before detection.  MediaPipe's two APIs label mirrored
# frames differently, so the tracker decides whether to swap the labels like this:
#   1. the value saved in calibration.json (set by the [H] key or by the automatic check),
#   2. else SWAP_HANDEDNESS below (None = automatic, True / False = force),
#   3. else the backend default (Tasks API: swapped, legacy API: not swapped).
# Whenever BOTH hands are visible and clearly apart, the tracker also checks the labels against
# their on-screen position (your left hand is on the left of the mirrored image) and
# auto-corrects the swap setting if the labels are consistently reversed.
SWAP_HANDEDNESS = None
AUTO_FIX_HANDEDNESS = True
HAND_SEPARATION_MIN = 0.20   # hands must be this far apart (fraction of frame width) to compare / relabel
HANDEDNESS_FIX_FRAMES = 12   # this many reversed frames in a row-ish before the swap flips

# ============================================================== 3. TRACKING
LOST_GRACE_S = 0.35          # keep a hand "alive" this long after detection drops (occlusion)
LABEL_SWAP_MAX_DIST = 0.25   # max jump (fraction of frame) to treat a re-labelled hand as the same hand
LANDMARK_SMOOTHING = 0.65    # EMA weight of the NEW landmarks (1.0 = off, 0.5 = very smooth)

# ============================================================== 4. HAND ROLES
PRIMARY_HAND = "Right"       # does cursor / clicks / scroll / drag.  Left-handed? use "Left"
MODIFIER_HAND = "Left" if PRIMARY_HAND == "Right" else "Right"   # modifier hand for 2-hand gestures

# ============================================================== 5. CALIBRATION & MAPPING
# Active region of the camera image that maps to the WHOLE screen: (x0, y0, x1, y1) in 0..1.
DEFAULT_ACTIVE_REGION = (0.20, 0.15, 0.80, 0.75)
SENSITIVITY = 1.0            # >1 = smaller hand movement for the same cursor travel
SENSITIVITY_MIN, SENSITIVITY_MAX, SENSITIVITY_STEP = 0.5, 3.0, 0.1
CALIBRATION_FILE = "calibration.json"
CALIBRATION_READY_S = 1.2    # time to move the finger to the next corner
CALIBRATION_HOLD_S = 1.5     # time to hold still on a corner
CALIBRATION_STABILITY = 0.05 # max wobble (fraction of frame) while holding
CALIBRATION_MIN_SIZE = (0.15, 0.12)   # smallest accepted region (width, height)

# ============================================================== 6. CURSOR ANTI-JITTER
# Three stacked stages:  landmark EMA (section 3)  ->  main filter  ->  short moving average  ->  dead-zone
# (unchanged: these now smooth the pinch-driven cursor)
SMOOTHING_MODE = "one_euro"  # "one_euro" (adaptive, recommended) | "ema" | "moving_average" | "none"
ONE_EURO_MIN_CUTOFF = 0.8    # lower = smoother when slow (less jitter, more lag)
ONE_EURO_BETA = 0.02         # higher = less lag when moving fast
ONE_EURO_D_CUTOFF = 1.0
CURSOR_EMA_ALPHA = 0.35      # "ema" mode: weight of the new sample (lower = smoother)
MOVING_AVERAGE_WINDOW = 5    # "moving_average" mode: frames
CURSOR_AVERAGE_FRAMES = 2    # extra moving average after the main filter (1 = off)
CURSOR_DEADZONE_PX = 3       # ignore cursor moves smaller than this many pixels

# ============================================================== 7. FINGER STATE / POSES
FINGER_EXTENDED_RATIO = 1.20 # |tip-wrist| / |pip-wrist| above this => finger is straight
THUMB_EXTENDED_RATIO = 1.00  # |thumb tip-pinky knuckle| / |thumb knuckle-pinky knuckle|
POSE_STABLE_FRAMES = 3       # a pose must persist this many frames to be accepted (de-flicker)
# PINCH_* = the "pinch" POSE used by the two-hand gestures (zoom, volume, brightness): unchanged.
PINCH_ON = 0.28              # thumb<->fingertip distance (in palm lengths) below which a pinch STARTS
PINCH_OFF = 0.42             # ...and above which it ENDS (hysteresis stops flicker)
PINCH_MIN_THUMB_RATIO = 0.85 # a pinch needs the thumb pointing out; a tucked thumb (fist) never counts
PINCH_CONFIRM_FRAMES = 2     # pinch must be seen this many frames in a row to start ...
PINCH_RELEASE_FRAMES = 2     # ... and its release this many frames in a row to end

# ============================================================== 8. ONE-HAND GESTURES
# CURSOR (Apple-Vision-Pro style): the cursor moves ONLY while index finger + thumb are pinched.
#   pinch + move the hand   -> cursor moves (relative: it continues from where it stopped, never jumps)
#   release the pinch       -> cursor stays exactly where it is
#   quick pinch tap, hand still        -> left click (two taps = double click)
#   middle finger + thumb tap          -> right click
#   index + middle up (PEACE)          -> scroll
#   close a FIST = grab (mouse down)   -> open the hand = drop (mouse up)
PINCH_CURSOR_GAIN = 1.0      # extra cursor speed on top of calibration + sensitivity ([+] / [-])
PINCH_MOVE_START_DIST = 0.012# the pinched hand must travel this far (fraction of frame width) before the
                             # cursor starts to move: keeps the cursor still while you tap to click
TAP_MAX_S = 0.35             # a pinch shorter than this (and without movement) is a click

# --- click-grade pinch detection (stricter than the pinch POSE above; used only for cursor + clicks)
CLICK_PINCH_ON = 0.25        # index<->thumb distance (palm lengths) below which a left-click pinch STARTS
CLICK_PINCH_OFF = 0.40       # ...and above which it ENDS
CLICK_CONFIRM_FRAMES = 3     # frames in a row needed to start it
CLICK_RELEASE_FRAMES = 2     # frames in a row needed to end it
RIGHT_CLICK_PINCH_ON = 0.22  # middle<->thumb: right-click pinch starts below this (tighter than left)
RIGHT_CLICK_PINCH_OFF = 0.36
RIGHT_CLICK_CONFIRM_FRAMES = 4   # right click needs a longer, more deliberate pinch
RIGHT_CLICK_MIN_INDEX_GAP = 0.45 # for a right click the index fingertip must be at least this far from the thumb
CLICK_AMBIGUITY_MARGIN = 0.10    # for a left click the middle fingertip must be this much farther than the index
CLICK_START_MAX_SPEED = 1.5  # a pinch cannot START while the hand moves faster than this (frame widths / second)
MIN_TRACK_FRAMES = 6         # a hand must be tracked this many frames before any click pinch can start
# (a click pinch also needs the fingers to have been clearly OPEN first, so a hand that merely
#  appears in the picture or a relaxed hand can never "click")

TAP_MIN_S = 0.06             # shorter pinches are treated as noise
PINCH_REARM_S = 0.10         # a new pinch is ignored this soon after the previous one ended
CLICK_MIN_INTERVAL_S = 0.12  # hard lower limit between two emitted clicks (anti double-fire)
POST_CLICK_FREEZE_S = 0.15   # no grab right after a click / drop
DOUBLE_CLICK_WINDOW_S = 0.45 # two taps within this time = double click
RIGHT_CLICK_MAX_S = 0.50     # middle+thumb tap must be shorter than this
RIGHT_CLICK_COOLDOWN_S = 0.60
GRAB_HOLD_S = 0.20           # a fist must be held this long to grab
GRAB_REQUIRES_OPEN_WITHIN_S = 2.0   # ...and must have been closed from a non-fist hand at most this long ago
                                    # (a hand resting as a fist can never grab)
DROP_MIN_FINGERS = 3         # "open hand" = at least this many of the 4 fingers extended ...
DROP_CONFIRM_FRAMES = 3      # ... for this many frames in a row
SCROLL_GAIN = 20.0           # wheel notches per full frame-height of hand travel
SCROLL_DEADZONE = 0.003      # per-frame vertical movement (fraction of frame) ignored as jitter
SCROLL_MAX_JUMP = 0.15       # per-frame movement larger than this is treated as a tracking glitch
SCROLL_INVERT = False        # False: hand up = page up.   True: "natural" scrolling

# ============================================================== 9. TWO-HAND GESTURES
ENABLE_TWO_HAND = True
ENABLE_ZOOM = True
ENABLE_WINDOW_SWITCH = True
ENABLE_VOLUME = True
ENABLE_BRIGHTNESS = True
ENABLE_MACROS = True
ENABLE_PAUSE_GESTURE = True
ENABLE_MEDIA_GESTURE = True

TWO_HAND_ENGAGE_S = 0.35     # both hands must hold the pose this long before a mode starts
WINDOW_ENGAGE_S = 0.45       # Alt+Tab mode needs a slightly longer, deliberate hold
TWO_HAND_RELEASE_GRACE_S = 0.25   # tolerate this long of pose flicker / hand loss before leaving a mode
MACRO_HOLD_S = 0.80          # macro poses must be held this long
ONE_SHOT_COOLDOWN_S = 1.5    # one-shot gestures (macros, pause, play/pause) cannot repeat faster

# Zoom: both hands pinch, then move them apart (zoom in) or together (zoom out)
ZOOM_STEP_NORM = 0.03        # change in hand distance (fraction of frame width) per zoom step
ZOOM_INVERT = False
ZOOM_METHOD = "keys" if IS_MAC else "scroll"   # "scroll" = Ctrl+wheel, "keys" = Ctrl/Cmd and +/-
ZOOM_SCROLL_NOTCHES = 1.0    # wheel notches per zoom step (scroll method)

# Window switch (Alt+Tab): modifier hand = FIST (holds Alt), primary hand = OPEN palm.
# Push the palm RIGHT = next window, LEFT = previous, return to the centre between steps
# (joystick style, works at any speed).  Open the fist to select the highlighted window.
WINDOW_PUSH_DX = 0.12        # push this far from where you started (fraction of frame width) = one step
WINDOW_REARM_DX = 0.05       # come back within this distance of the start to allow the next step
WINDOW_REPEAT_DELAY_S = 0.80 # keep pushing this long to start auto-repeat (0 = no repeat)
WINDOW_REPEAT_S = 0.50       # time between auto-repeated steps
WINDOW_INVERT = False

# Volume / brightness: modifier hand OPEN = volume, PEACE = brightness; primary hand PINCHES
# (like grabbing a slider knob) and moves.  Control is RELATIVE: up = more, down = less.
LEVEL_SOURCE = "vertical"    # "vertical" (hand height) | "pinch" (thumb-index opening)
LEVEL_GAIN = 1.6             # percent change per percent of travel
LEVEL_STEP_PERCENT = 4       # smallest change sent to the OS
LEVEL_DEADZONE = 0.015       # ignore hand drift smaller than this (fraction of frame)
LEVEL_CONTROLLER_POSES = ("PINCH",)   # use ("POINT",) with LEVEL_SOURCE = "pinch"

# Pause / resume the whole app: BOTH FISTS held together (side by side, close to each other)
PAUSE_HOLD_S = 1.2
PAUSE_MAX_GAP = (0.30, 0.25) # max distance between the fists (fraction of frame width, height)

# Play / pause the video: both index fingers pointing UP side by side (the "||" pause symbol)
MEDIA_HOLD_S = 0.9
MEDIA_MAX_GAP = (0.30, 0.15) # max distance between the two fingertips (fraction of width, height)
MEDIA_MAX_TILT_DEG = 35      # fingers must point (almost) straight up
MEDIA_KEY = "space" if IS_MAC else "playpause"   # mac: space (works when the player has focus)

# Macros: (LEFT-hand pose, RIGHT-hand pose) -> (label, key list).  Hold both poses MACRO_HOLD_S.
# Poses: OPEN FIST POINT PEACE THREE PINCH MIDDLE_PINCH.
if IS_MAC:
    _SHOW_DESKTOP, _SNIP = ["f11"], ["command", "shift", "4"]
else:
    _SHOW_DESKTOP, _SNIP = ["win", "d"], ["win", "shift", "s"]
MACROS = {
    ("OPEN", "OPEN"): ("Show desktop", _SHOW_DESKTOP),
    ("PEACE", "PEACE"): ("Screenshot tool", _SNIP),
}

# ============================================================== 10. OS INTEGRATION
PYAUTOGUI_FAILSAFE = False   # True = slamming the cursor into a corner aborts the program
# Units sent to pyautogui.scroll() per wheel notch on each OS
SCROLL_UNITS_PER_NOTCH = {"Windows": 120, "Darwin": 3, "Linux": 1}

# ============================================================== 11. UI
WINDOW_NAME = "Gesture Mouse"
