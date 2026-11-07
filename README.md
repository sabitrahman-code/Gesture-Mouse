# Gesture Mouse

Control your computer with hand gestures from an ordinary webcam. Real-time, one- and two-hand gestures, **pinch-to-move** cursor, anti-jitter smoothing, hand-to-screen calibration, fully tunable.

**Stack:** Python 3 · OpenCV (capture + overlay) · MediaPipe (21-point hand landmarks + handedness) ·
PyAutoGUI (mouse / keyboard) · NumPy

---

## 1. Install

Python 3.9 – 3.14 works. Older MediaPipe releases only had wheels for 3.9 – 3.12, but the current releases (1.x) ship
version-independent wheels for Windows, Linux and Apple-Silicon Macs, so Python 3.13 / 3.14 install fine.
Intel Macs have no recent MediaPipe wheel: use Python 3.12 there. Upgrade pip first: `python -m pip install --upgrade pip`.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS / Linux:  source .venv/bin/activate

pip install -r requirements.txt
# or manually:
pip install opencv-python mediapipe numpy pyautogui
pip install screen-brightness-control      # optional, only for the brightness gesture
```

| OS | Notes |
|----|-------|
| Windows | Works out of the box. |
| macOS | Allow **Camera** and **Accessibility** (System Settings → Privacy & Security) for your terminal / IDE. |
| Linux | Needs an **X11** session (Wayland blocks synthetic input). `sudo apt install python3-tk python3-dev`. Volume uses `pactl` or `amixer`. |

MediaPipe model: current MediaPipe releases use the *Tasks* API, which needs `hand_landmarker.task`.
It is **downloaded automatically on the first run** (needs internet once). Older releases use a built-in model.
The backend is chosen automatically (`HAND_BACKEND` in `config.py`).

## 2. Run (30-second start)

```bash
python selftest.py                 # optional: 37 webcam-free checks of the gesture logic
python main.py --dry-run --debug   # SAFE first run: nothing is clicked, a green virtual cursor is shown
python main.py                     # the real thing
```

1. **Calibrate** (starts automatically on the first run): hold your index fingertip still at your comfortable
   top-left, top-right, bottom-right and bottom-left reach (~1.5 s each).
2. **Check the labels:** your right hand must be labelled *Right*. If they look reversed press **H**.
3. **Move the cursor by pinching:** touch **index finger and thumb together** and move your hand. The cursor follows while
   you pinch (a green ring shows the pinch is driving the cursor) and **stays exactly where it is as soon as you let go**.
4. **Click with a quick pinch tap** (hand still). A yellow ring means "pinch seen, waiting to see whether it is a tap or a move".

Options: `--camera N` · `--width/--height` · `--dry-run` · `--debug` · `--no-preview` · `--calibrate` ·
`--skip-calibration` · `--primary-hand Left|Right` · `--sensitivity X`

Window hotkeys: **q / Esc** quit · **c** calibrate · **p** pause · **+ / −** sensitivity · **d** debug numbers ·
**h** swap Left/Right labels

> **Safety.** The cursor cannot move unless you are pinching (or dragging with a fist), so a hand resting in view does nothing.
> A fist can only grab if you closed it from an open / pointing / pinching hand. Hold **both fists side by side for ~1.2 s** to
> pause / resume everything from the camera. Alt and the mouse button are always released on exit. Failsafe corner-abort is
> off by default (`PYAUTOGUI_FAILSAFE`) because the cursor legitimately reaches corners.

---

## 3. Gesture cheat-sheet

Default roles: **Right hand = primary** (cursor, clicks, scroll, drag). **Left hand = modifier** (two-hand
gestures). Left-handed? `python main.py --primary-hand Left`.

### One hand (primary)

| Action | Gesture |
|--------|---------|
| **Move cursor** | **Pinch index + thumb and move your hand** (pinch-and-drag, like Apple Vision Pro). Release the pinch = the cursor stays where it is. Lift and re-pinch to keep going, exactly like lifting a finger on a trackpad |
| **Left click** | **Quick pinch tap** (index + thumb, shorter than 0.35 s) **without moving the hand** |
| **Double click** | Two quick taps (within 0.45 s); the second click lands exactly where the first did |
| **Right click** | **Quick tap of middle finger + thumb** (index finger stays away from the thumb) |
| **Drag & drop** | **Close your hand into a fist to grab** (hold ~0.2 s) → move the fist → **open your hand to release** (≥ 3 fingers extended). The grab happens at the cursor's current position, and the cursor then follows your fist |
| **Scroll** | Index + middle raised (**PEACE**), move the hand up / down |

How cursor movement works: the hand's movement is added to the cursor position (relative), using the same hand-to-screen scale as
your calibration and `+ / −` sensitivity, followed by the same smoothing as before. So there is **no jump when you pinch**, and the
cursor is stationary whenever you are not pinching. To cross the whole screen faster press `+` (more cursor travel per hand
movement) or re-pinch like a trackpad swipe.

### Two hands (hold the pose to engage; a progress bar shows the hold)

| Action | Gesture |
|--------|---------|
| **Zoom in / out** | Both hands **pinch**, move them apart / together (Ctrl + wheel, or Cmd +/− on macOS) |
| **Alt+Tab** | **Left FIST** (holds Alt) + **right OPEN palm**. **Push the right palm to the right = next window, to the left = previous**, return to the centre between steps (keep pushing for auto-repeat). **Open the fist to select** |
| **Volume** | **Left OPEN palm** + **right-hand pinch** ("grab the knob"): move up = louder, down = quieter |
| **Brightness** | **Left PEACE ✌** + **right-hand pinch**: move up / down |
| **Show desktop** | Both hands **OPEN** for 0.8 s (Win + D; F11 on macOS) |
| **Screenshot tool** | Both hands **PEACE** for 0.8 s (Win + Shift + S) |
| **Pause / resume gesture control** | **Both fists held side by side, close together**, for 1.2 s (orange frame while paused) |
| **Play / pause video** | **Both index fingers pointing straight up, side by side** (the **‖** pause symbol) for 0.9 s → media play/pause key |

Volume / brightness are **relative** (they change from the current level, with a dead-zone so tiny hand drift does
nothing) and the overlay shows the running total, e.g. `VOLUME +12%`.

Keep your left hand **out of view or relaxed** (not open / fist / peace) while working with the right hand alone, so
no two-hand gesture can start by accident.

---

## 4. How clicks stay accurate

Left and right clicks use a stricter **click-grade pinch** (separate from the pinch pose used by zoom / volume / brightness):

* **Tight distance, with hysteresis:** a left click needs thumb ↔ index tip distance below 0.25 palm-lengths and ends above 0.40;
  a right click is tighter still (0.22 / 0.36).
* **Confirmation frames:** left pinch must be seen 3 frames in a row, right pinch 4 frames, releases 2 frames. One-frame glitches
  do nothing.
* **Unambiguous:** for a left click the middle fingertip must be clearly farther from the thumb than the index fingertip; for a right
  click the index fingertip must stay at least 0.45 away. When both fingers touch the thumb, **nothing** fires.
* **Opened first:** a pinch only counts after the fingers were seen clearly open, so a hand that just appeared (or relaxed fingers) can
  never click. The hand must also have been tracked for 6 frames.
* **Not while the hand is moving fast** (motion blur creates fake pinches): a pinch cannot start above 1.5 frame-widths / second.
* **Thumb out:** a tucked thumb (fist) never counts as a pinch.
* **Taps never move the cursor:** the pinched hand must travel 1.2 % of the frame width before the cursor starts to move, and the cursor
  follows the **palm** (not the fingertips), so opening or closing the fingers cannot shift it.
* **Tap = short and still:** 0.06 – 0.35 s without movement = left click; right click 0.06 – 0.5 s. Longer holds do nothing.
* **Debounce:** minimum time between clicks, re-arm time after each pinch, per-click cooldown, double click re-uses the first position.

## 5. Left / Right hand fix (handedness)

The camera image is mirrored first (like a mirror), so your right hand appears on the right of the preview. MediaPipe's
two APIs label such mirrored frames differently — the current *Tasks* API comes out reversed. The tracker corrects this:

* whenever **both hands are visible and apart**, labels are taken from their **position** and compared with the model's labels; if
  they are consistently reversed, the swap flips by itself and is saved in `calibration.json`;
* if a single hand is still labelled wrongly, press **H** in the window (saved too), or set `SWAP_HANDEDNESS = True / False`
  in `config.py`. The label drawn near each wrist is the label the program uses.

## 6. Camera setup, lighting and sensitivity tips

* **Light the hand, not the background.** Even, soft light in front of you. Avoid a bright window or lamp *behind* you: the camera turns
  your hand into a silhouette and tracking gets noisy.
* **Frame rate depends on light.** In dim rooms webcams slow their shutter and drop below 30 FPS. Add light before changing any setting.
* **Plain background**, no skin-coloured wall, no other hands / faces with hands in view.
* **Camera at face or chest height**, hands 30 – 80 cm from the lens, **palm facing the camera**, fingers clearly separated, whole hand
  inside the picture. **Make the pinch face the camera** (thumb and index tips visible, not hidden behind the hand).
* **Calibrate for your sitting position** (`c`) and re-run it if you move the camera or chair.
* **Too sensitive / twitchy?** `−` key, `ONE_EURO_MIN_CUTOFF` down, `CURSOR_AVERAGE_FRAMES` up.
  **Too sluggish / too much arm movement?** `+` key, `PINCH_CURSOR_GAIN` up. **Gestures misread?** run `--debug` and tune (§7).
* Avoid gloves, very dark nails or strong rings, and fast whipping movements: motion blur makes MediaPipe lose the hand.
* **Rest your elbow** on the desk: small, relaxed hand movements are steadier than an outstretched arm.

## 7. Calibration, sensitivity and thresholds

A smaller rectangle of the camera image (blue box in the preview) defines how far your hand must travel to cross the screen.
**Calibration wizard** (`c`) stores it in `calibration.json`; **`+` / `−`** change sensitivity; delete the file to reset.

Everything else lives in **`config.py`**, whose header contains a *symptom → setting* table. Run
`python main.py --dry-run --debug`: for each hand the overlay shows its pose, `idx` / `mid` (thumb ↔ index / middle distance in
palm lengths), five finger bits (thumb…pinky, 1 = straight), `L` / `R` (1 = the left-click / right-click pinch is confirmed) and
`v` (hand speed). Do the gesture, read the numbers, and put the threshold between "gesture" and "no gesture".

| Setting | Meaning |
|---------|---------|
| `PINCH_CURSOR_GAIN`, `PINCH_MOVE_START_DIST` | extra cursor speed; how far the pinched hand must travel before the cursor moves (keeps taps still) |
| `CLICK_PINCH_ON/OFF`, `CLICK_CONFIRM_FRAMES`, `CLICK_RELEASE_FRAMES` | left-click pinch distance thresholds and confirmation frames (0.25 / 0.40, 3, 2) |
| `RIGHT_CLICK_PINCH_ON/OFF`, `RIGHT_CLICK_CONFIRM_FRAMES`, `RIGHT_CLICK_MIN_INDEX_GAP` | right-click equivalents (0.22 / 0.36, 4, 0.45) |
| `CLICK_AMBIGUITY_MARGIN`, `CLICK_START_MAX_SPEED`, `MIN_TRACK_FRAMES` | left/right disambiguation, fast-motion gate, minimum tracking time |
| `TAP_MIN_S`, `TAP_MAX_S`, `RIGHT_CLICK_MAX_S` | tap duration limits |
| `DOUBLE_CLICK_WINDOW_S`, `CLICK_MIN_INTERVAL_S`, `PINCH_REARM_S` | double-click gap, minimum time between clicks, re-arm time |
| `GRAB_HOLD_S`, `GRAB_REQUIRES_OPEN_WITHIN_S`, `DROP_MIN_FINGERS`, `DROP_CONFIRM_FRAMES` | fist grab time, "closed from an open hand" window, open-hand drop |
| `PINCH_ON/OFF`, `FINGER_EXTENDED_RATIO`, `POSE_STABLE_FRAMES` | pinch *pose* (zoom / volume / brightness), how straight a finger must be, pose debounce |
| `ONE_EURO_*`, `CURSOR_AVERAGE_FRAMES`, `CURSOR_DEADZONE_PX`, `LANDMARK_SMOOTHING` | anti-jitter stack (unchanged) |
| `SCROLL_*`, `WINDOW_*`, `ZOOM_STEP_NORM`, `LEVEL_*` | scroll, Alt+Tab, zoom, volume / brightness |
| `TWO_HAND_ENGAGE_S`, `MACRO_HOLD_S`, `PAUSE_HOLD_S`, `MEDIA_HOLD_S`, `ENABLE_*` | two-hand hold times and switches |
| `MACROS` | your own shortcuts: `(left pose, right pose) → (label, keys)` |

---

## 8. How it works

```
 webcam ─► ThreadedCamera ─► mirror ─► HandTracker ─► GestureEngine ─► MouseController ─► OS
 (thread, newest frame)               (MediaPipe)    (events)          (pyautogui)
```

| Module | Responsibility |
|--------|----------------|
| `main.py` | camera thread, main loop, overlay (pinch ring, progress bar, HUD), hotkeys, CLI |
| `hand_tracker.py` | MediaPipe wrapper (legacy + Tasks backends), up to **2 hands**, handedness fix, independent Left / Right tracks, landmark smoothing |
| `gesture_recognizer.py` | finger states, pinch detection (pose + click-grade), one-hand and two-hand state machines → `Event`s |
| `mouse_controller.py` | executes events (relative cursor, clicks, keys, volume, brightness, media key); **the only OS-touching file** |
| `calibration.py` | hand-to-screen scale, sensitivity, calibration wizard, saved handedness |
| `filters.py` | One Euro / EMA / moving average + post-average |
| `config.py` | every threshold, with the tuning guide |
| `selftest.py` | 37 webcam-free tests |

Stability features

* **Anti-jitter stack (unchanged):** landmark EMA → One Euro (or EMA / moving average) → short moving average → pixel dead-zone.
* **Confirmation frames and debounce** on every discrete gesture; one-shot gestures (macros, pause, play/pause) fire once per hold.
* **Hold-to-confirm progress bars** for two-hand poses; while one is being confirmed, one-hand actions are suppressed so a half-finished
  two-hand pose can never click, move the cursor or grab.
* **Occlusion handling:** a lost hand is kept (red skeleton) for 0.35 s; an active drag or Alt hold survives short tracking drops and is
  released if the hand stays gone.
* **Performance:** capture on its own thread (always the newest frame), 640×480 MJPG, light per-frame maths. Check the FPS counter.

## 9. Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera does not open | `--camera 1`; close Zoom / Teams / browser tabs using it; check OS camera permission |
| Left / Right labels reversed | show both hands apart (auto-fix), or press **H**; see §5 |
| Cursor does not move | you must be **pinching** (index + thumb touching, tips clearly visible to the camera) *and* moving the hand; check `L` = 1 in `--debug` while pinching |
| Cursor moves a little before a click | raise `PINCH_MOVE_START_DIST` (e.g. 0.018) |
| Tap does not click | keep the hand still, tap faster than 0.35 s; open the fingers fully between taps; raise `CLICK_PINCH_ON` (e.g. 0.30) |
| Click fires when I only move | `PINCH_MOVE_START_DIST` down so movement is detected sooner; lower `TAP_MAX_S` |
| Right click mixes with left click | keep the index finger clearly away from the thumb; raise `CLICK_AMBIGUITY_MARGIN` / `RIGHT_CLICK_MIN_INDEX_GAP` |
| Right click is hard to trigger | `RIGHT_CLICK_PINCH_ON` up (e.g. 0.27), `RIGHT_CLICK_CONFIRM_FRAMES` = 3 |
| Fist does not grab | open the hand first, then close it; hold ~0.2 s; check the `fing` bits in `--debug` (all four 0); lower `GRAB_HOLD_S` |
| Drops while dragging | `DROP_CONFIRM_FRAMES` up, `DROP_MIN_FINGERS` up; keep fingers curled |
| FPS below 30 | add light (§6), `CAMERA_FOURCC = "MJPG"`, 640×480, close heavy apps |
| Cursor jitters | lower `ONE_EURO_MIN_CUTOFF`, raise `CURSOR_AVERAGE_FRAMES` / `CURSOR_DEADZONE_PX` |
| Two-hand mode starts by accident | keep the left hand out of view, raise `TWO_HAND_ENGAGE_S`, or set `ENABLE_*` to `False` |
| Play/pause does nothing on macOS | the video player needs focus (the gesture sends Space); Windows / Linux send the media key |
| Volume / brightness does nothing | check the console for `[system]` messages; brightness needs `pip install screen-brightness-control` |
| macOS: nothing is clicked | grant Accessibility permission to your terminal |
| Linux: nothing is clicked | log in with an X11 session |

## 10. Extending

* **New macro:** add `("THREE", "OPEN"): ("My macro", ["ctrl", "shift", "n"])` to `MACROS` in `config.py`.
* **New gesture:** add a pose in `HandAnalyzer._raw_pose`, emit an `Event` from `OneHandGestures` / `TwoHandGestures`, handle it in
  `MouseController._handlers`.
* **pynput instead of pyautogui:** only `MouseController._pg()` and `SystemControls` talk to the OS.

## 11. Limitations

Gesture accuracy depends on lighting and hand orientation; MediaPipe is trained for palms and fingers roughly facing the camera.
Thresholds are sensible defaults, not guarantees: tune them with `--debug` for your camera, hands and setup. This version was
verified with 37 synthetic-hand tests and a synthetic Monte-Carlo run (900/900 noisy taps registered correctly, 0 false clicks from
10 minutes of idle hands), not yet with a live webcam.
