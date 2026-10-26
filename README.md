# Wrist Angle

Measures wrist angle with an OAK-D Lite while someone lifts or carries a solar panel, so different handle designs can be compared.

It uses MediaPipe body pose to find the elbow and wrist, then runs the MediaPipe hand model on a crop around the wrist to find the hand. The angle is measured between the forearm (elbow → wrist) and the hand (wrist → middle-finger knuckle), relative to a neutral posture you calibrate at the start of each session. The body pose model's own hand points are too coarse for this: they lag behind when the wrist bends, so they read well under the real angle.

This is a 2D measurement, so camera placement decides what gets measured:

| Camera view | Measures |
|---|---|
| Side view, perpendicular to the palm | Flexion / extension |
| Top view, looking down at the back of the hand | Radial / ulnar deviation |

## Files

| File | Purpose |
|---|---|
| `wrist_angle.py` | Live capture (or a video file): shows the angle and records trials to CSV |
| `analyze.py` | Summarizes trials and plots them on one chart |
| `requirements.txt` | Python packages |
| `pose_landmarker_full.task` | Pose model, downloaded automatically on first run |
| `hand_landmarker.task` | Hand model, downloaded automatically on first run |

## Setup (Windows, one time)

```bat
cd C:\Users\melsj\Documents\GitHub\wrist-angle
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Python 3.14 works. depthai, mediapipe and opencv all have Windows packages for it.

If `python -m venv venv` fails with "Unable to copy ... venvlauncher.exe", a venv is already active in that terminal. Run `deactivate` first, or skip creating it and just activate the existing one.

## Each session

```bat
cd C:\Users\melsj\Documents\GitHub\wrist-angle
venv\Scripts\activate.bat
```

### Make sure Windows has the camera, not WSL

The camera was passed through to WSL earlier with usbipd. If it's still attached to WSL, Windows can't see it. In an **admin PowerShell**:

```powershell
usbipd list            # Movidius MyriadX should NOT say "Attached"
usbipd detach --busid 2-9
usbipd unbind --all    # optional: removes all usbipd bindings
```

The BUSID (`2-9`) is the USB port the camera is plugged into. It changes if you use a different port, so check `usbipd list`.

### Record trials

```bat
python wrist_angle.py --side right --out direct_grip.csv
```

Keys in the video window:

| Key | Action |
|---|---|
| `n` | Set neutral. Hold the wrist straight, with forearm and hand clearly visible, for about 1 s |
| `r` | Start / stop a trial. Each start begins a new trial number in the same CSV |
| `q` | Quit and save |

Calibration only counts frames where the elbow, wrist and knuckles are visible. If it stays on "calibrating neutral… hold still", the camera can't see the arm.

Useful options:

| Option | Meaning |
|---|---|
| `--side left` / `right` | Which of the subject's arms to track (default `right`) |
| `--out FILE.csv` | Output file (default `trial_<date>_<time>.csv`) |
| `--save-video` | Also save the annotated video as an `.mp4` next to the CSV |
| `--video FILE.mp4` | Analyze a recorded video instead of the live camera |
| `--flip-sign` | Invert the angle so extension reads as positive (see below) |
| `--threshold 15` | Angle in degrees beyond which the display turns red |

**Check the sign the first time.** Bend your wrist back (extension). If the angle goes negative, use `--flip-sign` from then on. The sign depends on which side the camera views from.

### Compare trials

```bat
python analyze.py direct_grip.csv handle_v1.csv --labels "Direct grip" "Handle v1"
```

This prints one row per trial and shows a plot. Add `--plot compare.png` to save the plot instead.

| Column | Meaning |
|---|---|
| `dur s` | Trial length |
| `tracked%` | Share of frames where the arm was found. Low means the panel was hiding the hand |
| `peak +` / `peak -` | Largest angle in each direction |
| `mean\|a\|` | Average size of the angle |
| `>15deg s` | Seconds spent beyond the threshold (change it with `--threshold`) |

Gaps in the plot are frames where the arm wasn't visible. Those frames are left blank in the CSV rather than guessed. `--smooth N` sets the median filter window in frames (default 5, 1 turns it off).

## CSV format

| Column | Meaning |
|---|---|
| `trial` | Trial number within the file |
| `time_s` | Seconds since the trial started |
| `raw_angle_deg` | Angle before neutral is subtracted |
| `angle_deg` | Angle relative to neutral (blank if not tracked) |
| `tracked` | 1 if the arm was visible, 0 if not |

## Experiment tips

- Start with a stationary side-view setup comparing two handles before trying to track someone walking.
- Put the camera roughly perpendicular to the plane of wrist bending, with the forearm and hand in full view.
- Recalibrate neutral (`n`) for each person and whenever the camera moves.
- If `tracked%` is often low because the panel hides the hand, stick small high-contrast markers on the back of the hand and forearm.
- This measures wrist posture, not load. To show that a handle reduces force on the wrist, pair it with a force sensor.

## Troubleshooting

**Video freezes or stutters.** On startup the console prints the USB speed, and the bottom-left of the window shows the frame rate.
- `USB speed: HIGH` means USB2. Use a USB3 port (usually blue) and the cable that came with the camera.
- If the frame rate is well under 15 fps, pose detection is too slow on this computer. A lighter pose model would help (not set up yet).

**"No available devices" or "X_LINK_DEVICE_NOT_FOUND".** The camera is probably still attached to WSL (see above), or another program has it open.

**Running from WSL instead of Windows.** This didn't work. The camera reboots every time a script connects, and the usbipd passthrough to WSL is too slow to re-attach it before it gives up. Use Windows.
