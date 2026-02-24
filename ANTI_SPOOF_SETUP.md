# Anti-Spoof Model Setup (MiniFASNet ONNX)

This project now supports a trained anti-spoof model using ONNX (MiniFASNet-style) with fallback to heuristic scoring.

## 1) Install packages

Use your project venv:

- `pip install -r requirements.txt`

This installs `onnxruntime` used for anti-spoof inference.

## 2) Add the ONNX model file

Place your model file at:

- `models/anti_spoof_minifasnet.onnx`

Or set a custom path in `.env` with:

- `ATTENDANCE_ANTI_SPOOF_MODEL_PATH=...`

## 3) Verify `.env` settings

Required anti-spoof keys:

- `ATTENDANCE_ANTI_SPOOF_ENABLED=true`
- `ATTENDANCE_ANTI_SPOOF_BACKEND=auto`
- `ATTENDANCE_ANTI_SPOOF_MODEL_PATH=models/anti_spoof_minifasnet.onnx`
- `ATTENDANCE_ANTI_SPOOF_INPUT_SIZE=80`
- `ATTENDANCE_ANTI_SPOOF_LIVE_INDEX=0`
- `ATTENDANCE_ANTI_SPOOF_USE_ONNXRUNTIME=true`
- `ATTENDANCE_ANTI_SPOOF_THRESHOLD=0.62`
- `ATTENDANCE_ANTI_SPOOF_REQUIRED_FRAMES=3`
- `ATTENDANCE_ANTI_SPOOF_MARGIN=0.02`
- `ATTENDANCE_ANTI_SPOOF_MIN_PASS_RATIO=0.67`

## 4) How runtime selection works

- If ONNX model loads successfully, anti-spoof uses ONNX inference.
- If ONNX model is missing/invalid, it falls back to the heuristic anti-spoof scorer.

## 5) Tuning tips

If real users are rejected too often:

- Lower `ATTENDANCE_ANTI_SPOOF_THRESHOLD` by 0.02 at a time.
- Lower `ATTENDANCE_ANTI_SPOOF_REQUIRED_FRAMES` to 2.

If spoof still passes too often:

- Raise `ATTENDANCE_ANTI_SPOOF_THRESHOLD` by 0.02 at a time.
- Raise `ATTENDANCE_ANTI_SPOOF_REQUIRED_FRAMES` to 4 or 5.
- Raise `ATTENDANCE_ANTI_SPOOF_MIN_PASS_RATIO` toward `0.8`.

## 6) Run test

- `python .\main.py recognize --auto-schedule --camera-index 1`

Check logs for anti-spoof pass/fail lines.
