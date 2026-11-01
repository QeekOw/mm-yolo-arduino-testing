# YOLOv8 Pose Detection on Arduino UNO Q

In this lab you run a **YOLOv8 pose model** (17 body keypoints per person) on a test video, first on your
**laptop** and then on the **Arduino UNO Q**, and watch the performance numbers on the side of the preview.
Then you use **Qualcomm AI Hub** to **change the model's runtime** and **quantize it**, plug the new model in,
and run it again to compare.

```
 Part 1  Run on laptop      ──►  baseline numbers (ONNX Runtime, float)
 Part 2  Run on UNO Q       ──►  same model, on the board (App Lab "Run" button)
 Part 3  AI Hub (you!)      ──►  convert: change runtime + quantize
 Part 4  Plug it in & rerun ──►  compare FPS / latency / accuracy
```

---

## What's in this repo

```
mm-yolo-arduino-testing/
├── app.yaml               ← makes this folder an Arduino App Lab app
├── python/
│   ├── main.py            ← the program (inference + preview + metrics)
│   └── requirements.txt   ← packages App Lab installs on the board
├── models/
│   └── yolov8n-pose.onnx  ← the model we give you (YOLOv8n-pose, 640x640, float32)
├── assets/
│   └── test_video.mp4     ← test video (45 s, jumping jacks + burpees)
├── requirements.txt       ← packages for your laptop
└── tools/export_onnx.py   ← how the .onnx was made (instructors only)
```

## What you need

| | |
|---|---|
| Laptop | macOS (Apple Silicon), Windows 10/11 or Linux, with **Python 3.10 – 3.13** |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | Python package manager used in this course (plain `pip` also works) |
| Arduino UNO Q | already set up once with **Arduino App Lab** (Wi-Fi connected, password set) |
| Same network | laptop and UNO Q on the **same Wi-Fi** |
| Qualcomm AI Hub account | free, at <https://aihub.qualcomm.com> (needed for Part 3) |

---

## Part 1 — Run it on your laptop

### 1.1 Get the code

```bash
git clone https://github.com/QeekOw/mm-yolo-arduino-testing.git
cd mm-yolo-arduino-testing
```

### 1.2 Install the packages

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

<details><summary>Without uv (plain pip)</summary>

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
then use `python ...` instead of `uv run python ...` below.
</details>

### 1.3 Run

```bash
uv run python python/main.py
```

A window opens with the video, the detected skeletons, and a **side panel**:

| Panel line | Meaning |
|---|---|
| Model / Size | which model file is running and how big it is |
| Runtime | the inference engine (ONNX Runtime or LiteRT/TFLite) |
| Input | model input size (640 × 640) |
| **FPS** | frames per second (smoothed), counting pre-process + inference + post-process |
| Avg FPS | average since the start |
| **Inference** | time the model itself takes for one frame (ms) |
| Pre-process | resize/pad the frame to 640 × 640 (ms) |
| Post-proc | pick the people, remove duplicates (NMS), scale keypoints back (ms) |
| People / Keypoints | how many people and confident keypoints in this frame |

Press **q** to quit (or wait for the video to end). A summary is printed and saved to
`results/metrics_<model>_<runtime>_<cpu>.json`. **Write down your numbers.**

Useful options:

```bash
uv run python python/main.py --view web        # show in the browser at http://localhost:8000 instead of a window
uv run python python/main.py --view none       # no preview, just measure (fastest)
uv run python python/main.py --video 0         # use your webcam instead of the test video
uv run python python/main.py --help            # all options
```

---

## Part 2 — Run it on the Arduino UNO Q

The board has no screen, so the program **streams the preview to your laptop's browser**:
the UNO Q does all the work, your laptop just watches.

### 2.1 Open a terminal on the board

Use either one:

- **Wi-Fi (SSH):** `ssh arduino@<board-name>.local`
  (the board name and password are the ones you chose in App Lab's first setup; if `.local` does not work,
  use the board's IP address)
- **USB cable (adb):** `adb shell` (adb ships with App Lab, or install Android platform-tools)

Find the board's IP address, you will need it later:

```bash
hostname -I        # e.g. 192.168.0.139   (the first 192.168.x.x / 10.x.x.x number)
```

### 2.2 Put this app on the board

Apps live in `~/ArduinoApps/`. Clone the repo **into that folder** on the board:

```bash
cd ~/ArduinoApps
git clone https://github.com/QeekOw/mm-yolo-arduino-testing.git yolov8-pose-test
```

That's it. The folder has an `app.yaml`, so App Lab sees it as an app called **YOLOv8 Pose Test 🏃**.

### 2.3 Run it

**Option A, App Lab (one click):** open Arduino App Lab on your laptop, connect to your board, go to
**My Apps → YOLOv8 Pose Test** and press **Run**.

**Option B, terminal (on the board):**

```bash
arduino-app-cli app start user:yolov8-pose-test
arduino-app-cli app logs user:yolov8-pose-test      # see what it prints
arduino-app-cli app stop user:yolov8-pose-test      # stop it (prints + saves the summary)
```

The **first run takes 1–2 minutes** because the board downloads `onnxruntime` and `ai-edge-litert`.
Later runs start in a few seconds.

### 2.4 Watch the result

On your **laptop**, open a browser at

```
http://<board-ip>:8000          e.g. http://192.168.0.139:8000
```

You get the same view as Part 1, now with numbers from the UNO Q. On the board the video **loops forever**.
The console (App Lab's console or `app logs`) prints the average FPS every 100 frames. When you press
**Stop**, the summary is printed and saved on the board in
`~/ArduinoApps/yolov8-pose-test/results/`.

> The UNO Q's CPU is much slower than a laptop. Expect about **1 FPS** with the float model.
> That is exactly why we optimize the model in Part 3.

---

## Part 3 — Change the runtime and quantize with AI Hub (your turn)

Now **you** convert our model `models/yolov8n-pose.onnx` with Qualcomm AI Hub. You pick two things:

**Runtime**: which engine runs the model on the device:

| Runtime | `RUNTIME =` | Output file | Works with `main.py` |
|---|---|---|---|
| ONNX Runtime | `"onnx"` | `.onnx.zip` | ✅ |
| LiteRT (TensorFlow Lite) | `"tflite"` | `.tflite` | ✅ |
| QAIRT (Qualcomm AI Runtime) | `"qnn_dlc"` | `.dlc` | ❌ needs the Qualcomm NPU runtime, not covered here |

**Precision**: quantization:

| Precision | `PRECISION =` | Meaning |
|---|---|---|
| float | `"float"` | not quantized, 32-bit floats (default) |
| w8a8 | `"w8a8"` | **8-bit weights and 8-bit activations**: about 4× smaller, faster on CPUs/NPUs, a little less accurate |

> **Which device?** The UNO Q is not in AI Hub's device list, so we pick **`Arduino VENTUNO Q`**
> (same Qualcomm Dragonwing family). For LiteRT and ONNX the output file is portable: it runs on the
> UNO Q's CPU and on your laptop.

### 3.1 Log in to AI Hub (once)

1. Sign in at <https://aihub.qualcomm.com> → **Settings** → copy your **API token**.
2. In the repo folder on your laptop:

```bash
uv run qai-hub configure --api_token <YOUR_API_TOKEN>
uv run qai-hub list-devices | grep -i arduino      # should show "Arduino VENTUNO Q"
```

### 3.2 Write the conversion script

Create a new file **`convert.py`** in the repo folder (next to `README.md`) and type in the following.
Read the comments; each block is one step of what AI Hub does.

```python
import cv2
import qai_hub as hub

# ---------- YOUR CHOICES ----------
RUNTIME = "tflite"     # "tflite" or "onnx"
PRECISION = "float"    # "float" or "w8a8"
# ----------------------------------

device = hub.Device("Arduino VENTUNO Q")
model = "models/yolov8n-pose.onnx"          # the model we gave you

# Step 1 (only for w8a8): quantize.
# AI Hub must see real example inputs ("calibration data") to choose good 8-bit ranges,
# so we give it 50 frames from the test video, preprocessed like the model expects.
if PRECISION == "w8a8":
    cap = cv2.VideoCapture("assets/test_video.mp4")
    frames = []
    for i in range(0, 1350, 27):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        frames.append(cv2.dnn.blobFromImage(frame, 1 / 255, (640, 640), swapRB=True))

    quantize_job = hub.submit_quantize_job(
        model=model,
        calibration_data={"images": frames},     # "images" = the model's input name
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT8,
    )
    model = quantize_job.get_target_model()      # waits until done

# Step 2: compile = convert the model to the runtime you chose.
compile_job = hub.submit_compile_job(
    model=model,
    device=device,
    options=f"--target_runtime {RUNTIME}",
)

# Step 3: download the result into models/ (AI Hub adds the right file extension).
path = compile_job.get_target_model().download(f"models/yolov8n-pose_{PRECISION}")
print("Saved:", path)
```

### 3.3 Run it

```bash
uv run python convert.py
```

It uploads the model, prints a link to the job (open it to see what AI Hub is doing), waits, and downloads
the result. It takes about 1–3 minutes (w8a8 takes longer). You should now have, for example:

```
models/yolov8n-pose_float.tflite
```

Do it **several times** with different choices so you can compare. Suggested set:

| # | RUNTIME | PRECISION | You get |
|---|---|---|---|
| 1 | `tflite` | `float` | `models/yolov8n-pose_float.tflite` |
| 2 | `tflite` | `w8a8`  | `models/yolov8n-pose_w8a8.tflite` |
| 3 | `onnx`   | `float` | `models/yolov8n-pose_float.onnx.zip` |

> Running into trouble? Open the job link that was printed. AI Hub shows the error message there.

---

## Part 4 — Plug the new model in and run again

### 4.1 On your laptop

Either pass the model on the command line:

```bash
uv run python python/main.py --model models/yolov8n-pose_float.tflite
uv run python python/main.py --model models/yolov8n-pose_w8a8.tflite
uv run python python/main.py --model models/yolov8n-pose_float.onnx.zip     # zip is fine, no need to unzip
```

or open `python/main.py` and change the line near the top:

```python
MODEL = "models/yolov8n-pose_w8a8.tflite"
```

The side panel now shows the new **Model**, **Size** and **Runtime**. Compare FPS and inference time with Part 1.

### 4.2 On the UNO Q

1. **Copy the model to the board** (run this on your **laptop**, in the repo folder):

   ```bash
   scp models/yolov8n-pose_w8a8.tflite arduino@<board-ip>:~/ArduinoApps/yolov8-pose-test/models/
   ```

   (USB alternative: `adb push models/yolov8n-pose_w8a8.tflite /home/arduino/ArduinoApps/yolov8-pose-test/models/`)

2. **Point the app at it.** App Lab cannot pass command-line options, so edit the `MODEL` line.
   In App Lab, open **YOLOv8 Pose Test → python/main.py**, or on the board run
   `nano ~/ArduinoApps/yolov8-pose-test/python/main.py`:

   ```python
   MODEL = "models/yolov8n-pose_w8a8.tflite"
   ```

3. **Run** again (App Lab **Run** button, or `arduino-app-cli app restart user:yolov8-pose-test`) and refresh
   `http://<board-ip>:8000` in your browser.

Let it run for at least 100 frames, then press **Stop** and read the summary. Repeat for each model.

### 4.3 Fill in your results

| Model file | Runtime | Precision | Size (MB) | Laptop FPS | Laptop inference (ms) | UNO Q FPS | UNO Q inference (ms) |
|---|---|---|---|---|---|---|---|
| yolov8n-pose.onnx | ONNX Runtime | float | 13.5 | | | | |
| yolov8n-pose_float.tflite | LiteRT | float | | | | | |
| yolov8n-pose_w8a8.tflite | LiteRT | w8a8 | | | | | |
| yolov8n-pose_float.onnx.zip | ONNX Runtime | float | | | | | |

Questions to think about:

1. Which change helped more on the UNO Q, the **runtime** or the **precision**? Why might that be?
2. Watch the skeletons with the w8a8 model. Can you see any accuracy difference? Where (fast motion, crouching)?
3. The model's inference time is only part of the total. What else costs time on the board?

<details><summary>Instructor reference numbers (measured while preparing this lab)</summary>

| Model | MacBook (Apple M-series, CPU) inference | UNO Q inference |
|---|---|---|
| yolov8n-pose.onnx (float) | ~19 ms | BOARD_ONNX |
| yolov8n-pose_float.tflite | ~16 ms | BOARD_TFLITE |
| yolov8n-pose_w8a8.tflite | ~11 ms | BOARD_W8A8 |
| yolov8n-pose_float.onnx.zip | ~20 ms | BOARD_ONNXZIP |

w8a8 vs float on the test video: same people detected, keypoints move by about 6 px on average (640 × 360 frame).
</details>

---

## Troubleshooting

| Problem | Fix |
|---|---|
| App Lab: app status **failed** | Read the console / `arduino-app-cli app logs user:yolov8-pose-test`. Usually a wrong `MODEL` path (the file is not in `models/` on the board). |
| `Model not found: ...` | Check the file name. On the board: `ls ~/ArduinoApps/yolov8-pose-test/models/` |
| Browser can't open `http://<board-ip>:8000` | Same Wi-Fi? Correct IP (`hostname -I` on the board)? Is the app running (`arduino-app-cli app ps`)? Wait for the first-run install to finish. |
| Laptop: `No module named ai_edge_litert` | Intel Macs have no LiteRT package: run the `.tflite` models on the board only. |
| Laptop window doesn't open (Linux / WSL / remote) | Use `--view web` and open `http://localhost:8000`. |
| `qai-hub`: authentication error | Run `uv run qai-hub configure --api_token ...` again with a fresh token. |
| w8a8 model finds **no people** | You are probably quantizing a *stock* Ultralytics export. See "Why the model has 4 outputs" below. |
| Don't add `numpy` / `opencv` to `python/requirements.txt` | App Lab's Python image already ships its own OpenCV build; listing them breaks the install. |

## How it works (short)

1. **Pre-process**: the frame is letterboxed (resized + gray padding) to 640 × 640, BGR → RGB, values 0–1.
2. **Inference**: the model outputs 8400 candidate detections.
3. **Post-process**: keep candidates with score > 0.25, remove duplicates with NMS (IoU 0.45), and scale boxes
   and keypoints back to the original frame.
4. **Draw** skeletons and the side panel, then show it in a window or stream it as MJPEG to the browser.

**Why the model has 4 outputs.** The normal Ultralytics export packs box coordinates (0–640 pixels) and
confidence scores (0–1) into **one** output tensor. When you quantize to 8 bits, that tensor gets a single scale
of about 640 / 256 ≈ 2.5, so every score from 0 to 1 rounds to **0**, and the w8a8 model finds nobody.
Our model (`tools/export_onnx.py`) splits the output into `boxes`, `scores`, `kpt_xy` and `kpt_conf`, so each
part gets its own scale and quantization works. Remember this when you quantize your own models!

## Credits

- Model: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) `yolov8n-pose` (AGPL-3.0).
- Test video: "[Jumping jacks and burpees](https://commons.wikimedia.org/wiki/File:Jumping_jacks_and_burpees.webm)"
  by Taco fleur, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0), via Wikimedia Commons.
  Converted to MP4 and black bars cropped.
