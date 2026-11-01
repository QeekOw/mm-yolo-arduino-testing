# YOLOv8 Pose Detection on Arduino UNO Q

In this lab you run a **YOLOv8 pose model** (17 body keypoints per person) on the **Arduino UNO Q**.
The board processes a test video, and **you watch the result live in your laptop's browser**, with the
performance numbers in a panel on the side. Then you use **Qualcomm AI Hub** to **change the model's
runtime** and **quantize it**, put the new model on the board, and run it again to compare.

```
 Part 1  Run on the UNO Q         board runs the model ──stream──► your laptop's browser shows it
 Part 2  AI Hub (your turn)       on your laptop: change runtime + quantize the model
 Part 3  Plug it in & rerun       copy the new model to the board, run again, compare the numbers
```

> **Where does the work happen?** On the **UNO Q**. The board decodes the video, runs the model, draws
> the skeletons and the side panel, and sends the finished frames over Wi-Fi. Your laptop's browser only
> **displays** them. The side panel says `Device: UNO Q aarch64 x4` so you can always tell.

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
├── requirements.txt       ← packages for your laptop (Part 2)
└── tools/export_onnx.py   ← how the .onnx was made (instructors only)
```

## What you need

| | |
|---|---|
| Arduino UNO Q | already set up once with **Arduino App Lab** (Wi-Fi connected, board name + password set) |
| Laptop | on the **same Wi-Fi** as the board, with Arduino App Lab and a web browser |
| Python 3.10 – 3.13 + [uv](https://docs.astral.sh/uv/getting-started/installation/) | on the laptop, for Part 2 (plain `pip` also works) |
| Qualcomm AI Hub account | free, at <https://aihub.qualcomm.com> (Part 2) |

---

## Part 1 — Run it on the UNO Q

### 1.1 Open a terminal on the board

Use either one, from your laptop:

- **Wi-Fi (SSH):** `ssh arduino@<board-name>.local`
  (board name and password are the ones you chose in App Lab's first setup)
- **USB cable (adb):** `adb shell`

Find the board's IP address (you may need it later):

```bash
hostname -I        # e.g. "192.168.0.139 172.17.0.1 ..."  → use the first one
```

### 1.2 Put this app on the board

Apps live in `~/ArduinoApps/` on the board. Clone this repo **into that folder**:

```bash
cd ~/ArduinoApps
git clone https://github.com/QeekOw/mm-yolo-arduino-testing.git yolov8-pose-test
```

That's it. The folder contains an `app.yaml`, so App Lab now lists it as **YOLOv8 Pose Test 🏃**.

### 1.3 Run it

**Option A, App Lab (one click):** open Arduino App Lab on your laptop, connect to your board, go to
**My Apps → YOLOv8 Pose Test** and press **Run**.

**Option B, terminal (on the board):**

```bash
arduino-app-cli app start user:yolov8-pose-test     # start
arduino-app-cli app logs  user:yolov8-pose-test     # see what it prints
arduino-app-cli app stop  user:yolov8-pose-test     # stop (prints + saves the summary)
```

The **first run takes 1–2 minutes** because the board installs `onnxruntime` and `ai-edge-litert`.
Later runs start in a few seconds.

### 1.4 Watch the result on your laptop

The board serves the preview on port **8000**. On your **laptop**, open a browser at

```
http://<board-name>.local:8000          e.g.  http://mmlab.local:8000
http://<board-ip>:8000                  e.g.  http://192.168.0.139:8000   (if .local doesn't work)
```

> ⚠️ Not `localhost`: `localhost` means your own laptop, and nothing runs there.

You'll see the video with skeletons, plus this **side panel** (all numbers measured on the UNO Q):

| Panel line | Meaning |
|---|---|
| Model / Size | which model file is running and how big it is |
| Runtime | the inference engine (ONNX Runtime or LiteRT/TFLite) |
| Input | model input size (640 × 640) |
| Device | `UNO Q aarch64 x4` = the board's 4-core Arm CPU did the work |
| **FPS** | frames per second (smoothed), counting pre-process + inference + post-process |
| Avg FPS | average since the start |
| **Inference** | time the model itself takes for one frame (ms) |
| Pre-process | resize/pad the frame to 640 × 640 (ms) |
| Post-proc | pick the people, remove duplicates (NMS), scale keypoints back (ms) |
| People / Keypoints | how many people and confident keypoints in this frame |

The video **loops forever**. The console (App Lab's console, or `app logs`) prints the average every 100
frames. Let it run for **at least 100 frames**, then press **Stop**. The summary is printed and saved on
the board in `~/ArduinoApps/yolov8-pose-test/results/`. **Write down your numbers** (table in Part 3).

> Expect about **1 FPS** (≈ 770 ms per frame) with this float ONNX model. A phone-class CPU running a
> 640 × 640 model is slow. That is exactly what Part 2 is about.

---

## Part 2 — Change the runtime and quantize with AI Hub (your turn)

This part runs on your **laptop**. You convert our model `models/yolov8n-pose.onnx` with Qualcomm AI Hub
and pick two things:

**Runtime**: which engine runs the model on the device:

| Runtime | `RUNTIME =` | Output file | Works with `main.py` |
|---|---|---|---|
| ONNX Runtime | `"onnx"` | `.onnx.zip` | ✅ |
| LiteRT (TensorFlow Lite) | `"tflite"` | `.tflite` | ✅ |
| QAIRT (Qualcomm AI Runtime) | `"qnn_dlc"` | `.dlc` | ❌ needs Qualcomm's NPU runtime, not covered here |

**Precision**: quantization:

| Precision | `PRECISION =` | Meaning |
|---|---|---|
| float | `"float"` | not quantized, 32-bit floats (default) |
| w8a8 | `"w8a8"` | **8-bit weights and 8-bit activations**: about 4× smaller and faster, slightly less accurate |

> **Which device?** The UNO Q is not in AI Hub's device list, so we use **`Arduino VENTUNO Q`**
> (same Qualcomm Dragonwing family). LiteRT and ONNX files are portable and run on the UNO Q's CPU.

### 2.1 Set up your laptop (once)

```bash
git clone https://github.com/QeekOw/mm-yolo-arduino-testing.git
cd mm-yolo-arduino-testing
uv venv --python 3.12
uv pip install -r requirements.txt
```

<details><summary>Without uv (plain pip)</summary>

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
then type `python ...` instead of `uv run python ...` below.
</details>

### 2.2 Log in to AI Hub (once)

1. Sign in at <https://aihub.qualcomm.com> → **Settings** → copy your **API token**.
2. In the repo folder:

```bash
uv run qai-hub configure --api_token <YOUR_API_TOKEN>
uv run qai-hub list-devices
```

### 2.3 Write the conversion script

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
# AI Hub needs real example inputs ("calibration data") to choose good 8-bit ranges,
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

### 2.4 Run it

```bash
uv run python convert.py
```

It uploads the model, prints a link to the job (open it to watch AI Hub work), waits, and downloads the
result. This takes about 1–3 minutes (w8a8 takes longer). Run it **several times**, changing
`RUNTIME` / `PRECISION` each time:

| # | RUNTIME | PRECISION | You get |
|---|---|---|---|
| 1 | `tflite` | `float` | `models/yolov8n-pose_float.tflite` |
| 2 | `tflite` | `w8a8`  | `models/yolov8n-pose_w8a8.tflite` |
| 3 | `onnx`   | `float` | `models/yolov8n-pose_float.onnx.zip` |

> Something failed? Open the job link that was printed. AI Hub shows the error there.

---

## Part 3 — Put the new model on the UNO Q and run again

### 3.1 Copy the model to the board

On your **laptop**, in the repo folder:

```bash
scp models/yolov8n-pose_w8a8.tflite arduino@<board-name>.local:~/ArduinoApps/yolov8-pose-test/models/
```

(USB alternative: `adb push models/yolov8n-pose_w8a8.tflite /home/arduino/ArduinoApps/yolov8-pose-test/models/`)

### 3.2 Point the app at it

App Lab can't pass command-line options, so you change one line at the top of `python/main.py`.
In App Lab open **YOLOv8 Pose Test → python/main.py**, or on the board run
`nano ~/ArduinoApps/yolov8-pose-test/python/main.py`:

```python
MODEL = "models/yolov8n-pose_w8a8.tflite"       # was "models/yolov8n-pose.onnx"
```

(`.onnx.zip` files work as they are. No need to unzip.)

### 3.3 Run and compare

If the app is still running, **Stop** it first, then press **Run** in App Lab. In the terminal:
`arduino-app-cli app stop user:yolov8-pose-test` then `arduino-app-cli app start user:yolov8-pose-test`
(don't use `app restart`: in App CLI 0.14 it stops the app but fails to start it again).
Then refresh `http://<board-name>.local:8000` on your laptop. The panel now shows the new **Model**, **Size** and
**Runtime**. Let it run 100+ frames, press **Stop**, and fill in the table. Repeat for each model.

| Model file | Runtime | Precision | Size (MB) | UNO Q FPS | UNO Q inference (ms) | Skeletons look OK? |
|---|---|---|---|---|---|---|
| yolov8n-pose.onnx | ONNX Runtime | float | 13.5 | | | |
| yolov8n-pose_float.tflite | LiteRT | float | | | | |
| yolov8n-pose_w8a8.tflite | LiteRT | w8a8 | | | | |
| yolov8n-pose_float.onnx.zip | ONNX Runtime | float | | | | |

Questions to think about:

1. Which change helped more on the UNO Q, the **runtime** or the **precision**? Why might that be?
2. AI Hub's ONNX output vs our original ONNX: same speed or not? What did AI Hub change?
3. Watch the skeletons with w8a8. Do you see accuracy loss? Where (fast motion, crouching)?
4. Inference is only part of the total time per frame. What else costs time on the board?

<details><summary>Instructor reference numbers (UNO Q, 4 GB, measured while preparing this lab)</summary>

| Model | Size | UNO Q inference | UNO Q FPS |
|---|---|---|---|
| yolov8n-pose.onnx (ONNX Runtime, float) | 13.5 MB | 771 ms | 1.2 |
| yolov8n-pose_float.onnx.zip (AI Hub ONNX, float) | 11.6 MB | 769 ms | 1.2 |
| yolov8n-pose_float.tflite (LiteRT, float) | 13.4 MB | 380 ms | 2.4 |
| yolov8n-pose_w8a8.tflite (LiteRT, w8a8) | 3.6 MB | 243 ms | 3.6 |

Pre-process ≈ 30 ms, post-process ≈ 3–6 ms on the board. w8a8 vs float on the test video: the same
people are found and keypoints move about 6 px on average (640 × 360 frame).
</details>

---

## Optional — run it on your laptop too

To compare the board with your laptop's CPU, run on the laptop (after 2.1):

```bash
uv run python python/main.py                                          # opens a window
uv run python python/main.py --model models/yolov8n-pose_w8a8.tflite  # any converted model
uv run python python/main.py --view web    # browser at http://localhost:8000 instead of a window
uv run python python/main.py --video 0     # your webcam instead of the test video
```

Here the panel says e.g. `Device: arm64 x10`, meaning your laptop did the work. Press **q** to quit.
Results go to `results/` on the laptop.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| App Lab: app status **failed** | Read App Lab's console or `arduino-app-cli app logs user:yolov8-pose-test`. Usually a wrong `MODEL` path (the file isn't in `models/` on the board). |
| `Model not found: ...` | Check the name: `ls ~/ArduinoApps/yolov8-pose-test/models/` on the board. |
| Browser can't open the preview | Use the **board's** address, not `localhost`. Same Wi-Fi? Try the IP instead of `.local`. App running (`arduino-app-cli app ps`)? First run still installing? Wait 1–2 min. |
| `ssh: Could not resolve hostname ....local` | Use the IP address (`hostname -I` via `adb shell`, or your router's device list). |
| Laptop: `No module named ai_edge_litert` | Intel Macs have no LiteRT package. Run `.tflite` models on the board only. |
| `qai-hub`: authentication error | Run `uv run qai-hub configure --api_token ...` again with a fresh token. |
| w8a8 model finds **no people** | You are probably quantizing a *stock* Ultralytics export. See "Why the model has 4 outputs" below. |
| Editing `python/requirements.txt` | Don't add `numpy` or `opencv`. App Lab's Python image ships its own OpenCV build, and listing them breaks the install. |

## How it works (short)

1. **Pre-process**: the frame is letterboxed (resized + gray padding) to 640 × 640, BGR → RGB, values 0–1.
2. **Inference**: the model outputs 8400 candidate detections.
3. **Post-process**: keep candidates with score > 0.25, remove duplicates with NMS (IoU 0.45), and scale
   boxes and keypoints back to the original frame.
4. **Draw** skeletons and the side panel. On the board, each frame is JPEG-encoded and streamed (MJPEG over
   HTTP, port 8000, declared in `app.yaml`) to whoever opens the page.

**Why the model has 4 outputs.** The normal Ultralytics export packs box coordinates (0–640 pixels) and
confidence scores (0–1) into **one** output tensor. When you quantize to 8 bits, that tensor gets a single
scale of about 640 / 256 ≈ 2.5, so every score from 0 to 1 rounds to **0** and the w8a8 model finds
nobody. Our model (`tools/export_onnx.py`) splits the output into `boxes`, `scores`, `kpt_xy` and
`kpt_conf`, so each gets its own scale and quantization works. Remember this when you quantize your own models!

## Credits

- Model: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) `yolov8n-pose` (AGPL-3.0).
- Test video: "[Jumping jacks and burpees](https://commons.wikimedia.org/wiki/File:Jumping_jacks_and_burpees.webm)"
  by Taco fleur, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0), via Wikimedia Commons.
  Converted to MP4 and black bars cropped.
