"""YOLOv8 pose detection test for the workshop: laptop and Arduino UNO Q.

Runs a YOLOv8-pose model on a video and shows the annotated result with a
side panel of performance metrics.

    python python/main.py                                  # defaults below
    python python/main.py --model models/yolov8n-pose.tflite
    python python/main.py --view web                       # view in browser

In Arduino App Lab you cannot pass arguments, so edit MODEL below instead.
"""

import argparse
import json
import os
import platform
import signal
import sys
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# EDIT THESE (App Lab users: this is where you switch models)
# ---------------------------------------------------------------------------
MODEL = "models/yolov8n-pose.onnx"   # after Part 3: "models/yolov8n-pose.tflite"
VIDEO = "assets/test_video.mp4"
CONF = 0.25          # minimum person confidence
PORT = 8000          # browser preview port (must match app.yaml)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
IN_APP_LAB = Path("/app/app.yaml").exists()  # running on the board inside Arduino App Lab
DEVICE = f"{'UNO Q ' if IN_APP_LAB else ''}{platform.machine()} x{os.cpu_count()}"
PANEL_W = 300
MIN_VIEW_H = 540
KPT_CONF = 0.5
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
            (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
            (3, 5), (4, 6)]


# ----------------------------------------------------------------- runtimes --
class OnnxModel:
    runtime = "ONNX Runtime (CPU)"

    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.name, self.nchw = inp.name, inp.shape[1] == 3
        self.size = inp.shape[2] if self.nchw else inp.shape[1]

    def __call__(self, x):
        return self.sess.run(None, {self.name: x})


class TfliteModel:
    runtime = "LiteRT / TFLite (CPU)"

    def __init__(self, path):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            from tflite_runtime.interpreter import Interpreter
        self.it = Interpreter(model_path=str(path), num_threads=os.cpu_count())
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.outs = self.it.get_output_details()
        shape = self.inp["shape"]
        self.nchw = shape[1] == 3
        self.size = int(shape[2] if self.nchw else shape[1])

    def __call__(self, x):
        scale, zero = self.inp["quantization"]
        if scale:  # quantized model: float -> int
            x = np.clip(np.round(x / scale + zero), *_int_range(self.inp["dtype"]))
        self.it.set_tensor(self.inp["index"], x.astype(self.inp["dtype"]))
        self.it.invoke()
        ys = []
        for out in self.outs:
            y = self.it.get_tensor(out["index"])
            scale, zero = out["quantization"]
            ys.append((y.astype(np.float32) - zero) * scale if scale else y)
        return ys


def _int_range(dtype):
    info = np.iinfo(dtype)
    return info.min, info.max


def load_model(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        sys.exit(f"Model not found: {path}")
    if path.suffix == ".zip":  # AI Hub's ONNX Runtime download: model.onnx + model.data in a zip
        folder = path.with_suffix("")
        if not folder.exists():
            zipfile.ZipFile(path).extractall(folder)
        return OnnxModel(next(folder.rglob("*.onnx"))), path
    if path.suffix == ".onnx":
        return OnnxModel(path), path
    if path.suffix == ".tflite":
        return TfliteModel(path), path
    sys.exit(f"Unsupported model type {path.suffix!r} (use .onnx, .onnx.zip or .tflite)")


# ------------------------------------------------------- pre/post-process --
def preprocess(frame, size, nchw):
    """Letterbox to size x size, RGB, 0..1 float. Returns tensor + undo info."""
    h, w = frame.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (size - nh) // 2, (size - nw) // 2
    img = np.full((size, size, 3), 114, np.uint8)
    img[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh))
    x = img[:, :, ::-1].astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1) if nchw else x
    return np.ascontiguousarray(x[None]), (r, left, top)


def merge_outputs(outs):
    """Model outputs -> one (56, N) array: cx, cy, w, h, score, 17 x (x, y, conf).

    Our model has 4 outputs (see tools/export_onnx.py); a stock Ultralytics
    export has a single (1, 56, N) one. Outputs are matched by shape, so the
    order a converter puts them in does not matter.
    """
    if len(outs) == 1:
        p = outs[0][0]
        return p if p.shape[0] == 56 else p.T
    parts = {}
    for o in outs:
        o = o[0]
        name = {(2, 4): "boxes", (2, 1): "scores", (3, 2): "kpt_xy", (3, 1): "kpt_conf"}[(o.ndim, o.shape[-2])]
        parts[name] = o
    kpts = np.concatenate([parts["kpt_xy"], parts["kpt_conf"]], axis=1).reshape(51, -1)
    return np.vstack([parts["boxes"], parts["scores"], kpts])


def postprocess(outs, undo, conf_th):
    """Model outputs -> list of (box, score, keypoints[17x3])."""
    p = merge_outputs(outs).T
    p = p[p[:, 4] > conf_th]
    if not len(p):
        return []
    r, left, top = undo
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    people = []
    for i in nms(boxes, p[:, 4], 0.45):
        box = (boxes[i] - [left, top, left, top]) / r
        kpts = p[i, 5:].reshape(17, 3).copy()
        kpts[:, 0] = (kpts[:, 0] - left) / r
        kpts[:, 1] = (kpts[:, 1] - top) / r
        people.append((box, float(p[i, 4]), kpts))
    return people


def nms(boxes, scores, iou_th):
    """Non-maximum suppression in numpy (the board's OpenCV has no cv2.dnn)."""
    order, keep = scores.argsort()[::-1], []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while len(order):
        i, rest = order[0], order[1:]
        keep.append(i)
        wh = np.clip(np.minimum(boxes[i, 2:], boxes[rest, 2:]) - np.maximum(boxes[i, :2], boxes[rest, :2]), 0, None)
        inter = wh[:, 0] * wh[:, 1]
        order = rest[inter / (area[i] + area[rest] - inter) <= iou_th]
    return keep


# ------------------------------------------------------------------ drawing --
def draw_people(frame, people):
    for box, score, kpts in people:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 160, 0), 2)
        cv2.putText(frame, f"person {score:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 160, 0), 1, cv2.LINE_AA)
        ok = kpts[:, 2] > KPT_CONF
        for a, b in SKELETON:
            if ok[a] and ok[b]:
                cv2.line(frame, tuple(kpts[a, :2].astype(int)), tuple(kpts[b, :2].astype(int)),
                         (0, 255, 255), 2, cv2.LINE_AA)
        for (x, y, _), good in zip(kpts, ok):
            if good:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1, cv2.LINE_AA)


def draw_panel(frame, lines):
    if frame.shape[0] < MIN_VIEW_H:  # small videos: enlarge so the panel fits
        s = MIN_VIEW_H / frame.shape[0]
        frame = cv2.resize(frame, (round(frame.shape[1] * s), MIN_VIEW_H))
    h = frame.shape[0]
    panel = np.full((h, PANEL_W, 3), 30, np.uint8)
    y = 30
    for text, style in lines:
        if style == "title":
            cv2.putText(panel, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
            y += 30
        elif style == "gap":
            y += 10
        else:
            cv2.putText(panel, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
            y += 22
    return np.hstack([frame, panel])


# ------------------------------------------------------------- web preview --
class Stream:
    """Holds the latest JPEG and serves it as an MJPEG stream on http://<host>:port/."""

    def __init__(self, port):
        self.jpg, self.cond = None, threading.Condition()
        stream = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path != "/stream":
                    body = (b"<html><head><title>YOLOv8 Pose</title></head>"
                            b"<body style='margin:0;background:#111;display:flex;justify-content:center'>"
                            b"<img src='/stream' style='max-width:100%'></body></html>")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with stream.cond:
                            stream.cond.wait()
                            jpg = stream.jpg
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()

    def show(self, frame):
        jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        with self.cond:
            self.jpg = jpg
            self.cond.notify_all()


def has_display():
    if IN_APP_LAB:
        return False
    return sys.platform in ("darwin", "win32") or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


# --------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL, help=".onnx, .onnx.zip or .tflite file (default: %(default)s)")
    ap.add_argument("--video", default=VIDEO, help="video file, or a camera index like 0")
    ap.add_argument("--view", choices=["auto", "window", "web", "none"], default="auto",
                    help="window = OpenCV window, web = browser at http://<ip>:PORT, none = metrics only")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--loop", action="store_true", help="replay the video forever (always on in web view)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # show logs live in the App Lab console
    signal.signal(signal.SIGTERM, signal.default_int_handler)  # App Lab "Stop" -> still print summary

    view = args.view if args.view != "auto" else ("window" if has_display() else "web")
    loop = args.loop or view == "web"

    model, model_path = load_model(args.model)
    src = int(args.video) if args.video.isdigit() else str(ROOT / args.video)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {src}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    size_mb = model_path.stat().st_size / 1e6
    print(f"Model   : {model_path.name} ({size_mb:.1f} MB)")
    print(f"Runtime : {model.runtime}")
    print(f"Input   : {model.size}x{model.size}  ({'NCHW' if model.nchw else 'NHWC'})")
    print(f"Device  : {platform.system()} {platform.machine()}, {os.cpu_count()} CPU cores")

    stream = None
    if view == "web":
        stream = Stream(args.port)
        if IN_APP_LAB:
            print(f"Preview : on your LAPTOP open http://<board-name>.local:{args.port} "
                  f"or http://<board-ip>:{args.port}  (board IP: run `hostname -I` on the board)")
        else:
            print(f"Preview : open http://localhost:{args.port} in your browser")
    print("Press q in the window (or Ctrl+C) to stop.\n")

    stats = {"pre": [], "infer": [], "post": [], "total": [], "people": []}
    fps_ema, frame_i, t_start = 0.0, 0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if loop and total > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_i = 0
                    continue
                break
            frame_i += 1

            t0 = time.perf_counter()
            x, undo = preprocess(frame, model.size, model.nchw)
            t1 = time.perf_counter()
            out = model(x)
            t2 = time.perf_counter()
            people = postprocess(out, undo, args.conf)
            t3 = time.perf_counter()

            ms = [(t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t3 - t0) * 1e3]
            for k, v in zip(("pre", "infer", "post", "total"), ms):
                stats[k].append(v)
            stats["people"].append(len(people))
            fps = 1000 / ms[3]
            fps_ema = fps if fps_ema == 0 else 0.9 * fps_ema + 0.1 * fps

            draw_people(frame, people)
            n = len(stats["total"])
            panel = [
                ("YOLOv8 POSE TEST", "title"),
                (f"Model:   {model_path.name}", ""),
                (f"Size:    {size_mb:.1f} MB", ""),
                (f"Runtime: {model.runtime}", ""),
                (f"Input:   {model.size}x{model.size}", ""),
                (f"Device:  {DEVICE}", ""),
                ("", "gap"),
                ("PERFORMANCE", "title"),
                (f"FPS:         {fps_ema:5.1f}", ""),
                (f"Avg FPS:     {1000 / np.mean(stats['total']):5.1f}", ""),
                (f"Inference:   {ms[1]:6.1f} ms", ""),
                (f"Pre-process: {ms[0]:6.1f} ms", ""),
                (f"Post-proc:   {ms[2]:6.1f} ms", ""),
                (f"Total:       {ms[3]:6.1f} ms", ""),
                ("", "gap"),
                ("RESULTS", "title"),
                (f"People:      {len(people)}", ""),
                (f"Keypoints:   {sum(int((k[:, 2] > KPT_CONF).sum()) for _, _, k in people)}", ""),
                (f"Frame:       {frame_i}/{total if total > 0 else '-'}", ""),
                (f"Frames done: {n}", ""),
            ]
            canvas = draw_panel(frame, panel)

            if view == "window":
                cv2.imshow("YOLOv8 Pose - press q to quit", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif stream:
                stream.show(canvas)
            if n % 100 == 0:
                print(f"[{n:5d} frames] avg FPS {1000 / np.mean(stats['total']):5.1f} | "
                      f"avg inference {np.mean(stats['infer']):6.1f} ms")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if view == "window":
            cv2.destroyAllWindows()

    if not stats["total"]:
        return
    summary = {
        "model": model_path.name,
        "model_size_mb": round(size_mb, 2),
        "runtime": model.runtime,
        "device": f"{platform.system()} {platform.machine()} ({os.cpu_count()} cores)",
        "frames": len(stats["total"]),
        "wall_time_s": round(time.perf_counter() - t_start, 1),
        "avg_fps": round(1000 / np.mean(stats["total"]), 2),
        "avg_inference_ms": round(float(np.mean(stats["infer"])), 2),
        "p95_inference_ms": round(float(np.percentile(stats["infer"], 95)), 2),
        "avg_preprocess_ms": round(float(np.mean(stats["pre"])), 2),
        "avg_postprocess_ms": round(float(np.mean(stats["post"])), 2),
        "avg_people_per_frame": round(float(np.mean(stats["people"])), 2),
    }
    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        print(f"{k:22s}: {v}")
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"metrics_{model_path.stem}_{model_path.suffix[1:]}_{platform.machine()}.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
