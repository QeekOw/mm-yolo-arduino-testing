"""How models/yolov8n-pose.onnx was made (instructors only; students don't need this).

    pip install ultralytics onnx
    python tools/export_onnx.py

The stock Ultralytics export concatenates boxes (0..640 px), scores (0..1) and
keypoints into one (1, 56, 8400) tensor. Int8 quantization then uses one scale
for all of it and every score rounds to 0, so a w8a8 model detects nothing.
We cut the graph before that Concat and expose four separate outputs instead:

    boxes    (1, 4, 8400)      cx, cy, w, h in input pixels
    scores   (1, 1, 8400)      person confidence 0..1
    kpt_xy   (1, 17, 2, 8400)  keypoint x, y in input pixels
    kpt_conf (1, 17, 1, 8400)  keypoint confidence 0..1
"""

from pathlib import Path

import onnx
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "yolov8n-pose.onnx"
SPLIT = {  # tensor inside the Ultralytics graph -> new output name
    "/model.22/Mul_2_output_0": "boxes",
    "/model.22/Sigmoid_output_0": "scores",
    "/model.22/Mul_4_output_0": "kpt_xy",
    "/model.22/Sigmoid_1_output_0": "kpt_conf",
}

src = YOLO("yolov8n-pose.pt").export(format="onnx", imgsz=640, opset=17, simplify=True)
onnx.utils.extract_model(src, str(OUT), ["images"], list(SPLIT))

m = onnx.load(OUT)
for node in m.graph.node:
    node.input[:] = [SPLIT.get(n, n) for n in node.input]
    node.output[:] = [SPLIT.get(n, n) for n in node.output]
for o in m.graph.output:
    o.name = SPLIT[o.name]
# AI Hub rejects tensors listed both as outputs and in value_info
io = {v.name for v in list(m.graph.input) + list(m.graph.output)}
keep = [v for v in m.graph.value_info if v.name not in io and v.name not in SPLIT]
del m.graph.value_info[:]
m.graph.value_info.extend(keep)
onnx.checker.check_model(m, full_check=True)
onnx.save(m, OUT)
print("saved", OUT, [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
