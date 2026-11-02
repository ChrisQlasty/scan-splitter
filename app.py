"""Scan Splitter - splits scanned pages with multiple photos into separate files."""
import os
import subprocess
import sys
import threading
import uuid
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, Response

BASE = Path(__file__).parent
app = Flask(__name__, static_folder=str(BASE / "static"))

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
STATE = {"folder": None, "crops": {}}  # crop id -> dict(img, name)


# ---------------------------------------------------------------- detection
def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)  # handles unicode paths on Windows
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect


def split_scan(img):
    """Return list of cropped, deskewed photos found on the scan."""
    h, w = img.shape[:2]
    scale = 1200.0 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.int16)
    lab = cv2.GaussianBlur(lab, (5, 5), 0)

    # Background colour = median of a thin frame around the scan (works for light and dark lids)
    b = max(4, int(min(small.shape[:2]) * 0.01))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    diff = np.abs(lab - bg).sum(axis=2).astype(np.float32)
    mask = (diff > 28).astype(np.uint8) * 255

    k = max(3, int(min(small.shape[:2]) * 0.006))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k * 3, k * 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = small.shape[0] * small.shape[1] * 0.015
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        rect = cv2.minAreaRect(c)
        boxes.append(rect)

    # Sort top-to-bottom, left-to-right
    boxes.sort(key=lambda r: (round(r[0][1] / (small.shape[0] / 4)), r[0][0]))

    crops = []
    for (cx, cy), (bw, bh), ang in boxes[:20]:
        # shrink slightly to remove background edges
        rect = ((cx / scale, cy / scale), (max(bw / scale - 6, 10), max(bh / scale - 6, 10)), ang)
        pts = order_points(cv2.boxPoints(rect))
        tw = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
        th = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
        dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(pts, dst)
        crops.append(cv2.warpPerspective(img, M, (tw, th), flags=cv2.INTER_CUBIC))
    if not crops:  # nothing detected -> keep whole image
        crops.append(img)
    return crops


# ---------------------------------------------------------------- routes
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/choose-folder")
def choose_folder():
    # Run the native folder dialog in a separate process (safe on macOS & Windows)
    code = ("import tkinter as tk; from tkinter import filedialog; r=tk.Tk(); r.withdraw(); "
            "r.attributes('-topmost', True); print(filedialog.askdirectory(title='Choose folder with scans'))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()
    if not out:
        return jsonify(folder=None)
    STATE["folder"] = out
    files = sorted(p.name for p in Path(out).iterdir() if p.suffix.lower() in EXTS)
    return jsonify(folder=out, files=files)


@app.post("/api/process")
def process():
    """Process one file (called per file so the UI can show progress)."""
    name = request.json["file"]
    img = read_image(Path(STATE["folder"]) / name)
    if img is None:
        return jsonify(error=f"Cannot read {name}"), 400
    result = []
    for i, crop in enumerate(split_scan(img), 1):
        cid = uuid.uuid4().hex
        STATE["crops"][cid] = {"img": crop, "name": f"{Path(name).stem}_{i:02d}.jpg"}
        result.append({"id": cid, "name": STATE["crops"][cid]["name"]})
    return jsonify(crops=result)


@app.get("/api/thumb/<cid>")
def thumb(cid):
    img = STATE["crops"][cid]["img"]
    s = 320.0 / max(img.shape[:2])
    t = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", t, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.post("/api/reset")
def reset():
    STATE["crops"].clear()
    return jsonify(ok=True)


@app.post("/api/save")
def save():
    items = request.json["items"]  # [{id, rotation, keep}]
    out_dir = Path(STATE["folder"]) / "split"
    out_dir.mkdir(exist_ok=True)
    rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    saved = 0
    for it in items:
        if not it.get("keep", True):
            continue
        c = STATE["crops"][it["id"]]
        img = c["img"]
        r = it.get("rotation", 0) % 360
        if r:
            img = cv2.rotate(img, rot[r])
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        buf.tofile(str(out_dir / c["name"]))
        saved += 1
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(out_dir)])
    elif os.name == "nt":
        os.startfile(str(out_dir))
    return jsonify(saved=saved, folder=str(out_dir))


if __name__ == "__main__":
    port = 5123
    threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    print(f"\nScan Splitter is running at http://127.0.0.1:{port}\nClose this window to quit.\n")
    app.run(host="127.0.0.1", port=port)
