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


def _gaps(prof, max_w):
    """Narrow background runs (real gaps between photos), as (start, end) pairs."""
    bgline = prof < 0.02
    runs, i, n = [], 0, len(prof)
    while i < n:
        if bgline[i]:
            j = i
            while j < n and bgline[j]:
                j += 1
            if i > 0 and j < n and j - i <= max_w:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _trim(raw, x, y, w, h, max_t):
    """Remove pure-background margins of a cell, but at most max_t px per side (keeps white skies)."""
    sub = raw[y:y + h, x:x + w] > 0
    rows, cols = sub.mean(axis=1) >= 0.02, sub.mean(axis=0) >= 0.02
    if not rows.any() or not cols.any():
        return None
    t = min(int(np.argmax(rows)), max_t)
    b = min(int(np.argmax(rows[::-1])), max_t)
    l = min(int(np.argmax(cols)), max_t)
    r = min(int(np.argmax(cols[::-1])), max_t)
    return (x + l, y + t, w - l - r, h - t - b)


def _xy_cut(raw, box, max_w, min_part, depth=0):
    """Recursively cut a region along narrow full-width/height background gaps."""
    x, y, w, h = box
    sub = raw[y:y + h, x:x + w] > 0
    if depth < 12:
        for axis in (1, 0):  # try horizontal cuts first, then vertical
            length = h if axis == 1 else w
            for a, b in _gaps(sub.mean(axis=axis), max_w):
                if a >= min_part and length - b >= min_part:
                    if axis == 1:
                        parts = [(x, y, w, a), (x, y + b, w, h - b)]
                    else:
                        parts = [(x, y, a, h), (x + b, y, w - b, h)]
                    return [q for p in parts for q in _xy_cut(raw, p, max_w, min_part, depth + 1)]
    return [box]


def split_scan(img, background="white"):
    """Return list of cropped photos found on the scan."""
    h, w = img.shape[:2]
    scale = min(1.0, 2000.0 / max(h, w))  # keep enough resolution so narrow gaps survive
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.int16)

    # Background chosen by the user: white (L=255) or black (L=0), neutral colour (a=b=128)
    bg = np.array([255 if background == "white" else 0, 128, 128])
    # Scanner lid is very uniform (white ~255, neutral), bright photo areas like sky are not,
    # so a tight threshold separates them. Black lids are noisier -> looser threshold.
    lab = cv2.medianBlur(lab.astype(np.uint8), 5).astype(np.int16)
    thr = 8 if background == "white" else 24
    raw = (np.abs(lab - bg).sum(axis=2) > thr).astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # remove dust only

    sh, sw = raw.shape
    max_w = int(max(sh, sw) * 0.04)    # gaps between photos are narrow
    min_part = int(min(sh, sw) * 0.08)  # a photo is not thinner than this
    page = _trim(raw, 0, 0, sw, sh, max(sh, sw))
    if page is None:
        return [img]

    # Pass 1: cut the page along narrow gaps, trim each cell, drop near-empty cells
    cells = []
    for c in _xy_cut(raw, page, max_w, min_part):
        t = _trim(raw, *c, max_t=max(sh, sw))
        if t and t[2] > 0 and t[3] > 0 and (raw[t[1]:t[1] + t[3], t[0]:t[0] + t[2]] > 0).mean() > 0.15:
            cells.append(t)
    if not cells:
        return [img]

    # Pass 2: all photos share one aspect ratio and size. Learn it from the cells (median is
    # robust to a few bad cells), then check every cell against it: a cell that holds several
    # photos (no visible gap) is cut into an n_x x n_y grid of expected-size photos.
    def grid(cw, ch, p_short, p_long):
        """How many expected-size photos fit in a cell (tries portrait and landscape)."""
        best = None
        for pw, ph in ((p_short, p_long), (p_long, p_short)):
            nx, ny = max(1, round(cw / pw)), max(1, round(ch / ph))
            err = abs(np.log(cw / nx / pw)) + abs(np.log(ch / ny / ph))
            if best is None or err < best[0]:
                best = (err, nx, ny)
        return best[1], best[2]

    # Start from the smallest cell (it cannot hold less than one photo), then refine with the
    # median per-photo size over all cells.
    sx, sy, sw0, sh0 = min(cells, key=lambda c: c[2] * c[3])
    p_short, p_long = min(sw0, sh0), max(sw0, sh0)
    for _ in range(2):
        per = []
        for _, _, cw, ch in cells:
            nx, ny = grid(cw, ch, p_short, p_long)
            per.append(sorted((cw / nx, ch / ny)))
        p_short, p_long = (float(v) for v in np.median(per, axis=0))
    boxes = []
    for x, y, cw, ch in cells:
        nx, ny = grid(cw, ch, p_short, p_long)
        for j in range(ny):
            for i in range(nx):
                bx, by, bw, bh = x + i * cw // nx, y + j * ch // ny, cw // nx, ch // ny
                if nx * ny > 1 and (raw[by:by + bh, bx:bx + bw] > 0).mean() < 0.3:
                    continue  # grid slot is empty scanner area
                boxes.append(((bx + bw / 2, by + bh / 2), (bw, bh), 0.0))

    # Sort top-to-bottom, left-to-right
    row_h = p_short * 0.6
    boxes.sort(key=lambda r: (round(r[0][1] / row_h), r[0][0]))

    crops = []
    for (cx, cy), (bw, bh), ang in boxes[:30]:
        # shrink slightly to remove background edges
        rect = ((cx / scale, cy / scale), (max(bw / scale - 4, 10), max(bh / scale - 4, 10)), ang)
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
    background = request.json.get("background", "white")
    img = read_image(Path(STATE["folder"]) / name)
    if img is None:
        return jsonify(error=f"Cannot read {name}"), 400
    result = []
    for i, crop in enumerate(split_scan(img, background), 1):
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
