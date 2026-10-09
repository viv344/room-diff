"""Pick sharp, evenly spaced keyframes from a phone video."""
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np


def sharpness(img):
    g = cv2.cvtColor(cv2.resize(img, (320, int(320 * img.shape[0] / img.shape[1]))), cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()


def _source(video, max_side):
    """Return (frame_count, frame_iterator). OpenCV first; ffmpeg fallback for files it can't decode
    (e.g. 10-bit HDR HEVC from iPhones)."""
    cap = cv2.VideoCapture(video)
    ok = cap.isOpened() and cap.read()[0]
    if ok:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:  # missing metadata: count by grabbing
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            n = 1
            while cap.grab():
                n += 1
        cap.release()

        def it():
            c = cv2.VideoCapture(video)
            while True:
                good, f = c.read()
                if not good:
                    break
                yield f
            c.release()
        return n, it()
    cap.release()
    if not shutil.which("ffmpeg"):
        raise RuntimeError(f"OpenCV cannot decode {video} and ffmpeg is not installed")
    tmp = tempfile.mkdtemp()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", video, "-vf",
                    f"fps=10,scale='if(gt(iw,ih),{max_side},-2)':'if(gt(iw,ih),-2,{max_side})'",
                    "-q:v", "2", os.path.join(tmp, "%05d.jpg")], check=True)
    files = sorted(os.listdir(tmp))

    def it():
        for f in files:
            yield cv2.imread(os.path.join(tmp, f))
        shutil.rmtree(tmp, ignore_errors=True)
    return len(files), it()


def extract_frames(video, out_dir, n=40, max_side=960):
    """Split the video into n equal time bins and keep the sharpest frame of each bin.

    Motion blur is the main killer of SfM on handheld phone footage, so this matters more than
    uniform sampling. Returns the list of written file names (relative to out_dir).
    """
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(video):
        raise FileNotFoundError(video)
    total, frames = _source(video, max_side)
    bins = np.linspace(0, total, n + 1).astype(int)
    best = {}  # bin -> (score, frame)
    for i, frame in enumerate(frames):
        if frame is None:
            continue
        if max(frame.shape[:2]) > 2 * max_side:  # 4K: downscale early to keep memory in check
            s = 2 * max_side / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        b = min(int(np.searchsorted(bins, i, side="right") - 1), n - 1)
        sh = sharpness(frame)
        if b not in best or sh > best[b][0]:
            best[b] = (sh, frame)
    names = []
    for b in sorted(best):
        img = best[b][1]
        scale = max_side / max(img.shape[:2])
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        name = f"{b:04d}.jpg"
        cv2.imwrite(os.path.join(out_dir, name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        names.append(name)
    return names
