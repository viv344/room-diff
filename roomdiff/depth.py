"""Dense depth per frame: monocular metric depth, then aligned to the SfM geometry.

Dense MVS in COLMAP needs CUDA. Instead we predict depth for every keyframe with Depth Anything V2
(metric, indoor) and fit a per-frame scale+shift so it agrees with the triangulated SfM points.
That gives dense depth that's consistent with the shared reconstruction (and its metric scale).
"""
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import least_squares
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


class DepthModel:
    def __init__(self, model=MODEL):
        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.proc = AutoImageProcessor.from_pretrained(model)
        self.net = AutoModelForDepthEstimation.from_pretrained(model).eval()

    @torch.no_grad()
    def __call__(self, rgb):
        inp = self.proc(images=Image.fromarray(rgb), return_tensors="pt")
        out = self.net(**inp).predicted_depth  # 1 x h x w, meters
        d = torch.nn.functional.interpolate(out[None], size=rgb.shape[:2], mode="bilinear", align_corners=False)
        return d[0, 0].numpy().astype(np.float32)


def sparse_depths(view):
    """Camera-frame depth of the SfM points this view observes, with their pixel coords."""
    X = view.sparse_xyz @ view.T_cw[:3, :3].T + view.T_cw[:3, 3]
    ok = X[:, 2] > 1e-6
    return view.sparse_uv[ok], X[ok, 2]


def fit_scale_shift(pred, target):
    """Robust fit target ~= a * pred + b (soft-L1 loss so mismatched / moved points don't dominate)."""
    a0 = np.median(target) / max(np.median(pred), 1e-6)
    res = least_squares(lambda p: (p[0] * pred + p[1] - target) / np.maximum(target, 1e-3), [a0, 0.0],
                        loss="soft_l1", f_scale=0.05)
    a, b = res.x
    rel = np.abs(a * pred + b - target) / target
    return a, b, float(np.median(rel))


def sample(img, uv):
    h, w = img.shape[:2]
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
    return img[v, u]


def edge_mask(depth, thresh=0.06):
    """Mask 'flying pixels' on depth discontinuities, which monocular models smear across edges."""
    ld = np.log(np.maximum(depth, 1e-3))
    gx = cv2.Sobel(ld, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(ld, cv2.CV_32F, 0, 1, ksize=3) / 8
    bad = np.hypot(gx, gy) > thresh
    return cv2.dilate(bad.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def correction_field(depth, uv, z, sigma_frac=0.12, prior=3.0):
    """Smooth multiplicative correction so dense depth bends to agree with the SfM points.

    Monocular depth is locally sharp but globally warped (curved walls, tilted floors). We take the
    log-ratio SfM/prediction at every sparse point, reject outliers, and spread it with a wide
    Gaussian (normalised convolution). `prior` pseudo-weight pulls the field to 1 far from any point.
    """
    h, w = depth.shape
    pred = sample(depth, uv)
    ok = pred > 0
    r = np.log(z[ok] / pred[ok])
    keep = np.abs(r - np.median(r)) < 0.25
    u = np.clip(np.round(uv[ok][keep, 0]).astype(int), 0, w - 1)
    v = np.clip(np.round(uv[ok][keep, 1]).astype(int), 0, h - 1)
    num = np.zeros((h, w), np.float32)
    den = np.zeros((h, w), np.float32)
    np.add.at(num, (v, u), r[keep])
    np.add.at(den, (v, u), 1.0)
    s = sigma_frac * max(h, w)
    k = int(3 * s) | 1
    num = cv2.GaussianBlur(num, (k, k), s)
    den = cv2.GaussianBlur(den, (k, k), s)
    peak = 1.0 / (2 * np.pi * s * s)  # density of one point at its centre
    return np.exp(num / (den + prior * peak)).astype(np.float32)


def estimate_metric_scale(views, raw):
    """SfM is up-to-scale; the metric depth model tells us roughly how many metres one SfM unit is."""
    scales = []
    for v in views:
        uv, z = sparse_depths(v)
        if len(z) < 20:
            continue
        p = sample(raw[v.name], uv)
        scales.append(np.median(z / np.maximum(p, 1e-3)))
    return 1.0 / float(np.median(scales))  # metres per SfM unit


def align_depths(views, raw, log=print):
    """Per-frame scale/shift alignment of raw metric predictions to the (already metric) SfM points."""
    errs = []
    for v in views:
        d = raw[v.name]
        uv, z = sparse_depths(v)
        if len(z) >= 20:
            a, b, err = fit_scale_shift(sample(d, uv), z)
            if not (0.3 < a < 3.0):  # bad fit: fall back to pure scale
                a, b, err = np.median(z / np.maximum(sample(d, uv), 1e-3)), 0.0, 1.0
        else:
            a, b, err = 1.0, 0.0, 1.0
        al = (a * d + b).astype(np.float32)
        if len(z) >= 20:
            al *= correction_field(al, uv, z)
        al[edge_mask(al)] = 0
        al[(al < 0.1) | (al > 8.0)] = 0
        v.depth = al.astype(np.float32)
        errs.append(err)
    log(f"[depth] aligned {len(views)} depth maps, median rel. error vs SfM {np.median(errs):.3f}")
