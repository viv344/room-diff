"""Score a run on the synthetic scene: align true camera centres to the reconstruction (Umeyama
similarity), map ground-truth objects into the reconstructed frame, and print a top-down debug image.

    python scripts/eval_synthetic.py OUT_DIR [--frames 90 --keyframes 40]
"""
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from make_synthetic import trajectory  # noqa: E402


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = np.trace(np.diag(sig) @ E) / S.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--gt", default="examples/synthetic/gt.json")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--keyframes", type=int, default=40)
    a = ap.parse_args()
    sc = pickle.load(open(os.path.join(a.out, "work", "scene.pkl"), "rb"))
    bins = np.linspace(0, a.frames, a.keyframes + 1).astype(int)
    src, dst = [], []
    for sess, kind in (("before", "a"), ("after", "b")):
        poses = trajectory(kind, a.frames)
        for v in sc["views"]:
            if v.session == sess:
                T = poses[bins[int(os.path.basename(v.name)[:4])]]
                src.append(-T[:3, :3].T @ T[:3, 3])
                dst.append(v.center)
    s, R, t = umeyama(np.array(src), np.array(dst))
    err = np.linalg.norm((s * np.array(src) @ R.T + t) - np.array(dst), axis=1)
    print(f"camera alignment: scale {s:.3f} (true room units -> recon), RMS err {np.sqrt((err**2).mean()) / s:.3f} m (true)")
    f = lambda p: s * R @ np.array(p) + t
    gt = json.load(open(a.gt))
    truth = [("removed", f(m["from"]), m["name"]) for m in gt["moved"]] + \
            [("added", f(m["to"]), m["name"]) for m in gt["moved"]] + \
            [("added", f(m["at"]), m["name"]) for m in gt["added"]] + \
            [("removed", f(m["at"]), m["name"]) for m in gt["removed"]]
    res = json.load(open(os.path.join(a.out, "changes.json"))) if os.path.exists(os.path.join(a.out, "changes.json")) else None
    if res:
        det = []
        for c in res["changes"]:
            if c["type"] == "moved":
                det += [("removed", np.array(c["from"]), c["label"]), ("added", np.array(c["to"]), c["label"])]
            else:
                det.append((c["type"], np.array(c["centroid"]), c["label"]))
        hit = set()
        for kind, p, name in truth:
            d = [np.linalg.norm(q - p) / s if k == kind else 1e9 for k, q, _ in det]
            j = int(np.argmin(d)) if d else -1
            ok = j >= 0 and d[j] < 0.4
            if ok:
                hit.add(j)
            print(f"  GT {kind:7s} {name:12s} -> {'FOUND as ' + repr(det[j][2]) + f' ({d[j]:.2f} m off)' if ok else 'MISSED'}")
        print(f"recall {sum(1 for _ in hit)}/{len(truth)} ; false positives {len(det) - len(hit)}")
        print("moved pairs:", [(c['label'], round(c['distance_m'], 2)) for c in res['changes'] if c['type'] == 'moved'],
              "(true distance %.2f m)" % np.linalg.norm(np.subtract(gt['moved'][0]['to'], gt['moved'][0]['from'])))
    # top-down debug image
    P, C = sc["clouds"]["before"]
    Q, _ = sc["clouds"]["after"]
    allp = np.vstack([P, Q])
    lo, hi = allp[:, :2].min(0), allp[:, :2].max(0)
    sz = 700 / (hi - lo).max()
    img = np.full((720, 720, 3), 255, np.uint8)
    px = lambda p: (int((p[0] - lo[0]) * sz) + 10, int((hi[1] - p[1]) * sz) + 10)
    for p in P[::3]:
        img[px(p)[1], px(p)[0]] = (200, 200, 200)
    if res:
        for c in res["changes"]:
            col = {"removed": (0, 0, 255), "added": (0, 160, 0), "moved": (255, 120, 0)}[c["type"]]
            if c["type"] == "moved":
                cv2.arrowedLine(img, px(c["from"]), px(c["to"]), col, 2)
            else:
                cv2.circle(img, px(c["centroid"]), 6, col, 2)
            cv2.putText(img, c["label"][:14], px(c.get("to", c.get("centroid"))), 0, 0.4, col, 1)
    for kind, p, name in truth:
        cv2.drawMarker(img, px(p), (0, 0, 0), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(img, f"GT {kind[0]} {name}", (px(p)[0] + 6, px(p)[1] + 14), 0, 0.4, (0, 0, 0), 1)
    cv2.imwrite(os.path.join(a.out, "eval_topdown.png"), img)
    print("wrote", os.path.join(a.out, "eval_topdown.png"))


if __name__ == "__main__":
    main()
