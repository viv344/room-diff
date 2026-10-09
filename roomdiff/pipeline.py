"""End-to-end: two videos -> shared 3D model -> changes -> labelled report."""
import os
import pickle
import time

import cv2
import numpy as np
import open3d as o3d

from . import SESSIONS
from .changes import detect
from .depth import DepthModel, align_depths, estimate_metric_scale
from .frames import extract_frames
from .fusion import fuse
from .geometry import gravity_align, transform_world
from .sfm import run_sfm


def reconstruct(videos, out, n_frames=40, log=print):
    """Stages 1-4. Cached in out/work/scene.pkl so semantics/report can be re-run cheaply."""
    work = os.path.join(out, "work")
    cache = os.path.join(work, "scene.pkl")
    if os.path.exists(cache):
        log(f"[cache] loading {cache} (delete it to recompute)")
        with open(cache, "rb") as f:
            return pickle.load(f)
    t0 = time.time()
    for s, vid in zip(SESSIONS, videos):
        names = extract_frames(vid, os.path.join(work, "images", s), n=n_frames)
        log(f"[frames] {s}: {len(names)} keyframes from {vid}")
    views, pts, rgb, both, img_dir = run_sfm(os.path.join(work, "images"), os.path.join(work, "sfm"), log=log)
    for v in views:
        v.image = cv2.cvtColor(cv2.imread(os.path.join(img_dir, v.name)), cv2.COLOR_BGR2RGB)

    log("[depth] running Depth Anything V2 (metric indoor) on every keyframe")
    model = DepthModel()
    raw = {v.name: model(v.image) for v in views}
    s = estimate_metric_scale(views, raw)
    log(f"[geom] metric scale: 1 SfM unit = {s:.3f} m")
    pts = transform_world(views, pts, np.eye(4), s)
    M = gravity_align(views, pts, log=log)
    pts = transform_world(views, pts, M)
    align_depths(views, raw, log=log)

    clouds = {}
    for sess in SESSIONS:
        clouds[sess] = fuse([v for v in views if v.session == sess])
        log(f"[fuse] {sess}: {len(clouds[sess].points)} points")
    scene = {"views": views, "sparse": (pts, rgb), "anchors": pts[both],
             "clouds": {k: (np.asarray(c.points), np.asarray(c.colors)) for k, c in clouds.items()}}
    with open(cache, "wb") as f:
        pickle.dump(scene, f)
    log(f"[time] reconstruction took {time.time() - t0:.0f}s")
    return scene


def to_pcd(pc):
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc[0]))
    p.colors = o3d.utility.Vector3dVector(pc[1])
    return p


def run(before, after, out, n_frames=40, use_claude=None, log=print):
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    scene = reconstruct([before, after], out, n_frames=n_frames, log=log)
    b, a = to_pcd(scene["clouds"]["before"]), to_pcd(scene["clouds"]["after"])
    clusters, masks = detect(b, a, scene["views"], scene["anchors"], log=log)

    from .report import write_json, write_report
    from .semantics import interpret
    changes, summary = interpret(clusters, scene["views"], use_claude=use_claude, log=log)

    nb = {s: sum(v.session == s for v in scene["views"]) for s in SESSIONS}
    stats = {"keyframes registered": f"{nb['before']} before / {nb['after']} after",
             "points": f"{len(b.points)} before / {len(a.points)} after",
             "before-scan seen by after": f"{1 - masks['unobserved_before'].mean():.0%}",
             "after-scan seen by before": f"{1 - masks['unobserved_after'].mean():.0%}",
             "runtime": f"{time.time() - t0:.0f}s"}
    write_report(os.path.join(out, "report.html"), scene, changes, summary, stats)
    write_json(os.path.join(out, "changes.json"), changes, summary, stats)
    o3d.io.write_point_cloud(os.path.join(out, "before.ply"), b)
    o3d.io.write_point_cloud(os.path.join(out, "after.ply"), a)
    log(f"\n{summary}")
    for i, c in enumerate(changes, 1):
        log(f"  #{i} [{c.type}] {c.description}")
    log(f"\nreport: {os.path.join(out, 'report.html')}")
    return changes, summary
