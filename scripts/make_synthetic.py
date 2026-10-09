"""Render a synthetic "before/after" pair of room walkthrough videos with known changes.

Used to sanity-check the pipeline without a phone. Ground truth is written to gt.json.

    python scripts/make_synthetic.py --out examples/synthetic
"""
import argparse
import json
import os
import subprocess

import numpy as np
import open3d as o3d

W, H, F = 640, 480, 520.0
rng = np.random.default_rng(0)


def texture(base, size=512, seed=0):
    """Procedural multi-scale texture: gives SIFT plenty to latch onto."""
    r = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.float32)
    for s in (4, 8, 16, 32, 64):
        n = r.random((s, s, 3)).astype(np.float32)
        n = np.kron(n, np.ones((size // s, size // s, 1), np.float32))
        img += n / s ** 0.5
    img = img / img.max()
    img = 0.55 * np.array(base, np.float32) + 0.45 * img
    # sprinkle blobs/lines
    for _ in range(60):
        x, y = r.integers(0, size, 2)
        rad = r.integers(3, 18)
        yy, xx = np.ogrid[:size, :size]
        m = (xx - x) ** 2 + (yy - y) ** 2 < rad ** 2
        img[m] = r.random(3)
    return o3d.geometry.Image((np.clip(img, 0, 1) * 255).astype(np.uint8))


def box(size, center, color, seed):
    m = o3d.geometry.TriangleMesh.create_box(*size, create_uv_map=True, map_texture_to_each_face=True)
    m.translate(np.array(center) - np.array(size) / 2)
    return m, texture(color, seed=seed)


def sphere(r, center, color, seed):
    m = o3d.geometry.TriangleMesh.create_sphere(r, 40, create_uv_map=True)
    m.translate(center)
    return m, texture(color, seed=seed)


def cylinder(r, h, center, color, seed):
    m = o3d.geometry.TriangleMesh.create_cylinder(r, h, 40, 4, create_uv_map=True)
    m.translate(center)
    return m, texture(color, seed=seed)


def room():
    """5m x 4m x 2.6m room, z up. Static structure + furniture."""
    g = {}
    g["floor"] = box((5, 4, 0.05), (0, 0, -0.025), (0.55, 0.45, 0.35), 1)
    g["wall_n"] = box((5, 0.05, 2.6), (0, 2, 1.3), (0.8, 0.8, 0.7), 2)
    g["wall_s"] = box((5, 0.05, 2.6), (0, -2, 1.3), (0.7, 0.75, 0.8), 3)
    g["wall_e"] = box((0.05, 4, 2.6), (2.5, 0, 1.3), (0.75, 0.8, 0.7), 4)
    g["wall_w"] = box((0.05, 4, 2.6), (-2.5, 0, 1.3), (0.8, 0.7, 0.75), 5)
    g["table"] = box((1.2, 0.7, 0.75), (0.3, 0.6, 0.375), (0.5, 0.3, 0.15), 6)
    g["cabinet"] = box((0.5, 0.8, 1.6), (2.1, -1.2, 0.8), (0.3, 0.3, 0.5), 7)
    g["sofa"] = box((1.8, 0.8, 0.8), (-1.3, 1.5, 0.4), (0.2, 0.5, 0.3), 8)
    return g


def scenes():
    a = room()
    a["box_red"] = box((0.45, 0.45, 0.45), (-0.9, -0.6, 0.225), (0.85, 0.15, 0.1), 10)
    a["bin"] = cylinder(0.18, 0.5, (1.6, 0.0, 0.25), (0.15, 0.2, 0.85), 11)
    b = room()
    b["box_red"] = box((0.45, 0.45, 0.45), (0.9, -1.2, 0.225), (0.85, 0.15, 0.1), 10)  # moved
    b["ball"] = sphere(0.28, (-0.2, -1.1, 0.28), (0.9, 0.8, 0.1), 12)  # added
    # bin removed
    gt = {
        "moved": [{"name": "red box", "from": [-0.9, -0.6, 0.225], "to": [0.9, -1.2, 0.225]}],
        "added": [{"name": "yellow ball", "at": [-0.2, -1.1, 0.28]}],
        "removed": [{"name": "blue bin", "at": [1.6, 0.0, 0.25]}],
    }
    return a, b, gt


def look_at(eye, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.stack([r, -u, f])  # camera: x right, y down, z forward
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = -R @ eye
    return T


def trajectory(kind, n):
    t = np.linspace(0, 1, n)
    poses = []
    for s in t:
        if kind == "a":  # sweep along south side, looking north-ish across room
            ang = np.deg2rad(-150 + 120 * s)
            eye = np.array([1.6 * np.cos(ang) * 1.2, -1.5 + 0.2 * np.sin(6 * s), 1.4 + 0.05 * np.sin(9 * s)])
            eye[0] = -1.8 + 3.6 * s
            tgt = np.array([0.6 * np.cos(4 * s), 0.3, 0.5])
        else:  # loop along the walls looking into the room, lower height
            ang = np.deg2rad(-60 + 290 * s)
            eye = np.array([1.9 * np.cos(ang), 1.4 * np.sin(ang), 1.15 + 0.08 * np.sin(7 * s)])
            tgt = np.array([0.3 * np.cos(ang + 2.5), 0.3 * np.sin(ang + 2.5) - 0.2, 0.4])
        poses.append(look_at(eye, tgt))
    return poses


def render(geoms, poses, path):
    r = o3d.visualization.rendering.OffscreenRenderer(W, H)
    r.scene.set_background([1, 1, 1, 1])
    for name, (mesh, tex) in geoms.items():
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.albedo_img = tex
        r.scene.add_geometry(name, mesh, mat)
    K = o3d.camera.PinholeCameraIntrinsic(W, H, F, F, W / 2, H / 2)
    frames = os.path.splitext(path)[0] + "_frames"
    os.makedirs(frames, exist_ok=True)
    for i, T in enumerate(poses):
        r.setup_camera(K, T)
        img = np.asarray(r.render_to_image())
        o3d.io.write_image(f"{frames}/{i:04d}.png", o3d.geometry.Image(img))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "15", "-i", f"{frames}/%04d.png",
                    "-pix_fmt", "yuv420p", "-crf", "18", path], check=True)
    subprocess.run(["rm", "-rf", frames])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="examples/synthetic")
    ap.add_argument("--frames", type=int, default=90)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    a, b, gt = scenes()
    render(a, trajectory("a", args.frames), f"{args.out}/before.mp4")
    render(b, trajectory("b", args.frames), f"{args.out}/after.mp4")
    json.dump(gt, open(f"{args.out}/gt.json", "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
