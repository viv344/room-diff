"""Name the changes, pair removed+added into "moved", and describe them in plain language.

Default labeller is CLIP (zero-shot over a household vocabulary) - runs locally on CPU.
If ANTHROPIC_API_KEY is set, Claude looks at the before/after crops instead and writes richer
labels and descriptions; CLIP is still used for the appearance embedding that drives matching.
"""
import base64
import io
import json
import os
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

from .changes import project, tolerance

VOCAB = [
    "chair", "office chair", "stool", "table", "desk", "sofa", "armchair", "bed", "pillow", "cushion", "blanket",
    "lamp", "floor lamp", "plant", "potted plant", "vase", "box", "cardboard box", "storage box", "basket",
    "bag", "backpack", "suitcase", "shoe", "shoes", "jacket", "towel", "book", "stack of books", "laptop",
    "monitor", "keyboard", "television", "speaker", "guitar", "bottle", "water bottle", "mug", "cup", "bowl",
    "plate", "kettle", "toaster", "microwave", "fan", "heater", "trash can", "bucket", "ball", "toy",
    "teddy bear", "umbrella", "picture frame", "clock", "mirror", "shelf", "cabinet", "drawer unit",
    "side table", "coat rack", "ladder", "vacuum cleaner", "broom", "bicycle", "helmet", "printer",
    "headphones", "phone", "remote control", "toolbox", "paint can", "folder", "paper", "rug", "doormat",
]

COLORS = {"removed": "#e5484d", "added": "#30a46c", "moved": "#3e63dd"}


@dataclass
class Change:
    type: str  # removed | added | moved
    label: str
    description: str
    centroid: list
    extent: list
    confidence: float
    points: np.ndarray = field(repr=False, default=None)  # removed/added: the cluster; moved: new position
    old_points: np.ndarray = field(repr=False, default=None)  # moved only
    crops: dict = field(default_factory=dict)  # {"before": jpeg bytes, "after": jpeg bytes}
    extra: dict = field(default_factory=dict)

    def to_json(self):
        d = {"type": self.type, "label": self.label, "description": self.description,
             "centroid": [round(x, 3) for x in self.centroid], "size_m": [round(x, 2) for x in self.extent],
             "confidence": round(self.confidence, 2)}
        d.update(self.extra)
        return d


# ---------------------------------------------------------------- views & crops

def best_view(points, views, need_visible=True):
    """View where the cluster is most visible: many unoccluded points, large, near the image centre."""
    best, best_score = None, -1
    for v in views:
        h, w = v.depth.shape
        uv, z = project(v, points)
        ok = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        if ok.sum() < 10:
            continue
        d = v.depth[uv[ok, 1].astype(int), uv[ok, 0].astype(int)]
        if need_visible:
            vis = np.abs(d - z[ok]) <= tolerance(z[ok])
        else:  # for the "other" session: we want views that looked through where the object was
            vis = d > z[ok] + tolerance(z[ok])
        q = uv[ok][vis]
        if len(q) < 10:
            continue
        c = np.abs(q.mean(0) / [w, h] - 0.5).max()  # 0 = centred
        score = len(q) * (1.2 - c) * (ok.mean())
        if score > best_score:
            best, best_score = (v, q), score
    return best


def crop_box(points, view, pad=0.35, min_px=96):
    h, w = view.depth.shape
    uv, z = project(view, points)
    uv = uv[z > 0.1]
    lo, hi = np.percentile(uv, 2, axis=0), np.percentile(uv, 98, axis=0)
    c, s = (lo + hi) / 2, np.maximum((hi - lo) * (1 + 2 * pad), min_px)
    s = np.full(2, s.max())  # square crops read better on cards
    x0, y0 = np.clip(c - s / 2, 0, [w - 1, h - 1]).astype(int)
    x1, y1 = np.clip(c + s / 2, 1, [w, h]).astype(int)
    return x0, y0, x1, y1


def crop_image(view, box, size=320):
    x0, y0, x1, y1 = box
    img = Image.fromarray(view.image[y0:y1, x0:x1])
    img.thumbnail((size, size))
    return img


def jpeg(img):
    b = io.BytesIO()
    img.save(b, "JPEG", quality=88)
    return b.getvalue()


def corners(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def make_crops(cluster_pts, own_views, other_views):
    """Crop of the object in its own session + crop of the same 3D box in the other session."""
    out, shown = {}, None
    bv = best_view(cluster_pts, own_views, need_visible=True)
    if bv:
        v, _ = bv
        out["own"] = crop_image(v, crop_box(cluster_pts, v))
        shown = v.name
    box3d = corners(cluster_pts.min(0), cluster_pts.max(0))
    ov = best_view(cluster_pts, other_views, need_visible=False)
    if ov:
        v, _ = ov
        out["other"] = crop_image(v, crop_box(box3d, v))
    return out, shown


# ---------------------------------------------------------------- CLIP

class Clip:
    def __init__(self, model="openai/clip-vit-base-patch16", vocab=VOCAB):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model).eval()
        self.proc = CLIPProcessor.from_pretrained(model)
        self.vocab = list(vocab)
        with torch.no_grad():
            t = self.proc(text=[f"a photo of a {w}" for w in self.vocab], return_tensors="pt", padding=True)
            e = self.model.get_text_features(**t)
            e = e if isinstance(e, torch.Tensor) else e.pooler_output
            self.text = torch.nn.functional.normalize(e, dim=-1)

    @torch.no_grad()
    def embed(self, img):
        e = self.model.get_image_features(**self.proc(images=img, return_tensors="pt"))
        e = e if isinstance(e, torch.Tensor) else e.pooler_output
        return torch.nn.functional.normalize(e, dim=-1)[0]

    def classify(self, emb, k=3):
        p = (100 * emb @ self.text.T).softmax(-1)
        top = p.topk(k)
        return [(self.vocab[i], float(s)) for s, i in zip(top.values, top.indices)]


# ---------------------------------------------------------------- Claude (optional)

def claude_available():
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def claude_describe(changes, model="claude-opus-5-5", log=print):
    """Ask Claude to name each change from its before/after crops and write a short room summary."""
    import anthropic
    client = anthropic.Anthropic()
    content = [{"type": "text", "text": (
        "You are helping a robot understand what changed in a room between two phone recordings. "
        "A 3D change detector found the regions below. For each one you get a crop from the session where the "
        "object IS present and a crop of the same 3D region from the other session (where it is absent, or for "
        "'moved' the old and new positions). Geometry facts are from the 3D reconstruction and are reliable; "
        "the provisional CLIP label may be wrong.")}]
    for i, c in enumerate(changes):
        content.append({"type": "text", "text": f"\nChange {i}: type={c.type}, provisional label={c.label!r}, "
                                                f"size≈{np.round(c.extent, 2).tolist()} m, facts: {c.description}"})
        for k, b in c.crops.items():
            content.append({"type": "text", "text": f"{k} crop:"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                        "data": base64.standard_b64encode(b).decode()}})
    content.append({"type": "text", "text": (
        "\nReturn only JSON: {\"changes\": [{\"index\": int, \"label\": short noun phrase incl. colour if obvious, "
        "\"description\": one sentence for a person, keeping the geometric facts}], \"summary\": 2-3 sentence "
        "overview of what changed in the room}. If a region doesn't look like a real object change, say so in its "
        "description and prefix the label with 'uncertain:'.")})
    resp = client.beta.messages.create(
        model=model, max_tokens=16000, output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": content}])
    if resp.stop_reason == "refusal":
        raise RuntimeError("Claude declined the request")
    text = "".join(b.text for b in resp.content if b.type == "text")
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    for item in data.get("changes", []):
        c = changes[item["index"]]
        c.label, c.description = item["label"], item["description"]
    return data.get("summary")


# ---------------------------------------------------------------- matching & descriptions

def size_similarity(a, b):
    a, b = np.sort(a), np.sort(b)
    return float(np.mean(np.minimum(a, b) / np.maximum(a, b)))


def refine_motion(old, new):
    """ICP from old to new position, seeded with the pure centroid shift -> translation + yaw.

    We only see each object from a few sides, so rotation is ambiguous for symmetric things (boxes,
    bins); it is reported only when ICP aligns the two partial scans confidently.
    """
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(old))
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(new))
    T0 = np.eye(4)
    T0[:3, 3] = new.mean(0) - old.mean(0)
    r = o3d.pipelines.registration.registration_icp(
        src, dst, 0.05, T0, o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
    ok = r.fitness > 0.8
    T = r.transformation if ok else T0
    yaw = float(np.degrees(np.arctan2(T[1, 0], T[0, 0]))) if ok else 0.0
    moved = (old @ T[:3, :3].T + T[:3, 3]).mean(0) - old.mean(0)
    return moved, yaw, float(r.fitness)


def direction_words(d):
    """World frame: +y = the way the first 'before' frame looks, +x = its right, +z = up."""
    parts = []
    for val, pos, neg in ((d[1], "further away", "closer"), (d[0], "to the right", "to the left")):
        if abs(val) > 0.25 * np.linalg.norm(d[:2]) and abs(val) > 0.1:
            parts.append(pos if val > 0 else neg)
    return " and ".join(parts) if parts else "a short distance"


def placement(pts):
    zmin = np.percentile(pts[:, 2], 2)  # bottom few cm are trimmed by the floor filter / depth smoothing
    return "on the floor" if zmin < 0.25 else f"on a surface about {zmin:.1f} m above the floor"


def describe(c):
    size = "×".join(f"{x:.2f}" for x in c.extent)
    if c.type == "moved":
        e = c.extra
        rot = f", turned about {abs(e['rotation_deg']):.0f}°" if abs(e.get("rotation_deg", 0)) > 20 else ""
        return (f"The {c.label} moved {e['distance_m']:.2f} m {direction_words(np.array(e['displacement']))}{rot}; "
                f"it is now {placement(c.points)}.")
    if c.type == "added":
        return f"A {c.label} (≈{size} m) appeared {placement(c.points)}."
    return f"The {c.label} (≈{size} m) that stood {placement(c.points)} is gone."


def interpret(clusters, views, use_claude=None, log=print):
    vb = [v for v in views if v.session == "before"]
    va = [v for v in views if v.session == "after"]
    log("[semantics] loading CLIP")
    clip = Clip()
    items = []
    for c in clusters:
        own, other = (vb, va) if c.kind == "removed" else (va, vb)
        crops, shown = make_crops(c.points, own, other)
        emb = clip.embed(crops["own"]) if "own" in crops else None
        top = clip.classify(emb) if emb is not None else [("object", 0.0)]
        items.append(dict(cluster=c, crops=crops, emb=emb, top=top, view=shown))
        log(f"[semantics] {c.kind:7s} {len(c.points):5d} pts -> {top[0][0]} ({top[0][1]:.2f}); alt {top[1:]}")

    # pair removed with added objects that look alike and are about the same size -> "moved"
    rem = [i for i in items if i["cluster"].kind == "removed" and i["emb"] is not None]
    add = [i for i in items if i["cluster"].kind == "added" and i["emb"] is not None]
    pairs = []
    if rem and add:
        S = np.zeros((len(rem), len(add)))
        for a, r in enumerate(rem):
            for b, d in enumerate(add):
                app = float(r["emb"] @ d["emb"])
                col = 1 - min(1.0, np.linalg.norm(r["cluster"].colors.mean(0) - d["cluster"].colors.mean(0)) / 0.5)
                same_label = float(r["top"][0][0] == d["top"][0][0])
                S[a, b] = 0.45 * app + 0.25 * size_similarity(r["cluster"].extent, d["cluster"].extent) \
                    + 0.2 * col + 0.1 * same_label
        ra, ca = linear_sum_assignment(-S)
        for a, b in zip(ra, ca):
            log(f"[semantics] match? {rem[a]['top'][0][0]} -> {add[b]['top'][0][0]}: score {S[a, b]:.2f}")
            if S[a, b] >= 0.75:
                pairs.append((rem[a], add[b], S[a, b]))

    changes, used = [], set()
    for r, d, score in pairs:
        used |= {id(r), id(d)}
        old, new = r["cluster"].points, d["cluster"].points
        disp, yaw, fit = refine_motion(old, new)
        label = d["top"][0][0] if d["top"][0][1] >= r["top"][0][1] else r["top"][0][0]
        c = Change("moved", label, "", new.mean(0).tolist(), d["cluster"].extent.tolist(),
                   min(1.0, score), points=new, old_points=old,
                   crops={"before": jpeg(r["crops"]["own"]), "after": jpeg(d["crops"]["own"])},
                   extra={"from": old.mean(0).round(3).tolist(), "to": (old.mean(0) + disp).round(3).tolist(),
                          "displacement": disp.round(3).tolist(), "distance_m": float(np.linalg.norm(disp)),
                          "rotation_deg": round(yaw, 1), "icp_fitness": round(fit, 2),
                          "clip_top": [d["top"], r["top"]]})
        changes.append(c)
    for i in items:
        if id(i) in used:
            continue
        cl = i["cluster"]
        crops = {}
        if "own" in i["crops"]:
            crops["before" if cl.kind == "removed" else "after"] = jpeg(i["crops"]["own"])
        if "other" in i["crops"]:
            crops["after" if cl.kind == "removed" else "before"] = jpeg(i["crops"]["other"])
        crops = dict(sorted(crops.items(), key=lambda kv: kv[0] != "before"))
        conf = float(np.clip(0.5 * cl.score + 0.5 * min(1.0, cl.color_diff / 0.25), 0, 1))
        changes.append(Change(cl.kind, i["top"][0][0], "", cl.centroid.tolist(), cl.extent.tolist(), conf,
                              points=cl.points, crops=crops,
                              extra={"clip_top": i["top"], "free_space_ratio": round(cl.score, 2),
                                     "colour_change": round(cl.color_diff, 2)}))
    for c in changes:
        c.description = describe(c)

    summary = None
    if changes and (use_claude or (use_claude is None and claude_available())):
        try:
            log("[semantics] asking Claude to name and describe the changes")
            summary = claude_describe(changes, log=log)
        except Exception as e:  # never let the optional VLM break the pipeline
            log(f"[semantics] Claude unavailable ({type(e).__name__}: {e}); keeping CLIP labels")
    if summary is None:
        summary = summarise(changes)
    return changes, summary


def summarise(changes):
    if not changes:
        return "No meaningful changes were found in the parts of the room both recordings observed."
    parts = []
    for kind, verb in (("moved", "moved"), ("added", "appeared"), ("removed", "disappeared")):
        names = [c.label for c in changes if c.type == kind]
        if names:
            parts.append(f"{len(names)} object{'s' * (len(names) > 1)} {verb} ({', '.join(names)})")
    return "; ".join(parts).capitalize() + "."
