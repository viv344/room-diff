"""Visibility-aware 3D change detection.

A point from session A is only called "gone" if session B *looked through* the place where it
should be: B's depth at that pixel is clearly farther than the point (free-space violation). If
B's depth is nearer, the point was merely occluded; if B never looked there, it is unobserved.
Neither of those count as change - which is exactly how we separate real changes from
viewpoint differences, occlusion and incomplete coverage.
"""
from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree


def project(view, X):
    Xc = X @ view.T_cw[:3, :3].T + view.T_cw[:3, 3]
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = Xc[:, :2] / z[:, None] @ view.K[:2, :2].T + view.K[:2, 2]
    return uv, z


def tolerance(z):
    """Depth agreement tolerance; monocular depth error grows roughly linearly with distance."""
    return 0.06 + 0.06 * z


def visibility_votes(points, views, border=0.0, extra_tol=None, return_range=False):
    """For each point, count views that see it (consistent), see through it (free) or see in front of it.

    extra_tol: optional per-point slack added to the depth tolerance (uncertainty of the point itself).
    return_range: also return the mean camera distance over the views that see the point consistently.
    """
    n = len(points)
    zsum = np.zeros(n)
    same = np.zeros(n, np.int32)
    free = np.zeros(n, np.int32)
    occl = np.zeros(n, np.int32)
    for v in views:
        h, w = v.depth.shape
        # conservative "see-through": the whole 7x7 neighbourhood must be farther. Zeros (invalid) -> never free
        dmin = minimum_filter(np.where(v.depth > 0, v.depth, 0), size=7)
        uv, z = project(v, points)
        mx, my = border * w, border * h  # monocular depth is least reliable at the image borders
        ok = (z > 0.1) & (uv[:, 0] >= mx) & (uv[:, 0] < w - 1 - mx) & (uv[:, 1] >= my) & (uv[:, 1] < h - 1 - my)
        idx = np.where(ok)[0]
        u = np.round(uv[idx, 0]).astype(int)
        r = np.round(uv[idx, 1]).astype(int)
        d = v.depth[r, u]
        valid = d > 0
        tau = tolerance(z[idx]) + (0 if extra_tol is None else extra_tol[idx])
        s_ = valid & (np.abs(d - z[idx]) <= tau)
        same[idx[s_]] += 1
        zsum[idx[s_]] += z[idx[s_]]
        free[idx[valid & (dmin[r, u] > z[idx] + tau)]] += 1
        occl[idx[valid & (d < z[idx] - tau)]] += 1
    if return_range:
        return same, free, occl, np.where(same > 0, zsum / np.maximum(same, 1), np.inf)
    return same, free, occl


@dataclass
class Cluster:
    kind: str  # "removed" (present before, gone after) | "added"
    points: np.ndarray
    colors: np.ndarray
    score: float  # mean free-space ratio
    anchors: int = 0  # cross-session SfM points on the cluster (should be ~0 for a real change)
    color_diff: float = 0.0  # how different the other session looks where the cluster is

    @property
    def centroid(self):
        return self.points.mean(0)

    @property
    def bbox(self):
        return self.points.min(0), self.points.max(0)

    @property
    def extent(self):
        lo, hi = self.bbox
        return hi - lo


def photometric_difference(points, colors, dst_views, min_px=20):
    """How different does the other session's image look where the cluster should be?

    For a real removal the other session sees whatever was *behind* the object (floor, wall), which
    looks different. If it sees the same colour, the "change" is just the same surface reconstructed
    at a slightly different depth (typical monocular error on far, uniform walls). Compared on
    blurred images, so a few cm of misregistration doesn't matter.
    """
    diffs = []
    for v in dst_views:
        h, w = v.depth.shape
        uv, z = project(v, points)
        ok = (z > 0.1) & (uv[:, 0] > 0.05 * w) & (uv[:, 0] < 0.95 * w) & (uv[:, 1] > 0.05 * h) & (uv[:, 1] < 0.95 * h)
        if ok.sum() < min_px:
            continue
        u, r = uv[ok, 0].astype(int), uv[ok, 1].astype(int)
        free = v.depth[r, u] > z[ok] + tolerance(z[ok])
        if free.sum() < min_px:
            continue
        blur = cv2.GaussianBlur(v.image, (0, 0), 6).astype(np.float32) / 255
        diffs.append(np.linalg.norm(blur[r[free], u[free]].mean(0) - colors.mean(0)))
    return float(np.median(diffs)) if diffs else 1.0


def changed_points(src, dst, src_views, dst_views, min_free=3, ratio=0.7, dist=0.06, range_k=0.05):
    """Points of `src` cloud that the `dst` session observed to be empty space.

    A point that `src` only ever saw from far away is itself uncertain (monocular depth error grows
    with range), so its tolerance is widened by range_k * (distance it was observed from).
    Returns (mask_changed, mask_unobserved, free_ratio, displacement) over src points.
    """
    P = np.asarray(src.points)
    *_, src_range = visibility_votes(P, src_views, return_range=True)
    sigma = range_k * np.minimum(src_range, 6.0)
    same, free, _ = visibility_votes(P, dst_views, border=0.05, extra_tol=sigma)
    seen = same + free
    r = free / np.maximum(seen, 1)
    nn, _ = cKDTree(np.asarray(dst.points)).query(P, distance_upper_bound=1.0)
    nn = np.minimum(nn, 1.0)
    changed = (free >= min_free) & (r >= ratio) & (nn > np.maximum(dist, sigma))
    unobserved = seen == 0
    return changed, unobserved, r, nn


def cluster_changes(pcd, mask, ratio, nn, kind, anchors, dst_views, eps=0.05, min_points=100, min_extent=0.08,
                    floor_z=0.06, min_displacement=0.10, max_anchor_frac=0.01, min_color_diff=0.08):
    P = np.asarray(pcd.points)[mask]
    C = np.asarray(pcd.colors)[mask]
    R, D = ratio[mask], nn[mask]
    keep = P[:, 2] > floor_z  # floor-level differences (rug shifted, depth noise on the floor) are ignored on purpose
    P, C, R, D = P[keep], C[keep], R[keep], D[keep]
    if len(P) < min_points:
        return []
    sub = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    labels = np.asarray(sub.cluster_dbscan(eps=eps, min_points=8))
    out = []
    for l in range(labels.max() + 1):
        m = labels == l
        if m.sum() < min_points:
            continue
        c = Cluster(kind, P[m], C[m], float(R[m].mean()))
        if np.sort(c.extent)[-2] < min_extent:  # thin slivers along depth edges
            continue
        # surfaces that just shifted a few cm (depth error) rather than appearing/vanishing
        if np.median(D[m]) < min_displacement:
            continue
        # veto: SfM features matched in both videos sit on this "change" -> it's a depth error, not a change
        n_anchor = int((cKDTree(c.points).query(anchors, distance_upper_bound=0.05)[0] < 0.05).sum()) if len(anchors) else 0
        c.anchors = n_anchor
        if n_anchor > max(2, max_anchor_frac * len(c.points)):
            continue
        c.color_diff = photometric_difference(c.points, c.colors, dst_views)
        if c.color_diff < min_color_diff:
            continue
        out.append(c)
    return sorted(out, key=lambda c: -len(c.points))


def detect(before_pcd, after_pcd, views, anchors=np.zeros((0, 3)), log=print):
    vb = [v for v in views if v.session == "before"]
    va = [v for v in views if v.session == "after"]
    rem, unobs_b, rb, nb = changed_points(before_pcd, after_pcd, vb, va)
    add, unobs_a, ra, na = changed_points(after_pcd, before_pcd, va, vb)
    log(f"[change] {rem.sum()} 'gone' points, {add.sum()} 'new' points; "
        f"{unobs_b.mean():.0%} of before-cloud unobserved after, {unobs_a.mean():.0%} of after-cloud unobserved before")
    clusters = (cluster_changes(before_pcd, rem, rb, nb, "removed", anchors, va)
                + cluster_changes(after_pcd, add, ra, na, "added", anchors, vb))
    log(f"[change] {len(clusters)} change clusters")
    masks = {"removed": rem, "added": add, "unobserved_before": unobs_b, "unobserved_after": unobs_a}
    return clusters, masks
