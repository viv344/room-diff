"""Fuse one session's aligned depth maps into a clean, multi-view-consistent point cloud.

Back-project every depth map, voxelise, then keep only voxels that (a) at least `min_views`
frames agree on and (b) the session's own frames don't see straight through. That's the
free-space carving a TSDF gives you, written out explicitly so the same visibility test is
reused for change detection.
"""
import numpy as np
import open3d as o3d

from .changes import visibility_votes


def backproject(view, stride=2, max_depth=5.0):
    d = view.depth[::stride, ::stride]
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    u, v = u * stride, v * stride
    m = (d > 0) & (d < max_depth)
    z = d[m]
    K = view.K
    Xc = np.stack([(u[m] - K[0, 2]) / K[0, 0] * z, (v[m] - K[1, 2]) / K[1, 1] * z, z], 1)
    R, t = view.T_cw[:3, :3], view.T_cw[:3, 3]
    X = (Xc - t) @ R  # R^T (Xc - t)
    return X, view.image[::stride, ::stride][m] / 255.0


def fuse(views, voxel=0.02, min_views=2):
    pts, cols, ids = [], [], []
    for i, v in enumerate(views):
        X, C = backproject(v)
        pts.append(X)
        cols.append(C)
        ids.append(np.full(len(X), i, np.int32))
    P, C, I = np.concatenate(pts), np.concatenate(cols), np.concatenate(ids)
    key = np.floor(P / voxel).astype(np.int64)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    n = len(counts)
    # mean position/colour per voxel, and number of distinct frames that hit it
    S = np.zeros((n, 3)); np.add.at(S, inv, P)
    SC = np.zeros((n, 3)); np.add.at(SC, inv, C)
    P, C = S / counts[:, None], SC / counts[:, None]
    nviews = np.unique(np.stack([inv, I], 1), axis=0)[:, 0]
    nviews = np.bincount(nviews, minlength=n)
    keep = nviews >= min_views
    P, C = P[keep], C[keep]
    same, free, _ = visibility_votes(P, views)
    keep = (same >= min_views) & (free <= 0.5 * same)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[keep]))
    pcd.colors = o3d.utility.Vector3dVector(C[keep])
    pcd, _ = pcd.remove_statistical_outlier(20, 2.0)
    return pcd
