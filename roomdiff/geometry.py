"""World-frame conventions: metric units, z up, floor at z = 0."""
import numpy as np
import open3d as o3d


def transform_world(views, pts, M, scale=1.0):
    """Apply x' = scale * R x + t (M = [R|t], 4x4) to every camera and point."""
    R, t = M[:3, :3], M[:3, 3]
    for v in views:
        Rc, tc = v.T_cw[:3, :3], v.T_cw[:3, 3]
        # x_c = Rc x + tc, x = R^T (x' - t) / scale  ->  x_c = Rc R^T x'/scale - Rc R^T t/scale + tc
        Rn = Rc @ R.T
        tn = scale * tc - Rn @ t
        v.T_cw = np.eye(4)
        v.T_cw[:3, :3], v.T_cw[:3, 3] = Rn, tn  # depth also scales by `scale`
        if v.sparse_xyz is not None and len(v.sparse_xyz):
            v.sparse_xyz = scale * v.sparse_xyz @ R.T + t
    return scale * pts @ R.T + t


def rotation_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), np.dot(a, b)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def gravity_align(views, pts, log=print):
    """Rotate so +z is up and the floor is z=0.

    Initial up = average camera 'up' (people hold phones roughly upright), refined with RANSAC on the
    biggest roughly-horizontal plane below the cameras (the floor).
    """
    up = np.mean([-v.T_cw[:3, :3].T @ np.array([0, 1, 0]) for v in views], axis=0)
    up /= np.linalg.norm(up)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    cam_h = np.median([v.center @ up for v in views])
    rest = pcd
    cands = []  # horizontal planes below the cameras: (height, n_inliers, normal, points)
    for _ in range(6):
        if len(rest.points) < 50:
            break
        extent = np.linalg.norm(np.asarray(rest.points).std(0))
        (a, b, c, d), inl = rest.segment_plane(0.01 * extent + 1e-6, 3, 2000)
        n = np.array([a, b, c])
        P = np.asarray(rest.points)[inl]
        if abs(n @ up) > np.cos(np.deg2rad(25)) and np.median(P @ up) < cam_h:
            cands.append((np.median(P @ up), len(inl), n if n @ up > 0 else -n, P))
        rest = rest.select_by_index(inl, invert=True)
    floor = None
    if cands:  # lowest plane with decent support (table tops are higher, stray slivers are small)
        big = max(c[1] for c in cands)
        floor = min((c for c in cands if c[1] >= 0.25 * big), key=lambda c: c[0])[2:]
    if floor is not None:
        up = floor[0]
        log("[geom] floor plane found; using its normal as gravity")
    else:
        log("[geom] no floor plane found; using mean camera orientation as gravity")
    R = rotation_between(up, np.array([0, 0, 1.0]))
    # yaw: make the first 'before' camera look along +y so "left/right/forward" read naturally
    fwd = R @ (views[0].T_cw[:3, :3].T @ np.array([0, 0, 1.0]))
    yaw = np.arctan2(fwd[0], fwd[1])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R
    z0 = np.percentile((floor[1] if floor is not None else pts) @ R.T[:, 2], 50 if floor is not None else 2)
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, [0, 0, -z0]
    return M
