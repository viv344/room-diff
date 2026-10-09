"""Joint structure-from-motion over both recordings.

Both videos go into ONE COLMAP reconstruction, so the static parts of the room (walls, floor,
furniture that didn't move) tie the two sessions into a single shared coordinate frame. Changed
objects are a small fraction of the matches and get rejected by RANSAC as outliers.
"""
import os
from dataclasses import dataclass, field

import numpy as np
import pycolmap

from . import SESSIONS


@dataclass
class View:
    name: str  # e.g. "before/0003.jpg"
    session: str
    K: np.ndarray  # 3x3 pinhole intrinsics (undistorted)
    T_cw: np.ndarray  # 4x4 world -> camera
    image: np.ndarray = None  # HxWx3 RGB uint8
    depth: np.ndarray = None  # HxW metric depth, 0 = invalid
    sparse_uv: np.ndarray = field(default=None, repr=False)  # Nx2 pixels of triangulated points
    sparse_xyz: np.ndarray = field(default=None, repr=False)  # Nx3 world points

    @property
    def center(self):
        R, t = self.T_cw[:3, :3], self.T_cw[:3, 3]
        return -R.T @ t


def _pose(img):
    cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
    T = np.eye(4)
    T[:3, :4] = cfw.matrix()
    return T


def run_sfm(image_root, work, log=print):
    """image_root contains before/*.jpg and after/*.jpg. Returns (views, sparse_points_xyz, sparse_rgb)."""
    os.makedirs(work, exist_ok=True)
    db = os.path.join(work, "database.db")
    if os.path.exists(db):
        os.remove(db)
    names = sorted(os.path.join(s, f) for s in SESSIONS for f in os.listdir(os.path.join(image_root, s)))

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = 1024
    ext.sift.max_num_features = 6000
    log(f"[sfm] extracting SIFT features from {len(names)} frames")
    # one camera per folder = one intrinsics per recording (the two phones/zoom levels may differ)
    pycolmap.extract_features(db, image_root, image_names=names, camera_mode=pycolmap.CameraMode.PER_FOLDER,
                              reader_options=reader, extraction_options=ext, device=pycolmap.Device.cpu)
    log("[sfm] exhaustive matching (within and across recordings)")
    pycolmap.match_exhaustive(db, device=pycolmap.Device.cpu)

    log("[sfm] incremental mapping")
    opts = pycolmap.IncrementalPipelineOptions()
    opts.multiple_models = True
    recs = pycolmap.incremental_mapping(db, image_root, os.path.join(work, "sparse"), options=opts)
    if not recs:
        raise RuntimeError("SfM failed: no reconstruction. Record slower, with more texture and overlap.")
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    per = {s: sum(1 for im in rec.images.values() if im.name.startswith(s + "/")) for s in SESSIONS}
    log(f"[sfm] registered {per} of {len(names)} frames, {rec.num_points3D()} points")
    for s in SESSIONS:
        if per[s] < 5:
            raise RuntimeError(f"only {per[s]} '{s}' frames registered into the shared model; the two recordings "
                               "don't overlap enough (or the lighting changed a lot). Re-record with more overlap.")

    # undistort so that later stages can use a plain pinhole model
    und = os.path.join(work, "undistorted")
    rec_dir = os.path.join(work, "sparse_best")
    os.makedirs(rec_dir, exist_ok=True)
    rec.write(rec_dir)
    pycolmap.undistort_images(und, rec_dir, image_root)
    rec = pycolmap.Reconstruction(os.path.join(und, "sparse"))

    xyz_all = {pid: p.xyz for pid, p in rec.points3D.items()}
    views = []
    for im in sorted(rec.images.values(), key=lambda i: i.name):
        cam = rec.cameras[im.camera_id]
        uv, xyz = [], []
        for p2 in im.points2D:
            if p2.has_point3D():
                uv.append(p2.xy)
                xyz.append(xyz_all[p2.point3D_id])
        views.append(View(name=im.name, session=im.name.split("/")[0], K=cam.calibration_matrix(), T_cw=_pose(im),
                          sparse_uv=np.array(uv, float).reshape(-1, 2), sparse_xyz=np.array(xyz, float).reshape(-1, 3)))
    pts = np.array([p.xyz for p in rec.points3D.values()])
    rgb = np.array([p.color for p in rec.points3D.values()])
    # a feature track seen in BOTH recordings proves that surface is static
    sess_of = {iid: im.name.split("/")[0] for iid, im in rec.images.items()}
    both = np.array([len({sess_of[e.image_id] for e in p.track.elements}) == 2 for p in rec.points3D.values()])
    log(f"[sfm] {both.sum()} points tracked across both recordings (static anchors)")
    return views, pts, rgb, both, os.path.join(und, "images")
