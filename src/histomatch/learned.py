"""`--method learned`: place every photo with the trained slice2vol model instead of the search.

slice2vol places each photo on its own (no cutting-guide assumption, no stack prior) and returns, per
photo, a matrix from full-resolution photo pixel to scanner mm. histomatch's outputs, lesion transfer and
report all read ONE cutting orientation for the brain plus a slice position and an in-plane transform
per photo, so the independent placements are converted to that form here:

  * the cutting normal is the mean of the per-photo plane normals (outliers beyond 15 degrees dropped);
  * each photo's slice is where its tissue centroid falls along that normal;
  * each photo's in-plane transform is its own placement projected onto the common plane.

What is lost is each slab's own tilt relative to the common normal (a few degrees at most on a
consistently cut brain). A photo the model could not place inherits its neighbours' transform and an
interpolated slice, and is reported.
"""
import os

import numpy as np

ENV = "HISTOMATCH_WEIGHTS"
DEFAULT_WEIGHTS = os.path.join(os.path.expanduser("~"), ".cache", "histomatch", "slice2vol_v1", "best.pt")
MAX_NORMAL_DEG = 15.0


def find_weights(weights=None):
    """The checkpoint to use: the argument, else $HISTOMATCH_WEIGHTS, else the default location."""
    path = weights or os.environ.get(ENV) or DEFAULT_WEIGHTS
    if not os.path.isfile(path):
        raise ValueError(f"slice2vol weights not found at {path!r}: pass --weights PATH or set {ENV} "
                         "(the checkpoint's config.yaml must sit next to it)")
    return path


def _common_normal(normals):
    """Mean direction of unit normals (sign-aligned), dropping those far from the first estimate."""
    ref = normals[len(normals) // 2]
    n = np.array([m if m @ ref > 0 else -m for m in normals])
    mean = n.mean(0) / np.linalg.norm(n.mean(0))
    ok = np.degrees(np.arccos(np.clip(n @ mean, -1, 1))) <= MAX_NORMAL_DEG
    if ok.sum() >= 3:
        mean = n[ok].mean(0) / np.linalg.norm(n[ok].mean(0))
    return mean, np.degrees(np.arccos(np.clip(n @ mean, -1, 1)))


def learned_state(paths, photos, scale_f, mri, weights=None, log=print, tick=lambda: None):
    """Place `paths` in `mri` with slice2vol and return a histomatch state
    (Rg, Y, P, spacing, mirror; plus `placed`, a bool per photo)."""
    try:
        import torch
        from slice2vol.predict import load_model, predict_images, prepare_volume
    except ImportError as e:
        raise ValueError("--method learned needs the slice2vol package (pip install -e path/to/slice2vol): "
                         f"{e}") from e
    wpath = find_weights(weights)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(wpath, dev)
    vol = prepare_volume(mri.path, dev)
    res = []
    for p in paths:                                     # one at a time: bounded memory, one tick per photo
        res.extend(predict_images([p], vol, model, dev, mask=True, weights_name=os.path.basename(wpath)))
        tick()
    del model, vol
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    inv = np.linalg.inv(np.asarray(mri.affine, float))
    N, c = len(paths), np.asarray(mri.c, float)
    A = [None] * N                                      # per photo: (a, b, t) with voxel = a col + b row + t
    for i, r in enumerate(res):
        if r.get("ok") and r.get("matrix") is not None:
            M = inv @ np.asarray(r["matrix"], float)
            A[i] = (M[:3, 0], M[:3, 1], M[:3, 3])
    placed = np.array([a is not None for a in A])
    if placed.sum() < 3:
        raise ValueError(f"the learned model placed only {int(placed.sum())} of {N} photos; at least 3 are needed")

    normals = []
    for a in A:
        if a is not None:
            n = np.cross(a[0], a[1])
            normals.append(n / np.linalg.norm(n))
    ey, dev_deg = _common_normal(normals)
    if ey[1] < 0:                                       # same sense as the search: +y of the rotated frame ~ +array y
        ey = -ey
    ez = np.array([0.0, 0.0, 1.0]) - ey[2] * ey
    ez /= np.linalg.norm(ez)
    ex = np.cross(ey, ez)
    Rg = np.stack([ex, ey, ez], 1)                      # voxel = c + Rg @ o, as loftr_stage.upright leaves it

    Y, P = np.full(N, np.nan), np.zeros((N, 3, 3))
    f = float(scale_f)
    for i, a in enumerate(A):
        if a is None:
            continue
        av, bv, t = a
        cu, cv = [(float(v) + 0.5) / f - 0.5 for v in photos[i]["cen"]]        # tissue centroid, full-res pixel
        Y[i] = (av * cu + bv * cv + t - c) @ ey
        J = np.stack([[av @ ex, bv @ ex], [av @ ez, bv @ ez]])                # (col, row) -> (X, Z)
        t2 = np.array([(t - c) @ ex, (t - c) @ ez])
        Ji = np.linalg.inv(J)
        P[i, :2, :2] = f * Ji                                                  # (X, Z) -> reduced photo pixel
        P[i, :2, 2] = f * (-Ji @ t2) + 0.5 * f - 0.5
        P[i, 2, 2] = 1.0

    good = np.where(placed)[0]
    for i in np.where(~placed)[0]:                      # unplaced: neighbours' slice spacing and transform
        Y[i] = np.interp(i, good, Y[good]) if good[0] <= i <= good[-1] else \
            Y[good[0 if i < good[0] else -1]] + (i - good[0 if i < good[0] else -1]) * np.median(np.diff(Y[good]) / np.diff(good))
        j = good[np.argmin(np.abs(good - i))]
        P[i] = P[j]
        XZ = np.linalg.solve(P[j][:2, :2], np.array(photos[j]["cen"], float) - P[j][:2, 2])
        P[i][:2, 2] = np.array(photos[i]["cen"], float) - P[j][:2, :2] @ XZ    # same spot of the slice under its centroid
    steps = np.abs(np.diff(Y[good]) / np.diff(good))
    mirror = int(np.sign(-np.median([np.linalg.det(P[i][:2, :2]) for i in good])) or 1)
    log(f"learned placement: {int(placed.sum())} of {N} photos placed, plane normals within "
        f"{float(np.max(dev_deg)):.1f} deg of the common one (median {float(np.median(dev_deg)):.1f}), "
        f"median slab step {float(np.median(steps)):.1f} voxels")
    for i in np.where(~placed)[0]:
        log(f"  {os.path.basename(paths[i])}: not placed by the model, taken from its neighbours -- check qc")
    d = np.diff(Y[good])
    if len(d) and min((d > 0).mean(), (d < 0).mean()) > 0:
        log(f"  warning: {int(min((d > 0).sum(), (d < 0).sum()))} photo(s) out of order along the cutting normal -- check qc")
    return dict(Rg=Rg, Y=Y, P=P, spacing=float(np.median(steps)), mirror=mirror, placed=placed.tolist())


def count_inliers(st, mri, photos, scale_f, dev, tick=lambda: None):
    """LoFTR in-plane inliers of every photo on its slice, as a quality figure. The state is not changed."""
    from . import loftr_stage as LS
    R = LS.Renderer(mri, photos, scale_f, dev)
    out = []
    for i in range(len(photos)):
        pa, pb, _, _ = LS.match_fan(R, st, i, [0.0], conf_thr=0.2)
        out.append(LS.ransac_count(pa.cpu().numpy(), pb.cpu().numpy(), thr=3.0)[0])
        tick()
    return out
