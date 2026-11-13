"""White matter lesions: segment on the T1 (wmh_seg) and transfer onto the slab photos.

Every photo has a known plane in the MRI and a known in-plane transform, so each photo
pixel has a 3D position in the T1. The lesion mask is sampled there, giving a mask in the
photo's own pixel grid:
    on the face  - lesion on the photographed (anterior) cut surface
    in the slab  - lesion anywhere through the slab thickness behind that surface
"""
import csv
import importlib.util
import os
import sys
import types

import cv2
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from . import progress as PG
from scipy import ndimage as ndi

WORK_MAX = 1600        # photos are processed at this size (longest side), masks are saved at full size


def segment_wmh(t1_path, out_path, model="exvivo", log=print):
    """Run wmh_seg's nnU-Net model on the T1 and write a 0/1 mask on the T1 grid."""
    try:
        import pkg_resources  # noqa: F401  (wmh_seg needs it; removed from setuptools >= 81)
    except ModuleNotFoundError:
        shim = types.ModuleType("pkg_resources")
        shim.resource_filename = lambda pkg, path: os.path.join(
            os.path.dirname(importlib.util.find_spec(pkg).origin), path)
        sys.modules["pkg_resources"] = shim
    try:
        from wmh_seg import nnunet_segment
    except ImportError as e:
        raise ValueError("--wmh needs wmh_seg 1.6 or newer (pip install -U wmh_seg); "
                         f"an older version has no `nnunet_segment` with model selection: {e}")
    PG.start(1, f"Lesion segmentation ({model})")
    with torch.enable_grad():                      # leave wmh_seg's own grad handling untouched
        nnunet_segment(t1_path, out_path, model=model)
    PG.tick()
    PG.done()
    return out_path


def load_mask(path, mri):
    img = nib.load(path)
    if tuple(img.shape[:3]) != tuple(mri.vol.shape):
        raise ValueError(f"lesion mask {path} has shape {img.shape}, the T1 has {mri.vol.shape}; "
                         "the mask must be on the T1 grid")
    return (np.asarray(img.dataobj) > 0)


def transfer(st, mri, photos, scale_f, lesion, out_dir, thickness, dev, log=print):
    """Write lesion masks / overlays in photo space and a per-photo summary table."""
    for d in ("lesion_mask", "lesion_overlay", "lesion_slice"):
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)
    vol = torch.from_numpy(lesion.astype(np.float32)).to(dev)[None, None]
    shp = torch.tensor(lesion.shape, dtype=torch.float32, device=dev)
    c = (shp - 1) / 2
    Rg = torch.tensor(st["Rg"], dtype=torch.float32, device=dev)
    idx = np.rint(mri.c[1] + st["Y"]).astype(int)
    depths = torch.arange(0, max(float(thickness), 1.0), 2.0, device=dev)
    hi = np.percentile(mri.vol[::4, ::4, ::4], 99.5)
    rows = []
    PG.start(len(photos), "Lesions onto the photos")
    for i, p in enumerate(photos):
        PG.tick()
        full = cv2.imread(p["path"], cv2.IMREAD_COLOR)
        H, W = full.shape[:2]
        f = min(1.0, WORK_MAX / max(H, W))
        h, w = max(1, round(H * f)), max(1, round(W * f))
        # work pixel -> full pixel -> classical-stage pixel -> in-plane o coords
        vv, uu = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                                torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
        us = ((uu + 0.5) / f) * scale_f - 0.5
        vs = ((vv + 0.5) / f) * scale_f - 0.5
        Pi = torch.tensor(np.linalg.inv(st["P"][i]), dtype=torch.float32, device=dev)
        X = Pi[0, 0] * us + Pi[0, 1] * vs + Pi[0, 2]
        Z = Pi[1, 0] * us + Pi[1, 1] * vs + Pi[1, 2]
        Y = float(idx[i] - mri.c[1]) - depths                       # slab runs posteriorly from the face
        o = torch.stack([X[None].expand(len(Y), -1, -1), Y[:, None, None].expand(-1, h, w),
                         Z[None].expand(len(Y), -1, -1)], -1)
        g = ((c + o @ Rg.T + 0.5) / shp * 2 - 1).flip(-1)
        s = F.grid_sample(vol, g[None], mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0]
        tissue = cv2.resize(p["mask"].astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR) > 0
        face = (s[0] > 0.5).cpu().numpy() & tissue
        slab = (s.max(0).values > 0.5).cpu().numpy() & tissue

        mask = np.zeros((h, w), np.uint8)
        mask[slab] = 127
        mask[face] = 255
        cv2.imwrite(os.path.join(out_dir, "lesion_mask", f"{i}.png"),
                    cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST))

        # overlay: photo with lesions | rotated MRI slice with lesions
        ph = cv2.resize(full, (w, h), interpolation=cv2.INTER_AREA)
        ov = ph.copy()
        ov[face] = (0.55 * ov[face] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
        for m, col in ((slab, (0, 255, 255)), (face, (0, 0, 255))):
            cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, cnt, -1, col, max(1, round(max(h, w) / 600)))
        nx, _, nz = mri.vol.shape
        Xg, Zg = np.meshgrid(np.arange(nx) - mri.c[0], np.arange(nz) - mri.c[2], indexing="ij")
        pts = (mri.c + np.stack([Xg, np.full_like(Xg, idx[i] - mri.c[1]), Zg], -1) @ st["Rg"].T).reshape(-1, 3).T
        sl = ndi.map_coordinates(mri.vol, pts, order=1, cval=0).reshape(nx, nz)
        ls = ndi.map_coordinates(lesion.astype(np.float32), pts, order=1, cval=0).reshape(nx, nz) > 0.5
        cv2.imwrite(os.path.join(out_dir, "lesion_slice", f"{i}.png"),
                    np.ascontiguousarray(np.flipud(ls.T)).astype(np.uint8) * 255)
        mr = cv2.cvtColor(np.flipud(np.clip(sl / hi * 255, 0, 255).astype(np.uint8).T), cv2.COLOR_GRAY2BGR)
        mr = cv2.resize(mr, (round(nx * h / nz), h), interpolation=cv2.INTER_LINEAR)
        lm = cv2.resize(np.ascontiguousarray(np.flipud(ls.T)).astype(np.uint8), (mr.shape[1], h),
                        interpolation=cv2.INTER_NEAREST)
        cnt, _ = cv2.findContours(lm, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(mr, cnt, -1, (0, 0, 255), max(1, round(h / 600)))
        cv2.imwrite(os.path.join(out_dir, "lesion_overlay", f"{i}.jpg"), np.hstack([ov, mr]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

        # summary for targeted sampling
        mm_per_work_px = mri.vox * scale_f / (f * np.sqrt(abs(np.linalg.det(st["P"][i][:2, :2]))))
        lab, n = ndi.label(face)
        cx = cy = ""
        if n:
            sz = ndi.sum(face, lab, range(1, n + 1))
            cyw, cxw = ndi.center_of_mass(face, lab, int(np.argmax(sz)) + 1)
            cx, cy = round((cxw + 0.5) / f - 0.5), round((cyw + 0.5) / f - 0.5)
        rows.append(dict(n=i, photo=os.path.basename(p["path"]), slice_index=int(idx[i]),
                         face_lesion_mm2=round(float(face.sum()) * mm_per_work_px ** 2, 1),
                         slab_lesion_mm2=round(float(slab.sum()) * mm_per_work_px ** 2, 1),
                         face_lesion_count=int(n), largest_face_lesion_x_px=cx, largest_face_lesion_y_px=cy))
    PG.done()
    with open(os.path.join(out_dir, "lesions.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    log("")
    log(f"{'photo':<28}{'on face mm2':>12}{'in slab mm2':>12}")
    for r in rows:
        log(f"{r['photo']:<28}{r['face_lesion_mm2']:>12}{r['slab_lesion_mm2']:>12}")
    return rows
