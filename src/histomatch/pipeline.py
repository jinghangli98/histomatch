"""End-to-end matching: classical initialisation, then LoFTR refinement, then export."""
import json
import os
import time

import numpy as np


def run(photo_dir, t1_path, out_dir, use_loftr=True, spacing=24.5, wmh=False, lesion_mask=None,
        wmh_model="exvivo", ppt=False, fast=False, method="search", weights=None, log=print):
    """Match every photo in `photo_dir` (sorted numerically, anterior -> posterior) to `t1_path`.

    Writes angle_pos, match.json, resizedCam/, mri_slices/, qc/ and png.nii.gz to `out_dir`
    and returns the summary dict that is saved as match.json.

    With `wmh=True` the T1 is segmented with wmh_seg (or pass an existing mask on the T1 grid
    as `lesion_mask`) and the lesions are transferred onto every photo.

    `fast=True` takes only the better of the two coarse candidates of each direction/mirror
    class through the full fit and LoFTR (4 instead of 8): quicker, but it can miss the right
    orientation.

    `method="learned"` places every photo with the trained slice2vol model (`weights`: its checkpoint)
    instead of the classical search and LoFTR refinement; see learned.py. `use_loftr` then only decides
    whether LoFTR match counts are computed as a quality figure, and `fast` is ignored.
    """
    from . import classical as C
    from . import loftr_stage as LS
    from . import progress as PG

    if log is print:
        log = PG.write
    if method not in ("search", "learned"):
        raise ValueError(f"unknown method {method!r} (want 'search' or 'learned')")
    extra = (2 if wmh and not lesion_mask else 1 if lesion_mask else 0) + (1 if ppt else 0)
    PG.plan((3 if method == "learned" else 4) + (1 if use_loftr else 0) + extra)

    C.SPACING_VOX = float(spacing)
    t0 = time.time()
    paths = C.list_photos(photo_dir)
    if len(paths) < 3:
        raise ValueError(f"need at least 3 photos directly inside {photo_dir!r}, found {len(paths)}")
    log(f"{len(paths)} photos, anterior -> posterior: {os.path.basename(paths[0])} ... {os.path.basename(paths[-1])}")
    PG.start(len(paths), "Reading photos")
    photos, scale_f = C.load_photos(paths)
    mri = C.MRI(t1_path)
    log(f"MRI {mri.vol.shape} @ {mri.vox:.3f} mm, running on {C.DEV}  ({time.time() - t0:.0f}s)")
    os.makedirs(out_dir, exist_ok=True)
    N, per = len(photos), (len(LS.SCHED) + 1) * len(photos)
    if method == "learned":
        from . import learned as LN
        PG.start(N, "Placing photos (learned)")
        st = LN.learned_state(paths, photos, scale_f, mri, weights=weights, log=log, tick=PG.tick)
        if use_loftr:
            PG.start(N, "LoFTR check")
            st["loftr_inliers"] = LN.count_inliers(st, mri, photos, scale_f, C.DEV, tick=PG.tick)
        PG.start(N, "Writing outputs")
        summary = LS.export_state(st, mri, photos, scale_f, out_dir, thickness=spacing)
        summary["method"] = "learned"
        summary["placed"] = st["placed"]
        with open(os.path.join(out_dir, "match.json"), "w") as fh:
            json.dump(summary, fh, indent=1)
    elif not use_loftr:
        fits = C.refine(mri, photos, C.coarse_search(mri, photos, log), log, fast=fast)
        st = LS.state_from_model(fits[0][1], mri)
        PG.start(N, "Writing outputs")
        summary = LS.export_state(st, mri, photos, scale_f, out_dir, thickness=spacing)
    else:
        fits = C.refine(mri, photos, C.coarse_search(mri, photos, log), log, fast=fast)
        st = LS.state_from_model(fits[0][1], mri)
        PG.start(len(fits) * per, "LoFTR matching")
        classical_st = st
        # run LoFTR from every classical candidate and keep the one LoFTR agrees with most
        R = LS.Renderer(mri, photos, scale_f, C.DEV)
        best = None
        for sc, model in fits:
            cand = LS.loftr_refine(LS.state_from_model(model, mri), mri, photos, scale_f, log, dev=C.DEV, R=R,
                                   rescue_weak=False)
            tot = int(np.sum(cand["loftr_inliers"]))
            log(f"  candidate (classical score {sc:.3f}) -> {tot} LoFTR inliers")
            if best is None or tot > best[0]:
                best = (tot, cand, model)
        st = best[1]
        if min(st["loftr_inliers"]) < LS.WEAK:               # some photos unmatched: zoom / rotation rescue
            PG.extend(per)
            st = LS.loftr_refine(LS.state_from_model(best[2], mri), mri, photos, scale_f, log, dev=C.DEV, R=R)
        PG.start(2 * N, "Writing outputs")
        LS.export_state(classical_st, mri, photos, scale_f, os.path.join(out_dir, "classical"), thickness=spacing)
        summary = LS.export_state(st, mri, photos, scale_f, out_dir, thickness=spacing)
    PG.done()
    log("")
    log(f"rotation angles (x, y, z): {', '.join(f'{a:.2f}' for a in summary['angles_xyz_deg'])}")
    log(f"{'photo':<28}{'slice':>6}{'LoFTR matches':>15}")
    for s in summary["slabs"]:
        n = s.get("loftr_inliers")
        flag = "   <- few matches, check qc" if n is not None and n < LS.WEAK else ""
        log(f"{s['photo']:<28}{s['index']:>6}{'' if n is None else n:>15}{flag}")
    lesions = None
    if wmh or lesion_mask:
        from . import lesion as LE
        if not lesion_mask:
            lesion_mask = os.path.join(out_dir, "wmh.nii.gz")
            LE.segment_wmh(t1_path, lesion_mask, model=wmh_model, log=log)
        lesions = LE.transfer(st, mri, photos, scale_f, LE.load_mask(lesion_mask, mri), out_dir, spacing, C.DEV, log=log)
    if ppt:
        from . import report as RP
        name = os.path.basename(os.path.normpath(os.path.abspath(out_dir)))
        RP.build(out_dir, name, summary, st, mri, photos, lesions=lesions, log=log)
    PG.done()
    log(f"results in {out_dir}  ({time.time() - t0:.0f}s)")
    return summary
