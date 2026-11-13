"""LoFTR stage: nail down the cutting angle and the slice of every photo.

Starting from the classical estimate, every photo is matched (pretrained LoFTR) against
a small fan of parallel MRI slices around its estimated position. Each match is a 3D
point in the volume. If the estimated cutting plane is tilted, one side of a photo
matches on slices further forward and the other side further back, so one shared plane
tilt (a, b) plus one depth per slab is fitted robustly to all matches of all photos.
Repeated a few times with a shrinking fan, then each photo is aligned in-plane to its
final slice with a RANSAC similarity on the LoFTR matches.

State (rotated frame "o", identical to the manual workflow):
    native_voxel = c + Rg @ (o - c)          c = (shape - 1) / 2
    Y[i]   depth (o-frame axis 1, relative to c) of photo i
    P[i]   3x3 matrix: centred in-plane o coords (X, Z, 1) -> photo pixel (u, v, 1)
           in the down-sampled photo used by the classical stage
"""
import json
import math
import os

import cv2
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from . import progress as PG
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation

HR = 3          # LoFTR photo resolution relative to the classical-stage photo
RES = 2         # LoFTR images are sampled at 1/RES native voxel
AMP = True      # run LoFTR in fp16 autocast on CUDA: ~30 % less memory, 2-3x faster, same matches
MEM_FRAC = 0.7  # share of the free GPU memory one LoFTR call may use
_LOFTR = None
_PAIR_BYTES = {}   # (H, W, photo features cached) -> measured peak GPU bytes of one photo/slice pair


def get_loftr(dev):
    global _LOFTR
    if _LOFTR is None:
        import kornia.feature as KF
        _LOFTR = KF.LoFTR(pretrained="outdoor").to(dev).eval()
    return _LOFTR


def pairs_per_call(dev, key):
    """How many photo/slice pairs one LoFTR call may hold. LoFTR's coarse matching keeps several
    (tokens x tokens) matrices per pair, ~1 GB per pair for a 0.37 mm MRI, so the fan of slices
    must be split to what the GPU has left. The cost of one pair is measured on the first call."""
    if not str(dev).startswith("cuda"):
        return 4
    free, total = torch.cuda.mem_get_info(dev)
    alloc = torch.cuda.memory_allocated(dev)
    # the allocator's unused cache is ours too, but never count more than the card holds:
    # on WSL the cache can grow past the physical memory (it spills into system RAM)
    free = min(free + torch.cuda.memory_reserved(dev) - alloc, total - alloc)
    return max(1, int(MEM_FRAC * free / _PAIR_BYTES[key]))


def run_loftr(dev, ph, sl):
    """LoFTR of photo(s) `ph` (1 or B,1,H,W) against slices `sl` (B,1,H,W), split into calls that
    fit in GPU memory. With a single photo its backbone features are computed once and reused.
    Returns keypoints0, keypoints1, confidence, batch_indexes (index into `sl`)."""
    loftr = get_loftr(dev)
    cuda = str(dev).startswith("cuda")
    B = sl.shape[0]
    cache_photo = ph.shape[0] == 1 and hasattr(loftr, "extract_features")   # kornia >= 0.9
    key = tuple(sl.shape[-2:]) + (cache_photo,)
    outs, k = [], 0
    with torch.autocast("cuda", dtype=torch.float16, enabled=cuda and AMP):
        feats = loftr.extract_features(ph) if cache_photo else None
        while k < B:
            measure = cuda and key not in _PAIR_BYTES
            n = 1 if measure else min(B - k, pairs_per_call(dev, key))
            if measure:
                torch.cuda.synchronize(dev)
                torch.cuda.reset_peak_memory_stats(dev)
                base = torch.cuda.memory_allocated(dev)
            inp = {"image1": sl[k:k + n]}
            if feats is not None:
                inp["features0"] = {name: v.expand(n, -1, -1, -1) for name, v in feats.items()}
            else:
                inp["image0"] = ph[k:k + n] if ph.shape[0] > 1 else ph.expand(n, -1, -1, -1)
            o = loftr(inp)
            if measure:
                torch.cuda.synchronize(dev)
                _PAIR_BYTES[key] = max(torch.cuda.max_memory_allocated(dev) - base, 1)
            o["batch_indexes"] = o["batch_indexes"] + k
            outs.append(o)
            k += n
    out = {name: torch.cat([o[name] for o in outs]) for name in outs[0]}
    return {name: v if name == "batch_indexes" else v.float() for name, v in out.items()}


# ----------------------------------------------------------------------------- state
def state_from_model(model, mri):
    Rg = model.Rg().detach().cpu().numpy().astype(np.float64)
    shift = Rg.T @ model.T.detach().cpu().numpy().astype(np.float64)
    Y = model.slab_y().detach().cpu().numpy().astype(np.float64) + shift[1]
    P = []
    for i in range(model.N):
        k = math.exp(float(model.logk + model.ls[i]))
        c, s = math.cos(float(model.th[i])), math.sin(float(model.th[i]))
        SX, SZ = float(model.sh[i, 0]) + shift[0], float(model.sh[i, 1]) + shift[2]
        cu, cv = [float(v) for v in model.cen[i]]
        fl = model.flip
        P.append([[fl * c / k, fl * s / k, cu - fl * (c * SX + s * SZ) / k],
                  [s / k, -c / k, cv - (s * SX - c * SZ) / k],
                  [0, 0, 1]])
    st = dict(Rg=Rg, Y=Y, P=np.array(P), spacing=math.exp(float(model.logt)), mirror=int(model.flip))
    return upright(st)


def upright(st):
    """Spin the rotated frame about the cutting normal so its vertical axis is as close as
    possible to the MRI's own vertical axis, and move that spin into the per-photo transforms.
    Photos taken sideways are then rotated to fit the MRI, not the MRI to fit the photos."""
    Rg = st["Rg"]
    ey = Rg[:, 1]
    ez = np.array([0.0, 0, 1]) - ey[2] * ey
    ez /= np.linalg.norm(ez)
    ex = np.cross(ey, ez)
    Rn = np.stack([ex, ey, ez], 1)
    Q = Rn.T @ Rg                                    # o_new = Q @ o_old, a rotation about y
    Q2 = np.array([[Q[0, 0], Q[0, 2], 0], [Q[2, 0], Q[2, 2], 0], [0, 0, 1]])
    st["P"] = st["P"] @ np.linalg.inv(Q2)
    st["Rg"] = Rn
    return st


# ----------------------------------------------------------------------------- rendering
class Renderer:
    def __init__(self, mri, photos, scale_f, dev):
        self.dev, self.mri = dev, mri
        v = torch.from_numpy(mri.vol).to(dev)
        self.vol = (v / torch.quantile(v.flatten()[::97], 0.995)).clamp(0, 1)[None, None]
        self.shp = torch.tensor(mri.vol.shape, dtype=torch.float32, device=dev)
        self.c = (self.shp - 1) / 2
        nx, _, nz = mri.vol.shape
        self.W, self.H = nx * RES // 8 * 8, nz * RES // 8 * 8          # LoFTR needs /8
        xi = (torch.arange(self.W, device=dev) + 0.5) / RES - 0.5       # o-frame x index
        zi = (nz - 1) - ((torch.arange(self.H, device=dev) + 0.5) / RES - 0.5)   # row 0 = superior
        self.Zo, self.Xo = torch.meshgrid(zi - self.c[2], xi - self.c[0], indexing="ij")   # (H,W) centred
        self.photos = []
        for p in photos:
            full = cv2.imread(p["path"], cv2.IMREAD_GRAYSCALE)
            h, w = p["mask"].shape
            hi = cv2.resize(full, (w * HR, h * HR), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            m = cv2.resize(p["mask"].astype(np.float32), (w * HR, h * HR), interpolation=cv2.INTER_LINEAR)
            self.photos.append(torch.from_numpy(hi * (m > 0.5)).to(dev)[None, None])

    def photo(self, i, P):
        P = torch.tensor(P, dtype=torch.float32, device=self.dev)
        u = P[0, 0] * self.Xo + P[0, 1] * self.Zo + P[0, 2]
        v = P[1, 0] * self.Xo + P[1, 1] * self.Zo + P[1, 2]
        img = self.photos[i]
        h, w = img.shape[-2:]
        g = torch.stack([((u + 0.5) * HR) / w * 2 - 1, ((v + 0.5) * HR) / h * 2 - 1], -1)
        return F.grid_sample(img, g[None], mode="bilinear", padding_mode="zeros", align_corners=False)[0]

    def slices(self, Rg, Ys):
        Rg = torch.tensor(Rg, dtype=torch.float32, device=self.dev)
        Ys = torch.as_tensor(Ys, dtype=torch.float32, device=self.dev)
        o = torch.stack([self.Xo[None].expand(len(Ys), -1, -1), Ys[:, None, None].expand(-1, self.H, self.W),
                         self.Zo[None].expand(len(Ys), -1, -1)], -1)
        p = self.c + o @ Rg.T
        g = ((p + 0.5) / self.shp * 2 - 1).flip(-1)
        return F.grid_sample(self.vol, g[None], mode="bilinear", padding_mode="zeros",
                             align_corners=False)[0, 0][:, None]         # (B,1,H,W)

    def px_to_o(self, kp):
        """LoFTR pixel (col,row) -> centred in-plane o coords (X,Z)."""
        nz = self.mri.vol.shape[2]
        X = (kp[:, 0] + 0.5) / RES - 0.5 - self.c[0]
        Z = (nz - 1) - ((kp[:, 1] + 0.5) / RES - 0.5) - self.c[2]
        return torch.stack([X, Z], -1)


def match_fan(R, st, i, dys, conf_thr=0.3, max_disp=40.0):
    """LoFTR photo i against slices at Y[i]+dys. Returns photo XZ, MRI XZ, Y, conf (tensors)."""
    ph = R.photo(i, st["P"][i])
    Ys = st["Y"][i] + np.asarray(dys, dtype=np.float64)
    sl = R.slices(st["Rg"], Ys)
    out = run_loftr(R.dev, ph[None], sl)
    cf = out["confidence"]
    a, b = R.px_to_o(out["keypoints0"]), R.px_to_o(out["keypoints1"])
    yy = torch.as_tensor(Ys, dtype=torch.float32, device=R.dev)[out["batch_indexes"]]
    keep = (cf > conf_thr) & ((a - b).norm(dim=1) < max_disp)
    return a[keep], b[keep], yy[keep], cf[keep]


# ----------------------------------------------------------------------------- plane fit
def fit_tilt(M, Y0, max_deg, step_deg, w, spacing):
    """M: list per slab of (photoXZ, mriXZ, Y, conf). Fit Y = Y0_i + c_i + a*X + b*Z.

    Returns a, b, c (N,), peak weight (N,). Robust: for each (a,b) each slab votes with
    the height of the strongest mode of its residual histogram (bin width w).
    """
    dev = M[0][1].device
    N = len(M)
    X = torch.cat([m[1][:, 0] for m in M]); Z = torch.cat([m[1][:, 1] for m in M])
    r0 = torch.cat([m[2] - y0 for m, y0 in zip(M, Y0)])
    cf = torch.cat([m[3] for m in M])
    sl = torch.cat([torch.full((len(m[3]),), i, device=dev, dtype=torch.long) for i, m in enumerate(M)])
    if len(cf) == 0:
        return 0.0, 0.0, np.zeros(N), np.zeros(N)
    t = torch.tan(torch.deg2rad(torch.arange(-max_deg, max_deg + 1e-6, step_deg, device=dev)))
    A, B = torch.meshgrid(t, t, indexing="ij")
    A, B = A.flatten(), B.flatten()
    K = len(A)
    NB = int(math.ceil((r0.abs().max().item() + 60) / w)) * 2 + 1
    r = r0[None] - A[:, None] * X[None] - B[:, None] * Z[None]                  # (K,M)
    bins = (r / w).round().long().clamp(-(NB // 2), NB // 2) + NB // 2
    idx = (torch.arange(K, device=dev)[:, None] * N + sl[None]) * NB + bins
    hist = torch.zeros(K * N * NB, device=dev).scatter_add_(0, idx.flatten(), cf[None].expand(K, -1).flatten())
    hist = F.conv1d(hist.view(K * N, 1, NB), torch.tensor([[[0.5, 1.0, 0.5]]], device=dev), padding=1).view(K, N, NB)
    peak, arg = hist.max(dim=2)                                                  # (K,N)
    k = int(peak.sum(1).argmax())
    a, b = float(A[k]), float(B[k])
    c = joint_depths(hist[k], np.asarray(Y0, dtype=np.float64), w, spacing)
    rk = r[k]
    cc, pw = [], []
    for i in range(N):                                                           # refine each mode
        s = (sl == i) & ((rk - c[i]).abs() <= 1.5 * w)
        ws = cf[s].sum()
        cc.append(float((rk[s] * cf[s]).sum() / ws) if ws > 0 else float(c[i]))
        pw.append(float(ws))
    return a, b, np.array(cc), np.array(pw)


def joint_depths(hist, Y0, w, spacing, sig=4.0, lam=0.15, cap=16.0):
    """Pick one depth bin per slab jointly: strong match modes, but consecutive slabs
    should sit about one slab spacing apart (the stack is a rigid block of parallel cuts)."""
    N, NB = hist.shape
    pk = hist.max(dim=1, keepdim=True).values
    h = hist / pk.clamp(min=1e-6) * (pk / 20.0).clamp(max=1.0)         # weak slabs vote less
    off = (torch.arange(NB, device=hist.device) - NB // 2).float() * w
    best = h[0].clone()
    back = []
    for i in range(1, N):
        gap = (Y0[i - 1] - Y0[i]) + off[:, None] - off[None, :]                  # [prev bin, this bin]
        pen = lam * (((gap - spacing) / sig) ** 2).clamp(max=cap)
        tot = best[:, None] - pen
        m, j = tot.max(dim=0)
        back.append(j)
        best = m + h[i]
    j = int(best.argmax())
    sel = [j]
    for bk in reversed(back):
        j = int(bk[j]); sel.append(j)
    sel = sel[::-1]
    return off[torch.tensor(sel, device=hist.device)]


def tilt_update(st, a, b, c):
    """Apply plane Y = Y_i + c_i + a X + b Z: rotate the frame, keep photos glued to anatomy."""
    ey = np.array([-a, 1.0, -b]); ey /= np.linalg.norm(ey)
    ex = np.array([1.0, 0, 0]) - ey[0] * ey; ex /= np.linalg.norm(ex)
    ez = np.cross(ex, ey)
    Rd = np.stack([ex, ey, ez], 1)
    for i in range(len(st["Y"])):
        Yc = st["Y"][i] + c[i]
        Mi = np.array([[ex[0] + a * ex[1], ex[2] + b * ex[1], Yc * ex[1]],
                       [ez[0] + a * ez[1], ez[2] + b * ez[1], Yc * ez[1]],
                       [0, 0, 1]])
        st["P"][i] = st["P"][i] @ np.linalg.inv(Mi)
        st["Y"][i] = Yc * ey[1]
    st["Rg"] = st["Rg"] @ Rd


def inplane_update(st, i, src, dst, thr=4.0):
    """RANSAC similarity photo->MRI on in-plane o coords; returns inlier count."""
    if len(src) < 8:
        return 0
    S, inl = cv2.estimateAffinePartial2D(src.astype(np.float32), dst.astype(np.float32), method=cv2.RANSAC,
                                         ransacReprojThreshold=thr, maxIters=2000, confidence=0.999)
    if S is None or inl.sum() < 8:
        return 0
    sc = math.hypot(S[0, 0], S[1, 0]); ang = abs(math.degrees(math.atan2(S[1, 0], S[0, 0])))
    if not (0.8 < sc < 1.25 and ang < 25 and np.abs(S[:, 2]).max() < 60):
        return 0
    st["P"][i] = st["P"][i] @ np.linalg.inv(np.vstack([S, [0, 0, 1]]))
    return int(inl.sum())


def ransac_count(src, dst, thr=4.0):
    if len(src) < 8:
        return 0, None
    S, inl = cv2.estimateAffinePartial2D(src.astype(np.float32), dst.astype(np.float32), method=cv2.RANSAC,
                                         ransacReprojThreshold=thr, maxIters=1000, confidence=0.99)
    return (0, None) if S is None else (int(inl.sum()), S)


def rescue(R, st, i, cen, log, scales=(0.6, 0.7, 0.82, 1.0, 1.2, 1.4, 1.65), rots=(-45, -30, -15, 0, 15, 30, 45),
           dys=(-12.0, 0.0, 12.0), chunk=21):
    """Photo i has almost no matches: its zoom or rotation differs from the other photos.
    Try a grid of in-plane scales / rotations (about the slab centre) and nearby depths and
    keep the combination LoFTR agrees with most."""
    p0 = np.linalg.inv(st["P"][i]) @ np.array([cen[0], cen[1], 1.0])           # slab centre in o coords
    cands = []
    for sc in scales:
        for rd in rots:
            c, s_ = sc * math.cos(math.radians(rd)), sc * math.sin(math.radians(rd))
            S = np.array([[c, -s_, p0[0] - c * p0[0] + s_ * p0[1]], [s_, c, p0[1] - s_ * p0[0] - c * p0[1]], [0, 0, 1]])
            for dy in dys:
                cands.append((S, dy))
    best = (0, None, None, 0.0)
    for k in range(0, len(cands), chunk):
        cc = cands[k:k + chunk]
        ph = torch.stack([R.photo(i, st["P"][i] @ np.linalg.inv(S)) for S, _ in cc])
        sl = R.slices(st["Rg"], [st["Y"][i] + dy for _, dy in cc])
        out = run_loftr(R.dev, ph, sl)
        a, b = R.px_to_o(out["keypoints0"]).cpu().numpy(), R.px_to_o(out["keypoints1"]).cpu().numpy()
        bi, ok = out["batch_indexes"].cpu().numpy(), (out["confidence"] > 0.3).cpu().numpy()
        for j, (S, dy) in enumerate(cc):
            m = ok & (bi == j)
            n, S2 = ransac_count(a[m], b[m])
            if n > best[0]:
                best = (n, S, S2, dy)
    n, S, S2, dy = best
    if n >= 25:
        st["P"][i] = st["P"][i] @ np.linalg.inv(S) @ np.linalg.inv(np.vstack([S2, [0, 0, 1]]))
        st["Y"][i] += dy
    return n


# ----------------------------------------------------------------------------- driver
SCHED = [(36, 6, 10.0, 1.0), (36, 6, 10.0, 1.0), (15, 3, 4.0, 0.5), (6, 2, 2.0, 0.25), (4, 1, 1.0, 0.125)]
WEAK = 60        # a photo with fewer LoFTR inliers than this on its final slice is suspect


def loftr_refine(st, mri, photos, scale_f, log, dev="cuda", R=None, rescue_weak=True):
    R = R or Renderer(mri, photos, scale_f, dev)
    N = len(photos)
    def finalise():
        out = []
        for i in range(N):
            PG.tick()
            n = 0
            for _ in range(2):
                pa, pb, yy, cf = match_fan(R, st, i, [0.0], conf_thr=0.2)
                n = inplane_update(st, i, pa.cpu().numpy(), pb.cpu().numpy(), thr=3.0)
            out.append(n)
        return out

    def fans(sched):
        for rng, step, mdeg, sdeg in sched:
            dys = np.arange(-rng, rng + 1, step)
            M = []
            for i in range(N):
                M.append(match_fan(R, st, i, dys))
                PG.tick()
            w = max(step / 2.0, 1.5)
            a, b, c, pw = fit_tilt(M, st["Y"], mdeg, sdeg, w, st["spacing"])
            good = pw >= 4.0
            ninl = []
            for i in range(N):                               # in-plane fix from matches on the fitted plane
                pa, pb, yy, cf = M[i]
                r = yy - float(st["Y"][i]) - a * pb[:, 0] - b * pb[:, 1]
                s = (r - float(c[i])).abs() <= 1.5 * w
                ninl.append(inplane_update(st, i, pa[s].cpu().numpy(), pb[s].cpu().numpy()) if good[i] else 0)
            tilt_update(st, a, b, c)
            log(f"  loftr fan +-{rng}/{step}: tilt change ({math.degrees(math.atan(a)):+.2f}, {math.degrees(math.atan(b)):+.2f}) deg, "
                f"depth change mean|c| {np.abs(c).mean():.1f} vox, weak slabs {int((~good).sum())}, "
                f"matches {int(sum(len(m[3]) for m in M))}, in-plane inliers {int(np.sum(ninl))}")

    fans(SCHED)
    final = finalise()
    weak = [i for i in range(N) if final[i] < WEAK]
    if weak and rescue_weak:
        PG.extend(len(weak) + (len(SCHED) - 2) * N + N)
        got = []
        for i in weak:
            got.append(rescue(R, st, i, photos[i]["cen"], log))
            PG.tick()
        log(f"  rescue of {len(weak)} weak photos (zoom / rotation search): inliers {got}")
        fans(SCHED[2:])                                      # refit tilt and depths with the repaired photos
        final = finalise()
    st["loftr_inliers"] = final
    log(f"  loftr final in-plane inliers per photo: {final}")
    return st


# ----------------------------------------------------------------------------- export
def export_state(st, mri, photos, scale_f, out_dir, thickness=24.5):
    for d in ("resizedCam", "mri_slices", "qc"):
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)
    Rg, c = st["Rg"], mri.c
    idx = np.rint(c[1] + st["Y"]).astype(int)
    ang = Rotation.from_matrix(Rg.T).as_euler("xyz", degrees=True)
    nx, ny, nz = mri.vol.shape
    Xg, Zg = np.meshgrid(np.arange(nx) - c[0], np.arange(nz) - c[2], indexing="ij")
    hi = np.percentile(mri.vol[::4, ::4, ::4], 99.5)
    stack = np.zeros((nx, ny, nz), np.uint8)
    info, grays = [], []
    for i, p in enumerate(photos):
        PG.tick()
        o = np.stack([Xg, np.full_like(Xg, idx[i] - c[1]), Zg], -1)
        sl = ndi.map_coordinates(mri.vol, (c + o @ Rg.T).reshape(-1, 3).T, order=1, cval=0).reshape(nx, nz)
        P = st["P"][i]
        pu = (P[0, 0] * Xg + P[0, 1] * Zg + P[0, 2]).astype(np.float32)
        pv = (P[1, 0] * Xg + P[1, 1] * Zg + P[1, 2]).astype(np.float32)
        full = cv2.imread(p["path"], cv2.IMREAD_COLOR)
        kk = scale_f / math.sqrt(abs(np.linalg.det(P[:2, :2])))          # native vox per full-res photo px
        if kk < 1:
            full = cv2.GaussianBlur(full, (0, 0), 0.5 / kk)
        rgb = cv2.remap(full, (pu + 0.5) / scale_f - 0.5, (pv + 0.5) / scale_f - 0.5, cv2.INTER_LINEAR, borderValue=0)
        msk = cv2.remap(p["mask"].astype(np.float32), pu, pv, cv2.INTER_LINEAR, borderValue=0) > 0.5
        rgb[~msk] = 0
        img = np.flipud(np.transpose(rgb, (1, 0, 2)))                    # rows: superior at top, cols: +x
        sl_img = np.flipud(np.clip(sl / hi * 255, 0, 255).astype(np.uint8).T)
        cv2.imwrite(os.path.join(out_dir, "resizedCam", f"{i}.png"), img)
        cv2.imwrite(os.path.join(out_dir, "mri_slices", f"{i}.png"), sl_img)
        qc = cv2.cvtColor(sl_img, cv2.COLOR_GRAY2BGR)
        cnt, _ = cv2.findContours(np.ascontiguousarray(np.flipud(msk.T)).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(qc, cnt, -1, (0, 255, 255), 1)
        cv2.imwrite(os.path.join(out_dir, "qc", f"{i}.png"), np.hstack([img, qc, cv2.cvtColor(sl_img, cv2.COLOR_GRAY2BGR)]))
        grays.append(np.flipud(cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2GRAY)).T)
        row = dict(photo=os.path.basename(p["path"]), index=int(idx[i]), mm_per_photo_px=float(kk * mri.vox))
        if "loftr_inliers" in st:
            row["loftr_inliers"] = int(st["loftr_inliers"][i])
        info.append(row)
    # Stacked volume with a fixed slab thickness: the photos keep their order and every slab is
    # drawn exactly `spacing` voxels thick, on the regular grid that best fits the matched slices.
    order = np.argsort(-idx)
    top = float(np.mean(idx[order] + thickness * np.arange(len(order))))
    edges = np.rint(top - thickness * np.arange(len(order) + 1)).astype(int)
    for r, i in enumerate(order):
        stack[:, max(edges[r + 1] + 1, 0):min(edges[r], ny - 1) + 1, :] = grays[i][:, None, :]
        info[i]["stack_range"] = [int(edges[r + 1] + 1), int(edges[r])]
    nib.save(nib.Nifti1Image(stack, mri.affine), os.path.join(out_dir, "png.nii.gz"))
    with open(os.path.join(out_dir, "angle_pos"), "w") as fh:
        fh.write(", ".join(f"{a:.2f}" for a in ang) + "\n" + ", ".join(str(int(v)) for v in idx) + "\n")
    summary = dict(angles_xyz_deg=[float(a) for a in ang], indices=[int(v) for v in idx], Rg=Rg.tolist(),
                   mirror=st["mirror"], stack_thickness_vox=thickness, slabs=info)
    with open(os.path.join(out_dir, "match.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary
