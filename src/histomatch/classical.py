"""Classical stage: photo masks, MRI local-contrast volume, coarse search, refinement."""
import json
import math
import os
import re
import time

import cv2
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation
from skimage.filters import threshold_otsu

from . import progress as PG

torch.set_grad_enabled(False)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEV)
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
SPACING_VOX = 24.5      # typical slab spacing in native voxels (~9.15 mm)
AREA_REF_MM2 = 4400.0   # typical largest slab cross-section


# ----------------------------------------------------------------------------- photos
def list_photos(folder):
    fs = [f for f in os.listdir(folder)
          if f.lower().endswith(IMG_EXT) and not f.startswith(".")
          and os.path.isfile(os.path.join(folder, f))]

    def key(f):
        nums = re.findall(r"\d+", os.path.splitext(f)[0])
        return (int(nums[-1]) if nums else 0, f)
    return [os.path.join(folder, f) for f in sorted(fs, key=key)]


def tissue_mask(bgr):
    """Tissue = warm coloured pixels on a blue (or black) background."""
    b, g, r = [bgr[..., i].astype(np.float32) for i in range(3)]
    mx = bgr.max(2)
    if (mx < 16).mean() > 0.15:                       # background already removed
        m = mx > 24
    else:
        score = r - b
        m = score > threshold_otsu(score)
    k = max(1, round(max(m.shape) / 200))
    m = ndi.binary_opening(m, iterations=k)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sz = ndi.sum(m, lab, range(1, n + 1))
    keep = sz > 0.04 * sz.max()
    # a piece cut off by the photo edge is a neighbouring slab creeping into frame
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    for e in edge[edge > 0]:
        if e - 1 != sz.argmax():
            keep[e - 1] = False
    return np.isin(lab, np.where(keep)[0] + 1)


def load_photos(paths, maxdim=384):
    """Load all photos with ONE common downsampling factor (keeps relative scale)."""
    raw = [cv2.imread(p, cv2.IMREAD_COLOR) for p in paths]
    for p, im in zip(paths, raw):
        if im is None:
            raise IOError(f"cannot read {p}")
    f = maxdim / max(max(im.shape[:2]) for im in raw)
    photos = []
    for p, im in zip(paths, raw):
        PG.tick()
        sm = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        m = tissue_mask(sm)
        if m.sum() < 30:
            raise ValueError(f"no tissue found in {p}")
        rr, cc = np.nonzero(m)
        photos.append(dict(path=p, bgr=sm, mask=m, cen=(cc.mean(), rr.mean()),
                           area=float(m.sum()),
                           radius=float(np.hypot(cc - cc.mean(), rr - rr.mean()).max())))
    return photos, f


# ----------------------------------------------------------------------------- MRI
def gauss1d(sigma):
    r = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def blur(x, sigma, dims):
    """Separable gaussian blur of tensor x (B,1,...) along the listed spatial dims."""
    k = gauss1d(sigma)
    nd = x.dim() - 2
    conv = F.conv3d if nd == 3 else F.conv2d
    for d in dims:
        shape = [1, 1] + [1] * nd
        shape[2 + d] = -1
        pad = [0, 0] * nd
        pad[2 * (nd - 1 - d)] = pad[2 * (nd - 1 - d) + 1] = len(k) // 2
        x = conv(F.pad(x, pad, mode="replicate"), k.view(shape))
    return x


def local_contrast(x, sigma, eps):
    """(x - local mean) / local std ; x is (B,1,...)"""
    dims = list(range(x.dim() - 2))
    mu = blur(x, sigma, dims)
    var = (blur(x * x, sigma, dims) - mu * mu).clamp(min=0)
    return (x - mu) / torch.sqrt(var + eps * eps)


class MRI:
    def __init__(self, t1_path):
        img = nib.load(t1_path)
        self.path = t1_path
        self.vol = np.asarray(img.dataobj).astype(np.float32)
        self.affine = img.affine
        self.shape = np.array(self.vol.shape, dtype=np.float64)
        self.c = (self.shape - 1) / 2.0
        self.vox = float(np.mean(img.header.get_zooms()[:3]))
        v = self.vol / np.percentile(self.vol, 99)
        self.levels = {}
        for f in (8, 4, 2):
            lo = torch.from_numpy(ndi.zoom(ndi.gaussian_filter(v, f / 3.0), 1.0 / f, order=1)).to(DEV)
            L = local_contrast(lo[None, None], 2.0, 0.05)
            self.levels[f] = L.clamp(-3, 3)           # (1,1,X,Y,Z)

    def sample(self, f, pts):
        """Sample level-f contrast volume at native voxel coords pts (...,3)."""
        L = self.levels[f]
        shp = torch.tensor(self.shape, dtype=torch.float32)
        g = (pts + 0.5) / shp * 2 - 1                  # align_corners=False
        g = g.flip(-1)                                 # grid_sample wants (z,y,x) order
        return F.grid_sample(L, g.reshape(1, -1, 1, 1, 3), mode="bilinear",
                             padding_mode="zeros", align_corners=False).reshape(pts.shape[:-1])


# ----------------------------------------------------------------------------- geometry
def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rodrigues(w):
    """axis-angle (3,) -> rotation matrix, differentiable."""
    th = torch.sqrt((w * w).sum() + 1e-12)
    k = w / th
    K = torch.zeros(3, 3)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def photo_stack(photos, size=96):
    """Centroid-centred soft masks on a common canvas -> (N,size,size), px per canvas cell."""
    rad = max(p["radius"] for p in photos) * 1.15
    cell = 2 * rad / size
    out = []
    for p in photos:
        xs = p["cen"][0] + (np.arange(size) - (size - 1) / 2) * cell
        ys = p["cen"][1] + (np.arange(size) - (size - 1) / 2) * cell
        mx, my = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
        sm = cv2.GaussianBlur(p["mask"].astype(np.float32), (0, 0), max(cell / 2, 0.5))
        out.append(cv2.remap(sm, mx, my, cv2.INTER_LINEAR, borderValue=0))
    return torch.from_numpy(np.stack(out)).to(DEV), cell


# ----------------------------------------------------------------------------- coarse search
def coarse_search(mri, photos, log, f=8, flips=(1, -1)):
    L = mri.levels[f][0, 0]
    FL = torch.fft.rfftn(L).conj()
    shp = torch.tensor(L.shape, dtype=torch.float32)
    stack, cell = photo_stack(photos)                  # canvas rows = photo y (down)
    N, S, _ = stack.shape
    vol = stack[None, None]                            # depth = slab index
    idx = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float32) for n in L.shape],
                                     indexing="ij"), -1)
    nat = (idx + 0.5) * f - 0.5 - torch.tensor(mri.c, dtype=torch.float32)   # native vox, centred
    k0 = math.sqrt(AREA_REF_MM2 / mri.vox ** 2 / max(p["area"] for p in photos))
    t = SPACING_VOX
    res = []
    # rev = 1: the brain lies the other way round in the scan (anterior at the low end of
    # axis 1). Together with the mirror this gives four classes; all are searched, because
    # which one applies depends on the hemisphere and on how it was placed in the container.
    PG.start(2 * len(flips) * 7 * 7 * 12 * 4, "Coarse search")
    for rev, flip in [(r_, f_) for r_ in (0, 1) for f_ in flips]:
        for ax in range(-30, 31, 10):
            for az in range(-30, 31, 10):
                for ay in range(0, 360, 30):
                    Rg = rot_z(math.pi * rev) @ rot_x(math.radians(ax)) @ rot_z(math.radians(az)) @ rot_y(math.radians(ay))
                    o = nat @ torch.tensor(Rg, dtype=torch.float32)      # o = Rg^T n
                    for ks in (0.82, 0.92, 1.03, 1.15):
                        k = k0 * ks                                      # native vox per photo px
                        col = flip * o[..., 0] / (k * cell)              # canvas cells from centre
                        row = -o[..., 2] / (k * cell)
                        dep = (N * t / 2 - o[..., 1]) / t - 0.5
                        g = torch.stack([col / (S / 2), row / (S / 2), (dep + 0.5) / N * 2 - 1], -1)
                        P = F.grid_sample(vol, g[None], mode="bilinear", padding_mode="zeros",
                                          align_corners=False)
                        LP = local_contrast(P, 2.0, 0.05)[0, 0]
                        cc = torch.fft.irfftn(torch.fft.rfftn(LP) * FL, s=L.shape)
                        # cc[s] = sum_n LP(n) * L(n - s)  -> photo block must move by -s
                        j = int(cc.argmax())
                        sc = float(cc.flatten()[j] / (LP.norm() + 1e-6))
                        s = np.array(np.unravel_index(j, L.shape), dtype=np.float64)
                        s = np.where(s > np.array(L.shape) / 2, s - np.array(L.shape), s)
                        PG.tick()
                        res.append(dict(score=sc, Rg=Rg, T=-s * f, k=k, t=t, flip=flip,
                                        cls=(rev, flip),
                                        desc=f"rev{rev} flip{flip} ax{ax} az{az} ay{ay} ks{ks}"))
    res.sort(key=lambda r: -r["score"])
    out = []
    for r in res:                                       # the two best distinct orientations of each class
        same = [q for q in out if q["cls"] == r["cls"]]
        if len(same) < 2 and all(np.degrees(np.arccos(np.clip((np.trace(r["Rg"].T @ q["Rg"]) - 1) / 2, -1, 1))) > 12
                                 for q in same):
            out.append(r)
    for r in out:
        log(f"  coarse {r['score']:.3f}  {r['desc']}  T={np.round(r['T'])}")
    return out


# ----------------------------------------------------------------------------- refinement
class Model:
    """Global similarity + per-slab in-plane transform; renders photo/MRI pairs."""

    def __init__(self, mri, photos, init):
        self.mri, self.photos, self.N = mri, photos, len(photos)
        self.flip = init["flip"]
        self.cls = init["cls"]
        self.R0 = torch.tensor(init["Rg"], dtype=torch.float32)
        self.E = max(p["radius"] for p in photos) * 1.12        # half extent, photo px
        z = lambda *s: torch.zeros(*s, requires_grad=True)
        self.w = z(3)                                            # extra rotation (axis-angle)
        self.T = torch.tensor(init["T"], dtype=torch.float32).requires_grad_()
        self.logk = torch.tensor(math.log(init["k"])).requires_grad_()
        self.logt = torch.tensor(math.log(init["t"])).requires_grad_()
        self.th, self.sh, self.dy, self.ls = z(self.N), z(self.N, 2), z(self.N), z(self.N)
        self.masks = [torch.from_numpy(cv2.GaussianBlur(p["mask"].astype(np.float32), (0, 0), 1.0))[None, None].to(DEV)
                      for p in photos]
        self.cen = torch.tensor([p["cen"] for p in photos], dtype=torch.float32)

    def Rg(self):
        return self.R0 @ rodrigues(self.w)

    def slab_y(self):
        i = torch.arange(self.N, dtype=torch.float32)
        return (self.N / 2 - i) * torch.exp(self.logt) + self.dy

    def photo_coords(self, i, X, Z):
        """o-frame in-plane coords (native vox, relative to the slab origin) -> photo px."""
        k = torch.exp(self.logk + self.ls[i])
        c, s = torch.cos(self.th[i]), torch.sin(self.th[i])
        x, zz = X - self.sh[i, 0], Z - self.sh[i, 1]
        u = (c * x + s * zz) / k
        v = (-s * x + c * zz) / k
        return self.cen[i, 0] + self.flip * u, self.cen[i, 1] - v

    def score(self, f, slabs=None):
        """Mean per-slab edge correlation at MRI level f (differentiable)."""
        k = float(torch.exp(self.logk.detach()))
        n = int(2 * self.E * k / f) // 2 * 2 + 1
        lin = (torch.arange(n, dtype=torch.float32) - n // 2) * f
        Z, X = torch.meshgrid(lin, lin, indexing="ij")
        Rg, Y = self.Rg(), self.slab_y()
        c = torch.tensor(self.mri.c, dtype=torch.float32)
        tot = []
        for i in (range(self.N) if slabs is None else slabs):
            m = self.masks[i]
            pu, pv = self.photo_coords(i, X, Z)
            H, W = m.shape[-2:]
            g = torch.stack([(pu + 0.5) / W * 2 - 1, (pv + 0.5) / H * 2 - 1], -1)
            a = F.grid_sample(m, g[None], mode="bilinear", padding_mode="zeros", align_corners=False)
            a = local_contrast(a, 2.0, 0.05)[0, 0]
            o = torch.stack([X, Y[i].expand_as(X), Z], -1)
            b = self.mri.sample(f, c + self.T + o @ Rg.T)
            wgt = (a.abs().detach() * 4).clamp(max=1)
            tot.append((a * b).sum() / torch.sqrt((a * a).sum() * (wgt * b * b).sum() + 1e-6))
        return torch.stack(tot)

    def reg(self):
        return (0.02 * ((self.dy / 6) ** 2).mean() + 0.02 * ((self.ls / 0.05) ** 2).mean()
                + 0.004 * ((self.th / 0.3) ** 2).mean() + 0.004 * ((self.sh / 40) ** 2).mean()
                + 0.02 * ((self.logt - math.log(SPACING_VOX)) / 0.08) ** 2)

    def fit(self, f, iters, free):
        lrs = dict(w=0.004, T=1.0, logk=0.004, logt=0.003, th=0.01, sh=0.8, dy=0.4, ls=0.003)
        opt = torch.optim.Adam([dict(params=[getattr(self, n)], lr=lrs[n]) for n in free])
        with torch.enable_grad():
            for it in range(iters):
                PG.tick()
                opt.zero_grad()
                loss = -self.score(f).mean() + self.reg()
                loss.backward()
                opt.step()
        return float(self.score(f).mean())


def refine(mri, photos, cands, log):
    G = ["w", "T", "logk", "logt"]
    models = []
    PG.start(len(cands) * 120 + len({c["cls"] for c in cands}) * 400, "Refining the fit")
    for c in cands:
        m = Model(mri, photos, c)
        s = m.fit(4, 120, G)
        log(f"  global fit {s:.3f}  ({c['desc']})")
        models.append((s, m))
    models.sort(key=lambda x: -x[0])
    pick, seen = [], set()                              # the best candidate of every class goes on
    for x in models:
        if x[1].cls not in seen:
            seen.add(x[1].cls)
            pick.append(x)
    out = []
    for s, m in pick:
        m.fit(4, 150, G + ["th", "sh"])
        m.fit(4, 100, G + ["th", "sh", "dy", "ls"])
        s2 = m.fit(2, 150, G + ["th", "sh", "dy", "ls"])
        log(f"  full fit   {s2:.3f}")
        out.append((s2, m))
    out.sort(key=lambda x: -x[0])
    return out
