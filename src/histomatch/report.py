"""PowerPoint report of a histomatch run (--ppt)."""
import io
import os

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import progress as PG

DARK, INK, MUTED, ACCENT, TEAL, TINT, WHITE = "1F2933", "222B33", "6B7280", "B4472E", "2F6F73", "EEF2F4", "FFFFFF"
FONT = "Calibri"
SW, SH = 13.333, 7.5          # slide size, inches
WEAK = 60


def _jpeg(img, maxdim=1500, quality=88):
    """BGR array -> in-memory JPEG, downsized so the deck stays small."""
    f = min(1.0, maxdim / max(img.shape[:2]))
    if f < 1:
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return io.BytesIO(buf.tobytes()), img.shape[1] / img.shape[0]


def overview_image(st, mri, labels):
    """Sagittal cut through the rotated MRI with one line per matched slice."""
    nx, ny, nz = mri.vol.shape
    c = mri.c
    Yg, Zg = np.meshgrid(np.arange(ny) - c[1], np.arange(nz) - c[2], indexing="ij")
    pts = (c + np.stack([np.zeros_like(Yg), Yg, Zg], -1) @ st["Rg"].T).reshape(-1, 3).T
    sl = ndi.map_coordinates(mri.vol, pts, order=1, cval=0).reshape(ny, nz)
    hi = np.percentile(mri.vol[::4, ::4, ::4], 99.5)
    k = 3
    img = cv2.cvtColor(np.flipud(np.clip(sl / hi * 255, 0, 255).astype(np.uint8).T), cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
    idx = np.rint(c[1] + st["Y"]).astype(int)
    for n, y in enumerate(idx):
        x = int(round((y + 0.5) * k))
        cv2.line(img, (x, 34), (x, img.shape[0] - 1), (60, 110, 240), 2, cv2.LINE_AA)
        t = str(labels[n])
        (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(img, t, (x - tw // 2, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def build(out_dir, title, summary, st, mri, photos, lesions=None, log=print):
    try:
        from pptx import Presentation
        from pptx.dml.color import RGBColor
        from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
        from pptx.util import Inches, Pt
    except ImportError as e:
        raise ValueError(f"--ppt needs the python-pptx package (pip install python-pptx): {e}")

    rgb = RGBColor.from_string
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)
    title_only, blank = prs.slide_layouts[5], prs.slide_layouts[6]

    def text(slide, s, x, y, w, h, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        lines = s if isinstance(s, list) else [s]
        for j, ln in enumerate(lines):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            p.alignment = align
            r = p.add_run()
            r.text = ln
            r.font.size, r.font.bold, r.font.name = Pt(size), bold, FONT
            r.font.color.rgb = rgb(color)
        return tb

    def content_slide(heading):
        s = prs.slides.add_slide(title_only)
        t = s.shapes.title
        t.left, t.top, t.width, t.height = Inches(0.5), Inches(0.35), Inches(SW - 1.0), Inches(0.9)
        tf = t.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        r = p.add_run()
        r.text = heading
        r.font.size, r.font.bold, r.font.name = Pt(32), True, FONT
        r.font.color.rgb = rgb(INK)
        return s

    def picture(slide, img, x, y, w, h, maxdim=1500):
        """Place BGR image inside the box (x, y, w, h), centred, aspect preserved. Returns its box."""
        stream, ar = _jpeg(img, maxdim)
        pw, ph = (w, w / ar) if w / ar <= h else (h * ar, h)
        px, py = x + (w - pw) / 2, y + (h - ph) / 2
        slide.shapes.add_picture(stream, Inches(px), Inches(py), Inches(pw), Inches(ph))
        return px, py, pw, ph

    def stat(slide, value, label, x, y, w, size=40):
        text(slide, value, x, y, w, 0.75, size=size, color=TEAL, bold=True, anchor=MSO_ANCHOR.BOTTOM)
        text(slide, label, x, y + 0.78, w, 0.4, size=14, color=MUTED)

    slabs = summary["slabs"]
    N = len(slabs)
    ang = summary["angles_xyz_deg"]
    matches = [s.get("loftr_inliers") for s in slabs]
    has_loftr = matches[0] is not None
    weak = [n for n, m in enumerate(matches) if has_loftr and m < WEAK]
    les = {r["n"]: r for r in lesions} if lesions else {}

    # ---- 1. title
    s = prs.slides.add_slide(blank)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(DARK)
    text(s, title, 0.8, 2.3, SW - 1.6, 1.3, size=48, color=WHITE, bold=True, anchor=MSO_ANCHOR.BOTTOM)
    text(s, "Slab photographs matched to the T1 MRI", 0.8, 3.75, SW - 1.6, 0.5, size=22, color="CBD5DA")
    text(s, f"{N} photos, anterior to posterior   |   {os.path.basename(str(mri_path(mri)))}", 0.8, 4.45, SW - 1.6, 0.4,
         size=14, color="9AA7AE")

    # ---- 2. where the cuts are
    s = content_slide("Where each photo sits in the MRI")
    ov = overview_image(st, mri, list(range(N)))
    cv2.imwrite(os.path.join(out_dir, "overview.png"), ov)
    _, py, _, ph = picture(s, ov, 0.5, 1.55, 8.3, 5.0, maxdim=1800)
    text(s, "Side view of the rotated MRI, anterior on the right. Each line is the matched slice of one photo "
            "(numbered from 0, the most anterior).", 0.5, py + ph + 0.12, 8.3, 0.6, size=12, color=MUTED)
    stat(s, f"{ang[0]:.1f}, {ang[1]:.1f}, {ang[2]:.1f}", "MRI rotation angles x, y, z (degrees)", 9.3, 1.6, 3.6, size=28)
    stat(s, str(N), "photos matched", 9.3, 3.1, 3.6)
    if has_loftr:
        stat(s, str(int(np.median(matches))), "median LoFTR matches per photo", 9.3, 4.6, 3.6)
        text(s, (f"{len(weak)} photo(s) under {WEAK} matches: check them" if weak
                 else f"All photos have at least {WEAK} matches"), 9.3, 6.05, 3.6, 0.5, size=14,
             color=ACCENT if weak else INK, bold=bool(weak))

    # ---- 3. one slide per photo
    nx, _, nz = mri.vol.shape
    PG.start(len(slabs), "PowerPoint slides")
    for n, sl in enumerate(slabs):
        PG.tick()
        s = content_slide(f"Photo {n}: {sl['photo']}")
        qc = cv2.imread(os.path.join(out_dir, "qc", f"{n}.png"))
        w3 = qc.shape[1] // 3
        panels = []
        photo = cv2.imread(photos[n]["path"])
        lm = cv2.imread(os.path.join(out_dir, "lesion_mask", f"{n}.png"), cv2.IMREAD_GRAYSCALE) if les else None
        f = min(1.0, 1500 / max(photo.shape[:2]))
        photo = cv2.resize(photo, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        cap = "Original photo"
        if lm is not None:                                   # lesion on the photographed face, in red
            face = cv2.resize(lm, (photo.shape[1], photo.shape[0]), interpolation=cv2.INTER_NEAREST) == 255
            photo[face] = (0.55 * photo[face] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
            cnt, _ = cv2.findContours(face.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(photo, cnt, -1, (0, 0, 255), 2)
            cap = "Original photo, lesion in red"
        # show the photo in the same left-right sense as the aligned photo and the MRI
        if np.linalg.det(st["P"][n][:2, :2] @ np.diag([1.0, -1.0])) < 0:
            photo = np.ascontiguousarray(photo[:, ::-1])
            cap += "\n(flipped left-right to match the MRI)"
        panels.append((photo, cap, 3.5 if lm is not None else 3.9))
        pw_ = 2.55 if lm is not None else 3.0
        mr = qc[:, 2 * w3:]                                  # plain MRI slice
        al = qc[:, :w3]                                      # photo aligned to the MRI grid
        ls = cv2.imread(os.path.join(out_dir, "lesion_slice", f"{n}.png"), cv2.IMREAD_GRAYSCALE) if les else None

        def with_lesion(img, tissue_only=False, k=3):
            big = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
            m = cv2.resize(ls, (big.shape[1], big.shape[0]), interpolation=cv2.INTER_LINEAR) > 127
            if tissue_only:
                m &= cv2.resize(img, (big.shape[1], big.shape[0]), interpolation=cv2.INTER_NEAREST).max(2) > 0
            big[m] = (0.45 * big[m] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
            return big

        if ls is not None:
            panels.append((with_lesion(al, tissue_only=True), "Photo aligned to the MRI grid,\nlesion in red", pw_))
        else:
            panels.append((al, "Photo aligned to the MRI grid", pw_))
        panels.append((mr, f"MRI slice {sl['index']}", pw_))
        if ls is not None:
            panels.append((with_lesion(mr), "MRI slice with lesion in red", pw_))
        gap = 0.35
        x = (SW - sum(p[2] for p in panels) - gap * (len(panels) - 1)) / 2
        for img, cap, w in panels:
            px, py, pw, ph = picture(s, img, x, 1.45, w, 4.45)
            text(s, cap.split("\n"), x, 6.0, w, 0.5, size=12, color=MUTED, align=PP_ALIGN.CENTER)
            x += w + gap
        bits = [f"MRI slice {sl['index']}"]
        if has_loftr:
            bits.append(f"{sl['loftr_inliers']} LoFTR matches")
        if les:
            bits.append(f"lesion on face {les[n]['face_lesion_mm2']:.0f} mm2")
        bad = has_loftr and sl["loftr_inliers"] < WEAK
        if bad:
            bits.append("few matches: check this one by eye")
        text(s, "   |   ".join(bits), 0.5, 6.65, SW - 1.0, 0.4, size=16, color=ACCENT if bad else INK, bold=True,
             align=PP_ALIGN.CENTER)

    path = os.path.join(out_dir, f"{title}.pptx")
    prs.save(path)
    PG.done()
    log(f"PowerPoint report: {path}  ({len(prs.slides)} slides)")
    return path


def mri_path(mri):
    return getattr(mri, "path", "T1")
