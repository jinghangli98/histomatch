"""Command line entry point:  histomatch PHOTO_DIR T1 -o OUT_DIR"""
import argparse
import sys

from . import __version__

EPILOG = """\
outputs written to OUT_DIR
  angle_pos      line 1: rotation angles x, y, z (degrees); line 2: one slice index per photo
  match.json     the same plus per-photo details (LoFTR match count, mm per photo pixel)
  resizedCam/    N.png  each photo warped onto the grid of its MRI slice (N = 0 is the most anterior)
  mri_slices/    N.png  the matching slice of the rotated MRI
  qc/            N.png  photo | MRI with photo outline | MRI, for checking by eye
  png.nii.gz     the photos stacked on the T1 grid, every slab --spacing voxels thick
  classical/     the same outputs from the first (pre-LoFTR) stage, for comparison
with --ppt
  <OUT_DIR name>.pptx   report: where each photo sits in the MRI, then one slide per photo
with --wmh or --lesion-mask
  wmh.nii.gz        lesion mask on the T1 grid (only with --wmh)
  lesion_mask/      N.png  lesion mask at the full size of photo N: 255 = lesion on the photographed
                    face, 127 = lesion deeper in the slab (within --spacing voxels behind the face)
  lesion_overlay/   N.jpg  photo with lesions (red = on face, yellow outline = in slab) | MRI slice
  lesion_slice/     N.png  lesion mask on the grid of the rotated MRI slice (same grid as mri_slices/)
  lesions.csv       per photo: lesion area on the face / in the slab, position of the largest lesion

convention
  rotated[o] = T1[c + Rg @ (o - c)],  c = (shape - 1) / 2,
  Rg = scipy Rotation.from_euler("xyz", angles, degrees=True).as_matrix().T
  the slice of photo N is rotated[:, index_N, :]
"""


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="histomatch",
        description="Match coronal brain slab photographs to the T1 MRI they were cut from: "
                    "finds the cutting angle and the MRI slice of every photo.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", help="folder with the slab photos, one slab per photo; files directly in the folder "
                                   "are sorted by the last number in their name and taken as anterior -> posterior")
    ap.add_argument("t1", help="T1 MRI of the same brain (.nii or .nii.gz)")
    ap.add_argument("-o", "--out", required=True, help="output folder (created if missing)")
    ap.add_argument("--spacing", type=float, default=24.5,
                    help="expected slab thickness in MRI voxels (default: 24.5)")
    ap.add_argument("--wmh", action="store_true",
                    help="segment white matter lesions on the T1 with wmh_seg and transfer them onto the photos")
    ap.add_argument("--wmh-model", default="exvivo", choices=["exvivo", "unified", "invivo", "pmb"],
                    help="wmh_seg model used by --wmh (default: exvivo)")
    ap.add_argument("--lesion-mask", metavar="NIFTI",
                    help="use this existing lesion mask (on the T1 grid) instead of running wmh_seg")
    ap.add_argument("--ppt", action="store_true",
                    help="also write a PowerPoint report (<output folder name>.pptx) into the output folder")
    ap.add_argument("--no-loftr", action="store_true", help="classical stage only (faster, less accurate)")
    ap.add_argument("--fast", action="store_true",
                    help="follow up only 4 of the 8 starting orientations (about a minute quicker, "
                         "but can miss the right orientation)")
    ap.add_argument("--no-progress", action="store_true", help="do not draw progress bars")
    ap.add_argument("-v", "--verbose", action="store_true", help="print the progress of every search step")
    ap.add_argument("--version", action="version", version=f"histomatch {__version__}")
    a = ap.parse_args(argv)

    from .pipeline import run
    try:
        from . import progress as PG
        if a.no_progress:
            PG.ENABLED = False
        log = PG.write if a.verbose else (lambda m="": None if m.startswith("  ") else PG.write(m))
        run(a.photos, a.t1, a.out, use_loftr=not a.no_loftr, spacing=a.spacing, wmh=a.wmh,
            lesion_mask=a.lesion_mask, wmh_model=a.wmh_model, ppt=a.ppt, fast=a.fast, log=log)
    except (ValueError, IOError) as e:
        print(f"histomatch: error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
