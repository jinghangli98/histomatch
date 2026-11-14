# histomatch

Match photographs of coronal brain slabs to the T1 MRI of the same brain. Given a folder of
photos and a `T1.nii.gz`, it finds the cutting angle and the MRI slice each photo comes from,
and warps every photo onto its slice.

## Install

```
pip install -e . 
```

`--no-deps` keeps pip from touching an existing PyTorch/CUDA install. Without it, pip installs
the dependencies listed in `pyproject.toml`. A GPU is used when available; the first run
downloads the pretrained LoFTR "outdoor" weights (44 MB).

LoFTR runs in fp16 and its calls are split to fit the free GPU memory (the cost of one
photo/slice pair is measured on the first call), so an 8-12 GB card handles a 0.37 mm T1.

## Use

```
histomatch "path/to/CW23-36 anterior slab surface" path/to/T1.nii.gz -o results/CW23-36
```

Photos: one slab per photo, image files directly in the folder (sub-folders are ignored),
sorted by the last number in the file name and taken as anterior to posterior.

Options: `--spacing` expected slab thickness in MRI voxels (default 24.5), `--no-loftr`
classical stage only, `--fast` follow up only 4 of the 8 starting orientations (about a minute
quicker, can miss the right orientation). `histomatch -h` lists the outputs and the rotation convention.

## Outputs

| File | Content |
|---|---|
| `angle_pos` | rotation angles x, y, z, then one slice index per photo |
| `match.json` | the same plus per-photo LoFTR match count and scale |
| `resizedCam/N.png` | photo N warped onto the grid of its MRI slice |
| `mri_slices/N.png` | the matching slice of the rotated MRI |
| `qc/N.png` | photo, MRI with photo outline, MRI |
| `png.nii.gz` | photos stacked on the T1 grid, fixed slab thickness |
| `classical/` | the same outputs before the LoFTR stage |

## White matter lesions on the photos

```
histomatch "path/to/photos" path/to/T1.nii.gz -o results/CW23-36 --wmh
```

`--wmh` segments lesions on the T1 with [wmh_seg](https://github.com/jinghangli98/wmh_seg)
(`pip install -U wmh_seg`, version 1.6 or newer; ex-vivo model by default, `--wmh-model` to change) and transfers them onto
every photo. `--lesion-mask mask.nii.gz` uses an existing mask on the T1 grid instead.

| File | Content |
|---|---|
| `wmh.nii.gz` | lesion mask on the T1 grid (`--wmh` only) |
| `lesion_mask/N.png` | mask at the full size of photo N: 255 = lesion on the photographed face, 127 = lesion deeper in the slab |
| `lesion_overlay/N.jpg` | photo with lesions (red = on face, yellow outline = in slab) next to the MRI slice |
| `lesions.csv` | per photo: lesion area on the face and in the slab, pixel position of the largest lesion |

A photo with fewer than 60 LoFTR matches is flagged in the printed table; check its `qc` image.

## How it works

1. Tissue masks from the photo background colour.
2. The MRI is converted to local contrast, so no brain mask is needed.
3. Coarse search of the photo stack against the MRI over tilt, in-plane rotation, mirror and scale.
4. Gradient refinement of the global pose plus small per-photo adjustments.
5. LoFTR matches each photo against a fan of parallel MRI slices;
   the matches of all photos fix the shared cutting angle and each photo's slice, with
   consecutive slabs kept about one slab thickness apart.
