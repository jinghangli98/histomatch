# histomatch

Match photographs of coronal brain slabs to the T1 MRI of the same brain. Given a folder of
photos and a `T1.nii.gz`, it finds the cutting angle and the MRI slice each photo comes from,
and warps every photo onto its slice.

## Install

```
pip install -e . 
```

## Use

```
histomatch "path/to/CW23-36 anterior slab surface" path/to/T1.nii.gz -o results/CW23-36
```

Photos: one slab per photo, image files directly in the folder (sub-folders are ignored),
sorted by the last number in the file name and taken as anterior to posterior.

Options: `--spacing` expected slab thickness in MRI voxels (default 24.5)

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
histomatch "path/to/photos" path/to/T1.nii.gz -o results/CW23-36 --wmh --ppt
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

