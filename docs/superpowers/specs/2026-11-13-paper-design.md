# Paper design: automated MRI-guided lesion targeting in post mortem brain slabs

Date: 2026-11-13. Status: awaiting approval.

## Intent

Write a paper for post mortem brain researchers showing that the route from an ex vivo
7T T1 scan to lesion-marked slab photographs is now fully automatic. It is the sequel to
"Bridging Neuroimaging and Neuropathology" (Farhat, Li et al., PMC12204417), which
described the embedding, scanning and cutting-guide workflow but aligned MRI to photos by
hand (about 30 min per brain for guided slabs, longer for freehand) and reported neither
alignment error nor segmentation accuracy.

A reader should come away believing the two automatic stages are accurate enough to
replace the manual work, and be able to run them (`wmh_seg`, `histomatch`) on their own data.

## Goals

- Validate the post mortem lesion segmentation model (nnU-Net, 5-fold, 50 manually labelled cases).
- Validate slab-to-MRI alignment against the 40 manually matched brains, on the current code.
- Show the end result: lesions drawn on slab photos for targeted sampling.
- Release code with the paper.

## Non-goals

- Histology confirmation that transferred lesions contain lesion tissue (future work unless data appears).
- New method development. Known failures are reported, not fixed, unless a fix is small.
- Retraining or fine-tuning LoFTR.

## Anti-goals

- Reporting accuracy from runs that predate the current code.
- Claiming lesion transfer is validated when only its two inputs are.
- Treating manual matching as error-free ground truth.

## Constraints

- Segmentation numbers must come from out-of-fold predictions only.
- Photos and MRI are human tissue data; figures must carry no case identifiers.
- Target journal: Imaging Neuroscience; fallback Journal of Neuroimaging. (Claude's call, not yet confirmed by the user.)

## Approaches considered

1. **Follow-up note** (Journal of Neuroimaging / J Neurosci Methods): alignment accuracy, time saved, one lesion example. Least work, smallest claim.
2. **Stand-alone pipeline paper** (Imaging Neuroscience): both stages validated, rater agreement, ablation, positioning against FreeSurfer photo tools. All computational, data exists. **Chosen**: the user added the segmentation model and has 50 labelled cases, which makes the two-stage claim supportable.
3. **Imaging-to-pathology paper** (Acta Neuropathologica Communications): adds histology of sampled blocks. Most valuable, but depends on data not known to exist.

## Content

| Section | Experiment | Data |
|---|---|---|
| Segmentation | Dice, lesion-level sensitivity/precision, volume agreement, per-fold spread; in vivo model on the same scans as baseline; low-lesion cases for false positives | 50 labelled cases, out-of-fold |
| Alignment | Cutting-angle and slice-index error vs manual, per brain and per slab; classical-only vs classical+LoFTR; guided vs freehand (first two / last two) slabs; reversed and sideways cases; weak-match flag as a confidence signal | 40 matched brains, fresh run |
| Rater agreement | Second rater re-matches about 10 brains | New |
| End to end | Lesion overlays on photos for several brains; run time vs manual time | Brains in both sets |
| Related work | Position against FreeSurfer photo reconstruction and reference-free method; run them on hemispheres if feasible | Computational |
| Limits | Per-photo zoom variation, two slabs per photo, parallel-cut assumption | Existing failures |

Figures: pipeline overview, segmentation examples and scores, LoFTR matching and plane fit
(reuse explainer figures), error distributions, ablation, failure cases, end-to-end overlays.

## Decided by the user

- Audience: post mortem researchers.
- Include the post mortem segmentation model alongside the alignment.
- Segmentation was trained with nnU-Net 5-fold on 50 manually labelled cases.

## Claude's calls (open to correction)

- Journal choice and fallback.
- Histology left as future work.
- Second-rater subset of about 10 brains.
- Manuscript drafted in LaTeX under `histomatch/paper/`.

## Needed from the user

- Location of the nnU-Net results folder (fold validation predictions) and the 50 labels.
- Which of the 50 are also among the 40 aligned brains.
- A second rater, the author list, and whether any histology exists.

## Left to the builder

Statistical tests, exact figure layout, evaluation script structure, run order.
