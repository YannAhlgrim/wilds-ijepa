# Decisions

## 2026-11-24 — Frozen external baseline implementation

- **Verified library identifiers and dimensions.** `timm 1.0.31` registry/config
  inspection confirmed `vit_giant_patch14_dinov2.lvd142m` (1536),
  `vit_large_patch14_dinov2.lvd142m` (1024), `vit_huge_patch14_224.mae` (1280),
  `vit_huge_patch14_224.orig_in21k` (1280),
  `vit_large_patch14_clip_224.openai` (1024 before projection), and
  `vit_large_patch14_clip_336.openai` (1024 before projection). The DINOv2 names
  map to the same public weight files as torch.hub `dinov2_vitg14` and
  `dinov2_vitl14`. I use timm for all six encoders to share a single, locally
  installed implementation and its normalized intermediate-feature API rather
  than execute a remote hub repository. No public checkpoints were downloaded.
- **CLIP uses width 1024, not 768.** timm's OpenAI CLIP config advertises 768
  output classes because that is the text-alignment projection. The factory
  constructs `num_classes=0`, removes that projection/classifier, and exposes
  normalized pre-projection tokens. The requested `clip_vitl14_336` key uses the
  native 336-pixel OpenAI weights and CLIP normalization.
- **Pooling compatibility.** `TimmExternalEncoder` follows the existing I-JEPA
  token interface, so `ViTClassifier` and both optimizer/head paths remain the
  single probing implementation. `last4_avgpool_concat` calls timm
  `forward_intermediates(..., indices=4, norm=True, output_fmt="NLC")`, which
  matches I-JEPA's application of final norm to selected layers. Prefix slicing
  uses `num_prefix_tokens` and falls back to CLS plus `num_reg_tokens` /
  `num_register_tokens`; CLS and every register are excluded from patch means.
  `cls` returns only the normalized pre-projection CLS token as an ablation.
- **Preprocessing follows the milestone, with one recorded caveat.** CLIP uses
  the published OpenAI mean/std and every non-CLIP baseline uses the standard
  ImageNet mean/std explicitly in `data.normalization`. In particular, timm's
  metadata for `vit_huge_patch14_224.orig_in21k` advertises 0.5/0.5 original
  checkpoint normalization, whereas SPEC R3 explicitly requires ImageNet stats
  for non-CLIP baselines. The configs follow SPEC so the comparison protocol is
  explicit and consistent; this can be revisited separately if a strict
  checkpoint-native supervised-ViT preprocessing ablation is desired.
- **Legacy I-JEPA behavior is unchanged.** External names are dispatched only
  through an explicit registry. Legacy models still use `init_model` with the
  same arguments, including construction of its unused predictor. Missing
  `data.normalization` resolves to exactly the previous ImageNet tuple.
  `validation.log_per_epoch_f1` defaults to false, so existing runs do not add
  an ID-val pass or change their training CSV. New baseline configs enable it;
  their CSV is copied into the preserved eval directory before the existing
  training-directory cleanup. Evaluation of newly trained external probes sets
  `pretrained=false` because the complete trained model checkpoint restores the
  backbone, avoiding a redundant public-weight download.
- **Config/grid defaults are placeholders.** Reduced HP grids contain exactly
  8 single-seed combinations per encoder and effective batch 512. Five-seed
  grids are valid and launchable but their LR/head/representation constants are
  explicitly marked to be replaced with the OOD-validation-selected HP row;
  no such selection can be made without training.
- **Retention aggregation.** Each seed computes `R = OOD F1 / ID F1`; the mean
  and sample standard deviation of those ratios are reported. A zero/missing ID
  F1 is undefined (`null`) and excluded from the aggregate. Existing gap and
  summary fields are retained and the new retention fields are appended.
- **Per-class F1 conventions.** Class 0 defaults to `empty` and is excluded
  from species frequency terciles. Other classes are sorted by `(Source count,
  class id)` and split with `np.array_split` into tail/mid/head, giving stable
  terciles even with tied counts. Following WILDS macro-F1, each bin averages
  classes present in that split's ground truth; per-class CSVs retain all class
  counts and zero-support rows.
- **Rank diagnostic.** Features are mean-centered before SVD. Effective rank is
  the exponential entropy of normalized singular values, and top-k variance is
  the cumulative fraction of squared singular values. A standard SVD roundoff
  tolerance excludes numerical null directions. Extraction supports both I-JEPA
  and external encoder checkpoints, defaults to CPU and 5,000 deterministic
  first split images, and never implicitly downloads public model weights.
- **Local verification environment.** The host only exposes Python 3.14, for
  which the repository's torch 2.0.1 pin has no wheel. The ignored `.venv` uses
  the current CPU-only torch/torchvision wheels plus timm 1.0.31 for tests. This
  does not alter the cluster's torch pins. Large baseline architectures are
  mocked at their declared widths in unit tests; one real random-weight timm
  tiny ViT tests the actual token/intermediate API without downloads. The
  optional `sup_vitl16_in21k` suggestion was not added because the requested
  supervised baseline is ViT-H/14 and the six-model launch matrix is already
  complete.
