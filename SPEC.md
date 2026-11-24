# SPEC — Frozen-encoder baselines and analysis tools for the TMLR submission

## Goal
The paper "Evaluating I-JEPA under Distribution Shifts on iWildCam2020-WILDS" (target: TMLR)
compares frozen I-JEPA linear probes only against fine-tuned leaderboard entries. Reviewers will
require **same-protocol frozen baselines** and a few cheap diagnostics. This milestone adds the
code and configs so the author can launch these runs on the SLURM cluster. No training is run on
this machine (no GPU, no WILDS data).

## Scope
### M1 — Frozen baseline encoders in the linear-probe pipeline
Add support in `src/train_supervised.py` / `src/eval_wilds.py` (via a small encoder factory, e.g.
`src/models/external_encoders.py`) for these frozen encoders, loaded from public weights:

| key (`meta.model_name`) | source | patch/res | embed dim |
|---|---|---|---|
| `dinov2_vitg14` | torch.hub `facebookresearch/dinov2` `dinov2_vitg14` (or timm `vit_giant_patch14_dinov2.lvd142m`) | 14 / 224 | 1536 |
| `dinov2_vitl14` | torch.hub `dinov2_vitl14` (or timm `vit_large_patch14_dinov2.lvd142m`) | 14 / 224 | 1024 |
| `mae_vith14` | MAE ViT-H/14 IN-1K pretrain weights (facebookresearch/mae `mae_pretrain_vit_huge.pth`; or timm `vit_huge_patch14_224.mae`) | 14 / 224 | 1280 |
| `sup_vith14_in21k` | timm `vit_huge_patch14_224.orig_in21k` (supervised IN-21k) | 14 / 224 | 1280 |
| `clip_vitl14` | OpenAI CLIP ViT-L/14 image tower (timm `vit_large_patch14_clip_224.openai` or open_clip) | 14 / 224 | 1024 |

Verify identifiers/dims against the actual libraries; if a timm name differs, use the correct
one and document it in `DECISIONS.md`.

Requirements:
* R1. Encoders are frozen (`requires_grad=False`, `eval()` mode) exactly like the I-JEPA path.
* R2. Representations: support the existing `representation_type` values `last_avgpool` and
  `last4_avgpool_concat` (average of **patch tokens only**, excluding CLS/register tokens), so the
  protocol is identical to I-JEPA (which has no CLS token). Additionally support `cls` for
  models that have one (used only as an extra ablation).
* R3. Normalization: each encoder uses its own pretraining mean/std (CLIP uses CLIP stats;
  others ImageNet stats). Make this explicit in config (`data.normalization: imagenet|clip`).
  No data augmentation (same as I-JEPA runs).
* R4. Heads `linear` and `bn_linear`, optimizer, scheduler, gradient accumulation, early stopping,
  metric logging, `summary.json`, predictions CSVs — all reuse the existing code paths unchanged.
* R5. Existing I-JEPA configs and results must remain bit-for-bit unaffected (no default changes).

### M2 — Configs and grids
* `configs/supervised_<key>.yaml` for each encoder (based on `configs/supervised_vith14_224_in22k.yaml`).
* Reduced HP grid per encoder in `configs/grids/hp/<key>.yaml` usable with `tools/run_grid.py`:
  lr ∈ {1e-3, 1e-2}, head ∈ {linear, bn_linear}, representation ∈ {last_avgpool, last4_avgpool_concat},
  wd = 5e-4, effective batch 512 (micro-batch 16 × 32 accumulation), 1 seed → 8 runs/encoder.
* Seed grids `configs/grids/seeds/<key>.yaml` (5 seeds 0–4) to be filled with the selected config.
* A short `docs/baselines.md` with the exact launch commands (SLURM) and estimated GPU-hours.

### M3 — Analysis tools (CPU-only, run on saved outputs)
* `tools/retention.py` (or extend `aggregate_seeds.py`): add **relative retention**
  `R = F1_OOD / F1_ID` (mean ± std across seeds, computed per seed) next to the existing gap Δ.
* `tools/per_class_f1.py`: from `*_predictions.csv` + WILDS metadata/labels, compute per-class F1
  on ID/OOD test and aggregate macro-F1 by training-frequency bins (e.g. head/mid/tail species by
  Source-split count terciles) and the F1 of the `empty` class. Output CSV + a LaTeX-ready table.
  If labels need the WILDS dataset, read them via the `wilds` package from `root_path`.
* `tools/feature_rank.py`: for a given frozen encoder checkpoint, extract pooled features on N
  (default 5,000) images of a split and report effective rank (exp of entropy of normalized
  singular values) and the fraction of variance in the top-k PCs — a collapse diagnostic for the
  iWildCam-pretrained I-JEPA checkpoints vs. ImageNet ones.
* `tools/plot_val_curve.py`: from the training logs/CSV of a run, extract per-epoch OOD-val F1
  and ID-val F1 (if logged); output CSV usable by pgfplots. If per-epoch metrics are not currently
  logged, add logging of them to the training loop (append-only, no behavior change).

## Non-goals
* Running training or downloading the WILDS dataset or large checkpoints on this machine.
* Changing the I-JEPA pretraining code.
* Pushing to GitHub (the author will review and push).

## Acceptance criteria
1. `python -m pytest` (add `tests/` if none exists) passes on CPU and covers:
   a) each new encoder key builds with *random* weights (no download; `pretrained=False` path)
      at small size or mocked, returns features of the declared dim for `last_avgpool` and
      `last4_avgpool_concat`, and has zero trainable parameters;
   b) patch-token pooling excludes CLS/register tokens (unit test with a tiny ViT);
   c) `retention.py` and `per_class_f1.py` produce correct values on a synthetic fixture
      (hand-computed expected numbers);
   d) `feature_rank.py` returns rank ≈ d for isotropic random features and ≈ 1 for rank-1 features.
2. All new YAML configs parse and pass the same config validation the training entry point uses
   (dry-run / `--help`-level instantiation without data).
3. Existing configs load unchanged; `git diff` shows no modification of existing default values.
4. `docs/baselines.md` lists every launch command and expected outputs.
5. Engineer's report lists files changed, test output, and any identifier/dimension corrections.

## Constraints
* Python/PyTorch stack as in `requirements.txt`; adding `timm` (and optionally `open_clip_torch`)
  to requirements is allowed. Keep imports of optional libs lazy so the I-JEPA path doesn't need them.
* Local commits only; do not push.
