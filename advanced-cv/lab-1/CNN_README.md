# Part 1: full-coverage CNN MIL

Use cnn.ipynb, the soil competition data, and the latest soil-cv-utils dataset.
Select two T4 GPUs, Internet, and the WANDB_API_KEY secret. Then Save & Run All.
The existing secret binding is reused. W&B entity is public-team. There are no CV-summary or online smoke runs.

One bag is one whole sample, with every photo and every grid tile. No tiles are sampled out.
The encoder runs in chunks of 8 tiles with gradient checkpointing. It pools features before the curve head.
One sample curve receives one EMD loss. Two sample losses make one optimizer update.
BN statistics stay frozen. All five recipes use 16 epochs at most, with patience 5.
The 224 and 288 recipes use the same native crop rectangles. Each cache uses one LANCZOS resize from the original.

Coverage is checked by exact rectangle union, per photo. coverage_224.csv and coverage_288.csv must show 100% and zero missing pixels.
Overlap-corrected area weights avoid over-counting edges. Photos have equal sample weight.
This does not promise lossless resizing or prove the absolute physical scale.

Outputs in /kaggle/working:

- submission.csv: the selected recipe's four-fold mean prediction. No competition upload is sent.
- part1_mil/: weights, best/last checkpoints, OOF, camera curves, configs, logs, coverage, paired errors and hashes.
- part1_mil/CNN_RESULTS.md: current-run report, generated from results/cnn_results.csv.
- part1_mil/source/: code, pre-run hypotheses and environment with pyproject.toml, uv.lock and actual runtime versions.

HYPOTHESES.md contains pre-run predictions. CNN_RESULTS.md has the current run's numbers.
Only the MIL pipeline is supported by the inference notebook.

Every real fold reloads weights through the public inference path and checks one complete test sample.
The same AMP setting must reproduce test_predictions.csv within 0.001 percentage points.
Standalone inference reads tile size, grid, resolution, normalization and pooling from the saved config.

```bash
PYTHONPATH=part1_mil/source python -m soilcv.cnn.inference \
  --photo photo.jpg --ppm 4.6 \
  --checkpoint-dir part1_mil/runs/mil-r18-288/fold_0 \
  --device cuda:0 --output prediction.csv
```

Supply the actual PPM. Use all four selected fold folders for the submission ensemble.

The paired camera check reuses held-out features. Both camera label errors use the same paired samples.
Prediction disagreement is not label error. Camera and view can both change; this is not an iPhone test.
Best-epoch selection stays. A code TODO records EMA/fixed-epoch and extra-split checks for later.
