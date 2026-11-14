# Test inference

Use `from soilcv.inference.runner import run_checkpoints`.
Attach competition data, the current soil-cv-utils dataset, and `WANDB_API_KEY`.
Internet is needed to read W&B model artifacts. This code does not start W&B runs.

A string selects one checkpoint. It can be a run ID, run URL, or exact model artifact version.
A group selects four folds of one recipe from one training study:

```python
checkpoints = [
    {
        "name": "mil-r18-cosine-4fold",
        "checkpoints": [
            "public-team/soil-grain-size/mil-r18-cosine-fold-0:v0",
            "public-team/soil-grain-size/mil-r18-cosine-fold-1:v0",
            "public-team/soil-grain-size/mil-r18-cosine-fold-2:v0",
            "public-team/soil-grain-size/mil-r18-cosine-fold-3:v0",
        ],
    }
]
results = run_checkpoints(checkpoints, "/kaggle/working/submissions")
```

A complete group makes one CSV. Each fold uses its saved scale and input settings.
Predictions are joined by sample ID, then averaged with equal fold weights.
Folds 0, 1, 2, and 3 must each appear once. Mixed recipes or studies are rejected.
If a model is not ready, its whole group is skipped. A three-fold result is never saved as four-fold.

The manifest saves exact artifact versions, source runs, folds, and skipped references.
There is no automatic competition submit step. Re-running this notebook does not use a submission slot.
MIL CNN and full-frame ViT checkpoints are supported. The finished MIL source is checked with its original model and preprocessing code.
