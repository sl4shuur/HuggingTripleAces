# MIL inference

Use inference.ipynb, competition data, the latest soil-cv-utils dataset, Internet and WANDB_API_KEY.
Three MIL run references are filled. Two slots are empty.
Each slot accepts a run ID, full W&B/Forge run URL, or exact model artifact handle.
The helper uses only a committed MIL model artifact. It skips runs whose weights are not ready.
There is no need to wait for the other training experiments.

Each ready model writes one CSV in /kaggle/working/submissions_mil.
There is no averaging between slots, training or new W&B run.
The manifest records the exact artifacts, source runs, and skipped references.
Only the matching MIL code can load a checkpoint. CPU works; a GPU is faster.
All photos and grid tiles enter each sample bag.
