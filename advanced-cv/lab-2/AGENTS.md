# Project conventions

- Write source code, comments, configuration descriptions, and documentation in English.
- Do not create `__init__.py` files. Use Python namespace packages.
- Split domain models into files by responsibility in `alseg/core/`; do not collect all models in one file. Keep ABC contracts in `alseg/interfaces.py`.
- `alseg/adapters/` contains only classes implementing interfaces. Put standalone data helpers in `alseg/data.py` or another relevant module outside adapters.
- Keep entry points in root `script/` (next to `alseg/`). Share presentation/options through `alseg/cli_common.py`; keep that file there.
- Keep future experiment orchestration in `alseg/pipeline.py` and evaluation functions in `alseg/metrics.py` when implemented.
- Core models must not import adapters or entry-point code. Use ABCs at actual component boundaries, not for every helper.
- Use `rich_click` for all CLI commands.
- In notebooks, launch scripts using `%run` (preferred) or `!python`. Do not use `subprocess` to launch scripts or shell commands.
- Distribute project source to Kaggle as a source-code dataset and update that dataset for releases. Increase `[project].version` in `pyproject.toml` before every Kaggle dataset update; never publish changed source with the same version.
- Keep source on the read-only Kaggle input mount. Write caches, test scratch files, and run outputs only under `/kaggle/working/` or a temporary directory.
- Use the same source dataset version, project version, and experiment config for both teammates. Restart the kernel after switching source dataset versions.
- Never commit a Kaggle Jupyter server URL or access token.
