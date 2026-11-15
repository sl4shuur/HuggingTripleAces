# Project conventions

- Write source code, comments, configuration descriptions, and documentation in English.
- Keep domain models and invariants in `alseg/core/`, ABC contracts in `alseg/interfaces.py`, concrete integrations in `alseg/adapters/`, and commands in `alseg/cli/`. Keep future experiment orchestration in `alseg/pipeline.py` and evaluation functions in `alseg/metrics.py` when those modules are implemented.
- Core models must not import adapters or CLI code. Adapters implement relevant interfaces; use ABCs at actual component boundaries, not for every helper.
- Use `rich_click` for all CLI commands.
- In notebooks, launch scripts using `%run` (preferred) or `!python`. Do not use `subprocess` to launch scripts or shell commands.
- Distribute project source to Kaggle as a source-code dataset and update that dataset for releases. Increase `[project].version` in `pyproject.toml` before every Kaggle dataset update; never publish changed source with the same version.
- Keep source on the read-only Kaggle input mount. Write caches, test scratch files, and run outputs only under `/kaggle/working/` or a temporary directory.
- Use the same source dataset version, project version, and experiment config for both teammates. Restart the kernel after switching source dataset versions.
- Never commit a Kaggle Jupyter server URL or access token.
