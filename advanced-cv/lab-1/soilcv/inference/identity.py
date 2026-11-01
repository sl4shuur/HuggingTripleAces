"""Check the current source, or the one supported finished MIL study."""

import ast
import hashlib
import json
from pathlib import Path

from soilcv.artifacts import package_identity

MIL_STUDY_SOURCE = {
    "version": "0.1.1",
    "source_hash": "431a69bd03d926c50897eac4f358c2b14b0b7366ace8b9b0ab3ffaa5b6f83008",
    "folds_hash": "057fe8f93ba35d2ab7a3483063c95ca0079c4bcbb9978671c16ffc4119d9c1b2",
}
# These are the original model, curve head, and physical preprocessing sources.
# AST hashes ignore formatting changes, but include every executable statement.
MIL_DEPENDENCIES = {
    "bags.py": "f719a67c4854fc1504ff8d6f3812c9aed3dcc634fd4178fb640fd3363c7e0bfc",
    "cnn/model.py": "07f589623b09f7a46c0adc90345578ecb44fa14ee848e27713491930929b88d0",
    "nn.py": "210021c52e1afa5b9088a4b6e7485b6e39a048146c3651dc1d34f8acece9591f",
    "curves.py": "dbdc88eb37a39c30c3c1a15bb90d97f5abb80f784fa944e6416a1a5f3e67f38b",
}


def _syntax(value):
    if value is Ellipsis:
        return {"literal": "Ellipsis"}
    if isinstance(value, ast.AST):
        fields = {
            name: _syntax(child)
            for name, child in ast.iter_fields(value)
            if child is not None and child != []
        }
        return [type(value).__name__, fields]
    if isinstance(value, list):
        return [_syntax(child) for child in value]
    return value


def semantic_hash(path):
    # Python 3.13 changed ast.dump's handling of empty fields. Use our stable form.
    tree = ast.parse(Path(path).read_text())
    payload = json.dumps(_syntax(tree), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def check_model_source(config):
    saved = config["versions"]["soilcv"]
    current = package_identity()
    if saved == current:
        return
    if config["stage"] == "cnn_mil" and saved == MIL_STUDY_SOURCE:
        root = Path(__file__).parents[1]
        if current["folds_hash"] != saved["folds_hash"]:
            raise ValueError("The saved folds changed.")
        for relative, expected in MIL_DEPENDENCIES.items():
            if semantic_hash(root / relative) != expected:
                raise ValueError("MIL model dependency changed: " + relative)
        return
    raise ValueError("Use the source package saved with this model.")
