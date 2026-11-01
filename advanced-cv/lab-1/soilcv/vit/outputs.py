"""Save the full Transformer study in Kaggle Outputs."""

import json
import shutil

from soilcv.artifacts import file_hash, write_json
from soilcv.cnn.outputs import save_source

from .experiments import ROOT, jobs, verify_completed


def export_outputs():
    status = json.loads((ROOT / "status.json").read_text())
    if status["failed"] or len(status["jobs"]) != len(jobs()):
        raise RuntimeError("Export needs every planned fold job.")
    if any(job["state"] != "complete" for job in status["jobs"]):
        raise RuntimeError("Wait for all fold jobs to finish.")
    for recipe, fold, camera in jobs():
        name = recipe if camera is None else recipe + "__" + camera.lower()
        verify_completed(ROOT / "runs" / name / f"fold_{fold}")
    save_source(ROOT)
    files = [
        path
        for path in sorted(ROOT.rglob("*"))
        if path.is_file()
        and not {"wandb", "__pycache__"}.intersection(path.relative_to(ROOT).parts)
        and path.name not in {"outputs.json", "output_manifest.json"}
    ]
    write_json(
        ROOT / "output_manifest.json",
        {
            str(path.relative_to(ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
            for path in files
        },
    )
    submission = ROOT.parent / "submission.csv"
    shutil.copy2(ROOT / "submission.csv", submission)
    write_json(
        ROOT / "outputs.json",
        {
            "folder": str(ROOT),
            "files": len(files),
            "total_bytes": sum(path.stat().st_size for path in files),
            "submission": str(submission),
            "competition_submission_made": False,
        },
    )
    print("Saved study:", ROOT, "| top-level submission:", submission)
    return ROOT
