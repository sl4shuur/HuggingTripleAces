"""Run independent fold processes and keep their logs in the study folder."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .artifacts import write_json


def save_status(root, jobs, failed):
    temporary = Path(root) / "status.tmp.json"
    write_json(temporary, {"jobs": jobs, "failed": failed, "updated": time.time()})
    temporary.replace(Path(root) / "status.json")


def run_jobs(jobs, root, module, *, gpu_count=2):
    root = Path(root)
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "WANDB_CONSOLE": "off"}
    if not environment.get("WANDB_API_KEY"):
        raise ValueError("Read the WANDB_API_KEY secret in the notebook first.")
    jobs = [{**job, "state": "pending"} for job in jobs]
    active = {}
    failed = False
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        while any(job["state"] in {"pending", "running"} for job in jobs):
            for gpu, (process, stream, job) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                job.update(
                    state="complete" if code == 0 else "failed",
                    exit_code=code,
                    finished=time.time(),
                )
                failed = failed or code != 0
                print(
                    job["state"],
                    job["experiment"],
                    job["fold"],
                    job.get("train_camera"),
                    flush=True,
                )
                del active[gpu]
            for gpu in range(gpu_count):
                if gpu in active or failed:
                    continue
                job = next((job for job in jobs if job["state"] == "pending"), None)
                if job is None:
                    break
                suffix = (
                    "__" + job["train_camera"].lower()
                    if job.get("train_camera")
                    else ""
                )
                log_path = logs / f"{job['experiment']}{suffix}-fold-{job['fold']}.log"
                command = [
                    sys.executable,
                    "-m",
                    module,
                    "train",
                    "--experiment",
                    job["experiment"],
                    "--fold",
                    str(job["fold"]),
                ]
                if job.get("train_camera"):
                    command += ["--train-camera", job["train_camera"]]
                stream = log_path.open("a")
                try:
                    process = subprocess.Popen(
                        command,
                        env={**environment, "CUDA_VISIBLE_DEVICES": str(gpu)},
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
                except BaseException:
                    stream.close()
                    failed = True
                    raise
                job.update(
                    state="running",
                    gpu=gpu,
                    pid=process.pid,
                    started=time.time(),
                    log=str(log_path),
                )
                active[gpu] = (process, stream, job)
                print(
                    "started",
                    job["experiment"],
                    job["fold"],
                    job.get("train_camera"),
                    "GPU",
                    gpu,
                    flush=True,
                )
            save_status(root, jobs, failed)
            if failed and not active:
                break
            time.sleep(5)
    finally:
        for process, stream, job in active.values():
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stream.close()
            job.update(
                state="failed", finished=time.time(), exit_code=process.returncode
            )
            failed = True
        save_status(root, jobs, failed)
    if failed:
        raise RuntimeError("A worker failed. Read the study logs folder.")
    print("All", len(jobs), "fold jobs are complete.", flush=True)
    return jobs


def interrupted(signum, frame):
    raise KeyboardInterrupt("Study launcher stopped")


def install_signal_handler():
    signal.signal(signal.SIGTERM, interrupted)
