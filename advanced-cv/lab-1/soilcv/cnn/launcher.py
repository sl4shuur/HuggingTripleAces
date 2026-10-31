"""Run independent fold jobs on two GPUs and wait for all workers."""
import os
import signal
import subprocess
import sys
import time

from .experiments import ROOT, RECIPES
from soilcv.artifacts import write_json


def save_status(jobs, failed):
    temporary = ROOT/'status.tmp.json'
    write_json(temporary, {'jobs':jobs, 'failed':failed, 'updated':time.time()})
    temporary.replace(ROOT/'status.json')


def launch():
    environment = {**os.environ, 'PYTHONUNBUFFERED':'1', 'WANDB_CONSOLE':'off'}
    if not environment.get('WANDB_API_KEY'):
        raise ValueError('Read the WANDB_API_KEY secret in the notebook first.')
    jobs = [{'experiment':recipe['id'], 'fold':fold, 'state':'pending'}
            for recipe in RECIPES for fold in range(4)]
    active = {}
    failed = False
    logs = ROOT/'logs'
    logs.mkdir(exist_ok=True)
    try:
        while any(job['state'] in ['pending','running'] for job in jobs):
            for gpu, (process, stream, job) in list(active.items()):
                code = process.poll()
                if code is not None:
                    stream.close()
                    job.update(state='complete' if code == 0 else 'failed',
                               exit_code=code, finished=time.time())
                    failed = failed or code != 0
                    print(job['state'],job['experiment'],job['fold'],flush=True)
                    del active[gpu]
            for gpu in [0,1]:
                if gpu in active or failed:
                    continue
                job = next((job for job in jobs if job['state']=='pending'),None)
                if job is None:
                    break
                log_path = logs/f"{job['experiment']}-fold-{job['fold']}.log"
                stream = log_path.open('a')
                child_env = {**environment, 'CUDA_VISIBLE_DEVICES':str(gpu)}
                process = subprocess.Popen(
                    [sys.executable,'-m','soilcv.cnn.experiments','train',
                     '--experiment',job['experiment'],'--fold',str(job['fold'])],
                    env=child_env, stdout=stream, stderr=subprocess.STDOUT)
                job.update(state='running',gpu=gpu,pid=process.pid,
                           started=time.time(),log=str(log_path))
                active[gpu]=(process,stream,job)
                print('started',job['experiment'],job['fold'],'GPU',gpu,flush=True)
            save_status(jobs,failed)
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
            job.update(state='failed',finished=time.time(),exit_code=process.returncode)
            failed = True
        save_status(jobs,failed)
    if failed:
        raise RuntimeError('A CNN worker failed. Read part1_cnn/logs for the error.')
    print('All 32 CNN fold jobs are complete.',flush=True)


def interrupted(signum, frame):
    raise KeyboardInterrupt('CNN launcher stopped')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM,interrupted)
    launch()
