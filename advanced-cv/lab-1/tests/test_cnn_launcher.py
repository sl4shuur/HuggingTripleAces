"""A failed background job must stop the queue and release its workers."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


def load_launcher(monkeypatch, tmp_path):
    experiments = types.ModuleType('soilcv.cnn.experiments')
    experiments.ROOT = tmp_path
    experiments.RECIPES = [{'id':'first'},{'id':'second'}]
    monkeypatch.setitem(sys.modules, 'soilcv.cnn.experiments', experiments)
    path = Path(__file__).parents[1]/'soilcv/cnn/launcher.py'
    spec = importlib.util.spec_from_file_location('soilcv.cnn.launcher',path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    monkeypatch.setenv('WANDB_API_KEY','test-only')
    monkeypatch.setattr(launcher.time,'sleep',lambda seconds: None)
    return launcher


def test_failed_worker_stops_new_jobs(monkeypatch, tmp_path):
    launcher = load_launcher(monkeypatch,tmp_path)
    workers = []

    class Worker:
        def __init__(self, *args, **kwargs):
            self.pid = len(workers)+1
            workers.append(self)

        def poll(self):
            return 1 if self.pid==1 else 0

    monkeypatch.setattr(launcher.subprocess,'Popen',Worker)
    with pytest.raises(RuntimeError,match='worker failed'):
        launcher.launch()
    status = json.loads((tmp_path/'status.json').read_text())
    assert len(workers)==2
    assert status['failed']
    assert any(job['state']=='pending' for job in status['jobs'])
    assert all(job['state']!='running' for job in status['jobs'])


def test_spawn_error_stops_existing_worker(monkeypatch, tmp_path):
    launcher = load_launcher(monkeypatch,tmp_path)

    class Worker:
        pid = 123
        returncode = None
        terminated = False

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, **kwargs):
            return self.returncode

    worker = Worker()
    calls = []

    def spawn(*args, **kwargs):
        calls.append(args)
        if len(calls)==2:
            raise OSError('Cannot start another worker')
        return worker

    monkeypatch.setattr(launcher.subprocess,'Popen',spawn)
    with pytest.raises(OSError,match='Cannot start'):
        launcher.launch()
    status = json.loads((tmp_path/'status.json').read_text())
    assert worker.terminated
    assert status['failed']
    assert all(job['state']!='running' for job in status['jobs'])
