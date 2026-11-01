"""Resolve ready MIL runs without waiting for the full training queue."""
from pathlib import Path
import importlib.util
from types import SimpleNamespace
import pytest


def runner():
    path=Path(__file__).parents[1]/'inference/runner.py'
    spec=importlib.util.spec_from_file_location('mil_runner',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('reference',[
 'uvi5xre4',
 'public-team/soil-grain-size/uvi5xre4',
 'https://forge.coreweave.com/wandb/public-team/soil-grain-size/runs/uvi5xre4/files?nw=demo',
 'https://wandb.ai/public-team/soil-grain-size/runs/uvi5xre4'])
def test_run_references(reference):
    assert runner().run_path(reference)=='public-team/soil-grain-size/uvi5xre4'


def test_ready_model_is_selected_among_other_artifacts():
    model=SimpleNamespace(type='model',state='COMMITTED')
    run=SimpleNamespace(logged_artifacts=lambda:[SimpleNamespace(type='predictions',state='COMMITTED'),model])
    api=SimpleNamespace(run=lambda path:run)
    artifact,source=runner().resolve_model(api,'uvi5xre4')
    assert artifact is model and source is run


def test_incomplete_model_does_not_block_other_runs():
    run=SimpleNamespace(logged_artifacts=lambda:[SimpleNamespace(type='model',state='PENDING')])
    api=SimpleNamespace(run=lambda path:run)
    assert runner().resolve_model(api,'uvi5xre4')==(None,run)


def test_multiple_models_require_an_exact_version():
    run=SimpleNamespace(logged_artifacts=lambda:[SimpleNamespace(type='model',state='COMMITTED')]*2)
    api=SimpleNamespace(run=lambda path:run)
    with pytest.raises(ValueError,match='several model artifacts'):
        runner().resolve_model(api,'uvi5xre4')
