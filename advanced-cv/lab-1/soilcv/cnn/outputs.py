"""Save code, environment, and a complete study in Kaggle Outputs."""
import importlib.metadata
import json
from pathlib import Path
import shutil

from .experiments import ROOT
from soilcv.artifacts import file_hash, package_identity, save_environment, write_json


def save_source():
    package = Path(__file__).resolve().parents[1]
    source = ROOT/'source'
    target = source/'soilcv'
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(package,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['pyproject.toml','HYPOTHESES.md','release.json']:
        original = package.parent/name
        if original.is_file():
            shutil.copy2(original,source/name)
    environment = source/'environment'
    environment.mkdir(exist_ok=True)
    for name in ['pyproject.toml','uv.lock']:
        shutil.copy2(package/'cnn/assets'/name,environment/name)
    save_environment(environment/'runtime.json')
    installed = sorted(f'{p.metadata["Name"]}=={p.version}'
                       for p in importlib.metadata.distributions() if p.metadata['Name'])
    (environment/'requirements-installed.txt').write_text('\n'.join(installed)+'\n')
    return source


def export_outputs():
    status = json.loads((ROOT/'status.json').read_text())
    if status['failed'] or len(status['jobs'])!=32 or any(j['state']!='complete' for j in status['jobs']):
        raise RuntimeError('Export needs all 32 successful CV jobs.')
    save_source()
    for job in status['jobs']:
        folder = ROOT/'runs'/job['experiment']/f"fold_{job['fold']}"
        for name in ['best.pt','last.pt','weights.pt','config.json','oof.csv','test_predictions.csv']:
            if not (folder/name).is_file():
                raise FileNotFoundError(folder/name)
    files = [p for p in sorted(ROOT.rglob('*')) if p.is_file()
             and not {'wandb','__pycache__'}.intersection(p.relative_to(ROOT).parts)
             and p.name not in {'outputs.json','output_manifest.json'}]
    write_json(ROOT/'output_manifest.json',{
        str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in files})
    submission = ROOT.parent/'submission.csv'
    shutil.copy2(ROOT/'results/cnn_submission.csv',submission)
    write_json(ROOT/'outputs.json',{'folder':str(ROOT),'files':len(files),
                'total_bytes':sum(p.stat().st_size for p in files),
                'submission':str(submission),'competition_submission_made':False})
    print('Saved',len(files),'study files in',ROOT,flush=True)
    return ROOT
