"""MIL-only inference: one CSV for each ready W&B run or model artifact."""
import gc
import json
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import urlparse

DEFAULT_PROJECT='public-team/soil-grain-size'


def run_path(reference):
    """Accept a run ID, entity/project/run, or a W&B/Forge run URL."""
    if '://' in reference:
        parts=urlparse(reference).path.strip('/').split('/')
        if 'runs' not in parts:raise ValueError('Use a W&B run URL.')
        index=parts.index('runs')
        if index<2 or index+1>=len(parts):raise ValueError('Incomplete run URL.')
        return '/'.join([parts[index-2],parts[index-1],parts[index+1]])
    if '/' not in reference:return DEFAULT_PROJECT+'/'+reference
    if len(reference.split('/'))!=3:raise ValueError('Use entity/project/run_id or an exact model artifact.')
    return reference


def resolve_model(api,reference):
    if ':' in reference and '://' not in reference:
        artifact=api.artifact(reference,type='model');run=artifact.logged_by()
        if run is None:raise ValueError('Model artifact has no source run.')
        return artifact,run
    run=api.run(run_path(reference))
    models=[a for a in run.logged_artifacts() if a.type=='model' and a.state=='COMMITTED']
    if not models:return None,run
    if len(models)!=1:raise ValueError('This run has several model artifacts. Use an exact artifact version.')
    return models[0],run


def run_checkpoints(checkpoints,code_root,output_dir,*,device='auto'):
    import torch
    import wandb
    sys.path.insert(0,str(code_root))
    from torch.utils.data import DataLoader
    from soilcv.bags import make_grid_manifest,cache_grid,SampleBags,one_bag
    from soilcv.cnn.inference import load_model
    from soilcv.cnn.model import image_transform
    from soilcv.cnn.predict import predict_bags
    from soilcv.data import load_metadata
    from soilcv.curves import make_submission

    selected=[(i,r.strip()) for i,r in enumerate(checkpoints,1) if isinstance(r,str) and r.strip()]
    if not selected:raise ValueError('Fill at least one MIL run or artifact slot.')
    torch.set_num_threads(2)
    device=('cuda:0' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    data=next(Path('/kaggle/input').rglob('sample_submission.csv')).parent
    _,photos,template=load_metadata(data)
    photos=photos[photos.split.eq('test')].copy()
    api=wandb.Api();results=[];skipped=[]
    with tempfile.TemporaryDirectory(prefix='soil-mil-inference-') as temporary:
        temporary=Path(temporary)
        for index,reference in selected:
            print(f'[{index}] Resolve {reference}',flush=True)
            artifact,run=resolve_model(api,reference)
            if artifact is None:
                skipped.append({'slot':index,'run':run.url,'state':run.state,'reason':'No committed model artifact yet'})
                print('Skip: checkpoint is not ready.',flush=True)
                continue
            config=dict(run.config)
            if config.get('stage')!='cnn_mil':raise ValueError('Only MIL checkpoints are supported.')
            folder=temporary/f'model_{index:02d}'
            artifact.download(root=str(folder))
            (folder/'config.json').write_text(json.dumps(config,indent=2))
            model,recipe=load_model(folder,device)
            manifest=make_grid_manifest(photos,tile_mm=recipe['tile_mm'],output_px=recipe['pixels'])
            cache=temporary/'tiles';cache_grid(manifest,cache)
            dataset=SampleBags(manifest,cache,image_transform(recipe))
            loader=DataLoader(dataset,batch_size=1,collate_fn=one_bag,num_workers=0)
            predictions,_=predict_bags(model,loader,device=device,amp=True)
            name=re.sub(r'[^a-zA-Z0-9_-]','_',artifact.qualified_name.split('/')[-1])
            output=output_dir/f'submission_{index:02d}_{name}.csv'
            pending=output.with_suffix('.tmp.csv')
            frame=make_submission(predictions,template,pending);pending.replace(output)
            results.append({'slot':index,'artifact':artifact.qualified_name,'source_run':run.url,
                            'output':str(output),'device':device,'samples':len(frame)})
            (output_dir.parent/'inference_manifest.json').write_text(json.dumps({'predictions':results,'skipped':skipped},indent=2)+'\n')
            print('Saved:',output.name,frame.shape,flush=True)
            del model,loader,dataset
            gc.collect()
            if device.startswith('cuda'):torch.cuda.empty_cache()
    (output_dir.parent/'inference_manifest.json').write_text(json.dumps({'predictions':results,'skipped':skipped},indent=2)+'\n')
    if not results:raise RuntimeError('No MIL checkpoint is ready yet.')
    return results
