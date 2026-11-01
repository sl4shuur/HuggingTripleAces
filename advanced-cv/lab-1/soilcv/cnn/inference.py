"""Reload saved MIL models and predict from complete photos and their scale."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from soilcv.bags import make_grid_manifest,cache_grid,SampleBags,one_bag
from soilcv.artifacts import package_identity,write_json
from soilcv.curves import TARGET_COLUMNS,validate_curves
from .model import MILCNN,image_transform,NORMALIZE
from .predict import predict_bags


def load_model(folder,device='cpu'):
    folder=Path(folder);config=json.loads((folder/'config.json').read_text())
    if config['versions']['soilcv']!=package_identity():raise ValueError('Use the source package saved with this model.')
    recipe=config['params']
    if recipe['normalization']!=NORMALIZE or recipe['grid']!='full_grid_v1' or recipe['pooling']!='overlap_area_then_equal_photos':raise ValueError('Unsupported preprocessing.')
    model=MILCNN(recipe,np.array(recipe['train_mean_prior']),pretrained=False).to(device)
    model.load_state_dict(torch.load(folder/'weights.pt',map_location='cpu',weights_only=True))
    return model,recipe


def download_checkpoint(artifact_handle,destination):
    import wandb
    artifact=wandb.Api().artifact(artifact_handle,type='model');run=artifact.logged_by()
    if run is None:raise ValueError('Model artifact has no run config.')
    artifact.download(root=str(destination));write_json(Path(destination)/'config.json',dict(run.config))
    return Path(destination)


def predict_sample(photos,checkpoint_dirs,*,sample_id='query',device='cpu',amp=True):
    if not photos or not checkpoint_dirs:raise ValueError('Provide photos and saved model folders.')
    rows=[{'sample_id':sample_id,'photo':Path(p['path']).name,'path':str(Path(p['path']).resolve()),
           'ppm_est':float(p['ppm_est']),'device':p.get('device','unknown')} for p in photos]
    predictions=[]
    with tempfile.TemporaryDirectory(prefix='soil-mil-inference-') as temp:
        for folder in checkpoint_dirs:
            model,recipe=load_model(folder,device)
            manifest=make_grid_manifest(pd.DataFrame(rows),tile_mm=recipe['tile_mm'],output_px=recipe['pixels'])
            cache_grid(manifest,temp)
            dataset=SampleBags(manifest,temp,image_transform(recipe))
            loader=DataLoader(dataset,batch_size=1,collate_fn=one_bag,num_workers=0)
            curve,_=predict_bags(model,loader,device=device,amp=amp)
            predictions.append(curve[list(TARGET_COLUMNS)].to_numpy())
            del model
    values=np.mean(predictions,axis=0);validate_curves(values)
    result=pd.DataFrame(values,columns=TARGET_COLUMNS);result.insert(0,'sample_id',sample_id)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--photo',nargs='+',required=True,type=Path)
    parser.add_argument('--ppm',nargs='+',required=True,type=float)
    parser.add_argument('--checkpoint-dir',nargs='+',required=True,type=Path)
    parser.add_argument('--device',default='cpu');parser.add_argument('--sample-id',default='query')
    parser.add_argument('--output',type=Path,default=Path('prediction.csv'))
    args=parser.parse_args()
    if len(args.photo)!=len(args.ppm):parser.error('Supply one PPM per photo.')
    result=predict_sample([{'path':p,'ppm_est':ppm} for p,ppm in zip(args.photo,args.ppm)],args.checkpoint_dir,sample_id=args.sample_id,device=args.device)
    args.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(args.output,index=False)
