"""Predict one soil sample from photos and supplied pixels-per-mm values."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .experiments import SoilCNN, image_transform, NORMALIZE
from soilcv.artifacts import file_hash, write_json
from soilcv.data import TileDataset
from soilcv.tiles import make_tile_manifest, build_tile_cache
from soilcv.training import predict_tiles
from soilcv.curves import TARGET_COLUMNS, aggregate_predictions, validate_curves


def download_checkpoint(artifact_handle, destination):
    """Use W&B login/env credentials. Download weights and their original run config."""
    import wandb
    artifact = wandb.Api().artifact(artifact_handle, type='model')
    run = artifact.logged_by()
    if run is None:
        raise ValueError('The model artifact has no source run config.')
    config = dict(run.config)
    source = Path(__file__).with_name('experiments.py')
    if config['versions']['source'] != file_hash(source):
        raise ValueError('Use the soilcv/cnn/experiments.py version saved with this experiment.')
    folder = Path(destination)
    artifact.download(root=str(folder))
    write_json(folder/'config.json', config)
    return folder


def predict_sample(photos, checkpoint_dirs, *, sample_id='query', device='cpu'):
    """photos: [{'path': ..., 'ppm_est': ...}]. Each directory has weights.pt and config.json."""
    if not photos or not checkpoint_dirs:
        raise ValueError('Provide photos and at least one model checkpoint.')
    rows = [{'sample_id':sample_id, 'photo':Path(row['path']).name, 'path':str(Path(row['path']).resolve()),
             'ppm_est':float(row['ppm_est']), 'ppm_method':'user_supplied',
             'absolute_scale_verified':bool(row.get('absolute_scale_verified',False))} for row in photos]
    frame = pd.DataFrame(rows)
    curves = []
    with tempfile.TemporaryDirectory(prefix='soil-cnn-inference-') as temp:
        manifest = make_tile_manifest(frame, tile_mm=64, output_px=288)
        build_tile_cache(manifest, temp)
        for directory in checkpoint_dirs:
            directory = Path(directory)
            config = json.loads((directory/'config.json').read_text())
            recipe = config['params']
            if config['versions']['source'] != file_hash(Path(__file__).with_name('experiments.py')):
                raise ValueError('Checkpoint config and CNN source version differ.')
            if recipe['normalization'] != NORMALIZE:
                raise ValueError('Checkpoint uses different image normalization.')
            # The saved state overwrites this initialization. No ImageNet download is needed.
            prior = np.array(recipe.get('train_mean_prior', np.arange(1,12)/11*100))
            model = SoilCNN(recipe, prior, pretrained=False).to(device)
            model.load_state_dict(torch.load(directory/'weights.pt', map_location='cpu', weights_only=True))
            loader = DataLoader(TileDataset(manifest,temp,transform=image_transform(recipe)),batch_size=16,shuffle=False)
            prediction = aggregate_predictions(predict_tiles(model,loader,device=device))
            curves.append(prediction[list(TARGET_COLUMNS)].to_numpy())
            del model
    values = np.mean(curves, axis=0)
    validate_curves(values)
    result = pd.DataFrame(values,columns=TARGET_COLUMNS)
    result.insert(0,'sample_id',sample_id)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--photo',nargs='+',required=True,type=Path)
    parser.add_argument('--ppm',nargs='+',required=True,type=float)
    parser.add_argument('--checkpoint-dir',nargs='+',required=True,type=Path)
    parser.add_argument('--sample-id',default='query')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',type=Path,default=Path('cnn_prediction.csv'))
    args=parser.parse_args()
    if len(args.photo)!=len(args.ppm):parser.error('Give one PPM value for each photo.')
    result=predict_sample([{'path':path,'ppm_est':ppm} for path,ppm in zip(args.photo,args.ppm)],
                          args.checkpoint_dir,sample_id=args.sample_id,device=args.device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print(result.to_string(index=False))
