"""Full-coverage CNN MIL recipes for Kaggle. All scores are sample-level."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import argparse
import json
import math
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import models

from soilcv.artifacts import file_hash,identity_hash,package_identity,write_json
from soilcv.data import load_metadata,load_folds,DEFAULT_FOLDS_PATH
from soilcv.bags import make_grid_manifest,cache_grid,coverage_table,SampleBags,one_bag
from soilcv.curves import TARGET_COLUMNS
from soilcv.config import ExperimentConfig,seed_everything
from soilcv.nn import EMDLoss,make_param_groups
from soilcv.metrics import evaluate_samples
from soilcv.training import fit_supervised_fold
from soilcv.tracking import start_run,log_artifact
from .model import MILCNN,image_transform,NORMALIZE
from .predict import predict_bags,validate_bags

ROOT=Path(os.environ.get('SOILCV_CNN_OUTPUT_DIR','/kaggle/working/part1_mil'))
CACHE=Path(os.environ.get('SOILCV_CNN_CACHE_DIR','/tmp/soilcv-mil-tiles'))
HYPOTHESES_COMMIT='0af70f4'
BASE=dict(arch='resnet18',frozen=False,freeze_bn=True,augmentation=True,optimizer='adamw',
          backbone_lr=.0001,head_lr=.001,weight_decay=.01,schedule='constant',pixels=288,
          tile_mm=64,grid='full_grid_v1',pooling='overlap_area_then_equal_photos',
          chunk_size=8,accumulation_steps=2,epochs=16,patience=5,deterministic=True)
RECIPES=[dict(BASE,id='mil-r18-frozen-288',hypothesis='H-MIL-01',frozen=True,head_lr=.003),
         dict(BASE,id='mil-r18-288',hypothesis='H-MIL-02'),
         dict(BASE,id='mil-r18-224',hypothesis='H-MIL-03',pixels=224),
         dict(BASE,id='mil-r18-cosine',hypothesis='H-MIL-04',schedule='warmup_cosine'),
         dict(BASE,id='mil-convnext-288',hypothesis='H-MIL-05',arch='convnext_tiny')]


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    data=next(Path('/kaggle/input').rglob('Training_labels_updated.csv')).parent
    labels,photos,template=load_metadata(data)
    folds=load_folds(DEFAULT_FOLDS_PATH,labels.sample_id)
    source_hashes={name:file_hash(data/name) for name in ['Training_labels_updated.csv','ppm_updated.csv','sample_submission.csv']}
    grids={};all_manifests={}
    for pixels in sorted({r['pixels'] for r in RECIPES}):
        manifest=make_grid_manifest(photos,tile_mm=BASE['tile_mm'],output_px=pixels,image_root=data)
        all_manifests[pixels]=manifest
        grids[str(pixels)]=manifest.cache_id.iloc[0]
    columns=['sample_id','photo','tile_id','left','top','crop_px','photo_weight']
    if not all_manifests[224][columns].equals(all_manifests[288][columns]):
        raise ValueError('Resolution runs must use identical native crops and weights.')
    identity={'soilcv':package_identity(),'data':source_hashes,'grids':grids,'folds_hash':file_hash(DEFAULT_FOLDS_PATH),
              'recipes':identity_hash(RECIPES),'hypotheses_commit':HYPOTHESES_COMMIT}
    path=ROOT/'metadata.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous['identity']!=identity:raise ValueError('Code/data changed. Use a fresh output folder.')
        metadata=previous
    else:
        weight_info={}
        for arch,weights in [('resnet18',models.ResNet18_Weights.IMAGENET1K_V1),('convnext_tiny',models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)]:
            weights.get_state_dict(progress=False,check_hash=True)
            weight_path=Path(torch.hub.get_dir())/'checkpoints'/Path(weights.url).name
            weight_info[arch]={'enum':str(weights),'url':weights.url,'sha256':file_hash(weight_path)}
        metadata={'identity':identity,'run_id':time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+identity_hash(identity)[:10],
                  'data_root':str(data),'pretrained':weight_info}
        write_json(path,metadata)
    for pixels,manifest in all_manifests.items():
        cache_grid(manifest,CACHE,image_root=data)
        coverage=coverage_table(manifest)
        coverage.to_csv(ROOT/f'coverage_{pixels}.csv',index=False)
        manifest=manifest.merge(folds,on='sample_id',how='left',validate='many_to_one')
        manifest.to_json(ROOT/f'tiles_{pixels}.json',orient='records',indent=2)
    labels.to_json(ROOT/'labels.json',orient='records',indent=2)
    photos.to_json(ROOT/'photos.json',orient='records',indent=2)
    template.to_json(ROOT/'template.json',orient='records',indent=2)
    folds.to_csv(ROOT/'sample_folds.csv',index=False)
    write_json(ROOT/'recipes.json',RECIPES)
    print('Run:',metadata['run_id'],'| photos:',len(photos),'| tiles per resolution:',len(all_manifests[288]),flush=True)
    return metadata


def make_loader(manifest,recipe,labels=None,train=False,seed=242):
    dataset=SampleBags(manifest,CACHE,image_transform(recipe,train=train),labels)
    return DataLoader(dataset,batch_size=1,shuffle=train,num_workers=0,collate_fn=one_bag,
                      generator=torch.Generator().manual_seed(seed))


def verify_completed(folder,recipe_hash):
    result=json.loads((folder/'result.json').read_text())
    if result['recipe_hash']!=recipe_hash:raise ValueError('Existing result has other code/settings.')
    hashes=json.loads((folder/'artifact_hashes.json').read_text())
    required=['weights.pt','best.pt','last.pt','oof.csv','test_predictions.csv','config.json','history.json','compute.json','camera_predictions.csv','inference_check.json']
    if set(required)-set(hashes):raise ValueError('Incomplete run manifest.')
    for name,digest in hashes.items():
        if not (folder/name).is_file() or file_hash(folder/name)!=digest:raise ValueError('Missing/changed run artifact: '+name)
    return result


def run_fold(recipe,fold):
    torch.set_num_threads(2);seed_everything(42+fold,deterministic=True)
    metadata=json.loads((ROOT/'metadata.json').read_text())
    if metadata['identity']['soilcv']!=package_identity():raise ValueError('Package changed after prepare.')
    labels=pd.read_json(ROOT/'labels.json')
    manifest=pd.read_json(ROOT/f"tiles_{recipe['pixels']}.json")
    train=manifest[manifest.fold.notna() & manifest.fold.ne(fold)].copy()
    valid=manifest[manifest.fold.eq(fold)].copy();test=manifest[manifest.fold.isna()].copy()
    train_labels=labels[labels.sample_id.isin(train.sample_id)]
    valid_labels=labels[labels.sample_id.isin(valid.sample_id)]
    recipe_hash=identity_hash({'recipe':recipe,'identity':metadata['identity'],'fold':fold})
    folder=ROOT/'runs'/recipe['id']/f'fold_{fold}'
    if (folder/'result.json').exists():return verify_completed(folder,recipe_hash)
    prior=train_labels[list(TARGET_COLUMNS)].to_numpy().mean(axis=0)
    train_loader=make_loader(train,recipe,train_labels,train=True,seed=242+fold)
    valid_loader=make_loader(valid,recipe,valid_labels)
    model=MILCNN(recipe,prior).to('cuda:0')
    groups=make_param_groups(model.head,lr=recipe['head_lr'],weight_decay=recipe['weight_decay'])
    if not recipe['frozen']:groups+=make_param_groups(model.encoder,lr=recipe['backbone_lr'],weight_decay=recipe['weight_decay'])
    optimizer=torch.optim.AdamW(groups)
    scheduler=None
    if recipe['schedule']=='warmup_cosine':
        def factor(epoch):
            if epoch<3:return (epoch+1)/3
            return .05+.95*(1+math.cos(math.pi*min(1,(epoch-3)/max(1,recipe['epochs']-4))))/2
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    config=ExperimentConfig(recipe['id'],stage='cnn_mil',model=recipe['arch'],fold=fold,seed=42+fold,
        hypothesis_id=recipe['hypothesis'],output_root=str(ROOT/'runs'),
        versions={'soilcv':metadata['identity']['soilcv'],'run_id':metadata['run_id'],'identity':metadata['identity']},
        params={**recipe,'normalization':NORMALIZE,'train_mean_prior':prior.tolist(),
                'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'loss_units':'percent-log10'})
    with start_run(config) as run:
        oof,history=fit_supervised_fold(model,train_loader,valid_loader,valid_labels,optimizer,EMDLoss(),config,
            epochs=recipe['epochs'],device='cuda:0',amp=True,scheduler=scheduler,patience=recipe['patience'],
            accumulation_steps=recipe['accumulation_steps'],clip_grad=5,freeze_batchnorm=True,
            frozen_modules=(model.encoder,) if recipe['frozen'] else (),run=run,validation_fn=validate_bags)
        _,camera=predict_bags(model,valid_loader,device='cuda:0',amp=True,cameras=True)
        camera['fold']=fold;camera.to_csv(folder/'camera_predictions.csv',index=False)
        predictions,_=predict_bags(model,make_loader(test,recipe),device='cuda:0',amp=True)
        predictions.to_csv(folder/'test_predictions.csv',index=False)
        torch.save(model.state_dict(),folder/'weights.pt')
        # Reload through the public inference path, on one test sample, using AMP in both paths.
        from .inference import predict_sample
        sample=predictions.sample_id.iloc[0]
        rows=test[test.sample_id.eq(sample)].drop_duplicates('photo')
        supplied=[{'path':Path(metadata['data_root'])/r.image_path,'ppm_est':r.ppm_est,'device':r.device} for r in rows.itertuples()]
        check=predict_sample(supplied,[folder],sample_id=sample,device='cuda:0',amp=True)
        difference=float(np.abs(check[list(TARGET_COLUMNS)].to_numpy()-predictions.loc[predictions.sample_id.eq(sample),list(TARGET_COLUMNS)].to_numpy()).max())
        if difference>1e-3:raise ValueError('Reloaded inference differs from saved test curves.')
        write_json(folder/'inference_check.json',{'sample_id':sample,'max_absolute_difference':difference,'tolerance':1e-3,'amp':True})
        artifact=log_artifact(run,folder/'weights.pt',name=f"{recipe['id']}-fold-{fold}",artifact_type='model',metadata={'run_id':metadata['run_id'],'recipe_hash':recipe_hash})
        artifact.wait()
        log_artifact(run,folder/'oof.csv',name=f"{recipe['id']}-fold-{fold}-oof",artifact_type='predictions')
        result={'experiment':recipe['id'],'fold':fold,'run_id':metadata['run_id'],'recipe_hash':recipe_hash,
                **evaluate_samples(valid_labels,oof),**json.loads((folder/'compute.json').read_text()),
                'epochs_ran':len(history),'best_epoch':min(history,key=lambda r:r['val']['emd'])['epoch'],
                'run_url':run.url,'checkpoint_artifact':artifact.qualified_name}
    names=['weights.pt','best.pt','last.pt','oof.csv','test_predictions.csv','config.json','history.json','compute.json','camera_predictions.csv','inference_check.json']
    write_json(folder/'artifact_hashes.json',{name:file_hash(folder/name) for name in names})
    write_json(folder/'result.json',result)
    print('Complete:',recipe['id'],fold,'EMD:',result['emd'],flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','train'])
    parser.add_argument('--experiment',default='mil-r18-288');parser.add_argument('--fold',type=int,default=0)
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    else:run_fold(next(dict(r) for r in RECIPES if r['id']==args.experiment),args.fold)
