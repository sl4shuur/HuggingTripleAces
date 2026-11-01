"""Small CPU-only tests: coverage, whole bags, gradients, fit and reload inference."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image
import torch
from torch.utils.data import DataLoader
from soilcv.bags import make_grid_manifest,cache_grid,coverage_table,rectangle_coverage,SampleBags,one_bag
from soilcv.curves import TARGET_COLUMNS,validate_curves
from soilcv.nn import EMDLoss,freeze_bn_stats
from soilcv.config import ExperimentConfig
from soilcv.artifacts import package_identity,write_json
from soilcv.training import fit_supervised_fold
from soilcv.cnn.model import MILCNN,image_transform
from soilcv.cnn.predict import predict_bags,validate_bags
from soilcv.cnn.experiments import BASE
from soilcv.cnn.inference import predict_sample


def photos(tmp_path,sample='sample'):
    rows=[]
    for i in range(2):
        path=tmp_path/f'{sample}-{i}.png'
        pixels=np.random.default_rng(i).integers(0,256,(70,100,3),dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        rows.append({'sample_id':sample,'photo':path.name,'path':str(path),'ppm_est':1.,'device':'Motorola' if i==0 else 'Samsung'})
    return pd.DataFrame(rows)


def test_grid_covers_every_pixel_and_corrects_overlap(tmp_path):
    frame=photos(tmp_path)
    a=make_grid_manifest(frame,tile_mm=32,output_px=24)
    b=make_grid_manifest(frame,tile_mm=32,output_px=48)
    columns=['photo','tile_id','left','top','crop_px','photo_weight']
    pd.testing.assert_frame_equal(a[columns],b[columns])
    for _,rows in a.groupby('photo'):
        count=np.zeros((70,100),dtype=int)
        for r in rows.itertuples():count[r.top:r.top+r.crop_px,r.left:r.left+r.crop_px]+=1
        assert (count>0).all()
        expected=[(1/count[r.top:r.top+r.crop_px,r.left:r.left+r.crop_px]).sum()/7000 for r in rows.itertuples()]
        np.testing.assert_allclose(rows.photo_weight,expected)
    assert coverage_table(a).coverage.eq(1).all()
    cache_grid(a,tmp_path/'cache')
    row=a.iloc[0]
    with Image.open(frame.iloc[0].path) as source:
        direct=source.crop((row.left,row.top,row.left+32,row.top+32)).resize((24,24),Image.Resampling.LANCZOS)
    with Image.open(tmp_path/'cache'/row.tile_path) as cached:np.testing.assert_array_equal(cached,direct)
    Image.new('RGB',(24,24),'red').save(tmp_path/'cache'/row.tile_path)
    with pytest.raises(ValueError,match='checksum'):cache_grid(a,tmp_path/'cache')
    incomplete=np.array([[0,0,32,32]])
    assert rectangle_coverage(incomplete,100,70)[0]<1


def test_full_bag_fit_gradients_and_saved_inference(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(42)
    recipe=dict(BASE,pixels=32,tile_mm=32,chunk_size=2,augmentation=False,epochs=1)
    frame=photos(tmp_path,'train');other=photos(tmp_path,'valid')
    all_photos=pd.concat([frame,other],ignore_index=True)
    manifest=make_grid_manifest(all_photos,tile_mm=32,output_px=32)
    cache=tmp_path/'cache';cache_grid(manifest,cache)
    manifest['fold']=manifest.sample_id.map({'train':1,'valid':0})
    prior=np.linspace(1,100,11,dtype=np.float32)
    labels=pd.DataFrame([['train',*prior],['valid',*prior]],columns=['sample_id',*TARGET_COLUMNS])
    train=SampleBags(manifest[manifest.fold.eq(1)],cache,image_transform(recipe),labels)
    valid=SampleBags(manifest[manifest.fold.eq(0)],cache,image_transform(recipe),labels)
    item=train[0]
    assert len(item['image'])==len(manifest[manifest.sample_id.eq('train')])
    assert torch.isclose(item['bag_weights'].sum(),torch.tensor(1.))
    for name in set(item['photo']):
        assert float(item['bag_weights'][torch.tensor([p==name for p in item['photo']])].sum())==pytest.approx(.5)
    model=MILCNN(recipe,prior,pretrained=False);model.train();freeze_bn_stats(model)
    loss=EMDLoss()(model(item['image'],item['bag_weights']),item['target'])
    loss.backward()
    assert model.encoder.conv1.weight.grad.abs().sum()>0
    assert model.head.projection.weight.grad.abs().sum()>0
    model.zero_grad()
    config=ExperimentConfig('tiny-mil',stage='cnn_mil',model='resnet18',fold=0,output_root=str(tmp_path/'runs'),
             versions={'soilcv':package_identity()},params={**recipe,'normalization':{'mean':[.485,.456,.406],'std':[.229,.224,.225]},'train_mean_prior':prior.tolist()})
    loaders=[DataLoader(d,batch_size=1,collate_fn=one_bag) for d in [train,valid]]
    result,_=fit_supervised_fold(model,*loaders,labels[labels.sample_id.eq('valid')],torch.optim.AdamW(model.parameters(),lr=.0001),EMDLoss(),config,
          epochs=1,device='cpu',amp=False,freeze_batchnorm=True,validation_fn=validate_bags)
    validate_curves(result[list(TARGET_COLUMNS)])
    torch.save(model.state_dict(),config.output_dir/'weights.pt')
    expected,camera=predict_bags(model,loaders[1],cameras=True)
    assert len(camera)==2
    reloaded=predict_sample([{'path':r.path,'ppm_est':r.ppm_est,'device':r.device} for r in other.itertuples()],
                            [config.output_dir],sample_id='valid',device='cpu',amp=False)
    np.testing.assert_allclose(reloaded[list(TARGET_COLUMNS)],expected[list(TARGET_COLUMNS)],atol=1e-5)
