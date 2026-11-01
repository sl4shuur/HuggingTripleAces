"""Evaluate full sample bags; optional paired-device curves reuse the same features."""
import pandas as pd
import torch
from soilcv.curves import TARGET_COLUMNS,validate_curves
from soilcv.metrics import evaluate_samples
from soilcv.nn import evaluation_mode


def predict_bags(model,loader,*,device='cpu',amp=False,cameras=False):
    samples=[];devices=[]
    device=torch.device(device)
    with evaluation_mode(model),torch.inference_mode():
        for bag in loader:
            images=bag['image'].to(device);weights=bag['bag_weights'].to(device)
            with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):
                features=model.encode(images)
                curve=model.pool(features,weights).float().cpu().numpy()[0]
                samples.append({'sample_id':bag['sample_id'],**dict(zip(TARGET_COLUMNS,curve))})
                if cameras:
                    for name in sorted(set(bag['device'])):
                        mask=torch.tensor([d==name for d in bag['device']],device=device)
                        values=model.pool(features[mask],weights[mask]).float().cpu().numpy()[0]
                        devices.append({'sample_id':bag['sample_id'],'device':name,**dict(zip(TARGET_COLUMNS,values))})
    predictions=pd.DataFrame(samples)
    validate_curves(predictions[list(TARGET_COLUMNS)])
    return predictions,pd.DataFrame(devices)


def validate_bags(model,loader,labels,*,device='cpu',amp=False,repair=False):
    predictions,_=predict_bags(model,loader,device=device,amp=amp)
    return evaluate_samples(labels,predictions),predictions
