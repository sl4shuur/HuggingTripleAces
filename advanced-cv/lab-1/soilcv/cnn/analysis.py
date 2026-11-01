"""Generate current-run MIL results. Never mix numbers from different runs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
from matplotlib.patches import Rectangle
from soilcv.artifacts import write_json
from soilcv.curves import TARGET_COLUMNS,align_samples,make_submission
from soilcv.metrics import summarize_cv,emd_per_sample
from .experiments import ROOT,RECIPES,verify_completed


def collect_results():
    metadata=json.loads((ROOT/'metadata.json').read_text());run_id=metadata['run_id']
    status=json.loads((ROOT/'status.json').read_text())
    if status['failed'] or len(status['jobs'])!=4*len(RECIPES) or any(j['state']!='complete' for j in status['jobs']):
        raise RuntimeError('Wait for every MIL fold to finish.')
    labels=pd.read_json(ROOT/'labels.json');template=pd.read_json(ROOT/'template.json')
    folds=pd.read_csv(ROOT/'sample_folds.csv');output=ROOT/'results';output.mkdir(exist_ok=True)
    tables=[];checkpoints=[];predictions={};cameras=[]
    for recipe in RECIPES:
        name=recipe['id'];oof=[];test=[];results=[]
        for fold in range(4):
            folder=ROOT/'runs'/name/f'fold_{fold}'
            result=json.loads((folder/'result.json').read_text())
            if result['run_id']!=run_id:raise ValueError('Mixed run IDs.')
            verify_completed(folder,result['recipe_hash'])
            results.append(result);oof.append(pd.read_csv(folder/'oof.csv'))
            test.append(pd.read_csv(folder/'test_predictions.csv'))
            camera=pd.read_csv(folder/'camera_predictions.csv');camera['experiment']=name;cameras.append(camera)
            checkpoints.append({'experiment':name,'fold':fold,'run_id':run_id,'wandb_run':result['run_url'],'artifact':result['checkpoint_artifact']})
        oof=pd.concat(oof,ignore_index=True);predictions[name]=oof
        scores=summarize_cv(labels,oof,folds)
        oof.to_csv(output/f'{name}_oof.csv',index=False)
        tables.append({'experiment':name,'run_id':run_id,'arch':recipe['arch'],'pixels':recipe['pixels'],
                       'frozen':recipe['frozen'],'schedule':recipe['schedule'],'cv_emd':scores['emd'],'fold_std':scores['fold_std'],
                       'gpu_hours':sum(r['gpu_hours'] for r in results),
                       'peak_gpu_mb':max(r['peak_gpu_memory_mb'] for r in results),
                       'epochs_per_fold':','.join(str(r['epochs_ran']) for r in results),
                       'public_lb':'not submitted','private_lb':'not available'})
        submit=template.copy()
        submit[list(TARGET_COLUMNS)]=np.mean([align_samples(template,p)[list(TARGET_COLUMNS)].to_numpy() for p in test],axis=0)
        make_submission(submit,template,output/f'{name}_submission.csv')
    table=pd.DataFrame(tables).sort_values('cv_emd').reset_index(drop=True)
    table.to_csv(output/'cnn_results.csv',index=False)
    pd.DataFrame(checkpoints).to_csv(output/'cnn_checkpoints.csv',index=False)
    camera=pd.concat(cameras,ignore_index=True);camera.to_csv(output/'camera_oof.csv',index=False)
    camera_summary,camera_pairs=compare_cameras(labels,camera)
    camera_summary.to_csv(output/'camera_summary.csv',index=False);camera_pairs.to_csv(output/'camera_pairs.csv',index=False)
    y=labels[list(TARGET_COLUMNS)].to_numpy();baseline='mil-r18-288'
    baseline_errors=emd_per_sample(y,align_samples(labels,predictions[baseline])[list(TARGET_COLUMNS)].to_numpy())
    pairs=[]
    for name,p in predictions.items():
        errors=emd_per_sample(y,align_samples(labels,p)[list(TARGET_COLUMNS)].to_numpy())
        for sample,a,b in zip(labels.sample_id,baseline_errors,errors):
            pairs.append({'experiment':name,'sample_id':sample,'reference':baseline,'reference_emd':a,'emd':b,'delta_emd':b-a})
    pd.DataFrame(pairs).to_csv(output/'paired_sample_errors.csv',index=False)
    best=table.iloc[0]
    make_submission(pd.read_csv(output/f'{best.experiment}_submission.csv'),template,output/'cnn_submission.csv')
    write_json(output/'selection.json',{'run_id':run_id,'selected_experiment':best.experiment,'cv_emd':float(best.cv_emd),
          'selection_rule':'lowest sample-mean OOF EMD','test_prediction':'mean of four fold curves','competition_submission_made':False})
    write_report(table,camera_summary,metadata)
    return table,predictions,camera_summary


def compare_cameras(labels,camera):
    pairs=[]
    targets=labels.set_index('sample_id')[list(TARGET_COLUMNS)]
    for (name,sample),group in camera.groupby(['experiment','sample_id']):
        motorola=group[group.device.str.contains('motorola',case=False,na=False)]
        samsung=group[group.device.str.contains('samsung',case=False,na=False)]
        if len(motorola)!=1 or len(samsung)!=1:continue
        a=motorola[list(TARGET_COLUMNS)].to_numpy();b=samsung[list(TARGET_COLUMNS)].to_numpy()
        y=targets.loc[[sample]].to_numpy()
        pairs.append({'experiment':name,'sample_id':sample,'fold':int(group.fold.iloc[0]),
                      'motorola_emd':float(emd_per_sample(y,a)[0]),'samsung_emd':float(emd_per_sample(y,b)[0]),
                      'camera_disagreement':float(emd_per_sample(a,b)[0])})
    frame=pd.DataFrame(pairs,columns=['experiment','sample_id','fold','motorola_emd','samsung_emd','camera_disagreement'])
    summary=frame.groupby('experiment').agg(paired_samples=('sample_id','size'),motorola_emd=('motorola_emd','mean'),
                samsung_emd=('samsung_emd','mean'),camera_disagreement=('camera_disagreement','mean')).reset_index()
    return summary,frame


def plot_scores(table):
    fig,ax=plt.subplots(figsize=(9,3.6));order=table.sort_values('cv_emd',ascending=False)
    ax.barh(order.experiment,order.cv_emd,xerr=order.fold_std,capsize=3)
    ax.set(xlabel='Sample EMD (lower is better)',title='MIL: four fixed folds; error bars are fold std')
    fig.tight_layout();return fig


def plot_learning():
    fig,axes=plt.subplots(1,2,figsize=(11,3.5))
    for name,ax in zip(['mil-r18-288','mil-convnext-288'],axes):
        for fold in range(4):
            h=json.loads((ROOT/'runs'/name/f'fold_{fold}'/'history.json').read_text())
            ax.plot([r['epoch']+1 for r in h],[r['val']['emd'] for r in h],label=f'fold {fold}')
        ax.set(title=name,xlabel='Epoch',ylabel='Validation sample EMD');ax.legend()
    fig.tight_layout();return fig


def plot_coverage():
    metadata=json.loads((ROOT/'metadata.json').read_text())
    manifest=pd.read_json(ROOT/'tiles_288.json');photo=manifest.photo.iloc[0]
    rows=manifest[manifest.photo.eq(photo)];first=rows.iloc[0]
    with Image.open(Path(metadata['data_root'])/first.image_path) as raw:image=ImageOps.exif_transpose(raw).convert('RGB')
    fig,ax=plt.subplots(figsize=(10,5));ax.imshow(image)
    for row in rows.itertuples():
        ax.add_patch(Rectangle((row.left,row.top),row.crop_px,row.crop_px,fill=False,edgecolor='lime',linewidth=1))
    ax.set_title(f'{photo}: all native pixels covered; {len(rows)} tiles');ax.axis('off')
    fig.tight_layout();return fig


def write_report(table,cameras,metadata):
    best=table.iloc[0]
    text=['# CNN MIL results','',f"Run ID: `{metadata['run_id']}`.",
          f"Code hash: `{metadata['identity']['soilcv']['source_hash']}`.",
          f"Hypotheses commit: `{metadata['identity']['hypotheses_commit']}`.",
          'Exact data and grid hashes are in metadata.json. Numbers below come from results/cnn_results.csv.','',
          f'Best OOF EMD: {best.cv_emd:.3f}. Fold standard deviation: {best.fold_std:.3f}. Recipe: {best.experiment}.','',
          '| Recipe | EMD | Fold std | Fit GPU-hours | Peak MB |','| --- | ---: | ---: | ---: | ---: |']
    for r in table.itertuples():text.append(f'| {r.experiment} | {r.cv_emd:.3f} | {r.fold_std:.3f} | {r.gpu_hours:.3f} | {r.peak_gpu_mb:.0f} |')
    score=table.set_index('experiment').cv_emd
    text+=['','## What happened','',
       f"Frozen encoder: {score['mil-r18-frozen-288']:.3f}; tuned ResNet: {score['mil-r18-288']:.3f}.",
       f"Direct 224 px: {score['mil-r18-224']:.3f}; direct 288 px: {score['mil-r18-288']:.3f}.",
       f"Constant LR: {score['mil-r18-288']:.3f}; warmup/cosine: {score['mil-r18-cosine']:.3f}.",
       f"ConvNeXt: {score['mil-convnext-288']:.3f}. More capacity does not promise a better score.",
       'See paired_sample_errors.csv for changes on each of the same 24 samples. Small score gaps need care.','',
       'All photos have 100% rectangle coverage at both resolutions. This means no source area is skipped.',
       'Resizing still loses fine detail. PPM is still based on the no-crop assumption.',
       'All tiles are in each bag. Area weights avoid counting overlapping edges too often. Each photo has equal sample weight.',
       'The head receives pooled features. One sample curve receives one EMD loss.','',
       '## Camera check','',
       '| Recipe | Paired samples | Motorola EMD | Samsung EMD | Disagreement |',
       '| --- | ---: | ---: | ---: | ---: |']
    for r in cameras.itertuples():text.append(f'| {r.experiment} | {r.paired_samples} | {r.motorola_emd:.3f} | {r.samsung_emd:.3f} | {r.camera_disagreement:.3f} |')
    text+=['','Camera errors use the same paired held-out samples. Disagreement is EMD between device curves, not label error.',
           'Camera and view can both change. This check does not measure transfer to iPhone.','',
           '## Limits','',
           'The old tile-loss study is historical. Coverage, pooling, update count and budget also changed. This is not a loss-only ablation.',
           'We still select the best epoch on six validation samples. This can make CV optimistic. A TODO in code keeps the later robustness work.',
           'There is no new split search or seed search. Fold std is not a confidence interval.',
           'Each fold saves a reload test on one test sample. It must reproduce its saved AMP curve within 0.001 percentage points.',
           'The submission averages four fold models. OOF evaluates one held-out model per sample.',
           'Public LB: not submitted. Private LB: not available. No competition submission is sent.',
           'AI helped write this code. Read it before the defense.']
    path=ROOT/'CNN_RESULTS.md';path.write_text('\n'.join(text)+'\n');return path
