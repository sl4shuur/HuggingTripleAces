"""Build the Part 1 report from complete, saved sample-level predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from .experiments import ROOT, CACHE, RECIPES
from soilcv.curves import TARGET_COLUMNS, SIEVE_MM, align_samples, make_submission
from soilcv.metrics import summarize_cv, group_errors, emd_per_sample
from soilcv.artifacts import write_json, file_hash


def read_data():
    labels = pd.DataFrame(json.loads((ROOT/'labels.json').read_text()))
    folds = pd.read_csv(ROOT/'sample_folds.csv')
    return labels, folds


def collect_results():
    labels, folds = read_data()
    status = json.loads((ROOT/'status.json').read_text())
    if status['failed'] or any(job['state'] != 'complete' for job in status['jobs']):
        raise RuntimeError('Wait for all CNN jobs to complete before writing the final table.')
    output = ROOT/'results'
    output.mkdir(exist_ok=True)
    tables, groups, checkpoints = [], [], []
    oof_predictions = {}
    for recipe in RECIPES:
        name = recipe['id']
        fold_predictions, fold_results, test_predictions = [], [], []
        for fold in range(4):
            folder = ROOT/'runs'/name/f'fold_{fold}'
            fold_predictions.append(pd.read_csv(folder/'oof.csv'))
            fold_results.append(json.loads((folder/'result.json').read_text()))
            test_predictions.append(pd.read_csv(folder/'test_predictions.csv'))
            config = json.loads((folder/'config.json').read_text())
            if config['versions']['source'] != file_hash(Path(__file__).with_name('experiments.py')):
                raise ValueError('A result uses a different CNN source file.')
            checkpoints.append({'experiment':name,'fold':fold,'wandb_run':fold_results[-1]['run_url'],
                                'artifact':fold_results[-1]['checkpoint_artifact']})
        predictions = pd.concat(fold_predictions,ignore_index=True)
        scores = summarize_cv(labels,predictions,folds)
        oof_predictions[name] = predictions
        predictions.to_csv(output/f'{name}_oof.csv',index=False)
        jobs = [job for job in status['jobs'] if job['experiment']==name]
        tables.append({'experiment':name, 'arch':recipe['arch'],'optimizer':recipe['optimizer'],
                       'frozen_encoder':recipe['frozen'],'frozen_bn':recipe['freeze_bn'],
                       'augmentation':recipe['augmentation'],'schedule':recipe['schedule'],'pixels':recipe['pixels'],
                       'cv_emd':scores['emd'],'fold_std':scores['fold_std'],
                       'gpu_hours':sum(job['finished']-job['started'] for job in jobs)/3600,
                       'train_val_gpu_hours':sum(r['gpu_hours'] for r in fold_results),
                       'peak_gpu_mb':max(r['peak_gpu_memory_mb'] for r in fold_results),
                       'epochs_per_fold':','.join(str(r['epochs_ran']) for r in fold_results),
                       'public_lb':'not submitted','private_lb':'not available'})
        errors = group_errors(labels,predictions)
        errors.insert(0,'experiment',name)
        groups.append(errors)
        template = pd.DataFrame(json.loads((ROOT/'template.json').read_text()))
        aligned = [align_samples(template,pred)[list(TARGET_COLUMNS)].to_numpy() for pred in test_predictions]
        test_mean = template.copy()
        test_mean[list(TARGET_COLUMNS)] = np.mean(aligned,axis=0)
        make_submission(test_mean,template,output/f'{name}_submission.csv')
    # Same split and scoring for the two image-free references.
    y=labels[list(TARGET_COLUMNS)].to_numpy()
    fold_ids=folds.set_index('sample_id').loc[labels.sample_id,'fold'].to_numpy()
    reference_rows=[]
    for name in ['train_mean','train_median']:
        pred=np.zeros_like(y)
        for fold in range(4):
            train=y[fold_ids != fold]
            pred[fold_ids == fold]=train.mean(axis=0) if name=='train_mean' else np.median(train,axis=0)
        frame=labels[['sample_id']].copy(); frame[list(TARGET_COLUMNS)]=pred;frame['fold']=fold_ids
        scores=summarize_cv(labels,frame,folds)
        reference_rows.append({'baseline':name,'cv_emd':scores['emd'],'fold_std':scores['fold_std']})
    table=pd.DataFrame(tables).sort_values('cv_emd').reset_index(drop=True)
    table.to_csv(output/'cnn_results.csv',index=False)
    pd.concat(groups,ignore_index=True).to_csv(output/'cnn_group_errors.csv',index=False)
    pd.DataFrame(checkpoints).to_csv(output/'cnn_checkpoints.csv',index=False)
    pd.DataFrame(reference_rows).to_csv(output/'image_free_baselines.csv',index=False)
    best=table.iloc[0].experiment
    make_submission(pd.read_csv(output/f'{best}_submission.csv'),template,output/'cnn_submission.csv')
    write_json(output/'selection.json',{'selected_experiment':best,'selection_rule':'lowest sample-level four-fold OOF EMD',
                'cv_emd':float(table.iloc[0].cv_emd),'test_prediction':'mean of four fold models',
                'note':'OOF evaluates individual held-out fold models, not this four-model test ensemble.',
                'leaderboard_submission_made':False})
    return table, pd.DataFrame(reference_rows), oof_predictions


def plot_scores(table, baselines):
    fig,ax=plt.subplots(figsize=(10,4))
    order=table.sort_values('cv_emd',ascending=False)
    ax.barh(order.experiment,order.cv_emd,xerr=order.fold_std,color='#39789c',capsize=3)
    for row in baselines.itertuples(index=False):
        ax.axvline(row.cv_emd,linestyle='--',label=row.baseline)
    ax.set(xlabel='Sample-mean EMD (lower is better)',title='Four fixed sample folds; bars show fold standard deviation')
    ax.legend()
    fig.tight_layout()
    return fig


def plot_learning():
    names=['cnn-r18-frozen','cnn-r18-adamw','cnn-convnext-tiny']
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for ax,name in zip(axes,names):
        for fold in range(4):
            history=json.loads((ROOT/'runs'/name/f'fold_{fold}'/'history.json').read_text())
            ax.plot([row['epoch']+1 for row in history],[row['val']['emd'] for row in history],label=f'fold {fold}')
        ax.set(title=name,xlabel='Epoch',ylabel='Validation sample EMD')
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_failures(best_name, predictions):
    labels,_=read_data()
    pred=align_samples(labels,predictions)[list(TARGET_COLUMNS)].to_numpy()
    y=labels[list(TARGET_COLUMNS)].to_numpy()
    error=emd_per_sample(y,pred)
    worst=np.argsort(error)[-4:][::-1]
    manifest=pd.DataFrame(json.loads((ROOT/'tiles.json').read_text()))
    fig,axes=plt.subplots(4,2,figsize=(10,9))
    for pair,index in zip(axes,worst):
        sample=labels.loc[index,'sample_id']
        path=manifest.query('sample_id == @sample and tile_id == 1').iloc[0].tile_path
        with Image.open(CACHE/path) as image:pair[0].imshow(image)
        pair[0].set_title(sample);pair[0].axis('off')
        pair[1].semilogx(SIEVE_MM,y[index],marker='o',label='true')
        pair[1].semilogx(SIEVE_MM,pred[index],marker='o',label='OOF CNN')
        pair[1].set(title=f'EMD {error[index]:.2f}',xlabel='Size (mm)',ylabel='Passing (%)',ylim=(-2,103))
        pair[1].legend(fontsize=8)
    fig.suptitle(best_name+' | four largest OOF errors')
    fig.tight_layout()
    return fig


def hypothesis_outcomes(table):
    score=table.set_index('experiment').cv_emd
    checks=[('H-CNN-01','Frozen image model vs median baseline',score['cnn-r18-frozen']<85.213),
            ('H-CNN-02','Fine-tuning vs frozen encoder',score['cnn-r18-adamw']<score['cnn-r18-frozen']),
            ('H-CNN-03','AdamW recipe vs SGD recipe',score['cnn-r18-adamw']<score['cnn-r18-sgd']),
            ('H-CNN-04','Frozen BN vs updated BN',score['cnn-r18-adamw']<score['cnn-r18-train-bn']),
            ('H-CNN-05','Light augmentation vs none',score['cnn-r18-adamw']<score['cnn-r18-no-aug']),
            ('H-CNN-06','Warmup/cosine recipe vs constant LR',score['cnn-r18-adamw']<score['cnn-r18-constant-lr']),
            ('H-CNN-07','288 px vs 224 px',score['cnn-r18-adamw']<score['cnn-r18-224']),
            ('H-CNN-08','ConvNeXt vs tuned ResNet',score['cnn-convnext-tiny']<score['cnn-r18-adamw'])]
    return pd.DataFrame([{'hypothesis':h,'comparison':label,'CV_order_matches_prediction':bool(ok)} for h,label,ok in checks])


def write_report(table, baselines):
    output=ROOT/'results'
    best=table.iloc[0]
    reference=baselines.set_index('baseline').loc['train_median','cv_emd']
    lines=['# Part 1: CNNs','',
           f"Best fixed-fold OOF score: {best.cv_emd:.3f} +/- {best.fold_std:.3f} EMD with `{best.experiment}`.",
           f"The image-free median baseline is {reference:.3f}. Lower is better.",'',
           'We use 24 soil samples and the original four group folds. No sample is in both train and validation.',
           'Each T4 runs a separate fold. Input tiles are 64 mm using estimated PPM. Three views per photo are cached at 288 px.',
           'Validation means tiles per photo, then photos per sample. Training uses equal expected sample/photo/tile weight.','',
           'Hypotheses were committed before training: `804c059`. The smoke test is separate from the CV table.','',
           '| Experiment | CV EMD | Fold std | GPU-hours (job slots) | Peak GPU MB | Epochs per fold |',
           '| --- | ---: | ---: | ---: | ---: | --- |']
    for row in table.itertuples(index=False):
        lines.append(f'| {row.experiment} | {row.cv_emd:.3f} | {row.fold_std:.3f} | {row.gpu_hours:.3f} | {row.peak_gpu_mb:.0f} | {row.epochs_per_fold} |')
    lines += ['', 'GPU-hours above include each worker slot from process start to finish, including checkpoint/log work.',
              '`train_val_gpu_hours` in the CSV measures the shared fit function only. Peak memory is per selected GPU, not the sum of two cards.','',
              '## What the comparisons mean','',
              'CNN filters reuse local texture patterns. This can help with a small dataset. A local tile can miss large stones or the full mix.',
              'The surface image does not directly measure the full mass distribution. Fine grains can be smaller than a pixel.',
              'A larger model can overfit. Pretrained BatchNorm statistics may help when tile batches contain related views.','',
              'AdamW and SGD use different learning rates and weight decay. That comparison tests two optimizer recipes.',
              'The schedule comparison changes both warmup and cosine decay. It does not isolate warmup alone.',
              'The 224 px run uses the same physical crop as the 288 px run. It changes model resolution, not the requested area.','',
              '## Hypothesis checks','',
              '| Hypothesis | Comparison | Expected CV order observed? |','| --- | --- | --- |']
    for row in hypothesis_outcomes(table).itertuples(index=False):
        lines.append(f'| {row.hypothesis} | {row.comparison} | {"yes" if row.CV_order_matches_prediction else "no"} |')
    lines += ['', 'These are observed CV order checks, not significance tests. The optimizer-speed prediction needs the learning curves too.',
              'We keep all planned runs, including worse results. Eight recipes on 24 samples can overfit the validation choice.',
              'The four-fold standard deviation is not a confidence interval. Small score gaps need more checks.','',
              '## Checkpoints and inference','',
              'Every fold has its config, environment, OOF curves, best/last checkpoint, inference weights, test curves, and W&B model artifact.',
              'See `cnn_checkpoints.csv` for exact W&B artifact handles and run links.',
              'Use `python -m soilcv.cnn.inference` with photos, one supplied PPM per photo, and model directories containing config.json and weights.pt.',
              'The model can be rebuilt without downloading ImageNet weights. Use the matching soilcv/cnn/experiments.py source version.','',
              '## Submission status','',
              '`cnn_submission.csv` is ready. It averages test curves from the four fold models of the selected recipe.',
              'CV evaluates one held-out fold model per sample. It is not a CV estimate of that four-model test ensemble.',
              'No competition submission was made. Public LB is not submitted; private LB is not available.',
              'No test labels or test-specific rules were used to select the model.','',
              'AI helped write and run this code. Read the source and results before the defense.']
    lines += ['', '## Observed outcomes', ''] + ['- '+line for line in interpret_results(table)]
    path=output/'cnn_report.md'
    path.write_text('\n'.join(lines)+'\n')
    return path


def interpret_results(table):
    s=table.set_index('experiment').cv_emd
    lines=[
        f"Frozen ResNet EMD is {s['cnn-r18-frozen']:.3f}; tuned ResNet EMD is {s['cnn-r18-adamw']:.3f}. " +
        ('Tuning helps here, but the gain is small.' if s['cnn-r18-adamw']<s['cnn-r18-frozen'] else 'Tuning does not help here. Few samples can make fine-tuning overfit.'),
        f"AdamW recipe: {s['cnn-r18-adamw']:.3f}; SGD recipe: {s['cnn-r18-sgd']:.3f}. The score gap does not isolate optimizer choice or prove learning speed.",
        f"Frozen BN stats: {s['cnn-r18-adamw']:.3f}; updated BN stats: {s['cnn-r18-train-bn']:.3f}. " +
        ('This fits the concern about running statistics from related tile batches.' if s['cnn-r18-adamw']<s['cnn-r18-train-bn'] else 'The expected BN ranking did not hold. This split does not show a benefit from frozen statistics.'),
        f"Light augmentation: {s['cnn-r18-adamw']:.3f}; no augmentation: {s['cnn-r18-no-aug']:.3f}. " +
        ('Light augmentation helps in this recipe.' if s['cnn-r18-adamw']<s['cnn-r18-no-aug'] else 'The expected gain did not hold. Some colour changes may remove useful cues; this run does not test each transform alone.'),
        f"Warmup/cosine: {s['cnn-r18-adamw']:.3f}; constant LR: {s['cnn-r18-constant-lr']:.3f}. " +
        ('The schedule prediction was wrong. A full LR from the start works better in this short run; many best checkpoints are early.' if s['cnn-r18-constant-lr']<s['cnn-r18-adamw'] else 'The schedule recipe helps here. Warmup alone was not isolated.'),
        f"288 px: {s['cnn-r18-adamw']:.3f}; 224 px: {s['cnn-r18-224']:.3f}. Compare this gap with fold variation before claiming that more pixels help.",
        f"ConvNeXt-Tiny: {s['cnn-convnext-tiny']:.3f}; matched ResNet recipe: {s['cnn-r18-adamw']:.3f}. " +
        ('The stronger model improves this matched recipe, but uses more compute.' if s['cnn-convnext-tiny']<s['cnn-r18-adamw'] else 'The expected architecture gain did not hold. More capacity and different pretrained features do not ensure a gain on 18 train samples per fold.'),
    ]
    return lines
