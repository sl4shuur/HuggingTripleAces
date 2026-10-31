"""Part 1 CNN recipes. Shared data, training, metrics, and tracking come from soilcv."""
import os
import sys
from pathlib import Path

import argparse
import gc
import json
import math
import random
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torchvision import models, transforms as T

from soilcv.config import ExperimentConfig, seed_everything, seed_worker
from soilcv.data import load_metadata, load_folds, DEFAULT_FOLDS_PATH, TileDataset, sample_weights
from soilcv.tiles import make_tile_manifest, build_tile_cache
from soilcv.curves import TARGET_COLUMNS, aggregate_predictions, validate_curves
from soilcv.metrics import evaluate_samples, group_errors
from soilcv.nn import MassCurveHead, EMDLoss, make_param_groups, set_trainable
from soilcv.training import fit_supervised_fold, predict_tiles
from soilcv.tracking import start_run, log_artifact
from soilcv.artifacts import package_identity, file_hash, identity_hash, write_json, save_environment

ROOT = Path(os.environ.get('SOILCV_CNN_OUTPUT_DIR', '/kaggle/working/part1_cnn'))
CACHE = Path(os.environ.get('SOILCV_CNN_CACHE_DIR', '/tmp/soilcv-cnn-tiles'))
HYPOTHESES_COMMIT = '804c059'
NORMALIZE = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
BASE = dict(arch='resnet18', frozen=False, freeze_bn=True, augmentation=True,
            optimizer='adamw', backbone_lr=0.0001, head_lr=0.001, weight_decay=0.01,
            schedule='warmup_cosine', pixels=288, batch_size=16, epochs=24, patience=6)
RECIPES = [
    dict(BASE, id='cnn-r18-frozen', hypothesis='H-CNN-01', frozen=True, head_lr=0.003),
    dict(BASE, id='cnn-r18-adamw', hypothesis='H-CNN-02'),
    dict(BASE, id='cnn-r18-sgd', hypothesis='H-CNN-03', optimizer='sgd', backbone_lr=0.001, head_lr=0.01, weight_decay=0.0001),
    dict(BASE, id='cnn-r18-train-bn', hypothesis='H-CNN-04', freeze_bn=False),
    dict(BASE, id='cnn-r18-no-aug', hypothesis='H-CNN-05', augmentation=False),
    dict(BASE, id='cnn-r18-constant-lr', hypothesis='H-CNN-06', schedule='constant'),
    dict(BASE, id='cnn-r18-224', hypothesis='H-CNN-07', pixels=224),
    dict(BASE, id='cnn-convnext-tiny', hypothesis='H-CNN-08', arch='convnext_tiny', freeze_bn=False),
]


def prepare():
    started = time.perf_counter()
    ROOT.mkdir(exist_ok=True, parents=True)
    data_dir = next(Path('/kaggle/input').rglob('Training_labels_updated.csv')).parent
    labels, photos, template = load_metadata(data_dir)
    folds = load_folds(DEFAULT_FOLDS_PATH, labels.sample_id)
    manifest = make_tile_manifest(photos, tile_mm=64, output_px=288, image_root=data_dir)
    build_tile_cache(manifest, CACHE, image_root=data_dir)
    manifest = manifest.merge(folds[['sample_id', 'fold']], on='sample_id', how='left', validate='many_to_one')
    manifest.to_json(ROOT / 'tiles.json', orient='records', indent=2)
    labels.to_json(ROOT / 'labels.json', orient='records', indent=2)
    template.to_json(ROOT / 'template.json', orient='records', indent=2)
    folds.to_csv(ROOT / 'sample_folds.csv', index=False)
    weight_info = {}
    for name, weights in [('resnet18', models.ResNet18_Weights.IMAGENET1K_V1),
                          ('convnext_tiny', models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)]:
        state = weights.get_state_dict(progress=False, check_hash=True)
        path = Path(torch.hub.get_dir()) / 'checkpoints' / Path(weights.url).name
        weight_info[name] = {'enum': str(weights), 'url': weights.url, 'sha256': file_hash(path)}
        del state
        gc.collect()
    metadata = {'soilcv': package_identity(), 'labels_hash': file_hash(data_dir / 'Training_labels_updated.csv'),
                'folds_hash': file_hash(DEFAULT_FOLDS_PATH), 'cache_id': manifest.cache_id.iloc[0],
                'pretrained': weight_info, 'torch': torch.__version__, 'torchvision': torchvision.__version__,
                'hypotheses_commit': HYPOTHESES_COMMIT, 'prepare_seconds': time.perf_counter()-started}
    write_json(ROOT / 'metadata.json', metadata)
    write_json(ROOT / 'recipes.json', RECIPES)
    print('Prepared', len(labels), 'train samples and', len(manifest), 'tiles.', flush=True)
    print('Cache:', metadata['cache_id'], flush=True)
    return metadata


class QuarterTurn:
    def __call__(self, image):
        angle = random.choice([0, 90, 180, 270])
        return image.rotate(angle)


def image_transform(recipe, train=False):
    steps = [T.Resize((recipe['pixels'], recipe['pixels']), interpolation=T.InterpolationMode.BILINEAR, antialias=True)]
    if train and recipe['augmentation']:
        steps += [QuarterTurn(), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                  T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)]
    return T.Compose([*steps, T.ToTensor(), T.Normalize(**NORMALIZE)])


class SoilCNN(nn.Module):
    def __init__(self, recipe, train_mean, *, pretrained=True):
        super().__init__()
        if recipe['arch'] == 'resnet18':
            self.encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None, progress=False)
            size = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        else:
            self.encoder = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None, progress=False)
            size = self.encoder.classifier[2].in_features
            self.encoder.classifier[2] = nn.Identity()
        self.head = MassCurveHead(size)
        mass = np.maximum(np.diff(np.r_[0, train_mean]), 0.001)
        with torch.no_grad():
            nn.init.normal_(self.head.projection.weight, std=0.001)
            self.head.projection.bias.copy_(torch.tensor(np.log(mass), dtype=torch.float32))
        if recipe['frozen']:
            set_trainable(self.encoder, False)

    def forward(self, image):
        return self.head(self.encoder(image))


def learning_rate_factor(epoch, epochs):
    if epoch < 3:
        return (epoch+1)/3
    progress = min(1, max(0, (epoch-3)/max(1, epochs-4)))
    return 0.05 + 0.95 * (1+math.cos(math.pi*progress))/2


def run_fold(recipe, fold):
    torch.set_num_threads(2)
    seed_everything(42 + fold)
    metadata = json.loads((ROOT / 'metadata.json').read_text())
    labels = pd.DataFrame(json.loads((ROOT / 'labels.json').read_text()))
    manifest = pd.DataFrame(json.loads((ROOT / 'tiles.json').read_text()))
    train_tiles = manifest[manifest.fold.notna() & manifest.fold.ne(fold)].copy()
    valid_tiles = manifest[manifest.fold.eq(fold)].copy()
    test_tiles = manifest[manifest.fold.isna()].copy()
    train_labels = labels[labels.sample_id.isin(train_tiles.sample_id)]
    valid_labels = labels[labels.sample_id.isin(valid_tiles.sample_id)]
    train_mean = train_labels[list(TARGET_COLUMNS)].to_numpy().mean(axis=0)
    recipe_hash = identity_hash({'recipe': recipe, 'metadata': {k: v for k, v in metadata.items() if k != 'prepare_seconds'}, 'source': file_hash(__file__)})
    output = ROOT / 'runs' / recipe['id'] / f'fold_{fold}'
    if (output / 'result.json').exists():
        result = json.loads((output / 'result.json').read_text())
        if result['recipe_hash'] != recipe_hash:
            raise ValueError('Existing result uses different code or settings. Use a new experiment ID.')
        print('Already complete:', recipe['id'], fold, flush=True)
        return result
    sampler = WeightedRandomSampler(sample_weights(train_tiles), len(train_tiles), replacement=True,
                                    generator=torch.Generator().manual_seed(142+fold))
    loader_args = dict(batch_size=recipe['batch_size'], num_workers=2, pin_memory=True,
                       worker_init_fn=seed_worker, persistent_workers=False)
    train_loader = DataLoader(TileDataset(train_tiles, CACHE, labels, image_transform(recipe, train=True)), sampler=sampler,
                              generator=torch.Generator().manual_seed(242+fold), **loader_args)
    valid_loader = DataLoader(TileDataset(valid_tiles, CACHE, labels, image_transform(recipe)), shuffle=False,
                              generator=torch.Generator().manual_seed(342+fold), **loader_args)
    test_loader = DataLoader(TileDataset(test_tiles, CACHE, transform=image_transform(recipe)), shuffle=False, **loader_args)
    model = SoilCNN(recipe, train_mean).to('cuda:0')
    parameters = make_param_groups(model.head, lr=recipe['head_lr'], weight_decay=recipe['weight_decay'])
    if not recipe['frozen']:
        parameters += make_param_groups(model.encoder, lr=recipe['backbone_lr'], weight_decay=recipe['weight_decay'])
    if recipe['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(parameters)
    else:
        optimizer = torch.optim.SGD(parameters, momentum=0.9)
    scheduler = None
    if recipe['schedule'] == 'warmup_cosine':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: learning_rate_factor(epoch, recipe['epochs']))
    config = ExperimentConfig(recipe['id'], stage='cnn', model=recipe['arch'], fold=fold, seed=42+fold,
        hypothesis_id=recipe['hypothesis'], output_root=str(ROOT/'runs'),
        versions={'soilcv': metadata['soilcv'], 'folds': metadata['folds_hash'], 'data': metadata['labels_hash'],
                  'cache': metadata['cache_id'], 'hypotheses_commit': HYPOTHESES_COMMIT, 'source': file_hash(__file__)},
        params={**recipe, 'normalization': NORMALIZE, 'pretrained': metadata['pretrained'][recipe['arch']],
                'train_mean_prior': train_mean.tolist(), 'sampling': 'equal_sample_photo_tile',
                'loss_units': 'percent-log10', 'amp_dtype': 'float16', 'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES')})
    with start_run(config) as run:
        predictions, history = fit_supervised_fold(model, train_loader, valid_loader, valid_labels, optimizer, EMDLoss(), config,
            epochs=recipe['epochs'], device='cuda:0', amp=True, scheduler=scheduler, scheduler_step='epoch',
            patience=recipe['patience'], clip_grad=5.0, freeze_batchnorm=recipe['freeze_bn'],
            frozen_modules=(model.encoder,) if recipe['frozen'] else (), run=run)
        test_predictions = aggregate_predictions(predict_tiles(model, test_loader, device='cuda:0', amp=True))
        validate_curves(test_predictions[list(TARGET_COLUMNS)])
        test_predictions.to_csv(output/'test_predictions.csv', index=False)
        errors = group_errors(valid_labels, predictions)
        errors.to_csv(output/'group_errors.csv', index=False)
        torch.save(model.state_dict(), output/'weights.pt')
        result = {'experiment': recipe['id'], 'fold': fold, 'recipe_hash': recipe_hash,
                  **evaluate_samples(valid_labels, predictions), 'epochs_ran': len(history),
                  'best_epoch': min(history, key=lambda row: row['val']['emd'])['epoch'], 'run_url': run.url,
                  **json.loads((output/'compute.json').read_text())}
        artifact = log_artifact(run, output/'weights.pt', name=f"{recipe['id']}-fold-{fold}", artifact_type='model',
                                metadata={'recipe': recipe, 'weights': metadata['pretrained'][recipe['arch']],
                                          'code': file_hash(__file__), 'fold': fold})
        artifact.wait()
        result['checkpoint_artifact'] = artifact.qualified_name
        log_artifact(run, output/'oof.csv', name=f"{recipe['id']}-fold-{fold}-oof", artifact_type='predictions')
        run.summary['result/checkpoint_artifact'] = result['checkpoint_artifact']
        write_json(output/'result.json', result)
        print(json.dumps(result), flush=True)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'train'])
    parser.add_argument('--experiment', default='cnn-r18-adamw')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    else:
        recipe = next(dict(recipe) for recipe in RECIPES if recipe['id'] == args.experiment)
        if args.smoke:
            recipe.update(id='cnn-smoke', hypothesis='H-CNN-00', epochs=2, patience=None)
        run_fold(recipe, args.fold)
