"""Shared encoder -> weighted sample features -> one valid grain-size curve."""
import random
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision import models, transforms as T
from soilcv.nn import MassCurveHead, set_trainable

NORMALIZE={'mean':[0.485,0.456,0.406],'std':[0.229,0.224,0.225]}


class QuarterTurn:
    def __call__(self,image):return image.rotate(random.choice([0,90,180,270]))


def image_transform(recipe,train=False):
    # Cache already has the final resolution: never resize a cached tile again.
    steps=[]
    if train and recipe['augmentation']:
        steps=[QuarterTurn(),T.RandomHorizontalFlip(),T.RandomVerticalFlip(),
               T.ColorJitter(brightness=.15,contrast=.15,saturation=.1,hue=.02)]
    return T.Compose([*steps,T.ToTensor(),T.Normalize(**NORMALIZE)])


class MILCNN(nn.Module):
    def __init__(self,recipe,train_mean,*,pretrained=True):
        super().__init__()
        if recipe['arch']=='resnet18':
            self.encoder=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            size=self.encoder.fc.in_features;self.encoder.fc=nn.Identity()
        elif recipe['arch']=='convnext_tiny':
            self.encoder=models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
            size=self.encoder.classifier[2].in_features;self.encoder.classifier[2]=nn.Identity()
        else:raise ValueError('Unknown CNN architecture.')
        self.head=MassCurveHead(size)
        mass=np.maximum(np.diff(np.r_[0,train_mean]),.001)
        with torch.no_grad():
            nn.init.normal_(self.head.projection.weight,std=.001)
            self.head.projection.bias.copy_(torch.tensor(np.log(mass),dtype=torch.float32))
        self.chunk_size=recipe['chunk_size']
        self.frozen=recipe['frozen']
        if self.frozen:set_trainable(self.encoder,False)

    def encode(self,images):
        features=[]
        for chunk in images.split(self.chunk_size):
            if self.training and not self.frozen:
                features.append(checkpoint(self.encoder,chunk,use_reentrant=False))
            else:features.append(self.encoder(chunk))
        return torch.cat(features)

    def pool(self,features,weights):
        weights=weights.float()
        return self.head((features.float()*weights[:,None]).sum(dim=0,keepdim=True)/weights.sum())

    def forward(self,images,weights):
        return self.pool(self.encode(images),weights)
