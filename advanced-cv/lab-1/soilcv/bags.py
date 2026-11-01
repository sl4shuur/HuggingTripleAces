"""Full-photo physical grids and complete sample bags for MIL."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from .artifacts import file_hash, identity_hash, write_json
from .curves import TARGET_COLUMNS


def grid_starts(length, side):
    if side < 1 or side > length:
        raise ValueError('Physical tile must fit inside the photo.')
    positions = list(range(0,length-side+1,side))
    if positions[-1] != length-side:
        positions.append(length-side)
    return positions


def rectangle_coverage(rectangles, width, height):
    """Exact union area and overlap-corrected tile weights, without a huge pixel mask."""
    rectangles = np.asarray(rectangles,dtype=int)
    if rectangles.ndim!=2 or rectangles.shape[1]!=4 or len(rectangles)==0:
        raise ValueError('Use non-empty left, top, right, bottom rectangles.')
    if (rectangles[:,:2]<0).any() or (rectangles[:,2:] > [width,height]).any() or (rectangles[:,2:]<=rectangles[:,:2]).any():
        raise ValueError('A rectangle is outside the photo.')
    xs=np.unique(np.r_[0,width,rectangles[:,[0,2]].ravel()])
    ys=np.unique(np.r_[0,height,rectangles[:,[1,3]].ravel()])
    count=np.zeros((len(ys)-1,len(xs)-1),dtype=int)
    slices=[]
    for left,top,right,bottom in rectangles:
        x0,x1=np.searchsorted(xs,[left,right]);y0,y1=np.searchsorted(ys,[top,bottom])
        slices.append((slice(y0,y1),slice(x0,x1)))
        count[y0:y1,x0:x1]+=1
    area=np.diff(ys)[:,None]*np.diff(xs)[None,:]
    covered=int(area[count>0].sum())
    shared=area/np.maximum(count,1)
    weights=np.array([shared[region].sum()/(width*height) for region in slices])
    return covered/(width*height), weights


def make_grid_manifest(photos, *, tile_mm=64, output_px=288, image_root=None):
    if tile_mm<=0 or not np.isfinite(tile_mm) or not isinstance(output_px,int) or output_px<1:
        raise ValueError('Use a positive physical size and integer resolution.')
    if photos.empty or photos.duplicated(['sample_id','photo']).any():
        raise ValueError('Use unique non-empty photo rows.')
    rows=[]
    for photo in photos.sort_values(['sample_id','photo']).itertuples(index=False):
        path=Path(photo.path);ppm=float(photo.ppm_est)
        if not np.isfinite(ppm) or ppm<=0:raise ValueError('PPM must be positive.')
        with Image.open(path) as raw:
            width,height=ImageOps.exif_transpose(raw).size
        side=round(tile_mm*ppm)
        boxes=[(x,y,x+side,y+side) for y in grid_starts(height,side) for x in grid_starts(width,side)]
        coverage,weights=rectangle_coverage(boxes,width,height)
        if coverage!=1 or not np.isclose(weights.sum(),1):raise ValueError('Incomplete photo coverage.')
        digest=file_hash(path)
        for index,((left,top,right,bottom),weight) in enumerate(zip(boxes,weights)):
            rows.append({'sample_id':str(photo.sample_id),'photo':str(photo.photo),'tile_id':index,
                         'device':str(getattr(photo,'device','unknown')),
                         'image_path':str(path.relative_to(image_root)) if image_root else str(path.resolve()),
                         'source_hash':digest,'left':left,'top':top,'crop_px':side,
                         'width':width,'height':height,'ppm_est':ppm,'tile_mm':tile_mm,
                         'output_px':output_px,'photo_weight':float(weight),'coverage':coverage,
                         'interpolation':'LANCZOS','schema':'full_grid_v1'})
    result=pd.DataFrame(rows)
    cache_id=identity_hash(result.to_dict('records'))
    result['cache_id']=cache_id
    result['tile_path']=[f'{cache_id}/{i:06d}.png' for i in range(len(result))]
    return result


def cache_grid(manifest, root, *, image_root=None):
    root=Path(root);checksums={}
    cache=root/manifest.cache_id.iloc[0];cache.mkdir(parents=True,exist_ok=True)
    saved=cache/'checksums.json'
    previous=json.loads(saved.read_text()) if saved.exists() else {}
    for name,group in manifest.groupby('image_path',sort=False):
        path=Path(image_root)/name if image_root else Path(name)
        if not group.source_hash.eq(file_hash(path)).all():raise ValueError('Source image changed.')
        with Image.open(path) as raw:
            image=ImageOps.exif_transpose(raw).convert('RGB')
            for row in group.itertuples(index=False):
                target=root/row.tile_path
                if target.exists() and row.tile_path in previous:
                    if previous.get(row.tile_path)!=file_hash(target):
                        raise ValueError('Tile checksum mismatch: '+row.tile_path)
                else:
                    crop=image.crop((row.left,row.top,row.left+row.crop_px,row.top+row.crop_px))
                    crop.resize((row.output_px,row.output_px),Image.Resampling.LANCZOS).save(target)
                checksums[row.tile_path]=file_hash(target)
    write_json(saved,checksums)
    return checksums


def coverage_table(manifest):
    rows=[]
    for (sample,photo),group in manifest.groupby(['sample_id','photo'],sort=False):
        row=group.iloc[0];side=group.crop_px.to_numpy()
        boxes=np.column_stack([group.left,group.top,group.left+side,group.top+side])
        coverage,weights=rectangle_coverage(boxes,int(row.width),int(row.height))
        if coverage!=1 or not np.allclose(weights,group.photo_weight):
            raise ValueError('Coverage or overlap weights changed.')
        rows.append({'sample_id':sample,'photo':photo,'device':row.device,'width':row.width,'height':row.height,
                     'tiles':len(group),'coverage':coverage,'missing_native_pixels':0})
    return pd.DataFrame(rows)


class SampleBags:
    """Every item contains every tile of every photo of one sample. No subsampling."""
    def __init__(self, manifest, cache_root, transform, labels=None):
        self.manifest=manifest.reset_index(drop=True).copy()
        self.root=Path(cache_root);self.transform=transform
        self.samples=sorted(self.manifest.sample_id.unique())
        self.targets=None if labels is None else labels.set_index('sample_id')[list(TARGET_COLUMNS)]

    def __len__(self):return len(self.samples)

    def __getitem__(self,index):
        import torch
        sample=self.samples[index]
        rows=self.manifest[self.manifest.sample_id.eq(sample)]
        images=[]
        for row in rows.itertuples(index=False):
            with Image.open(self.root/row.tile_path) as image:
                images.append(self.transform(image.convert('RGB')))
        # Each photo has unit area weight. All photos then get equal sample weight.
        weights=rows.photo_weight.to_numpy()/rows.photo.nunique()
        result={'image':torch.stack(images),'bag_weights':torch.tensor(weights,dtype=torch.float32),
                'sample_id':sample,'photo':rows.photo.tolist(),'device':rows.device.tolist()}
        if self.targets is not None:
            result['target']=torch.tensor(self.targets.loc[sample].to_numpy(dtype=np.float32))[None]
        return result


def one_bag(items):
    if len(items)!=1:raise ValueError('Use batch_size=1; accumulate complete sample losses.')
    return items[0]
