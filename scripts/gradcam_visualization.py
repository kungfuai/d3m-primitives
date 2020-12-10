from glob import glob

from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
import torch 
from rsp.data import bilinear_upsample, BANDS
from tifffile import imread
from sklearn.preprocessing import LabelEncoder
from PIL import Image

def _bilinear_upsample(x, n=120):
    dtype = x.dtype
    assert len(x.shape) == 2
    if (x.shape[0] == n) and (x.shape[1] == n):
        return x
    else:
        x = x.astype(np.float)
        x = Image.fromarray(x)
        x = x.resize((n, n), Image.BILINEAR)
        x = np.array(x)
        x = x.astype(dtype)
    
    return x

def load_image(inpath: str):
    return np.stack([
        _bilinear_upsample(imread(f'{inpath}_B01.tif'), n=120),
        _bilinear_upsample(imread(f'{inpath}_B02.tif'), n=120),
        _bilinear_upsample(imread(f'{inpath}_B03.tif'), n=120),
        _bilinear_upsample(imread(f'{inpath}_B04.tif'), n=120),
        _bilinear_upsample(imread(f'{inpath}_B05.tif'), n=120),
        _bilinear_upsample(imread(f'{inpath}_B06.tif'), n=120),
        _bilinear_upsample(imread(f'{inpath}_B07.tif'), n=120),
        _bilinear_upsample(imread(f'{inpath}_B08.tif'), n=120),
        _bilinear_upsample(imread(f'{inpath}_B8A.tif'), n=120),
        _bilinear_upsample(imread(f'{inpath}_B09.tif'), n=120),
        _bilinear_upsample(imread(f'{inpath}_B11.tif'), n=120),
        _bilinear_upsample(imread(f'{inpath}_B12.tif'), n=120),
    ])

def load_inputs(datapath: str = '/test_data/bigearth-100-single/*/*.tif'):
    fnames  = sorted(glob(datapath))
    imnames = sorted(list(set(['_'.join(f.split('_')[:-1]) for f in fnames])))
    imgs =  [load_image(xx) for xx in tqdm(imnames)]
    
    y = [i.split('/')[3] for i in imnames]
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(y)
    
    return imgs, label_encoder, y

def deshape(img: np.ndarray):
    img = np.stack((img[3], img[2], img[1]))
    return img.transpose(1,2,0)

def show_cam_on_image(
    image,
    img: np.ndarray, 
    mask: np.ndarray, 
    mask_strength: float = 0.4
):
    img = deshape(img)
    img = (img / 4096) ** 0.5
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255 * mask_strength
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(f"imgs/img_{image}.jpg", np.uint8(255 * img))
    cv2.imwrite(f"cams/cam_{image}.jpg", np.uint8(255 * cam))

all_images, label_encoder, tgts = load_inputs('/test_data/bigearth-100-single-multi/*/*.tif')
masks = pd.read_pickle('cam_explanations.pkl') # images x nclasses
preds = pd.read_pickle('mlp_preds.pkl')

all_expl_classes = [
    'Broad_leaved_forest',
    'Coniferous_forest',
    'Continuous_urban_fabric',
    'Mixed_forest',
    'Non_irrigated_arable_land',
    'Pastures',
    'Sea_and_ocean',
    'Water_bodies'
]

for image in range(masks.shape[0]):

    pred = label_encoder.inverse_transform(preds.iloc[image, 0].reshape(1,))[0]
    print(f'Image: {image}, Predicted: {pred}, Label: {tgts[image]}')
    expl_class = label_encoder.transform(np.array([pred]))[0]
    show_cam_on_image(
        image,
        all_images[image], 
        masks.iloc[image, expl_class],
    )
