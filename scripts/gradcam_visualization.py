##
## Demo script: Grad-Cam visualization on BigEarthNet satellite imagery
##
## Requires
##      a) Featurization and classifier model artifacts
##      b) Small sample of BigEarthNet images. Available sample data
##         is located in `data/bigearth-100-single-multi`. I recommend
##         using these images:
##              test_data/bigearth-100-single-multi/Coniferous_forest/S2B_MSIL2A_20180525T94031_80_19
##              test_data/bigearth-100-single-multi/Continuous_urban_fabric/S2B_MSIL2A_20180515T112109_75_21
##              test_data/bigearth-100-single-multi/Non_irrigated_arable_land/S2A_MSIL2A_20170803T094031_65_25
##              test_data/bigearth-100-single-multi/Non_irrigated_arable_land/S2B_MSIL2A_20170914T093029_31_24
##              test_data/bigearth-100-single-multi/Pastures/S2A_MSIL2A_20170717T113321_31_64
##      c) Dependencies in `requirements.txt`:
##
## Steps
##      1) Load BigEarthNet image
##      2) Featurize image
##      3) Make prediction & generate explanations
##      4) Display explanations
##
## Notes
##      You might observe that many explanations are concentrated in the corners of images, which
##      is an artifact (or pathology) that derives from the data augmentation procedure in the
##      self-supervised training. The explanations helped us uncover this behavior!
##

import types
import os

import cv2
from tifffile import imread
from PIL import Image
import numpy as np
import torch
from torch import nn
from rsp.data import bilinear_upsample, BANDS
from rsp.moco_r50.data import sentinel_augmentation_valid
from rsp.moco_r50.inference import moco_r50

np.random.seed(0)
torch.manual_seed(0)

IMAGE_PTH = (
    "test_data/bigearth-100-single-multi/Pastures/S2A_MSIL2A_20170717T113321_38_10"
)
FEATURIZER_PTH = "featurizer-weights.pth"
CLASSIFIER_PTH = "classifier-weights.pth"
OUTPUT_DIR = "explanations"

CLASSES = {
    0: "Broad_leaved_forest",
    1: "Coniferous_forest",
    2: "Continuous_urban_fabric",
    3: "Non_irrigated_arable_land",
    4: "Pastures",
    5: "Sea_and_ocean",
    6: "Water_bodies",
}

## Step 1: Load BigEarthNet image


def load_image(inpath):
    """ load input image """
    img_bands = [imread(f"{inpath}_{band}.tif") for band in BANDS]
    img = np.stack([bilinear_upsample(img_band) for img_band in img_bands])
    return img


## Step 2: Featurize image


def build_featurizer(device):
    """ load featurization model"""

    model = moco_r50(FEATURIZER_PTH, map_location=device)

    def forward(self, x):
        """ patch forward to eliminate pooling, flattening + fully connected"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    model.forward = types.MethodType(forward, model)

    model = model.to(device)
    model = model.eval()

    return model


def featurize(img, device):
    """ featurize input image """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    featurizer = build_featurizer(device)

    img = img[:12].transpose(1, 2, 0) / 10_000
    img = sentinel_augmentation_valid()(image=img)["image"]

    with torch.no_grad():
        img = img.unsqueeze(0).to(device)
        features = featurizer(img)

    return features


## Step 3: Make prediction & generate explanations


def get_one_hots(shape, device):
    """ get list of one hot outputs for each class """
    one_hots = []
    for i in range(len(CLASSES)):
        one_hot = np.zeros(shape, dtype=np.float32)
        one_hot[:, i] = 1
        one_hots.append(one_hot)
    one_hots = [torch.from_numpy(o_h).to(device) for o_h in one_hots]
    one_hots = [o_h.requires_grad_(True) for o_h in one_hots]
    return one_hots


def get_mask(classifier, inputs, logits, one_hot):
    """ get GradCam mask """
    one_hot = torch.sum(one_hot * logits)
    classifier.zero_grad()
    one_hot.backward(retain_graph=True)
    grads = inputs.grad.cpu().data.numpy()
    features = inputs.cpu().data.numpy()
    weights = np.mean(grads, axis=(2, 3))
    cam_mask = np.sum(weights[:, :, np.newaxis, np.newaxis] * features, axis=1)
    return cam_mask[0]


def resize_mask(mask):
    """ resize mask to dimension of input image"""
    mask = np.maximum(mask, 0)
    mask = cv2.resize(mask, (120, 120))
    mask = mask - np.min(mask)
    if np.max(mask) > 0:
        mask = mask / np.max(mask)
    return mask


def predict(features, device):
    """ make prediction & generate explanations """

    classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(features.shape[1], features.shape[1]),
        nn.BatchNorm1d(features.shape[1]),
        nn.ReLU(),
        nn.Linear(features.shape[1], len(CLASSES)),
    )

    classifier = classifier.to(device)
    classifier.load_state_dict(torch.load(CLASSIFIER_PTH, map_location=device))
    classifier = classifier.eval()

    features.requires_grad = True
    logits = classifier(features)

    _, prediction = torch.max(logits, 1)
    prediction = prediction.cpu().data.numpy()[0]
    pred_class = CLASSES[prediction]

    one_hots = get_one_hots(logits.shape, device)
    masks = []
    for i, one_hot in enumerate(one_hots):
        mask = get_mask(classifier, features, logits, one_hot)
        mask = resize_mask(mask)
        masks.append(mask)

    return pred_class, masks


## Step 5: Display Explanations


def display(img, mask, idx, mask_strength: float = 0.4):
    """ display explanations on image """
    img = np.stack((img[3], img[2], img[1])).transpose(1, 2, 0)
    img = (img / 4096) ** 0.5
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255 * mask_strength
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(f"{OUTPUT_DIR}/expl_class_{CLASSES[i]}.jpg", np.uint8(255 * cam))


## Run
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img = load_image(IMAGE_PTH)
features = featurize(img, device)
pred_class, explanations = predict(features, device)
for i, explanation in enumerate(explanations):
    display(img, explanation, i)

print(f"Predicted class = {pred_class}")
print(f"True class = {IMAGE_PTH.split('/')[-2]}")
