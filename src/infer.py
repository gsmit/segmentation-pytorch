import os
import cv2
import time
import yaml
import torch
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from torch.utils.data import DataLoader
from skimage.util.shape import view_as_windows
from segmentation_models_pytorch import Unet, UnetPlusPlus, DeepLabV3Plus
from monai.metrics import DiceMetric, compute_roc_auc

# TODO: Iplement YAML file containing inference parameters below.

config = '../config/training.yml'

data = [
    '../images/image01.tif',
    '../images/image02.tif',
    '../images/image03.tif'
]

masks = [
    '../masks/image01.tif',
    '../masks/image02.tif',
    '../masks/image03.tif'
]

# evaluation metric
criterion = DiceMetric(include_background=False, reduction='mean')

# load config file
with open(config, 'r') as yaml_file:
    cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

# patch settings
tile_size = 768
overlap = 128
original_spacing = 0.5
target_spacing = 2.0
mask_threshold = 0.5
verbose = True

# calculate overlap and shapes
half_overlap = int(overlap / 2)
step_size = tile_size - overlap
resize_factor = int(target_spacing / original_spacing)
window_shape = (tile_size, tile_size, 3)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 1 if len(cfg['classes']) == 1 else (len(cfg['classes']) + 1)
activation = 'sigmoid' if num_classes == 1 else 'softmax2d'

aux_params = dict(
    pooling='avg',  # one of 'avg', 'max'
    dropout=cfg['dropout'],  # dropout ratio, default is None
    activation='sigmoid',  # activation function, default is None
    classes=num_classes)  # define number of output labels

num_classes = 1 if len(cfg['classes']) == 1 else (len(cfg['classes']) + 1)
activation = 'sigmoid' if num_classes == 1 else 'softmax2d'
background = False if cfg['ignore_channels'] else True

aux_params = dict(
    pooling=cfg['pooling'],  # one of 'avg', 'max'
    dropout=cfg['dropout'],  # dropout ratio, default is None
    activation='sigmoid',  # activation function, default is None
    classes=num_classes)  # define number of output labels

# configure model
models = {
    'unet': Unet(
        encoder_name=cfg['encoder_name'],
        encoder_weights=cfg['encoder_weights'],
        decoder_use_batchnorm=cfg['use_batchnorm'],
        classes=num_classes,
        activation=activation,
        aux_params=aux_params),
    'unetplusplus': UnetPlusPlus(
        encoder_name=cfg['encoder_name'],
        encoder_weights=cfg['encoder_weights'],
        decoder_use_batchnorm=cfg['use_batchnorm'],
        classes=num_classes,
        activation=activation,
        aux_params=aux_params),
    'deeplabv3plus': DeepLabV3Plus(
        encoder_name=cfg['encoder_name'],
        encoder_weights=cfg['encoder_weights'],
        classes=num_classes,
        activation=activation,
        aux_params=aux_params)}

assert cfg['architecture'] in models.keys()
model = models[cfg['architecture']]

# load model weights
checkpoint = os.path.join(cfg['checkpoint_dir'], cfg['checkpoint_name'])
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()
model.to(device)


def predict_array(m):
    start = time.time()

    m = m / 255.
    m = m.transpose((0, 3, 1, 2))
    m = torch.Tensor(m)  # transform to torch tensor

    dataset = torch.utils.data.TensorDataset(m)
    loader = DataLoader(dataset, batch_size=4, num_workers=2, shuffle=False)

    pred = []
    for batch_idx, x in enumerate(loader):
        x = x[0].to(device)
        with torch.no_grad():
            y_pred, _ = model.forward(x)  # seg + clf output
            y_pred = y_pred.cpu().numpy()
            pred.append(y_pred)

    pred = np.concatenate(pred, axis=0)
    pred = pred.transpose((0, 2, 3, 1))
    print(f'Finished predicting in {round(time.time() - start, 1)}s')

    return pred


def apply_tta(m):
    result = np.zeros(shape=(m.shape[0], m.shape[1], m.shape[2], 1), dtype=np.float32)

    # rotate 0
    array = m
    array = predict_array(array)
    result = result + array

    # rotate 90
    array = np.rot90(m=m, k=1, axes=(1, 2))
    array = predict_array(array)
    array = np.rot90(m=array, k=3, axes=(1, 2))
    result = result + array

    # rotate 180
    array = np.rot90(m=m, k=2, axes=(1, 2))
    array = predict_array(array)
    array = np.rot90(m=array, k=2, axes=(1, 2))
    result = result + array

    # rotate 270
    array = np.rot90(m=m, k=3, axes=(1, 2))
    array = predict_array(array)
    array = np.rot90(m=array, k=1, axes=(1, 2))
    result = result + array

    # flip, rotate 0
    array = np.flip(m=m, axis=1)
    array = predict_array(array)
    array = np.flip(m=array, axis=1)
    result = result + array

    # flip, rotate 90
    array = np.rot90(m=np.flip(m=m, axis=1), k=1, axes=(1, 2))
    array = predict_array(array)
    array = np.flip(m=np.rot90(m=array, k=3, axes=(1, 2)), axis=1)
    result = result + array

    # flip, rotate 180
    array = np.rot90(m=np.flip(m=m, axis=1), k=2, axes=(1, 2))
    array = predict_array(array)
    array = np.flip(m=np.rot90(m=array, k=2, axes=(1, 2)), axis=1)
    result = result + array

    # flip, rotate 270
    array = np.rot90(m=np.flip(m=m, axis=1), k=3, axes=(1, 2))
    array = predict_array(array)
    array = np.flip(m=np.rot90(m=array, k=1, axes=(1, 2)), axis=1)
    result = result + array

    # average predictions
    result = result / 8.0
    # result = result / 1.0

    return result


def calculate_padding(border_size, block_size):
    # calculate padding pixels for resizing
    padding = block_size - (border_size % block_size)
    padding = padding if padding != block_size else 0
    return padding


def load_tiff(image_path):
    # read tiff image and preprocess
    tiff_image = tiff.imread(image_path)
    tiff_image = np.squeeze(tiff_image)  # remove dimensions of size one
    tiff_image = tiff_image.transpose((1, 2, 0)) if tiff_image.shape[0] == 3 else tiff_image
    return tiff_image


def run_inference():

    # keep track of scores
    scores = []

    for e, (image_path, mask_path) in enumerate(zip(data, masks)):

        print('Preprocessing image...') if verbose else None
        print(image_path)

        # select and load image
        image = load_tiff(image_path=image_path)
        hwc = image.shape

        # add padding for correctly resizing image
        resize_pad_h = calculate_padding(hwc[0], resize_factor)
        resize_pad_w = calculate_padding(hwc[1], resize_factor)
        image = np.pad(image, ((0, resize_pad_h), (0, resize_pad_w), (0, 0)), constant_values=255)

        # resize image
        hwc_padded = image.shape
        new_h = int(image.shape[0] / resize_factor)
        new_w = int(image.shape[1] / resize_factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)  # cv2 applies (w,h,c) and not (h,w,c)
        hwc_resized = image.shape

        # add padding for creating tiles
        tile_pad_h = step_size - (image.shape[0] % step_size)
        tile_pad_w = step_size - (image.shape[1] % step_size)
        image = np.pad(image, ((0, tile_pad_h), (0, tile_pad_w), (0, 0)), constant_values=0)

        # add padding as overlap for all tiles
        image = np.pad(image, ((half_overlap, half_overlap), (half_overlap, half_overlap), (0, 0)), constant_values=0)

        # create tiles
        image_shape = image.shape
        image = view_as_windows(image, window_shape, step=step_size)
        image = np.squeeze(image)  # remove dimensions of size one

        # resize tiles
        tiles_h = image.shape[0]
        tiles_w = image.shape[1]
        image = image.reshape((-1, tile_size, tile_size, 3))

        print('Making predictions...') if verbose else None

        # make predictions
        predictions = apply_tta(image)
        predictions = predictions.squeeze()
        print(predictions.shape)

        predictions = np.round(predictions, decimals=0).astype(np.uint8)
        tiles = predictions[:, half_overlap:-half_overlap, half_overlap:-half_overlap]
        tiles = tiles.reshape((tiles_h, tiles_w, (tile_size - overlap), (tile_size - overlap), num_classes))
        tiles = tiles.squeeze()

        del predictions  # clean up memory
        image = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)

        for i in range(tiles_h):
            for j in range(tiles_w):
                patch = tiles[i, j]
                a_0 = i * (tile_size - overlap)
                a_1 = (i + 1) * (tile_size - overlap)
                b_0 = j * (tile_size - overlap)
                b_1 = (j + 1) * (tile_size - overlap)
                image[a_0:a_1, b_0:b_1] = patch

        print('Finishing up...') if verbose else None

        del tiles  # clean up memory
        image = np.where(image >= mask_threshold, 1, 0).astype(np.uint8)
        image = image[:hwc_resized[0], :hwc_resized[1]]
        image = cv2.resize(image, (hwc_padded[1], hwc_padded[0]), interpolation=cv2.INTER_NEAREST)
        image = image[:hwc[0], :hwc[1]]
        image = np.round(image, decimals=0).astype(np.uint8)

        # load mask file
        mask = load_tiff(image_path=mask_path)
        assert image.shape == mask.shape

        plt.imshow(mask)
        plt.show()

        # calculate score
        score = f1_score(image.flatten(), mask.flatten())
        scores.append(score)
        print('Score:', score)

    print('Average:', np.mean(scores))
    print('Configuration:', config)


if __name__ == '__main__':
    run_inference()
