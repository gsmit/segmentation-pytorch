import os
import json
import torch
import random
import numpy as np
import torch.backends.cudnn
import albumentations as albu
import matplotlib.pyplot as plt

from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from sklearn.metrics import jaccard_score, f1_score


def mask_to_rle(img):
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_to_mask(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[1] * shape[0], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def dice_score(prediction, mask, average):
    y_pred = prediction.flatten()
    y_true = mask.flatten()
    return f1_score(y_true, y_pred, average=average)


def iou_score(prediction, mask, average):
    y_pred = prediction.flatten()
    y_true = mask.flatten()
    return jaccard_score(y_true, y_pred, average=average)


def count_elements(array, exclude=0):
    count = np.bincount(array[array != exclude])
    return exclude if count.size == 0 else np.argmax(count)


def save_predictions(out_path, index, image, ground_truth_mask, predicted_mask, average):
    titles = ['Image', 'Ground Truth Mask', 'Predicted Mask']
    images = [image, ground_truth_mask, predicted_mask]
    plt.figure(figsize=(16, 5))

    for i, (name, image) in enumerate(zip(titles, images)):
        plt.subplot(1, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name)

        plt.imshow(image, vmin=0, vmax=1)

    out_name = os.path.join(out_path, f'predictions_{str(index).zfill(2)}.png')
    plt.savefig(out_name)
    plt.close('all')


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """Wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Constructs preprocessing augmentation."""

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor)]

    return albu.Compose(_transform)
