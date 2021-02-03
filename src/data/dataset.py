import random
import skimage.io
import numpy as np
import pandas as pd
import tissuemix as tm
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):

    def __init__(self, df_path, transform=None, normalize=None, tissuemix=None,
                 probability=0.0, blending=False, warping=False, color=False):
        self.df = pd.read_csv(df_path)
        self.transform = transform
        self.normalize = normalize

        # tissue mix params
        self.tissuemix = tissuemix
        self.probability = probability
        self.blending = blending
        self.warping = warping
        self.color = color

        # extract indices that contain background patch
        self.background = self.df.loc[(self.df['label'] == 0) & (self.df['background'] == 1)].index.values.tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['path']
        mask_path = self.df.iloc[idx]['target']
        image = skimage.io.imread(img_path)
        mask = skimage.io.imread(mask_path)
        mask = mask[:, :, np.newaxis]

        if self.tissuemix and self.df.iloc[idx]['label'] == 1:
            if random.random() >= self.probability:
                bg_idx = random.choice(self.background)
                bg_path = self.df.iloc[bg_idx]['path']
                background = skimage.io.imread(bg_path)
                mask = mask[:, :, 0]

                image, mask = tm.apply_tissuemix(
                    target=image,
                    target_mask=mask,
                    background=background,
                    background_mask=mask,
                    blend=self.blending,
                    color=self.color,
                    warp=self.warping)
                mask = mask[:, :, np.newaxis]

        if self.transform:  # image and mask augmentations
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.normalize:
            if self.normalize == 'simple':
                image = image / 255.
            elif self.normalize == 'imagenet':
                mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
                std = np.array([58.395, 57.120, 57.375], dtype=np.float32)
                image = image.astype(np.float32)
                image = image - mean
                image = image * np.reciprocal(std, dtype=np.float32)
            else:
                image = image / 255.

        # reshape from HWC to CHW format
        image = image.transpose((2, 0, 1)).astype(np.float32)
        mask = mask.transpose((2, 0, 1)).astype(np.float32)

        return image, mask
