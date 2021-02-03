import os
import yaml
import torch
import shutil
import neptune
import argparse
import skimage.io
import numpy as np
import matplotlib.pyplot as plt

from monai.losses import DiceLoss, GeneralizedDiceLoss
from monai.metrics import DiceMetric, compute_roc_auc

from segmentation_models_pytorch import Unet, UnetPlusPlus, DeepLabV3Plus
from segmentation_models_pytorch.utils.metrics import Fscore, IoU, Accuracy

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.optim import Adam, AdamW, RMSprop
from torch.nn import BCELoss

# from data import class_labels
from data.dataset import SegmentationDataset
from data.transform import load_train_transform, load_valid_transform
from utils.helpers import seed_everything, save_predictions, count_elements
from trainer.trainer import Trainer


def main(cfg):
    """Runs main training procedure."""

    print('Starting training...')
    print('Current working directory is:', os.getcwd())

    # fix random seeds for reproducibility
    seed_everything(seed=cfg['seed'])

    # neptune logging
    neptune.init(
        project_qualified_name=cfg['neptune_project_name'],
        api_token=cfg['neptune_api_token'])

    neptune.create_experiment(
        name=cfg['neptune_experiment'],
        params=cfg)

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

    # configure loss
    losses = {
        'dice_loss': DiceLoss(include_background=background, softmax=False, batch=cfg['combine']),
        'generalized_dice': GeneralizedDiceLoss(include_background=background, softmax=False, batch=cfg['combine'])}

    assert cfg['loss'] in losses.keys()
    loss = losses[cfg['loss']]

    # configure optimizer
    optimizers = {
        'adam': Adam([dict(params=model.parameters(), lr=cfg['lr'])]),
        'adamw': AdamW([dict(params=model.parameters(), lr=cfg['lr'])]),
        'rmsprop': RMSprop([dict(params=model.parameters(), lr=cfg['lr'])])}

    assert cfg['optimizer'] in optimizers.keys()
    optimizer = optimizers[cfg['optimizer']]

    # configure metrics
    metrics = {
        'dice_score': DiceMetric(include_background=background, reduction='mean'),
        'dice_smp': Fscore(threshold=cfg['rounding'], ignore_channels=cfg['ignore_channels']),
        'iou_smp': IoU(threshold=cfg['rounding'], ignore_channels=cfg['ignore_channels']),
        'generalized_dice': GeneralizedDiceLoss(include_background=background, softmax=False, batch=cfg['combine']),
        'dice_loss': DiceLoss(include_background=background, softmax=False, batch=cfg['combine']),
        'cross_entropy': BCELoss(reduction='mean'),
        'accuracy': Accuracy(ignore_channels=cfg['ignore_channels'])}

    assert all(m['name'] in metrics.keys() for m in cfg['metrics'])
    metrics = [(metrics[m['name']], m['name'], m['type']) for m in cfg['metrics']]  # tuple of (metric, name, type)

    # configure scheduler
    schedulers = {
        'steplr': StepLR(optimizer, step_size=cfg['step_size'], gamma=0.5),
        'cosine': CosineAnnealingLR(optimizer, cfg['epochs'], eta_min=cfg['eta_min'], last_epoch=-1)}

    assert cfg['scheduler'] in schedulers.keys()
    scheduler = schedulers[cfg['scheduler']]

    # configure augmentations
    train_transform = load_train_transform(transform_type=cfg['transform'], patch_size=cfg['patch_size'])
    valid_transform = load_valid_transform(patch_size=cfg['patch_size'])

    train_dataset = SegmentationDataset(
        df_path=cfg['train_data'],
        transform=train_transform,
        normalize=cfg['normalize'],
        tissuemix=cfg['tissuemix'],
        probability=cfg['probability'],
        blending=cfg['blending'],
        warping=cfg['warping'],
        color=cfg['color'])

    valid_dataset = SegmentationDataset(
        df_path=cfg['valid_data'],
        transform=valid_transform,
        normalize=cfg['normalize'])

    # save intermediate augmentations
    if cfg['eval_dir']:
        default_dataset = SegmentationDataset(
            df_path=cfg['train_data'],
            transform=None,
            normalize=None)

        transform_dataset = SegmentationDataset(
            df_path=cfg['train_data'],
            transform=None,
            normalize=None,
            tissuemix=cfg['tissuemix'],
            probability=cfg['probability'],
            blending=cfg['blending'],
            warping=cfg['warping'],
            color=cfg['color'])

        for idx in range(0, min(500, len(default_dataset)), 10):
            image_input, image_mask = default_dataset[idx]
            image_input = image_input.transpose((1, 2, 0))
            image_input = image_input.astype(np.uint8)

            image_mask = image_mask.transpose(1, 2, 0)  # Why do we need transpose here?
            image_mask = image_mask.astype(np.uint8)
            image_mask = image_mask.squeeze()
            image_mask = image_mask * 255

            image_transform, _ = transform_dataset[idx]
            image_transform = image_transform.transpose((1, 2, 0)).astype(np.uint8)

            idx_str = str(idx).zfill(3)
            skimage.io.imsave(os.path.join(cfg['eval_dir'], f'{idx_str}a_image_input.png'),
                              image_input, check_contrast=False)
            plt.imsave(os.path.join(cfg['eval_dir'], f'{idx_str}b_image_mask.png'),
                       image_mask, vmin=0, vmax=1)
            skimage.io.imsave(os.path.join(cfg['eval_dir'], f'{idx_str}c_image_transform.png'),
                              image_transform, check_contrast=False)

        del transform_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        num_workers=cfg['workers'],
        shuffle=True)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg['batch_size'],
        num_workers=cfg['workers'],
        shuffle=False)

    trainer = Trainer(
        model=model,
        device=cfg['device'],
        save_checkpoints=cfg['save_checkpoints'],
        checkpoint_dir=cfg['checkpoint_dir'],
        checkpoint_name=cfg['checkpoint_name'])

    trainer.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        num_classes=num_classes)

    trainer.fit(
        train_loader,
        valid_loader,
        epochs=cfg['epochs'],
        scheduler=scheduler,
        verbose=cfg['verbose'],
        loss_weight=cfg['loss_weight'])

    # validation inference
    best_model = model
    best_model.load_state_dict(torch.load(os.path.join(cfg['checkpoint_dir'], cfg['checkpoint_name'])))
    best_model.to(cfg['device'])
    best_model.eval()

    # setup directory to save plots
    if os.path.isdir(cfg['plot_dir']):
        # remove existing dir and content
        shutil.rmtree(cfg['plot_dir'])
    # create absolute destination
    os.makedirs(cfg['plot_dir'])

    # valid dataset without transformations and normalization for image visualization
    valid_dataset_vis = SegmentationDataset(
        df_path=cfg['valid_data'],
        transform=valid_transform,
        normalize=None)

    if cfg['save_checkpoints']:
        for n in range(len(valid_dataset)):
            image_vis = valid_dataset_vis[n][0].astype('uint8')
            image_vis = image_vis.transpose((1, 2, 0))

            image, gt_mask = valid_dataset[n]
            gt_mask = gt_mask.transpose((1, 2, 0))
            gt_mask = gt_mask.squeeze()

            x_tensor = torch.from_numpy(image).to(cfg['device']).unsqueeze(0)
            pr_mask, _ = best_model.predict(x_tensor)
            pr_mask = pr_mask.cpu().numpy().round()
            pr_mask = pr_mask.squeeze()

            save_predictions(
                out_path=cfg['plot_dir'],
                index=n,
                image=image_vis,
                ground_truth_mask=gt_mask,
                predicted_mask=pr_mask,
                average='macro')


if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser(description='Train segmentation model.')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path')
    args = parser.parse_args()

    # load config file
    with open(args.config, 'r') as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # run training process
    main(cfg=config)
