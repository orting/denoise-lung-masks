# pylint: disable=missing-module-docstring, missing-function-docstring
import argparse
from datetime import datetime

import torchio as tio
import pytorch_lightning as pl
from monai.networks.nets import AutoEncoder
from monai import losses

from data_modules.transformations import RandomMask
from denoise_lung_masks import LungDataModule, AutoEncoderWrapper
from parameters import get_params

def main():
    description = '''Train an autoencoder to reconstruct a corrupted 3d lung mask.
    Masks are corrupted to resemble failure to segment high density pathologies.
    '''
    version_help = '''Three versions of the experiments.
    0 : Use 5mm isotropic resolution and always corrupt mask.
    1 : Use 2.5mm isotropic resolution and always corrupt.
    2 : Use 2.5mm isotropic resolution and corrupt 3/4.
    '''
    parser = argparse.ArgumentParser(description = description)
    parser.add_argument('version', type=int, choices=(0,1,2), help=version_help)
    args = parser.parse_args()

    params = get_params(args.version)
    transforms = {
        'train' : tio.Compose([
            RandomMask(params['mask_sampler'], prob=params['random_mask_prob']),
            tio.RandomAffine(
                scales=0.25,
                degrees=15,
                translation=20,
                image_interpolation='nearest'
            )
        ])
    }
    data_module = LungDataModule(
        params['data_dir'],
        params['data_info_file'],
        params['batch_size'],
        padding=params['padding'],
        transforms=transforms,
        num_workers=16,
        extra_data_loader_kwargs={'pin_memory' : True}
    )
    data_module.prepare_data()
    data_module.setup()

    net = AutoEncoder(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 32, 64, 64, 128, 128),
        strides=(  1,  2,  1,  2,  1,   2,   1),
    )
    loss_function = losses.DiceLoss()
    model = AutoEncoderWrapper(net, loss_function , learning_rate=1e-4, use_sigmoid=True)

    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        every_n_epochs=10,
        save_top_k=2,
        filename='{epoch}-{step}-{train_loss:.2f}-{val_loss:.2f}'
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=-1,
        auto_select_gpus=True,
        log_every_n_steps=2,
        callbacks=[model_checkpoint],
        max_epochs=500,
    )

    start = datetime.now()
    print('Training started at', start)
    trainer.fit(model=model, datamodule=data_module)
    print('Training duration:', datetime.now() - start)

if __name__ == '__main__':
    main()
