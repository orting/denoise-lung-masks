# pylint: disable=missing-module-docstring, missing-function-docstring
import argparse
import os

import numpy as np
import torch

from data_modules.transformations import RandomMask
from denoise_lung_masks import LungDataModule, AutoEncoderWrapper
from .parameters import get_params

def main():
    # pylint: disable=too-many-locals
    parser = argparse.ArgumentParser()
    parser.add_argument('model_checkpoint', type=str)
    parser.add_argument('outdir', type=str)
    parser.add_argument('version', type=int, choices=(0,1,2))
    parser.add_argument('--with-corruptions', action='store_true')
    args = parser.parse_args()

    for ds_name in ('train', 'validation', 'test'):
        os.makedirs(os.path.join(args.outdir, ds_name), exist_ok=True)

    params = get_params(args.version)
    if args.with_corruptions:
        transforms = {
            'train' : RandomMask(params['mask_sampler'], prob=1.0),
            'validation' : RandomMask(params['mask_sampler'], prob=1.0),
            'test' : RandomMask(params['mask_sampler'], prob=1.0)
        }
    else:
        transforms = None

    # pylint: disable=duplicate-code
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

    model = AutoEncoderWrapper.load_from_checkpoint(args.model_checkpoint)
    model.eval()
    loaders = {
        'train' : data_module.train_dataloader(False), #pylint: disable=too-many-function-args
        'validation' : data_module.val_dataloader(),
        'test' : data_module.test_dataloader()
    }
    with torch.no_grad():
        for ds_name, loader in loaders.items():
            offset = 0
            for batch in loader:
                recons, masks, images = model.predict_step(batch)
                for im_idx in range(recons.shape[0]):
                    outpath = os.path.join(args.outdir, ds_name, f'{offset+im_idx:03d}.npz')
                    np.savez(
                        outpath,
                        recon=recons[im_idx].squeeze().numpy(),
                        mask=masks[im_idx].squeeze().numpy().astype('uint8'),
                        image=images[im_idx].squeeze().numpy()
                    )
                offset += recons.shape[0]


if __name__ == '__main__':
    main()
