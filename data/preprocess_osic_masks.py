'''Preprocess lung masks from OSIC Fibrosis Progression challenge.
See data_modules.chest.OSICLungHeartTracheaDataModule for details about the data
'''
import os
from data_modules.chest import OSICLungHeartTracheaDataModule
import torchio as tio
import torch

def main():
    '''Produces 2.5mm and 5mm isotropic volumes'''
    data_dir = 'osic_fibrosis_masks'
    data_info = f'{data_dir}/data-info.csv'
    transforms_all = {
        '2.5mm' : tio.Compose([
            tio.transforms.ToCanonical(),
            tio.transforms.Resample((2.5,2.5,2.5)),
            tio.transforms.CropOrPad((140,140,120)),
            tio.transforms.Clamp(0, 1)
        ]),
        '5mm' : tio.Compose([
            tio.transforms.ToCanonical(),
            tio.transforms.Resample((5,5,5)),
            tio.transforms.CropOrPad((70,70,60)),
            tio.transforms.Clamp(0, 1)
        ])
    }
    download = True
    for resolution, transforms in transforms_all.items():
        transforms = {
            'train' : transforms,
            'validation' : transforms,
            'test' : transforms,
        }
        data_module = OSICLungHeartTracheaDataModule(
            data_dir,
            data_info,
            batch_size=1,
            download=download,
            transforms=transforms,
            use_anatomies=['lung']
        )
        data_module.prepare_data()
        data_module.setup()
        # We only need to do the download/unpack part once and it takes some time
        download = False

        for ds_name, loader in [('train', data_module.train_dataloader()),
                                ('validation', data_module.val_dataloader()),
                                ('test', data_module.test_dataloader())]:
            out_dir = os.path.join(data_dir, 'preprocessed', f'lung-{resolution}', ds_name)
            os.makedirs(out_dir, exist_ok=True)
            for i, sample in enumerate(loader):
                if sample['lung-valid']:
                    lung = sample['lung']['data'][0]
                    out_path = os.path.join(out_dir, f'{i:03}.pt')
                    torch.save(lung, out_path)
                else:
                    print(sample)

if __name__ == '__main__':
    main()
