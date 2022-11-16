'''See class LungDataModule'''
import os
import csv
import glob

import torch
import torch.nn.functional as F
import torchio as tio

from data_modules.base_data_module import BaseDataModule

__all__ = [
    'LungDataModule'
]

class LungDataModule(BaseDataModule):
    '''Data module for loading preprocessed lung segmentations.
    '''
    # pylint: disable=too-many-instance-attributes
    def __init__(self,
                 data_dir,
                 data_info_path,
                 batch_size,
                 padding,
                 **kwargs,
                 ):
        '''
        Parameters
        ----------
        data_dir : str
          The root data directory. It is expected that it contains the following sub directories
          train/
          validation/
          test/

        data_info_path : str
          EITHER
           an existing csv file containing *at least* the following named columns
             dataset,path
           all other columns are ignored.
           The dataset column must contain values from {train, validation, test, predict}.
          OR
           a path to store data info in.
          If the file exists it will be used, if it does not exist it will be generated.

        batch_size : int
          Batch size

        padding : None or whatever torch.functional.pad accepts
          If not None torch.functional.pad is used to pad images
          
        '''
        # pylint: disable=too-many-arguments
        super().__init__(data_info_path, batch_size, **kwargs)
        self.data_dir = data_dir
        self.padding = padding

    def create_subject(self, row):
        if self.padding is None:
            lung = torch.load(row.path).to(dtype=torch.float32)
        else:
            lung = F.pad(torch.load(row.path).to(dtype=torch.float32), self.padding)
        subject_kwargs = {
            'image' : tio.ScalarImage(tensor=lung),
            'mask' : tio.LabelMap(tensor=lung)
        }
        return tio.Subject(**subject_kwargs)

    def prepare_data(self):
        '''Do the following
        1. Create data info file
        '''
        if not os.path.exists(self.data_info_path):
            self._create_data_info()

    def _create_data_info(self):
        header = ['dataset', 'path']
        rows = []
        for dataset in ('train', 'validation', 'test'):
            for path in glob.glob(os.path.join(self.data_dir, dataset, '*.pt')):
                rows.append((dataset, path))

        with open(self.data_info_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)
            writer.writerows(rows)
