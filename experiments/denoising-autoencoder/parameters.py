'''Parameters for denoising-autoencoder experiment'''
from data_modules.transformations import MotherOfGaussianSpheres

def get_params(version):
    '''version should be in (0,1,2)'''
    assert version in (0,1,2)
    if version == 0:
        params = {
            'data_dir' : '../../data/osic_fibrosis_masks/preprocessed/lung-5mm/',
            'data_info_file' : 'ae-data-info-5mm.csv',
            'batch_size' : 16,
            'mask_sampler' : MotherOfGaussianSpheres(
                num_clusters = 5,
                num_samples = 100,
                dilation_radius = (3,5),
                sigma = 10,
                holes_as_background = True
            ),
            'padding' : (2,2,1,1,1,1),
            'random_mask_prob' : 1,
        }
    else:
        params = {
            'data_dir' : '../../data/osic_fibrosis_masks/preprocessed/lung-2.5mm/',
            'data_info_file' : 'ae-data-info-2.5mm.csv',
            'batch_size' : 8,
            'mask_sampler' : MotherOfGaussianSpheres(
                num_clusters = 40,
                num_samples = 800,
                dilation_radius = (6,10),
                sigma = 20,
                holes_as_background = True
            ),
            'padding' : (0,0,2,2,2,2),
        }
        if version == 1:
            params['random_mask_prob'] = 1
        else:
            params['random_mask_prob'] = 0.75
            
    return params
