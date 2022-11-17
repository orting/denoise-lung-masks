# Denoise lung masks
Experiments with an autoencoder to reconstruct corrupted 3d lung mask.
Masks are corrupted to resemble failure to segment high density pathologies

## Data
Run `data/preprocess_osic_masks.py` to unpack and preprocess the lung masks. It will produce both 5mm and 2.5mm lung masks.
If the data archive is not already downloaded to the `data/osic_fibrosis_masks/` directory, the script will print instructions for downloading.

Note that the preprocessing will take some time.

A data-info file defining dataset splits is provided in `osic_fibrosis_masks/data-info.csv`. If you wish to run experiments with different data splits, either delete the file or change the path in `preprocess_osic_masks.main.data_info`.


## Prepare module
The files in `src/denoise_lung_masks` are needed for the experiments. Either run `reinstall_package.sh` to install `denoise_lung_masks` as a python package, or create a symlink to `src/denoise_lung_masks` in the experiment directory.

## Run experiments
Experiments are in `experiments/`. 

### Denoising autoencoder
Directory `experiments/denoising-autoencoder`.

Train an autoencoder to reconstruct a corrupted 3d lung mask. Masks are corrupted to resemble failure to segment high density pathologies.

There are three versions of the experiments.

| Name      | Description                                           |
|-----------|-------------------------------------------------------|
| Version 0 | Use 5mm isotropic resolution and always corrupt mask. |
| Version 1 | Use 2.5mm isotropic resolution and always corrupt.    |
| Version 2 | Use 2.5mm isotropic resolution and corrupt 3/4.       |


Parameters for each version are stored in `parameters.py`. Adjust `batch_size` as needed, version 1 and 2 requires around 20MB GPU RAM.


### Train
Train each version with

	python train.py <version-number>
	
The two models with lowest validation loss are kept.

Approximate runtime on RTX3090

| Name      | Approximate wall clock time |
|-----------|-------------------------------------------------------|
| Version 0 | 11 min :|
| Version 1 | :|
| Version 2 | :|



### Predict
Predict each version with

	python predict.py <model-checkpoint> <outdir> <version-number> [--with-corruptions]
	
The flag `--with-corruptions` will enable data corruption on all samples before prediction.

