# Denoise lung masks
Experiments with an autoencoder to reconstruct corrupted 3d lung mask.
Masks are corrupted to resemble failure to segment high density pathologies

## Data
Run `data/preprocess_osic_masks.py` to unpack and preprocess the lung masks. It will produce both 5mm and 2.5mm lung masks.
If the data archive is not already downloaded to the `data/osic_fibrosis_masks/` directory, the script will print instructions for downloading.

Note that the preprocessing will take some time.

A data-info file defining dataset splits is provided in `osic_fibrosis_masks/data-info.csv`. If you wish to run experiments with different data splits, either delete the file or change the path in `preprocess_osic_masks.main.data_info`.
