# pylint: disable=missing-module-docstring, missing-function-docstring
import argparse

import numpy as np
import napari


def main():
    parser = argparse.ArgumentParser(
        description='Visualize predictions from denoising experiments using napari'
    )
    parser.add_argument('inpath', type=str, nargs='+')
    args = parser.parse_args()

    viewer = napari.Viewer()
    for i, path in enumerate(args.inpath):
        loaded = np.load(path)
        viewer.add_image(loaded['recon'], name=f'recon-{i}')
        if 'seg' in loaded:
            viewer.add_image(loaded['seg'], name=f'seg-{i}')
        if 'image' in loaded:
            viewer.add_labels(loaded['image'].astype('uint8'), name=f'image-{i}', color={1:'teal'})
        viewer.add_labels(loaded['mask'].astype('uint8'), name=f'mask-{i}')

    napari.run()
    
if __name__ == '__main__':
    main()
