# pylint: disable=missing-module-docstring, missing-function-docstring
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description='Estimate lung volume from denoise lung mask experiments. '
        'Measurements are printed to stdout in csv format'
    )
    parser.add_argument('inpath', type=str, nargs='+',
                        help="Path to .npz file with 'mask', 'recon', 'image'"
                        )
    parser.add_argument('--plot-path', type=str, help='Path to store plot', default=None)
    parser.add_argument('--show-plot', action='store_true')
    args = parser.parse_args()

    print('path,mask_volume,recon_volume,bad_mask_volume,recon_mask_ratio,bad_mask_mask_ratio')
    if args.plot_path:
        values = []
    for path in args.inpath:
        loaded = np.load(path)
        mask_volume = (loaded['mask'] > 0).sum()
        #recon_volume = round(loaded['recon'].sum())
        recon_volume = (loaded['recon'] > 0.5).sum()
        bad_mask_volume = (loaded['image'] > 0).sum()
        recon_mask_ratio = round(recon_volume/mask_volume,2)
        bad_mask_mask_ratio = round(bad_mask_volume/mask_volume, 2)
        row =  [
            path,
            mask_volume,
            recon_volume,
            bad_mask_volume,
            recon_mask_ratio,
            bad_mask_mask_ratio,
        ]
        print(*row, sep=',')
        if args.plot_path is not None:
            values.append(row[1:])
            
    if args.plot_path is not None:
        values = np.array(values)
        _, axs = plt.subplots(1,3,figsize=(30,10))
        axs[0].plot(values[:,0], 'b+')
        axs[0].plot(values[:,1], 'rx')
        axs[0].plot(values[:,2], 'go')
        axs[0].legend(('ref', 'recon', 'corrupted'))
        axs[0].set_title('Voxel counts')

        axs[1].plot(values[:,3], 'rx')
        axs[1].legend(('recon/ref',))
        axs[1].set_title('(reconstructed mask volume)/(reference mask volume)')

        axs[2].plot(values[:,4], 'go')
        axs[2].legend(['corrupted/ref'])
        axs[1].set_title('(corrupted mask volume)/(reference mask volume)')
        plt.tight_layout()
        plt.savefig(args.plot_path)
        if args.show_plot:
            plt.show()

    
if __name__ == '__main__':
    main()
