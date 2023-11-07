


"""Montage images or masks"""
import argparse
from normalization import Normalize
import datetime
import logging
import numpy as np
import os
import imageio

logger = logging.getLogger("Montage")
# logger.propagate = False
now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
print('Timestamp', TIMESTAMP)
fink_log_dir = './finkbeiner_logs'
if not os.path.exists(fink_log_dir):
    os.makedirs(fink_log_dir)
logname = os.path.join(fink_log_dir, f'Montage-log_{TIMESTAMP}.log')
fh = logging.FileHandler(logname)
# fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.warning('Running Montage from Database.')


class Montage:
    def __init__(self, opt):
        self.opt = opt
        self.Norm = Normalize(self.opt)
        _, self.analysisdir = self.Norm.get_raw_and_analysis_dir()
        self.montage_folder_name = 'MontagedImages'
        self.montagedir = os.path.join(self.analysisdir, self.montage_folder_name)

    def run(self, savebool=True):
        tiledata_df = self.Norm.get_flatfields()
        tiledata_df = tiledata_df.sort_values(by=['timepoint', 'well', 'tile'])

        groups = tiledata_df.groupby(by=['timepoint', 'well'])
        for name, df in groups:
            self.single_montage(df)
        with open(self.opt.outfile, 'w') as f:
            f.write(f'Montaged Images.')
        print('Done.')

    def single_montage(self, df, savebool=True):
        """
                # Get Robo0/3/4 montage order indexes relative to
        # regular left to right and top to bottom order.
        # like
        # 3 2 1                 1 2 3
        # 4 5 6     relative to 4 5 6
        # 9 8 7                 7 8 9
        # But here we start from 0
        """
        images = []
        savepath = None
        mont = None
        overlap = 0  # TODO: use overlap?
        logger.warning(f'Length of df: {len(df)} and max tile: {df.tile.max()}')
        print(f'Length of df: {len(df)} and max tile: {df.tile.max()}')
        df = df.sort_values('tile')
        if len(df) == df.tile.max():
            for i, row in df.iterrows():
                f = row[self.opt.tiletype]
                # overlap = row.overlap
                if not savepath:
                    name = os.path.basename(f)
                    name = name.split('.t')[0] + '_MONTAGE.tif'
                    welldir = os.path.join(self.montagedir, row.well)
                    if not os.path.exists(welldir):
                        os.makedirs(welldir)
                    savepath = os.path.join(welldir, name)
                img = imageio.v3.imread(f)
                cleaned_im = self.Norm.image_correction[self.opt.img_norm_name](img, row.tile)
                images.append(cleaned_im)
        num_tiles = len(images)
        print(num_tiles)
        logger.warning(f'Num tiles: {num_tiles}')
        side = int(np.sqrt(num_tiles))
        if num_tiles:
            h, w = np.shape(images[0])
            mont = np.zeros((int(h * side), int(w * side)), dtype=np.uint16)
            for i in range(side):
                for j in range(side):
                    #TODO: map montages for legacy montage, new montages, and ixm montages
                    if self.opt.montage_pattern == 'legacy':
                        if i%2==0:
                            k = side - (j+1)
                        else:
                            k = j
                    else:
                        k = j
                    mont[i * h:(i + 1) * h, j * w:(j + 1) * w] = images[i * side + k]
            if savebool:
                imageio.v3.imwrite(savepath, mont)
        return mont


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dict',
        help='path to pickle',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp.pkl'
    )
    parser.add_argument(
        '--outfile',
        help='Text status',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp_output.txt'
    )
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--tiletype', choices=['filename', 'maskpath', 'trackedmaskpath'], type=str,
                        help='Montage image, binary mask, or tracked mask.')
    parser.add_argument('--img_norm_name', choices=['division', 'subtraction', 'identity'], type=str,
                        help='Image normalization method using flatfield image.')
    parser.add_argument('--montage_pattern', choices=['standard', 'legacy'], help="Montage snaking with 3 2 1 4 5 6 9 8 7 pattern.")
    parser.add_argument("--wells_toggle",
                        help="Chose whether to include or exclude specified wells.")
    parser.add_argument("--timepoints_toggle",
                        help="Chose whether to include or exclude specified timepoints.")
    parser.add_argument("--channels_toggle", default='include',
                        help="Chose whether to include or exclude specified channels.")
    parser.add_argument("--chosen_wells", "-cw",
                        dest="chosen_wells", default='',
                        help="Specify wells to include or exclude")
    parser.add_argument("--chosen_timepoints", "-ct",
                        dest="chosen_timepoints", default='',
                        help="Specify timepoints to include or exclude.")
    parser.add_argument("--chosen_channels", "-cc",
                        dest="chosen_channels",
                        help="Specify channels to include or exclude.")
    parser.add_argument('--tile', default=0, type=int, help="Select single tile to segment. Default is to segment all tiles.")
    args = parser.parse_args()
    print(args)
    Mt = Montage(args)
    Mt.run()
