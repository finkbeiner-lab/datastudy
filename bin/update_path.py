#!/opt/conda/bin/python

"""Update path on database after moving from TM
"""
from sql import Database
import os
import datetime
import argparse
import logging


class UpdatePath:
    def __init__(self, opt):
        self.opt = opt
        self.Db = Database()
        self.robofolder = dict(TM='ThinkingMicroscope', ROBO4='Robo4Images',ROBO3='Robo3Images')
        # self.origfolder = dict(TM=['D:/Images', 'X:'], ROBO4=['E:/Images'],ROBO3=['C:/Test']) #Original
        self.origfolder = dict(TM=['D:/Images', 'Z:/ThinkingMicroscope',], ROBO4=['E:/Images'],ROBO3=['C:/Test'])
        # self.targetfolder = {'D:/Images':'/gladstone/finkbeiner/robodata',
        #                      'X:':'/gladstone/finkbeiner/robodata',
        #                      'E:/Images':'/gladstone/finkbeiner/robodata','C:/Test':'/gladstone/finkbeiner/robodata'} #Original
        self.targetfolder = {'D:/Images':'/gladstone/finkbeiner/robodata',
                             'Z:/ThinkingMicroscope':'/gladstone/finkbeiner/robodata',
                             'E:/Images':'/gladstone/finkbeiner/robodata','C:/Test':'/gladstone/finkbeiner/robodata'}

    def build_target_folder(self, src_folder, microscope):
        prefix = self.targetfolder[src_folder]
        # if microscope=='TM' and src_folder in ['X:']: #Original
        if microscope=='TM' and src_folder in ['Z:']:
            return prefix
        else:
            return os.path.join(prefix, self.robofolder[microscope])
        

    def run(self):
        # Add to database
        now = datetime.datetime.now()
        analysisdate = f'{now.year}-{now.month:02}-{now.day:02}'
        microscope = self.Db.get_table_value(
            tablename='experimentdata', column='microscope', kwargs=dict(experiment=self.opt.experiment))
        author = self.Db.get_table_value(
            tablename='experimentdata', column='researcher', kwargs=dict(experiment=self.opt.experiment))
        exp_uuid = self.Db.get_table_uuid(
            tablename='experimentdata', kwargs=dict(experiment=self.opt.experiment))
        print(microscope)
        assert microscope[0][0] in ['TM', 'ROBO4', 'ROBO3','IXM','ROBO5'], 'Experiment is not from Thinking Microscope or Robo4. Check if null or from other microscope.........'
        microscope = microscope[0][0]
        print(f'Updating paths for microscope: {microscope}')
        analysisdir = os.path.join(
            f'/gladstone/finkbeiner/linsley/GXYTMPS/{author[0][0]}/GXYTMP-{self.opt.experiment}')
        imagedir = os.path.join(
            f'/gladstone/finkbeiner/robodata/{self.robofolder[microscope]}/{self.opt.experiment}')
        if not os.path.exists(analysisdir):
            os.makedirs(analysisdir)
        self.Db.update(tablename='experimentdata',
                       update_dct=dict(imagedir=imagedir,
                                       analysisdir=analysisdir,
                                       analysisdate=analysisdate),
                       kwargs=dict(experiment=self.opt.experiment))

        # /gladstone/finkbeiner/robodata/ThinkingMicroscope/20230828-2-msneuron-cry2/F8
        # \\ -> /  Convert windows slashes to unix
        self.Db.update_slashes('tiledata', exp_uuid=exp_uuid)
        for src_folder in self.origfolder[microscope]:
            target_folder = self.build_target_folder(src_folder, microscope)
            
            self.Db.update_prefix_path('tiledata', exp_uuid=exp_uuid, old_string=src_folder,
                                    new_string=target_folder)  #f'/gladstone/finkbeiner/robodata/{self.robofolder[microscope]}'
            print(f'Done: Replaced {src_folder} with {target_folder}')


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
    parser.add_argument(
        '--experiment', default='0907-FB-1-JL-gedi-test', type=str)
    parser.add_argument(
        '--analysisdir', default='gladstone/finkbeiner/linsley/GXYTMPS/Kaushik', help='Analysis directory path')
    args = parser.parse_args()
    print(args)
    Up = UpdatePath(args)
    Up.run()
