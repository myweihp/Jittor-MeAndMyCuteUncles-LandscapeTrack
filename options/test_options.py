"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from ast import parse
from email.policy import default
from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        parser.add_argument('--save_per_img', action='store_true', help='if specified, save per image')
        parser.add_argument('--show_corr', action='store_true', help='if specified, save bilinear upsample correspondence')
        parser.add_argument('--outname', type=str, default='landscape2', help='saves results name')

        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=512, load_size=512, display_winsize=512)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')

        # parser.add_argument('--input_path', type=str, default='./dataset/val_B', help='input semantic image path')
        parser.add_argument('--output_path', type=str, default='./results', help='output landscape image path')

        parser.add_argument('--oasis_name', type=str, default='label2coco', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--seed', type=int, default=42, help='random seed')
        # parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--oasis_checkpoints_dir', type=str, default='./oasis/checkpoints', help='oasis models are saved here')
        parser.add_argument('--no_spectral_norm', action='store_true', help='this option deactivates spectral norm in all layers')
        # parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        # parser.add_argument('--dataroot', type=str, default='./datasets/cityscapes/', help='path to dataset root')
        # parser.add_argument('--dataset_mode', type=str, default='coco', help='this option indicates which dataset should be loaded')
        # parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
        parser.add_argument('--num_res_blocks', type=int, default=6, help='number of residual blocks in G and D')
        parser.add_argument('--channels_G', type=int, default=64, help='# of gen filters in first conv layer in generator')
        parser.add_argument('--param_free_norm', type=str, default='batch', help='which norm to use in generator before SPADE')
        parser.add_argument('--spade_ks', type=int, default=3, help='kernel size of convs inside SPADE')
        parser.add_argument('--no_EMA', action='store_true', help='if specified, do *not* compute exponential moving averages')
        parser.add_argument('--EMA_decay', type=float, default=0.9999, help='decay in exponential moving averages')
        parser.add_argument('--no_3dnoise', action='store_true', default=False, help='if specified, do *not* concatenate noise to label maps')
        parser.add_argument('--z_dim', type=int, default=64, help='dimension of the latent z vector')
        parser.add_argument('--oasis_ckpt_iter', type=str, default='best', help='which epoch to load to evaluate a model')
        self.isTrain = False
        return parser
