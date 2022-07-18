import os
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
class LANDSCAPEVALDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        preprocess_mode = 'resize_and_crop' if is_train else 'none'
        # parser.set_defaults(preprocess_mode=preprocess_mode)
        parser.set_defaults(preprocess_mode=preprocess_mode)
        if is_train:
            # parser.set_defaults(preprocess_mode='resize_and_crop')
            # if is_train:
            #     parser.set_defaults(load_size=286)
            # else:
            parser.set_defaults(load_size=512)
            parser.set_defaults(crop_size=384)
            parser.set_defaults(display_winsize=512)
            parser.set_defaults(label_nc=29)
            parser.set_defaults(contain_dontcare_label=True)
            parser.set_defaults(cache_filelist_read=False)
            parser.set_defaults(cache_filelist_write=False)
        else:
            size = 512
            preprocess_mode = 'scale_width'
            parser.set_defaults(preprocess_mode='scale_width')
            parser.set_defaults(load_size=size)
            parser.set_defaults(crop_size=size)
            parser.set_defaults(aspect_ratio=4/3)
            parser.set_defaults(display_winsize=size)
            parser.set_defaults(label_nc=29)
            parser.set_defaults(contain_dontcare_label=True)

        # parser.set_defaults(load_size=256)
        # parser.set_defaults(crop_size=256)
        # parser.set_defaults(aspect_ratio=4/3)
        # parser.set_defaults(display_winsize=256)
        # parser.set_defaults(label_nc=29)
        # parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        return parser

    def get_paths(self, opt):
        root = "dataset/landscape_dataset_ori_oasis"
        phase = 'val'
        subfolder = 'validation' if opt.phase == 'test' else 'training'
        # subfolder = 'training'
        cache = False if opt.phase == 'test' else True
        all_images = sorted(make_dataset(root + '/' + subfolder, recursive=True, read_cache=cache, write_cache=False))
        image_paths = []
        label_paths = []
        for p in all_images:
            # if '_%s_' % phase not in p:
            #     continue
            if p.endswith('.jpg'):
                image_paths.append(p)
            elif p.endswith('.png'):
                label_paths.append(p)
        # print(label_paths)
        return label_paths, image_paths

    def get_ref(self, opt):
        extra = '_test' if opt.phase == 'test' else ''
        with open('./data/landscape_ref{}.txt'.format(extra)) as fd:
            lines = fd.readlines()
        ref_dict = {}
        for i in range(len(lines)):
            items = lines[i].strip().split(',')
            key = items[0]
            if opt.phase == 'test':
                val = items[1:]
            else:
                val = [items[1], items[-1]]
            ref_dict[key] = val
        train_test_folder = ('training', 'validation')
        return ref_dict, train_test_folder

