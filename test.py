import os
import jittor as jt
import jittor.nn as nn
import pandas as pd
jt.flags.use_cuda = 1
import data
from options.test_options import TestOptions
from models.networks import define_G, define_Corr
from data.base_dataset import get_params, get_transform
from PIL import Image
# from oasis.models.models_org import OASIS_model

#CUDA_VISIBLE_DEVICES=2 python test_direct.py --maskmix --PONO --PONO_C --use_attention

def preprocess_input(opt, data):
    label_map = data['label']
    bs, _, h, w = label_map.size()
    nc = opt.label_nc + 1 if opt.contain_dontcare_label \
        else opt.label_nc
    input_label = jt.zeros([bs, nc, h, w])
    input_semantics = jt.scatter(input_label, 1, label_map, jt.ones(1))
    label_map = data['label_ref']
    label_ref = jt.zeros([bs, nc, h, w])
    ref_semantics = jt.scatter(label_ref, 1, label_map, jt.ones(1))
    
    return data['label'], input_semantics, data['image'], data['self_ref'], data['ref'], data['label_ref'], ref_semantics


opt = TestOptions().parse()
opt.oasis_semantic_nc = 29

# dataloader = data.create_dataloader(opt)
opt.preprocess_mode = 'scale_width'
opt.load_size = 512
opt.crop_size = 512
opt.aspect_ratio = 1.33
opt.label_nc = 29
opt.display_winsize = 512
opt.contain_dontcare_label = True
size = 512



save_root = opt.output_path
if not os.path.exists(save_root):
    os.makedirs(save_root)
def get_label_tensor(opt, path):

    label = Image.open(path)
    params1 = get_params(opt, label.size)
    transform_label = get_transform(opt, params1, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(label)

    return label_tensor, params1


netG = define_G(opt)
netCorr = define_Corr(opt)

netG.load('ckpt/182_net_G.pkl')
netCorr.load('ckpt/182_net_Corr.pkl')
netG.eval()
netCorr.eval()

f = open('./ref_cos.txt', 'r')
ref_dict = {}
for line in f:
    a = line.strip().split(',')
    ref_dict[a[0]] = a[1]


for fname, refname in ref_dict.items():
    input_semantic, params1 = get_label_tensor(opt, os.path.join(opt.input_path,fname.replace('jpg','png')))

    # ref_dict = get_reference_data()
    ref_image_fname = refname
    dataroot = './ref_imgs/'
    ref_image_fpath = dataroot + ref_image_fname
    ref_semantic_fpath = dataroot + refname.replace('jpg','png')
    ref_semantic, params = get_label_tensor(opt, ref_semantic_fpath)
    ref_image = Image.open(ref_image_fpath).convert('RGB')
    transform_image = get_transform(opt, params)
    ref_image = transform_image(ref_image)
    
    input_label = jt.Var(input_semantic).unsqueeze(0)
    ref_image = jt.Var(ref_image).unsqueeze(0)
    ref_label = jt.Var(ref_semantic).unsqueeze(0)

    # process input and ref label into semantic maps
    label_map = input_label
    bs, _, h, w = label_map.size()
    nc = opt.label_nc + 1 if opt.contain_dontcare_label \
        else opt.label_nc
    zero_label = jt.zeros([bs, nc, h, w])
    input_semantics = jt.scatter(zero_label, 1, label_map, jt.ones(1))
    label_map = ref_label
    zero_ref = jt.zeros([bs, nc, h, w])
    ref_semantics = jt.scatter(zero_ref, 1, label_map, jt.ones(1))

    with jt.no_grad():
        generate_out = {}
        coor_out = netCorr(ref_image, None, input_semantics, ref_semantics, alpha=1)

        if opt.CBN_intype == 'mask':
            CBN_in = input_semantics
        elif opt.CBN_intype == 'warp':
            CBN_in = coor_out['warp_out']
        elif opt.CBN_intype == 'warp_mask':
            CBN_in = jt.concat((coor_out['warp_out'], input_semantics), dim=1)

        generate_out['fake_image'] = netG(input_semantics, warp_out=CBN_in)
        out = {**generate_out, **coor_out}
        out['input_semantics'] = input_semantics
        out['ref_semantics'] = ref_semantics
        print(os.path.join(save_root,fname))
        jt.save_image(generate_out['fake_image'].squeeze(0), os.path.join(save_root,fname),  
                    padding=0, normalize=True)
