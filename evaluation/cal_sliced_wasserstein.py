import os
import time
import re
import bisect
from collections import OrderedDict
import numpy as np
import scipy.ndimage
import scipy.misc
import importlib
from PIL import Image
import sys
#import config


#----------------------------------------------------------------------------
# Evaluate one or more metrics for a previous training run.
# To run, uncomment one of the appropriate lines in config.py and launch train.py.
def import_module(module_or_obj_name):
    parts = module_or_obj_name.split('.')
    parts[0] = {'np': 'numpy', 'tf': 'tensorflow'}.get(parts[0], parts[0])
    for i in range(len(parts), 0, -1):
        try:
            module = importlib.import_module('.'.join(parts[:i]))
            relative_obj_name = '.'.join(parts[i:])
            return module, relative_obj_name
        except:
            pass
    raise ImportError(module_or_obj_name)

def find_obj_in_module(module, relative_obj_name):
    obj = module
    for part in relative_obj_name.split('.'):
        obj = getattr(obj, part)
    return obj

def import_obj(obj_name):
    module, relative_obj_name = import_module(obj_name)
    return find_obj_in_module(module, relative_obj_name)

def evaluate_metrics(fake_imgs, real_imgs, real_passes, minibatch_size=20, metrics=['swd']):
    metric_class_names = {
        'swd':      'sliced_wasserstein.API',
        'fid':      'frechet_inception_distance.API',
        'is':       'inception_score.API',
        'msssim':   'ms_ssim.API',
    }

    assert fake_imgs.shape[0] == real_imgs.shape[0]
    num_images = fake_imgs.shape[0]
    # Initialize metrics.
    metric_objs = []
    for name in metrics:
        class_name = metric_class_names.get(name, name)
        print('Initializing %s...' % class_name)
        class_def = import_obj(class_name)
        image_shape = [3] + [256, 256]
        obj = class_def(num_images=num_images, image_shape=image_shape, image_dtype=np.uint8, minibatch_size=minibatch_size)
        mode = 'warmup'
        obj.begin(mode)
        for idx in range(10):
            obj.feed(mode, np.random.randint(0, 256, size=[minibatch_size]+image_shape, dtype=np.uint8))
        obj.end(mode)
        metric_objs.append(obj)

    # Print table header.
    print()
    print('%-10s%-12s' % ('Snapshot', 'Time_eval'), end='')
    for obj in metric_objs:
        for name, fmt in zip(obj.get_metric_names(), obj.get_metric_formatting()):
            print('%-*s' % (len(fmt % 0), name), end='')
    print()
    print('%-10s%-12s' % ('---', '---'), end='')
    for obj in metric_objs:
        for fmt in obj.get_metric_formatting():
            print('%-*s' % (len(fmt % 0), '---'), end='')
    print()

    # Feed in reals.
    for title, mode in [('Reals', 'reals'), ('Reals2', 'fakes')][:real_passes]:
        print('%-10s' % title, end='')
        time_begin = time.time()
        [obj.begin(mode) for obj in metric_objs]
        for begin in range(0, num_images, minibatch_size):
            end = min(begin + minibatch_size, num_images)
            if mode == 'fakes':
                images = fake_imgs[begin:end]
            else:
                images = real_imgs[begin:end]
            if images.shape[1] == 1:
                images = np.tile(images, [1, 3, 1, 1]) # grayscale => RGB
            [obj.feed(mode, images) for obj in metric_objs]
        results = [obj.end(mode) for obj in metric_objs]
        print('------')
        for obj, vals in zip(metric_objs, results):
            for val, fmt in zip(vals, obj.get_metric_formatting()):
                print(fmt % val, end='')
        print()

def get_image(folder):
    files = os.listdir(folder)
    imgs_path = [it for it in files if (it.endswith('.jpg') or it.endswith('.png'))]
    print('load {} imgs'.format(len(imgs_path)))
    img_list = []
    for path in imgs_path:
        img = Image.open(os.path.join(folder, path))
        img_list.append(np.array(img)[np.newaxis, :].transpose(0,3,1,2))
    imgs = np.concatenate(img_list, axis=0)
    return imgs

fake_imgs = get_image(sys.argv[1])
real_imgs = get_image(sys.argv[2])
#real_imgs = real_imgs[:len(fake_imgs)]
evaluate_metrics(fake_imgs, real_imgs, 2, minibatch_size=20, metrics=['swd'])

#python cal_sliced_wasserstein.py /mnt/blob/Output/SPADE_Exemplar/output/test_per_img/ade20k_exemplar_stage3_InoiseCwarpmask_perc0.01_attn_baseline2 /mnt/blob/Dataset/ADEChallengeData2016/fid_256/