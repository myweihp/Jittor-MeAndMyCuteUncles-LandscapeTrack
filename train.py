# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import jittor as jt
jt.flags.use_cuda = 1
jt.cudnn.set_max_workspace_ratio(0.0)
import jittor.nn as nn
import sys

from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.util import print_current_errors
from trainers.pix2pix_trainer import Pix2PixTrainer

from tensorboardX import SummaryWriter
writer = SummaryWriter('./loss')
import os

opt = TrainOptions().parse()
opt.dataroot = opt.input_path
print(' '.join(sys.argv))

dataloader = data.create_dataloader(opt)
len_dataloader = len(dataloader)

iter_counter = IterationCounter(opt, len(dataloader))

trainer = Pix2PixTrainer(opt, resume_epoch=iter_counter.first_epoch)

save_root = os.path.join(os.path.dirname(opt.checkpoints_dir), 'output', opt.name)
for epoch in iter_counter.training_epochs():
    opt.epoch = epoch
    if not opt.maskmix:
        print('inject nothing')
    elif opt.maskmix and opt.noise_for_mask and epoch > opt.mask_epoch:
        print('inject noise')
    else:
         print('inject mask')
    print('real_reference_probability is :{}'.format(dataloader.real_reference_probability))
    print('hard_reference_probability is :{}'.format(dataloader.hard_reference_probability))
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        # jt.sync_all()
        # jt.gc()
        print(i)
        iter_counter.record_one_iteration()
        #use for Domain adaptation loss
        p = min(float(i + (epoch - 1) * len_dataloader) / 50 / len_dataloader, 1)
        alpha = 2. / (1. + np.exp(-10 * p, dtype="float32")) - 1

        trainer.run_one_step(data_i, alpha=alpha, iter=i)

        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            losses_key= losses.keys()
            losses_value = losses.values()
            for j in range(9):
                mean = jt.mean(list(losses_value)[j])
                writer.add_scalar(list(losses_key)[j], mean.data, iter_counter.total_steps_so_far) 
            try:
                print_current_errors(opt, epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)
            except OSError as err:
                print(err)

        if iter_counter.needs_displaying():
            if not os.path.exists(save_root + opt.name):
                os.makedirs(save_root + opt.name)
            imgs_num = data_i['label'].shape[0]
            if opt.dataset_mode == 'celebahq':
                data_i['label'] = data_i['label'][:,::2,:,:]
            elif opt.dataset_mode == 'celebahqedge':
                data_i['label'] = data_i['label'][:,:1,:,:]
            elif opt.dataset_mode == 'deepfashion':
                data_i['label'] = data_i['label'][:,:3,:,:]
            if data_i['label'].shape[1] == 3:
                label = data_i['label']
            else:
                label = data_i['label'].expand(-1, 3, -1, -1).float() / data_i['label'].max()

            cycleshow = None
            if opt.warp_cycle_w > 0:
                cycleshow = trainer.out['warp_cycle'] if opt.warp_patch else nn.interpolate(trainer.out['warp_cycle'], scale_factor=opt.warp_stride)
                if opt.two_cycle:
                    cycleshow = jt.concat((cycleshow, nn.interpolate(trainer.out['warp_i2r'], scale_factor=opt.warp_stride), nn.interpolate(trainer.out['warp_i2r2i'], scale_factor=opt.warp_stride)), 0)

            if cycleshow is not None:
                imgs = jt.concat((label, data_i['ref'], trainer.out['warp_out'], cycleshow, trainer.get_latest_generated(), data_i['image']), 0)
            else:
                imgs = jt.concat((label, data_i['ref'], trainer.out['warp_out'], trainer.get_latest_generated(), data_i['image']), 0)
            
            try:
                jt.save_image(imgs, save_root + opt.name + '/' + str(epoch) + '_' + str(iter_counter.total_steps_so_far) + '.png',  
                        nrow=imgs_num, padding=0, normalize=True)
            except OSError as err:
                print(err)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                (epoch, iter_counter.total_steps_so_far))
            try:
                trainer.save('latest')
                trainer.save(epoch)
                iter_counter.record_current_iter()
            except OSError as err:
                print(err)

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        try:
            trainer.save('latest')
            trainer.save(epoch)
        except OSError as err:
            print(err)


print('Training was successfully finished.')
