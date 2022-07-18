# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import jittor as jt
from models.pix2pix_model import Pix2PixModel
from models.networks.generator import EMA
import util.util as util


class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt, resume_epoch=0):
        self.opt = opt
        self.pix2pix_model = Pix2PixModel(opt)
 
            
        if opt.use_ema:
            self.netG_ema = EMA(opt.ema_beta)
            for name, param in self.pix2pix_model.netG.named_parameters():
                if param.requires_grad:
                    self.netG_ema.register(name, param)
            self.netCorr_ema = EMA(opt.ema_beta)
            for name, param in self.pix2pix_model.net['netCorr'].named_parameters():
                if param.requires_grad:
                    self.netCorr_ema.register(name, param)

        self.generated = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.pix2pix_model.create_optimizers(opt)
            self.old_lr = opt.lr
            if opt.continue_train and opt.which_epoch == 'latest':
                checkpoint = jt.load(os.path.join(opt.checkpoints_dir, opt.name, 'optimizer.pkl'))
                self.optimizer_G.load_state_dict(checkpoint['G'])
                self.optimizer_D.load_state_dict(checkpoint['D'])
        self.last_data, self.last_netCorr, self.last_netG, self.last_optimizer_G = None, None, None, None

    def run_one_step(self, data, alpha=1, iter=0):

        input_label, input_semantics, real_image, self_ref, ref_image, ref_label, ref_semantics = self.pix2pix_model.preprocess_input(data, )

        # training generator
        if iter % self.opt.D_steps_per_G == 0:
            # self.optimizer_G.zero_grad()
            # self.pix2pix_model.set_grad_D(False)
            # self.pix2pix_model.set_grad_G(True)
            g_losses, out = self.pix2pix_model(input_label, input_semantics, real_image, self_ref, ref_image, ref_label, ref_semantics, mode='generator', alpha=alpha)
            g_loss = sum(g_losses.values()).mean()

            self.optimizer_G.step(g_loss)
            self.g_losses = g_losses
            self.out = out
            if self.opt.use_ema:
                self.netG_ema(self.pix2pix_model.netG)
                self.netCorr_ema(self.pix2pix_model.netCorr)
      
        # traning D
        GforD = {}
        GforD['fake_image'] = self.out['fake_image']
        GforD['adaptive_feature_seg'] = self.out['adaptive_feature_seg']
        GforD['adaptive_feature_img'] = self.out['adaptive_feature_img']

        # self.pix2pix_model.set_grad_D(True)
        # self.pix2pix_model.set_grad_G(False)
        d_losses = self.pix2pix_model(input_label, input_semantics, real_image, self_ref, ref_image, ref_label, ref_semantics, mode='discriminator', GforD=GforD)
        d_loss = sum(d_losses.values()).mean()
        self.optimizer_D.step(d_loss)
        self.d_losses = d_losses
        
    def run_generator_one_step(self, data, alpha=1):
        self.optimizer_G.zero_grad()
        g_losses, out = self.pix2pix_model(data, mode='generator', alpha=alpha)
        g_loss = sum(g_losses.values()).mean()
        # g_loss.backward()
        self.optimizer_G.step(g_loss)
        self.g_losses = g_losses
        self.out = out
        if self.opt.use_ema:
            self.netG_ema(self.pix2pix_model.netG)
            self.netCorr_ema(self.pix2pix_model.netCorr)

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        GforD = {}
        GforD['fake_image'] = self.out['fake_image']
        GforD['adaptive_feature_seg'] = self.out['adaptive_feature_seg']
        GforD['adaptive_feature_img'] = self.out['adaptive_feature_img']
        d_losses = self.pix2pix_model(data, mode='discriminator', GforD=GforD)
        d_loss = sum(d_losses.values()).mean()
        # d_loss.backward()
        self.optimizer_D.step(d_loss)
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.out['fake_image']

    def update_learning_rate(self, epoch): # TODO 我看不懂，但我大受震撼
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model.save(epoch)
        if self.opt.use_ema:
            self.netG_ema.assign(self.pix2pix_model.netG)
            util.save_network(self.pix2pix_model.netG, 'G_ema', epoch, self.opt)
            self.netG_ema.resume(self.pix2pix_model.netG)

            self.netCorr_ema.assign(self.pix2pix_model.netCorr)
            util.save_network(self.pix2pix_model.netCorr, 'netCorr_ema', epoch, self.opt)
            self.netCorr_ema.resume(self.pix2pix_model.netCorr)
        if epoch == 'latest':
            jt.save({'G': self.optimizer_G.state_dict(),
                        'D': self.optimizer_D.state_dict(),
                        'lr':  self.old_lr,
                        }, os.path.join(self.opt.checkpoints_dir, self.opt.name, 'optimizer.pkl'))

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr

    def update_fixed_params(self):
        for param in self.pix2pix_model.netCorr.parameters():
            param.requires_grad = True
        G_params = [{'params': self.pix2pix_model.netG.parameters(), 'lr': self.opt.lr*0.5}]
        G_params += [{'params': self.pix2pix_model.netCorr.parameters(), 'lr': self.opt.lr*0.5}]
        if self.opt.no_TTUR:
            beta1, beta2 = self.opt.beta1, self.opt.beta2
            G_lr = self.opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr = self.opt.lr / 2

        self.optimizer_G = jt.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2), eps=1e-3)