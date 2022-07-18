import jittor as jt
import jittor.nn as nn
import models.networks as networks
import util.util as util
# from tensorboardX import SummaryWriter


def nll_loss_nd(output, target):
    new_target = target.unsqueeze(1)
    return jt.gather(output, 1, new_target).squeeze(1)

############lisa labelmix################
def generate_labelmix(label, fake_image, real_image):
    # target_map = jt.argmax(label, dim = 1, keepdims = True)[0]
    target_map = label.clone()

    # print(jt.argmax(label, dim = 1, keepdims = True))
    all_classes = jt.unique(target_map)
    # all_classes = np.unique(target_map.data)
    for c in all_classes:
        target_map[target_map == c] = jt.randint(0,2,(1,))
    target_map = target_map.float()
    mixed_image = target_map*real_image+(1-target_map)*fake_image
    return mixed_image, target_map

# writer = SummaryWriter('./loss')
class Pix2PixModel(nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = jt.Var
        self.ByteTensor = jt.Var
        self.alpha = 1
        self.initialize_networks(opt)
        ############lisa labelmix################
        # self.oasis_D = networks.discriminator.OASIS_Discriminator(opt)
        # self.oasis_D.requires_grad_ = False
        self.labelmix_function = nn.MSELoss()
        # set loss functions
        if opt.isTrain:
            self.vggnet_fix = networks.correspondence.VGG19_feature_color_jittor(vgg_normal_correct=opt.vgg_normal_correct)
            self.vggnet_fix.load_state_dict(jt.load('models/vgg19_conv.pth'))
            self.vggnet_fix.eval()
            for param in self.vggnet_fix.parameters():
                param.requires_grad = False

            self.contextual_forward_loss = networks.ContextualLoss_forward(opt)

            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = nn.L1Loss()
            self.MSE_loss = nn.MSELoss()
            if opt.which_perceptual == '5_2':
                self.perceptual_layer = -1
            elif opt.which_perceptual == '4_2':
                self.perceptual_layer = -2

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def execute(self, input_label, input_semantics, real_image, self_ref, ref_image, ref_label, ref_semantics, mode, GforD=None, alpha=1):
        # input_label, input_semantics, real_image, self_ref, ref_image, ref_label, ref_semantics = self.preprocess_input(data, )
        # print("exec start", mode, jt.unique(input_label))
        self.alpha = alpha
        generated_out = {}
        if mode == 'generator':
            g_loss, generated_out = self.compute_generator_loss(input_label, 
                input_semantics, real_image, ref_label, ref_semantics, ref_image, self_ref)
            # print("exec done", mode, jt.unique(input_label))
            out = {}
            out['fake_image'] = generated_out['fake_image']
            out['input_semantics'] = input_semantics
            out['ref_semantics'] = ref_semantics
            out['warp_out'] = None if 'warp_out' not in generated_out else generated_out['warp_out']
            out['warp_mask'] = None if 'warp_mask' not in generated_out else generated_out['warp_mask']
            out['adaptive_feature_seg'] = None if 'adaptive_feature_seg' not in generated_out else generated_out['adaptive_feature_seg']
            out['adaptive_feature_img'] = None if 'adaptive_feature_img' not in generated_out else generated_out['adaptive_feature_img']
            out['warp_cycle'] = None if 'warp_cycle' not in generated_out else generated_out['warp_cycle']
            out['warp_i2r'] = None if 'warp_i2r' not in generated_out else generated_out['warp_i2r']
            out['warp_i2r2i'] = None if 'warp_i2r2i' not in generated_out else generated_out['warp_i2r2i']
            return g_loss, out

        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image, GforD, label=input_label)
            return d_loss
        elif mode == 'inference':
            out = {}
            with jt.no_grad():
                out = self.inference(input_semantics, 
                        ref_semantics=ref_semantics, ref_image=ref_image, self_ref=self_ref)
            out['input_semantics'] = input_semantics
            out['ref_semantics'] = ref_semantics
            return out
        else:
            raise ValueError("|mode| is invalid")

    def set_grad_D(self, require_grad=True):
        self.netD.requires_grad_(require_grad)
        if self.opt.weight_domainC > 0 and self.opt.domain_rela:
            self.netDomainClassifier.requires_grad_(require_grad)

    def set_grad_G(self, require_grad=True):
        self.netG.requires_grad_(require_grad)
        self.netCorr.requires_grad_(require_grad)

    def create_optimizers(self, opt):
        G_params, D_params = list(), list()
        G_params += [{'params': self.netG.parameters(), 'lr': opt.lr*0.5}]
        G_params += [{'params': self.netCorr.parameters(), 'lr': opt.lr*0.5}]
        if opt.isTrain:
            D_params += list(self.netD.parameters())
            if opt.weight_domainC > 0 and opt.domain_rela:
                D_params += list(self.netDomainClassifier.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = jt.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2), eps=1e-3)
        optimizer_D = jt.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        util.save_network(self.netCorr, 'Corr', epoch, self.opt)
        if self.opt.weight_domainC > 0 and self.opt.domain_rela: 
            util.save_network(self.netDomainClassifier, 'DomainClassifier', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        
        self.netG = networks.define_G(opt)
        self.netD = networks.define_D(opt) if opt.isTrain else None
        self.netCorr = networks.define_Corr(opt)
        self.netDomainClassifier = networks.define_DomainClassifier(opt) if opt.weight_domainC > 0 and opt.domain_rela else None

        if not opt.isTrain or opt.continue_train:
            self.netG = util.load_network(self.netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                self.netD = util.load_network(self.netD, 'D', opt.which_epoch, opt)
            self.netCorr = util.load_network(self.netCorr, 'Corr', opt.which_epoch, opt)
            if opt.weight_domainC > 0 and opt.domain_rela:
                self.netDomainClassifier = util.load_network(self.netDomainClassifier, 'DomainClassifier', opt.which_epoch, opt)
            if (not opt.isTrain) and opt.use_ema:
                self.netG = util.load_network(self.netG, 'G_ema', opt.which_epoch, opt)
                self.netCorr = util.load_network(self.netCorr, 'netCorr_ema', opt.which_epoch, opt)
        # return net
        #return netG_stage1, netD_stage1, netG, netD, netE, netCorr

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):


        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = jt.zeros([bs, nc, h, w])
        input_semantics = jt.scatter(input_label, 1, label_map, jt.ones(1))
        label_map = data['label_ref']
        label_ref = jt.zeros([bs, nc, h, w])
        ref_semantics = jt.scatter(label_ref, 1, label_map, jt.ones(1))
        
        return data['label'], input_semantics, data['image'], data['self_ref'], data['ref'], data['label_ref'], ref_semantics

    def get_ctx_loss(self, source, target):
        contextual_style5_1 = jt.mean(self.contextual_forward_loss(source[-1], target[-1].detach())) * 8
        contextual_style4_1 = jt.mean(self.contextual_forward_loss(source[-2], target[-2].detach())) * 4
        contextual_style3_1 = jt.mean(self.contextual_forward_loss(nn.avg_pool2d(source[-3], 2), nn.avg_pool2d(target[-3].detach(), 2))) * 2
        if self.opt.use_22ctx:
            contextual_style2_1 = jt.mean(self.contextual_forward_loss(nn.avg_pool2d(source[-4], 4), nn.avg_pool2d(target[-4].detach(), 4))) * 1
            # self.trainwriter.add_scalar('contextual_style', contextual_style5_1 + contextual_style4_1 + contextual_style3_1 + contextual_style2_1, self.opt.use_22ctx)
            # writer.add_scalar('contextual_style', contextual_style5_1 + contextual_style4_1 + contextual_style3_1 + contextual_style2_1, self.opt.use_22ctx)
            return contextual_style5_1 + contextual_style4_1 + contextual_style3_1 + contextual_style2_1
        return contextual_style5_1 + contextual_style4_1 + contextual_style3_1

    def compute_generator_loss(self, input_label, input_semantics, real_image, ref_label=None, ref_semantics=None, ref_image=None, self_ref=None):
        G_losses = {}
        generate_out = self.generate_fake(
            input_semantics, real_image, ref_semantics=ref_semantics, ref_image=ref_image, self_ref=self_ref)

        if 'loss_novgg_featpair' in generate_out and generate_out['loss_novgg_featpair'] is not None:
            G_losses['no_vgg_feat'] = generate_out['loss_novgg_featpair']
            # Domain alignment loss
        if self.opt.warp_cycle_w > 0: #no
            if not self.opt.warp_patch: 
                ref = nn.avg_pool2d(ref_image, self.opt.warp_stride)
            else:
                ref = ref_image

            G_losses['G_warp_cycle'] = nn.l1_loss(generate_out['warp_cycle'], ref) * self.opt.warp_cycle_w
            if self.opt.two_cycle:
                real = nn.avg_pool2d(real_image, self.opt.warp_stride)
                G_losses['G_warp_cycle'] += nn.l1_loss(generate_out['warp_i2r2i'], real) * self.opt.warp_cycle_w
                
        if self.opt.warp_self_w > 0: #no

            sample_weights = (self_ref[:, 0, 0, 0] / (sum(self_ref[:, 0, 0, 0]) + 1e-5)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            G_losses['G_warp_self'] = jt.mean(nn.l1_loss(generate_out['warp_out'], real_image, reduce=False) * sample_weights) * self.opt.warp_self_w

        pred_fake, pred_real, seg, fake_cam_logit, real_cam_logit = self.discriminate(
            input_semantics, generate_out['fake_image'], real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True, input_label,
                                            for_discriminator=False) * self.opt.weight_gan
        # GAN loss
        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = jt.zeros(1)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
                    # writer.add_scalar('GAN_Feat_loss', GAN_Feat_loss, i)
            G_losses['GAN_Feat'] = GAN_Feat_loss
        # GAN feat loss
        fake_features = self.vggnet_fix(generate_out['fake_image'], ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
        sample_weights = (self_ref[:, 0, 0, 0] / (sum(self_ref[:, 0, 0, 0]) + 1e-5)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        loss = 0
        for i in range(len(generate_out['real_features'])):
            loss += weights[i] * util.weighted_l1_loss(fake_features[i], generate_out['real_features'][i].detach(), sample_weights)
            # writer.add_scalar('loss', loss, i)
        G_losses['fm'] = loss * self.opt.lambda_vgg * self.opt.fm_ratio
        
        #Exemplar translation losses
        feat_loss = util.mse_loss(fake_features[self.perceptual_layer], generate_out['real_features'][self.perceptual_layer].detach())
        G_losses['perc'] = feat_loss * self.opt.weight_perceptual

        # contextual loss
        G_losses['contextual'] = self.get_ctx_loss(fake_features, generate_out['ref_features']) * self.opt.lambda_vgg * self.opt.ctx_w

        if self.opt.warp_mask_losstype != 'none':

            ref_label = nn.interpolate(ref_label, scale_factor=0.25, mode='nearest').astype(jt.int32).squeeze(1)
            gt_label = nn.interpolate(input_label, scale_factor=0.25, mode='nearest').astype(jt.int32).squeeze(1)
            weights = []

            for i in range(ref_label.shape[0]):
                
                ref_label_uniq = jt.unique(ref_label[i])
                gt_label_uniq = jt.unique(gt_label[i])
                zero_label = [it for it in gt_label_uniq if it not in ref_label_uniq]
                weight = jt.ones_like(gt_label[i]).float()
                for j in zero_label:
                    weight[gt_label[i] == j] = 0
                weight[gt_label[i] == 0] = 0 #no loss from unknown class
                weights.append(weight.unsqueeze(0))
            weights = jt.concat(weights, dim=0)
            G_losses['mask'] = (nn.cross_entropy_loss(generate_out['warp_mask'], gt_label, reduction=None) * weights).sum() / (weights.sum() + 1e-5) * self.opt.weight_mask
            # print(generate_out['warp_mask'].shape, gt_label.shape)
            # G_losses['mask'] = (nll_loss_nd(jt.log(generate_out['warp_mask'] + 1e-10), gt_label) * weights).sum() / (weights.sum() + 1e-5) * self.opt.weight_mask

        #self.fake_image = fake_image
        return G_losses, generate_out

    def loss_labelmix(self, mask, output_D_mixed, output_D_fake, output_D_real):
        # print("output_D_mixed", len(output_D_mixed))
        # print("output_D_real", len(output_D_real))
        loss = 0
        for i in range(len(output_D_mixed)):
            output_D_mixed_i = output_D_mixed[i]
            output_D_real_i = output_D_real[i]
            output_D_fake_i = output_D_fake[i]
            if isinstance(output_D_mixed_i, list):
                output_D_mixed_i = output_D_mixed_i[-1]
                output_D_real_i = output_D_real_i[-1]
                output_D_fake_i = output_D_fake_i[-1]
            n, c, h, w = output_D_fake_i.shape
            mask_i = nn.interpolate(mask, (h,w), mode='nearest')
            mixed_D_output_i = mask_i * output_D_real_i + (1-mask_i) * output_D_fake_i
            loss += self.labelmix_function(mixed_D_output_i, output_D_mixed_i)
        ############lisa labelmix################
        return loss

    def compute_discriminator_loss(self, input_semantics, real_image, GforD, label, ref_label=None, ref_semantics=None, ref_image=None, self_ref=None):
        D_losses = {}
        with jt.no_grad():
            #fake_image, _, _, _, _ = self.generate_fake(input_semantics, real_image, VGG_feat=False)
            fake_image = GforD['fake_image'].detach()
            fake_image.start_grad()
            
        pred_fake, pred_real, seg, fake_cam_logit, real_cam_logit = self.discriminate(
            input_semantics, fake_image, real_image)
        # print("pred_fake", pred_fake[-1][-1].shape)
        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False, label,
                                            for_discriminator=True) * self.opt.weight_gan
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                            label,for_discriminator=True) * self.opt.weight_gan
        ###########lisa labelmix################
        if not self.opt.no_labelmix:

            ## generate_labelmix
            # mask = jt.argmax(input_semantics, dim = 1, keepdims = True)[0]
            all_classes = jt.unique(label)
            
            for c in all_classes:
                label[label==c] = jt.randint(0,2,(1,))

            mask = label.float()
            mixed_inp = mask * real_image + (1-mask)*fake_image
            # print(mixed_inp.shape)
            # mixed_inp, mask = generate_labelmix(label, output_D_fake, output_D_real)
            fake_concat = jt.concat([input_semantics, fake_image], dim=1)
            real_concat = jt.concat([input_semantics, real_image], dim=1)
            mixed_concat = jt.concat([input_semantics, mixed_inp], dim=1)
            output_D_mixed, seg, _ = self.netD(mixed_concat)
            output_D_fake, seg, _ = self.netD(fake_concat)
            output_D_real, seg, _ = self.netD(real_concat)
            loss_D_lm = self.opt.lambda_labelmix * self.loss_labelmix(mask, output_D_mixed, output_D_fake,output_D_real)
            D_losses['D_Fake'] += loss_D_lm
            D_losses['D_real'] += loss_D_lm
        else:
            loss_D_lm = None

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, ref_semantics=None, ref_image=None, self_ref=None):
        generate_out = {}
        #print(ref_image.max())
        ref_relu1_1, ref_relu2_1, ref_relu3_1, ref_relu4_1, ref_relu5_1 = self.vggnet_fix(ref_image, ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
        
        coor_out = self.netCorr(ref_image, real_image, input_semantics, ref_semantics, alpha=self.alpha)
        
        generate_out['ref_features'] = [ref_relu1_1, ref_relu2_1, ref_relu3_1, ref_relu4_1, ref_relu5_1]
        generate_out['real_features'] = self.vggnet_fix(real_image, ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
        
        if self.opt.CBN_intype == 'mask':
            CBN_in = input_semantics
        elif self.opt.CBN_intype == 'warp':
            CBN_in = coor_out['warp_out']
        elif self.opt.CBN_intype == 'warp_mask':
            CBN_in = jt.concat((coor_out['warp_out'], input_semantics), dim=1)

        generate_out['fake_image'] = self.netG(input_semantics, warp_out=CBN_in)

        generate_out = {**generate_out, **coor_out}
        return generate_out

    def inference(self, input_semantics, ref_semantics=None, ref_image=None, self_ref=None):
        generate_out = {}
        coor_out = self.netCorr(ref_image, None, input_semantics, ref_semantics, alpha=self.alpha)
        if self.opt.CBN_intype == 'mask':
            CBN_in = input_semantics
        elif self.opt.CBN_intype == 'warp':
            CBN_in = coor_out['warp_out']
        elif self.opt.CBN_intype == 'warp_mask':
            CBN_in = jt.concat((coor_out['warp_out'], input_semantics), dim=1)

        generate_out['fake_image'] = self.netG(input_semantics, warp_out=CBN_in)
        generate_out = {**generate_out, **coor_out}
        return generate_out

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = jt.concat([input_semantics, fake_image], dim=1)
        real_concat = jt.concat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = jt.concat([fake_concat, real_concat], dim=0)
        seg = None
        discriminator_out, seg, cam_logit = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)
        fake_cam_logit, real_cam_logit = None, None
        if self.opt.D_cam > 0:
            fake_cam_logit = jt.concat([it[:it.shape[0]//2] for it in cam_logit], dim=1)
            real_cam_logit = jt.concat([it[it.shape[0]//2:] for it in cam_logit], dim=1)
        #fake_cam_logit, real_cam_logit = self.divide_pred(cam_logit)

        return pred_fake, pred_real, seg, fake_cam_logit, real_cam_logit

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = jt.exp(0.5 * logvar)
        eps = jt.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def compute_D_seg_loss(self, out, gt):
        fake_seg, real_seg = self.divide_pred([out])
        fake_seg_loss = nn.cross_entropy(fake_seg[0][0], gt)
        real_seg_loss = nn.cross_entropy(real_seg[0][0], gt)

        down_gt = nn.interpolate(gt.unsqueeze(1).float(), scale_factor=0.5, mode='nearest').squeeze().int()
        fake_seg_loss_down = nn.cross_entropy(fake_seg[0][1], down_gt)
        real_seg_loss_down = nn.cross_entropy(real_seg[0][1], down_gt)

        seg_loss = fake_seg_loss + real_seg_loss + fake_seg_loss_down + real_seg_loss_down
        return seg_loss