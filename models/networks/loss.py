
import jittor as jt
import jittor.nn as nn
from models.networks.architecture import VGG19
from models.networks.correspondence import VGG19_feature_color_jittor


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=jt.Var, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        # self.load_logging()
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))



    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = jt.full([1], self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = jt.full([1], self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = jt.zeros(1)
            self.zero_tensor.stop_grad()
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, real_label_tensor, for_discriminator=True):

        # if self.gan_mode == 'original':  # cross entropy loss
        #     target_tensor = self.get_target_tensor(input, target_is_real)
        #     loss = F.binary_cross_entropy_with_logits(input, target_tensor)
        #     return loss        
        ############lisa get_class_balancing#############
        weight_map = get_class_balancing(self.opt, input, real_label_tensor)
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = nn.binary_cross_entropy_with_logits(input, target_tensor)
            if target_is_real:
                loss = loss * weight_map[:, 0, :, :]
            else:
                loss = loss
            return loss

        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return nn.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = jt.minimum(input - 1, self.get_zero_tensor(input))
                    loss = -jt.mean(minval)
                else:
                    minval = jt.minimum(-input - 1, self.get_zero_tensor(input))
                    loss = -jt.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -jt.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, real_label_tensor,for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        # writer = SummaryWriter(log_dir="summary_pic") 
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, real_label_tensor, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = jt.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss

            return loss / len(input)
        else:
            return self.loss(input, target_is_real,self.real_label_tensor, for_discriminator)


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids, vgg_normal_correct=False):
        super(VGGLoss, self).__init__()
        self.vgg_normal_correct = vgg_normal_correct
        if vgg_normal_correct:
            self.vgg = VGG19_feature_color_jittor(vgg_normal_correct=True)
        else:
            self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def execute(self, x, y):
        if self.vgg_normal_correct:
            x_vgg, y_vgg = self.vgg(x, ['r11', 'r21', 'r31', 'r41', 'r51'], preprocess=True), self.vgg(y, ['r11', 'r21', 'r31', 'r41', 'r51'], preprocess=True)
        else:
            x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        #     writer.add_scalar("VGGloss", loss.detach(), i)
        # writer.close() 
        return loss


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def execute(self, mu, logvar):
        kldloss=-0.5 * jt.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # writer.add_scalar("KLDloss", kldloss.detach())
        # writer.close() 
        return -0.5 * jt.sum(1 + logvar - mu.pow(2) - logvar.exp())

def reciprocal(tensor):
    return jt.divide(1,tensor)

def get_class_balancing(opt, input, label):
    if not opt.no_balancing_inloss:
        class_occurence = jt.sum(label, dims=(0, 2, 3))
        if opt.contain_dontcare_label:
            class_occurence[0] = 0
        num_of_classes = (class_occurence > 0).sum()
        coefficients = reciprocal(class_occurence) * label.numel() / (num_of_classes * label.shape[1])
        integers = jt.argmax(label, dim=1, keepdims=True)[0]
        if opt.contain_dontcare_label:
            coefficients[0] = 0
        weight_map = coefficients[integers]
    else:
        weight_map = jt.ones_like(input[:, :, :, :])
    return weight_map