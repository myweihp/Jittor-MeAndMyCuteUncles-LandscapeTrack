# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import jittor as jt
import jittor.nn as nn
from models.networks.base_network import BaseNetwork
from models.networks.generator import AdaptiveFeatureGenerator, DomainClassifier, ReverseLayerF
from util.util import vgg_preprocess
import util.util as util

def my_squeeze(tensor):
    shape = tensor.shape
    tgt_shape = list(filter(lambda i:i!=1, shape))
    if shape[0]==1:
        tgt_shape = [1] + tgt_shape
    # print("tgt_shape", tgt_shape)
    return tensor.reshape(tgt_shape)
class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ResidualBlock, self).__init__()
        self.padding1 = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.padding2 = nn.ReflectionPad2d(padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn2 = nn.InstanceNorm2d(out_channels)

    def execute(self, x):
        residual = x
        out = self.padding1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.prelu(out)
        return out

class WTA_scale(jt.Function): 

    def execute(self, input, scale=1e-4):

        activation_max, index_max = jt.argmax(input, -1, keepdims=True)
        input_scale = input * scale  # default: 1e-4

        output_max_scale = jt.where(input == activation_max, input, input_scale)

        mask = (input == activation_max).astype(jt.float32)
        # ctx.save_for_backward(input, mask)
        self.input = input
        self.mask = mask
        return output_max_scale

    def grad(self, grad_output):
        input = self.input
        mask = self.mask
        mask_ones = jt.ones_like(mask)
        mask_small_ones = jt.ones_like(mask) * 1e-4

        grad_scale = jt.where(mask == 1, mask_ones, mask_small_ones)
        grad_input = grad_output * grad_scale
        return grad_input

class VGG19_feature_color_jittor(nn.Module):
    ''' 
    NOTE: there is no need to pre-process the input 
    input tensor should range in [0,1]
    '''

    def __init__(self, pool='max', vgg_normal_correct=False, ic=3):
        super(VGG19_feature_color_jittor, self).__init__()
        self.vgg_normal_correct = vgg_normal_correct

        self.conv1_1 = nn.Conv2d(ic, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def execute(self, x, out_keys, preprocess=True):
        ''' 
        NOTE: input tensor should range in [0,1]
        '''
        out = {}
        if preprocess:
            x = vgg_preprocess(x, vgg_normal_correct=self.vgg_normal_correct)
        out['r11'] = nn.relu(self.conv1_1(x))
        out['r12'] = nn.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = nn.relu(self.conv2_1(out['p1']))
        out['r22'] = nn.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = nn.relu(self.conv3_1(out['p2']))
        out['r32'] = nn.relu(self.conv3_2(out['r31']))
        out['r33'] = nn.relu(self.conv3_3(out['r32']))
        out['r34'] = nn.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = nn.relu(self.conv4_1(out['p3']))
        out['r42'] = nn.relu(self.conv4_2(out['r41']))
        out['r43'] = nn.relu(self.conv4_3(out['r42']))
        out['r44'] = nn.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = nn.relu(self.conv5_1(out['p4']))
        out['r52'] = nn.relu(self.conv5_2(out['r51']))
        out['r53'] = nn.relu(self.conv5_3(out['r52']))
        out['r54'] = nn.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]

class NoVGGCorrespondence(BaseNetwork):
    # input is Al, Bl, channel = 1, range~[0,255]
    def __init__(self, opt):
        self.opt = opt
        super().__init__()

        opt.spade_ic = opt.semantic_nc
        self.adaptive_model_seg = AdaptiveFeatureGenerator(opt)
        opt.spade_ic = 3
        self.adaptive_model_img = AdaptiveFeatureGenerator(opt)
        del opt.spade_ic
        if opt.weight_domainC > 0 and (not opt.domain_rela):
            self.domain_classifier = DomainClassifier(opt)

        if 'down' not in opt:
            opt.down = 4
        if opt.warp_stride == 2:
            opt.down = 2
        assert (opt.down == 2) or (opt.down == 4)
        self.down = opt.down
        self.feature_channel = 64
        self.in_channels = self.feature_channel * 4
        self.inter_channels = 256
        
        coord_c = 3 if opt.use_coordconv else 0
        label_nc = opt.semantic_nc if opt.maskmix else 0
    
        self.layer = nn.Sequential(
            ResidualBlock(self.feature_channel * 4 + label_nc + coord_c, self.feature_channel * 4 + label_nc + coord_c, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4 + label_nc + coord_c, self.feature_channel * 4 + label_nc + coord_c, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4 + label_nc + coord_c, self.feature_channel * 4 + label_nc + coord_c, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4 + label_nc + coord_c, self.feature_channel * 4 + label_nc + coord_c, kernel_size=3, padding=1, stride=1))

        self.phi = nn.Conv2d(in_channels=self.in_channels + label_nc + coord_c, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channels + label_nc + coord_c, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.upsampling_bi = nn.Upsample(scale_factor=opt.down, mode='bilinear') #for show
        if opt.warp_bilinear:
            self.upsampling = nn.Upsample(scale_factor=opt.down, mode='bilinear')
        else:
            self.upsampling = nn.Upsample(scale_factor=opt.down)
        self.zero_tensor = None

        # model = [nn.ReflectionPad2d(1),
        #         nn.Conv2d(opt.semantic_nc, 128, kernel_size=3, padding=0, stride=1),
        #         nn.InstanceNorm2d(128),
        #         nn.PReLU(),
        #         nn.ReflectionPad2d(1),
        #         nn.Conv2d(128, self.feature_channel * 2, kernel_size=3, padding=0, stride=1),
        #         nn.InstanceNorm2d(self.feature_channel * 2),
        #         nn.PReLU()]
        # self.layer_mask_head = nn.Sequential(*model)
        # self.layer_mix = nn.Conv2d(in_channels=self.feature_channel * 6, out_channels=self.feature_channel * 4, kernel_size=1, stride=1, padding=0)

    def addcoords(self, x):
        bs, _, h, w = x.shape
        xx_ones = jt.ones([bs, h, 1], dtype=x.dtype)
        xx_range = jt.arange(w, dtype=x.dtype).unsqueeze(0).repeat([bs, 1]).unsqueeze(1)
        xx_channel = nn.matmul(xx_ones, xx_range).unsqueeze(1)

        yy_ones = jt.ones([bs, 1, w], dtype=x.dtype)
        yy_range = jt.arange(h, dtype=x.dtype).unsqueeze(0).repeat([bs, 1]).unsqueeze(-1)
        yy_channel = nn.matmul(yy_range, yy_ones).unsqueeze(1)

        xx_channel = xx_channel.float() / (w - 1)
        yy_channel = yy_channel.float() / (h - 1)
        xx_channel = 2 * xx_channel - 1
        yy_channel = 2 * yy_channel - 1

        rr_channel = jt.sqrt(jt.pow(xx_channel, 2) + jt.pow(yy_channel, 2))
        
        concat = jt.concat((x, xx_channel, yy_channel, rr_channel), dim=1)
        return concat

    def execute(self,
                ref_img,
                real_img,
                seg_map,
                ref_seg_map,
                temperature=0.01,
                detach_flag=False,
                WTA_scale_weight=1,
                alpha=1,
                return_corr=False):
        coor_out = {}
        batch_size = ref_img.shape[0]
        image_height = ref_img.shape[2]
        image_width = ref_img.shape[3]
        feature_height = int(image_height / self.opt.down)
        feature_width = int(image_width / self.opt.down)

        # print("corr:", ref_img.dtype, real_img.dtype, seg_map.dtype, ref_seg_map.dtype)

        if self.opt.mask_noise: #add noise to mask
            noise = jt.randn_like(seg_map) * 0.1
            noise.requires_grad = False
            noise[seg_map == 0] = 0
            seg_input = seg_map + noise
        else:
            seg_input = seg_map
        adaptive_feature_seg = self.adaptive_model_seg(seg_input, seg_input)
        adaptive_feature_img = self.adaptive_model_img(ref_img, ref_img)
        adaptive_feature_seg = util.feature_normalize(adaptive_feature_seg)
        adaptive_feature_img = util.feature_normalize(adaptive_feature_img)
        if self.opt.isTrain and self.opt.novgg_featpair > 0:
            adaptive_feature_img_pair = self.adaptive_model_img(real_img, real_img)
            adaptive_feature_img_pair = util.feature_normalize(adaptive_feature_img_pair)
            coor_out['loss_novgg_featpair'] = nn.l1_loss(adaptive_feature_seg, adaptive_feature_img_pair) * self.opt.novgg_featpair
  
        if self.opt.use_coordconv:
            adaptive_feature_seg = self.addcoords(adaptive_feature_seg)
            adaptive_feature_img = self.addcoords(adaptive_feature_img)
        
        seg = nn.interpolate(seg_map, size=adaptive_feature_seg.size()[2:], mode='nearest')
        ref_seg = nn.interpolate(ref_seg_map, size=adaptive_feature_img.size()[2:], mode='nearest')
        if self.opt.maskmix: #True
            cont_features = self.layer(jt.concat((adaptive_feature_seg, seg), 1))
            if self.opt.noise_for_mask and ((not self.opt.isTrain) or (self.opt.isTrain and self.opt.epoch > self.opt.mask_epoch)):
                noise = jt.randn_like(ref_seg) * 0.01
                noise.requires_grad = False
                ref_features = self.layer(jt.concat((adaptive_feature_img, noise), 1))
            else:
                ref_features = self.layer(jt.concat((adaptive_feature_img, ref_seg), 1))
        else:
            cont_features = self.layer(adaptive_feature_seg)
            ref_features = self.layer(adaptive_feature_img)

        # pairwise cosine similarity
        theta = self.theta(cont_features)
        if self.opt.match_kernel == 1:
            theta = theta.view(batch_size, self.inter_channels, -1)  # 2*256*(feature_height*feature_width)
        else: #this
            theta = nn.unfold(theta, kernel_size=self.opt.match_kernel, padding=int(self.opt.match_kernel // 2))
        dim_mean = 1 if self.opt.PONO_C else -1 #1
        theta = theta - theta.mean(dim=dim_mean, keepdims=True)  # center the feature
        theta_norm = jt.norm(theta, 2, 1, keepdim=True) #+ sys.float_info.epsilon
        theta = jt.divide(theta, theta_norm)
        theta_permute = theta.permute(0, 2, 1)  # 2*(feature_height*feature_width)*256
        phi = self.phi(ref_features)
        if self.opt.match_kernel == 1:
            phi = phi.view(batch_size, self.inter_channels, -1)  # 2*256*(feature_height*feature_width)
        else:
            phi = nn.unfold(phi, kernel_size=self.opt.match_kernel, padding=int(self.opt.match_kernel // 2))
        phi = phi - phi.mean(dim=dim_mean, keepdims=True)  # center the feature
        phi_norm = jt.norm(phi, 2, 1, keepdim=True) #+ sys.float_info.epsilon
        phi = jt.divide(phi, phi_norm)

        f = nn.matmul(theta_permute, phi)  # 2*(feature_height*feature_width)*(feature_height*feature_width)
        if detach_flag:
            f = f.detach()
        
        #f_similarity = f.unsqueeze(dim=1)
        # similarity_map = jt.max(f_similarity, -1, keepdim=True)[0]
        # similarity_map = similarity_map.view(batch_size, 1, feature_height, feature_width)

        # f can be negative
        if WTA_scale_weight == 1:
            f_WTA = f
        else:
            f_WTA = WTA_scale.apply(f, WTA_scale_weight)
        f_WTA = f_WTA / temperature
        # print("f_WTA", f_WTA.shape, f_WTA.dtype)
        if return_corr:
            return f_WTA
        # print(f_WTA.shape)
        f_WTA_tmp = my_squeeze(f_WTA)
        # print(f_WTA_tmp.shape)
        f_div_C = nn.softmax(f_WTA_tmp, dim=-1)  # 2*1936*1936; softmax along the horizontal line (dim=-1)
        # print("f_div_C", f_div_C.shape, f_div_C.dtype, f_div_C)
        # downsample the reference color
        if self.opt.warp_patch:
            ref = nn.unfold(ref_img, self.opt.down, stride=self.opt.down)
        else:
            ref = nn.avg_pool2d(ref_img, self.opt.down)
            channel = ref.shape[1]
            ref = ref.view(batch_size, channel, -1)
        ref = ref.permute(0, 2, 1)
        y = nn.matmul(f_div_C, ref)  # 2*1936*channel
        if self.opt.warp_patch:
            y = y.permute(0, 2, 1)
            y = nn.fold(y, 256, self.opt.down, stride=self.opt.down)
        else:
            y = y.permute(0, 2, 1)
            y = y.view(batch_size, channel, feature_height, feature_width)  # 2*3*44*44
        if (not self.opt.isTrain) and self.opt.show_corr:
            coor_out['warp_out_bi'] = y if self.opt.warp_patch else self.upsampling_bi(y)
        coor_out['warp_out'] = y if self.opt.warp_patch else self.upsampling(y) 
        if self.opt.warp_mask_losstype == 'direct' or self.opt.show_warpmask: #yes
            ref_seg = nn.interpolate(ref_seg_map.int(), scale_factor= 1/self.opt.down, mode='nearest').float()
            channel = ref_seg.shape[1]
            ref_seg = ref_seg.view(batch_size, channel, -1)
            ref_seg = ref_seg.permute(0, 2, 1)
            # print(f_div_C.shape, ref_seg.shape)
            warp_mask = nn.matmul(f_div_C, ref_seg)  # 2*1936*channel
            # warp_mask = f_div_C @ ref_seg
            warp_mask = warp_mask.permute(0, 2, 1)
            coor_out['warp_mask'] = warp_mask.view(batch_size, channel, feature_height, feature_width)  # 2*3*44*44
        elif self.opt.warp_mask_losstype == 'cycle':
            f_div_C_v = nn.softmax(f_WTA.transpose(1, 2), dim=-1)  # 2*1936*1936; softmax along the vertical line
            seg = nn.interpolate(seg_map, scale_factor=1 / self.opt.down, mode='nearest')
            channel = seg.shape[1]
            seg = seg.view(batch_size, channel, -1)
            seg = seg.permute(0, 2, 1)
            warp_mask_to_ref = nn.matmul(f_div_C_v, seg)  # 2*1936*channel
            warp_mask = nn.matmul(f_div_C, warp_mask_to_ref)  # 2*1936*channel
            warp_mask = warp_mask.permute(0, 2, 1)
            coor_out['warp_mask'] = warp_mask.view(batch_size, channel, feature_height, feature_width)  # 2*3*44*44
        else:
            warp_mask = None

        if self.opt.warp_cycle_w > 0: #no
            f_div_C_v = nn.softmax(f_WTA.transpose(1, 2), dim=-1)
            if self.opt.warp_patch:
                y = nn.unfold(y, self.opt.down, stride=self.opt.down)
                y = y.permute(0, 2, 1)
                warp_cycle = nn.matmul(f_div_C_v, y)
                warp_cycle = warp_cycle.permute(0, 2, 1)
                warp_cycle = nn.fold(warp_cycle, 256, self.opt.down, stride=self.opt.down)
                coor_out['warp_cycle'] = warp_cycle
            else:
                channel = y.shape[1]
                y = y.view(batch_size, channel, -1).permute(0, 2, 1)
                warp_cycle = nn.matmul(f_div_C_v, y).permute(0, 2, 1)
                coor_out['warp_cycle'] = warp_cycle.view(batch_size, channel, feature_height, feature_width)
                if self.opt.two_cycle:
                    real_img = nn.avg_pool2d(real_img, self.opt.down)
                    real_img = real_img.view(batch_size, channel, -1)
                    real_img = real_img.permute(0, 2, 1)
                    warp_i2r = nn.matmul(f_div_C_v, real_img).permute(0, 2, 1)  #warp input to ref
                    warp_i2r = warp_i2r.view(batch_size, channel, feature_height, feature_width)
                    warp_i2r2i = nn.matmul(f_div_C, warp_i2r.view(batch_size, channel, -1).permute(0, 2, 1))
                    coor_out['warp_i2r'] = warp_i2r
                    coor_out['warp_i2r2i'] = warp_i2r2i.permute(0, 2, 1).view(batch_size, channel, feature_height, feature_width)

        return coor_out
