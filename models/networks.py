import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class ResnetGenerator(nn.Module):
    def __init__(self, ngf=64, img_size=256, light=False):
        super(ResnetGenerator, self).__init__()
        self.light = light

        self.ConvBlock1 = nn.Sequential(nn.ReflectionPad2d(3),
                                       nn.Conv2d(3, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                                       nn.InstanceNorm2d(ngf),
                                       nn.ReLU(True))

        self.HourGlass1 = HourGlass(ngf, ngf)
        self.HourGlass2 = HourGlass(ngf, ngf)

        # Down-Sampling
        self.DownBlock1 = nn.Sequential(nn.ReflectionPad2d(1),
                                        nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=0, bias=False),
                                        nn.InstanceNorm2d(ngf * 2),
                                        nn.ReLU(True))

        self.DownBlock2 = nn.Sequential(nn.ReflectionPad2d(1),
                                        nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=0, bias=False),
                                        nn.InstanceNorm2d(ngf*4),
                                        nn.ReLU(True))

        # Encoder Bottleneck
        self.EncodeBlock1 = ResnetBlock(ngf*4)
        self.EncodeBlock2 = ResnetBlock(ngf*4)
        self.EncodeBlock3 = ResnetBlock(ngf*4)
        self.EncodeBlock4 = ResnetBlock(ngf*4)

        # Class Activation Map
        self.gap_fc = nn.Linear(ngf*4, 1)
        self.gmp_fc = nn.Linear(ngf*4, 1)
        self.conv1x1 = nn.Conv2d(ngf*8, ngf*4, kernel_size=1, stride=1)
        self.relu = nn.ReLU(True)

        # Gamma, Beta block
        if self.light:
            self.FC = nn.Sequential(nn.Linear(ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear(ngf*4, ngf*4),
                                    nn.ReLU(True))
        else:
            self.FC = nn.Sequential(nn.Linear(img_size//4*img_size//4*ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear(ngf*4, ngf*4),
                                    nn.ReLU(True))

        # Decoder Bottleneck
        self.DecodeBlock1 = ResnetSoftAdaLINBlock(ngf*4)
        self.DecodeBlock2 = ResnetSoftAdaLINBlock(ngf*4)
        self.DecodeBlock3 = ResnetSoftAdaLINBlock(ngf*4)
        self.DecodeBlock4 = ResnetSoftAdaLINBlock(ngf*4)

        # Up-Sampling
        self.UpBlock1 = nn.Sequential(nn.Upsample(scale_factor=2),
                                      nn.ReflectionPad2d(1),
                                      nn.Conv2d(ngf*4, ngf*2, kernel_size=3, stride=1, padding=0, bias=False),
                                      LIN(ngf*2),
                                      nn.ReLU(True))

        self.UpBlock2 = nn.Sequential(nn.Upsample(scale_factor=2),
                                      nn.ReflectionPad2d(1),
                                      nn.Conv2d(ngf*2, ngf, kernel_size=3, stride=1, padding=0, bias=False),
                                      LIN(ngf),
                                      nn.ReLU(True))

        self.HourGlass3 = HourGlass(ngf, ngf)
        self.HourGlass4 = HourGlass(ngf, ngf, False)

        self.ConvBlock2 = nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(3, 3, kernel_size=7, stride=1, padding=0, bias=False),
                                        nn.Tanh())

    def forward(self, x):
        x = self.ConvBlock1(x)
        x = self.HourGlass1(x)
        x = self.HourGlass2(x)

        x = self.DownBlock1(x)
        x = self.DownBlock2(x)

        x = self.EncodeBlock1(x)
        content_features1 = F.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1)
        x = self.EncodeBlock2(x)
        content_features2 = F.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1)
        x = self.EncodeBlock3(x)
        content_features3 = F.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1)
        x = self.EncodeBlock4(x)
        content_features4 = F.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1)

        gap = F.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = F.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)

        if self.light:
            x_ = F.adaptive_avg_pool2d(x, 1)
            style_features = self.FC(x_.view(x_.shape[0], -1))
        else:
            style_features = self.FC(x.view(x.shape[0], -1))

        x = self.DecodeBlock1(x, content_features4, style_features)
        x = self.DecodeBlock2(x, content_features3, style_features)
        x = self.DecodeBlock3(x, content_features2, style_features)
        x = self.DecodeBlock4(x, content_features1, style_features)

        x = self.UpBlock1(x)
        x = self.UpBlock2(x)

        x = self.HourGlass3(x)
        x = self.HourGlass4(x)
        out = self.ConvBlock2(x)

        return out, cam_logit, heatmap


class ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConvBlock, self).__init__()
        self.dim_out = dim_out

        self.ConvBlock1 = nn.Sequential(nn.InstanceNorm2d(dim_in),
                                        nn.ReLU(True),
                                        nn.ReflectionPad2d(1),
                                        nn.Conv2d(dim_in, dim_out//2, kernel_size=3, stride=1, bias=False))

        self.ConvBlock2 = nn.Sequential(nn.InstanceNorm2d(dim_out//2),
                                        nn.ReLU(True),
                                        nn.ReflectionPad2d(1),
                                        nn.Conv2d(dim_out//2, dim_out//4, kernel_size=3, stride=1, bias=False))

        self.ConvBlock3 = nn.Sequential(nn.InstanceNorm2d(dim_out//4),
                                        nn.ReLU(True),
                                        nn.ReflectionPad2d(1),
                                        nn.Conv2d(dim_out//4, dim_out//4, kernel_size=3, stride=1, bias=False))

        self.ConvBlock4 = nn.Sequential(nn.InstanceNorm2d(dim_in),
                                        nn.ReLU(True),
                                        nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False))

    def forward(self, x):
        residual = x

        x1 = self.ConvBlock1(x)
        x2 = self.ConvBlock2(x1)
        x3 = self.ConvBlock3(x2)
        out = torch.cat((x1, x2, x3), 1)

        if residual.size(1) != self.dim_out:
            residual = self.ConvBlock4(residual)

        return residual + out


class HourGlass(nn.Module):
    def __init__(self, dim_in, dim_out, use_res=True):
        super(HourGlass, self).__init__()
        self.use_res = use_res

        self.HG = nn.Sequential(HourGlassBlock(dim_in, dim_out),
                                ConvBlock(dim_out, dim_out),
                                nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1, bias=False),
                                nn.InstanceNorm2d(dim_out),
                                nn.ReLU(True))

        self.Conv1 = nn.Conv2d(dim_out, 3, kernel_size=1, stride=1)

        if self.use_res:
            self.Conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1)
            self.Conv3 = nn.Conv2d(3, dim_out, kernel_size=1, stride=1)

    def forward(self, x):
        ll = self.HG(x)
        tmp_out = self.Conv1(ll)

        if self.use_res:
            ll = self.Conv2(ll)
            tmp_out_ = self.Conv3(tmp_out)
            return x + ll + tmp_out_

        else:
            return tmp_out


class HourGlassBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(HourGlassBlock, self).__init__()

        self.ConvBlock1_1 = ConvBlock(dim_in, dim_out)
        self.ConvBlock1_2 = ConvBlock(dim_out, dim_out)
        self.ConvBlock2_1 = ConvBlock(dim_out, dim_out)
        self.ConvBlock2_2 = ConvBlock(dim_out, dim_out)
        self.ConvBlock3_1 = ConvBlock(dim_out, dim_out)
        self.ConvBlock3_2 = ConvBlock(dim_out, dim_out)
        self.ConvBlock4_1 = ConvBlock(dim_out, dim_out)
        self.ConvBlock4_2 = ConvBlock(dim_out, dim_out)

        self.ConvBlock5 = ConvBlock(dim_out, dim_out)

        self.ConvBlock6 = ConvBlock(dim_out, dim_out)
        self.ConvBlock7 = ConvBlock(dim_out, dim_out)
        self.ConvBlock8 = ConvBlock(dim_out, dim_out)
        self.ConvBlock9 = ConvBlock(dim_out, dim_out)

    def forward(self, x):
        skip1 = self.ConvBlock1_1(x)
        down1 = F.avg_pool2d(x, 2)
        down1 = self.ConvBlock1_2(down1)

        skip2 = self.ConvBlock2_1(down1)
        down2 = F.avg_pool2d(down1, 2)
        down2 = self.ConvBlock2_2(down2)

        skip3 = self.ConvBlock3_1(down2)
        down3 = F.avg_pool2d(down2, 2)
        down3 = self.ConvBlock3_2(down3)

        skip4 = self.ConvBlock4_1(down3)
        down4 = F.avg_pool2d(down3, 2)
        down4 = self.ConvBlock4_2(down4)

        center = self.ConvBlock5(down4)

        up4 = self.ConvBlock6(center)
        up4 = F.upsample(up4, scale_factor=2)
        up4 = skip4 + up4

        up3 = self.ConvBlock7(up4)
        up3 = F.upsample(up3, scale_factor=2)
        up3 = skip3 + up3

        up2 = self.ConvBlock8(up3)
        up2 = F.upsample(up2, scale_factor=2)
        up2 = skip2 + up2

        up1 = self.ConvBlock9(up2)
        up1 = F.upsample(up1, scale_factor=2)
        up1 = skip1 + up1

        return up1


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias=False):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetSoftAdaLINBlock(nn.Module):
    def __init__(self, dim, use_bias=False):
        super(ResnetSoftAdaLINBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = SoftAdaLIN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = SoftAdaLIN(dim)

    def forward(self, x, content_features, style_features):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, content_features, style_features)
        out = self.relu1(out)

        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, content_features, style_features)
        return out + x


class ResnetAdaLINBlock(nn.Module):
    def __init__(self, dim, use_bias=False):
        super(ResnetAdaLINBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaLIN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaLIN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class SoftAdaLIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(SoftAdaLIN, self).__init__()
        self.norm = adaLIN(num_features, eps)

        self.w_gamma = Parameter(torch.zeros(1, num_features))
        self.w_beta = Parameter(torch.zeros(1, num_features))

        self.c_gamma = nn.Sequential(nn.Linear(num_features, num_features),
                                     nn.ReLU(True),
                                     nn.Linear(num_features, num_features))
        self.c_beta = nn.Sequential(nn.Linear(num_features, num_features),
                                    nn.ReLU(True),
                                    nn.Linear(num_features, num_features))
        self.s_gamma = nn.Linear(num_features, num_features)
        self.s_beta = nn.Linear(num_features, num_features)

    def forward(self, x, content_features, style_features):
        content_gamma, content_beta = self.c_gamma(content_features), self.c_beta(content_features)
        style_gamma, style_beta = self.s_gamma(style_features), self.s_beta(style_features)

        w_gamma, w_beta = self.w_gamma.expand(x.shape[0], -1), self.w_beta.expand(x.shape[0], -1)
        soft_gamma = (1. - w_gamma) * style_gamma + w_gamma * content_gamma
        soft_beta = (1. - w_beta) * style_beta + w_beta * content_beta

        out = self.norm(x, soft_gamma, soft_beta)
        return out


class adaLIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaLIN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out


class LIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(LIN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                 nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                  nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        self.model = nn.Sequential(*model)

    def forward(self, input):
        x = self.model(input)

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap


class RhoClipper(object):
    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):
        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w


class WClipper(object):
    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):
        if hasattr(module, 'w_gamma'):
            w = module.w_gamma.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.w_gamma.data = w

        if hasattr(module, 'w_beta'):
            w = module.w_beta.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.w_beta.data = w
