# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# 08.09 change pad

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from models.architecture import Attention
from models.architecture import ResnetBlock as ResnetBlock
from models.architecture import SPADEResnetBlock as SPADEResnetBlock
from models.base_network import BaseNetwork
from models.normalization import equal_lr, get_nonspade_norm_layer
from models.sync_batchnorm import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G="spectralspadesyncbatch3x3")
        return parser

    def __init__(self):
        super().__init__()

        nf = 64

        self.sw, self.sh = self.compute_latent_vector_size()
        semantic_nc = 19
        ic = 0 + 3 + semantic_nc
        self.fc = nn.Conv2d(ic, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf)
        self.attn = Attention(4 * nf, "spectral" in "spectralspadesyncbatch3x3")
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf)

        final_nc = nf

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self):
        num_up_layers = 5

        sw = 256 // (2 ** num_up_layers)
        sh = round(sw / 1)

        return sw, sh

    def forward(self, input, warp_out=None):
        seg = input if warp_out is None else warp_out

        # we downsample segmap and run convolution
        x = F.interpolate(seg, size=(self.sh, self.sw))
        x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)

        x = self.up(x)
        if self.opt.use_attention:
            x = self.attn(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x


class AdaptiveFeatureGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G="spectralspadesyncbatch3x3")
        parser.add_argument(
            "--num_upsampling_layers",
            choices=("normal", "more", "most"),
            default="normal",
            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator",
        )

        return parser

    def __init__(self):
        # TODO: kernel=4, concat noise, or change architecture to vgg feature pyramid
        super().__init__()
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = 64
        norm_layer = get_nonspade_norm_layer("spectralinstance")
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=1, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, 3, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=1, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, 3, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=1, padding=pw))

        self.actvn = nn.LeakyReLU(0.2, False)

        nf = 64
        self.head_0 = SPADEResnetBlock(8 * nf, 8 * nf, use_se=False)
        self.G_middle_0 = SPADEResnetBlock(8 * nf, 8 * nf, use_se=False)
        self.G_middle_1 = SPADEResnetBlock(8 * nf, 4 * nf, use_se=False)

    def forward(self, input, seg):
        x = self.layer1(input)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))

        x = self.head_0(x, seg)

        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)

        return x


class ShallowAdaptiveFeatureGenerator(BaseNetwork):
    def __init__(self, spade_resnet=True, ch=256, ic=None, inch=3):
        super().__init__()
        self.spade_resnet = spade_resnet
        kw = 3
        ndf = 64
        self.ch_ratio = int(ch / 64)
        norm_layer = get_nonspade_norm_layer("spectralinstance")
        self.layer1 = norm_layer(nn.Conv2d(inch, ndf, 3, stride=1, padding=1))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 4, 3, stride=1, padding=1))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=1, padding=1))

        self.actvn = nn.LeakyReLU(0.2, False)

        nf = 64
        ic = 3 if ic is None else ic
        if spade_resnet:
            self.G_middle_1 = SPADEResnetBlock(8 * nf, self.ch_ratio * nf, ic=ic, use_se=False)

    def forward(self, input, seg=None):
        if seg is None:
            seg = input

        x = self.layer1(input)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))

        if self.spade_resnet:
            x = self.G_middle_1(x, seg)
            return x
        return x


class ShallowAdaptiveFeatureGeneratorForLabel(BaseNetwork):
    def __init__(self, label_nch, encoder_hidden_dim=64, spade_resnet=True, ch=256):
        super().__init__()
        self.spade_resnet = spade_resnet
        kernel_size = 3
        self.ch_ratio = int(ch / 64)
        norm_layer = get_nonspade_norm_layer("spectralinstance")
        self.layer1 = norm_layer(nn.Conv2d(label_nch, encoder_hidden_dim, 3, stride=1, padding=1))
        self.layer2 = norm_layer(nn.Conv2d(encoder_hidden_dim * 1, encoder_hidden_dim * 4, 3, stride=1, padding=1))
        self.layer3 = norm_layer(
            nn.Conv2d(encoder_hidden_dim * 4, encoder_hidden_dim * 8, kernel_size, stride=1, padding=1)
        )

        self.actvn = nn.LeakyReLU(0.2, False)

        if spade_resnet:
            self.G_middle_1 = SPADEResnetBlock(
                8 * encoder_hidden_dim, self.ch_ratio * encoder_hidden_dim, use_se=False, ic=label_nch
            )

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))

        if self.spade_resnet:
            x = self.G_middle_1(x, input)
            return x
        return x


class ReverseGenerator(BaseNetwork):
    def __init__(self, opt, ic, oc, size):
        super().__init__()
        self.opt = opt
        self.downsample = True if size == 256 else False
        nf = opt.ngf
        opt.spade_ic = ic
        if opt.warp_reverseG_s:
            self.backbone_0 = SPADEResnetBlock(4 * nf, 4 * nf, opt)
        else:
            self.backbone_0 = SPADEResnetBlock(4 * nf, 8 * nf, opt)
            self.backbone_1 = SPADEResnetBlock(8 * nf, 8 * nf, opt)
            self.backbone_2 = SPADEResnetBlock(8 * nf, 8 * nf, opt)
            self.backbone_3 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.backbone_4 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.backbone_5 = SPADEResnetBlock(2 * nf, nf, opt)
        del opt.spade_ic
        if self.downsample:
            kw = 3
            pw = int(np.ceil((kw - 1.0) / 2))
            ndf = opt.ngf
            norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
            self.layer1 = norm_layer(nn.Conv2d(ic, ndf, kw, stride=1, padding=pw))
            self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, 4, stride=2, padding=pw))
            self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=1, padding=pw))
            self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 4, 4, stride=2, padding=pw))
            self.up = nn.Upsample(scale_factor=2)
        self.actvn = nn.LeakyReLU(0.2, False)
        self.conv_img = nn.Conv2d(nf, oc, 3, padding=1)

    def forward(self, x):
        input = x
        if self.downsample:
            x = self.layer1(input)
            x = self.layer2(self.actvn(x))
            x = self.layer3(self.actvn(x))
            x = self.layer4(self.actvn(x))
        x = self.backbone_0(x, input)
        if not self.opt.warp_reverseG_s:
            x = self.backbone_1(x, input)
            x = self.backbone_2(x, input)
            x = self.backbone_3(x, input)
        if self.downsample:
            x = self.up(x)
        x = self.backbone_4(x, input)
        if self.downsample:
            x = self.up(x)
        x = self.backbone_5(x, input)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        return x


class DomainClassifier(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        nf = opt.ngf
        kw = 4 if opt.domain_rela else 3
        pw = int((kw - 1.0) / 2)
        self.feature = nn.Sequential(
            nn.Conv2d(4 * nf, 2 * nf, kw, stride=2, padding=pw),
            SynchronizedBatchNorm2d(2 * nf, affine=True),
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(2 * nf, nf, kw, stride=2, padding=pw),
            SynchronizedBatchNorm2d(nf, affine=True),
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(nf, int(nf // 2), kw, stride=2, padding=pw),
            SynchronizedBatchNorm2d(int(nf // 2), affine=True),
            nn.LeakyReLU(0.2, False),
        )  # 32*8*8
        model = [nn.Linear(int(nf // 2) * 8 * 8, 100), SynchronizedBatchNorm1d(100, affine=True), nn.ReLU()]
        if opt.domain_rela:
            model += [nn.Linear(100, 1)]
        else:
            model += [nn.Linear(100, 2), nn.LogSoftmax(dim=1)]
        self.classifier = nn.Sequential(*model)

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x.view(x.shape[0], -1))
        return x


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class EMA:
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}
        self.original = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                decay = self.mu
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]
