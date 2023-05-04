"""
Mostly copy-paste from mmsegmentation
"""

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from timm.models.layers import trunc_normal_, get_act_layer


class ConvModule(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            norm_layer=nn.BatchNorm2d,
            act_type='relu',
    ):
        super(ConvModule, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.norm = norm_layer(out_channels)
        self.act = get_act_layer(act_type)()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class BasicConvBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            num_convs=2,
            stride=1,
            dilation=1,
            norm_layer=nn.BatchNorm2d,
            act_type='relu',
    ):
        super(BasicConvBlock, self).__init__()

        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=1 if i == 0 else dilation,
                    norm_layer=norm_layer,
                    act_type=act_type
                )
            )

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        out = self.convs(x)
        return out


class DeconvModule(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        norm_layer=nn.BatchNorm2d,
        act_type='relu',
        *,
        kernel_size=4,
        scale_factor=2
    ):
        super(DeconvModule, self).__init__()

        assert (kernel_size - scale_factor >= 0) and\
               (kernel_size - scale_factor) % 2 == 0,\
               f'kernel_size should be greater than or equal to scale_factor '\
               f'and (kernel_size - scale_factor) should be even numbers, '\
               f'while the kernel size is {kernel_size} and scale_factor is '\
               f'{scale_factor}.'

        stride = scale_factor
        padding = (kernel_size - scale_factor) // 2
        deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        norm = norm_layer(out_channels)
        activate = get_act_layer(act_type)()

        self.deconv_upsamping = nn.Sequential(deconv, norm, activate)

    def forward(self, x):

        out = self.deconv_upsamping(x)
        return out


class InterpConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_layer=nn.BatchNorm2d,
                 act_type='relu',
                 *,
                 conv_first=False,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 upsample_cfg=dict(
                     scale_factor=2, mode='bilinear', align_corners=False)):
        super(InterpConv, self).__init__()

        conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm_layer=norm_layer,
            act_type=act_type
        )
        upsample = nn.Upsample(**upsample_cfg)
        if conv_first:
            self.interp_upsample = nn.Sequential(conv, upsample)
        else:
            self.interp_upsample = nn.Sequential(upsample, conv)

    def forward(self, x):
        out = self.interp_upsample(x)
        return out


class UpConvBlock(nn.Module):

    def __init__(self,
                 conv_block,
                 in_channels,
                 skip_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 norm_layer=nn.BatchNorm2d,
                 act_type='relu',
                 upsample_layer=InterpConv,
                 ):
        super(UpConvBlock, self).__init__()

        self.conv_block = conv_block(
            in_channels=2 * skip_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            norm_layer=norm_layer,
            act_type=act_type,
        )

        self.upsample = upsample_layer(
            in_channels=in_channels,
            out_channels=skip_channels,
            norm_layer=norm_layer,
            act_type=act_type
        )

    def forward(self, skip, x):
        """Forward function."""

        x = self.upsample(x)
        out = torch.cat([skip, x], dim=1)
        out = self.conv_block(out)

        return out


class UNet(nn.Module):

    def __init__(self, config, **kwargs):
        super(UNet, self).__init__()

        spec_dict = {k.lower(): v for k, v in dict(config.UNET).items()}
        spec_dict.update(**kwargs)
        in_channels = spec_dict['in_channels']
        base_channels = spec_dict['base_channels']
        num_stages = spec_dict['num_stages']
        strides = spec_dict['strides']
        enc_num_convs = spec_dict['enc_num_convs']
        dec_num_convs = spec_dict['dec_num_convs']
        downsamples = spec_dict['downsamples']
        enc_dilations = spec_dict['enc_dilations']
        dec_dilations = spec_dict['dec_dilations']
        norm_type = spec_dict['norm_type']
        act_type = spec_dict['act_type']
        upsample_type = spec_dict['upsample_type']

        assert len(strides) == num_stages, \
            'The length of strides should be equal to num_stages, '\
            f'while the strides is {strides}, the length of '\
            f'strides is {len(strides)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(enc_num_convs) == num_stages, \
            'The length of enc_num_convs should be equal to num_stages, '\
            f'while the enc_num_convs is {enc_num_convs}, the length of '\
            f'enc_num_convs is {len(enc_num_convs)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(dec_num_convs) == (num_stages-1), \
            'The length of dec_num_convs should be equal to (num_stages-1), '\
            f'while the dec_num_convs is {dec_num_convs}, the length of '\
            f'dec_num_convs is {len(dec_num_convs)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(downsamples) == (num_stages-1), \
            'The length of downsamples should be equal to (num_stages-1), '\
            f'while the downsamples is {downsamples}, the length of '\
            f'downsamples is {len(downsamples)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(enc_dilations) == num_stages, \
            'The length of enc_dilations should be equal to num_stages, '\
            f'while the enc_dilations is {enc_dilations}, the length of '\
            f'enc_dilations is {len(enc_dilations)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(dec_dilations) == (num_stages-1), \
            'The length of dec_dilations should be equal to (num_stages-1), '\
            f'while the dec_dilations is {dec_dilations}, the length of '\
            f'dec_dilations is {len(dec_dilations)}, and the num_stages is '\
            f'{num_stages}.'

        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.base_channels = base_channels

        if norm_type == 'bn':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.out_dims = []

        for i in range(num_stages):
            enc_conv_block = []
            if i != 0:
                if strides[i] == 1 and downsamples[i - 1]:
                    enc_conv_block.append(nn.MaxPool2d(kernel_size=2))
                upsample = (strides[i] != 1 or downsamples[i - 1])
                self.decoder.append(
                    UpConvBlock(
                        conv_block=BasicConvBlock,
                        in_channels=base_channels * 2**i,
                        skip_channels=base_channels * 2**(i - 1),
                        out_channels=base_channels * 2**(i - 1),
                        num_convs=dec_num_convs[i - 1],
                        stride=1,
                        dilation=dec_dilations[i - 1],
                        norm_layer=norm_layer,
                        act_type=act_type,
                        upsample_layer=InterpConv if upsample_type == 'interp' else DeconvModule,
                    )
                )

            enc_conv_block.append(
                BasicConvBlock(
                    in_channels=in_channels,
                    out_channels=base_channels * 2**i,
                    num_convs=enc_num_convs[i],
                    stride=strides[i],
                    dilation=enc_dilations[i],
                    norm_layer=norm_layer,
                    act_type=act_type,
                ))

            self.encoder.append((nn.Sequential(*enc_conv_block)))
            in_channels = base_channels * 2**i
            self.out_dims.append(in_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        self._check_input_divisible(x)
        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
        dec_outs = [x]
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)

        return dec_outs

    def _check_input_divisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 1
        for i in range(1, self.num_stages):
            if self.strides[i] == 2 or self.downsamples[i - 1]:
                whole_downsample_rate *= 2
        assert (h % whole_downsample_rate == 0) \
            and (w % whole_downsample_rate == 0),\
            f'The input image size {(h, w)} should be divisible by the whole '\
            f'downsample rate {whole_downsample_rate}, when num_stages is '\
            f'{self.num_stages}, strides is {self.strides}, and downsamples '\
            f'is {self.downsamples}.'
