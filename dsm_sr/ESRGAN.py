import torch.nn as nn
import torch

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf, gc=32, res_scale=0.2):
        super(ResidualDenseBlock, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(nf + 0 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(nf + 1 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(nf + 2 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(nf + 3 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer5 = nn.Sequential(nn.Conv2d(nf + 4 * gc, nf, 3, padding=1, bias=True), nn.LeakyReLU())

        self.res_scale = res_scale

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(torch.cat((x, layer1), 1))
        layer3 = self.layer3(torch.cat((x, layer1, layer2), 1))
        layer4 = self.layer4(torch.cat((x, layer1, layer2, layer3), 1))
        layer5 = self.layer5(torch.cat((x, layer1, layer2, layer3, layer4), 1))
        return layer5.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, nf, gc=32, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.layer1 = ResidualDenseBlock(nf, gc)
        self.layer2 = ResidualDenseBlock(nf, gc)
        self.layer3 = ResidualDenseBlock(nf, gc, )
        self.res_scale = res_scale

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out.mul(self.res_scale) + x


def upsample_block(nf, scale_factor=2):
    block = []
    for _ in range(scale_factor//2):
        block += [
            nn.Conv2d(nf, nf * (2 ** 2), 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU()
        ]

    return nn.Sequential(*block)

class ESRGAN(nn.Module):
    def __init__(self, in_channels, out_channels, nf=64, gc=32, scale_factor=4, n_basic_block=23):
        super(ESRGAN, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels,nf, 3,1,1))

        basic_block_layer = []

        for _ in range(n_basic_block):
            basic_block_layer += [ResidualInResidualDenseBlock(nf, gc)]

        self.basic_block = nn.Sequential(*basic_block_layer)

        self.conv2 = nn.Sequential(nn.Conv2d(nf, nf, 3,1,1))
        # self.multihead_atten = MultiHeadSelfAttention(nf)
        self.upsample = upsample_block(nf, scale_factor=scale_factor)
        self.conv3 = nn.Sequential(nn.Conv2d(nf, nf, 3,1,1),  nn.LeakyReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(nf, out_channels, 3,1,1))

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.basic_block(x1)
        x = self.conv2(x)
        # x = self.multihead_atten(x)
        x = self.upsample(x + x1)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class Self_Attention (nn.Module):
      def __init__(self, in_dim):
        super(Self_Attention,self).__init__()
        self.query_conv=nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.key_conv=nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.value_conv=nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma=nn.Parameter(torch.zeros(1))
        self.softmax=nn.Softmax(dim=-1)
      def forward(self,x):
        B,C,W,H=x.size()
        query=self.query_conv(x).view(B, -1, W*H)
        query=query.permute(0,2,1)
        key=self.key_conv(x).view((B, -1, W*H))
        energy = torch.bmm(query, key)
        attention=self.softmax(energy).permute(0,2,1)
        value = self.value_conv(x).view(B, -1, W*H)
        out = torch.bmm(value, attention).view(B, C, W,H)
        out= self.gamma*out+x
        return out #, attention

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__() 
        self.heads = nn.ModuleList([Self_Attention(in_channels) for _ in range(num_heads)])
        self.conv = nn.Conv2d(num_heads * in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        multi_head_output = torch.cat(head_outputs, dim=1)
        out = self.conv(multi_head_output)
        return out
    
######### ESRGAN PRETRAINED #################
# import functools
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# def make_layer(block, n_layers):
#     layers = []
#     for _ in range(n_layers):
#         layers.append(block())
#     return nn.Sequential(*layers)


# class ResidualDenseBlock_5C(nn.Module):
#     def __init__(self, nf=64, gc=32, bias=True):
#         super(ResidualDenseBlock_5C, self).__init__()
#         # gc: growth channel, i.e. intermediate channels
#         self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
#         self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
#         self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
#         self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
#         self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#         # initialization
#         # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

#     def forward(self, x):
#         x1 = self.lrelu(self.conv1(x))
#         x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
#         x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
#         x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
#         x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
#         return x5 * 0.2 + x


# class RRDB(nn.Module):
#     '''Residual in Residual Dense Block'''

#     def __init__(self, nf, gc=32):
#         super(RRDB, self).__init__()
#         self.RDB1 = ResidualDenseBlock_5C(nf, gc)
#         self.RDB2 = ResidualDenseBlock_5C(nf, gc)
#         self.RDB3 = ResidualDenseBlock_5C(nf, gc)

#     def forward(self, x):
#         out = self.RDB1(x)
#         out = self.RDB2(out)
#         out = self.RDB3(out)
#         return out * 0.2 + x


# class RRDBNet(nn.Module):
#     def __init__(self, in_nc, out_nc, nf, nb, gc=32):
#         super(RRDBNet, self).__init__()
#         RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

#         self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
#         self.RRDB_trunk = make_layer(RRDB_block_f, nb)
#         self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         #### upsampling
#         self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#     def forward(self, x):
#         fea = self.conv_first(x)
#         trunk = self.trunk_conv(self.RRDB_trunk(fea))
#         fea = fea + trunk

#         fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         out = self.conv_last(self.lrelu(self.HRconv(fea)))

#         return out


######### Real ESRGAN pretrained ##
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from arch_util import default_init_weights, make_layer, pixel_unshuffle


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x



class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
    
if __name__ == "__main__":
    model = RRDBNet(1,1,scale=4,num_feat=64,num_block=23,num_grow_ch=32)#ESRGAN(1,1,scale_factor=4) #
    print(model)
    x = torch.randn(4,1,64,64)
    out = model(x)
    print(out.shape)