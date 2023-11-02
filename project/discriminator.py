import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SpectralNorm
import functools
import numpy as np 
import torch.nn.functional as F

class Blocks(nn.Module):
  """ Defines the CNN blocks for the model """

  def __init__(self, in_channels, out_channels, stride):
    """ Constructs CNN blocks 

    Args:
        in_channels (int): the number of input channels 
        out_channels (_type_): the number of out channels
        stride (int): stride number 
    """
    super(Blocks, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        ## to use spectral normalization instead of batch normalization
        # SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)),
        nn.LeakyReLU(0.2),
    )

  def forward(self, x):
    return self.conv(x)  

class Self_Attention (nn.Module):
      """ Defines the self-attention mechanism"""

      def __init__(self, in_dim):
        """Constructs the self-attention 

        Args:
            in_dim (int): number of input channels
        """
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
        return out, attention

class Discriminator(nn.Module):
  """ Defines the pixel based discriminator """
  def __init__(self, in_channels, features):
    """COnstructs the pixel discriminator

    Args:
        in_channels (int): number of input channels
        features (int): number of feature channels 
    """
    super(Discriminator, self).__init__()
    self.first_layer= nn.Sequential(
        nn.Conv2d(in_channels, features, 3, 2 ,1),
        nn.LeakyReLU(0.2),
    )
    self.Block1 = Blocks(features, features*2, stride=2)
    self.Block2 = Blocks(features*2, features*2, stride=1)
    self.Block3 = Blocks(features*2, features*4, stride=2)
    self.Block4 = Blocks(features*4, features*4, stride=1)
    self.Block5 = Blocks(features*4, features*8, stride=2)
    self.Block6 = Blocks(features*8, features*8, stride=1)
    self.Block7 = Blocks(features*8, features*8, stride=2)
    self.Block8 = Blocks(features*8, features*8, stride=2)
    # self.attention = Self_Attention(features*8)
    self.Block9 = nn.Sequential(
        nn.Conv2d(features*8, features*4, 3, 2, 1),
        
        nn.LeakyReLU(0.2),
    )
    self.final_layer = nn.Sequential(
        nn.Linear(features*4, 1),
    )

  def forward(self, x):
    x =  self.first_layer(x)
    x =  self.Block1(x)
    x =  self.Block2(x)
    x =  self.Block3(x)
    x =  self.Block4(x)
    x =  self.Block5(x)
    x =  self.Block6(x)
    x =  self.Block7(x)
    x =  self.Block8(x)
    # x,_ = self.attention(x) # to use the self-attention for discriminator
    x = self.Block9(x)
    x = x.view(x.size(0), -1)
    return self.final_layer(x)

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                #SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            #SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
        
class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = SpectralNorm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out
    
def test():
  x = torch.randn(8,1,256,256)
#   disc = Discriminator(1,128)
  # disc = NLayerDiscriminator(1,64)
  disc = UNetDiscriminatorSN(1,64,skip_connection=False)
  print(disc)
  out = disc(x)
  print(out.shape)
  
#   print(out.shape)

if __name__ == "__main__":
    test()    