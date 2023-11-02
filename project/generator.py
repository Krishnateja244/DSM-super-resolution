import torch
import torch.nn as nn
import torchvision
from torchview import draw_graph
import segmentation_models_pytorch as smp
import cv2

class CNNBlocks(nn.Module):
  def __init__(self, in_channels ):
    super(CNNBlocks, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        # nn.BatchNorm2d(in_channels),
        nn.PReLU(),
        nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        # nn.BatchNorm2d(in_channels),
    )
  def forward(self, x):
      return self.conv(x)+x


class PixelShuffle(nn.Module):
  def __init__(self, in_channels, out_channels, upscale_factor):
    super(PixelShuffle, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels,out_channels, 3, 1, 1),
        nn.PixelShuffle(upscale_factor),
        nn.PReLU(),
    )
  def forward(self,x):
    return self.conv(x)

class SrganGenerator(nn.Module):
  def __init__(self, in_channels, features,scale):
    super(SrganGenerator, self).__init__()
    self.first_layer = nn.Sequential(
        nn.Conv2d(in_channels, features, 3, 1, 1),
        nn.PReLU(),
    )
    self.RB1 = CNNBlocks(features)
    self.RB2 = CNNBlocks(features)
    self.RB3 = CNNBlocks(features)
    self.RB4 = CNNBlocks(features)
    # These are added by me (kt)
    self.RB5 = CNNBlocks(features)
    self.RB6 = CNNBlocks(features)
    self.RB7 = CNNBlocks(features)
    self.RB8 = CNNBlocks(features)
    self.scale = scale

    if self.scale == 2:
        self.mid_layer = nn.Sequential(
            nn.Conv2d(features, features*2, 3, 1, 1),
            nn.PReLU(),
        )
        # self.attention = Self_Attention(features*2)
        # self.multihead_attention = MultiHeadSelfAttention(features*2)
        self.PS2 = PixelShuffle(features*2, features*4,2)
    else:
        self.mid_layer = nn.Sequential(
            nn.Conv2d(features, features*4, 3, 1, 1),
            nn.PReLU(),
        )
        # self.attention = Self_Attention(features*4)
        # self.multihead_attention = MultiHeadSelfAttention(features*4)
        self.PS1 = PixelShuffle(features*4, features*8,2)
        self.PS2 = PixelShuffle(features*2, features*4,2)

    self.final_layer = nn.Sequential(
        nn.Conv2d(features, 1, 3, 1, 1),
        # nn.Tanh(),
        # nn.PReLU()
    )


  def forward(self, x):
    x1 = self.first_layer(x)
    x2 = self.RB1(x1)
    x3 = self.RB2(x2)
    x4 = self.RB3(x3)
    x5 = self.RB4(x4)
    x6 = self.RB5(x5)
    x7 = self.RB5(x6)
    x8 = self.RB6(x7)
    x9 = self.RB7(x8)
    x10 = self.RB8(x9)
    x11 = self.mid_layer(x10+x1)
    # x11 = self.attention(x11)
    # x11 = self.multihead_attention(x11) 
    if self.scale ==2:
        x13 = self.PS2(x11)
    else:
        x12 = self.PS1(x11)
        x13 = self.PS2(x12)
    return self.final_layer(x13)

class SRGAN_rgb(nn.Module):
    def __init__(self, num_channels=1,feat_channels=64):
        super(SRGAN_rgb, self).__init__()
        self.srgan = SrganGenerator(1,64,4)
        self.feature_extractor =  torch.nn.Sequential(
                smp.Unet('resnet50', classes=64, in_channels=4))
        self.upsample_torch = torch.nn.Upsample(scale_factor=4, mode='bicubic')
        self.last_afconcat = nn.Sequential(
            nn.Conv2d(64,1,3,1,1),
        )

    def forward(self, x,im):
        x1 = self.srgan(x)
        x_bicubic = self.upsample_torch(x)
        x2_rgb = self.feature_extractor(torch.cat([im, x_bicubic-x_bicubic.mean((1,2,3), keepdim=True) ], 1))
        x2 = self.last_afconcat(x1+x2_rgb.detach())
        x3_rgb = self.last_afconcat(x2_rgb)
        return x2,x3_rgb

class Encoder(nn.Module):
    def __init__(self, in_channels=1, num_filters=128):
        super(Encoder, self).__init__()

        self.encoder_blocks = nn.Sequential(
            self._conv_block(in_channels, num_filters, batch_norm=False),  
            self._conv_block(num_filters, num_filters,batch_norm=False),
        )
        self.decoder = SrganGenerator(128,128,4) 
        
    def _conv_block(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, batch_norm=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        encoded_features = self.encoder_blocks(x)
        out = self.decoder(encoded_features) 
        return out
        
class Self_Attention (nn.Module):
      def __init__(self, in_dim):
        super(Self_Attention,self).__init__()
        self.query_conv=nn.Conv2d(in_dim, in_dim//16, kernel_size=1)
        self.key_conv=nn.Conv2d(in_dim, in_dim//16, kernel_size=1)
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
    
#####################################################
if __name__ == "__main__":
    
    gen = SrganGenerator(1,128,4)
    x1 = torch.randn(8,1,64,64)
    print(gen)
    out = gen(x1)
    print(out.shape) 
