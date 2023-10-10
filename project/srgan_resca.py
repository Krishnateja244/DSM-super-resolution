import torch
import torch.nn as nn
import torchvision

# residual blocks
class CNNBlocks(nn.Module):
    def __init__(self, in_channels ):
        super(CNNBlocks, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        )
    def forward(self, x):
        return self.conv(x)+x

# upsample blocks 
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
        return out
      
# residual channel attention
class Resca(nn.Module):
    def __init__(self,in_channels,reduction):
        super(Resca,self).__init__()
        self.block_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels,in_channels//reduction,1),
            nn.PReLU(),
            nn.Conv2d(in_channels//reduction,in_channels,1),
        )
        self.block_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels,in_channels//reduction,1),
            nn.PReLU(),
            nn.Conv2d(in_channels//reduction,in_channels,1),
        )
        self.sig = nn.Sigmoid()

    def forward(self,x):
        x1 = self.block_max(x)
        x2 = self.block_avg(x)
        x3 = x1+x2
        x4 = self.sig(x3)

        return x4*x

class ResCAB(nn.Module):
    def __init__(self,in_channels,reduction):
        super(ResCAB,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,1,1),
            nn.PReLU(),
            nn.Conv2d(in_channels,in_channels,3,1,1),
            Resca(in_channels,reduction)
        )
        # self.self_atten = Self_Attention(in_channels)
        
    def forward(self,x):
        
        return x+self.block(x)#+self.self_atten(x) 


class SRCAGAN(nn.Module):
    def __init__(self,in_channels,features,res_blocks,atten_blocks,reduction):
        super(SRCAGAN,self).__init__()

        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels,features,3,1,1)
        )
        blocks = [CNNBlocks(features) for i in range(res_blocks)]
        self.residual_blocks = nn.Sequential(*blocks)
        
        self.mid_conv = nn.Conv2d(features,features,3,1,1)

        self.res_atten = [ResCAB(features,reduction) for i in range(atten_blocks)]
        self.atten_blocks = nn.Sequential(*self.res_atten)
        # self.self_atten = Self_Attention(features)
        self.mid_layer = nn.Sequential(
            nn.Conv2d(features, features, 3, 1, 1),
        )
        self.PS1 = PixelShuffle(features, features*4,2)
        self.PS2 = PixelShuffle(features,features*4,2)
        self.final_layer = nn.Sequential(
            nn.Conv2d(features, in_channels, 3, 1, 1)
        )
    def forward(self,x):
        x1 = self.first_layer(x)
        x2 = self.residual_blocks(x1)
        x3 = self.mid_conv(x2)
        x4 = self.atten_blocks(x3+x)
        # x4 = self.self_atten(x4)
        x5 = self.mid_layer(x4+x)
        x5 = self.PS1(x5)
        x7 = self.PS2(x5)
        return self.final_layer(x7)

if __name__ == "__main__":
    
    gen = SRCAGAN(1,128,0,10,8)
    print(gen)
    x = torch.randn(8,1,64,64)
    out = gen(x)
    print(out.shape) 




