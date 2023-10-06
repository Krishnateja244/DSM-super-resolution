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

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        self.rpi = self.calculate_rpi_sa()

    def forward(self, x,mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        x_windows = self.window_partition(x, self.window_size[0])  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[0], 1)
        x = x_windows
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        attn_windows = x
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[0], c)
        shifted_x = self.window_reverse(attn_windows, self.window_size[0], h, w)  # b h' w' c

        # reverse cyclic shift
        
        attn_x = shifted_x
        attn_x = attn_x.view(b, h * w, c)
        x = attn_x
        return x

    def calculate_rpi_sa(self):
        # calculate relative position index for SA
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[0])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[0] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[0] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    def window_partition(self,x, window_size):
        """
        Args:
            x: (b, h, w, c)
            window_size (int): window size

        Returns:
            windows: (num_windows*b, window_size, window_size, c)
        """
        b, h, w, c = x.shape
        x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
        return windows
    
    def window_reverse(windows, window_size, h, w):
        """
        Args:
            windows: (num_windows*b, window_size, window_size, c)
            window_size (int): Window size
            h (int): Height of image
            w (int): Width of image

        Returns:
            x: (b, h, w, c)
        """
        b = int(windows.shape[0] / (h * w / window_size / window_size))
        x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
        return x

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
        # self.window_atten = WindowAttention(in_channels,window_size=(16,16),num_heads=6)
        # self.self_atten = Self_Attention(in_channels)
        
    def forward(self,x):
        
        return x+self.block(x)#+self.self_atten(x) #+self.window_atten(x)


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




