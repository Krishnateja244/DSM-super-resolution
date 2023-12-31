"""
This code base was taken from the 
https://github.com/d-li14/efficientnetv2.pytorch
"""

import torch
import torch.nn as nn
import math
from torchsummary import summary
import cv2

__all__ = ['effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl']


def _make_divisible(v, divisor, min_value=None):
    """ Function ensures that all layers have a channel number that is divisible by 8

    Args:
        v (int): value
        divisor (int): divisor. Defaults to 8
        min_value (_type_, optional): _description_. Defaults to None.

    Returns:
        new_v: new value
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

 
class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.LeakyReLU(0.2, True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
    )

def interpolate(x,height,width,scale):
    batched = torch.FloatTensor(x.shape[0], 1, x.shape[2]*scale, x.shape[-1]*scale)
    for i in range(0,x.shape[0]):
        reshaped = cv2.resize(x[i].reshape(x[i].shape[1],x[i].shape[2],1),(height*scale,width*scale), interpolation=cv2.INTER_NEAREST)
        batched[i] = torch.tensor(reshaped.reshape(1,height*scale,width*scale))
    return batched.cuda()


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.LeakyReLU(0.2),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.LeakyReLU(0.2),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.LeakyReLU(0.2),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            )


    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
class Self_Attention (nn.Module):
      """ Defines the Self-attention mechanism"""
      def __init__(self, in_dim):
        """ Constructs the self-attention mechanism

        Args:
            in_dim (int): feature layer dimension
        """
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
    """ Defines the Multi-head attention mechanism"""

    def __init__(self, in_channels, num_heads=8):
        """ Constructs the multi-ead attention mechanism

        Args:
            in_channels (int): channels dimension
            num_heads (int, optional): number of heads for multi-head mechanism. Defaults to 8.
        """
        super(MultiHeadSelfAttention, self).__init__() 
        self.heads = nn.ModuleList([Self_Attention(in_channels) for _ in range(num_heads)])
        self.conv = nn.Conv2d(num_heads * in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        multi_head_output = torch.cat(head_outputs, dim=1)
        out = self.conv(multi_head_output)
        return out
      
class EffNetV2(nn.Module):
    """ Defines the EffectientNetv2 model"""

    def __init__(self, cfgs, width_mult=1.):
        """COnstructs the EffNetv2 model

        Args:
            cfgs (list): model configs
            width_mult (float, optional): width multiplier. Defaults to 1..
        """
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        
        layers = [conv_3x3_bn(1, input_channel, 1)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # self.multihead = MultiHeadSelfAttention(input_channel)
        self.pixel_suffel = nn.PixelShuffle(2) 
        self.conv = conv_1x1_bn(1,1)

    def forward(self, x):
        
        x_int = x
        x = self.features(x)
        # x = self.multihead(x)
        x = self.pixel_suffel(x)
        x = self.pixel_suffel(x)
    
        x = x + interpolate(x_int.cpu().detach().numpy(),x_int.shape[-1],x_int.shape[-1],4)
        x = self.conv(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def effnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  2, 1, 0],
        [4,  48,  4, 1, 0],
        [4,  64,  4, 1, 0],
        [4, 64,  4, 1, 1],
        [6, 160,  9, 1, 1],
        [6, 16, 15, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_m(**kwargs):
    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  3, 1, 0],
        [4,  48,  5, 2, 0],
        [4,  80,  5, 2, 0],
        [4, 160,  7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512,  5, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_l(**kwargs):
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  7, 2, 0],
        [4,  96,  7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640,  7, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_xl(**kwargs):
    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  8, 2, 0],
        [4,  96,  8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640,  8, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)

if __name__=="__main__":
    
    x = torch.randn(1,1,64,64).cuda()
    print(x.shape)
    model = effnetv2_s().cuda()
    print(model)
    out = model(x)
    print(out.shape)
 
