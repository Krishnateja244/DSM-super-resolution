import torch
import torch.nn as nn
import torchvision
from torchview import draw_graph
import segmentation_models_pytorch as smp

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
        # self.PS1 = PixelShuffle(features*4, features*8,2)
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
    # x6 = self.mid_layer(x5+x1)
    # x7 = self.PS1(x6)
    # x8 = self.PS2(x7)
    # return self.final_layer(x8)
    x11 = self.mid_layer(x10+x1)
    # x11 = self.attention(x11)
    # x11 = self.multihead_attention(x11) 
    if self.scale ==2:
        # x12 = self.PS1(x11)
        x13 = self.PS2(x11)
    else:
        x12 = self.PS1(x11)
        x13 = self.PS2(x12)
    return self.final_layer(x13)

#### encoder ##################

# class _Residual_Block(nn.Module): 
#     def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
#         super(_Residual_Block, self).__init__()
        
#         midc=int(outc*scale)
        
#         # if inc is not outc:
#         #   self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
#         # else:
#         #   self.conv_expand = None
          
#         self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=2, padding=1, groups=groups, bias=False)
#         # self.bn1 = nn.BatchNorm2d(midc)
#         self.relu1 = nn.LeakyReLU(0.2, inplace=False)
#         # self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
#         # self.bn2 = nn.BatchNorm2d(outc)
#         # self.relu2 = nn.LeakyReLU(0.2, inplace=False)
        
#     def forward(self, x): 
#         if self.conv_expand is not None:
#           identity_data = self.conv_expand(x)
#         else:
#           identity_data = x

#         output = self.relu1(self.bn1(self.conv1(x)))
#         # output = self.conv2(output)
#         # output = self.relu2(self.bn2(torch.add(output,identity_data)))
#         return output

# class Encoder(nn.Module):
#     def __init__(self, cdim=3, hdim=512, channels=[128,256], image_size=256):
#         super(Encoder, self).__init__() 
        
#         # assert (2 ** len(channels)) * 4 == image_size
        
#         self.hdim = hdim
#         cc = channels[0]
#         self.main = nn.Sequential(
#                 nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
#                 # nn.BatchNorm2d(cc),
#                 nn.LeakyReLU(0.2),                
#                 # nn.AvgPool2d(2),
#               )
              
#         sz = image_size//2
#         for ch in channels[1:]:
#             self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
#             # self.main.add_module('down_to_{}'.format(sz//2), nn.AvgPool2d(2))
#             cc, sz = ch, sz//2
        
#         self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))    
#         self.cov =  nn.Sequential(
#             nn.Conv2d(256,64,3,1,1),
#             nn.LeakyReLU(0.2)
#         )   
#         self.generator = SrganGenerator(64,128,4)            
#         # self.fc = nn.Linear((cc)*4*4, 2*hdim)           
    
#     def forward(self, x):        
#         y = self.main(x)
#         y = self.cov(y)
#         print(y.shape)
#         y = self.generator(y)
#         # y = self.fc(y)
#         # mu, logvar = y.chunk(2, dim=1)                
#         return y
class Encoder(nn.Module):
    def __init__(self, in_channels=1, num_filters=128):
        super(Encoder, self).__init__()

        self.encoder_blocks = nn.Sequential(
            self._conv_block(in_channels, num_filters, batch_norm=False),  # Initial conv without batch norm
            self._conv_block(num_filters, num_filters,batch_norm=False),
            # self._conv_block(num_filters * 2, num_filters * 4),
            # self._conv_block(num_filters * 4, num_filters * 8)
        )
        self.decoder = SrganGenerator(128,128,4) #newM has 1 channel as input to decoder but newM_2 has 128 channels
        # self.last_layer = nn.Conv2d(1,1,3,1,1) # added for newM_2   removed in NewM3
        
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
        out = self.decoder(encoded_features) #added +x in newM_2 removed for newM3
        # out = self.last_layer(out) #added for new M_2 , removed in NewM3
        return out

class Multi_SrganGen(nn.Module):

    def __init__(self,in_channels,features) -> None:
        super(Multi_SrganGen,self).__init__()

        self.srgan = SrganGenerator(in_channels,features)
        self.mid_layer = nn.Sequential(
            nn.Conv2d(features, features*2, 3, 1, 1),
            nn.PReLU(),
        )
        self.g1 = nn.Sequential(*list(self.srgan.children())[:-4],self.mid_layer,PixelShuffle(256,1024,2))
        self.g2 = nn.Sequential(*list(self.srgan.children())[:-4],self.mid_layer)

        self.PS2 = PixelShuffle(features*2, features*4,2)

        self.final_layer = nn.Sequential(
            nn.Conv2d(features, in_channels, 3, 1, 1),
        )

    def forward(self,inp1,inp2):
        x1 = self.g1(inp1)
        x2 = self.g2(inp2)
        feat = x1+x2
        x3 = self.PS2(feat)
        return self.final_layer(x3)

class SRCNN(nn.Module):
    def __init__(self, num_channels=1,feat_channels=64):
        super(SRCNN, self).__init__()
        # self.firstlayer = nn.Conv2d(num_channels, feat_channels, kernel_size=3,stride=1,padding=1)
        # # self.rgbfirstlayer = nn.Conv2d(3,64,3,1,1)
        # self.conv_block = nn.Sequential(
        #     nn.Conv2d(feat_channels, feat_channels, kernel_size=3, stride=1,padding=1),
        #     nn.PReLU(),
        #     nn.Conv2d(feat_channels, feat_channels, kernel_size=3,stride=1, padding=1))
        # self.identity_block = nn.Conv2d(64, feat_channels, kernel_size=1,stride=1, padding=0)
        # self.conv_block_2 = nn.Sequential(
        #     nn.Conv2d(feat_channels,feat_channels,1,1,0),
        #     nn.PReLU(),
        #     nn.Conv2d(feat_channels,32,3,1,1),
        #     nn.PReLU(),
        #     nn.Conv2d(32,32,3,1,1)
        # )
        # self.conv_block_3 = nn.Sequential(
        #     nn.Conv2d(32,32,3,1,1),
        #     nn.PReLU(),
        #     nn.Conv2d(32,32,3,1,1),
        # )
        # self.last_layer = nn.Conv2d(32,1,1,1,0)
        
        # # self.rgb_last_layer = nn.Sequential(
        # #     nn.Conv2d(feat_channels,32,3,1,1),
        # #     nn.PReLU(),
        # #     nn.Conv2d(32,1,1,1,0)
        # # )
        self.srgan = SrganGenerator(1,128,4)
        self.feature_extractor =  torch.nn.Sequential(
                smp.Unet('resnet50', classes=1, in_channels=3))
                # torch.nn.AvgPool2d(kernel_size=2, stride=2) )

        self.last_afconcat = nn.Sequential(
            nn.Conv2d(1,1,3,1,1),
            nn.PReLU(),
            nn.Conv2d(1,1,3,1,1),
        )
        
    def forward(self, x,im):
        # x1 = self.firstlayer(x)
        # # x1_rgb = self.rgbfirstlayer(im)
        # #print(x1_rgb.shape)
        # x2 = self.conv_block(x1)
        # # x2_rgb = self.conv_block(x1_rgb)
        # # x3_rgb = self.identity_block(x2_rgb+x1_rgb)
        # # x4_rgb = self.conv_block(x3_rgb)
        # # #print(x4_rgb.shape)
        # # x5_rgb = self.identity_block(x3_rgb+x4_rgb)
        # # x6_rgb = self.identity_block(x5_rgb+x1_rgb)

        # # print(x6_rgb.shape)
        # x6_rgb = self.feature_extractor(im)
        
        # x3 = self.identity_block(x2+x1) #+x6_rgb.detach())

        # x4 = self.conv_block(x3)
        
        # x5 = self.identity_block(x3+x4)
        # x6 = self.conv_block(x5)
        # x7 = self.identity_block(x5+x6+x1)
        # x8 = self.conv_block_2(x7)
        # x9 = self.conv_block_3(x8)
        # x10 = self.last_layer(x9)
        # x10 = self.last_afconcat(x10+x6_rgb.detach())
        # x6_rgb = self.rgb_last_layer(x6_rgb)

        x1 = self.srgan(x)
        x6_rgb = self.feature_extractor(im)
        x10 = self.last_afconcat(x1+x6_rgb.detach())
       
        return x10,x6_rgb
        
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
    
#### pix2pixhd generator

class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]      

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]#, nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)] #, nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)  

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


########### kSENIA CODE ######################3
class ChannelAttention(nn.Module):
    def init(self,in_planes,ratio=16):
        super(ChannelAttention,self).init()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.conv2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv2(self.relu1(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu1(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def init(self, kernel_size=7):
        super(SpatialAttention, self).init()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

#####################################################
if __name__ == "__main__":
    
    gen = SrganGenerator(1,128,4)
    # gen_2 = Encoder(1,128)
    gen_2 = SRCNN(1,64) #SrganGenerator(1,128,4)
    #print(gen_2)
    # gen_2 = nn.Sequential(*list(gen.children())[:-3], PixelShuffle(512,2048,2))
    # gen_3 = nn.Sequential(*list(gen.children())[:-2],nn.Conv2d(256,1,3,1,1))
    # print(gen_3)
    x1 = torch.randn(8,1,64,64)
    x2 = torch.randn(8,3,256,256)
    # print(gen_2)
    out = gen(x1)
    #out,out_2 = gen_2(x1,x2)
    print(out.shape) 
    # print(out_2.shape)
    #model_graph= draw_graph( SRCNN(1,64),input_size=[(1,1,64,64),(1,3,256,256)], expand_nested=True)
    #model_graph.visual_graph.render("srgan_unet_rgb.png",format='png')
    # out_2 = gen_2(x)
    # out_3 = gen_3(x2)
    # print(out_2.shape)
    # print(out_3.shape)
    # gen = Multi_SrganGen(1,128)
    # out = gen(x1,x2)
    # print(out.shape)
