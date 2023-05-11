import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.involution_naive import involution
from utils.antialias import Downsample as downsamp
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
import torch.utils.checkpoint as cp

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

class single_conv_3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv_3, self).__init__()     #将父类的初始化内容放在子类的初始化内容中
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),     #卷积操作，参数1是输入图像的通道数，参数2是输出特征图的通道数，3指的是卷积核大小，padding=1代表在特征图外边包裹了一层0
            nn.PReLU()
        )       #相当于将许多模块整合，使forward里面调用时可以少一些调用操作

    def forward(self, x):       #这里是在使用single_conv(x)时会调用Module中的__call__函数，call调用forward，而forward在子类中又重写了
        return self.conv(x)     #所以会自动调用forward函数

class single_conv_1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv_1, self).__init__()     #将父类的初始化内容放在子类的初始化内容中
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),     #卷积操作，参数1是输入图像的通道数，参数2是输出特征图的通道数，3指的是卷积核大小，padding=1代表在特征图外边包裹了一层0
            nn.PReLU()
        )       #相当于将许多模块整合，使forward里面调用时可以少一些调用操作

    def forward(self, x):       #这里是在使用single_conv(x)时会调用Module中的__call__函数，call调用forward，而forward在子类中又重写了
        return self.conv(x)     #所以会自动调用forward函数

##########################################################################
##---------- Dual Complementary Feature Extraction (DCFE) ----------
class Bottleneck(nn.Module):
    """Bottleneck block for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of conv2. Default: 4.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None.
        style (str): ``"pytorch"`` or ``"caffe"``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: "pytorch".
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = involution(self.mid_channels, 3, self.conv2_stride)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out
class DCFE(nn.Module):
    def __init__(self, in_ch):
        super(DCFE, self).__init__()

        self.conv_stream = nn.Sequential(
            single_conv_1(in_ch, in_ch),
            single_conv_3(in_ch, in_ch),
            single_conv_1(in_ch, in_ch)
        )
        # self.inv_stream = nn.Sequential(
        #     single_conv_1(in_ch, in_ch*4),
        #     Bottleneck(in_ch*4, in_ch*4),
        #     single_conv_1(in_ch*4, in_ch)
        #)
    def forward(self, x):
        out = self.conv_stream(x)
        # inv_out = self.inv_stream(x)
        # out = out + inv_out
        return out



##########################################################################
##---------- Resizing Feature Map ----------
class ResidualDownSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualDownSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels,   1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias),
                                nn.PReLU(),
                                downsamp(channels=in_channels,filt_size=3,stride=2),
                                nn.Conv2d(in_channels, in_channels*2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(downsamp(channels=in_channels,filt_size=3,stride=2),
                                nn.Conv2d(in_channels, in_channels*2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top+bot
        return out

class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualDownSample(in_channels))
            in_channels = int(in_channels * stride)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class ResidualUpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualUpSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels,   1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1,bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, in_channels//2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
                                nn.Conv2d(in_channels, in_channels//2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top+bot
        return out

class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualUpSample(in_channels))
            in_channels = int(in_channels // stride)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

##########################################################################
##---------- Collaborative-attention-guided Feature Fusion ----------
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
       
class CGFF(nn.Module):
    def __init__(self, in_numchw, height=3,reduction=8,bias=False):
        super(CGFF, self).__init__()
        
        self.height = height    #height为3代表三分支
        d = max(int(in_numchw/reduction),4)   #代表输入特征图的通道数压缩8倍
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)     #全局平均池化
        self.compress = ChannelPool()
        self.conv_du = nn.Sequential(nn.Conv2d(in_numchw, d, 1, padding=0, bias=bias), nn.PReLU())    #这个是对一维向量压缩通道的卷积

        self.fcs = nn.ModuleList([])    #这个是softmax前的三个卷积，此时通道数又变为输入通道数
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_numchw, kernel_size=1, stride=1,bias=bias))
        
        self.spatial = nn.ModuleList([])
        for i in range(self.height):
            self.spatial.append(BasicConv(2, 1, 5, stride=1, padding=2, relu=False))
        
        self.softmax = nn.Softmax(dim=1)
        self.final = nn.Conv2d(in_numchw*2, in_numchw, 1, padding=0, bias=bias)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]  #此时的inp维度排序为3，b，c，h，w
        n_feats =  inp_feats[0].shape[1]    #为c通道数
        n_feats_h =  inp_feats[0].shape[2]
        n_feats_w =  inp_feats[0].shape[3]

        inp_feats = torch.cat(inp_feats, dim=1)     #此时的维度变为b，3c，h，w
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        #这里是对inp维度进行重新排列，按照一行一行顺序排列下来，b不动，相当于原来的1×3c变为3×c，由于之前是级联得到3c，是有顺序的
        #因此按照view的格式排下来结构不会乱
        feats_U = torch.sum(inp_feats, dim=1)   #此时维度为b，c，h，w
        feats_S = self.avg_pool(feats_U)    #全局平均池化
        feats_Z = self.conv_du(feats_S)     #对池化得到的向量进一步卷积压缩通道数

        attention_vectors = [fc(feats_Z) for fc in self.fcs]    #一个列表，包含三个维度为（b，c，1，1）的向量
        attention_vectors = torch.cat(attention_vectors, dim=1)     #此时维度为b，3c，1，1
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)      #b，3，c，1，1
        # stx()
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)

        feats_compress = self.compress(feats_U)
        spatial_attention_vector = [torch.sigmoid(sp(feats_compress)) for sp in self.spatial]
        spatial_attention_vector = torch.cat(spatial_attention_vector, dim=1)
        spatial_attention_vector = spatial_attention_vector.view(batch_size, self.height, 1, n_feats_h, n_feats_w)
        feats_P = torch.sum(inp_feats*spatial_attention_vector, dim=1)*(1/3)
        

        feats_G = torch.cat([feats_V, feats_P], dim=1)
        feats_F = self.final(feats_G)
        
        return feats_F
        


##########################################################################
##---------- Multi-Scale Complementary Feature Fusion (MCFF) ----------
#height为3代表网络中的三分支，width为2代表网络中用了两个DCFE，stride为2代表上采样或者下采样时通道数变化的倍数，n_feat
#代表尺度为1的特征图的通道数
class MCFF(nn.Module):
    def __init__(self, n_feat, height, width, stride, bias):    
        super(MCFF, self).__init__()

        self.n_feat, self.height, self.width = n_feat, height, width
        self.blocks = nn.ModuleList([nn.ModuleList([DCFE(int(n_feat*stride**i))]*width) for i in range(height)])
        #二维模型列表，包含着模型中所有可能用到的DCFE
        INDEX = np.arange(0,width, 2)
        FEATS = [int((stride**i)*n_feat) for i in range(height)]    #代表三个支路中各自特征图的通道数
        SCALE = [2**i for i in range(1,height)]     #代表下采样或者上采样的尺度变换因子

        self.last_up   = nn.ModuleDict()    #定义一个模型字典，存放最后的上采样操作
        for i in range(1,height):
            self.last_up.update({f'{i}': UpSample(int(n_feat*stride**i),2**i,stride)})

        self.down = nn.ModuleDict()     #定义一个模型字典，存放中间尺度交互中的下采样操作
        self.up   = nn.ModuleDict()     #定义一个模型字典，存放中间尺度交互中的上采样操作

        i=0
        SCALE.reverse()     #反转列表
        for feat in FEATS:
            for scale in SCALE[i:]:
                self.down.update({f'{feat}_{scale}': DownSample(feat,scale,stride)})
            i+=1
        #这里的目的是将中间所有可能用到的下采样操作按照字典索引的方式统计一下
        i=0
        FEATS.reverse()
        for feat in FEATS:
            for scale in SCALE[i:]:                
                self.up.update({f'{feat}_{scale}': UpSample(feat,scale,stride)})
            i+=1
        #这里的目的是将中间所有可能用到的上采样操作按照字典索引的方法统计

        self.conv_out = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=bias)
        #模型最后用到的卷积

        
        self.selective_kernel = nn.ModuleList([CGFF(n_feat*stride**i, height) for i in range(height)])
        #这里是一个模型列表，里面存放的是中间尺度交互的三次CGFF
        


    def forward(self, x):
        inp = x.clone()     #复制一份输入，为了后续的残差做准备
        #col 1 only
        blocks_out = []     #这里存放的是经过第一步下采样+DCFE得到的三个特征图
        for j in range(self.height):
            if j==0:
                inp = self.blocks[j][0](inp)    #1×尺度不经过下采样，直接经过DCFE
            else:
                inp = self.blocks[j][0](self.down[f'{inp.size(1)}_{2}'](inp))   #另外两个支路需要下采样
            blocks_out.append(inp)

        #rest of grid
        for i in range(1,self.width):   #i仅取到1
            #Mesh
            # Replace condition(i%2!=0) with True(Mesh) or False(Plain)
            # if i%2!=0:
            if True:
                tmp=[]
                for j in range(self.height):
                    TENSOR = []
                    nfeats = (2**j)*self.n_feat
                    for k in range(self.height):
                        TENSOR.append(self.select_up_down(blocks_out[k], j, k)) 
                    #这里的两个循环是实现的是送入CGFF前完成的所有上采样和下采样
                    selective_kernel_fusion = self.selective_kernel[j](TENSOR)      #在不同尺度上送入3个CGFF
                    tmp.append(selective_kernel_fusion)     #存放的是3个CGFF的3个输出
            #Plain
            else:
                tmp = blocks_out
            #Forward through either mesh or plain
            for j in range(self.height):
                blocks_out[j] = self.blocks[j][i](tmp[j])      #过第二个DGFE

        #Sum after grid
        out=[]
        for k in range(self.height):
            out.append(self.select_last_up(blocks_out[k], k))  #根据所在支路进行最后的上采样

        out = self.selective_kernel[0](out)     #过最后的SKFF

        out = self.conv_out(out)
        out = out + x

        return out

    def select_up_down(self, tensor, j, k):     #该函数的功能是判断输入tebsor是需要上采样还是下采样还是不变
        if j==k:
            return tensor
        else:
            diff = 2 ** np.abs(j-k)
            if j<k:
                return self.up[f'{tensor.size(1)}_{diff}'](tensor)
            else:
                return self.down[f'{tensor.size(1)}_{diff}'](tensor)


    def select_last_up(self, tensor, k):
        if k==0:
            return tensor
        else:
            return self.last_up[f'{k}'](tensor)
            

##########################################################################
##---------- Scale Interaction Pyramid Block (SIPB) ----------
class SIPB(nn.Module):
    def __init__(self, n_feat, n_MCFF, height, width, stride, bias=False):      
        super(SIPB, self).__init__()

        module_body = [MCFF(n_feat, height, width, stride, bias) for _ in range(n_MCFF)]
        module_body.append(conv(n_feat, n_feat, kernel_size=3))
        self.body = nn.Sequential(*module_body)
        
        

    def forward(self, x):
        res = self.body(x)
        res += x

        return res


##########################################################################
##---------- SIPNet  -----------------------
class SIPNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=32, kernel_size=3, stride=2, n_MCFF = 2, n_SIPB=3, height=3, width=2, bias=False):
        super(SIPNet, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=bias)

        modules_body = [SIPB(n_feat, n_MCFF, height, width, stride, bias) for _ in range(n_SIPB)]
        self.body = nn.Sequential(*modules_body)

        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=bias)

    def forward(self, x):
        h = self.conv_in(x)
        h = self.body(h)
        h = self.conv_out(h)
        h += x
        return h















