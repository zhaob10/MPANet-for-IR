
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.self_attention_1 import WindowAttention
from torch.nn import Softmax


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()     #将父类的初始化内容放在子类的初始化内容中
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),     #卷积操作，参数1是输入图像的通道数，参数2是输出特征图的通道数，3指的是卷积核大小，padding=1代表在特征图外边包裹了一层0
            nn.LeakyReLU(0.2, inplace=False)
        )       #相当于将许多模块整合，使forward里面调用时可以少一些调用操作

    def forward(self, x):       #这里是在使用single_conv(x)时会调用Module中的__call__函数，call调用forward，而forward在子类中又重写了
        return self.conv(x)     #所以会自动调用forward函数


class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)      #反卷积实现上采样，经过计算可知，此处实现的是2倍上采样

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]     #size()取的是这个tensor的尺寸，以元组的形式返回，这里计算的是x2和x1之间空间尺寸的差距
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))      #对x1进行填充，输入四维的元组分别代表上、下、左右四个方向填充的大小

        x = x2 + x1
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)     #这个卷积是用来最后输出恢复结果时用到的1×1卷积

    def forward(self, x):
        x = self.conv(x)
        return x

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
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

class BasicINConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, relu_slope=0.2, use_IN = False):
        super(BasicINConv, self).__init__()
        self.indentity = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_IN:
            self.norm = nn.InstanceNorm2d(out_channels//2, affine=True)
        self.use_IN = use_IN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_IN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.indentity(x)
        return out

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class DCFE_Branch(nn.Module):
    def __init__(self, is_chw, in_size):
        super(DCFE_Branch, self).__init__()
        self.is_chw = is_chw
        self.conv1 = BasicINConv(in_size, in_size, use_IN=False)
        self.compress_h = ChannelPool()
        self.compress_w = ChannelPool()
        self.compress_c = ChannelPool()
        self.conv2 = BasicINConv(4, 4, use_IN=True)
        self.conv3 = BasicConv(2, 1, 3, 1, 1, relu=False,bn=False)
    
    def forward(self, x):
        if self.is_chw == 0:
            h_perm1 = x.permute(0,2,1,3).contiguous()
            h_perm2 = x.permute(0,2,1,3).contiguous()
            h_mid1 = self.compress_h(h_perm1)
            h_mid2 = self.compress_h(h_perm2)
            h_mid = torch.cat([h_mid1, h_mid2], dim=1)
            h_mid = self.conv2(h_mid)
            h_mid_atn1, h_mid_atn2 = torch.chunk(h_mid, 2, dim=1)
            h_atn1 = self.conv3(h_mid_atn1)
            h_atn2 = self.conv3(h_mid_atn2)
            h_atn1 = torch.sigmoid(h_atn1)
            h_atn2 = torch.sigmoid(h_atn2)
            h_out11 = h_perm1 * h_atn1
            h_out22 = h_perm2 * h_atn2
            h_out1 = h_out11.permute(0,2,1,3).contiguous()
            h_out2 = h_out22.permute(0,2,1,3).contiguous()
            h_out = (1/2)*(h_out1 + h_out2)
            return h_out
        elif self.is_chw == 1:
            w_perm1 = x.permute(0,3,2,1).contiguous()
            w_perm2 = x.permute(0,3,2,1).contiguous()
            w_mid1 = self.compress_w(w_perm1)
            w_mid2 = self.compress_w(w_perm2)
            w_mid = torch.cat([w_mid1, w_mid2], dim=1)
            w_mid = self.conv2(w_mid)
            w_mid_atn1, w_mid_atn2 = torch.chunk(w_mid, 2, dim=1)
            w_atn1 = self.conv3(w_mid_atn1)
            w_atn2 = self.conv3(w_mid_atn2)
            w_atn1 = torch.sigmoid(w_atn1)
            w_atn2 = torch.sigmoid(w_atn2)
            w_out11 = w_perm1 * w_atn1
            w_out22 = w_perm2 * w_atn2
            w_out1 = w_out11.permute(0,3,2,1).contiguous()
            w_out2 = w_out22.permute(0,3,2,1).contiguous()
            w_out = (1/2)*(w_out1 + w_out2)
            return w_out
        else:
            c_res1 = self.conv1(x)
            c_res2 = self.conv1(x)
            c_mid1 = self.compress_c(c_res1)
            c_mid2 = self.compress_c(c_res2)
            c_mid = torch.cat([c_mid1, c_mid2], dim=1)
            c_mid = self.conv2(c_mid)
            c_mid_atn1, c_mid_atn2 = torch.chunk(c_mid, 2, dim=1)
            c_atn1 = self.conv3(c_mid_atn1)
            c_atn2 = self.conv3(c_mid_atn2)
            c_atn1 = torch.sigmoid(c_atn1)
            c_atn2 = torch.sigmoid(c_atn2)
            c_out1 = c_res1 * c_atn1
            c_out2 = c_res2 * c_atn2
            c_out = (1/2)*(c_out1 + c_out2)
            return c_out

class DCFE(nn.Module):
    def __init__(self, in_ch):
        super(DCFE, self).__init__()

        self.h_stream = DCFE_Branch(0, in_ch)
        self.w_stream = DCFE_Branch(1, in_ch)
        self.c_stream = DCFE_Branch(2, in_ch)
        
    def forward(self, x):
        out1 = self.h_stream(x)
        out2 = self.w_stream(x)
        out3 = self.c_stream(x)
        out = (1/3)*(out1 + out2 +out3)
        out += x
        return out


def INF(B,H,W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)
    #这里是先定义一个浮点型的正无穷，放到GPU上，在列上重复H次，[inf,inf,..,inf],diag是产生一个以inf为对角元素的对角矩阵
    #该矩阵的其余元素均为0，unsqueeze为该矩阵的外围套上一个维度，最后将该矩阵整体重复B*W次，inf变-inf

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)     #通过卷积产生q
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        #先将[b,c,h,w]变为[b,w,c,h],view之后变为[b*w,c,h],再转换为[b*w,h,c]
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        #先将[b,c,h,w]变为[b,h,c,w],view之后变为[b*h,c,w],再转换为[b*h,w,c]
        proj_key = self.key_conv(x)     #卷积产生k
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        #先将[b,c,h,w]变为[b,w,c,h],view之后变为[b*w,c,h]
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        #先将[b,c,h,w]变为[b,h,c,w],view之后变为[b*h,c,w]
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        #先将[b,c,h,w]变为[b,w,c,h],view之后变为[b*w,c,h]
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        #先将[b,c,h,w]变为[b,h,c,w],view之后变为[b*h,c,w]
        a = self.INF(m_batchsize, height, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        #先是两个矩阵相乘，bmm操作的矩阵必须是三维的，得到[b*w,h,h],在对角上加负无穷，展开成[b,w,h,h],最后维度转换成[b,h,w,h]
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)     #[b,h,w,w]
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))    #[b,h,w,h+w]，在最后的维度上做softmax

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #先取出H上的q×k转置，转换维度为[b,w,h,h],再变换为[b*w,h,h]
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        #先取出W上的q×k转置，[b,h,w,w], 再变换为[b*h,w,w]
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        #[b*w,c,h] @ [b*w,h,h]=[b*w,c,h],再转换为[b,w,c,h]->[b,c,h,w]
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #[b*h,c,w] @ [b*h,w,w]=[b*h,c,w],再转换为[b,h,c,w]->[b,c,h,w]
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x

class FusedAttention(nn.Module):
    def __init__(self, in_dim):
        super(FusedAttention, self).__init__()

        self.localatn = WindowAttention(in_dim)
        self.globalatn1 = CrissCrossAttention(in_dim)
        self.globalatn2 = CrissCrossAttention(in_dim)
    
    def forward(self, x):
        out1 = self.localatn(x)
        out1 = self.globalatn1(out1)
        out = self.globalatn2(out1)
        return out + x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.inc = nn.Sequential(
            single_conv(3, 64),     #这里面输入特征图通道数为6是因为将噪声图和噪声等级图级联了？
            single_conv(64, 64)
        )

        self.enhance1 = DCFE(64)

        self.down1 = nn.AvgPool2d(2)    #平均池化，窗口大小和步长均为2，因此实现的是2倍下采样
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.enhance2 = DCFE(128)

        self.down2 = nn.AvgPool2d(2)    
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )

        self.enhance3 = DCFE(256)

        self.global_atn = FusedAttention(256)

        self.up1 = up(256)
        self.conv3 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.enhance4 = DCFE(128)

        self.up2 = up(128)
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.enhance5 = DCFE(64)

        self.outc = outconv(64, 3)

    def forward(self, x):
        inx = self.inc(x)
        inx = self.enhance1(inx)

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)
        conv1 = self.enhance2(conv1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)
        conv2 = self.enhance3(conv2)
        conv2 = self.global_atn(conv2)

        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)
        conv3 = self.enhance4(conv3)

        up2 = self.up2(conv3, inx)
        conv4 = self.conv4(up2)
        conv4 = self.enhance5(conv4)

        out = self.outc(conv4)
        return out


