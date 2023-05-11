import torch.nn as nn
from mmcv.cnn import ConvModule


class involution(nn.Module):

    def __init__(self,
                 channels,
                 kernel_size,
                 stride):
        super(involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=channels // reduction_ratio,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv2 = ConvModule(
            in_channels=channels // reduction_ratio,
            out_channels=kernel_size**2 * self.groups,
            kernel_size=1,
            stride=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape      #取出产生权重的维度
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)       #这里是先将得到的卷积核进行维度转换，相当于将out_channels拆解，unsqueeze表示在第2个维度上加一个维度
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out
        #对involution代码的理解
        #第一行是判断步长是否为1，如果不为1则通过池化进行下采样，但是一般都设置为1，经过两次卷积，第一次是压缩通道数，第二次是扩展通道数
        #第二行是获取得到的卷积核的各项参数，主要获得的是batch，h和w，这时的h和w是和输入特征图相同的，卷积核的内容集中在c中，也即out_channels=kernel_size**2 * self.groups
        #第三行是将weight展平，这里groups相当于对于某个位置上卷积核的个数，kernel_size**2指的是那个二维卷积核，unsqueeze的作用是在
        #kernel_size**2这个维度外面再加上一个维度，相当于套一个括号，其目的是为了使这一步的tensor的维度情况与后面第四行out的维度相对应
        #第四行首先对输入的特征图进行滑动窗口展平，这里Unfold的第一个参数是窗口大小，第二个维度是dilation，第三个维度是填充情况，若kernel_size
        #为3，则padding为1，代表在特征图外围加一圈0，如果这时候步长为1，则整个involution前后特征图的尺寸不变
        #整个第四行代码的目的在于按照窗口对输入特征图进行reshape，同时在通道维上进行分组，分组的个数为self.groups，每组的通道数为self.group_channelsself.group_channels
        #第五行实现的是将卷积核与整理后的特征图相乘相加，最后整理为与输入特征图size相同的输出特征图。