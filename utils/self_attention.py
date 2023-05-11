import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax

def INF(B,C,H,W,num):
    mask = []
    # m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16 = []
    #4×4个像素,开始从左上到右下依次编号1,2,...,16,相当于对整张图做的self-attention共分16类,下面算每类的编码
    if(num==0):
        m1 = [[1.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]
        m1 = torch.tensor(m1).cuda().unsqueeze(0).unsqueeze(0).repeat(B,C,H//4,W//4)
        return m1
    elif(num==1):
        m2 = [[0.,1.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]
        m2 = torch.tensor(m2).cuda().unsqueeze(0).unsqueeze(0).repeat(B,C,H//4,W//4)
        return m2
    elif(num==2):
        m3 = [[0.,0.,1.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]
        m3 = torch.tensor(m3).cuda().unsqueeze(0).unsqueeze(0).repeat(B,C,H//4,W//4)
        return m3
    elif(num==3):
        m4 = [[0.,0.,0.,1.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]
        m4 = torch.tensor(m4).cuda().unsqueeze(0).unsqueeze(0).repeat(B,C,H//4,W//4)
        return m4
    elif(num==4):
        m5 = [[0.,0.,0.,0.],[1.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]
        m5 = torch.tensor(m5).cuda().unsqueeze(0).unsqueeze(0).repeat(B,C,H//4,W//4)
        return m5
    elif(num==5):
        m6 = [[0.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]
        m6 = torch.tensor(m6).cuda().unsqueeze(0).unsqueeze(0).repeat(B,C,H//4,W//4)
        return m6
    elif(num==6):
        m7 = [[0.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]
        m7 = torch.tensor(m7).cuda().unsqueeze(0).unsqueeze(0).repeat(B,C,H//4,W//4)
        return m7
    elif(num==7):
        m8 = [[0.,0.,0.,0.],[0.,0.,0.,1.],[0.,0.,0.,0.],[0.,0.,0.,0.]]
        m8 = torch.tensor(m8).cuda().unsqueeze(0).unsqueeze(0).repeat(B,C,H//4,W//4)
        return m8
    elif(num==8):
        m9 = [[0.,0.,0.,0.],[0.,0.,0.,0.],[1.,0.,0.,0.],[0.,0.,0.,0.]]
        m9 = torch.tensor(m9).cuda().unsqueeze(0).unsqueeze(0).repeat(B,C,H//4,W//4)
        return m9
    elif(num==9):
        m10 = [[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,0.]]
        m10 = torch.tensor(m10).cuda().unsqueeze(0).unsqueeze(0).repeat(B,C,H//4,W//4)
        return m10
    elif(num==10):
        m11 = [[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,0.]]
        m11 = torch.tensor(m11).cuda().unsqueeze(0).unsqueeze(0).repeat(B,C,H//4,W//4)
        return m11
    elif(num==11):
        m12 = [[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,1.],[0.,0.,0.,0.]]
        m12 = torch.tensor(m12).cuda().unsqueeze(0).unsqueeze(0).repeat(B,C,H//4,W//4)
        return m12
    elif(num==12):
        m13 = [[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[1.,0.,0.,0.]]
        m13 = torch.tensor(m13).cuda().unsqueeze(0).unsqueeze(0).repeat(B,C,H//4,W//4)
        return m13
    elif(num==13):
        m14 = [[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,1.,0.,0.]]
        m14 = torch.tensor(m14).cuda().unsqueeze(0).unsqueeze(0).repeat(B,C,H//4,W//4)
        return m14
    elif(num==14):
        m15 = [[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,1.,0.]]
        m15 = torch.tensor(m15).cuda().unsqueeze(0).unsqueeze(0).repeat(B,C,H//4,W//4)
        return m15
    elif(num==15):
        m16 = [[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,1.]]
        m16 = torch.tensor(m16).cuda().unsqueeze(0).unsqueeze(0).repeat(B,C,H//4,W//4)
        return m16

class ChessBoardAttention(nn.Module):
    """ Chessboard Attention Module """
    def __init__(self, in_dim):
        super(ChessBoardAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,kernel_size=1)
        self.softmax = Softmax(dim=-1)
        self.pool = nn.MaxPool2d(kernel_size=4,stride=4)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batchsize, channel, height, width = x.size()
        proj_query = self.query_conv(x) #[b,c//8,h,w]
        # proj_query_HW = proj_query.permute(0,2,3,1).contiguous().view(batchsize,height*width,-1)    #[b,h*w,c//8]
        proj_key = self.key_conv(x) #[b,c//8,h,w]
        # proj_key_HW = proj_key.view(batchsize,-1,height*width)  #[b,c//8,h*w]
        proj_value = self.value_conv(x)     #[b,c,h,w]
        out_tmp = []
        for num in range(16):
            window_cross = self.INF(batchsize,channel//8,height,width,num)
            q = proj_query*window_cross
            q = self.pool(q)
            q_hw = q.permute(0,2,3,1).contiguous().view(batchsize,height*width//16,-1)
            k = proj_key*window_cross
            k = self.pool(k)
            k_hw = k.view(batchsize,-1,height*width//16)
            window = self.INF(batchsize,channel,height,width,num)
            v = proj_value*window
            v = self.pool(v)
            v_hw = v.view(batchsize,-1,height*width//16)
            attn_hw = torch.bmm(q_hw,k_hw)
            attn_hw = self.softmax(attn_hw)
            out_hw = torch.bmm(v_hw,attn_hw.permute(0,2,1)).view(batchsize,-1,height//4,width//4)
            # for i in range(height//4):
            #     for j in range(width//4):
            #         window[:, :, i*4:i*4+4, j*4:j*4+4] = window[:, :, i*4:i*4+4, j*4:j*4+4]*out_hw[:, :, i:i+1, j:j+1]
            window = window.view(batchsize, channel, height//4, 4, width//4, 4)
            window_trans = window.permute(0, 1, 2, 4, 3, 5).contiguous()
            out_hw = out_hw.view(batchsize, channel, height//4, 1, width//4, 1)
            out_hw_trans = out_hw.permute(0, 1, 2, 4, 3, 5).contiguous()
            out_11 = window_trans*out_hw_trans
            out_1 = out_11.permute(0, 1, 2, 4, 3, 5).contiguous().view(batchsize, channel, height, width)
            out_tmp.append(out_1)
        
        out = out_tmp[0]
        for i in range(1,16):
            out = out + out_tmp[i]
        
        return self.gamma*out + x


        # proj_value_HW = proj_value.view(batchsize,-1,height*width)   #[b,c,h*w]
        # attn_HW = (torch.bmm(proj_query_HW, proj_key_HW)+self.INF(batchsize,height,width))  #[b,h*w,h*w]
        # attn_HW = self.softmax(attn_HW)

        # out_HW = torch.bmm(proj_value_HW, attn_HW.permute(0,2,1)).view(batchsize,-1,height,width)
        # return self.gamma*out_HW + x

def window_partition(x, win_size):
    B, H, W, C = x.shape
    x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)
    return windows

def window_reverse(windows, win_size, H, W):
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, in_dim):
        super(WindowAttention,self).__init__()
        self.to_q = nn.Conv2d(in_dim, in_dim//4, kernel_size=1)
        self.to_k = nn.Conv2d(in_dim, in_dim//4, kernel_size=1)
        self.to_v = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.softmax = Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, _, H, W = x.shape
        xclone = x.permute(0,2,3,1).contiguous()
        x_window = window_partition(xclone, 4) #[b*Nh*Nw,4,4,c]
        x_window_at = x_window.permute(0,3,1,2).contiguous() #[b*Nh*Nw,c,4,4]
        B_, C, winh, winw = x_window_at.shape
        proj_query = self.to_q(x_window_at) #[b*Nh*Nw,c//4,4,4]
        proj_query_win = proj_query.permute(0,2,3,1).contiguous().view(B_,winh*winw,C//4) #[b*Nh*Nw,4*4,c//4]
        proj_key = self.to_k(x_window_at)   #[b*Nh*Nw,c//4,4,4]
        proj_key_win = proj_key.view(B_,C//4,winh*winw)   #[b*Nh*Nw,c//4,4*4]
        proj_value = self.to_v(x_window_at) #[b*Nh*Nw,c,4,4]
        proj_value_win = proj_value.view(B_,C,winh*winw)  #[b*Nh*Nw,c,4*4]
        attn_win = torch.bmm(proj_query_win,proj_key_win) #[b*Nh*Nw,4*4,4*4]
        attn_win = self.softmax(attn_win)

        out_win1 = torch.bmm(proj_value_win, attn_win.permute(0,2,1)).permute(0,2,1).contiguous().view(B_,winh,winw,C)
        out_win11 = window_reverse(out_win1, 4, H, W) #[B,H,W,C]
        out_win = out_win11.permute(0,3,1,2).contiguous()   #[B,C,H,W]

        return self.gamma*out_win + x

class GlobalAttention(nn.Module):
    def __init__(self,in_dim):
        super(GlobalAttention,self).__init__()

        self.stage1 = WindowAttention(in_dim)
        self.stage2 = ChessBoardAttention(in_dim)

    def forward(self, x):
        out1 = self.stage1(x)
        out = self.stage2(out1)
        # out += x
        return out









        
