import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax

def chess_partition(x, win_size):
    B, H, W, C = x.shape
    x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, H // win_size, W // win_size, C)
    return windows

def chess_reverse(windows, win_size, H, W):
    B = int(windows.shape[0] // (win_size * win_size))
    x = windows.view(B, win_size, win_size, H // win_size, W // win_size, -1)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, H, W, -1)
    return x

class ChessBoardAttention(nn.Module):
    """ Chessboard Attention Module """
    def __init__(self, in_dim):
        super(ChessBoardAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,kernel_size=1)
        self.softmax = Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, _, H, W = x.shape
        xclone = x.permute(0,2,3,1).contiguous() #[B,H,W,C]
        x_chess = chess_partition(xclone, 8) #[B*win*win,numh,numw,C]
        x_chess_at = x_chess.permute(0,3,1,2).contiguous() #[B*win*win,C,numh,numw]
        B_, C, numh, numw = x_chess_at.shape
        proj_query = self.query_conv(x_chess_at) #[B*win*win,C//4,numh,numw]
        proj_query_chess = proj_query.permute(0,2,3,1).contiguous().view(B_, numh*numw, C//4) #[B*win*win,numh*numw,C//4]
        proj_key = self.key_conv(x_chess_at) #[B*win*win,C//4,numh,numw]
        proj_key_chess = proj_key.view(B_, C//4, numh*numw) #[B*win*win,C//4,numh*numw]
        proj_value = self.value_conv(x_chess_at) #[B*win*win,C,numh,numw]
        proj_value_chess = proj_value.view(B_, C, numh*numw) #[B*win*win,C,numh*numw]
        attn_chess = torch.bmm(proj_query_chess, proj_key_chess) #[B*win*win,numh*numw,numh*numw]
        attn_chess = self.softmax(attn_chess)

        out_chess1 = torch.bmm(proj_value_chess, attn_chess.permute(0,2,1)).permute(0,2,1).contiguous().view(B_, numh, numw, C)
        out_chess11 = chess_reverse(out_chess1, 8, H, W) #[B, H, W, C]
        out_chess = out_chess11.permute(0, 3, 1, 2).contiguous()
        
        return self.gamma*out_chess + x
        


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
        return out 









        
