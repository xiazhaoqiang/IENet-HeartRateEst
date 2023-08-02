import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from mmcv.runner import load_checkpoint
from mmaction.utils import get_root_logger
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import numpy as np
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
#from ..builder import BACKBONES
import fusion_strategy


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows

def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

#先来看没有Shift的基于Window的注意力机制是如何做的，传统的Transformer都是基于全局来计算注意力的，因此计算复杂度十分高。而Swin
# #Transformer则将注意力的计算限制在每个窗口内，进而减少了计算量，主要区别是在原始计算Attention的公式中的Q,K时加入了相对位置编码

class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个注意力头对应的通道数
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        # 设置一个形状为(2*Wd-1*2*(Wh-1) * 2*(Ww-1), nH)的可学习变量 ,用于后续的位置编码
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH
        # 获取窗口内每对token的相对位置索引
        # get pair-wise relative position index for each token inside the window
        # 得到 window 内的表格坐标
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        # 下面两行其实和 mask生成的是做的操作类似  也就是将数据展平后 广播后相减 得到相对坐标
        #利用广播机制 ,分别在第二维 ,第一维 ,插入一个维度 ,进行广播相减 ,得到 3, Wd*Wh*Ww, Wd*Wh*Ww的张量
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        #因为采取的是相减 ,所以得到的索引是从负数开始的 ,所以加上偏移量 ,让其从0开始
        # 但是上面的相对坐标和定义的 relative_position_bias_table还对应不上 relative_coords 取值范围 (-w + 1) ~ (w - 1)
        # 所以在dim=[1, 2] 维度 才加上 self.window_size[0] - 1 取值范围 变成 0 ~ (2w - 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        # 后续我们需要将其展开成一维偏移量 而对于(1 ,2)和(2 ,1)这两个坐标 在二维上是不同的,
        # 但是通过将x,y坐标相加转换为一维偏移的时候,他的偏移量是相等的,所以对其做乘法以进行区分
        # 最后在 dim=2 的维度上 乘以 2w - 1 所以在这个维度上取值范围为 0 ~  (2w - 2) * (2w - 1) = (2w - 1)**2 - (2w - 1)
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        #在最后一维上进行求和 ,展开成一个一维坐标 ,并注册为一个不参与网络学习的常量
        # 最后求和后得到的相对坐标范围 0 ~ (2w - 1)**2 - (2w - 1) + (2w - 2) = (2w - 1)**2 - 1
        # OKay 到此为止终于得到范围为 0 ~ (2w - 1)**2 - 1 和 上面的 relative_position_bias_table对应上了
        # 所以每次只需要用相对索引去 relative_position_bias_table 表格中取值就行了
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # 截断正态分布初始化
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        # numWindows*B, N, C ,其中N=window_size_d * window_size_h * window_size_w， 将特征拉平了，用于计算注意力
        B_, N, C = x.shape
        # 然后经过self.qkv这个全连接层后进行reshape到(3, numWindows*B, num_heads,N, c//num_heads)
        # 3表示3个向量,刚好分配给q,k,v,
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        # 根据公式,对q乘以一个scale缩放系数,
        # 然后与k（为了满足矩阵乘要求，需要将最后两个维度调换）进行相乘.
        # 得(numWindows*B, num_heads, N, N)的attn张量
        q = q * self.scale  # selfattention公式里的根号下dk
        attn = q @ k.transpose(-2, -1)

        # 之前我们针对位置编码设置了个形状为(2*Wd-1*2*(Wh-1) * 2*(Ww-1), numHeads)的可学习变量.
        # 我们用计算得到的相对编码位置索引self.relative_position_index选取,
        # 得到形状为(nH, Wd*Wh*Ww, Wd*Wh*Ww)的编码,加到attn张量上

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        # 剩下就是跟transformer一样的softmax，dropout,与V矩阵乘,再经过一层全连接层和dropout
        # SW-MSA前向传播中不同的代码地方如下
        # 就是比W-MSA在attn结果上多加了一个mask的值，使不想要的位置的值无限小，softmax后就会被忽略，从而达到mask的效果。
        if mask is not None:
            # mask.shape =  nW, N, N,  其中N = Wd*Wh*Ww
            nW = mask.shape[0]
            # 将mask加到attention的计算结果再进行softmax,
            # 由于mask的值设置为-100,softmax后就会忽略掉对应的值,从而达到mask的效果
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#context_enhance
class Context_enhance1(nn.Module):
    def __init__(self):
        super(Context_enhance1, self).__init__()
        self.conv1 = nn.Conv3d(32, 32, kernel_size=(2, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.norm1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=(2, 3, 3), dilation=3, stride=(1, 1, 1), padding=(2, 3, 3))
        self.norm2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 32, kernel_size=(2, 3, 3), dilation=5, stride=(1, 1, 1), padding=(3, 5, 5))
        self.norm3 = nn.BatchNorm3d(32)

    def forward(self, x):

        # 将输入张量按通道数均分为三个分支
        c = x.size(1) // 3
        x1 = x[:, :c, :, :, :]
        x2 = x[:, c:2*c, :, :, :]
        x3 = x[:, 2*c:, :, :, :]

        # 分别对三个分支进行组空洞卷积
        out1 = self.norm1(self.conv1(x1))
        out1 = out1[:, :, :-1, :, :]
        out2 = self.norm2(self.conv2(x2))
        out2 = out2[:, :, :-1, :, :]
        out3 = self.norm3(self.conv3(x3))
        out3 = out3[:, :, :-1, :, :]

        # 将三个分支的输出特征级联，恢复到原始形状
        out = torch.cat([out1, out2, out3], dim=1)

        return out

class Context_enhance2(nn.Module):
    def __init__(self):
        super(Context_enhance2, self).__init__()
        self.conv1 = nn.Conv3d(64, 64, kernel_size=(2, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.norm1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=(2, 3, 3), dilation=3, stride=(1, 1, 1), padding=(2, 3, 3))
        self.norm2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 64, kernel_size=(2, 3, 3), dilation=5, stride=(1, 1, 1), padding=(3, 5, 5))
        self.norm3 = nn.BatchNorm3d(64)

    def forward(self, x):

        # 将输入张量按通道数均分为三个分支
        c = x.size(1) // 3
        x1 = x[:, :c, :, :, :]
        x2 = x[:, c:2*c, :, :, :]
        x3 = x[:, 2*c:, :, :, :]

        # 分别对三个分支进行组空洞卷积
        out1 = self.norm1(self.conv1(x1))
        out1 = out1[:, :, :-1, :, :]
        out2 = self.norm2(self.conv2(x2))
        out2 = out2[:, :, :-1, :, :]
        out3 = self.norm3(self.conv3(x3))
        out3 = out3[:, :, :-1, :, :]

        # 将三个分支的输出特征级联，恢复到原始形状
        out = torch.cat([out1, out2, out3], dim=1)

        return out

class Context_enhance3(nn.Module):
    def __init__(self):
        super(Context_enhance3, self).__init__()
        self.conv1 = nn.Conv3d(128, 128, kernel_size=(2, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.norm1 = nn.BatchNorm3d(128)
        self.conv2 = nn.Conv3d(128, 128, kernel_size=(2, 3, 3), dilation=3, stride=(1, 1, 1), padding=(2, 3, 3))
        self.norm2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 128, kernel_size=(2, 3, 3), dilation=5, stride=(1, 1, 1), padding=(3, 5, 5))
        self.norm3 = nn.BatchNorm3d(128)

    def forward(self, x):

        # 将输入张量按通道数均分为三个分支
        c = x.size(1) // 3
        x1 = x[:, :c, :, :, :]
        x2 = x[:, c:2*c, :, :, :]
        x3 = x[:, 2*c:, :, :, :]

        # 分别对三个分支进行组空洞卷积
        out1 = self.norm1(self.conv1(x1))
        out1 = out1[:, :, :-1, :, :]
        out2 = self.norm2(self.conv2(x2))
        out2 = out2[:, :, :-1, :, :]
        out3 = self.norm3(self.conv3(x3))
        out3 = out3[:, :, :-1, :, :]

        # 将三个分支的输出特征级联，恢复到原始形状
        out = torch.cat([out1, out2, out3], dim=1)

        return out


class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # 默认大小 7
        self.window_size = window_size
        # 进行 SW-MSA shift-size 7//2=3
        # 进行 W-MSA shift-size 0
        self.shift_size = shift_size
        # multi self attention 最后神经网络的隐藏层的维度
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint=use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        # 层归一化
        self.context_enhance1 = Context_enhance1()
        self.context_enhance2 = Context_enhance2()
        self.context_enhance3 = Context_enhance3()
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # 隐藏层维度增加的比率
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 最后接一个多层感知机网络
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # 可以看出 上面 结构是  layer normal +  W-MSA/SW-MSA + layer normal + mlp

    def forward_part1(self, x, mask_matrix):
        ## context enhance
        #x = rearrange(x, 'n d h w c -> n c d h w')
        #if x.size(1) == 96:
            #x = self.context_enhance1(x)
        #elif x.size(1) == 192:
            #x = self.context_enhance2(x)
        #else x.size(1) == 384:
            #x = self.context_enhance3(x)
        #x = rearrange(x, 'n c d h w -> n d h w c')
        B, D, H, W, C = x.shape
        # 1 先计算出当前block的window_size, 和shift_size
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        # 2 经过一个layer_norm
        x = self.norm1(x)
        # pad feature maps to multiples of window size  # pad一下特征图避免除不开
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # 3 判断是否需要对特征图进行shift
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # 4 将特征图切成一个个的窗口（都是reshape操作）
            # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # 5 通过attn_mask是否为None判断进行W-MSA还是SW-MSA
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask = attn_mask)  # B*nW, Wd*Wh*Ww, C
        # 6 把窗口在合并回来，看成4的逆操作，同样都是reshape操作
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # 7 如果之前shitf过，也要还原回去
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
        # 去掉pad
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
            # 经过FFN
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.

                Args:
                    x: Input feature, tensor size (B, D, H, W, C).
                    mask_matrix: Attention mask for cyclic shift.
        """
        # tranformer的常规操作，包含MSA、残差连接、dropout、FFN，只不过MSA变成W-MSA或者SW-MSA
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x

class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # 用全连接层把C由4C->2C，因为是4个cat一起所以是4C
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)  # 层归一化
        x = self.reduction(x)  # 全连接层 降维

        return x

# cache each stage results
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    # 切片操作
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

#流程图中每个stage,对应代码中的BasicLayer,由若干个block组成,而block的数目由depths列表中的元素决定,这里是[2,2,6,2]. 每个block就是W-MSA（window-multihead self attention）或者SW-MSA（shift window multihead self
# #attention）,一般有偶数个block,两种SA交替出现,比如6个block,0,2,4是W-MSA,1,3,5是SW-MSA. 前三个stage的最后会用PatchMerging进行下采样.(代码中是前三个stage每个stage最后，流程图上画的是后三个，每个stage最前面做，其实是一样的).
# #操作为将临近2*2范围内的patch(即4个为一组)按通道cat起来,经过一个layernorm和linear层, 实现维度下采样、特征加倍的效果，具体见PatchMerging类注释

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,  # 以第一层为例 为96
                 depth,  # 以第一层为例 为2
                 num_heads,  # 以第一层为例 为3
                 window_size=(1, 7, 7),  # (8,7,7)
                 mlp_ratio=4.,
                 qkv_bias=False,  # true
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,  # 以第一层为例 为[0, 0.01818182]
                 norm_layer=nn.LayerNorm,
                 downsample=None,  # PatchMerging
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size  # (8,7,7)
        self.shift_size = tuple(i // 2 for i in window_size)  # (4,3,3)
        self.depth = depth  # 2
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,  # 96
                num_heads=num_heads,  # 3
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                # 第一个block的shiftsize=(0,0,0)，也就是W-MSA不进行shift，第2个shiftsize=(4,3,3)
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,  # true
                qk_scale=qk_scale,  # None
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])  # depth = 2

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]  # 1*8
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]  # 56/7 *7
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]  # 56/7 *7
        # 计算一个attention_mask用于SW-MSA，怎么shitfed以及mask如何推导见后文
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        # 以第一个stage为例，里面有2个block，第一个block进行W-MSA，第二个block进行SW-MSA
        # 如何W-MSA SW-MSA 见下述
        for blk in self.blocks:
            x = blk(x, attn_mask)
        # 改变形状，把C放到最后一维度（因为PatchMerging里有layernom和全连接层）
        x = x.view(B, D, H, W, -1)
        # 用PatchMerging 进行patch的拼接和全连接层 实现下采样
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x


# 预处理图片序列到patch_embed，对应流程图中的Linear Embedding,具体做法是用3d卷积,从BCDHW->B,C,D,Wh,Ww 即(B,96,T/4,H/4,W/4),以后都假设HW为224X224，T为32，那么形状为（B,96,8,56,56），最后经过一层dropout，至此预处理结束
# #。要注意的是,其实在stage 1之前,即预处理完成后,已经是流程图上的T/4 × H/4 × W/4 × 96。主要函数实现：

class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=(2,4,4), in_chans=3, embed_dim=96, norm_layer=None):   # default  patch_size=(2,4,4), in_chans=3, embed_dim=96, norm_layer=None
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()  #BCDHW
        #DHW正好对应patch_size[0],patch_size[1],patch_size[2],防止除不开先pad
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww, 其中D Wh Ww表示经过3d卷积后特征的大小
        if self.norm is not None:  #默认会使用nn.LayerNorm,所以下面程序必运行
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)   #B, C, D, Wh, Ww -> B, C, D*Wh*Ww ->B,D*Wh*Ww, C
            x = self.norm(x)  #因为要层归一化，所以要拉成上面的形状，把C放在最后
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)   #又拉回 B, C, D, Wh, Ww

        return x


#@BACKBONES.register_module()
class SwinTransformer3D(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 pretrained=None,
                 pretrained2d=True,
                 patch_size=(2, 4, 4),  # default patch_size=(4, 4, 4)
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 4],   # default patch_size=(2,2,6,2)
                 num_heads=[3, 6, 12, 24],  # default patch_size=(3, 6, 12, 24)
                 window_size=(2, 7, 7),   # default window_size=(2, 7, 7)
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        # 预处理图片序列到patch_embed,对应流程图中的Linear Embedding,
        # 具体做法是用3d卷积,形状变化为BCDHW -> B,C,D,Wh,Ww 即(B,96,T/4,H/4,W/4),
        # 要注意的是,其实在stage 1之前,即预处理完成后,已经是流程图上的T/4 × H/4 × W/4 × 96

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # ViT在输入会给embedding进行位置编码.实验证明位置编码效果不好
        # 所以Swin-T把它作为一个可选项(self.ape),Swin-T是在计算Attention的时候做了一个相对位置编码
        # 这里video-Swin-T 直接去掉了位置编码
        # ViT会单独加上一个可学习参数,作为分类的token.
        # 而Swin-T则是直接做平均,输出分类,有点类似CNN最后的全局平均池化层

        # 经过一层dropout,至此预处理结束
        self.pos_drop = nn.Dropout(p=drop_rate)
        # 如果patch_size变为[2,4,4],则T维度则会下降2倍，所以最后需将T维度上采样，已使得可以与实际的标签维度匹配
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(384, 384, [3, 1, 1], stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(384),
            nn.ELU(),
        )

        # 流程图中每个stage,即代码中的BasicLayer,由若干个block组成,
        # 而block的数目由depths列表中的元素决定,这里是[2,2,6,2].
        # 每个block就是W-MSA(window-multihead self attention)或者SW-MSA(shift window multihead self attention),
        # 一般有偶数个block,两种SA交替出现,比如6个block,0,2,4是W-MSA,1,3,5是SW-MSA.
        # 前三个stage的最后会用PatchMerging进行下采样(代码中是前三个stage每个stage最后,流程图上画的是后三个,每个stage最前面做,其实是一样的)
        # 操作为将临近2*2范围内的patch(即4个为一组)按通道cat起来,经过一个layernorm和linear层, 实现维度下采样、特征加倍的效果,具体见PatchMerging类注释

        # stochastic depth
        # 随机深度,用这个来让每个stage中的block数目随机变化,达到随机深度的效果
        # torch.linspace()生成0到0.2的12个数构成的等差数列,如下
        # [0, 0.01818182, 0.03636364, 0.05454545, 0.07272727 0.09090909,
        # 0.10909091, 0.12727273, 0.14545455, 0.16363636, 0.18181818, 0.2]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):  # 流程图中的4个stage,对应代码中4个layers，流程图中每个stage,即代码中的BasicLayer,由若干个block组成
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),  # 96 x 2^n,对应流程图上的C,2C,4C,8C
                depth=depths[i_layer],  # [2,2,6,2]
                num_heads=num_heads[i_layer],  # [3, 6, 12, 24]
                window_size=window_size,  # (8,7,7)
                mlp_ratio=mlp_ratio,  # 4
                qkv_bias=qkv_bias,  # True
                qk_scale=qk_scale,  # None
                drop=drop_rate,  # 0
                attn_drop=attn_drop_rate,  # 0
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # 依据上面算的dpr
                norm_layer=norm_layer,  # nn.LayerNorm
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,  # 前三个stage后要用PatchMerging下采样
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))  # 96*8

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)

        self._freeze_stages()



    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


    def inflate_weights(self, logger):
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1, 1,self.patch_size[0], 1, 1) / self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            wd = self.window_size[0]
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                        size=(2 * self.window_size[1] - 1, 2 * self.window_size[2] - 1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2,
                                                                                                                   L2).permute(
                        1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2 * wd - 1, 1)

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()


    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights(logger)
            else:
                # Directly load 3D model.
                load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')


    def forward(self, x):
        """Forward function."""
        # path_embed 就是模型结构的 Patch Partition 图片变成 (B, H//4 * W//4, embed_dim)
        x = self.patch_embed(x)

        x = self.pos_drop(x)
        # 经过4个 3D swin transformer block
        for layer in self.layers:
            x = layer(x.contiguous())

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')
        x = self.upsample(x)
        return x
        #return x,rppg,hr  #b,c,t,h,w

        # Fusion

    def fusion(self, en1, en2, p_type):
        # attention weight
        fusion_function = fusion_strategy.attention_fusion_weight

        f = fusion_function(en1, en2, p_type)
        return f


    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer3D, self).train(mode)
        self._freeze_stages()
        
class rPPG_estimation(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense5 = nn.AdaptiveAvgPool3d((128, 1, 1))  # 把图像拉成1*1
        #self.dense6 = nn.Conv3d(768, 384, (1, 1, 1))  # 把维度拉1  # patch_size 为[1,4,4],为384， 为[2,4,4]则是768
        self.dense7 = nn.Conv3d(384,1,(1,1,1))

    def forward(self, x):
        x = self.dense5(x)
        #x = self.dense6(x)
        x = self.dense7(x)
        rppg = x.squeeze(1).squeeze(-1).squeeze(-1)
        return rppg

class fusion(nn.Module):
    # 融合模块
    def __init__(self):
        super(fusion, self).__init__()
        # self.RGB_data = RGB_data
        # self.NIR_data = NIR_data
        # self.p_type = p_type
        self.RGB_feature_extraction = SwinTransformer3D()
        self.NIR_feature_extraction = SwinTransformer3D()
        # attention weight
        self.fusion_function = fusion_strategy.attention_fusion_weight
        self.rPPG_net = rPPG_estimation()
        #f = fusion_function(RGB_feature, NIR_feature, p_type)
    def forward(self,RGB_data, NIR_data, p_type):
        RGB_feature = self.RGB_feature_extraction(RGB_data)
        NIR_feature = self.NIR_feature_extraction(NIR_data)
        fusion_feature = self.fusion_function(RGB_feature, NIR_feature, p_type)
        # fusion_feature = RGB_feature + NIR_feature
        RGB_rPPG = self.rPPG_net(RGB_feature)
        NIR_rPPG = self.rPPG_net(NIR_feature)
        fusion_rppg = self.rPPG_net(fusion_feature)
        return RGB_rPPG, NIR_rPPG, fusion_rppg