import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils
import numpy as np
from torch.autograd import Variable

device = torch.device("cuda"if torch.cuda.is_available()else"cpu")

EPSILON = 1e-5


def attention_fusion_weight(tensor1, tensor2, p_type):

    f_spatial_vector = spatial_vector_fusion(tensor1, tensor2, p_type)
    f_channel_vector = channel_vector_fusion(tensor1, tensor2, p_type)

    tensor_f = (f_spatial_vector + f_channel_vector)

    return tensor_f


def spatial_vector_fusion(tensor1, tensor2, p_type):
    shape = tensor1.size()
    # calculate spatial vector attention
    spatial_vector_p1 = spatial_vector_attention(tensor1, p_type)
    spatial_vector_p2 = spatial_vector_attention(tensor2, p_type)

    # get weight map
    spatial_vector_p_w1 = torch.exp(spatial_vector_p1) / (torch.exp(spatial_vector_p1) + torch.exp(spatial_vector_p2) + EPSILON)
    spatial_vector_p_w2 = torch.exp(spatial_vector_p2) / (torch.exp(spatial_vector_p1) + torch.exp(spatial_vector_p2) + EPSILON)

    spatial_vector_p_w1 = spatial_vector_p_w1.repeat(1, 1, 1, shape[3], shape[4])
    spatial_vector_p_w1 = spatial_vector_p_w1.to(device)
    spatial_vector_p_w2 = spatial_vector_p_w2.repeat(1, 1, 1, shape[3], shape[4])
    spatial_vector_p_w2 = spatial_vector_p_w2.to(device)

    tensor_f = spatial_vector_p_w1 * tensor1 + spatial_vector_p_w2 * tensor2

    return tensor_f


def channel_vector_fusion(tensor1, tensor2, spatial_type='mean'):
    shape = tensor1.size()
    # calculate channel vector attention
    channel_vector_1 = channel_vector_attention(tensor1, spatial_type)
    channel_vector_2 = channel_vector_attention(tensor2, spatial_type)

    channel_vector_w1 = torch.exp(channel_vector_1) / (torch.exp(channel_vector_1) + torch.exp(channel_vector_2) + EPSILON)
    channel_vector_w2 = torch.exp(channel_vector_2) / (torch.exp(channel_vector_1) + torch.exp(channel_vector_2) + EPSILON)

    channel_vector_w1 = channel_vector_w1.repeat(1, shape[1], 1, 1, 1)
    channel_vector_w1 = channel_vector_w1.to(device)
    channel_vector_w2 = channel_vector_w2.repeat(1, shape[1], 1,  1, 1)
    channel_vector_w2 = channel_vector_w2.to(device)

    tensor_f = channel_vector_w1 * tensor1 + channel_vector_w2 * tensor2

    return tensor_f


# spatial vector_attention
def spatial_vector_attention(tensor, type="l1_mean"):
    shape = tensor.size()
    b = shape[0]
    c = shape[1]
    t = shape[2]
    h = shape[3]
    w = shape[4]
    spatial_vector = torch.zeros(b, c, t, 1, 1)
    if type == "l1_mean":
        # torch.norm是对输入的Tensor求范数
        # input(Tensor) – 输入张量
        # p(float, optional) – 范数计算中的幂指数值
        # torch.norm(input, p, dim, out=None,keepdim=False) → Tensor  返回输入张量给定维dim 上每行的p 范数。
        spatial_vector = torch.norm(tensor, p=1, dim=[3, 4], keepdim=True) / (h * w)
    elif type == "l2_mean":
        spatial_vector = torch.norm(tensor, p=2, dim=[2, 3], keepdim=True) / (h * w)
    elif type == "linf":
            for i in range(c):
                tensor_1 = tensor[0,i,:,:]
                spatial_vector[0,i,0,0] = torch.max(tensor_1)
            ndarray = tensor.cpu().numpy()
            max = np.amax(ndarray,axis=(2,3))
            tensor = torch.from_numpy(max)
            spatial_vector = tensor.reshape(1,c,1,1)
            spatial_vector = spatial_vector.to(device)
    return spatial_vector


# # channel vector attention
def channel_vector_attention(tensor, type='l1_mean'):

    shape = tensor.size()
    b = shape[0]
    c = shape[1]
    t = shape[2]
    h = shape[3]
    w = shape[4]
    channel_vector = torch.zeros(b, 1, t, 1, 1)
    if type == 'l1_mean':
        channel_vector = torch.norm(tensor, p=1, dim=[1], keepdim=True) / c
    elif type == "l2_mean":
        channel_vector = torch.norm(tensor, p=2, dim=[1], keepdim=True) / c
    elif type == "linf":
        channel_vector, indices = tensor.max(dim=1, keepdim=True)
        channel_vector = channel_vector / c
        channel_vector = channel_vector.to(device)
    return channel_vector



