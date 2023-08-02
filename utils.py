from torch.utils import data
import os
import joblib
from args_fusion import args
import torch.nn as nn
import torch
import numpy as np
from scipy.signal import welch
from sklearn import metrics
from torch.autograd import Variable
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Mydata_hr(data.Dataset):
    def __init__(self,RGB_data_path=True,rPPG_data_path=True,NIR_data_path=True,hr_gt_path=True): 
        RGB_data = [os.path.join(RGB_data_path,x) for x in os.listdir(RGB_data_path)]
        rPPG_data = [os.path.join(rPPG_data_path,y) for y in os.listdir(rPPG_data_path)]
        NIR_data = [os.path.join(NIR_data_path,w) for w in os.listdir(NIR_data_path)]
        HR_gt = [os.path.join(hr_gt_path,z) for z in os.listdir(hr_gt_path)]
        self.RGB_data = RGB_data
        self.rPPG_data = rPPG_data
        self.NIR_data = NIR_data
        self.HR_gt = HR_gt
    def __getitem__(self,index):
        rPPG_path = self.rPPG_data[index]
        #label = np.load(y_path)
        rPPG_label = joblib.load(rPPG_path)

        RGB_path = self.RGB_data[index]
        #data = np.load(x_path)
        RGB_data = joblib.load(RGB_path)
        NIR_path = self.NIR_data[index]
        # data = np.load(x_path)
        NIR_data = joblib.load(NIR_path)
        #data = data.reshape(3,128,64,64)
        gt_path = self.HR_gt[index]
        gt = joblib.load(gt_path)
        return RGB_data, rPPG_label, NIR_data, gt
    def __len__(self):
        return len(self.RGB_data)


class Mydata_dual(data.Dataset):
    def __init__(self,RGB_data_path=True,rPPG_data_path=True,NIR_data_path=True): 
        RGB_data = [os.path.join(RGB_data_path,x) for x in os.listdir(RGB_data_path)]
        rPPG_data = [os.path.join(rPPG_data_path,y) for y in os.listdir(rPPG_data_path)]
        #HR_gt = [os.path.join(HR_gt_path,z) for z in os.listdir(HR_gt_path)]
        NIR_data = [os.path.join(NIR_data_path,w) for w in os.listdir(NIR_data_path)]
        self.RGB_data = RGB_data
        self.rPPG_data = rPPG_data
        #self.HR_gt = HR_gt
        self.NIR_data = NIR_data
    def __getitem__(self,index):
        rPPG_path = self.rPPG_data[index]
        #label = np.load(y_path)
        rPPG_label = joblib.load(rPPG_path)

        #gt_path = self.HR_gt[index]
        #gt = joblib.load(gt_path)

        RGB_path = self.RGB_data[index]
        #data = np.load(x_path)
        RGB_data = joblib.load(RGB_path)
        NIR_path = self.NIR_data[index]
        # data = np.load(x_path)
        NIR_data = joblib.load(NIR_path)
        #data = data.reshape(3,128,64,64)
        return RGB_data, rPPG_label, NIR_data
    def __len__(self):
        return len(self.RGB_data)


class Neg_Pearson(nn.Module):
    def __init__(self):
        super(Neg_Pearson,self).__init__()
        return
    def forward(self, preds, labels):
        loss_p = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])
            sum_y = torch.sum(labels[i])
            sum_xy = torch.sum(preds[i]*labels[i])
            sum_x2 = torch.sum(torch.pow(preds[i],2))
            sum_y2 = torch.sum(torch.pow(labels[i],2))
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
            if (pearson>=0):
                loss_p += 1 - pearson
            else:
                loss_p += 1 - torch.abs(pearson)
        #loss += 1 - pearson
        loss_person = loss_p/preds.shape[0]
        return loss_person

        
class Pearson1(nn.Module):   # 测试用，评估HR之间的相关性
    def __init__(self):
        super(Pearson1, self).__init__()
        return
    def forward(self, preds, labels):
        loss = 0
        sum_x = torch.sum(preds)
        sum_y = torch.sum(labels)
        sum_xy = torch.sum(preds * labels)
        sum_x2 = torch.sum(torch.pow(preds, 2))
        sum_y2 = torch.sum(torch.pow(labels, 2))
        N = preds.shape[0]
        pearson = (N * sum_xy - sum_x * sum_y) / (
            torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))
        if (pearson >= 0):
            loss += pearson
        else:
            loss += torch.abs(pearson)
        # loss += 1 - pearson
        return loss
class std(nn.Module):
    def __init__(self):
        super(std, self).__init__()
        return

    def forward(self, preds, labels):
        error = torch.abs(preds - labels)
        std = np.std(error.cpu().detach().numpy())
        return torch.tensor(std)
        
        
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print(f'-- new folder "{path}" --')
    else:
        print(f'-- the folder "{path}" is already here --')
        
class SelfDefinedRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return torch.where(inp < 1., torch.zeros_like(inp), inp)

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        return grad_output * torch.where(inp < 1., torch.zeros_like(inp),
                                         torch.ones_like(inp))


class Relu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = SelfDefinedRelu.apply(x)
        return out


def compute_complex_absolute_given_k(output, k, N):
    two_pi_n_over_N = Variable(2 * math.pi * torch.arange(0, N, dtype=torch.float), requires_grad=True) / N
    hanning = Variable(torch.from_numpy(np.hanning(N)).type(torch.FloatTensor), requires_grad=True).view(1, -1)

    k = k.type(torch.FloatTensor).cuda()
    two_pi_n_over_N = two_pi_n_over_N.cuda()
    hanning = hanning.cuda()

    output = output.view(1, -1) * hanning
    output = output.view(1, 1, -1).type(torch.cuda.FloatTensor)
    k = k.view(1, -1, 1)
    two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
    complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                       + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2

    return complex_absolute


def PSD_absolute(output, Fs, bpm_range=None):
    output = output.view(1, -1)

    N = output.size()[1]

    unit_per_hz = Fs / N
    feasible_bpm = bpm_range / 60.0
    k = feasible_bpm / unit_per_hz

    # only calculate feasible PSD range [0.7,4]Hz
    psd_absolute = compute_complex_absolute_given_k(output, k, N)

    return (1.0 / psd_absolute.sum()) * psd_absolute / torch.max((1.0 / psd_absolute.sum()) * psd_absolute)   # max-value normalization
    
def multi_wise(activate_norm_psd_absolute,bpm_range):
    hr = torch.sum((activate_norm_psd_absolute.reshape(-1)).mul(bpm_range))
    return hr
    
class fre_loss(nn.Module):  #fre_loss
    def __init__(self):
        super(fre_loss, self).__init__()
        return

    def forward(self, preds, labels, Fs, bpm_range):
        psd1 = PSD_absolute(preds, Fs, bpm_range)
        psd2 = PSD_absolute(labels, Fs, bpm_range)
        euclidean_distance = torch.norm(psd1 - psd2, dim=1)  # calculate Euclidean distance
        loss = torch.mean(euclidean_distance)  # calculate mean value as the loss
        inverse_distance = 1 / (loss + 1e-8)
        return inverse_distance