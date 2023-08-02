import numpy as np
from sklearn import metrics
import time
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
import torch.optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import *
from args_fusion_dual import args
from net_dual import fusion
import cv2
from matplotlib import pyplot as plt
import datetime
import torch.nn as nn
import joblib
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class hr_predition(nn.Module):
    def __init__(self):
        super(hr_predition, self).__init__()
        self.PSD_absolute = PSD_absolute
        self.gate_function = Relu()
        self.final_HR = multi_wise 
    def forward(self,rppg,Fs,bpm_range):
        final_hr = torch.zeros(rppg.size()[0])  # save HR value
        for n in torch.arange(0,(rppg.size()[0]),dtype=torch.long):  # psd function only can calculate one rPPG siganl, so need for circel in corresponding batch
            norm_psd_absolute = self.PSD_absolute(rppg[n,:], Fs, bpm_range)  # PSD calculation
            activate_norm_psd_absolute = self.gate_function(norm_psd_absolute)  # frequency-mask activation function 
            hr = self.final_HR(activate_norm_psd_absolute,bpm_range)  # obtain hr
            final_hr[n] = hr  
        final_hr = final_hr.cuda()
        return final_hr
        

class hr_net(nn.Module):
    def __init__(self):
        super(hr_net, self).__init__()
        self.rPPG_model = fusion()
        self.rPPG_model.load_state_dict(torch.load("/scratch/project_2001654/liulili/3D-swin-transformer/OBF_results/models/epochs_300_initial_lr_0.0001_step_size_100_video_RGB+NIR_alpha_0.5/"
                                                   "Current_epoch_298_Thu_Sep_22_21_10_35_2022_0.06322486_128_RGB+NIR.model"),False)
                                        
        self.conv1 = torch.nn.Conv1d(1,16,3,padding=1,bias=False)
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.conv2 = torch.nn.Conv1d(16,1,3,padding=1,bias=False)
        self.bn2 = torch.nn.BatchNorm1d(1)
        self.hr_estimation = hr_predition()
    def forward(self, RGB_data, NIR_data, p_type, Fs, bpm_range, e):
		if e < 40:
			with torch.no_grad():
				_,_,fusion_rppg = self.rPPG_model(RGB_data, NIR_data, p_type)
		else:
			with torch.enable_grad():
				_,_,fusion_rppg = self.rPPG_model(RGB_data, NIR_data, p_type)
        fusion_rppg = fusion_rppg.unsqueeze(1)
        fusion_rppg = self.bn1(self.conv1(fusion_rppg))
        fusion_rppg = self.bn2(self.conv2(fusion_rppg))
        fusion_rppg = fusion_rppg.squeeze(1)
        hr = self.hr_estimation(fusion_rppg,Fs,bpm_range)
        return fusion_rppg,hr

def load_model(path):

    SwinFuse_model = fusion()
    SwinFuse_model.load_state_dict(torch.load(path), False)

    SwinFuse_model.eval()
    SwinFuse_model.cuda()

    return SwinFuse_model

def main():
# train path
    # load data, RGB、NIR、rPPG gt、HR gt
    train_data = Mydata_hr(args.path1_train,args.path2_train,args.path3_train,args.path4_train)
    test_data = Mydata_hr(args.path1_test,args.path2_test,args.path3_test,args.path4_test)
    train(train_data,test_data)


def train(train_data,test_data):
    batch_size = args.batch_size
    SwinFuse_model = hr_net()  # model
    print(SwinFuse_model)
    gate_function = Relu()
    l1_loss = Neg_Pearson()  # Neg_Pearson loss
    l2_loss = torch.nn.L1Loss()  # mae loss
    # parameters
    alhpa = 0.4
    bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
    Fs = 30  # 采样率
    Fs = torch.tensor(Fs)
    Fs = Fs.cuda()
    HR_Pearson = Pearson1()   # metric r
    HR_std = std()           # sd
    optimizer = Adam(SwinFuse_model.parameters(), lr=args.initial_lr,weight_decay=0.00005)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma,last_epoch=-1)  #  https://blog.csdn.net/qyhaill/article/details/103043637
    print("\n初始化的学习率：", optimizer.defaults['lr'])
    best_mae = 100  # initialization 
    f_type = args.fusion_type[0]
    # metrics
    total_mae = np.zeros(args.epochs)
    total_rmse = np.zeros(args.epochs)
    total_sd = np.zeros(args.epochs)
    total_r = np.zeros(args.epochs)
    total_rppg_r = np.zeros(args.epochs)

    data_loader_train = DataLoader(train_data, batch_size, shuffle=True)
    data_loader_test = DataLoader(test_data, batch_size, shuffle=True)

    # model to GPU
    if args.cuda:
        SwinFuse_model.cuda()

    start = datetime.datetime.now()
    tbar = trange(args.epochs)
    print('\nStart training.....')
	for e in tbar:
		# load training database
		running_loss = 0.0
		running_loss2 = 0.0
		SwinFuse_model.train()
		# train
		for j, (RGB_data, rPPG_label, NIR_data,hr_gt) in enumerate(data_loader_train):  
			RGB_data = RGB_data.permute(0, 4, 1, 2, 3)   # 将B D H W C -> B C D H W
			NIR_data = NIR_data.permute(0, 4, 1, 2, 3)
			if args.cuda:
				RGB_data, rPPG_label, NIR_data,hr_gt = RGB_data.cuda(), rPPG_label.cuda(), NIR_data.cuda(), hr_gt.cuda()
			RGB_data = RGB_data.float()
			rPPG_label = rPPG_label.float()
			NIR_data = NIR_data.float()
			hr_gt = hr_gt.float()
			fusion_rppg,hr = SwinFuse_model(RGB_data,NIR_data,f_type,Fs, bpm_range)
			#loss1 = l1_loss(rppg, y_train)
			loss2 = l2_loss(hr, hr_gt)
			loss = alhpa * loss2
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			running_loss += loss.data  
			running_loss2 += loss2.data

		# 打印每一个epoch的loss
		num_batch = j + 1
		mesg = "{}\tEpoch {}:\t loss2: {:.6f}\t loss: {:.6f}".format(time.ctime(),e + 1,running_loss2 / num_batch,running_loss / num_batch)
		tbar.set_description(mesg)
		print("\n第%d个epoch的学习率：%f" % (e + 1, optimizer.param_groups[0]['lr']))
		scheduler.step()  

		print("Waiting Test!")
		SwinFuse_model.eval()
		final_mae = 0
		final_rmse = 0
		final_sd = 0
		final_r = 0
		final_rppg_r = 0
		with torch.no_grad():
			for i, (RGB_data_test, rPPG_label_test, NIR_data_test,hr_gt_test) in enumerate(data_loader_test): 
				RGB_data_test = RGB_data_test.permute(0, 4, 1, 2, 3)  # 将B D H W C -> B C D H W
				NIR_data_test = NIR_data_test.permute(0, 4, 1, 2, 3)
				if args.cuda:
					RGB_data_test, rPPG_label_test, NIR_data_test,hr_gt_test = RGB_data_test.cuda(), rPPG_label_test.cuda(), NIR_data_test.cuda(), hr_gt_test.cuda()
				RGB_data_test = RGB_data_test.float()
				rPPG_label_test = rPPG_label_test.float()
				NIR_data_test = NIR_data_test.float()
				hr_gt_test = hr_gt_test.float()
				fusion_rppg_test, hr_test = SwinFuse_model(RGB_data_test,NIR_data_test,f_type,Fs, bpm_range)
				#hr_gt_test = hr_gt_test.squeeze(-1)
				loss1_test = l1_loss(fusion_rppg_test, rPPG_label_test) 
				loss2_test = l2_loss(hr_test, hr_gt_test) 
				SD = HR_std(hr_test,hr_pred_by_gt_test)   
				R = HR_Pearson(hr_test,hr_pred_by_gt_test)  
				RMSE = np.sqrt(metrics.mean_squared_error(hr_test.cpu().numpy(), hr_pred_by_gt_test.cpu().numpy()))  
				final_mae += loss2_test.data
				final_sd += SD.data
				final_r += R.data
				final_rmse += RMSE
				final_rppg_r += loss1_test.data
		# every epoch metrics 
		num = i + 1 
		mae = final_mae / num   
		sd = final_sd /num
		r = final_r / num
		rmse = final_rmse / num
		rppg_r = final_rppg_r / num
		# all epoch metrics
		total_mae[e] = mae
		total_rmse[e] = rmse
		total_sd[e] = sd
		total_r[e] = r
		total_rppg_r[e] = rppg_r

		if mae <= best_mae:
			best_mae = mae
		print('\nbest_mae:%.4f' % best_mae)

	print(datetime.datetime.now() - start)



if __name__ == "__main__":
    main()