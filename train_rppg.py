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
#from utils import Mydata_dual
#from utils import Neg_Pearson
#from utils import Pearson
#from utils import std
#from utils import mkdir
#from utils import SelfDefinedRelu
#from utils import Relu
#from utils import PSD_absolute
#from utils import compute_complex_absolute_given_k
from args_fusion_dual import args
#from net import SwinTransformer3D
from net_dual import fusion
import cv2
from matplotlib import pyplot as plt
import datetime
import torch.nn as nn
import joblib
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main():
# train path
    # 载入RGB数据，rppg信号，hr_gt,NIR数据
    train_data = Mydata_dual(args.rPPG_path1_train,args.rPPG_path2_train,args.rPPG_path3_train)
    test_data = Mydata_dual(args.path1_test,args.path2_test,args.path3_test)
    train(train_data,test_data)


def train(train_data,test_data):
    batch_size = args.rPPG_batch_size
    epochs = args.rPPG_epochs
    SwinFuse_model = fusion()
    if args.resume is not None:
        print('Resuming, initializing using weight from {}.'.format(args.resume))
        SwinFuse_model.load_state_dict(torch.load(args.resume))
    print(SwinFuse_model)
    l1_loss = Neg_Pearson()  # 第一类损失，rppg信号的损失，负皮尔森相关系数
    l2_loss = fre_loss()
    gate_function = Relu()
    bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
    Fs = 30  # 采样率
    Fs = torch.tensor(Fs)
    Fs = Fs.cuda()
    #optimizer = optim.SGD(SwinFuse_model.parameters(), lr=args.initial_lr,momentum=0.9)
    optimizer = Adam(SwinFuse_model.parameters(), lr=args.initial_lr,weight_decay=0.00005)
    scheduler = StepLR(optimizer, step_size=args.rPPG_step_size, gamma=args.gamma,last_epoch=-1)  #参见2.2  https://blog.csdn.net/qyhaill/article/details/103043637
    print("\n初始化的学习率：", optimizer.defaults['lr'])
    alhpa = 0.3 #  损失函数的加权系数
    best_r = 100  # 初始化最优的MAE，以便后续保存模型
    eps = 1e-6  # 防止分母为0
    eps = np.array(eps)
    eps = torch.from_numpy(eps)
    f_type = args.fusion_type[0]
    # 最终的metrics
    total_r = np.zeros(epochs)

    data_loader_train = DataLoader(train_data, batch_size, shuffle=True)
    data_loader_test = DataLoader(test_data, batch_size, shuffle=True)

    # 将model转换为GPU
    if args.cuda:
        SwinFuse_model.cuda()

    start = datetime.datetime.now()
    tbar = trange(epochs)
    print('\nStart training.....')
    
	for e in tbar:
		# load training database
		running_loss = 0.0
		running_loss1 = 0.0
		running_loss2 = 0.0
		SwinFuse_model.train()
		# 训练
		for j, (RGB_data, rPPG_label, NIR_data) in enumerate(data_loader_train):  #x_train数据，y_train为rppg标签，gt为心率标签
			#ht_gt = ht_gt / 60   # 将心率值转换为频率
			RGB_data = RGB_data.permute(0, 4, 1, 2, 3)   # 将B D H W C -> B C D H W
			NIR_data = NIR_data.permute(0, 4, 1, 2, 3)
			if args.cuda:
				RGB_data, rPPG_label, NIR_data = RGB_data.cuda(), rPPG_label.cuda(), NIR_data.cuda()
			RGB_data = RGB_data.float()
			rPPG_label = rPPG_label.float()
			NIR_data = NIR_data.float()
			RGB_rPPG, NIR_rPPG,fusion_rppg = SwinFuse_model(RGB_data,NIR_data,f_type)  # x, 网络输出的特诊， rppg  hr 则为信号和心率值
			
			loss1 = l1_loss(fusion_rppg, rPPG_label)  # time_loss
			loss2 = l2_loss(fusion_rppg, rPPG_label,Fs,bpm_range) #fre_loss
			loss = alhpa * loss1 + alhpa * loss2
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			running_loss += loss.data  # 一个epoch下来损失的累加
			running_loss1 += loss1.data
			running_loss2 += loss2.data
		# 打印每一个迭代的loss
		num_batch = j+1
		mesg = "{}\tEpoch {}:\t loss: {:.6f}".format(time.ctime(),e + 1, running_loss/num_batch)
		tbar.set_description(mesg)
		print("\n第%d个epoch的学习率：%f" % (e+1, optimizer.param_groups[0]['lr']))
		scheduler.step()  # 调整学习率

		print("Waiting Test!")
		SwinFuse_model.eval()
		final_r = 0
		with torch.no_grad():
			for i, (RGB_data_test, rPPG_label_test, NIR_data_test) in enumerate(data_loader_test):   #x_test数据，y_test为rppg标签，gt为心率标签
				RGB_data_test = RGB_data_test.permute(0, 4, 1, 2, 3)  # 将B D H W C -> B C D H W
				NIR_data_test = NIR_data_test.permute(0, 4, 1, 2, 3)
				if args.cuda:
					RGB_data_test, rPPG_label_test, NIR_data_test = RGB_data_test.cuda(), rPPG_label_test.cuda(), NIR_data_test.cuda()
				RGB_data_test = RGB_data_test.float()
				rPPG_label_test = rPPG_label_test.float()
				NIR_data_test = NIR_data_test.float()
				# x, 网络输出的特征， rppg  hr 则为信号和心率值
				RGB_rPPG_test,NIR_rPPG_test,fusion_rppg_test = SwinFuse_model(RGB_data_test,NIR_data_test,f_type)

				loss1_test = l1_loss(fusion_rppg_test, rPPG_label_test)  # 评估指标 rPPPG 信号的皮尔森相关系数

				final_r += loss1_test.data  # rPPG信号之间的相关系数

		# 每一个epoch的metrics 值
		num = i + 1  # 记录有多少个batchsize，从而平均误差值，即测试集样本数/batch_size
		r = final_r / num
		total_r[e] = r

		if r <= best_r:
			best_r = r
			#torch.save(SwinFuse_model, 'best_rppg.pkl')
		print('\nbest_r:%.4f' % best_r)
	
	print(datetime.datetime.now() - start)



if __name__ == "__main__":
    main()
