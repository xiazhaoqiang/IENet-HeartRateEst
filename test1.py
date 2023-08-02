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



def hr_predition(rppg,Fs,bpm_range,gate_function):
    final_hr = torch.zeros(rppg.size()[0]) 
    for n in torch.arange(0,(rppg.size()[0]),dtype=torch.long):
        norm_psd_absolute = PSD_absolute(rppg[n,:], Fs, bpm_range) 
        activate_norm_psd_absolute = gate_function(norm_psd_absolute) 
        hr = multi_wise(activate_norm_psd_absolute,bpm_range) 
        final_hr[n] = hr  
    final_hr = final_hr.cuda()
    return final_hr


def main():
    # MR
    test_data = Mydata_hr(args.path1_test,args.path2_test,args.path3_test,args.path4_test)  
    batch_size = args.hr_pred_batch_size
    epochs = args.hr_pred_epoch
    SwinFuse_model = fusion()                                  
                                                   
    # OBF                                            
       
    SwinFuse_model.load_state_dict(torch.load("./model/Current_RGB+NIR.model"),False)                                           
    gate_function = Relu()

    bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
    Fs = 30  
    Fs = torch.tensor(Fs)
    Fs = Fs.cuda()         

    
    f_type = args.fusion_type[0]

    data_loader_test = DataLoader(test_data, batch_size, shuffle=False)


    if args.cuda:
        SwinFuse_model.cuda()

    start = datetime.datetime.now()
    tbar = trange(epochs)

    for e in tbar:
        print("Waiting Test!")
        SwinFuse_model.eval()

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
                #_,_,fusion_rppg_test = SwinFuse_model(RGB_data_test,NIR_data_test,f_type)  
                RGB_rPPG, NIR_rPPG,fusion_rppg_test = SwinFuse_model(RGB_data_test,NIR_data_test,f_type)  
                rPPG_label_test = (rPPG_label_test - torch.mean(rPPG_label_test)) / torch.std(rPPG_label_test)
                hr_test = hr_predition(fusion_rppg_test,Fs,bpm_range,gate_function)
                hr_gt_test = hr_gt_test.squeeze(-1)
                print("hr_test", hr_test)
                print("hr_gt_test", hr_gt_test) 


    print(datetime.datetime.now() - start)



if __name__ == "__main__":
    main()