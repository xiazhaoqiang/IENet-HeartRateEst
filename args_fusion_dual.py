class args():
    # training args
    epochs = 50  #default 50  train HR_net epoch
    batch_size = 16  #default  # train HR_net batch_size
    rPPG_epochs = 100  # train rPPG_net epoch
    rPPG_batch_size = 2  # train rPPG_net batch_size
    hr_pred_epoch = 1     # solely predict hr value epoch，测试程序为test_hr.py
    hr_pred_batch_size = 1 #18   # solely predict hr value batch_size，测试程序为test_hr.py

    fusion_type = ['l1_mean','l2_mean','linf']
    # HR_net
    #path1_train = '/scratch/project_2006012/OBF_dataset/Train/OBF_RGBPKL/'  # RGB data path
    #path2_train = '/scratch/project_2006012/OBF_dataset/Train/OBF_PulsePKL/'  # rPPG signal data path
    #path4_train = '/scratch/project_2006012/OBF_dataset/Train/OBF_GTPKL/'     # hr_gt data path
    #path3_train = '/scratch/project_2006012/OBF_rPPGtest/Test/OBF_NIRPKL/'    # NIR data path
   
    
    # rPPG_net 
    rPPG_path1_train = '/scratch/project_2006012/OBF_dataset/Train/OBF_RGBPKL/'  # RGB data path
    rPPG_path2_train = '/scratch/project_2006012/OBF_dataset/Train/OBF_PulsePKL/'  # rPPG signal data path
    rPPG_path4_train = '/scratch/project_2006012/OBF_dataset/Train/OBF_GTPKL/'     # hr_gt data path
    rPPG_path3_train = '/scratch/project_2006012/OBF_dataset/Train/OBF_NIRPKL/'    # NIR data path
    
    rPPG_path1_test = '/scratch/project_2006012/OBF_rPPGtest/Test/OBF_RGBPKL/'  # RGB data path
    rPPG_path2_test = '/scratch/project_2006012/OBF_rPPGtest/Test/OBF_PulsePKL/'  #  rPPG signal data path
    rPPG_path4_test = '/scratch/project_2006012/OBF_rPPGtest/Test/OBF_GTPKL/'     # hr_gt data path
    rPPG_path3_test = '/scratch/project_2006012/OBF_rPPGtest/Test/OBF_NIRPKL/'    # NIR data path


    save_rPPG_model_dir = "rPPG_models/"  # "path to folder where trained rPPG model will be saved."
    save_rPPG_results_dir = 'rPPG_results/'
    save_hr_model_dir = "hr_models/"  # "path to folder where trained hr model will be saved."
    save_hr_results_dir = 'hr_results/'

    cuda = 1  # "set it to 1 for running on GPU, 0 for CPU"


    initial_lr = 1e-4  # "learning rate"  
    step_size = 25  # train hr_net, evert step_size epochs，update parameters；
    rPPG_step_size = 50  # train rPPG_net, evert step_size epochs，update parameters；
    gamma =0.5   # update lr scale factor
    log_interval = 5  # "number of images after which the training loss is logged"
    log_iter = 1
    resume = None

