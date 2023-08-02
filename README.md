# IENet-HeartRateEst
This repo is used for the paper "Information-enhanced Network for Noncontact Heart Rate Estimation from Facial Videos", which is published in IEEE TCSVT.

## Requirements
- pytorch=1.11.0 +
- python=3.9 +
- torchvision
- numpy
- opencv-python
- anaconda (suggeted)

## Configuration
```bash
# Create your virtual environment using anaconda
conda create -n IENet-HeartRateEst python=3.9

# Activate your virtual environment
conda activate IENet-HeartRateEst

### Usage
```bash
# Clone our code
git clone https://github.com/xiazhaoqiang/IENet-HeartRateEst

# Remember to activate your virtual enviroment before running our code
conda activate IENet-HeartRateEst

# Replicate our method on heart rate estimation from facial videos 
net_dual.py # CET & MDEF combination module  
fusion_strategy.py # defination of MDEF  
train_rppg.py # train CET and MDEF modules to extract the rPPG signals  
args_fusion_dual.py # experimental parameters setting  
utils.py # refers to utility functions  
train_hr.py # train FEE module cascaded with CET and MDEF modules  
test.py  # test rppg and hr  
train_rppg.sh # slurm train shell

```
## Highlights
- This project proposes a novel information-enhanced network for HR estimation based on multimodal (e.g., RGB and NIR) sources.  In the network, context and modal difference information are sequentially enhanced from spatiotemporal and modal views for accurately describing HR-aware features, while maximum frequency information is enhanced for inhibiting heartbeat noise.
- Context-enhanced video Swin-Transformer (CET) module is exploited to extract useful rPPG signal features from facial visible-light and near-infrared videos.
- Novel modal difference enhanced fusion (MDEF) module is designed to acquire a fused rPPG signal,  which is taken as the input of the frequency- enhanced estimation (FEE) module to obtain the corresponding HR value. 

## Architecture of our remote heart rate estimation
![flowchart](https://github.com/xiazhaoqiang/IENet-HeartRateEst/blob/8d77ec02f5193e803b9da7634f6d7b99f19015f4/results/flowchart.png)

## Comparison Examples
1. rPPG signals comparison
!(https://github.com/xiazhaoqiang/IENet-HeartRateEst/blob/8d77ec02f5193e803b9da7634f6d7b99f19015f4/results/rPPG_comparison.png)


2. HR value comparison
!(https://github.com/xiazhaoqiang/IENet-HeartRateEst/results/HR_comparison.png)


4. Fre_spectrum comparison
!(https://github.com/xiazhaoqiang/IENet-HeartRateEst/results/fre_spectrum.png)


5. B-A comparison
!(https://github.com/xiazhaoqiang/IENet-HeartRateEst/results/B-A.png)


## Citation
If you find this code is useful for your research, please consider to cite our paper. Lili Liu, Zhaoqiang Xia, Xiaobiao Zhang, Jinye Peng, Xiaoyi Feng, Guoying Zhao, Information-enhanced Network for Noncontact Heart Rate Estimation from Facial Videos,  IEEE Transactions on Circuits and Systems for Video Technology.

```bash
@article{Liu2023Information,
  title={Information-enhanced Network for Noncontact Heart Rate Estimation from Facial Videos},
  author={Lili Liu, Zhaoqiang Xia, Xiaobiao Zhang, Jinye Peng, Xiaoyi Feng and Guoying Zhao},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={0},
  pages={1-16},
  year={2023},
  publisher={IEEE}
}
```
