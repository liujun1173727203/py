from scipy.ndimage import gaussian_filter
import torch
import numpy as np
import torch.nn.functional as F

def msemap (input,recon):



    #这两句是对图片求2范式，可以不求
    input = F.normalize(input,p=2)
    recon =  F.normalize(recon,p=2)
    _,_,h,w=input.shape
    a_map=(input-recon)**2
    a_map=a_map.sum(1,keepdim=True)

    anomaly_map=a_map.squeeze().cpu().numpy()
    anomaly_map=gaussian_filter(anomaly_map,sigma=4)
    anomaly_map=torch.tensor(anomaly_map).cuda()
    anomaly_map=anomaly_map.unsqueeze(0)
    anomaly_map=anomaly_map.unsqueeze(0)
    #print(anomaly_map.shape)

    return anomaly_map

def compute_mse(x, x_hat):
    loss = torch.mean((x - x_hat) ** 2, dim=1,keepdim=True)  # 按通道求平均
    return loss