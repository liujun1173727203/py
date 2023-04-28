import torch
import os
import numpy as np
import torch.nn as nn
from criterions.msgms import MSGMSLoss
from criterions.ssim import SSIMLoss
from utils.utils import mean_smoothing , savefig
from utils.auroc import compute_auroc
from sklearn.metrics import roc_auc_score
from collections import OrderedDict
import cv2
from torch.utils.data import DataLoader
import argparse
from model.SUNet import SUNet_model
import yaml
from dataload.data_RGB import get_test_data

mse = nn.MSELoss(reduction='mean')#损失函数mse
#L1_loss = nn.L1Loss()#损失函数采用L1损失
msgms = MSGMSLoss(num_scales=3, in_channels=3)#辅助损失函数
ssim = SSIMLoss(kernel_size=11, sigma=1.5)#辅助损失函数

with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)


parser = argparse.ArgumentParser(description='Demo Image Restoration')
parser.add_argument('--input_dir', default='./datasets/mvtec_anomaly_detection', type=str, help='Input images')
parser.add_argument('--window_size', default=8, type=int, help='window size')
parser.add_argument('--result_dir', default='./result', type=str, help='Directory for results')
parser.add_argument('--weights',
                    default='./checkpoints/recon/models/model_best.pth', type=str,
                    help='Path to weights')

args = parser.parse_args()
Train = opt['TRAINING']#加载训练的参数
class_type=Train['CLASS_TYPE']#类别类型


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

inp_dir = args.input_dir
out_dir = args.result_dir

os.makedirs(out_dir, exist_ok=True)

train_dataset = get_test_data(inp_dir,  Train['TRAIN_PS'],class_type)
print(train_dataset)
print(len(train_dataset))




# files = natsorted(glob(os.path.join(inp_dir, '*.jpg'))
#                   + glob(os.path.join(inp_dir, '*.JPG'))
#                   + glob(os.path.join(inp_dir, '*.png'))
#                   + glob(os.path.join(inp_dir, '*.PNG')))

if len(train_dataset) == 0:
    raise Exception(f"No files found at {inp_dir}")

train_loader = DataLoader(dataset=train_dataset, batch_size=1,
                          shuffle=True, num_workers=0, drop_last=False)
epoch = len(train_dataset)
# Load corresponding model architecture and weights
model = SUNet_model(opt)
model.cuda()

load_checkpoint(model, args.weights)
model.eval()

print('restoring images......')
gt_list_px_lxl = []
pred_list_px_lxl = []
gt_list_img_lxl = []
pred_list_img_lxl = []
image_yuan = []
reconst = []
y_label=[]
augs = []

image_map = []
image_gt =[]




for file_ in train_loader:
    #print(file_)

    input = file_[0][0].cuda()
    #print(input.shape)
    input_augment = file_[0][1]
    input_y = file_[1]
    input_mask = file_[2]
    #print(input_y)

    mb_amap=0

    with torch.no_grad():
        output = model(input_augment.cuda())
        #restored = torch.clamp(restored, 0, 1)#将输入的张量每个元素的范围限制在（0，1）之间，返回一个新的张量
        mb_amap += msgms(input,output,as_loss=False)
    #print(mb_amap,mb_amap.shape)
    #图像级参数
    img_map = mean_smoothing(mb_amap)
    image_map.extend(img_map.squeeze(1).detach().cpu().numpy())
    image_gt.extend(input_mask.detach().cpu().numpy())
    image_yuan.extend(input.permute(0, 2, 3, 1).detach().cpu().numpy())
    reconst.extend(output.permute(0, 2, 3, 1).detach().cpu().numpy())
    augs.extend(input_augment.permute(0, 2, 3, 1).detach().cpu().numpy())

    #print(mb_amap,mb_amap.shape)
    pixel_map = mean_smoothing(mb_amap)
    pixel_map= np.asarray(pixel_map.cpu())
    #print(mb_amap,mb_amap.shape)
    max_anomaly_score = pixel_map.max()
    min_anomaly_score = pixel_map.min()
    map =(pixel_map - min_anomaly_score)/(max_anomaly_score-min_anomaly_score)


    gt_np = np.asarray(input_mask)
    #print(gt_np,gt_np.shape)
    #像素级
    gt_list_px_lxl.extend(gt_np[0].flatten())
    pred_list_px_lxl.extend(map[0].flatten())








image_map = np.array(image_map)
image_map = (image_map-image_map.min())/(image_map.max()-image_map.min())
image_map = list(image_map)
auroc = compute_auroc( epoch,np.array(image_map),np.array(image_gt))#图像级得分
print("auroc: %.3f"%(auroc))

pixel_auc = roc_auc_score(gt_list_px_lxl, pred_list_px_lxl)
print('pixel_auc:',pixel_auc)
# print(gt_list_img_lxl)
# print(pred_list_img_lxl)
# img_auc = roc_auc_score(gt_list_img_lxl,pred_list_img_lxl)
# print('img_auc:',img_auc)



savefig(class_type,epoch, image_yuan,reconst,image_gt,image_map, augs,)
print(" infer end________________________________________")



    # output = output.cpu()
    # imagenet_mean = np.array([0.485, 0.456, 0.406])
    # imagenet_std = np.array([0.229, 0.224, 0.225])
    #
    # #save original img
    # mean = torch.as_tensor(imagenet_mean)[None, :, None, None]
    # std = torch.as_tensor(imagenet_std)[None, :, None, None]
    # img1 =  output* std + mean
    # img2 = file_[0][0]* std + mean
    #
    #
    #
    # img = ToPILImage()(img1[0, :])
    # img.show()
    # img = ToPILImage()(img2[0, :])
    # img.show()



