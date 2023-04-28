import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import ToPILImage
import os
import numpy as np
import torch.nn as nn
from criterions.msgms import MSGMSLoss
from criterions.ssim import SSIMLoss
from criterions.msemap import msemap, compute_mse
from utils.utils import mean_smoothing , savefig , save_mse_fig, mean_ssim_smoothing
from utils.auroc import compute_auroc
from sklearn.metrics import roc_auc_score
from skimage import img_as_ubyte
from collections import OrderedDict
import math
from natsort import natsorted
from glob import glob
import cv2
from torch.utils.data import DataLoader
import argparse
from model.SUNet import SUNet_model
import yaml
from data_RGB import get_training_data, get_validation_data,get_test_data

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
                    default='./checkpoints/recon_3_4__carpet/models/model_best.pth', type=str,
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

def reconstruct(mb_img, cutout_size: int, model) :
    _, _, h, w = mb_img.shape
    num_disjoint_masks = 3  # num_disjoint_masks=3
    disjoint_masks = create_disjoint_masks((h, w), cutout_size, num_disjoint_masks)  # cutout_sizes: [2, 4, 8, 16]
    #print(num_disjoint_masks,cutout_size)
    mb_reconst = 0
    for mask in disjoint_masks:
        mb_cutout = mb_img * mask
        mb_inpaint = model(mb_cutout)
        mb_reconst += mb_inpaint * (1 - mask)

    return mb_reconst


def create_disjoint_masks(
        img_size: [int, int],  # 图像输入的尺寸大小
        cutout_size: int = 8,  # 随机选择2，4，8，16里面的一个值，默认为8
        num_disjoint_masks: int = 3,  #
) :
    img_h, img_w = img_size  # 256，256
    grid_h = math.ceil(img_h / cutout_size)  # 返回一个不小于img_h / cutout_size的整数，256/8=32
    grid_w = math.ceil(img_w / cutout_size)  # 返回一个不小于img_w / cutout_size的整数，256/8=32
    num_grids = grid_h * grid_w  # 将图片分成32*32个大小为8*8的网格，[0,1,....,1023]共1024个
    disjoint_masks = []  # mask掉的网格列表
    for grid_ids in np.array_split(np.random.permutation(num_grids),num_disjoint_masks):  # np.random.permutation随机返回一个打乱的序列， np.array_split每num_disjoint_masks=3个为一组
        flatten_mask = np.ones(num_grids)  # [1,1,1..,1]1024个1
        flatten_mask[grid_ids] = 0  # 将刚才分组的下表序列的值都变成0
        mask = flatten_mask.reshape((grid_h, grid_w))  # 将flatten_mask变为32*32的格式 赋值个mask
        mask = mask.repeat(cutout_size, axis=0).repeat(cutout_size, axis=1)  # 沿着x,y轴复制8份，变成256*256
        mask = torch.tensor(mask, requires_grad=False, dtype=torch.float)
        mask = mask.cuda()
        disjoint_masks.append(mask)  # 一张图片产生三张不同位置的mask

    return disjoint_masks

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

mse_maps=[]
gms_maps=[]
ssim_maps=[]
cutout_size = 4


for file_ in train_loader:
    #print(file_)

    input = file_[0][0].cuda()
    #print(input.shape)
    input_augment = file_[0][1]
    input_y = file_[1]
    input_mask = file_[2]

    #print(input_y)

    mb_amap=0
    mse_map =0
    gms_map = 0
    ssim_map=0
    with torch.no_grad():
        output = reconstruct(input, cutout_size, model)
        #restored = torch.clamp(restored, 0, 1)#将输入的张量每个元素的范围限制在（0，1）之间，返回一个新的张量
        #mb_amap += msgms(input,output,as_loss=False)
        mse_map=compute_mse(input, output)
        gms_map=msgms(input,output,as_loss=False)
        ssim_map=ssim(input,output,as_loss=False)
        print(ssim_map.shape)
        #ssim_map=ssim(input,output,as_loss=False)
        mb_amap =compute_mse(input, output)+msgms(input,output,as_loss=False)
    #print(mb_amap,mb_amap.shape)
    #图像级参数
    img_map = mean_smoothing(mb_amap)
    gms_maps.extend(gms_map.squeeze(1).detach().cpu().numpy())
    mse_maps.extend(mse_map.squeeze(1).detach().cpu().numpy())
    ssim_maps.extend(ssim_map.permute(0, 2, 3, 1).squeeze(1).detach().cpu().numpy())


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

mse_maps=np.array(mse_maps)
mse_maps=(mse_maps-mse_maps.min())/(mse_maps.max()-mse_maps.min())
mse_maps=list(mse_maps)
gms_maps=np.array(gms_maps)
gms_maps=(gms_maps-gms_maps.min())/(gms_maps.max()-gms_maps.min())
gms_maps=list(gms_maps)

ssim_maps=np.array(ssim_maps)
ssim_maps=(ssim_maps-ssim_maps.min())/(ssim_maps.max()-ssim_maps.min())
#ssim_maps=list(ssim_maps)




save_mse_fig(class_type,epoch, image_yuan,reconst,image_gt,image_map, augs,mse_maps,gms_maps,list(ssim_maps))
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



