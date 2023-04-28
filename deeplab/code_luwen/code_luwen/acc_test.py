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
                    default='./checkpoints/Denoising/models/model_best.pth', type=str,
                    help='Path to weights')

args = parser.parse_args()
Train = opt['TRAINING']#加载训练的参数
class_type = Train["CLASS_TYPE"]


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
image_yuan=[]
reconst=[]
gt=[]
amap=[]
y_label=[]




for file_ in train_loader:
    #print(file_)

    input = file_[0][0].cuda()
    #print(input.shape)
    input_augment = file_[0][1].cuda()
    input_y = file_[1].cuda()
    input_mask = file_[2].cuda()

    mb_amap=0

    with torch.no_grad():
        output = model(input)
        #restored = torch.clamp(restored, 0, 1)#将输入的张量每个元素的范围限制在（0，1）之间，返回一个新的张量
        mb_amap += msgms(input,output,as_loss=False)
    mb_amap = mean_smoothing(mb_amap)
    amap.extend(mb_amap.squeeze(1).detach().cpu().numpy())
    image_yuan.extend(input.permute(0,2,3,1).detach().cpu().numpy())
    reconst.extend(output.permute(0,2,3,1).detach().cpu().numpy())
    gt.extend(input_mask.detach().cpu().numpy())
    y_label.extend(input_y.detach().cpu().numpy())

ep_amap = np.array(amap)
#图像级
image_ep_amap = ep_amap
image_ep_amap = image_ep_amap.reshape(image_ep_amap.shape[0],-1).max(axis=1)
y_label=np.asarray(y_label)
#print(ep_amap.max(axis=1))
image_score= roc_auc_score(y_label,image_ep_amap)#图片级得分
print("image_score: %.3f"%(image_score))
#图像级
ep_amap = (ep_amap-ep_amap.min())/(ep_amap.max()-ep_amap.min())
amap = list(ep_amap)

auroc = compute_auroc( epoch,np.array(amap),np.array(gt))#图像级得分
print("auroc: %.3f"%(auroc))

savefig(epoch, image_yuan,reconst,gt,amap)



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



