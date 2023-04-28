import torch
from torchvision.transforms import ToPILImage
import os
import numpy as np
from collections import OrderedDict
import cv2
from torch.utils.data import DataLoader
import argparse
from model.SUNet import SUNet_model
import yaml
from dataload.data_RGB import get_test_data

with open('../training.yaml', 'r') as config:
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

train_dataset = get_test_data(inp_dir,  Train['TRAIN_PS'])
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
# Load corresponding model architecture and weights
model = SUNet_model(opt)
model.cuda()

load_checkpoint(model, args.weights)
model.eval()

print('restoring images......')


for file_ in train_loader:
    #print(file_)

    input = file_[0][0].cuda()
    #print(input.shape)
    input_augment = file_[0][1].cuda()
    input_y = file_[1].cuda()
    input_mask = file_[2].cuda()
    #input_ = TF.to_tensor(img).unsqueeze(0).cuda()

    with torch.no_grad():
        restored = model(input)
        #restored = torch.clamp(restored, 0, 1)#将输入的张量每个元素的范围限制在（0，1）之间，返回一个新的张量
        restored = restored.cpu()

    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    #save original img
    mean = torch.as_tensor(imagenet_mean)[None, :, None, None]
    std = torch.as_tensor(imagenet_std)[None, :, None, None]
    img1 =  restored* std + mean
    img2 = file_[0][0]* std + mean



    img = ToPILImage()(img1[0, :])
    img.show()
    img = ToPILImage()(img2[0, :])
    img.show()


#     restored = img_as_ubyte(restored[0])
#
#     f = os.path.splitext(os.path.split(file_)[-1])[0]
#     save_img((os.path.join(out_dir, f + '.png')), restored)
#
# print(f"Files saved at {out_dir}")
# print('finish !')