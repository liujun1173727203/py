import os

import torch
import yaml
from utils.utils import mean_smoothing
from utils.auroc import compute_auroc
from sklearn.metrics import roc_auc_score
from utils import network_parameters
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from criterions.msgms import MSGMSLoss
from criterions.ssim import SSIMLoss

import time
import utils
import numpy as np
import random
from dataload.data_RGB import get_training_data, get_validation_data, get_test_data

from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from model.SUNet import SUNet_model

## Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

## Load yaml configuration file
with open('../training.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']#加载训练的参数
OPT = opt['OPTIM']#加载可调参的参数


## Build Model
print('==> Build the model')
model_restored = SUNet_model(opt)
p_number = network_parameters(model_restored)
model_restored.cuda()

## Training model path direction
mode = opt['MODEL']['MODE']#去噪 Denoising

model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
utils.mkdir(model_dir)
train_dir = Train['TRAIN_DIR']#训练数据集的路径
val_dir = Train['VAL_DIR']#验证数据集的路径
test_dir= Train['TEST_DIR']
class_type=Train['CLASS_TYPE']#类别类型

## GPU
gpus = ','.join([str(i) for i in opt['GPU']])#输入GPU的个数，经行遍历，也就是GPU的使用编码
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
device_ids = [i for i in range(torch.cuda.device_count())]#设备的索引
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
if len(device_ids) > 1:
    model_restored = nn.DataParallel(model_restored, device_ids=device_ids)

## Log
log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')#日志的保存路径
utils.mkdir(log_dir)#创建路径
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')#写入的路径

## Optimizer 定义优化器
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])#2e-4
optimizer = optim.Adam(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

## Scheduler (Strategy)  学习率的采用策略
warmup_epochs = 3
#OPT['EPOCHS'] =200
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                        eta_min=float(OPT['LR_MIN']))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

## Resume (Continue training by a pretrained model)#是否预训练，默认为False
if Train['RESUME']:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restored, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------')

## Loss
mse = nn.MSELoss(reduction='mean')#损失函数mse
L1_loss = nn.L1Loss()#损失函数采用L1损失
msgms = MSGMSLoss(num_scales=3, in_channels=3)#辅助损失函数
ssim = SSIMLoss(kernel_size=11, sigma=1.5)#辅助损失函数


## DataLoaders   TRAIN_PS=256 加载数据集，大小为256    batch_size=4
print('==> Loading datasets')

train_dataset = get_training_data(train_dir,  Train['TRAIN_PS'], class_type)
print(len(train_dataset))




train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
                          shuffle=True, num_workers=0, drop_last=False)
val_dataset = get_validation_data(val_dir,  Train['VAL_PS'],class_type)
#print(val_dataset.__dict__)
print(len(val_dataset))
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0,
                        drop_last=False)
test_dataset = get_test_data(test_dir,  Train['TRAIN_PS'],class_type)
print(test_dataset)

test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0,
                        drop_last=False)
# Show the training configuration
print(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])}
    Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
    Model parameters:   {p_number}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}
    GPU:                {'GPU' + str(device_ids)}''')
print('------------------------------------------------------------------')

# Start training!
print('==> Training start: ')
best_precision = 0

best_epoch = 0

total_start_time = time.time()


for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    val_epoch_loss=0
    epoch_loss = 0
    train_id = 1

    model_restored.train()#模型训练开始
    for i, data in enumerate(tqdm(train_loader), 0):
        # Forward propagation
        # print(data[0][0])
        # print(data[0][1])
        # print(i)

        # imagenet_mean = np.array([0.485, 0.456, 0.406])
        # imagenet_std = np.array([0.229, 0.224, 0.225])
        #
        # # save original img
        # # mean = torch.as_tensor(imagenet_mean)[None, :, None, None]
        # # std = torch.as_tensor(imagenet_std)[None, :, None, None]
        # # img1 = DATA[0][0][i] * std + mean
        # # img2 = DATA[0][1][i]* std + mean
        #
        #
        #
        # # img = ToPILImage()(img1[0, :])
        # # img.show()
        # # img = ToPILImage()(img2[0, :])
        # # img.show()
        for param in model_restored.parameters():
            param.grad = None
        #训练代码
        input = data[0][0].cuda()
        input_augment = data[0][1].cuda()
        input_y = data[1].cuda()
        input_mask =data[2].cuda()
        model_out = model_restored(input_augment)
        # Compute loss
        #loss = Charbonnier_loss(restored, target)
        #loss = L1_loss(model_out, input)

        mse_loss = mse(input, model_out)
        # print(mse_loss.shape)
        msgms_loss = msgms(input, model_out)
        ssim_loss = ssim(input, model_out)

        loss = mse_loss + msgms_loss + ssim_loss


        # Back propagation
        loss.backward()

        epoch_loss += loss.item()
        optimizer.step()

    writer.add_scalar('train_epoch_loss', epoch_loss, epoch)
    train_epoch_mean_loss = epoch_loss / len(train_loader.dataset)
    writer.add_scalar('train_epoch_mean_loss', train_epoch_mean_loss, epoch)


    #修改的验证代码
    if epoch % Train['VAL_AFTER_EVERY'] == 0:#每一次都评估一下
        model_restored.eval()
        test_epoch_loss=0
        map=[]
        image_yuan = []
        reconst = []
        gt = []
        y_label = []
        gt_list_px_lxl=[]
        pred_list_px_lxl=[]

        for ii, data_test in enumerate(test_loader, 0):
            test_input = data_test[0][0].cuda()
            test_input_Augment = data_test[0][1].cuda()
            test_input_y = data_test[1].cuda()
            test_input_mask =data_test[2].cuda()
            msgms_map=0
            with torch.no_grad():
                test_model_out = model_restored(test_input)
                msgms_map += msgms(test_input, test_model_out, as_loss=False)

            msgms_map = mean_smoothing(msgms_map)
            pixel_map = mean_smoothing(msgms_map)
            map.extend(msgms_map.squeeze(1).detach().cpu().numpy())
            image_yuan.extend(test_input.permute(0, 2, 3, 1).detach().cpu().numpy())
            reconst.extend(test_model_out.permute(0, 2, 3, 1).detach().cpu().numpy())
            gt.extend(test_input_mask.detach().cpu().numpy())
            y_label.extend(test_input_y.detach().cpu().numpy())



            #计算像素级得分需要的参数

            pixel_map= np.asarray(pixel_map.cpu())
            #print(mb_amap,mb_amap.shape)
            max_anomaly_score = pixel_map.max()
            min_anomaly_score = pixel_map.min()
            map_pixel =(pixel_map - min_anomaly_score)/(max_anomaly_score-min_anomaly_score)


            gt_np = np.asarray(test_input_mask.detach().cpu())
            #print(gt_np,gt_np.shape)
            #像素级
            gt_list_px_lxl.extend(gt_np[0].flatten())
            pred_list_px_lxl.extend(map_pixel[0].flatten())


            #计算loss
            test_mse_loss = mse(test_input, test_model_out)
            # print(mse_loss.shape)
            test_msgms_loss = msgms(test_input, test_model_out)
            test_ssim_loss = ssim(test_input, test_model_out)

            test_loss = test_mse_loss + test_msgms_loss + test_ssim_loss
            test_epoch_loss += test_loss.item()
        ep_amap = np.array(map)
        #图像级
        image_ep_amap = ep_amap
        image_ep_amap = image_ep_amap.reshape(image_ep_amap.shape[0], -1).max(axis=1)
        y_label = np.asarray(y_label)
        # print(ep_amap.max(axis=1))
        image_score = roc_auc_score(y_label, image_ep_amap)  # 图片级得分
        print("image_score: %.3f" % (image_score))
        #图像级
        ep_amap = (ep_amap - ep_amap.min()) / (ep_amap.max() - ep_amap.min())
        amap = list(ep_amap)
        auroc = compute_auroc(epoch, np.array(amap), np.array(gt))  # 图像级得分
        print("image_auc: %.3f" % (auroc))
        #像素级得分
        pixel_auc = roc_auc_score(gt_list_px_lxl, pred_list_px_lxl)
        print('pixel_auc:', pixel_auc)

        #将精度传入日志
        writer.add_scalar('precision_image',image_score,epoch)
        #print("auroc: %.3f" % (auroc))
        writer.add_scalar('precision', auroc, epoch)
        #savefig(epoch, image_yuan, reconst, gt, amap)
        #将损失写入tensorboard中
        writer.add_scalar('test_epoch_loss', test_epoch_loss, epoch)
        test_epoch_mean_loss=test_epoch_loss/len(test_loader.dataset)
        writer.add_scalar('test_epoch_mean_loss', test_epoch_mean_loss, epoch)


        # Save the best precision model of validation
        if auroc > best_precision :
            best_precision = auroc
            best_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_best.pth"))
        print("[epoch %d loss: %.4f --- best_epoch %d Best_precision %.4f]" % (
            epoch, test_epoch_mean_loss, best_epoch, best_precision))
    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              train_epoch_mean_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model_restored.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))


    writer.add_scalar('train_lr', scheduler.get_lr()[0], epoch)
writer.close()

total_finish_time = (time.time() - total_start_time)  # seconds
print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))