import os

import torch
import yaml

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

train_dataset = get_training_data(train_dir,  Train['TRAIN_PS'])
print(len(train_dataset))




train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
                          shuffle=True, num_workers=0, drop_last=False)
val_dataset = get_validation_data(val_dir,  Train['VAL_PS'])
#print(val_dataset.__dict__)
print(len(val_dataset))
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0,
                        drop_last=False)
test_dataset = get_test_data(test_dir,  Train['TRAIN_PS'])
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
best_loss =10000

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

        for ii, data_val in enumerate(val_loader, 0):
            val_input = data_val[0][0].cuda()
            val_input_Augment = data[0][1].cuda()
            val_input_y = data[1].cuda()
            val_input_mask =data[2].cuda()

            with torch.no_grad():
                val_model_out = model_restored(val_input_Augment)

            val_mse_loss = mse(val_input, val_model_out)
            # print(mse_loss.shape)
            val_msgms_loss = msgms(val_input, val_model_out)
            val_ssim_loss = ssim(val_input, val_model_out)

            val_loss = val_mse_loss + val_msgms_loss + val_ssim_loss
            val_epoch_loss += val_loss.item()
        writer.add_scalar('val_epoch_loss', val_epoch_loss, epoch)
        val_epoch_mean_loss=val_epoch_loss/len(val_loader.dataset)
        writer.add_scalar('val_epoch_mean_loss', val_epoch_mean_loss, epoch)


        # Save the best loss model of validation
        if val_epoch_mean_loss< best_loss:
            best_loss = val_epoch_mean_loss
            best_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_best.pth"))
        print("[epoch %d loss: %.4f --- best_epoch %d Best_loss %.4f]" % (
            epoch, val_epoch_mean_loss, best_epoch, best_loss))
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