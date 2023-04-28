import os

import torch
import yaml
from utils.auroc import compute_auroc
from utils import network_parameters
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from criterions.msgms import MSGMSLoss
from criterions.ssim import SSIMLoss
from criterions.loss import SSIM, FocalLoss
from dataload.data_loader import MVTecDRAEMTrainDataset, MVTecDRAEMTestDataset

import math
from utils.utils import mean_smoothing
import time
import utils
import numpy as np
import random
from dataload.data_RGB import get_training_data, get_validation_data, get_test_data

from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from model.SUNet import SUNet_model
from model.model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork

from sklearn.metrics import roc_auc_score, average_precision_score
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

def weights_init(m):#权重初始化
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
## Build Model 重建模型的定义
print('==> Build the model')
model_restored = SUNet_model(opt)
p_number = network_parameters(model_restored)
model_restored.cuda()
#分割模型的定义
model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
seg_net_p_number = network_parameters(model_seg)
model_seg.cuda()
model_seg.apply(weights_init)

## Training model path direction
mode = opt['MODEL']['MODE']#去噪 Denoising

model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
utils.mkdir(model_dir)
train_dir = Train['TRAIN_DIR']#训练数据集的路径
val_dir = Train['VAL_DIR']#验证数据集的路径
test_dir= Train['TEST_DIR']
class_type=Train['CLASS_TYPE']#类别类型
train_aug_dir = Train['TRAIN_AUG_DIR']


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


## GPU
gpus = ','.join([str(i) for i in opt['GPU']])#输入GPU的个数，经行遍历，也就是GPU的使用编码
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
device_ids = [i for i in range(torch.cuda.device_count())]#设备的索引
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
if len(device_ids) > 1:
    model_restored = nn.DataParallel(model_restored, device_ids=device_ids)
    model_seg = nn.DataParallel(model_seg, device_ids=device_ids)

## Log
log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')#日志的保存路径
utils.mkdir(log_dir)#创建路径
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')#写入的路径

"""
optimizer = torch.optim.Adam([
                                      {"params": model.parameters(), "lr": args.lr},
                                      {"params": model_seg.parameters(), "lr": args.lr}])#优化器

scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)#学习策略

"""

## Optimizer 定义优化器
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])#2e-4
optimizer = optim.Adam([{"params":model_restored.parameters()},{"params":model_seg.parameters()}], lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

## Scheduler (Strategy)  学习率的采用策略
warmup_epochs = 3
#OPT['EPOCHS'] =200
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                        eta_min=float(OPT['LR_MIN']))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

## Resume (Continue training by a pretrained model)#是否预训练，默认为False
#这一段不用看
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

## RIAD中的Loss
mse = nn.MSELoss(reduction='mean')#损失函数mse
L1_loss = nn.L1Loss()#损失函数采用L1损失
msgms = MSGMSLoss(num_scales=3, in_channels=3)#辅助损失函数
ssim = SSIMLoss(kernel_size=11, sigma=1.5)#辅助损失函数
#dream中的loss
loss_l2 = torch.nn.modules.loss.MSELoss()#l损失
loss_ssim = SSIM()#结构相似度损失
loss_focal = FocalLoss()#focal损失


## DataLoaders   TRAIN_PS=256 加载数据集，大小为256    batch_size=4
print('==> Loading datasets')


train_dataset = MVTecDRAEMTrainDataset(train_dir+"/"+class_type+"/train/good/",train_aug_dir,resize_shape=[256,256])
train_loader = DataLoader(train_dataset, batch_size=OPT['BATCH'], shuffle=False, num_workers=0)
print(len(train_dataset))

test_dataset = MVTecDRAEMTestDataset(test_dir+"/"+class_type+"/test",resize_shape=[256,256])
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=0)
print(len(test_dataset))


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
cutout_size = 4
best_precision = 0

best_epoch = 0

total_start_time = time.time()


for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    val_epoch_loss=0
    epoch_loss = 0
    train_id = 1
    #可视化代码
    #run_name = 'DRAEM_test_'+str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"+obj_name+'_'
    #visualizer = TensorboardVisualizer(log_dir=os.path.join(log_path, run_name+"/"))#每个对象测试可视化
    #训练代码
    model_restored.train()#模型训练开始
    model_seg.train()
    for i, data in enumerate(tqdm(train_loader), 0):
       
        for param in model_restored.parameters():
            param.grad = None
        for param in model_seg.parameters():
            param.grad = None
        #训练代码



        input = data["image"].cuda()
        input_augment = data["augmented_image"].cuda()
        input_y = data["has_anomaly"].cuda()
        input_mask =data["anomaly_mask"].cuda()
        
        model_out = reconstruct(input_augment, cutout_size, model_restored)
        joined_in = torch.cat((model_out,input_augment), dim=1)
        out_mask = model_seg(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)


        l2_loss = loss_l2(model_out, input)
        ssim_loss = loss_ssim(model_out, input)
        segment_loss =loss_focal(out_mask_sm,input_mask)
        loss = l2_loss+ ssim_loss+ segment_loss

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
        model_seg.eval()
        test_epoch_loss=0
        
        img_dim = 256
        


        total_pixel_scores = np.zeros((img_dim * img_dim * len(test_dataset)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(test_dataset)))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []

        display_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
        display_gt_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
        display_out_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
        display_in_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
        cnt_display = 0
        display_indices = np.random.randint(len(test_loader), size=(16,))

        for ii, data_test in enumerate(test_loader, 0):
            test_input = data_test["image"].cuda()#输入

            is_normal = data_test["has_anomaly"].detach().numpy()[0,0]
            anomaly_score_gt.append(is_normal)#所谓的异常标记

            test_input_mask =data_test["mask"]#mask
            true_mask_cv = test_input_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))#mask图片转换通道位置，满足需要

            with torch.no_grad():
                test_model_rec = reconstruct(test_input, cutout_size, model_restored)
                joined_in = torch.cat((test_model_rec.detach(), test_input), dim=1)
                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

            if ii in display_indices:
                t_mask = out_mask_sm[:, 1:, :, :]
                display_images[cnt_display] = test_model_rec[0]
                display_gt_images[cnt_display] = test_input[0]
                display_out_masks[cnt_display] = t_mask[0]
                display_in_masks[cnt_display] = test_input_mask[0]
                cnt_display += 1
            out_mask_cv = out_mask_sm[0, 1, :, :].detach().cpu().numpy()

            out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[:, 1:, :, :], 21, stride=1,
                                                               padding=21 // 2).cpu().detach().numpy()
            image_score = np.max(out_mask_averaged)#图像级得分

            anomaly_score_prediction.append(image_score)#图像级列表

            flat_true_mask = true_mask_cv.flatten()#真实的mask
            flat_out_mask = out_mask_cv.flatten()#预测的mask
            total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
            total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
            mask_cnt += 1
        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
        ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
        
        print("AUC Image:  " +str(auroc))
        print("AP Image:  " +str(ap))
        print("AUC Pixel:  " +str(auroc_pixel))
        print("AP Pixel:  " +str(ap_pixel))
        print("==============================")
        #将精度传入日志
        writer.add_scalar('pixel',auroc_pixel,epoch)
        writer.add_scalar('AP_pixel', ap_pixel, epoch)
        #print("auroc: %.3f" % (auroc))
        writer.add_scalar('image', auroc, epoch)
        writer.add_scalar('AP_image', ap, epoch)
        #savefig(epoch, image_yuan, reconst, gt, amap)


        # Save the best precision model of validation
        if auroc > best_precision :
            best_precision = auroc
            best_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_best.pth"))
            
            torch.save({'epoch': epoch,
                        'state_dict': model_seg.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_seg_best.pth"))
        print("[epoch %d  --- best_epoch %d Best_precision %.4f ]" % (
            epoch, best_epoch, best_precision))
    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model_restored.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))
    torch.save({'epoch': epoch,
                'state_dict': model_seg.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_seg_latest.pth"))


    writer.add_scalar('train_lr', scheduler.get_lr()[0], epoch)
writer.close()

total_finish_time = (time.time() - total_start_time)  # seconds
print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))