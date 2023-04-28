import numpy as np
import torch
from numpy import ndarray as NDArray
from torch import Tensor
import random
import math



# cutout_size = random.choice([2,4,8,16])
# num_disjoint_masks=3

def create_disjoint_masks(
        img_size: [int, int],#图像输入的尺寸大小
        cutout_size: int = 8,#随机选择2，4，8，16里面的一个值，默认为8
        num_disjoint_masks: int = 3,
    ):

        img_h, img_w = img_size#256，256
        grid_h = math.ceil(img_h / cutout_size)#返回一个不小于img_h / cutout_size的整数，256/8=32
        grid_w = math.ceil(img_w / cutout_size)#返回一个不小于img_w / cutout_size的整数，256/8=32
        num_grids = grid_h * grid_w#将图片分成32*32个大小为8*8的网格，[0,1,....,1023]共1024个
        disjoint_masks = []#mask掉的网格列表
        for grid_ids in np.array_split(np.random.permutation(num_grids), num_disjoint_masks):#np.random.permutation随机返回一个打乱的序列， np.array_split每num_disjoint_masks=3个为一组
            flatten_mask = np.ones(num_grids)#[1,1,1..,1]1024个1
            flatten_mask[grid_ids] = 0#将刚才分组的下表序列的值都变成0
            mask = flatten_mask.reshape((grid_h, grid_w))#将flatten_mask变为32*32的格式 赋值个mask 
            mask = mask.repeat(cutout_size, axis=0).repeat(cutout_size, axis=1)#沿着x,y轴复制8份，变成256*256
            #mask = torch.tensor(mask, requires_grad=False, dtype=torch.float)
            #mask = mask.to(self.cfg.params.device)
            disjoint_masks.append(mask)#一张图片产生三张不同位置的mask

        return disjoint_masks[0]
class make_mask(object):
    """Base class for both cutpaste variants with common operations"""
    def __init__(self, cutout_size=8,num_disjoint_masks=4, transform=None):
        self.transform = transform
        self.num_disjoint_masks=num_disjoint_masks
        self.cutout_size=cutout_size
    def __call__(self, org_img):
        # apply transforms to both images

        input=np.asarray(org_img)
        #print(input.shape)
        h, w, c = input.shape
        num_disjoint_masks = self.num_disjoint_masks# num_disjoint_masks=3
        disjoint_masks = create_disjoint_masks((h, w), self.cutout_size, self.num_disjoint_masks)#cutout_sizes: [2, 4, 8, 16]
        # a=disjoint_masks
        # b=disjoint_masks
        # c=disjoint_masks
        # three_wei_disjoint_masks=np.array([a,b,c])
        # three_wei_disjoint_masks=three_wei_disjoint_masks.transpose(1,2,0)

        #print(three_wei_disjoint_masks.shape)

        input[:,:,0] = input[:,:,0] * disjoint_masks
        input[:,:,1] = input[:,:,1] * disjoint_masks
        input[:,:,2] = input[:,:,1] * disjoint_masks
        #print(input.shape)
        if self.transform:
            org_img = self.transform(org_img)
            img = self.transform(input)
        return org_img




