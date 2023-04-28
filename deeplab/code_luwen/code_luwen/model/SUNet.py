import torch.nn as nn
from model.SUNet_detail import SUNet


class SUNet_model(nn.Module):
    def __init__(self, config):
        super(SUNet_model, self).__init__()
        self.config = config
        self.swin_unet = SUNet(img_size=config['SWINUNET']['IMG_SIZE'],#256
                               patch_size=config['SWINUNET']['PATCH_SIZE'],#4
                               in_chans=3,
                               out_chans=3,
                               embed_dim=config['SWINUNET']['EMB_DIM'],#96
                               depths=config['SWINUNET']['DEPTH_EN'],#[8,8,8,8]
                               num_heads=config['SWINUNET']['HEAD_NUM'],#[8,8,8,8]
                               window_size=config['SWINUNET']['WIN_SIZE'],#8
                               mlp_ratio=config['SWINUNET']['MLP_RATIO'],#0.1
                               qkv_bias=config['SWINUNET']['QKV_BIAS'],
                               qk_scale=config['SWINUNET']['QK_SCALE'],#8
                               drop_rate=config['SWINUNET']['DROP_RATE'],#0
                               drop_path_rate=config['SWINUNET']['DROP_PATH_RATE'],#0.1
                               ape=config['SWINUNET']['APE'],#False
                               patch_norm=config['SWINUNET']['PATCH_NORM'],#ture
                               use_checkpoint=config['SWINUNET']['USE_CHECKPOINTS'])#False

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits
