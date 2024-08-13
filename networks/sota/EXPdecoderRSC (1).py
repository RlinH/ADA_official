import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
bceloss = torch.nn.BCELoss()
class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        elif backbone == 'mixstyle':
            low_level_inplanes = 24
        elif backbone == 'ADA':
            low_level_inplanes = 24    
        else:
            low_level_inplanes = 24 

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv_1 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                          BatchNorm(256),
                                          nn.ReLU(),
                                          nn.Dropout(0.5),
                                          nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                          BatchNorm(256),
                                          nn.ReLU())
        self.last_conv_2 = nn.Sequential(nn.Dropout(0.1),
                                          nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x, low_level_feat,gt,epoch):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv_1(x)
        if not self.training:
            # 如果不是在训练模式，直接进行正常的前向传播
            out = self.last_conv_2(x)
            return out, x
        mode = 'S'
        visualize = False
        total_epochs =200
        if mode =='D':
            # 动态调整遮挡比例
            if epoch <= 18:
                percent = 1/100.0
            elif epoch <= 38:
                percent = 1/90.5
            elif epoch <= 58:
                percent = 1/85.0
            elif epoch <= 78:
                percent = 1/62.5
            else:
                percent = 1/50.0
        elif mode =='F':
            percent = 0.25
        elif mode =='IL':
            start_percent = 0.05  # 初始遮挡比例
            end_percent  = 0.25  # 结束时的遮挡比例
            percent = start_percent + (end_percent - start_percent) * epoch / total_epochs
        elif mode =='DL':
            start_percent = 0.25  # 初始遮挡比例
            end_percent  = 0.02   # 结束时的遮挡比例
            percent = start_percent + (end_percent - start_percent) * epoch / total_epochs
        elif mode=='S':
            # 动态调整遮挡比例
            
            if epoch <= 100:
                total_epochs = 100
                start_percent = 0.05  # 初始遮挡比例
                end_percent  = 0.02   # 结束时的遮挡比例
                percent = start_percent + (end_percent - start_percent) * epoch / total_epochs
                
                
            else:
                percent = 0
            
        #percent = 0.35
        # RSC技术的应用
        # 1. 切换到评估模式来计算梯度
        self.eval()

        # 2. 复制x用于梯度计算
        x_new = x.detach().requires_grad_(True)
        out_new = self.last_conv_2(x_new)
        out_new = F.interpolate(out_new, size=(512,512), mode='bilinear', align_corners=True)
        
        out_s =torch.sigmoid(out_new)
        loss = bceloss(out_s, gt)
        self.zero_grad()
        loss.backward()
        if visualize == True:
            x_numpy = x_new.cpu().detach().numpy()
            np.save(f'./innerdata/x_new_{epoch}.npy', x_numpy)
        
        # 3. 识别并生成遮挡掩码
        # 使用梯度信息来确定要遮挡的区域
        gradients = x_new.grad
        if visualize == True:
            gradients_numpy = gradients.cpu().detach().numpy()
            np.save(f'./innerdata/gradientsdata_{epoch}.npy', gradients_numpy)
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        # 保存梯度图数据
        if visualize == True:
            pooled_gradients_numpy = pooled_gradients.cpu().detach().numpy()
            np.save(f'./innerdata/pooled_gradients_numpy_{epoch}.npy', pooled_gradients_numpy)
        # 根据percent调整遮挡的区域
        # 使用绝对值来找到最大的变化（无论正负）
        #abs_pooled_gradients = torch.abs(pooled_gradients)
        threshold = torch.quantile(pooled_gradients, 1-percent)
        # 计算基于绝对值的阈值
        #threshold = torch.quantile(abs_pooled_gradients, 1 - percent)
        
        if visualize == True:
            mask_numpy = mask.cpu().detach().numpy()
            np.save(f'./innerdata/mask_numpy_{epoch}.npy', mask_numpy)
        #domain1
        #mask = torch.where(pooled_gradients >= threshold, 0 *torch.ones_like(pooled_gradients), torch.ones_like(pooled_gradients))    
        #domain2
        mask = torch.where(pooled_gradients >= threshold, torch.rand_like(pooled_gradients), torch.ones_like(pooled_gradients))
        mask = mask.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # 将 mask 调整为 [1, 256, 1, 1]

        # 4. 在应用掩码后重新计算前向传播
        self.train()
        out_masked = self.last_conv_2(x*mask)

        # 返回最终结果
        return out_masked, x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)
                            

if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    input = torch.rand(5, 3, 512, 512)
    gt = torch.rand(5, 1, 512, 512)
    epoch= 0
    output,_ = model(input,gt,epoch)
    print(output.size())


