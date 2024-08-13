import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import torch.utils.model_zoo as model_zoo
class ControlShift(nn.Module):
    def __init__(self):
        super(ControlShift, self).__init__()
        # 定义平均池化层，将224x224x3的图像缩减为1x1x3
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # MLP接受3个特征输入（因为池化后每个通道的平均值），输出6个参数（每个通道的scale和shift）
        self.fc = nn.Linear(3, 6)  # 输出维度修改为6
    def forward(self, image):
        # 应用平均池化
        x = self.avg_pool(image)
        # 将数据展平为一维向量
        x = torch.flatten(x, 1)
        # 应用MLP获取scale和shift参数
        x = self.fc(x)
        x = torch.tanh(x)
        # 分割scale和shift参数，使其对应每个通道
        scale, shift = x.split(3, dim=1)  # 修改为3，因为现在每个通道有独立的参数
        # 重新调整scale和shift的形状以匹配图像的形状
        scale = scale.view(-1, 3, 1, 1)  # 确保scale和shift的形状与通道数相匹配
        shift = shift.view(-1, 3, 1, 1)
        # 应用scale和shift调整图像
        #image = image * scale + shift
        #
        image = image * (1+scale) + shift
        return image
    
def conv_bn(inp, oup, stride, BatchNorm):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm(oup),
        nn.ReLU6(inplace=True)
    )


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class AdaptiveNormalization(nn.Module):
    def __init__(self, num_features):
        super(AdaptiveNormalization, self).__init__()
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        return x * self.scale + self.shift

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, dilation, expand_ratio, BatchNorm):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.kernel_size = 3
        self.dilation = dilation

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, 1, 1, bias=False),
                BatchNorm(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, 1, bias=False),
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, 1, bias=False),
                BatchNorm(oup),
            )

    def forward(self, x):
        x_pad = fixed_padding(x, self.kernel_size, dilation=self.dilation)
        if self.use_res_connect:
            x = x + self.conv(x_pad)
        else:
            x = self.conv(x_pad)
        return x


class ADA(nn.Module):
    def __init__(self, output_stride=8, BatchNorm=None, width_mult=1., pretrained=True):
        super(ADA, self).__init__()
        self.conshift = ControlShift()
        block = InvertedResidual
        input_channel = 32
        current_stride = 1
        rate = 1
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.features = [conv_bn(3, input_channel, 2, BatchNorm)]
        #self.features.append(AdaptiveNormalization(input_channel))
        current_stride *= 2
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            if current_stride == output_stride:
                stride = 1
                dilation = rate
                rate *= s
            else:
                stride = s
                dilation = 1
                current_stride *= s
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, stride, dilation, t, BatchNorm))
                    
                else:
                    self.features.append(block(input_channel, output_channel, 1, dilation, t, BatchNorm))
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)
        self._initialize_weights()

        if pretrained:

            self._load_pretrained_model()

        self.low_level_features = self.features[0:4]
        self.high_level_features = self.features[4:]

    def forward(self, image):
        
        standardized_images = torch.zeros_like(image)
        for i in range(image.shape[0]):
            img_mean = image[i].mean()
            img_std = image[i].std()
            epsilon = 1e-8
            standardized_images[i] = (image[i] - img_mean) / (img_std + epsilon)
        x =  standardized_images
        
        aug_x = self.conshift(x)
        
        normalized_aug_images = torch.zeros_like(aug_x)
        for i in range(aug_x.shape[0]):
            img_mean = aug_x[i].mean()
            img_std = aug_x[i].std()
            normalized_aug_images[i] = (aug_x[i] - img_mean) / (img_std + 1e-8)

        x = normalized_aug_images
        
        low_level_feat = self.low_level_features(x)
        x = self.high_level_features(low_level_feat)
        return x, low_level_feat

    def OLD_load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('http://jeff95.me/models/mobilenet_v2-6a65762b.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('http://jeff95.me/models/mobilenet_v2-6a65762b.pth')
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in pretrain_dict.items():
            if k in state_dict and state_dict[k].size() == v.size():
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict, strict=False)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == "__main__":
    input = torch.rand(1, 3, 512, 512)
    model = ADA(output_stride=16, BatchNorm=nn.BatchNorm2d)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
