import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import torch.utils.model_zoo as model_zoo

class BezierTrans(nn.Module):
    def __init__(self):
        super(BezierTrans, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 用于计算图像通道的平均值
        self.fc = nn.Linear(3, 4)  # MLP将3个通道的平均值映射到P1和P2（每个2个值）
        
        # 固定的控制点
        self.P0 = torch.tensor([0.0, 0.0], requires_grad=False)
        self.P3 = torch.tensor([1.0, 1.0], requires_grad=False)
        self.t_values = torch.linspace(0, 1, steps=100000).view(-1, 1).to('cuda:0')

    def forward(self, batch_images):
        batch_size = batch_images.shape[0]
        avg_values = self.avg_pool(batch_images).view(batch_size, -1)
        control_points = self.fc(avg_values)  # 每个图像产生4个值，分别用于P1和P2

        # 分割P1和P2
        P1 = control_points[:, :2].view(batch_size, 1, 2).to(batch_images.device)
        P2 = control_points[:, 2:].view(batch_size, 1, 2).to(batch_images.device)
        P1 = torch.sigmoid(P1)
        P2 = torch.sigmoid(P2)

        #t_values = torch.linspace(0, 1, steps=256).view(-1, 1).to(batch_images.device)
        
        bezier_curve = self.compute_bezier_curve(self.t_values, P1, P2, batch_size)
        
        enhanced_images = torch.zeros_like(batch_images)

        for i in range(3):  # 对每个颜色通道应用映射
            channel = batch_images[:, i, :, :]
            #channel_indices = (channel * 255).long()
            channel_indices = (channel * 99999).long()
            for j in range(batch_size):  # 为每个图像应用其特定的Bezier曲线
                enhanced_images[j, i, :, :] = bezier_curve[j, channel_indices[j], 1]

        return enhanced_images
    
    def compute_bezier_curve(self, t_values, P1, P2, batch_size):
        device = t_values.device  # Get the device from t_values
        P0 = self.P0.to(device).view(1, 1, 2).expand(batch_size, t_values.size(0), 2)
        P3 = self.P3.to(device).view(1, 1, 2).expand(batch_size, t_values.size(0), 2)

        # Ensure P1 and P2 are on the same device
        P1 = P1.to(device)
        P2 = P2.to(device)

        # Compute the Bezier curve
        bezier_curve = (1 - t_values)**3 * P0 + \
                       3 * (1 - t_values)**2 * t_values * P1.expand(batch_size, t_values.size(0), 2) + \
                       3 * (1 - t_values) * t_values**2 * P2.expand(batch_size, t_values.size(0), 2) + \
                       t_values**3 * P3

        return bezier_curve


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
        self.bezier = BezierTrans()
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
        
#         standardized_images = torch.zeros_like(image)
#         for i in range(image.shape[0]):
#             img_mean = image[i].mean()
#             img_std = image[i].std()
#             epsilon = 1e-8
#             standardized_images[i] = (image[i] - img_mean) / (img_std + epsilon)
#         x =  standardized_images
        
        standardized_images = torch.zeros_like(image)
        for i in range(image.shape[0]):
            img_min = image[i].min()
            img_max = image[i].max()
            epsilon = 1e-8
            standardized_images[i] = (image[i] - img_min) / (img_max - img_min + epsilon)
        x =  standardized_images
        
        
        
        x = self.bezier(x)
        x = self.conshift(x)
#         normalized_aug_images = torch.zeros_like(x)
#         for i in range(x.shape[0]):
#             img_min = x[i].min()
#             img_max = x[i].max()
#             epsilon = 1e-8
#             normalized_aug_images[i] = (x[i] - img_min) / (img_max - img_min + epsilon)

#         x = normalized_aug_images 
        
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
