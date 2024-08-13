import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import torch.utils.model_zoo as model_zoo
import numpy as np
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# self.fc = nn.Sequential(
#     DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1),  # 深度可分离卷积
#     nn.ReLU(),
#     nn.AdaptiveAvgPool2d((4, 4)),  # 使用更大的池化窗口
#     nn.Flatten(),
#     nn.Linear(channels * 4 * 4, channels * h * w_fft * 2),  # 减少全连接层的输入维度
#     nn.Tanh()
# )  
    
    
class GlobalFilter(nn.Module):
    def __init__(self, channels=3, h=384, w=384):
        super(GlobalFilter, self).__init__()
        self.h, self.w = h, w
        w_fft = w // 2 + 1  # 计算rfft2的输出尺寸
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),  # 使用深度可分离卷积
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # 使用更大的池化窗口
            nn.Flatten(),
            nn.Linear(channels * 4 * 4, channels * h * w_fft * 2),  # 减少全连接层的输入维度
            nn.Tanh()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.h and W == self.w, "Input height and width must match initialized dimensions ({}, {})".format(self.h, self.w)
        
        # 通过全连接层获得权重
        new_weights = self.fc(x)
        new_weights = new_weights.view(B, C, self.h, W // 2 + 1, 2)
        weight_complex = torch.view_as_complex(new_weights)
        
        # 应用rfft2
        x_fft = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        
        # 在频域应用权重
        x_filtered = x_fft * weight_complex  # 使用乘法而非加法以应用滤波
        
        # 应用逆FFT
        x_ifft = torch.fft.irfft2(x_filtered, s=(H, W), dim=(2, 3), norm='ortho')
        
        # 添加残差连接
        return x_ifft + x

def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v)
    Vc = torch.stack((Vc.real, Vc.imag), -1)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    x_shape = X.shape
    X = X.float()
    N = x_shape[-1]
    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.fft.ifft(torch.complex(V[..., 0], V[..., 1]), dim=1)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def fft_encode(x):
    return torch.fft.fft2(x)

def fft_decode(X):
    return torch.fft.ifft2(X).real


class DctFftEncoding(object):
    def __init__(self, vec_dim, mask_size=384, drop_prob=0.01):
        self.vec_dim = vec_dim
        self.mask_size = mask_size
        self.drop_prob = drop_prob  # 新增的参数，用于控制丢弃概率
        assert vec_dim <= mask_size * mask_size
        self.dct_vector_coords = self.get_dct_vector_coords(r=mask_size)

    def encode(self, masks, dim=None):
        if dim is None:
            dct_vector_coords = self.dct_vector_coords[:self.vec_dim]
        else:
            dct_vector_coords = self.dct_vector_coords[:dim]
        masks = masks.view([-1, self.mask_size, self.mask_size]).to(dtype=torch.float32)
        dct_all = dct_2d(masks, norm='ortho')
        xs, ys = dct_vector_coords[:, 0], dct_vector_coords[:, 1]
        dct_vectors = dct_all[:, xs, ys]
        fft_vectors = fft_encode(masks)
        return dct_vectors, fft_vectors

    def decode(self, dct_vectors, fft_vectors, dim=None):
        device = dct_vectors.device

        if dim is None:
            dct_vector_coords = self.dct_vector_coords[:self.vec_dim]
        else:
            dct_vector_coords = self.dct_vector_coords[:dim]
            dct_vectors = dct_vectors[:, :dim]

        # 随机丢弃部分DCT vectors
        mask = torch.rand(dct_vectors.shape, device=device) > self.drop_prob
        dct_vectors = dct_vectors * mask

        N = dct_vectors.shape[0]
        dct_trans = torch.zeros([N, self.mask_size, self.mask_size], dtype=dct_vectors.dtype).to(device)
        xs, ys = dct_vector_coords[:, 0], dct_vector_coords[:, 1]
        dct_trans[:, xs, ys] = dct_vectors

        mask_rc_dct = idct_2d(dct_trans, norm='ortho')

        # 随机丢弃部分FFT vectors
        mask = torch.rand(fft_vectors.shape, device=device) > self.drop_prob
        fft_vectors = fft_vectors * mask

        mask_rc_fft = fft_decode(fft_vectors)
        return mask_rc_dct, mask_rc_fft

    def get_dct_vector_coords(self, r=128):
        dct_index = []
        for i in range(r):
            if i % 2 == 0:
                index = [(i-j, j) for j in range(i+1)]
                dct_index.extend(index)
            else:
                index = [(j, i-j) for j in range(i+1)]
                dct_index.extend(index)
        for i in range(r, 2*r-1):
            if i % 2 == 0:
                index = [(i-j, j) for j in range(i-r+1,r)]
                dct_index.extend(index)
            else:
                index = [(j, i-j) for j in range(i-r+1,r)]
                dct_index.extend(index)
        dct_idxs = np.asarray(dct_index)
        return dct_idxs


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


class FFNET(nn.Module):
    def __init__(self, output_stride=8, BatchNorm=None, width_mult=1., pretrained=True):
        super(FFNET, self).__init__()
        # self.conshift = ControlShift()
        # self.bezier = BezierTrans()
        self.fft_filter = GlobalFilter()
        # 初始化 DCT-FFT 编码器
        self.coder = DctFftEncoding(vec_dim=15000, mask_size=384)
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
        
        # standardized_images = torch.zeros_like(image)
        # for i in range(image.shape[0]):
        #     img_mean = image[i].mean()
        #     img_std = image[i].std()
        #     epsilon = 1e-8
        #     standardized_images[i] = (image[i] - img_mean) / (img_std + epsilon)
        # x =  standardized_images
        
        standardized_images = torch.zeros_like(image)
        for i in range(image.shape[0]):
            img_min = image[i].min()
            img_max = image[i].max()
            epsilon = 1e-8
            standardized_images[i] = (image[i] - img_min) / (img_max - img_min + epsilon)
        x =  standardized_images
        if self.training:
            # 对每个通道进行编码和解码
            dct_vectors_list = []
            fft_vectors_list = []
            reconstructed_dct_channels = []
            reconstructed_fft_channels = []
            for c in range(3):  # 对RGB三个通道分别进行处理
                dct_vectors, fft_vectors = self.coder.encode(x[:, c, :, :])
                reconstructed_dct_channel, reconstructed_fft_channel = self.coder.decode(dct_vectors, fft_vectors)

                dct_vectors_list.append(dct_vectors)
                fft_vectors_list.append(fft_vectors)
                reconstructed_dct_channels.append(reconstructed_dct_channel)
                reconstructed_fft_channels.append(reconstructed_fft_channel)

            # 组合通道
            reconstructed_dct_image = torch.stack(reconstructed_dct_channels, dim=1).real
            reconstructed_fft_image = torch.stack(reconstructed_fft_channels, dim=1).real
            x= reconstructed_dct_image
            
            
        #x = self.fft_filter(x)
         
        
#         x = self.bezier(x)
#         x = self.conshift(x)
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
    input = torch.rand(8, 3, 384, 384)
    model = FFNET(output_stride=16, BatchNorm=nn.BatchNorm2d)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
