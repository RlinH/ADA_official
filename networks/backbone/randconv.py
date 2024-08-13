#!/usr/bin/env python
"""
convolution layer whose weights can be randomized

Created by zhenlinxu on 12/28/2019
"""
import torch
import torch.nn as nn
from torch.nn import Conv2d
import torch.nn.functional as F
import torchvision.transforms as transforms
import math
import numpy as np
import random
from collections.abc import Sequence


class RandConvModule(nn.Module):
    def __init__(self, net=None, kernel_size=3, in_channels=3, out_channels=3,
                 rand_bias=False,
                 mixing=False,
                 identity_prob=0.0, distribution='kaiming_normal',
                 data_mean=None, data_std=None, clamp_output=False,
                 ):
        """

        :param net:
        :param kernel_size:
        :param in_channels:
        :param out_channels:
        :param rand_bias:
        :param mixing: "random": output = (1-alpha)*input + alpha* randconv(input) where alpha is a random number sampled
                            from a distribution defined by res_dist
        :param identity_prob:
        :param distribution:
        :param data_mean:
        :param data_std:
        :param clamp_output:
        """

        super(RandConvModule, self).__init__()

        # if the input is not normalized, we need to normalized with given mean and std (tensor of size 3)
        self.register_buffer('data_mean', None if data_mean is None else torch.tensor(data_mean).reshape(3, 1, 1))
        self.register_buffer('data_std', None if data_std is None else torch.tensor(data_std).reshape(3, 1, 1))

        # adjust output range based on given data mean and std, (clamp or norm)
        # clamp with clamp the value given that the was image pixel values [0,1]
        # normalize will linearly rescale the values to the allowed range
        # The allowed range is ([0, 1]-data_mean)/data_std in each color channel
        self.clamp_output = clamp_output
        if self.clamp_output:
            assert (self.data_mean is not None) and (self.data_std is not None), "Need data mean/std to do output range adjust"
        self.register_buffer('range_up', None if not self.clamp_output else (torch.ones(3).reshape(3, 1, 1) - self.data_mean) / self.data_std)
        self.register_buffer('range_low', None if not self.clamp_output else (torch.zeros(3).reshape(3, 1, 1) - self.data_mean) / self.data_std)

        if isinstance(kernel_size, Sequence) and len(kernel_size) == 1:
            kernel_size = kernel_size[0]

        if mixing:
            out_channels = in_channels

        # generate random conv layer
        print("Add RandConv layer with kernel size {}, output channel {}".format(kernel_size, out_channels))
        self.randconv = MultiScaleRandConv2d(in_channels=in_channels, out_channels=out_channels, kernel_sizes=kernel_size,
                                             stride=1, rand_bias=rand_bias,
                                             distribution=distribution,
                                             clamp_output=self.clamp_output,
                                             range_low=self.range_low,
                                             range_up=self.range_up,
                                             )


        # mixing mode
        self.mixing = mixing # In the mixing mode, a mixing connection exists between input and output of random conv layer
        # self.res_dist = res_dist
        self.res_test_weight = None
        if self.mixing:
            assert in_channels == out_channels or out_channels == 1, \
                'In mixing mode, in/out channels have to be equal or out channels is 1'
            self.alpha = random.random()  # sample mixing weights from uniform distributin (0, 1)

        self.identity_prob = identity_prob  # the probability that use original input

    def forward(self, input):
        """assume that the input is whightened"""

        ######## random conv ##########
        if not (self.identity_prob > 0 and torch.rand(1) < self.identity_prob):
            # whiten input and go through randconv
            output = self.randconv(input)

            if self.mixing:
                output = (self.alpha*output + (1-self.alpha)*input)

            if self.clamp_output:
                output = torch.max(torch.min(output, self.range_up), self.range_low)
        else:
            output = input

        return output

    def parameters(self, recurse=True):
        return self.randconv.parameters()

    def trainable_parameters(self, recurse=True):
        return self.randconv.trainable_parameters()

    def whiten(self, input):
        return (input - self.data_mean) / self.data_std

    def dewhiten(self, input):
        return input * self.data_std + self.data_mean

    def randomize(self):
        self.randconv.randomize()

        if self.mixing:
            self.alpha = random.random()

    def set_test_res_weight(self, w):
        self.res_test_weight = w

class RandConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, rand_bias=True,
                 distribution='kaiming_normal',
                 clamp_output=None, range_up=None, range_low=None, **kwargs):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param rand_bias:
        :param distribution:
        :param clamp_output:
        :param range_up:
        :param range_low:
        :param kwargs:
        """
        super(RandConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=rand_bias, **kwargs)

        self.rand_bias = rand_bias
        self.distribution = distribution

        self.clamp_output = clamp_output
        self.register_buffer('range_up', None if not self.clamp_output else range_up)
        self.register_buffer('range_low', None if not self.clamp_output else range_low)
        if self.clamp_output:
            assert (self.range_up is not None) and (self.range_low is not None), "No up/low range given for adjust"


    def randomize(self):
        new_weight = torch.zeros_like(self.weight)
        with torch.no_grad():
            if self.distribution == 'kaiming_uniform':
                nn.init.kaiming_uniform_(new_weight, nonlinearity='conv2d')
            elif self.distribution == 'kaiming_normal':
                nn.init.kaiming_normal_(new_weight, nonlinearity='conv2d')
            elif self.distribution == 'kaiming_normal_clamp':
                fan = nn.init._calculate_correct_fan(new_weight, 'fan_in')
                gain = nn.init.calculate_gain('conv2d', 0)
                std = gain / math.sqrt(fan)
                with torch.no_grad():
                    new_weight.normal_(0, std)
                    new_weight = new_weight.clamp(-2*std, 2*std)
            elif self.distribution == 'xavier_normal':
                nn.init.xavier_normal_(new_weight)
            else:
                raise NotImplementedError()

        self.weight = nn.Parameter(new_weight.detach())
        if self.bias is not None and self.rand_bias:
            # new_bias = self.bias.clone().detach()
            new_bias = torch.zeros_like(self.bias)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(new_bias, -bound, bound)
            self.bias = nn.Parameter(new_bias)

    def forward(self, input):
        output = super(RandConv2d, self).forward(input)

        if self.clamp_output == 'clamp':
            output = torch.max(torch.min(output, self.range_up), self.range_low)
        elif self.clamp_output == 'norm':
            output_low = torch.min(torch.min(output, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
            output_up = torch.max(torch.max(output, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
            output = (output - output_low)/(output_up-output_low)*(self.range_up-self.range_low) + self.range_low

        return output


class MultiScaleRandConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes,
                 rand_bias=True, distribution='kaiming_normal',
                 clamp_output=False, range_up=None, range_low=None, **kwargs
                 ):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size: sequence of kernel size, e.g. (1,3,5)
        :param bias:
        """
        super(MultiScaleRandConv2d, self).__init__()

        self.clamp_output = clamp_output
        self.register_buffer('range_up', None if not self.clamp_output else range_up)
        self.register_buffer('range_low', None if not self.clamp_output else range_low)
        if self.clamp_output:
            assert (self.range_up is not None) and (self.range_low is not None), "No up/low range given for adjust"

        self.multiscale_rand_convs = nn.ModuleDict(
            {str(kernel_size): RandConv2d(in_channels, out_channels, kernel_size, padding = kernel_size // 2,
                                          rand_bias=rand_bias, distribution=distribution,
                                          clamp_output=self.clamp_output,
                                          range_low=self.range_low, range_up=self.range_up,
                                          **kwargs) for kernel_size in kernel_sizes})

        self.scales = kernel_sizes
        self.n_scales = len(kernel_sizes)
        self.randomize()

    def randomize(self):
        self.current_scale = str(self.scales[random.randint(0, self.n_scales-1)])
        self.multiscale_rand_convs[self.current_scale].randomize()

    def forward(self, input):
        output = self.multiscale_rand_convs[self.current_scale](input)
        return output



class data_whiten_layer(nn.Module):
    def __init__(self, data_mean, data_std):
        super(data_whiten_layer, self).__init__()
        self.register_buffer('data_mean', None if data_mean is None else torch.tensor(data_mean).reshape(3, 1, 1))
        self.register_buffer('data_std', None if data_std is None else torch.tensor(data_std).reshape(3, 1, 1))

    def forward(self, input):
        return (input - self.data_mean) / self.data_std
    
    
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import torch.utils.model_zoo as model_zoo

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


class RandConv(nn.Module):
    def __init__(self, output_stride=8, BatchNorm=None, width_mult=1., pretrained=True):
        super(RandConv, self).__init__()
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
        self.features = [RandConvModule(in_channels=3, out_channels=input_channel, kernel_size=[1 ,3])]
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

    def forward(self, x):
        low_level_feat = self.low_level_features(x)
        x = self.high_level_features(low_level_feat)
        return x, low_level_feat

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('http://jeff95.me/models/mobilenet_v2-6a65762b.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

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
    model = RandConv(output_stride=16, BatchNorm=nn.BatchNorm2d)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())

    