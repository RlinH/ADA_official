import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
class Args:
    def __init__(self):
        self.gpu = '0'
        self.resume = None
        self.dataset = 'Domain1'
        self.model = 'Deeplab'
        self.batch_size = 8
        self.group_num = 1
        self.max_epoch = 200
        self.stop_epoch = 200
        self.warmup_epoch = -1
        self.interval_validate = 5
        self.interval_save = 10
        self.lr = 1e-3
        self.lr_decrease_rate = 0.98
        self.lr_decrease_epoch = 1
        self.weight_decay = 0
        self.momentum = 0.99
        # self.data_dir = './Fundus'
        self.out_stride = 16
        self.sync_bn = True
        self.freeze_bn = False
        self.no_augmentation = False  # This is how you handle `store_true` action. Default to False.
        self.method = 'mobilenet'
        self.expid = 'level1'

args = Args()
from datetime import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import os.path as osp

# PyTorch includes
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import yaml
#from train_process import Trainer
from medpy.metric import binary
# Custom includes
from dataloaders import fundus_dataloader
from dataloaders import custom_transforms as trans
from utils.prostae_utils import postprocessing, _connectivity_region_analysis
#from networks.deeplabRSC import *
from networks.deeplabv3 import *

# GIN
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from pdb import set_trace


class GradlessGCReplayNonlinBlock(nn.Module):
    def __init__(self, out_channel = 32, in_channel = 3, scale_pool = [1, 3], layer_id = 0, use_act = True, requires_grad = False, **kwargs):
        """
        Conv-leaky relu layer. Efficient implementation by using group convolutions
        """
        super(GradlessGCReplayNonlinBlock, self).__init__()
        self.in_channel     = in_channel
        self.out_channel    = out_channel
        self.scale_pool     = scale_pool
        self.layer_id       = layer_id
        self.use_act        = use_act
        self.requires_grad  = requires_grad
        assert requires_grad == False

    def forward(self, x_in, requires_grad = False):
        """
        Args:
            x_in: [ nb (original), nc (original), nx, ny ]
        """
        # random size of kernel
        idx_k = torch.randint(high = len(self.scale_pool), size = (1,))
        k = self.scale_pool[idx_k[0]]

        nb, nc, nx, ny = x_in.shape

        ker = torch.randn([self.out_channel * nb, self.in_channel , k, k  ], requires_grad = self.requires_grad  ).cuda()
        shift = torch.randn( [self.out_channel * nb, 1, 1 ], requires_grad = self.requires_grad  ).cuda() * 1.0

        x_in = x_in.view(1, nb * nc, nx, ny)
        x_conv = F.conv2d(x_in, ker, stride =1, padding = k //2, dilation = 1, groups = nb )
        x_conv = x_conv + shift
        if self.use_act:
            x_conv = F.leaky_relu(x_conv)

        x_conv = x_conv.view(nb, self.out_channel, nx, ny)
        return x_conv


class GINGroupConv(nn.Module):
    def __init__(self, out_channel = 3, in_channel = 3, interm_channel = 2, scale_pool = [1, 3 ], n_layer = 4, out_norm = 'frob', **kwargs):
        '''
        GIN
        '''
        super(GINGroupConv, self).__init__()
        self.scale_pool = scale_pool # don't make it tool large as we have multiple layers
        self.n_layer = n_layer
        self.layers = []
        self.out_norm = out_norm
        self.out_channel = out_channel

        self.layers.append(
            GradlessGCReplayNonlinBlock(out_channel = interm_channel, in_channel = in_channel, scale_pool = scale_pool, layer_id = 0).cuda()
                )
        for ii in range(n_layer - 2):
            self.layers.append(
            GradlessGCReplayNonlinBlock(out_channel = interm_channel, in_channel = interm_channel, scale_pool = scale_pool, layer_id = ii + 1).cuda()
                )
        self.layers.append(
            GradlessGCReplayNonlinBlock(out_channel = out_channel, in_channel = interm_channel, scale_pool = scale_pool, layer_id = n_layer - 1, use_act = False).cuda()
                )

        self.layers = nn.ModuleList(self.layers)


    def forward(self, x_in):
        if isinstance(x_in, list):
            x_in = torch.cat(x_in, dim = 0)

        nb, nc, nx, ny = x_in.shape

        alphas = torch.rand(nb)[:, None, None, None] # nb, 1, 1, 1
        alphas = alphas.repeat(1, nc, 1, 1).cuda() # nb, nc, 1, 1

        x = self.layers[0](x_in)
        for blk in self.layers[1:]:
            x = blk(x)
        mixed = alphas * x + (1.0 - alphas) * x_in

        if self.out_norm == 'frob':
            _in_frob = torch.norm(x_in.view(nb, nc, -1), dim = (-1, -2), p = 'fro', keepdim = False)
            _in_frob = _in_frob[:, None, None, None].repeat(1, nc, 1, 1)
            _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim = (-1,-2), p = 'fro', keepdim = False)
            _self_frob = _self_frob[:, None, None, None].repeat(1, self.out_channel, 1, 1)
            mixed = mixed * (1.0 / (_self_frob + 1e-5 ) ) * _in_frob

        return mixed

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch
import random

class ExtendedTransform:
    def __init__(self, angle_range=(-30, 30), horizontal_flip=True, vertical_flip=True, crop_size=(300, 300), resize_size=(384, 384)):
        self.angle_range = angle_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.random_crop = transforms.RandomCrop(crop_size)
        self.random_rotation = transforms.RandomRotation(angle_range)  # Added rotation
        self.resize_size = resize_size  # Size to resize the images and masks to

    def adjust_gamma(self, image, gamma):
        """ Adjust the gamma of an image. """
        return TF.adjust_gamma(image, gamma)

    def __call__(self, sample):
        image, mask = sample['image'], sample['label']

        # Apply random rotation
        angle = random.uniform(*self.angle_range)  # Get a random angle
        image = TF.rotate(image, angle)  # Rotate image
        mask = TF.rotate(mask, angle)  # Rotate mask similarly to keep alignment

        # Random horizontal flip
        if self.horizontal_flip and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flip
        if self.vertical_flip and random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Apply random crop
        if random.random() > 0.5:
            i, j, h, w = self.random_crop.get_params(image, self.random_crop.size)
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

        # Resize both the image and the mask to the specified size
        image = TF.resize(image, self.resize_size, antialias=True)
        mask = TF.resize(mask, self.resize_size, antialias=True)

        return {'image': image, 'label': mask, 'img_name': sample['img_name']}
import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class Prostate(Dataset):
    def __init__(self, domain_indices=None, base_dir=None, split='train', num=None, transform=None, seed=42):
        self.base_dir = base_dir
        self.num = num
        self.domain_name = ['Domain1', 'Domain2', 'Domain3', 'Domain4', 'Domain5', 'Domain6']
        self.domain_indices = domain_indices if domain_indices is not None else [0]
        self.split = split
        self.transform = transform  # Now expects a transform
        self.seed = seed  # seed for random operations to ensure reproducibility

        # Set random seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Load image IDs from all specified domains
        self.id_path = []
        for domain_idx in self.domain_indices:
            full_id_path = os.listdir(os.path.join(self.base_dir, self.domain_name[domain_idx], 'image'))
            if self.num is not None:
                full_id_path = full_id_path[:self.num]
            
            # Shuffle and split data
            random.shuffle(full_id_path)
            split_idx = int(0.9 * len(full_id_path))  # 90% for training
            if self.split == 'train':
                self.id_path.extend([(domain_idx, id) for id in full_id_path[:split_idx]])
            elif self.split == 'test':
                self.id_path.extend([(domain_idx, id) for id in full_id_path[split_idx:]])
            elif self.split == 'full':
                self.id_path.extend([(domain_idx, id) for id in full_id_path])
        
        print("total {} samples for {}".format(len(self.id_path), self.split))
    
    def __len__(self):
        return len(self.id_path)
    
    def __getitem__(self, index):
        domain_idx, id = self.id_path[index]
        img = np.load(os.path.join(self.base_dir, self.domain_name[domain_idx], 'image', id))
        mask = np.load(os.path.join(self.base_dir, self.domain_name[domain_idx], 'mask', id))
        
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()  # img torch.Size([3, 384, 384])
        mask = torch.from_numpy(mask).float()  # gt torch.Size([384, 384])
        mask = mask.unsqueeze(0)
        
        sample = {'image': img, 'label': mask, 'img_name': id}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

# Define a transform if needed
train_transform = None

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

"""
Extended from implementations by Dr. Jo Schlemper
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
import numpy as np
import math


class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.eye(depth).cuda()

    def forward(self, X_in):
        """
        Args:
            added
        """
        n_dim = X_in.dim()
        output_size = X_in.size() + torch.Size([self.depth]) #[nb/nx/nz, ...., nc]
        num_element = X_in.numel()
        X_in = X_in.data.long().view(num_element)
        out = Variable(self.ones.index_select(0, X_in)).view(output_size)
        if n_dim > 1:
            return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float() # [nb/nx/n\, nc, ny]
        else:
            return out.float()

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes

    def forward(self, input, target):
        """
        input: logits : nb x nc x H x W
        target: dense mask
        """
        smooth = 1e-5
        batch_size = input.size(0)

        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)

        inter = torch.sum(input * target, 2)# + smooth # originally there is a smooth on inter which I believe to be a bug!
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score

class SoftDiceScore(nn.Module):
    """ This is not a loss function! don't use this. To train a network, use ``SoftDiceLoss'' above.
    """

    def __init__(self, n_classes, ignore_chan0 = True):
        """
        Args:
            ignore_chan0: ignore the channel 0 of the segmentation (usually for the background)
        """
        super(SoftDiceScore, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.ignore = ignore_chan0

    def forward(self, input, target):
        """ expects input to already be in one_hot encoded data """
        smooth = 1e-5
        batch_size = input.size(0)

        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        input = self.one_hot_encoder(torch.argmax(input, 1))
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)

        if self.ignore == True:
            input = input[:,1:, ...]
            target = target[:,1:, ...]


        inter = torch.sum(input * target, 2)# + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        # preserve class axis
        score = torch.sum(2.0 * inter / union, 0)
        score = score / (float(batch_size))
        print(score.cpu())

        return score

class Efficient_DiceScore(nn.Module):
    """ WARNING: This is not a loss function! don't use this. To train a network, use ``SoftDiceLoss'' above.
        Improving memory efficiency
    """

    def __init__(self, n_classes, ignore_chan0 = True):
        """
        Args:
            ignore_chan0: ignore the channel 0 of the segmentation (usually for the background)
        """
        super(Efficient_DiceScore, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.ignore = ignore_chan0

    def forward(self, global_input, global_target, dense_input = False, cutfold = 5):
        """
        Args:
            input: logits, or dense mask otherwise
                    for logits:  with a shape [nz/nb/1, nc, nx, ny]
                    for dense masks: shape [nz/nb/1, 1, nx, ny]
                    cutfold: split the input volume to <cutfold> folds
            target: dense mask instead, always
        """
        assert global_input.dim() == 4
        smooth = 1e-7
        nz = global_input.size(0)
        foldsize = nz // cutfold + 1 #  actual size

        niter = nz // foldsize

        if nz % foldsize == 0:
            pass
        else:
            niter += 1

        assert niter * foldsize >= nz

        global_inter = 0
        global_nz_pred = 0
        global_nz_gth = 0

        # start the loop
        for ii in range( niter ):
            input = global_input[ii * foldsize : (ii + 1) * foldsize, ...].clone()
            target = global_target[ii * foldsize : (ii + 1) * foldsize, ...].clone()

            if dense_input != True:
                # input is logits
                input = F.softmax(input, dim=1).view(-1, self.n_classes) # nxyz, nc
                input = self.one_hot_encoder(torch.argmax(input, 1)) # nxyz, nc
            else:
                input = self.one_hot_encoder( input.view(-1) )


            target = self.one_hot_encoder( target.view(-1)  ) #nxyz, nc

            if self.ignore == True:
                input = input[:,1:, ...]
                target = target[:,1:, ...]

            try:
                inter = torch.sum(input * target, 0) # + smooth # summing over pixel, keep dimension
                nz_pred = torch.sum(input, 0)
                nz_gth = torch.sum(target, 0)

                flat_inter = [] # place holder
            except:
                # magic numbver, probably due to cuda mememory mechanism
                MAGIC_NUMBER = 14000000
                if input.shape[0] < MAGIC_NUMBER:
                    raise ValueError


                flat_inter = input * target
                total_shape = input.shape[0]
                inter = 0
                nz_pred = 0
                nz_gth = 0

                # iterate through it
                for ii in range(total_shape // MAGIC_NUMBER + 1): # python and pytorch allows going over ...
                    inter += torch.sum(flat_inter[MAGIC_NUMBER * ii: MAGIC_NUMBER * (ii+1) ], 0)
                    nz_pred += torch.sum(input[MAGIC_NUMBER * ii: MAGIC_NUMBER * (ii+1) ], 0)
                    nz_gth += torch.sum(target[MAGIC_NUMBER * ii: MAGIC_NUMBER * (ii+1) ], 0)

            del input
            del target
            del flat_inter

            global_inter += inter
            global_nz_pred += nz_pred
            global_nz_gth  += nz_gth

        global_union = global_nz_pred + global_nz_gth + smooth
        score = 2.0 * global_inter / global_union

        return score

class My_CE(nn.CrossEntropyLoss):
    def __init__(self, nclass, weight, batch_size):
        super(My_CE, self).__init__(weight = weight)
        self.nclass = nclass
        self.one_hot_encoder = One_Hot(nclass)
        self.batch_size = batch_size

    def forward(self, inputs, targets, eps = 0.01):
        #target = targets.contiguous().view(batch_size, -1)
        if not isinstance(targets, torch.LongTensor):
            #targets = targets.LongTensor().cuda()
            targets = torch.squeeze(targets, 1)
            targets = targets.cuda()
        out = super(My_CE, self).forward(inputs, targets)

        return out


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        # Apply sigmoid activation to the input logits
        #input = torch.sigmoid(input)
        
        # Flatten the tensors
        input = input.view(-1)
        target = target.view(-1)
        
        # Compute the intersection and the union
        intersection = (input * target).sum()
        union = input.sum() + target.sum()
        
        # Compute Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Dice Loss is 1 - Dice coefficient
        return 1 - dice
from datetime import datetime
import os
import os.path as osp
import timeit
from torchvision.utils import make_grid
import time

import numpy as np
import pytz
import torch
import torch.nn.functional as F
import torch.nn as nn
import SimpleITK as sitk
from tensorboardX import SummaryWriter

from tqdm import tqdm

import socket
from utils.metrics import *
from utils.Utils import *
weights = torch.tensor([2.0, 1.0])
bceloss = torch.nn.BCELoss()
mseloss = torch.nn.MSELoss()
bcelogitsloss = nn.BCEWithLogitsLoss(pos_weight=weights)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def dice_coeff(predictions, target_map, threshold=0.75):
    # 将预测值二值化，大于等于阈值的设置为1，小于阈值的设置为0
    preds_thresh = (predictions >= threshold).float()
    
    # 确保预测和目标图的形状相同
    assert preds_thresh.shape == target_map.shape, "Shape mismatch between predictions and target map"

    # 计算分子，即两者相交的部分
    intersection = (preds_thresh * target_map).sum(dim=[2, 3])  # 对宽和高的维度进行求和

    # 计算分母，即两者各自的和
    preds_sum = preds_thresh.sum(dim=[2, 3])
    target_sum = target_map.sum(dim=[2, 3])

    # 计算dice系数
    dice = (2. * intersection + 1e-6) / (preds_sum + target_sum + 1e-6)  # 添加一个小常数防止除以零

    # 返回dice系数的平均值，或者根据需要返回每个样本的dice系数
    return dice.mean()

criterionCE = nn.CrossEntropyLoss().cuda()  # Cross Entropy Los
criterionCons = torch.nn.KLDivLoss()
class Trainer(object):

    def __init__(self, cuda, multiply_gpu, model, optimizer, scheduler, val_loader, domain_loader, out, max_epoch, stop_epoch=None,
                 lr=1e-3, interval_validate=10, interval_save=10, batch_size=8, warmup_epoch=10,domain2_loader=None):
        self.cuda = cuda
        self.multiply_gpu = multiply_gpu
        self.warmup_epoch = warmup_epoch
        self.model = model
        self.optim = optimizer
        self.scheduler = scheduler
        self.lr = lr
        # self.lr_decrease_rate = lr_decrease_rate
        # self.lr_decrease_epoch = lr_decrease_epoch
        self.batch_size = batch_size

        self.val_loader = val_loader
        self.domain_loader = domain_loader
        self.time_zone = 'Asia/Shanghai'
        self.timestamp_start = datetime.now(pytz.timezone(self.time_zone))

        self.interval_validate = interval_validate
        self.interval_save = interval_save
        self.domain2_loader = domain2_loader

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss_seg',
            'valid/loss_CE',
            'valid/cup_dice',
            'valid/disc_dice',
            'elapsed_time',
            'best_epoch'
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        log_dir = os.path.join(self.out, 'tensorboard',
                               datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        self.writer = SummaryWriter(log_dir=log_dir)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.stop_epoch = stop_epoch if stop_epoch is not None else max_epoch
        self.best_mean_dice = 0.0
        self.best_epoch = -1
        
        
        # config for gin

        self.datasetTest=2 
        self.img_transform_node = GINGroupConv(out_channel = 3, n_layer = 4,
                                               interm_channel = 2, out_norm = 'frob').cuda()
         # Define loss functions
        self.n_cls = 2
        #self.criterionDice = DiceLoss().cuda() # Dice loss

        # Use plain CE + Dice loss, not using WCE
        #self.criterionDice = DiceLoss().cuda()  # Dice loss
        


    def validate_prostate(self,val_loader=None):
        
        training = self.model.training
        self.model.eval()

        val_loss = 0.0
        total_dice = 0.0
        data_num_cnt = 0

        with torch.no_grad():
            for batch_idx, sample in enumerate(val_loader):
                data = sample['image']
                target_map = sample['label']
                data = data.cuda()
                target_map = target_map.cuda()
                if args.method == "RSC":
                    predictions, _ = self.model(data,None,None)
                elif args.method == "ADA":
                    predictions, _ = self.model(data,None,None)
                else:
                    predictions, _ = self.model(data)

                # 确保 target_map 的形状和 predictions 一致
                #target_map = target_map.unsqueeze(1)

                loss = F.binary_cross_entropy_with_logits(predictions, target_map)
                loss_data = loss.data.item()
                if np.isnan(loss_data):
                    raise ValueError('loss is nan while validating')
                val_loss += loss_data

                # 计算 Dice 系数
                # Assume 'predictions' is a tensor of logits from your model
                pred = torch.sigmoid(predictions)
                pred[pred > 0.75] = 1
                pred[pred <= 0.75] = 0
                dice_score = dice_coeff(pred, target_map)
                total_dice += dice_score.item()  # 累加 Dice 系数
                data_num_cnt += 1  # 计数样本数量

            # 计算平均损失和平均 Dice 系数
            avg_loss = val_loss / data_num_cnt
            avg_dice = total_dice / data_num_cnt

            print(f'Average Validation Loss: {avg_loss:.4f}')
            print(f'Average Dice Score: {avg_dice:.4f}')
            
            if self.epoch % 5 == 0 or self.epoch == 199:
                torch.save({
                        'model_state_dict': self.model.module.state_dict() if self.multiply_gpu else self.model.state_dict(),
                    }, osp.join(self.out, 'checkpoint_%d.pth.tar' % self.epoch))
                # Optionally, switch back to training mode
            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.now(pytz.timezone(self.time_zone)) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [''] + [avg_loss]+ [avg_dice] + [elapsed_time] + [self.best_epoch]
                log = map(str, log)
                f.write(','.join(log) + '\n')
            if training:
                self.model.train()
    def train_abepoch(self):
        self.model.train()
        self.running_seg_loss = 0.0

        start_time = timeit.default_timer()
        for batch_idx, sample in enumerate(self.domain_loader):

            iteration = batch_idx + self.epoch * len(self.domain_loader)
            self.iteration = iteration

            assert self.model.training

            self.optim.zero_grad()

            # train
            for param in self.model.parameters():
                param.requires_grad = True

            image = sample['image'].cuda()
            target_map = sample['label'].cuda()

            
            if args.method == "RSC":
                pred, _ = self.model(image,target_map,self.epoch)
            elif args.method == "ADA":
                pred, _ = self.model(image,target_map,self.epoch)
            else:
                pred, _ = self.model(image)
            
            #pred, _ = self.model(image,target_map,self.epoch)
            #pred, _ = self.model(image)
            pred = torch.sigmoid(pred)
            
            
            # 使用 unsqueeze 添加一个新的通道维度
            #target_map= target_map.unsqueeze(1)  # 在第二个维度上添加，形状变为 [8, 1, 384, 384]
            
            loss_seg = bceloss(pred, target_map)

            self.running_seg_loss += loss_seg.item()
            self.running_seg_loss /= len(self.domain_loader)

            loss_seg_data = loss_seg.data.item()
            if np.isnan(loss_seg_data):
                raise ValueError('loss is nan while training')

            loss_seg.backward()

            self.optim.step()

    
    def train_epoch(self):
        self.model.train()
        self.running_seg_loss = 0.0
        self.running_dice_loss = 0.0
        self.running_wce_loss = 0.0
        self.running_consist_loss = 0.0

        start_time = timeit.default_timer()
        for batch_idx, sample in enumerate(self.domain_loader):
            iteration = batch_idx + self.epoch * len(self.domain_loader)
            self.iteration = iteration

            assert self.model.training

            self.optim.zero_grad()

            # train
            for param in self.model.parameters():
                param.requires_grad = True

            image = sample['image'].cuda()
            target_map = sample['label'].cuda()

            # Apply augmentations
            pred1, _ = self.model(self.img_transform_node(image))
            pred2, _ = self.model(self.img_transform_node(image))
            pred3, _ = self.model(self.img_transform_node(image))
            
#             pred1, _ = self.model((image))
#             pred2, _ = self.model((image))
#             pred3, _ = self.model((image))
            
            
            self._nb_current = image.shape[0]
            pred_all = torch.cat([pred1, pred2, pred3], dim=0)
            pred_all_prob = torch.sigmoid(pred_all)
            pred_avg = 1.0 / 3 * ( pred_all_prob[: self._nb_current] + pred_all_prob[self._nb_current : self._nb_current * 2] + pred_all_prob[self._nb_current * 2: ]) # efficient implementation inspired by Xu et al. (Randconv)
            pred_avg = torch.cat([pred_avg  for ii in range(3)], dim = 0)
            pred_all = F.logsigmoid(pred_all)# according to pytorch 1.3 documentation, input is log_prob, target is prob
            loss_consist = criterionCons(pred_all, pred_avg)
            lambda_consist = 10.0
            
            # Calculate segmentation losses (Dice + WCE)
            pred1 = torch.sigmoid(pred1) 
            pred2 = torch.sigmoid(pred2) 
            pred3 = torch.sigmoid(pred3) 
            # Compute the average prediction
            pred_avg = (pred1 + pred2 + pred3) / 3
            loss_dice = DiceLoss(pred_avg,target_map)
            loss_wce = bceloss(pred_avg, target_map)
            
            # print('loss_dice',loss_dice)
            # print('loss_wce',loss_wce)
            lambda_dice = 1.0
            lambda_wce = 1.0
            lambda_Seg = 1.0

            loss_seg = (loss_dice * lambda_dice + loss_wce * lambda_wce) * lambda_Seg

            self.running_seg_loss += loss_seg.item()
            self.running_dice_loss += loss_dice.item()
            self.running_wce_loss += loss_wce.item()

            loss_consist = lambda_consist * loss_consist
            self.running_consist_loss += loss_consist.item()

            # Combine all losses
            total_loss = loss_seg #+ loss_consist

            total_loss_data = total_loss.data.item()
            if np.isnan(total_loss_data):
                raise ValueError('loss is nan while training')

            total_loss.backward()
            self.optim.step()

        self.running_seg_loss /= len(self.domain_loader)
        self.running_dice_loss /= len(self.domain_loader)
        self.running_wce_loss /= len(self.domain_loader)
        self.running_consist_loss /= len(self.domain_loader)

        print(f"Epoch [{self.epoch}/{self.max_epoch}], "
              f"Segmentation Loss: {self.running_seg_loss:.4f}, "
              f"Dice Loss: {self.running_dice_loss:.4f}, "
              f"WCE Loss: {self.running_wce_loss:.4f}, "
              f"Consistency Loss: {self.running_consist_loss:.4f}")

    
    def test_prostate(self):
        model.eval()
        batch_size = 8
        data_dir = os.path.join('./dataset/prostate')
        domain_list = ['ISBI', 'ISBI_1.5', 'I2CVB', 'UCL', 'BIDMC', 'HK']
        test3d_prostate_dir = domain_list[self.datasetTest]
        file_list = [item for item in os.listdir(os.path.join(data_dir, test3d_prostate_dir)) if 'segmentation' not in item]

        tbar = tqdm(file_list, ncols=150)

        val_dice = 0.0
        total_num = 0
        for file_name in tbar:
            itk_image = sitk.ReadImage(os.path.join(data_dir, test3d_prostate_dir, file_name))
            itk_mask = sitk.ReadImage(os.path.join(data_dir, test3d_prostate_dir, file_name.replace('.nii.gz', '_segmentation.nii.gz')))

            image = sitk.GetArrayFromImage(itk_image)
            mask = sitk.GetArrayFromImage(itk_mask)

            image /= 255.0

            mask[mask==2] = 1
            pred_y = np.zeros(mask.shape)

            #### channel 3 ####
            frame_list = [kk for kk in range(1, image.shape[0] - 1)]

            for ii in range(int(np.floor(image.shape[0] // batch_size))):
                vol = np.zeros([batch_size, 3, image.shape[1], image.shape[2]])

                for idx, jj in enumerate(frame_list[ii * batch_size : (ii + 1) * batch_size]):
                    vol[idx, ...] = image[jj - 1 : jj + 2, ...].copy()
                vol = torch.from_numpy(vol).float().cuda()
                pred = sigmoid(model(vol)[0])
                pred_student = sigmoid(model(vol)[0]).detach().data.cpu().numpy()

                for idx, jj in enumerate(frame_list[ii * batch_size : (ii + 1) * batch_size]):
                    ###### Ignore slices without prostate region ######
                    if np.sum(mask[jj, ...]) == 0:
                        continue
                    pred_y[jj, ...] = pred_student[idx, ...].copy()


            processed_pred_y = _connectivity_region_analysis(pred_y)
            dice_coeff = binary.dc(np.asarray(processed_pred_y, dtype=bool),
                                np.asarray(mask, dtype=bool))
            val_dice += dice_coeff
            total_num += 1

        val_dice /= total_num
        print('val_dice : {}'.format(val_dice))
        return val_dice 

    def train(self):
        
        for epoch in tqdm(range(self.max_epoch), desc='Training Progress'):
            self.epoch = epoch
            self.train_epoch()
            if self.stop_epoch == self.epoch:
                print('Stop epoch at %d' % self.stop_epoch)
                break

            self.scheduler.step()
            self.writer.add_scalar('lr', get_lr(self.optim), self.epoch * (len(self.domain_loader)))
            
            if (self.epoch) > 50:
                #self.test_prostate()
                self.validate_prostate(self.val_loader)
                # self.validate_prostate(self.domain2_loader)

                
        self.writer.close()


import torch
import torch.nn.functional as F
import numpy as np
from utils.transform import collate_fn_tr_styleaug, collate_fn_ts
class Projector(nn.Module):
    def __init__(self, output_size=1024):
        super(Projector, self).__init__()
        self.conv = nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = None
        self.output_size = output_size

    def forward(self, x_in):
        x = self.conv(x_in)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        if self.fc is None:
            self.fc = nn.Linear(x.size(1), self.output_size).to(x.device)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x

def dice_coeff(pred, target):
    smooth = 1e-5
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def iou_coeff(pred, target):
    smooth = 1e-5
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def validate_prostate(val_loader=None):
    model.eval()

    val_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    data_num_cnt = 0

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            
            data = sample['image'].cuda().to(dtype=torch.float32)

            target_map = sample['label'].cuda().to(dtype=torch.float32)

            if args.method == "RSC":
                predictions, _ = model(data,None,None)
            elif args.method == "ADA":
                predictions, _ = model(data,None,None)
            else:
                predictions, _ = model(data)

            loss = F.binary_cross_entropy_with_logits(predictions, target_map)
            loss_data = loss.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')
            val_loss += loss_data

            pred = torch.sigmoid(predictions)
            pred[pred > 0.75] = 1
            pred[pred <= 0.75] = 0

            dice_score = dice_coeff(pred, target_map)
            total_dice += dice_score.item()

            iou_score = iou_coeff(pred, target_map)
            total_iou += iou_score.item()

            data_num_cnt += 1

        avg_loss = val_loss / data_num_cnt
        avg_dice = total_dice / data_num_cnt
        avg_iou = total_iou / data_num_cnt

        #print(f'Average Validation Loss: {avg_loss:.4f}')
        print(f'Dice: {avg_dice:.4f}')
        print(f'IoU: {avg_iou:.4f}')
        
if __name__ == '__main__':
    import os.path as osp
    from datetime import datetime

    # 获取当前时间
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    #'Domain1','Domain2','Domain3','Domain4','Domain5','Domain6'
    now = datetime.now()
    for x in ['Domain4']:
        #args.method = 
        args.dataset = x
        args.out = osp.join('logs_train','Visualization','Prostate', args.dataset, "CISDG",current_time)

        os.makedirs(args.out)
        with open(osp.join(args.out, 'config.yaml'), 'w') as f:
            yaml.safe_dump(args.__dict__, f, default_flow_style=False)

        multiply_gpu = False
        if (args.gpu).find(',') != -1:
            multiply_gpu = True
        cuda = torch.cuda.is_available()

        torch.manual_seed(42)
        if cuda:
            torch.cuda.manual_seed(42)

        # 1. dataset

        #train_transform = None
        train_transform = ExtendedTransform()
        test_transform = None

        # Example of creating a DataLoader with multiple domains
        base_dir = './dataset/prostate'

        single_dataset =int(args.dataset[-1])-1

        trainset = Prostate(domain_indices=[single_dataset], base_dir=base_dir, split='full', transform=train_transform)

        trainloader = DataLoader(trainset, batch_size=8, num_workers=10,
                             shuffle=True, drop_last=True, pin_memory=True, worker_init_fn=seed_worker)

        testset = Prostate(domain_indices=[1],base_dir='./dataset/prostate', split='full', transform=test_transform)

        testloader = DataLoader(testset, batch_size=8, num_workers=0,
                             shuffle=True, drop_last=True, pin_memory=True, worker_init_fn=seed_worker)

        testset2 = Prostate(domain_indices=[1],base_dir='./dataset/prostate', split='full', transform=test_transform)

        testloader2 = DataLoader(testset2, batch_size=8, num_workers=0,
                             shuffle=True, drop_last=True, pin_memory=True, worker_init_fn=seed_worker)
        if args.method=='RSC' or 'SLAUG':
            method = 'mobilenet'
        else:
            method = args.method
        # 2. modelFFNET
        if method == 'CCSDG':
            model = DeepLab(num_classes=1, backbone=method, output_stride=args.out_stride,
                                        sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
        else:
            model = DeepLab(num_classes=1, backbone=method, output_stride=args.out_stride,
                                        sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)

        # model = DeepLab(num_classes=1, backbone='mixstyle', output_stride=args.out_stride,
        #                             sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
        if cuda:
            model = model.cuda()

        start_epoch = 0
        start_iteration = 0

        optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        #optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99, nesterov=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.lr_decrease_epoch, gamma=args.lr_decrease_rate)


        trainer = Trainer(
            cuda=cuda,
            multiply_gpu=multiply_gpu,
            model=model,
            optimizer=optim,
            scheduler=scheduler,
            lr=args.lr,
            val_loader=testloader,
            domain_loader=trainloader,
            out=args.out,
            max_epoch=args.max_epoch,
            stop_epoch=args.stop_epoch,
            interval_validate=args.interval_validate,
            interval_save=args.interval_save,
            batch_size=args.batch_size,
            warmup_epoch=args.warmup_epoch,
            domain2_loader = testloader2
        )
        trainer.epoch = start_epoch
        trainer.iteration = start_iteration
        trainer.train()

        testset = Prostate(domain_indices=[0],base_dir='./dataset/prostate', split='full', transform=test_transform)

        testloader = DataLoader(testset, batch_size=8, num_workers=0,
                             shuffle=True, drop_last=True, pin_memory=True, worker_init_fn=seed_worker)


        testset2 = Prostate(domain_indices=[1],base_dir='./dataset/prostate', split='full', transform=test_transform)

        testloader2 = DataLoader(testset2, batch_size=8, num_workers=0,
                             shuffle=True, drop_last=True, pin_memory=True, worker_init_fn=seed_worker)

        testset3 = Prostate(domain_indices=[2], base_dir='./dataset/prostate', split='full', transform=test_transform)

        testloader3 = DataLoader(testset3, batch_size=8, num_workers=0,
                             shuffle=True, drop_last=True, pin_memory=True, worker_init_fn=seed_worker)

        testset4 = Prostate(domain_indices=[3],base_dir='./dataset/prostate', split='full', transform=test_transform)

        testloader4 = DataLoader(testset4, batch_size=8, num_workers=0,
                             shuffle=True, drop_last=True, pin_memory=True, worker_init_fn=seed_worker)

        testset5 = Prostate(domain_indices=[4],base_dir='./dataset/prostate', split='full', transform=test_transform)

        testloader5 = DataLoader(testset5, batch_size=8, num_workers=0,
                             shuffle=True, drop_last=True, pin_memory=True, worker_init_fn=seed_worker)

        testset6 = Prostate(domain_indices=[5],base_dir='./dataset/prostate', split='full', transform=test_transform)

        testloader6 = DataLoader(testset6, batch_size=8, num_workers=0,
                             shuffle=True, drop_last=True, pin_memory=True, worker_init_fn=seed_worker)
        print(f'-----Dataset: {args.dataset}--------')
        validate_prostate(val_loader=testloader)
        validate_prostate(val_loader=testloader2)
        validate_prostate(val_loader=testloader3)
        validate_prostate(val_loader=testloader4)
        validate_prostate(val_loader=testloader5)
        validate_prostate(val_loader=testloader6)
        print(f'-----Dataset: {args.dataset}--------')