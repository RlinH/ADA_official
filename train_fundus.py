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
        self.data_dir = './dataset/Fundus'
        self.out_stride = 16
        self.sync_bn = True
        self.freeze_bn = False
        self.no_augmentation = False  # This is how you handle `store_true` action. Default to False.
        self.method = 'ADA'
args = Args()
from datetime import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import os.path as osp
isRSC = True
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
from networks.deeplabRSC import *
#from networks.deeplabv3 import *
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'fundus':
            return '../../../../data/disc_cup_split/'  # foler that contains leftImg8bit/
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

from glob import glob
import random
class FundusSegmentation(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir=Path.db_root_dir('fundus'),
                 dataset='refuge',
                 split='train',
                 testid=None,
                 transform=None,
                 image_list=None
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        if image_list==None:
            self._base_dir = base_dir
            self.image_list = []
            self.split = split

            self.image_pool = []
            self.label_pool = []
            self.img_name_pool = []

            self._image_dir = os.path.join(self._base_dir, dataset, split, 'image')
            print(self._image_dir)
            imagelist = glob(self._image_dir + "/*.png")
            for image_path in imagelist:
                gt_path = image_path.replace('image', 'mask')
                self.image_list.append({'image': image_path, 'label': gt_path, 'id': testid})
            ###
            '''
            if split=='train/ROIs':
                self._image_dir = os.path.join(self._base_dir, 'Domain3', split, 'image')
                print(self._image_dir)
                imagelist = glob(self._image_dir + "/*.png")
                for image_path in imagelist:
                    gt_path = image_path.replace('image', 'mask')
                    self.image_list.append({'image': image_path, 'label': gt_path, 'id': testid})
            '''
            ###
            #self.image_list = self.image_list[:90]
            self.transform = transform
            # self._read_img_into_memory()
            # Display stats
            print('Number of images in {}: {:d}'.format(split, len(self.image_list)))
        else:
            self.image_list = image_list
            self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        _img = Image.open(self.image_list[index]['image']).convert('RGB')
        _target = Image.open(self.image_list[index]['label'])
        if _target.mode == 'RGB':
            _target = _target.convert('L')
        _img_name = self.image_list[index]['image'].split('/')[-1]

        # _img = self.image_pool[index]
        # _target = self.label_pool[index]
        # _img_name = self.img_name_pool[index]
        sample = {'image': _img, 'label': _target, 'img_name': _img_name}

        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample['image'],sample['label'],sample['img_name']

    def _read_img_into_memory(self):

        img_num = len(self.image_list)
        for index in range(img_num):
            self.image_pool.append(Image.open(self.image_list[index]['image']).convert('RGB'))
            _target = Image.open(self.image_list[index]['label'])
            if _target.mode == 'RGB':
                _target = _target.convert('L')
            self.label_pool.append(_target)
            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool.append(_img_name)


    def __str__(self):
        return 'Fundus(split=' + str(self.split) + ')'
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch
import random

class ExtendedTransform:
    def __init__(self, angle_range=(-30, 30), horizontal_flip=True, vertical_flip=True, crop_size=(300, 300), resize_size=(352, 352)):
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
# Define a transform if needed
train_transform = None

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
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
from torch.cuda.amp import autocast, GradScaler
from tqdm.notebook import tqdm
from networks.unet_ccsdg import UNetCCSDG, Projector
import socket
from utils.metrics import *
from utils.Utils import *
weights = torch.tensor([2.0, 1.0])
bceloss = torch.nn.BCELoss()
mseloss = torch.nn.MSELoss()
bcelogitsloss = nn.BCEWithLogitsLoss(pos_weight=weights)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import math

def cosine_annealing_lr(epoch, max_epochs, initial_lr):
    return initial_lr * 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))

def adjust_learning_rate_cosine(optimizer, epoch, initial_lr, max_epochs):
    lr = cosine_annealing_lr(epoch, max_epochs, initial_lr)
    for i in range(len(optimizer.param_groups)):
        optimizer.param_groups[i]['lr'] = lr
    return lr


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent


def adjust_learning_rate(optimizer, epoch, initial_lr, max_epochs, exponent=0.9):
    lr = poly_lr(epoch, max_epochs, initial_lr, exponent)
    for i in range(len(optimizer.param_groups)):
        optimizer.param_groups[i]['lr'] = lr
    return lr

criterion = nn.BCEWithLogitsLoss()
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
        
        self.projector = Projector()
        self.projector = self.projector.cuda()
        
        self.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, nesterov=True)
        # self.optimizer_prompt = torch.optim.SGD(list(self.projector.parameters()) + [self.model.backbone.channel_prompt],
        #                             lr=1e-2, momentum=0.99, nesterov=True) 
        #self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        #self.optimizer_prompt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
            
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

        log_dir = os.path.join('../tf-logs/', 'tensorboard',args.EXPID)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.stop_epoch = stop_epoch if stop_epoch is not None else max_epoch
        self.best_mean_dice = 0.0
        self.best_epoch = -1
        
        self.datasetTest=2 


    def validate_prostate(self,val_loader=None,dataset_name= None):
        
        training = self.model.training
        self.model.eval()

        val_loss = 0.0
        val_fundus_dice = {'cup': 0.0, 'disc': 0.0}
        data_num_cnt = 0.0
        metrics = []

        with torch.no_grad():
            for batch_idx, sample in enumerate(val_loader):
                data = torch.from_numpy(sample['data']).cuda().to(dtype=torch.float32)
                target_map = torch.from_numpy(sample['seg']).cuda().to(dtype=torch.float32)
                if args.method == "RSC":
                    predictions, _ = self.model(data,None,None)
                elif args.method == "ADA":
                    predictions, _ = self.model(data,None,None)
                else:
                    predictions,_ = self.model(data,None,None)

                
                # 确保 target_map 的形状和 predictions 一致
                #target_map = target_map.unsqueeze(1)
                loss = F.binary_cross_entropy_with_logits(predictions, target_map)
                loss_data = loss.data.item()
                if np.isnan(loss_data):
                    raise ValueError('loss is nan while validating')
                val_loss += loss_data

                dice_cup, dice_disc = dice_coeff_2label(predictions, target_map)
                val_fundus_dice['cup'] += np.sum(dice_cup)
                val_fundus_dice['disc'] += np.sum(dice_disc)
                data_num_cnt += float(dice_cup.shape[0])

            val_loss /= data_num_cnt
            val_fundus_dice['cup'] /= data_num_cnt
            val_fundus_dice['disc'] /= data_num_cnt
            metrics.append((val_loss, val_fundus_dice['cup'], val_fundus_dice['disc']))

            # self.writer.add_scalar(f'val_data/loss_CE/{dataset_name}', val_loss, self.epoch * (len(self.domain_loader)))
            # self.writer.add_scalar(f'val_data/val_CUP_dice/{dataset_name}', val_fundus_dice['cup'], self.epoch * (len(self.domain_loader)))
            # self.writer.add_scalar(f'val_data/val_DISC_dice/{dataset_name}', val_fundus_dice['disc'], self.epoch * (len(self.domain_loader)))
            self.writer.add_scalar(f'val_data/loss_CE/{dataset_name}', val_loss, self.epoch )
            self.writer.add_scalar(f'val_data/val_CUP_dice/{dataset_name}', val_fundus_dice['cup'], self.epoch )
            self.writer.add_scalar(f'val_data/val_DISC_dice/{dataset_name}', val_fundus_dice['disc'], self.epoch )

            mean_dice = (val_fundus_dice['cup'] + val_fundus_dice['disc'])/2
            is_best = mean_dice > self.best_mean_dice
            if is_best:
                self.best_epoch = self.epoch + 1
                self.best_mean_dice = mean_dice

                torch.save({
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'arch': self.model.__class__.__name__,
                    'optim_state_dict': self.optim.state_dict(),
                    'model_state_dict': self.model.module.state_dict() if self.multiply_gpu else self.model.state_dict(),
                    'learning_rate_gen': get_lr(self.optim),
                    'best_mean_dice': self.best_mean_dice,
                }, osp.join(self.out, 'checkpoint_%d.pth.tar' % self.best_epoch))
            else:
                if self.epoch > 195:
                    torch.save({
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'arch': self.model.__class__.__name__,
                        'optim_state_dict': self.optim.state_dict(),
                        'model_state_dict': self.model.module.state_dict() if self.multiply_gpu else self.model.state_dict(),
                        'learning_rate_gen': get_lr(self.optim),
                        'best_mean_dice': self.best_mean_dice,
                    }, osp.join(self.out, 'checkpoint_%d.pth.tar' % (self.epoch + 1)))

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.now(pytz.timezone(self.time_zone)) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [''] + list(metrics) + [elapsed_time] + [self.best_epoch]
                log = map(str, log)
                f.write(','.join(log) + '\n')
            self.writer.add_scalar('best_model_epoch', self.best_epoch, self.epoch * (len(self.domain_loader)))
            if training:
                self.model.train()
                


    def train_FULLtrain(self):
        self.model.train()
        self.running_seg_loss = 0.0
        criterion = nn.BCEWithLogitsLoss()
        start_time = timeit.default_timer()

        for batch_idx, batch in enumerate(self.domain_loader):
            iteration = batch_idx + self.epoch * len(self.domain_loader)
            self.iteration = iteration

            assert self.model.training

            # self.optim.zero_grad()

            # Enable training for model parameters
            for param in self.model.parameters():
                param.requires_grad = True

            data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
            fda_data = torch.from_numpy(batch['fda_data']).cuda().to(dtype=torch.float32)
            GLA_data = torch.from_numpy(batch['GLA']).cuda().to(dtype=torch.float32)
            seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)

            tau = 0.1

            # Training with fda_data
            self.optimizer.zero_grad()
            if isRSC == True:
                output, _ = self.model(fda_data, seg, self.epoch)
            else:
                output, _ = self.model(fda_data)
            loss = criterion(output, seg)
            loss.backward()
            self.optimizer.step()

            # Training with GLA_data
            self.optimizer.zero_grad()
            if isRSC == True:
                output, _ = self.model(GLA_data, seg, self.epoch)
            else:
                output, _ = self.model(GLA_data)
            loss = criterion(output, seg)
            loss.backward()
            self.optimizer.step()

            # Training with data
            self.optimizer.zero_grad()
            if isRSC == True:
                output, _ = self.model(data, seg, self.epoch)
            else:
                output, _ = self.model(data)
            loss = criterion(output, seg)
            loss.backward()
            self.optimizer.step()
            
    def train(self):

        for epoch in tqdm(range(self.max_epoch), desc='Training Progress'):
            #lr = adjust_learning_rate(self.optimizer, self.epoch, initial_lr=0.01, max_epochs=200)
            
            lr = adjust_learning_rate_cosine(self.optimizer, self.epoch, initial_lr=0.001, max_epochs=200)
            self.epoch = epoch

            self.train_FULLtrain()
            if self.stop_epoch == self.epoch:
                print('Stop epoch at %d' % self.stop_epoch)
                break

            
            self.validate_prostate(self.val_loader,'Domain2')
            self.validate_prostate(self.domain2_loader,'Domain4')
            self.writer.add_scalar('lr', get_lr(self.optimizer), self.epoch * (len(self.domain_loader)))
            
            self.writer.close()
        

args.EXPID = "2080-0812-repeat1"
import numpy as np
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from utils.fourier import FDA_source_to_target_np
from utils.normalize import normalize_image
from utils.slaug import LocationScaleAugmentation


def fourier_augmentation_reverse(data, fda_beta=0.1):
    this_fda_beta = round(0.05+np.random.random() * fda_beta, 2)
    lowf_batch = data[::-1]
    fda_data = FDA_source_to_target_np(data, lowf_batch, L=this_fda_beta)
    return fda_data


def sl_augmentation(image, mask):
    location_scale = LocationScaleAugmentation(vrange=(0., 255.), background_threshold=0.01)
    GLA = location_scale.Global_Location_Scale_Augmentation(image.copy())
    LLA = location_scale.Local_Location_Scale_Augmentation(image.copy(), mask.copy().astype(np.int32))
    return GLA, LLA


def get_train_transform(patch_size=(512, 512)):
    tr_transforms = []
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))
    #0723v2
    tr_transforms.append(GammaTransform(gamma_range=(0.8, 1.2), invert_image=False, per_channel=True, p_per_sample=0.15))
    # tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))
    #0723v2
    tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.8, 1.2), different_sigma_per_channel=True,
                                               p_per_channel=0.5, p_per_sample=0.15))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

def get_structure_destroyed_transform(patch_size=(512, 512)):
    tr_transforms = []
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def collate_fn_tr(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'seg': label, 'name': name}
    tr_transforms = get_train_transform(patch_size=image.shape[-2:])
    data_dict = tr_transforms(**data_dict)
    return data_dict


def collate_fn_ts(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'seg': label, 'name': name}
    data_dict['data'] = normalize_image(data_dict['data'])
    
    return data_dict


def collate_fn_tr_styleaug(batch):
    
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'seg': label, 'name': name}
    # print(f"Data shape: {data_dict['data'].shape}, Seg shape: {data_dict['seg'].shape}")
    tr_transforms = get_train_transform(patch_size=image.shape[-2:])
    data_dict = tr_transforms(**data_dict)
    fda_data = fourier_augmentation_reverse(data_dict['data'])
    #mask = data_dict['seg'][:,0,:,:]
    mask = data_dict['seg'][:,1,:,:]
    mask =  np.expand_dims(mask, axis=1) 
    GLA, LLA = sl_augmentation(data_dict['data'], mask)
    
    data_dict['fda_data'] = normalize_image(fda_data)
    data_dict['data'] = normalize_image(data_dict['data'])
    data_dict['GLA'] = normalize_image(GLA)
    data_dict['LLA'] = normalize_image(LLA)
    
    return data_dict

import torch
import torch.nn.functional as F
import numpy as np
# from utils.transform import collate_fn_tr_styleaug, collate_fn_ts
from utils.polypDataset import PolypDataset
def calculate_iou(pred, target):
    target = target.data.cpu()
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu()
    pred[pred > 0.75] = 1
    pred[pred <= 0.75] = 0
    return iou_coefficient_numpy(pred[:, 0, ...], target[:, 0, ...]), iou_coefficient_numpy(pred[:, 1, ...], target[:, 1, ...])

def iou_coefficient_numpy(binary_segmentation, binary_gt_label):

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation, dtype=bool)
    binary_gt_label = np.asarray(binary_gt_label, dtype=bool)

    # compute the intersection
    intersection = np.logical_and(binary_segmentation, binary_gt_label)

    # compute the union
    union = np.logical_or(binary_segmentation, binary_gt_label)

    # count the number of True pixels in the intersection and union
    intersection = np.sum(intersection.astype(float), axis=(1, 2))
    union = np.sum(union.astype(float), axis=(1, 2))

    # compute the IoU coefficient
    iou_value = intersection / (union + 1e-6)  # Add a small value to avoid division by zero

    # return it
    return iou_value
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
    val_fundus_dice = {'cup': 0.0, 'disc': 0.0}
    val_fundus_iou = {'cup': 0.0, 'disc': 0.0}
    data_num_cnt = 0.0
    metrics = []

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            
            data = torch.from_numpy(sample['data']).cuda().to(dtype=torch.float32)

            target_map = torch.from_numpy(sample['seg']).cuda().to(dtype=torch.float32)

            if args.method == "RSC":
                predictions, _ = model(data,None,None)
            elif args.method == "ADA":
                predictions, _ = model(data,None,None)
            else:
                predictions, _ = model(data,None,None)

            dice_cup, dice_disc = dice_coeff_2label(predictions, target_map)
            val_fundus_dice['cup'] += np.sum(dice_cup)
            val_fundus_dice['disc'] += np.sum(dice_disc)

            iou_cup, iou_disc = calculate_iou(predictions, target_map)
            val_fundus_iou['cup'] += np.sum(iou_cup)
            val_fundus_iou['disc'] += np.sum(iou_disc)

            data_num_cnt += float(dice_cup.shape[0])

        val_loss /= data_num_cnt
        val_fundus_dice['cup'] /= data_num_cnt
        val_fundus_dice['disc'] /= data_num_cnt
        val_fundus_iou['cup'] /= data_num_cnt
        val_fundus_iou['disc'] /= data_num_cnt
        metrics.append((val_loss, val_fundus_dice['cup'], val_fundus_dice['disc'], val_fundus_iou['cup'], val_fundus_iou['disc']))
        
        mean_dice = (val_fundus_dice['cup'] + val_fundus_dice['disc']) / 2
        mean_iou = (val_fundus_iou['cup'] + val_fundus_iou['disc']) / 2
        #print('val_data/loss_CE', val_loss)
        print(f'CUP_dice: {val_fundus_dice["cup"]:.4f}')
        print(f'DISC_dice: {val_fundus_dice["disc"]:.4f}')
        print(f'MEAN_dice: {mean_dice:.4f}')
        print(f'CUP_iou: {val_fundus_iou["cup"]:.4f}')
        print(f'DISC_iou: {val_fundus_iou["disc"]:.4f}')
        print(f'MEAN_iou: {mean_iou:.4f}')

if __name__ == '__main__':
    import os.path as osp
    from datetime import datetime

    # 获取当前时间
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    now = datetime.now()
    for x in ['Domain3']:#
        args.dataset = x
        args.out = osp.join('logs_train','Fundus', args.dataset, args.method,current_time)

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
        composed_transforms_tr = transforms.Compose([
        trans.RandomScaleCrop(512),
        trans.RandomRotate(),
        trans.RandomFlip(),
        trans.elastic_transform(),
        trans.add_salt_pepper_noise(),
        trans.adjust_light(),
        trans.eraser(),
        trans.Normalize_tf(),
        trans.ToTensor()
    ])

        composed_transforms_ts = transforms.Compose([
            # fundus_trans.RandomCrop(512),
            trans.Resize(512),
            trans.Normalize_tf(),  # this function separates (raw ground truth mask) into (2 masks)
            trans.ToTensor()
        ])

        if args.no_augmentation:
            domain = FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset,
                                                          split='train/ROIs', transform=composed_transforms_ts)
        else:
            domain = FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset,
                                                          split='train/ROIs', transform=composed_transforms_tr)
        domain_loader = DataLoader(domain, batch_size=args.batch_size, shuffle=True, num_workers=10, pin_memory=True ,collate_fn=collate_fn_tr_styleaug)

        domain_val = FundusSegmentation(base_dir=args.data_dir, dataset='Domain2',
                                                          split='test/ROIs', transform=composed_transforms_ts)
        domain_loader_val = DataLoader(domain_val, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                       pin_memory=True, collate_fn=collate_fn_ts)

        domain2_val = FundusSegmentation(base_dir=args.data_dir, dataset='Domain4',
                                                          split='test/ROIs', transform=composed_transforms_ts)
        domain2_loader_val = DataLoader(domain2_val, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                       pin_memory=True, collate_fn=collate_fn_ts)


        method = args.method
        model = DeepLab(num_classes=2, backbone=method, output_stride=args.out_stride,
                                    sync_bn=args.sync_bn, freeze_bn=args.freeze_bn, dataset= 'Fundus')

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
            val_loader=domain_loader_val,
            domain_loader=domain_loader,
            out=args.out,
            max_epoch=args.max_epoch,
            stop_epoch=args.stop_epoch,
            interval_validate=args.interval_validate,
            interval_save=args.interval_save,
            batch_size=args.batch_size,
            warmup_epoch=args.warmup_epoch,
            domain2_loader = domain2_loader_val
        )
        trainer.epoch = start_epoch
        trainer.iteration = start_iteration
        trainer.train()


        domain_val = FundusSegmentation(base_dir=args.data_dir, dataset='Domain1',
                                                          split='test/ROIs', transform=composed_transforms_ts)
        testloader = DataLoader(domain_val, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                       pin_memory=True, collate_fn=collate_fn_ts)

        domain2_val = FundusSegmentation(base_dir=args.data_dir, dataset='Domain2',
                                                          split='test/ROIs', transform=composed_transforms_ts)
        testloader2 = DataLoader(domain2_val, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                       pin_memory=True, collate_fn=collate_fn_ts)

        domain3_val = FundusSegmentation(base_dir=args.data_dir, dataset='Domain3',
                                                          split='test/ROIs', transform=composed_transforms_ts)
        testloader3 = DataLoader(domain3_val, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                       pin_memory=True, collate_fn=collate_fn_ts)


        domain4_val = FundusSegmentation(base_dir=args.data_dir, dataset='Domain4',
                                                          split='test/ROIs', transform=composed_transforms_ts)
        testloader4 = DataLoader(domain4_val, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                       pin_memory=True, collate_fn=collate_fn_ts)


        output_file='validation_results_Fundus.txt'
        with open(output_file, 'a') as f:
            f.write(f'-----Dataset: {args.dataset}--------')

        print(f'-----Dataset: {args.dataset}--------')
        validate_prostate(val_loader=testloader)
        validate_prostate(val_loader=testloader2)
        validate_prostate(val_loader=testloader3)
        validate_prostate(val_loader=testloader4)
        print(f'-----Dataset: {args.dataset}--------')