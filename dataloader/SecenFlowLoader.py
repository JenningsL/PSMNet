import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import preprocess
import listflowfile as lt
import readpfm as rp
import numpy as np
from data_util import get_sparse_disp

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return rp.readPFM(path)


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader= disparity_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]


        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL, scaleL = self.dploader(disp_L)



        if self.training:
           w, h = left_img.size
           th, tw = 256, 512

           x1 = random.randint(0, w - tw)
           y1 = random.randint(0, h - th)

           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

           dataL = np.ascontiguousarray(dataL,dtype=np.float32)
           dataL = dataL[y1:y1 + th, x1:x1 + tw]

           processed = preprocess.get_transform(augment=False)
           left_img   = processed(left_img)
           right_img  = processed(right_img)

           return left_img, right_img, dataL, get_sparse_disp(dataL, erase_ratio=0.8)
        else:
           w, h = left_img.size
           left_img = left_img.crop((w-960, h-544, w, h))
           right_img = right_img.crop((w-960, h-544, w, h))
           processed = preprocess.get_transform(augment=False)
           left_img       = processed(left_img)
           right_img      = processed(right_img)

           dataL = np.pad(dataL, ((max(544-h, 0), 0), (max(960-w, 0), 0)), 'constant', constant_values=0)
           dataL = dataL[max(dataL.shape[0]-544, 0):dataL.shape[0], max(dataL.shape[1]-960, 0):dataL.shape[1]]

           return left_img, right_img, dataL, get_sparse_disp(dataL, erase_ratio=0.8)

    def __len__(self):
        return len(self.left)
