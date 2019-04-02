import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
import preprocess

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return Image.open(path)


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader= disparity_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def get_sparse_disp(self, disp_L):
        '''set some non zero disparity to zero'''
        w, h = disp_L.shape
        valid_ind = np.where(disp_L!=0)
        valid_ind = np.array(zip(valid_ind[0], valid_ind[1]))
        # how many points to be set to 0
        sample_num = int(len(valid_ind) * 0.7)
        #print('origin: ', np.sum(disp_L!=0)/float(w*h))
        ind = np.random.choice(range(0, len(valid_ind)), replace=False,size=sample_num)
        sample_ind = valid_ind[ind]
        sparse_disp_L = np.copy(disp_L)
        y_ind = np.squeeze(sample_ind[:,0])
        x_ind = np.squeeze(sample_ind[:,1])
        sparse_disp_L[y_ind,x_ind] = 0
        #print('sparse:', np.sum(sparse_disp_L!=0)/float(w*h))
        sparse_disp_L = np.expand_dims(sparse_disp_L, axis=0)
        return sparse_disp_L

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)

        if self.training:
           w, h = left_img.size
           th, tw = 256, 512

           x1 = random.randint(0, w - tw)
           y1 = random.randint(0, h - th)

           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
           dataL = dataL[y1:y1 + th, x1:x1 + tw]

           processed = preprocess.get_transform(augment=False)
           left_img   = processed(left_img)
           right_img  = processed(right_img)
           return left_img, right_img, dataL, self.get_sparse_disp(dataL)
        else:
           w, h = left_img.size

           left_img = left_img.crop((w-1248, h-352, w, h))
           right_img = right_img.crop((w-1248, h-352, w, h))
           w1, h1 = left_img.size

           dataL = dataL.crop((w-1248, h-352, w, h))
           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256

           processed = preprocess.get_transform(augment=False)
           left_img       = processed(left_img)
           right_img      = processed(right_img)

           return left_img, right_img, dataL, self.get_sparse_disp(dataL)

    def __len__(self):
        return len(self.left)
