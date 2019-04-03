import os
import sys
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import cv2
import numpy as np
import preprocess
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'kitti'))
import kitti_util
from kitti_object import *

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

class KITTIObjectLoader(data.Dataset):
    def __init__(self, kitti_path, split):
        self.kitti_path = kitti_path
        self.kitti_dataset = kitti_object(kitti_path, 'training')
        self.left_dir = os.path.join(self.kitti_path, 'training', 'image_2')
        self.right_dir = os.path.join(self.kitti_path, 'training', 'image_3')
        self.disp_dir = os.path.join(self.kitti_path, 'training/disparity')
        self.frame_ids = self.load_split_ids(split)
        self.baseline = 0.5379

    def load_split_ids(self, split):
        with open(os.path.join(self.kitti_path, split + '.txt')) as f:
            return [line.rstrip('\n') for line in f]

    def generate_sparse_disparity(self):
        if not os.path.isdir(self.disp_dir):
            os.mkdir(self.disp_dir)
        for frame_id in self.frame_ids:
            data_idx = int(frame_id)
            pc_velo = self.kitti_dataset.get_lidar(data_idx)
            calib = self.kitti_dataset.get_calibration(data_idx) # 3 by 4 matrix
            image = self.kitti_dataset.get_image(data_idx)
            img_height, img_width = image.shape[0:2]
            _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:,0:3],
                calib, 0, 0, img_width, img_height, True)
            pc_velo = pc_velo[img_fov_inds, :]
            pc_rect = calib.project_velo_to_rect(pc_velo[:,0:3])
            disp_L = np.zeros((img_height, img_width), dtype=np.float32)
            p2d = calib.project_rect_to_image(pc_rect)
            for i in range(len(p2d)):
                u = int(round(p2d[i][0]))
                v = int(round(p2d[i][1]))
                if u < 0 or u >= img_width or v < 0 or v >= img_height:
                    continue
                disp_L[v][u] = calib.P[0][0] * self.baseline / pc_rect[i][2]
            fname = os.path.join(self.disp_dir, '{:06}.npy'.format(data_idx))
            np.save(fname, disp_L)
            # fname = os.path.join(self.disp_dir, '{:06}.png'.format(data_idx))
            # cv2.imwrite(fname, disp_L.astype(np.int32))

    def __getitem__(self, index):
        left = os.path.join(self.left_dir, '{0}.png'.format(self.frame_ids[index]))
        right = os.path.join(self.right_dir, '{0}.png'.format(self.frame_ids[index]))
        # disp_L = os.path.join(self.disp_dir, '{0}.png'.format(self.frame_ids[index]))
        disp_L = os.path.join(self.disp_dir, '{0}.npy'.format(self.frame_ids[index]))
        left_img = Image.open(left).convert('RGB')
        right_img = Image.open(right).convert('RGB')
        # dataL = Image.open(disp_L)
        dataL = np.load(disp_L)

        target_w = 1248
        target_h = 352
        w, h = left_img.size

        # this will add zero padding
        left_img = left_img.crop((w-target_w, h-target_h, w, h))
        right_img = right_img.crop((w-target_w, h-target_h, w, h))
        w1, h1 = left_img.size

        # load sparse disparity as float, need to implement crop on numpy
        # print('origin size: ', dataL.shape)
        dataL = np.pad(dataL, ((max(target_h-h, 0), 0), (max(target_w-w, 0), 0)), 'constant', constant_values=0)
        # print('after padding: ', dataL.shape)
        dataL = dataL[max(dataL.shape[0]-target_h, 0):dataL.shape[0], max(dataL.shape[1]-target_w, 0):dataL.shape[1]]
        # print('after crop: ', dataL.shape)

        processed = preprocess.get_transform(augment=False)
        left_img = processed(left_img)
        right_img = processed(right_img)

        return left_img, right_img, dataL

    def __len__(self):
        return len(self.frame_ids)


if __name__ == '__main__':
    loader = KITTIObjectLoader(sys.argv[1], sys.argv[2])
    loader.generate_sparse_disparity()
    # print(loader[0])
    # disp = loader[0][2]
    # print(torch.FloatTensor(disp).sign())
