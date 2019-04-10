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
import time
import functools
import traceback
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'kitti'))
import kitti_util
from kitti_object import *
import psutil
from data_util import get_sparse_disp, crop_np_matrix
from utils.vis import visualize_disparity

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def trace_unhandled_exceptions(func):
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except:
            print 'Exception in '+func.__name__
            traceback.print_exc()
    return wrapped_func

class KITTIObjectLoader(data.Dataset):
    def __init__(self, kitti_path, split, training=False):
        self.kitti_path = kitti_path
        self.kitti_dataset = kitti_object(kitti_path, 'training')
        self.left_dir = os.path.join(self.kitti_path, 'training', 'image_2')
        self.right_dir = os.path.join(self.kitti_path, 'training', 'image_3')
        self.disp_dir = os.path.join(self.kitti_path, 'training/disparity')
        self.frame_ids = self.load_split_ids(split)
        self.training = training
        self.baseline = 0.5379

    def load_split_ids(self, split):
        with open(os.path.join(self.kitti_path, split + '.txt')) as f:
            return [line.rstrip('\n') for line in f]

    def filter_occlusion(self, disp):
        def is_occluded(disp, x, y, r=8):
            for u in range(max(x-r, 0), min(x+r, disp.shape[1])):
                for v in range(max(y-r, 0), min(y+r, disp.shape[0])):
                    if disp[v][u] - disp[y][x] > 5:
                        return True
        for y in range(disp.shape[0]):
            for x in range(disp.shape[1]):
                if disp[y][x] < 5:
                    continue
                if is_occluded(disp, x, y):
                    disp[y][x] = 0

    @trace_unhandled_exceptions
    def generate_sparse_disparity(self, start, end):
        if not os.path.isdir(self.disp_dir):
            os.mkdir(self.disp_dir)
        for frame_id in self.frame_ids[start:end]:
            print(frame_id)
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
            start = time.time()
            self.filter_occlusion(disp_L)
            print(time.time() - start)
            fname = os.path.join(self.disp_dir, '{:06}.npy'.format(data_idx))
            np.save(fname, disp_L)
            # color_disp = cv2.applyColorMap(np.minimum(256, disp_L*2).astype(np.uint8), cv2.COLORMAP_JET)
            # cv2.imshow('disp', color_disp)
            # cv2.waitKey(0)
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

        if self.training:
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = crop_np_matrix(dataL, th, tw)

            processed = preprocess.get_transform(augment=False)
            left_img   = processed(left_img)
            right_img  = processed(right_img)

            # visualize_disparity(dataL)
            return left_img, right_img, dataL, get_sparse_disp(dataL, erase_ratio=0.8)

        else:
            target_w = 1248
            target_h = 352
            w, h = left_img.size

            # this will add zero padding
            left_img = left_img.crop((w-target_w, h-target_h, w, h))
            right_img = right_img.crop((w-target_w, h-target_h, w, h))
            w1, h1 = left_img.size

            dataL = crop_np_matrix(dataL, target_h, target_w)

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)
            # sparse = get_sparse_disp(dataL, erase_ratio=0.9)
            #visualize_disparity(dataL)
            # visualize_disparity(sparse)

            return left_img, right_img, dataL, get_sparse_disp(dataL, erase_ratio=0.8)

    def __len__(self):
        return len(self.frame_ids)


def process_range(kitti_path, split, start, end):
    loader = KITTIObjectLoader(kitti_path, split)
    loader.generate_sparse_disparity(start, end)

if __name__ == '__main__':
    # from multiprocessing import Pool
    # import math
    # loader = KITTIObjectLoader(sys.argv[1], sys.argv[2])
    # p = Pool()
    # worker_num = psutil.cpu_count()
    # print('Found %d core' % worker_num)
    # assert(len(loader) > worker_num)
    # batch = int(math.ceil(len(loader) / float(worker_num)))
    # results = []
    # for i in range(worker_num):
    #     start = i * batch
    #     end = min(len(loader), (i+1) * batch)
    #     p.apply_async(process_range, args=(sys.argv[1], sys.argv[2], start, end))
    # print 'Waiting for all subprocesses done...'
    # p.close()
    # p.join()
    # print 'Done'

    loader = KITTIObjectLoader(sys.argv[1], sys.argv[2], training=True)
    print(loader[0])
