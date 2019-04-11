from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
#import skimage
#import skimage.io
#import skimage.transform
import numpy as np
import time
import math
from utils import preprocess
from models import *
from models import unet_refine

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--datatype', default='kitti_2015',
                    help='datatype, sceneflow, kitti_2015, kitti_2012')
parser.add_argument('--datapath', default='./Kitti/object',
                    help='datapath')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--loadmodel_refine', default=None,
                    help='load refine model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--output', default='./output',
                    help='output path')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.datatype == 'kitti_2015':
    from dataloader import KITTIloader2015 as ls
    from dataloader import KITTILoader as DA
elif args.datatype == 'kitti_2012':
    from dataloader import KITTIloader2012 as ls
    from dataloader import KITTILoader as DA
elif args.datatype == 'sceneflow':
    from dataloader import listflowfile as ls
    from dataloader import SecenFlowLoader as DA
elif args.datatype == 'kitti_object':
    from dataloader.KITTIObjectLoader import KITTIObjectLoader
else:
    print('unknown datatype: ', args.datatype)
    sys.exit()

if args.datatype == 'kitti_object':
    dataloader = KITTIObjectLoader(args.datapath, 'trainval')
else:
    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(args.datapath)
    dataloader = DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False)


if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

refine_model = unet_refine.resnet34(pretrained=True)

model = nn.DataParallel(model, device_ids=[0])
model.cuda()
refine_model = nn.DataParallel(refine_model)
refine_model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'], strict=False)
if args.loadmodel_refine is not None:
    print('Loading refine model')
    state_dict = torch.load(args.loadmodel_refine)
    refine_model.load_state_dict(state_dict)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR, sparse_disp_L, refine=True):
        model.eval()
        refine_model.eval()

        if args.cuda:
           imgL = torch.FloatTensor(imgL).cuda()
           imgR = torch.FloatTensor(imgR).cuda()
           sparse_disp_L = torch.FloatTensor(sparse_disp_L).cuda()

        imgL, imgR, sparse_disp_L = Variable(imgL), Variable(imgR), Variable(sparse_disp_L)
        with torch.no_grad():
            output = model(imgL,imgR, sparse_disp_L)
            if refine:
                output = refine_model(imgL, output, sparse_disp_L)
        output = torch.squeeze(output)
        pred_disp = output.data.cpu().numpy()

        return pred_disp


def main():
   processed = preprocess.get_transform(augment=False)
   if not os.path.isdir(args.output):
       os.mkdir(args.output)
   for inx in range(len(dataloader)):
       if args.datatype == 'kitti_object':
           frame_id = dataloader.frame_ids[inx]
       else:
           frame_id = str(inx)
       imgL_o, imgR_o, sparse_disp_L, _ = dataloader[inx]
       imgL = imgL_o.numpy()
       imgR = imgR_o.numpy()
       imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
       imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])
       sparse_disp_L = np.reshape(sparse_disp_L,[1,sparse_disp_L.shape[0],sparse_disp_L.shape[1]])

       # pad to (384, 1248)
       '''
       top_pad = 384-imgL.shape[2]
       left_pad = 1248-imgL.shape[3]
       imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
       imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
       sparse_disp_L = np.lib.pad(sparse_disp_L,((0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
       '''
       start_time = time.time()
       pred_disp = test(imgL,imgR,sparse_disp_L,refine=False)
       print('%s: time = %.2f' %(frame_id, time.time() - start_time))
       '''
       top_pad   = 384-imgL_o.shape[0]
       left_pad  = 1248-imgL_o.shape[1]
       img = pred_disp[top_pad:,:-left_pad]
       '''
       print(pred_disp.shape)
       np.save(os.path.join(args.output, frame_id+'.npy'), pred_disp)

if __name__ == '__main__':
   main()
