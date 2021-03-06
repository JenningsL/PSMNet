from __future__ import print_function
import argparse
import os
import sys
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
import copy

from models import *
from models import unet_refine
import criteria

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--decay_epochs', type=int ,default=10,
                    help='decay_epochs')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datatype', default='kitti_2015',
                    help='datatype, sceneflow, kitti_2015, kitti_2012')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/training/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='./trained/submission_model.tar',
                    help='load model')
parser.add_argument('--loadmodel_refine', default=None,
                    help='load refine model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
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

if args.datatype != 'kitti_object':
    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(args.datapath)
    TrainImgLoader = torch.utils.data.DataLoader(
             DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True),
             batch_size=12, shuffle=True, num_workers=8, drop_last=False)
    TestImgLoader = torch.utils.data.DataLoader(
             DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False),
             batch_size=6, shuffle=False, num_workers=4, drop_last=False)
else:
    TrainImgLoader = torch.utils.data.DataLoader(
             KITTIObjectLoader(args.datapath, 'train', training=True),
             batch_size=12, shuffle=True, num_workers=8, drop_last=False)
    TestImgLoader = torch.utils.data.DataLoader(
             KITTIObjectLoader(args.datapath, 'val', training=False),
             batch_size=6, shuffle=False, num_workers=4, drop_last=False)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

#refine_model = unet_refine.resnet50(pretrained=True)
refine_model = unet_refine.resnet34(pretrained=True, rgbd=True)
#refine_model = unet_refine.resnet18(pretrained=True, rgbd=True)

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()
    refine_model = nn.DataParallel(refine_model)
    refine_model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    #pretrained_dict = {k:v for k,v in state_dict1['state_dict'].items() if 'refine_depth' not in k}
    model.load_state_dict(state_dict['state_dict'])
if args.loadmodel_refine is not None:
    print('Loading refine model')
    state_dict = torch.load(args.loadmodel_refine)
    refine_model.load_state_dict(state_dict['state_dict'])

print('Number of PSMNet parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
print('Number of RefineNet parameters: {}'.format(sum([p.data.nelement() for p in refine_model.parameters()])))

#optimizer = optim.Adam(refine_model.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizer = optim.SGD(refine_model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.2**(epoch//args.decay_epochs))

smoothness_criterion = criteria.SmoothnessLoss()
def train(imgL,imgR,disp_L,sparse_disp_L):
        model.eval()
        refine_model.train()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))
        sparse_disp_L = Variable(torch.FloatTensor(sparse_disp_L))
        disp_L = Variable(torch.FloatTensor(disp_L))

        if args.cuda:
            imgL, imgR, disp_true, disp_true_sparse = imgL.cuda(), imgR.cuda(), disp_L.cuda(), sparse_disp_L.cuda()

        #---------
        mask = (disp_true > 0)
        mask.detach_()
        #----

        optimizer.zero_grad()
        if args.model == 'stackhourglass':
            with torch.no_grad():
                #output1, output2, output3 = model(imgL,imgR,disp_true_sparse)
                output3 = model(imgL,imgR,disp_true_sparse)
            #output1 = refine_model(imgL, output1, disp_true_sparse)
            #output2 = refine_model(imgL, output2, disp_true_sparse)
            output3 = refine_model(imgL, output3, disp_true_sparse)
            assert not torch.isnan(output3).any()
            #output1 = torch.squeeze(output1,1)
            #output2 = torch.squeeze(output2,1)
            smooth_loss = smoothness_criterion(output3)
            output3 = torch.squeeze(output3,1)
            #loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)
            loss = F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) + 0.1 * smooth_loss
        elif args.model == 'basic':
            output = model(imgL,imgR)
            output = torch.squeeze(output3,1)
            loss = F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)

        loss.backward()
        optimizer.step()

        #return loss.data[0]
        return loss.data

def test(imgL,imgR,disp_true, sparse_disp_L):
        model.eval()
        refine_model.eval()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))
        # dont use pixel with sparse disp for evaluation
        without_sparse_mask = sparse_disp_L == 0 # (4, 544, 960)
        disp_true *= without_sparse_mask.float()
        sparse_disp_L = Variable(torch.FloatTensor(sparse_disp_L))
        if args.cuda:
            imgL, imgR, disp_true_sparse = imgL.cuda(), imgR.cuda(), sparse_disp_L.cuda()
        with torch.no_grad():
            output3 = model(imgL,imgR, disp_true_sparse)
            output3 = refine_model(imgL, output3, disp_true_sparse)
            output3 = torch.squeeze(output3,1)
        pred_disp = output3.data.cpu()

        # ignore padding
        if args.datatype == 'sceneflow':
            pred_disp = pred_disp[:,4:,:]
            disp_true = disp_true[:,4:,:]

        #computing 3-px error for kitti#
        disp_true_bk = copy.deepcopy(disp_true)
        true_disp = disp_true
        index = np.argwhere(true_disp>0)
        disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
        max_pixel_err = 3 #3
        max_ratio_err = 0.05 #0.05
        correct = (disp_true[index[0][:], index[1][:], index[2][:]] < max_pixel_err)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*max_ratio_err)
        torch.cuda.empty_cache()

        err_3px = 1-(float(torch.sum(correct))/float(len(index[0])))

        # end-point-error for sceneflow
        disp_true = disp_true_bk
        mask1 = disp_true < 192
        mask2 = disp_true > 0
        mask = mask1 * mask2
        if len(disp_true[mask])==0:
           err_endpoint = 0
        else:
           err_endpoint = torch.mean(torch.abs(pred_disp[mask]-disp_true[mask]))

        return err_3px*100, err_endpoint

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
       lr = 0.001
    else:
       lr = 0.0001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def main():
    max_acc=0
    max_epo=0
    start_full_time = time.time()

    start_epoch = 1
    #if args.loadmodel_refine is not None:
    #    start_epoch = state_dict['epoch']
    #assert(start_epoch < args.epochs+1)
    for epoch in range(start_epoch, args.epochs+1):
        total_train_loss = 0
        #adjust_learning_rate(optimizer,epoch)
        scheduler.step()
        print('epoch {0} learning rate {1}'.format(epoch, get_learning_rate(optimizer)))
        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, sparse_disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()

            loss = train(imgL_crop,imgR_crop, disp_crop_L, sparse_disp_crop_L)
            print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))
        ## Test ##
        total_err_3px = 0
        total_err_endpoint = 0
        for batch_idx, (imgL, imgR, disp_L, sparse_disp_L) in enumerate(TestImgLoader):
            err_3px, err_endpoint = test(imgL,imgR, disp_L, sparse_disp_L)
            print('Iter %d 3-px error in val = %.3f' %(batch_idx, err_3px))
            print('Iter %d endpoint err in val = %.3f' %(batch_idx, err_endpoint))
            total_err_3px += err_3px
            total_err_endpoint += err_endpoint

        print('epoch %d total 3-px error in val = %.3f' %(epoch, total_err_3px/len(TestImgLoader)))
        print('total endpoint error = %.3f' %(total_err_endpoint/len(TestImgLoader)))

        #SAVE
        savefilename = args.savemodel+'finetune_'+str(epoch)+'.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': refine_model.state_dict()
        }, savefilename)
        #torch.save(refine_model.state_dict(), savefilename)
    print('full finetune time = %.2f HR' %((time.time() - start_full_time)/3600))
    print(max_epo)
    print(max_acc)


if __name__ == '__main__':
   main()
