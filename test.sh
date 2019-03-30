#!/bin/bash
export CUDA_VISIBLE_DEVICES=4 && python submission.py --maxdisp 192 \
                     --model stackhourglass \
                     --KITTI 2015 \
                     --datapath /data/ssd/public/jlliu/Kitti/object/training/ \
                     --loadmodel ./model/kitti_2015 \

 #--datapath /data/ssd/public/jlliu/scene_flow/testing/ \
