#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 && python submission.py --maxdisp 192 \
                     --model stackhourglass \
                     --kitti_path /data/ssd/public/jlliu/Kitti/object \
                     --loadmodel trained/finetune_90.tar

#--loadmodel ./model/kitti_2015 \

