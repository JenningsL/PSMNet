#!/bin/bash

export CUDA_VISIBLE_DEVICES=5 && python finetune.py --maxdisp 192 \
                   --model stackhourglass \
                   --datatype 2015 \
                   --datapath /data/ssd/public/jlliu/scene_flow/training/ \
                   --epochs 300 \
                   --loadmodel model/kitti_2015 \
                   --savemodel ./trained/

