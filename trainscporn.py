#me: train_res50.sh
# Author: plustang
# mail: 
# Created Time: Mon 20 Feb 2017 10:59:03 PM EDT
#########################################################################
#!/bin/bash
#!/usr/bin/env sh
#PRETRAINED_MODEL=/home/zhangzheng/Workplace/NSFW/open_nsfw-master/nsfw_model/resnet50tune.caffemodel
#t=$(date +%Y-%m-%d_%H:%M:%S) 
#LOG=./log/tuan_res50_$t.log
GLOG_logtostderr=1 /home/zhangzheng/Software/caffe/caffe/build/tools/caffe train \
    --solver=/home/zhangzheng/Workplace/NSFW/open_nsfw-master/nsfw_model/train.prototxt \
    --weights=/home/zhangzheng/Workplace/NSFW/open_nsfw-master/nsfw_model/resnet50tune.caffemodel \
    --gpu=2,3 2>&1 | tee $LOG

