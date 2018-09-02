#!/bin/bash

machine_num=$(cat hosts | wc -l)
worker_num=8
server_num=2
/root/zl_workspace/incubator-mxnet/tools/launch.py -n ${worker_num} -s ${server_num} -H hosts --launcher ssh \
python code/train_imagenet.py --network alexnet \
							 --num-classes 8 \
							 --data-train /root/zl_workspace/dataset/imagenet8/mxnet-format/train_rec.rec \
							 --num-examples 10400 \
							 --num-epochs 100 \
							 --loss 'ce' \
							 --disp-batches 1 \
							 --batch-size 256 \
							 --lr 0.01 \
							 --lr-step-epochs 5,10 \
							 --kv-store dist_sync \
							 > dis_alexnet.log 2>&1 &
