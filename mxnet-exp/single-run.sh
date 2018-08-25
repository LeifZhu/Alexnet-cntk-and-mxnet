#!/bin/bash

python code/train_imagenet.py --network alexnet \
			 --num-classes 8 \
			 --data-train /home/leizhu/dataset/imagenet8/mxnet-format/train_rec.rec \
			 --data-val /home/leizhu/dataset/imagenet8/mxnet-format/val_rec.rec \
			 --num-examples 10400 \
			 --num-epochs 20 \
			 --loss 'ce' \
			 --disp-batches 1 \
			 --batch-size 256 \
			 --lr 0.01 \
			 --lr-step-epochs 5,10
