#!/bin/bash

MX_PATH="~/git_repo/incubator-mxnet"
DATA_PATH="~/dataset/imagenet8/raw"


python ${MX_PATH}/tools/im2rec.py ${DATA_PATH}/train_rec ${DATA_PATH}/train/ --recursive --list --num-thread 8 &&
python ${MX_PATH}/tools/im2rec.py ${DATA_PATH}/train_rec ${DATA_PATH}/train/ --resize 256 --quality 90 --recursive --num-thread 8 &&
python ${MX_PATH}/tools/im2rec.py ${DATA_PATH}/val_rec ${DATA_PATH}/validation/ --recursive --list --num-thread 8 &&
python ${MX_PATH}/tools/im2rec.py ${DATA_PATH}/val_rec ${DATA_PATH}/validation/ --resize 256 --quality 90 --recursive --no-shuffle --num-thread 8 &&
mkdir -p ${DATA_PATH}/../mxnet-format/ &&
mv *.lst *.idx *.rec ${DATA_PATH}/../mxnet-format/ 
