#!/bin/bash

num=$(cat hosts | wc -l)
cat hosts | xargs -i{} -P ${num} ssh {} "bash /root/zl_workspace/Alexnet-cntk-and-mxnet/mxnet-exp/kill_local.sh" 
