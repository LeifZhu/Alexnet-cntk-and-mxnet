#!/bin/bash

for((i=3;i<=37;i++));do
    ssh "ssd$(printf "%02d" $i)" "pip uninstall -y mxnet; pip install mxnet-mkl"
done
