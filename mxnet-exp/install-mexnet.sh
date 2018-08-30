#!/bin/bash

for((i=5;i<=37;i++));do
    ssh "ssd$(printf "%02d" $i)" pip install mxnet
done
