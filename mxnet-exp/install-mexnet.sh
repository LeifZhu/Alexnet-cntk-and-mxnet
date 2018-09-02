#!/bin/bash

for((i=3;i<=37;i++));do
    ssh "ssd$(printf "%02d" $i)" "pip uninstall -y mxnet-mkl && pip install mxnet-mkl && python -c 'import mxnet; print(mxnet.__version__)'"
done
