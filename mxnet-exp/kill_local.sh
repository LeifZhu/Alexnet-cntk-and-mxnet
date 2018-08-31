#!/bin/bash


ps aux | grep "monitor_mx_run.sh" | grep -v "grep" | awk {'print $2'} | xargs kill
ps aux | grep "python code/train_imagenet.py" | grep -v "grep" | awk {'print $2'} | xargs kill
