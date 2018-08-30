#!/bin/bash

cat hosts | while read host;  do ssh -o "StrictHostKeyChecking no" $host "pkill -f python"; done
