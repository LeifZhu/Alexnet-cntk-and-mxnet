#!/bin/bash

function run()
{
	target=1.5	
	logfile=mx_alexnet.log
	export MXNET_CPU_WORKER_NTHREADS=$1
	export MXNET_CPU_PRIORITY_NTHREADS=$2
	export MXNET_CPU_NNPACK_NTHREADS=$3
	export MXNET_EXEC_ENABLE_INPLACE=$4
	python code/train_imagenet.py --network alexnet \
				 --num-classes 8 \
				 --data-train /root/zl_workspace/dataset/imagenet8/mxnet-format/train_rec.rec \
				 --data-val /root/zl_workspace/dataset/imagenet8/mxnet-format/val_rec.rec \
				 --num-examples 10400 \
				 --num-epochs 20 \
				 --loss 'ce' \
				 --disp-batches 1 \
				 --batch-size 256 \
				 --lr 0.01 \
				 --lr-step-epochs 5,10 \
				 >$logfile 2>&1 &
	
	begin_time=$(date +%s)
	sleep 10

	for((;;))
	do
		tail -n1 $logfile
		loss=$(tail -n1 $logfile | sed -n 's/.*\scross-entropy=\([0-9]\+.[0-9]\+\)*/\1/p') 
		if [ -n "$loss" ] 
		then
			if [ $(echo "${loss} < ${target}" | bc) -eq 1 ]
			then
				break
			else
				sleep 10
			fi
		else
			sleep 10
		fi
	done
	end_time=`date +%s`
	run_time=$((end_time - begin_time))
	echo "$1, $2, $3, $4, $run_time" >> "$5"
	echo "training finshed! Cost ${run_time} s."
	ps aux | grep "python code/train_imagenet.py" | grep -v "grep" | awk {'print $2'} | xargs kill
	echo "killed process successfully!"
	rm -f $logfile
	echo "finished removal successfully!"
}

ps aux | grep "python code/train_imagenet.py" | grep -v "grep" | awk {'print $2'} | xargs kill
touch results.csv
echo "mcwn, mcpn, mcnn, meei, run_time" > results.csv

for mcwn in 1 2 4 8; do
	for mcpn in 4 1 2 8; do
		for mcnn in 4 1 2 8; do
			for meei in true false; do
				if [ $(($mcwn + $mcpn + $mcnn)) -le 16 ]
				then
					run $mcwn $mcpn $mcnn $meei results.csv
					echo "waiting for restart..."
					sleep 10
				fi
			done
		done
	done
done

