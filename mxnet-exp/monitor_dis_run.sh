#!/bin/bash

function run()
{
	interval=2
	target=1.5
	logfile=dis_alexnet.log
	export MXNET_KVSTORE_REDUCTION_NTHREADS=$1
	export MXNET_KVSTORE_BIGARRAY_BOUND=$2
	kvstore=$3
	server_num=$4
	worker_num=$5

	#launch training
		
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
								 --lr-step-epochs 30,60 \
								 --kv-store $kvstore \
								 > $logfile 2>&1 &
	
	# counting time	
	begin_time=$(date +%s)
	compline=$((${worker_num}*2))
	for((;;))
	do
		tail -n1 $logfile
		loss=$(tail -n${compline} dis_alexnet.log | sed -n 's/.*\scross-entropy=\([0-9]\+.[0-9]\+\)*/\1/p' | sort | head -n1)
		if [ -n "$loss" ] 
		then
			if [ $(echo "${loss} < ${target}" | bc) -eq 1 ]
			then
				break
			else
				sleep $interval
			fi
		else
			sleep $interval
		fi
	done
	end_time=$(date +%s)
	run_time=$((end_time - begin_time))
	echo "$1, $2, $3, $4, $5, $run_time" >> "$6"
	echo "training finshed! Cost ${run_time} s."
	./kill_dis.sh
	echo "killed process successfully!"
	rm -f $logfile
	echo "finished removal successfully!"
}

./kill_dis.sh
rfile=results_dis.csv
touch $rfile
echo "kvrn, kvbb, kvstore, snum, wnum, run_time" > $rfile

# test
# run 4 1000000 dist_sync 5 5 $rfile

# serch
for kvrn in 2 5 8; do
	for kvbb in 1000000 10000; do
		for kvstore in dist_sync dist_async; do
			for snum in 2 5 10; do
				for wnum in 2 5 10; do 
					run $kvrn $kvbb $kvstore $snum $wnum $rfile
					echo "waiting for restart..."
					sleep 10
				done	
			done
		done
	done
done

