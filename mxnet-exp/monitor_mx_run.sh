#!/bin/bash



function run()
{
	target=1.5
	cntk configFile=alexnet.cntk \
		precision="$1" \
		numCPUThreads=$2 \
		hyperCompressMemory=$3 \
		forceDeterministicAlgorithms=$4 \
		DataDir="/root/zl_workspace/dataset/imagenet8/raw" \
		>/dev/null &

	begin_time=$(date +%s)

	sleep 30

	for((;;))
	do
		tail -n1 Output/AlexNet
		loss=$(tail -n1 Output/AlexNet | sed -n 's/.*ce = \([0-9]\+.[0-9]\+\).*/\1/p')
		if [ $(echo "${loss} < ${target}" | bc) -eq 1 ]
		then
			break
		else
			sleep 10
		fi
	done
	end_time=`date +%s`
	run_time=$((end_time - begin_time))
	echo "$1, $2, $3, $4, $run_time" >> $5
	echo "training finshed! Cost ${run_time} s."
	ps aux | grep "cntk configFile=brain_alexnet.cntk" | grep -v "grep" |awk {'print $2'} | xargs kill
	rm -rf Output
	echo "finished removal successfully!"
}

ps aux | grep "cntk configFile=brain_alexnet.cntk" | grep -v "grep" |awk {'print $2'} | xargs kill
rm -rf Output
touch results.csv
echo "" > results.csv
echo "prec, nct, hcm, fda, run_time" >> results.csv

for fda in false true; do
	for hcm in false true; do
		for nct in 0 2 4 8 16; do
			for prec in float double; do
				run $prec $nct $hcm $fda results.csv
				echo "waiting for restart..."
				sleep 10
			done
		done
	done
done

