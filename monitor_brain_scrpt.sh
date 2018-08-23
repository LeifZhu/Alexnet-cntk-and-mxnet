#!/bin/bash


ps aux | grep "cntk configFile=brain_alexnet.cntk" | grep -v "grep" |awk {'print $2'} | xargs kill
rm -rf Output

target=1.5
cntk configFile=brain_alexnet.cntk >/dev/null &
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
echo "training finshed! Cost ${run_time} s."
ps aux | grep "cntk configFile=brain_alexnet.cntk" | grep -v "grep" |awk {'print $2'} | xargs kill
rm -rf Output
echo "finished removal successfully!"
