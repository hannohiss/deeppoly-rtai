#!/bin/bash

for net in fc_linear fc_base fc_w fc_d fc_dw fc6_base fc6_w fc6_d fc6_dw conv_linear conv_base conv6_base conv_d skip skip_large skip6 skip6_large
do
	echo Evaluating network ${net}...
	for spec in `ls test_cases/${net}`
	do
		echo Evaluating spec ${spec}
		python code/verifier.py --net ${net} --spec test_cases/${net}/${spec}
	done
done
