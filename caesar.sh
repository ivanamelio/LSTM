#!/bin/bash


DRP[4]=1.
DRP[8]=1.
DRP[16]=1 #0.8
DRP[32]=1 #0.65
DRP[64]=1 #0.5
DRP[128]=0.9
DRP[256]=0.75 

NTS[4]=7000
NTS[8]=8000
NTS[16]=12000
NTS[32]=15000
NTS[64]=18000
NTS[100]=21000
NTS[128]=23000 #to increase!!!
NTS[156]=25000
NTS[256]=30000


for index in 1 2 3 4 5
do
	for num_hidden in 116 
	do
		echo $index  $num_hidden  ${DRP[$num_hidden]}
		python hyper_lstm.py $num_hidden output_gate=yes 1.  21000 #${NTS[$num_hidden]}
		#python var_mem_LSTM.py $num_hidden output_gate=yes ${DRP[$num_hidden]} 6 ${NTS[$num_hidden]}
		#python var_mem_LSTM.py $num_hidden output_gate=no ${DRP[$num_hidden]} 6 ${NTS[$num_hidden]}	
	done
done

