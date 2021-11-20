#!/bin/sh 


dir1="data/output/csv"
files=$(ls $dir1) 
header="	input_vars_id	input_vars	satdeficit	eval_fraction	regtypes	tree_maxdepth	subdomain_sizes	refstd	samplestd	samplecorr	exectime"

for file in $files
do 
	echo "${header}\n""$(awk 'FNR > 1' ${dir1}/${file})" > ${dir1}/${file}
done
