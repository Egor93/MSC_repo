#!/bin/sh

dir1="data/output/csv"
dir2=$1
files=$(ls $dir1) 
header="	input_vars_id	input_vars	satdeficit	eval_fraction	regtypes	tree_maxdepth	subdomain_sizes	refstd	samplestd	samplecorr	exectime"
for file in $files
do
	if [ -e "${dir2}/${file}" ]
	then
		#cat "${dir1}/${file}" "${dir2}/${file}"
		awk 'FNR==1 && NR!=1{next;}{print}'  "${dir1}/${file}" "${dir2}/${file}" > "${dir1}/${file}"
	else
		echo "${dir2}/${file}" does not exist 
	fi
done

