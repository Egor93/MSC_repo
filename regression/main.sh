#!/bin/bash 
# call example $ ./main.sh singleplot

initialize () 
{
	# activate MSC environment (includes sklearn)
	#source /home/egordeev/.local/bin/pyenvs/MSC_env_hegoa/bin/activate

	######## INPUT FILES/DIRECTORIES SETUP #########
	SRC_DIR="${PWD}/src"
	NETCDFDIR="${PWD}/../../002_Data/"
	SETUP_DIR="${PWD}/data/input/setup"


	######## EXPERIMENT SETUP #########
	subdomain_sizes=(1 05 025 0125 00625)

	# its important to PRESERVE THE SAME SEQUENCE of extra vars for every run!!!
	#    		var0, var1, var2, var3
	#extra_inputvars="None"

	mode1="binary subgroup comparison"
	# mode 1: unique invars run divided into 2 subgroups 
	#        with and without key setup variable
	#        TaylorPlot - each group of runs=separate key= separate label
	mode2="input variables reduction"
	# mode 2: unique invars  run, TaylorPlot - each reduction sequence=separate label
	mode3="determinicity check"
	# mode 3: compare several runs with same setup each,
	#         check impact of nondeterminicity of ML algorithm on the exp results 


	MODE="binary subgroup comparison"

	if [[ "$MODE" == "$mode1" ]]
	then 
		# binary subgroup comparison
		echo "MODE SELECTION:${MODE}"
		subgroup_key="input_vars"
		subgroup_val="qlm"
		root_inputvars="qtm,qsm,pm,tm"
		extra_inputvars="qlm,skew_l,var_l,var_t"
		nexprepeat=0
	elif [[ "$MODE" == "$mode2" ]]
	then
		#input variables reduction
		echo "MODE SELECTION:${MODE}"
		root_inputvars="qtm,qsm,pm,tm"
		extra_inputvars="qlm,skew_l,var_l,var_t"
		nexprepeat=0
	elif [[ "$MODE" == "$mode3" ]]
	then
		# ML determinicity check 
		echo "MODE SELECTION:${MODE}"
		root_inputvars="qtm,qsm,pm,tm"
		extra_inputvars="None"
		nexprepeat=10
		split_dataset_randomly="True"
	else
		echo "no valid mode chosen"
	fi
	

	#nexprepeat=10

    	#regtypes="decision_tree,gradient_boost,random_forest"
    	regtypes="decision_tree"
	

	######## OUTPUT DIRECTORIES #########
	RESULT_DIR="${PWD}/data/output"
	PLOTOUT_DIR="${RESULT_DIR}/img/taylor_plot"

	if [[ $nexprepeat -eq 0 ]]
	then
		# Default behaviour, no repetitons
		setup_csv="${SETUP_DIR}/setup.csv"
		CSVOUT_DIR="${RESULT_DIR}/csv"
	else
		# Modified behaviour, with repetitons
		setup_csv="${SETUP_DIR}/setup_repeat.csv"
		CSVOUT_DIR="${RESULT_DIR}/csv/repeat"
	fi


	######## PLOTTING SETUP #########
	multiplot="False"
}

setup_experiments ()
{
	setup_csv=$1
	root_inputvars=$2
	subdomain_sizes=$3
	regtypes=$4
	extra_inputvars=$5
	nexprepeat=$6
	# generate setup in any case
	echo  -----generate csv setup file ${setup_csv}
	python ${SRC_DIR}/setup_experiments.py -o $setup_csv -r ${root_inputvars} -e ${extra_inputvars} -s ${subdomain_sizes[@]} -t ${regtypes} -N ${nexprepeat}
	#python -m ipdb ${SRC_DIR}/setup_experiments.py -o $setup_csv -r ${root_inputvars} -e ${extra_inputvars} -s ${subdomain_sizes[@]} -t ${regtypes} -N ${nexprepeat}
}


run_experiments () 
{
	NETCDFDIR=$1
	setup_csv=$2
	CSVOUT_DIR=$3
	nexprepeat=$4
	split_randomly=$5
	echo  -----running experiments using SETUP ${setup_csv}
	echo "-----chosen following subdomain sizes (degrees)" : ${subdomain_sizes[@]} 
	# TODO: multirun - run the same setup several times
	echo ${nexprepeat}
	python -m ipdb ${SRC_DIR}/ML_performance.py -n ${NETCDFDIR} -s ${setup_csv} -o ${CSVOUT_DIR} -N ${nexprepeat} -R ${split_randomly}
	#python ${SRC_DIR}/ML_performance.py -n ${NETCDFDIR} -s ${setup_csv} -o ${CSVOUT_DIR} -N ${nexprepeat} -R ${split_randomly}
	# TODO if expout_R0.csv exists proceed with the following experiment

}


visualize ()
{

	multiplot=$1 # True of False
	CSVOUT_DIR=$2
	PLOTOUT_DIR=$3
	nexprepeat=$4
	root_inputvars=$5
	subgroup_key=$6
	subgroup_val=$7
	# visualization of the experiment results as PNG file
	echo "-----Taylor diagram will be plotted, multiplot = ${multiplot}"
	#python  ${SRC_DIR}/Taylor_plot.py -i ${CSVOUT_DIR} -o ${PLOTOUT_DIR} -m ${multiplot} -N ${nexprepeat} -R ${root_inputvars}
	python -m ipdb  ${SRC_DIR}/Taylor_plot.py \
		-i ${CSVOUT_DIR} \
		-o ${PLOTOUT_DIR} \
		-m ${multiplot} \
		-N ${nexprepeat} \
		-R ${root_inputvars}\
		-k ${subgroup_key} \
		-v ${subgroup_val}

}


# establish run order
main ()
{
	initialize	

	# setup - create csv table of ML runs parameters for each experiment
	# TODO: need to manually delete setup csv to generate new one
	#setup_experiments ${setup_csv} ${root_inputvars} ${subdomain_sizes} ${regtypes} ${extra_inputvars} ${nexprepeat}
	#run_experiments ${NETCDFDIR} ${setup_csv} ${CSVOUT_DIR} ${nexprepeat} ${split_dataset_randomly}
	visualize ${multiplot} ${CSVOUT_DIR} ${PLOTOUT_DIR} ${nexprepeat} ${root_inputvars} ${subgroup_key} ${subgroup_val}
	#return_status=$?
	#if [ $return_status -eq 0 ]
	#then
		#visualize ${multiplot}
	#else 
		#echo "-----generation of output file failed, execution stops"
	#fi
}

set -x
main
set +x
#"$@"
