#!/bin/bash 
# call example $ ./main.sh singleplot

initialize () 
{
	# activate MSC environment (includes sklearn)
	#source /home/egordeev/.local/bin/pyenvs/MSC_env_hegoa/bin/activate

	SRC_DIR="${PWD}/src"

	######## I/O FILES/DIRECTORIES SETUP #########
	NETCDFDIR="${PWD}/../../002_Data/"
	#NETCDFDIR="/home/egordeev/002_Data/"
	SETUP_DIR="${PWD}/data/input/setup"
	setup_csv="${SETUP_DIR}/setup.csv"
	#subdomain_sizes=(1 05 025 0125 00625 003125)
	#subdomain_sizes=(00625 003125)

	######## EXPERIMENT SETUP #########
	subdomain_sizes=(1 05 025 0125 00625)
    	root_inputvars="qtm,qsm,pm,tm"
      	extra_inputvars="qlm,skew_l,var_l,var_t"
    	#regtypes=("decision_tree" "gradient_boost" "random_forest")
    	regtypes="decision_tree,gradient_boost"
	
	######## OUTPUT DIRECTORIES #########
	RESULT_DIR="${PWD}/data/output"
	CSVOUT_DIR="${RESULT_DIR}/csv"
	PLOTOUT_DIR="${RESULT_DIR}/img/taylor_plot"

	######## PLOTTING SETUP #########
	multiplot="False"
}

setup_experiments ()
{
	setup_csv=$1
	root_inputvars=$2
	extra_inputvars=$3
	subdomain_sizes=$4
	regtypes=$5
	# if input csv file exists
	if [ ! -e $setup_csv ]
	# TODO: check if setup.csv should be updated
	then
		echo  -----generate csv setup file ${setup_csv}
		python ${SRC_DIR}/setup_experiments.py -o $setup_csv -r ${root_inputvars} -e ${extra_inputvars} -s ${subdomain_sizes[@]} -t ${regtypes} 
		# debug option: python -m ipdb
		#python -m ipdb ${SRC_DIR}/setup_experiments.py -o $setup_csv -r ${root_inputvars} -e ${extra_inputvars} -s ${subdomain_sizes[@]} -t ${regtypes[@]} 
		return $?
	else
		echo "-----${setup_csv} already generated"
	fi
}


run_experiments () 
{
	NETCDFDIR=$1
	setup_csv=$2
	CSVOUT_DIR=$3
	echo  -----running experiments using SETUP ${setup_csv}
	echo "-----chosen following subdomain sizes (degrees)" : ${subdomain_sizes[@]} 
	python ${SRC_DIR}/ML_performance.py -n ${NETCDFDIR} -s ${setup_csv} -o ${CSVOUT_DIR} 
	#python -m ipdb ${SRC_DIR}/ML_performance.py -n ${NETCDFDIR} -s ${setup_csv} -o ${CSVOUT_DIR} 
	# TODO if expout_R0.csv exists proceed with the following experiment

}


visualize ()
{

	multiplot=$1 # True of False
	# visualization of the experiment results as PNG file
	echo "-----Taylor diagram will be plotted, multiplot = ${multiplot}"
	python ${SRC_DIR}/Taylor_plot.py -i ${CSVOUT_DIR} -o ${PLOTOUT_DIR} -m ${multiplot} 
	#python -m ipdb ${SRC_DIR}/Taylor_plot.py -i ${CSVOUT_DIR} -o ${PLOTOUT_DIR} -m ${multiplot} 
}


# establish run order
main ()
{
	initialize	
	# setup - create csv table of ML runs parameters for each experiment
	#setup_experiments ${setup_csv} ${root_inputvars} ${extra_inputvars} ${subdomain_sizes} ${regtypes}

	#run_experiments ${NETCDFDIR} ${setup_csv} ${CSVOUT_DIR} 
	visualize ${multiplot}
	#return_status=$?
	#if [ $return_status -eq 0 ]
	#then
		#visualize ${multiplot}
	#else 
		#echo "-----generation of output file failed, execution stops"
	#fi
}

#set -x
main 
#set +x
#"$@"
