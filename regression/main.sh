#! /bin/bash 
# call example $ ./main.sh singleplot

initialize () 
{
	# activate MSC environment (includes sklearn)
	#source /home/egordeev/.local/bin/pyenvs/MSC_env_hegoa/bin/activate

	SRC_DIR="${PWD}/src"

	######## SETUP AND INPUT #########
	NETCDFDIR="${PWD}/../../002_Data/"
	#NETCDFDIR="/home/egordeev/002_Data/"
	SETUP_DIR="${PWD}/data/input/setup"
	setup_csv="${SETUP_DIR}/setup.csv"
	#subdomain_sizes=(1 05 025 0125 00625 003125)
	#subdomain_sizes=(00625 003125)
	subdomain_sizes=(1 05)
	
	######## OUTPUT DIRECTORIES #########
	RESULT_DIR="${PWD}/data/ouput"
	JSONOUT_DIR="${RESULT_DIR}/json"
	CSVOUT_DIR="${RESULT_DIR}/csv"
	PLOTOUT_DIR="${RESULT_DIR}/img"
	#output_json="${JSONOUT_DIR}/ML_performance_out_${subdomain_sizes[0]}_${subdomain_sizes[-1]}.json"
}


setup_experiments ()
{
	setup_csv=$1
	rootvars=$2
	extravars=$3
	subdomain_sizes=$4
	# if input csv file exists
	if [ ! -e $setup_csv ]
	then
		echo  -----generate input csv setup file ${setup_csv}
		python ${SRC_DIR}/setup_experiments.py -o $setup_csv -r ${rootvars} -e ${extravars} -s ${subdomain_sizes[@]} 
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
	echo  -----output json is ${output_json}
	# if output json file exists
	if [ ! -e $output_json ]
	then
		echo "-----calling ML_performance.py, generating JSON results"
		echo "-----chosen following subdomain sizes (degrees)" : ${subdomain_sizes[@]} 
		python ${SRC_DIR}/ML_performance.py -n ${NETCDFDIR} -s ${setup_csv} -o ${CSVOUT_DIR} 

		#python3 ${SRC_DIR}/ML_performance.py -s ${subdomain_sizes[@]} -o $output_json & disown
		return $?
	else
		echo "-----${output_json} already generated"
	fi
}


visualize ()
{

	output_type=$1
	echo "-----proceed to visualization in  PNG"
	# visualization of the JSON file as PNG file
	if [[ ${output_type} == "singleplot" ]]
	then
		echo "-----one Taylor diagram per PNG file will be plotted"
		output_taylor="${PLOTOUT_DIR}/Taylor_plot"
		# resolution is appended to the plot name in Taylor_plot.py plot 
		python ${SRC_DIR}/Taylor_plot.py -f ${output_json} -o ${output_taylor} -t singleplot
	elif [[ ${output_type} == "multiplot" ]]
	then
		echo "-----multiple Taylor diagrams per PNG file will be plotted"
		output_taylor="${PLOTOUT_DIR}/Taylor_plot_${subdomain_sizes[0]}_${subdomain_sizes[-1]}"
		python ${SRC_DIR}/Taylor_plot.py -f ${output_json} -o ${output_taylor} -t multiplot
	else
		echo "-----incorrect output type was chosen!!"
	fi
}


# establish run order
main ()
{
	output_type=$1	
	initialize	
	# setup - create csv table of ML runs parameters for each experiment
	setup_experiments ${setup_csv} ${inputvars} ${extravars} ${subdomain_sizes}  

	run_experiments ${NETCDFDIR} ${setup_csv} ${CSVOUT_DIR} 
	return_status=$?
	if [ $return_status -eq 0 ]
	then
		visualize ${output_type}
	else 
		echo "-----generation of json output failed, execution stops"
	fi
}

#set -x
#main $1
#set +x

"$@"
