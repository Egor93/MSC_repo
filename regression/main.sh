#! /bin/bash 
# call example $ ./main.sh singleplot

initialize () 
{
	#subdomain_sizes=(1 05 025 0125 00625 003125)
	#subdomain_sizes=(00625 003125)
	subdomain_sizes=(1 05)

	SETUP_DIR="${PWD}/data/input/setup"
	JSON_DIR="${PWD}/data/output/json"
	PLOT_DIR="${PWD}/data/output/img"
	SRC_DIR="${PWD}/src"
	# activate MSC environment (includes sklearn)
	source /home/egordeev/.local/bin/pyenvs/MSC_env_hegoa/bin/activate

	output_json="${JSON_DIR}/ML_performance_out_${subdomain_sizes[0]}_${subdomain_sizes[-1]}.json"
	input_csv="${SETUP_DIR}/setup.csv"
	#inputvars=
}


setup_experiments()
{
	input_csv=$1
	rootvars=$2
	extravars=$3
	# if input csv file exists
	if [ ! -e $input_csv ]
	then
		echo  -----generate input csv setup file ${input_csv}
		python ${SRC_DIR}/setup_experiments.py -o $input_csv -r ${rootvars} -e ${extravars}
		return $?
	else
		echo "-----${input_csv} already generated"
	fi
}


generate_json () 
{
	output_json=$1
	echo  -----output json is ${output_json}
	# if output json file exists
	if [ ! -e $output_json ]
	then
		echo "-----calling ML_performance.py, generating JSON results"
		echo "-----chosen following subdomain sizes (degrees)" : ${subdomain_sizes[@]} 
		python ${SRC_DIR}/ML_performance.py -s ${subdomain_sizes[@]} -o $output_json 
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
		output_taylor="${PLOT_DIR}/Taylor_plot"
		# resolution is appended to the plot name in Taylor_plot.py plot 
		python ${SRC_DIR}/Taylor_plot.py -f ${output_json} -o ${output_taylor} -t singleplot
	elif [[ ${output_type} == "multiplot" ]]
	then
		echo "-----multiple Taylor diagrams per PNG file will be plotted"
		output_taylor="${PLOT_DIR}/Taylor_plot_${subdomain_sizes[0]}_${subdomain_sizes[-1]}"
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
	setup_experiments ${input_csv} ${inputvars} ${extravars}
	generate_json ${output_json}
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
