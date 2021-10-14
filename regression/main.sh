#! /bin/bash 
subdomain_sizes=(1 05 025 0125 00625 003125)
#subdomain_sizes=(00625 003125)
#subdomain_sizes=(1 05)


# activate MSC environment (includes sklearn)
source /home/egordeev/.local/bin/pyenvs/MSC_env_hegoa/bin/activate

output_json="ML_performance_out_${subdomain_sizes[0]}_${subdomain_sizes[-1]}.json"
echo  output json is ${output_json}

# if output json file exists
if [ ! -e $output_json ]
then
	echo "calling ML_performance.py, generating JSON results"
	echo "chosen following subdomain sizes (degrees)" : ${subdomain_sizes[@]} 
	python3 ML_performance.py -s ${subdomain_sizes[@]} -o $output_json & disown
else
	echo "${output_json} already generated"

fi


output_type=$1
echo "proceed to visualization in  PNG"
# visualization of the JSON file as PNG file
if [ ${output_type} == "singleplot" ]
then
	echo "one Taylor diagram per PNG file will be plotted"
	output_taylor="Taylor_plot"
	python Taylor_plot.py -f ${output_json} -o ${output_taylor} -t singleplot
elif [ ${output_type} == "multiplot" ]
then
	echo "multiple Taylor diagrams per PNG file will be plotted"
	output_taylor="Taylor_plot_${subdomain_sizes[0]}_${subdomain_sizes[-1]}"
	python Taylor_plot.py -f ${output_json} -o ${output_taylor} -t multiplot
else
	echo "incorrect output type was chosen!!"
fi
	

