#! /bin/bash 
subdomain_sizes=(1 05 025 0125 00625 003125)
#subdomain_sizes=(00625 003125)
#subdomain_sizes=(1 05)

output_file="ML_performance_out_${subdomain_sizes[0]}_${subdomain_sizes[-1]}.json"

# activate MSC environment (includes sklearn)
source /home/egordeev/.local/bin/pyenvs/MSC_env_hegoa/bin/activate

echo "chosen following subdomain sizes (degrees)" : ${subdomain_sizes[@]} 
python3 ML_performance.py -s ${subdomain_sizes[@]} -o $output_file & disown
