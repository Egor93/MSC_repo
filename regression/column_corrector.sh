file=$1

dir1="data/output/csv"
files=$(ls $dir1) 
for file in $files
do
	corrected_file="$(awk 'BEGIN {OFS="\t"}; { if ($8 == "25") ($8 = "025");
				if ($8 == "5") ($8 = "05"); 
				if ($8 == "125") ($8 = "0125");
				if ($8 == "625") ($8 = "00625");
				if ($8 == "3125") ($8 = "003125");
					print $0}' ${dir1}/${file})"
	echo "${corrected_file}">${dir1}/${file}
done

