cd /home/igor/UNI/Master\ Project/Script/Data
pwd /home/igor/UNI/Master Project/Script/Data
#a= cat README_on_douze_files.txt |grep cl_l
ncfiles=$(ls | grep douze |grep nc)
for file in $ncfiles
  do
    ncatted -a longname,cl_l,c,c,"LES liquid cloud fraction" $file
  done


# get a list of attributes
# Read longname from README_on douze files
