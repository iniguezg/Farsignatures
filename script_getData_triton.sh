#!/bin/bash
#SBATCH --job-name=egonet_fits_text
#SBATCH --output=egonet_fits_text_%a.out
#SBATCH --array=179,194,356,360
#SBATCH --time=05-00
#SBATCH --mem-per-cpu=2G

#analysis 2
#time=01-00
#partition=batch
# srun python script_getData.py $1

#analysis 3
#time=01:00:00
#partition=short
#srun python script_getData.py

#analysis 5
#call: --array=1-895, text: --array=1-409
#call: --array=124,133,136,160,182,201,205,253,259,305,309,318,458,45,465,475,476,478,504,538,542,563,580,585,596,626,629,633,636,641,643,65,662,685,689,697,721,760,801,829,847,854,878,90
#text: --array=179,194,356,360
#find . ! -name . -prune -type f -name 'egonet_props_text_*' > filenames_text.txt
n=$SLURM_ARRAY_TASK_ID
filename=`sed -n "${n} p" filenames_text.txt`
srun python script_getData.py ${filename}
