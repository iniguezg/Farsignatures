#!/bin/bash
#SBATCH --job-name=egonet_fits_test
#SBATCH --output=egonet_fits_test_%a.out
#SBATCH --array=1-2
#SBATCH --time=01:00:00
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
#find . ! -name . -prune -type f -name 'egonet_props_text_*' > filenames_text.txt
n=$SLURM_ARRAY_TASK_ID
filename=`sed -n "${n} p" filenames_test.txt`
srun python script_getData.py ${filename}
