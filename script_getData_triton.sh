#!/bin/bash
#SBATCH --job-name=egonet_fits_piece
#SBATCH --output=egonet_fits_piece_%a.out
#SBATCH --array=1,4-11,13,15,18-25,27
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
#--job-name=egonet_fits_text
#--output=egonet_fits_text_%a.out
#call: --array=1-895, text: --array=1-409
#nsims:400, extra:300
#find . ! -name . -prune -type f -name 'egonet_props_text_*' > filenames_text.txt

#n=$SLURM_ARRAY_TASK_ID
#filename=`sed -n "${n} p" filenames_text.txt`
#srun python script_getData.py ${filename}


#analysis 12
#--array=1-28

n=$SLURM_ARRAY_TASK_ID
filename=`sed -n "${n} p" filenames_pieces.txt`
srun python script_getData.py ${filename}
