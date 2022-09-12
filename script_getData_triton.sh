#!/bin/bash
#SBATCH --time=05-00
#SBATCH --mem-per-cpu=2G


## analysis 2: get ego network properties for all datasets ##

#time=01-00
#partition=batch

# srun python script_getData.py $1


## analysis 3: get parameters for all datasets ##

#time=01:00:00
#partition=short

#srun python script_getData.py


## analysis 5: fit activity model to ego networks in all datasets ##

#SMALL DATASETS
#--job-name=egonet_fits_QA
#--output=egonet_fits_QA_%a.out
#--array=1-3

# case $SLURM_ARRAY_TASK_ID in
#    1)  SEED="QA_nets QA_askubuntu.evt"  ;;
#    2)  SEED="QA_nets QA_mathoverflow.evt"  ;;
#    3)  SEED="QA_nets QA_superuser.evt"  ;;
# esac
# srun python script_getData.py $SEED


#LARGE DATASETS
#--job-name=egonet_fits_text
#--output=egonet_fits_text_%a.out
#call: --array=1-895, text: --array=1-409
#nsims:400, extra:300
#find . ! -name . -prune -type f -name 'egonet_props_text_*' > filenames_text.txt

#n=$SLURM_ARRAY_TASK_ID
#filename=`sed -n "${n} p" filenames_text.txt`
#srun python script_getData.py ${filename}


## analysis 9: compute connection kernel for all ego networks in all datasets ##

srun python script_getData.py $1


## analysis 12: fit activity model to ego networks per time period in all datasets ##

#--job-name=egonet_fits_piece
#--output=egonet_fits_piece_%a.out
#--array=1-28

# n=$SLURM_ARRAY_TASK_ID
# filename=`sed -n "${n} p" filenames_pieces.txt`
# srun python script_getData.py ${filename}
