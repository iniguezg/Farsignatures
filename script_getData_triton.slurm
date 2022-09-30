#!/bin/bash

module load anaconda

#analysis 2/9
# sbatch -J text -o text.out script_getData_triton.sh text
# sbatch -J call -o call.out script_getData_triton.sh call

#analysis 3
# sbatch -J params_data_parallel -o params_data_parallel.out script_getData_triton.sh

#analysis 5/9/11/12
# sbatch script_getData_triton.sh

#analysis 12
# sbatch -J egonet_piece_0_test -o egonet_piece_0_test_%a.out --array=1-2 script_getData_triton.sh text 0 400
# sbatch -J egonet_piece_0_text -o egonet_piece_0_text_%a.out --array=1-409 script_getData_triton.sh text 0 400
sbatch -J egonet_piece_1_text -o egonet_piece_1_text_%a.out --array=1-409 script_getData_triton.sh text 1 400
sbatch -J egonet_piece_0_call -o egonet_piece_0_call_%a.out --array=1-895 script_getData_triton.sh call 0 400
sbatch -J egonet_piece_1_call -o egonet_piece_1_call_%a.out --array=1-895 script_getData_triton.sh call 1 400

#plot figure 1
# sbatch -J figure1 -o figure1.out script_getData_triton.sh
