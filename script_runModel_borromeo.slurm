#!/bin/bash

sbatch -J alpha_-0.999_t_1.1_k_100 -o alpha_-0.999_t_1.1_k_100.out script_runModel.sh -0.999 1.1

sbatch -J alpha_-0.99_t_2.0_k_100 -o alpha_-0.99_t_2.0_k_100.out script_runModel.sh -0.99 2.0

sbatch -J alpha_-0.9_t_11.0_k_100 -o alpha_-0.9_t_11.0_k_100.out script_runModel.sh -0.9 11.0

sbatch -J alpha_0.0_t_101.0_k_100 -o alpha_0.0_t_101.0_k_100.out script_runModel.sh 0.0 101.0

sbatch -J alpha_9.0_t_1001.0_k_100 -o alpha_9.0_t_1001.0_k_100.out script_runModel.sh 9.0 1001.0


sbatch -J alpha_-0.9_t_1.1_k_100 -o alpha_-0.9_t_1.1_k_100.out script_runModel.sh -0.9 1.1

sbatch -J alpha_0.0_t_2.0_k_100 -o alpha_0.0_t_2.0_k_100.out script_runModel.sh 0.0 2.0

sbatch -J alpha_9.0_t_11.0_k_100 -o alpha_9.0_t_11.0_k_100.out script_runModel.sh 9.0 11.0

sbatch -J alpha_99.0_t_101.0_k_100 -o alpha_99.0_t_101.0_k_100.out script_runModel.sh 99.0 101.0

sbatch -J alpha_999.0_t_1001.0_k_100 -o alpha_999.0_t_1001.0_k_100.out script_runModel.sh 999.0 1001.0


sbatch -J alpha_9.0_t_1.1_k_100 -o alpha_9.0_t_1.1_k_100.out script_runModel.sh 9.0 1.1

sbatch -J alpha_99.0_t_2.0_k_100 -o alpha_99.0_t_2.0_k_100.out script_runModel.sh 99.0 2.0

sbatch -J alpha_999.0_t_11.0_k_100 -o alpha_999.0_t_11.0_k_100.out script_runModel.sh 999.0 11.0

sbatch -J alpha_9999.0_t_101.0_k_100 -o alpha_9999.0_t_101.0_k_100.out script_runModel.sh 9999.0 101.0

sbatch -J alpha_99999.0_t_1001.0_k_100 -o alpha_99999.0_t_1001.0_k_100.out script_runModel.sh 99999.0 1001.0
