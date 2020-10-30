#!/bin/bash

sbatch -J alpha_-0.7_t_2.0_k_1000 -o alpha_-0.7_t_2.0_k_1000.out script_runModel.sh -0.7 2.0

sbatch -J alpha_-0.7_t_10.0_k_1000 -o alpha_-0.7_t_10.0_k_1000.out script_runModel.sh -0.7 10.0

sbatch -J alpha_-0.7_t_100.0_k_1000 -o alpha_-0.7_t_100.0_k_1000.out script_runModel.sh -0.7 100.0

sbatch -J alpha_-0.7_t_1000.0_k_1000 -o alpha_-0.7_t_1000.0_k_1000.out script_runModel.sh -0.7 1000.0

sbatch -J alpha_0.0_t_2.0_k_1000 -o alpha_0.0_t_2.0_k_1000.out script_runModel.sh 0.0 2.0

sbatch -J alpha_0.0_t_10.0_k_1000 -o alpha_0.0_t_10.0_k_1000.out script_runModel.sh 0.0 10.0

sbatch -J alpha_0.0_t_100.0_k_1000 -o alpha_0.0_t_100.0_k_1000.out script_runModel.sh 0.0 100.0

sbatch -J alpha_0.0_t_1000.0_k_1000 -o alpha_0.0_t_1000.0_k_1000.out script_runModel.sh 0.0 1000.0

sbatch -J alpha_9.0_t_2.0_k_1000 -o alpha_9.0_t_2.0_k_1000.out script_runModel.sh 9.0 2.0

sbatch -J alpha_9.0_t_10.0_k_1000 -o alpha_9.0_t_10.0_k_1000.out script_runModel.sh 9.0 10.0

sbatch -J alpha_9.0_t_100.0_k_1000 -o alpha_9.0_t_100.0_k_1000.out script_runModel.sh 9.0 100.0

sbatch -J alpha_9.0_t_1000.0_k_1000 -o alpha_9.0_t_1000.0_k_1000.out script_runModel.sh 9.0 1000.0
