#!/bin/bash

srun -J CNS_bt_symmetric -o CNS_bt_symmetric.out python script_getData.py Copenhagen_nets CNS_bt_symmetric.evt

srun -J email -o email.out python script_getData.py greedy_walk_nets email.evt

srun -J fb -o fb.out python script_getData.py greedy_walk_nets fb.evt

srun -J forum -o forum.out python script_getData.py greedy_walk_nets forum.evt

srun -J messages -o messages.out python script_getData.py greedy_walk_nets messages.evt

srun -J MPC_UEu -o MPC_UEu.out python script_getData.py MPC_UEu_net MPC_UEu.evt

srun -J MPC_Wu_SD01 -o MPC_Wu_SD01.out python script_getData.py SMS_net MPC_Wu_SD01.evt

srun -J MPC_Wu_SD02 -o MPC_Wu_SD02.out python script_getData.py SMS_net MPC_Wu_SD02.evt

srun -J MPC_Wu_SD03 -o MPC_Wu_SD03.out python script_getData.py SMS_net MPC_Wu_SD03.evt

srun -J pok -o pok.out python script_getData.py greedy_walk_nets pok.evt

srun -J sexcontact_events -o sexcontact_events.out python script_getData.py sex_contacts_net sexcontact_events.evt

#to run on cluster:
#sbatch script_getData.sh
