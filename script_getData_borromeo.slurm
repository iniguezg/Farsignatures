#!/bin/bash

# sbatch -J CNS_bt_symmetric -o CNS_bt_symmetric.out script_getData.sh Copenhagen_nets CNS_bt_symmetric.evt
#
# sbatch -J CNS_calls -o CNS_calls.out script_getData.sh Copenhagen_nets CNS_calls.evt
#
# sbatch -J CNS_sms -o CNS_sms.out script_getData.sh Copenhagen_nets CNS_sms.evt
#
# sbatch -J email -o email.out script_getData.sh greedy_walk_nets email.evt
#
# sbatch -J eml2 -o eml2.out script_getData.sh greedy_walk_nets eml2.evt

sbatch -J fb -o fb.out script_getData.sh greedy_walk_nets fb.evt 300

sbatch -J forum -o forum.out script_getData.sh greedy_walk_nets forum.evt 700

# sbatch -J messages -o messages.out script_getData.sh greedy_walk_nets messages.evt
#
# sbatch -J MPC_UEu -o MPC_UEu.out script_getData.sh MPC_UEu_net MPC_UEu.evt
#
# sbatch -J MPC_Wu_SD01 -o MPC_Wu_SD01.out script_getData.sh SMS_net MPC_Wu_SD01.evt
#
# sbatch -J MPC_Wu_SD02 -o MPC_Wu_SD02.out script_getData.sh SMS_net MPC_Wu_SD02.evt
#
# sbatch -J MPC_Wu_SD03 -o MPC_Wu_SD03.out script_getData.sh SMS_net MPC_Wu_SD03.evt

sbatch -J pok -o pok.out script_getData.sh greedy_walk_nets pok.evt 900

# sbatch -J sexcontact_events -o sexcontact_events.out script_getData.sh sex_contacts_net sexcontact_events.evt
