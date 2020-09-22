#! /usr/bin/env python

### SCRIPT FOR GETTING DATA PROPERTIES IN FARSIGNATURES PROJECT ###

#import modules
import pandas as pd
from os.path import expanduser

import data_misc as dm


## RUNNING DATA SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#input arguments
	loadflag = 'n' #load flag ( 'y', 'n' )

	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files

	# datasets = [ ('Copenhagen_nets', 'CNS_bt_symmetric.evt'), ('Copenhagen_nets', 'CNS_calls.evt'), ('Copenhagen_nets', 'CNS_sms.evt'), ('greedy_walk_nets', 'email.evt'), ('greedy_walk_nets', 'eml2.evt'), ('greedy_walk_nets', 'fb.evt'), ('greedy_walk_nets', 'forum.evt'), ('greedy_walk_nets', 'messages.evt'), ('MPC_UEu_net', 'MPC_UEu.evt'), ('SMS_net', 'MPC_Wu_SD01.evt'), ('SMS_net', 'MPC_Wu_SD02.evt'), ('SMS_net', 'MPC_Wu_SD03.evt'), ('greedy_walk_nets', 'pok.evt'), ('sex_contacts_net', 'sexcontact_events.evt') ]
	datasets = [ ('greedy_walk_nets', 'forum.evt') ]


	# ## analysis 1: format data (Bluetooth, Call, SMS) from Copenhagen Networks Study ##
	#
	# events_bt, events_call, events_sms = dm.format_data_CNS( root_data, loadflag )


	# ## analysis 2: get ego network properties for all datasets ##
	#
	# for dataname, eventname in datasets: #loop through considered datasets
	# 	print( 'dataset name: ' + eventname[:-4] ) #print output
	#
	# 	#prepare ego network properties
	# 	egonet_props, egonet_acts = dm.egonet_props_acts( dataname, eventname, root_data, 'y', saveloc )


	# ## analysis 3: get parameters for all datasets ##
	#
	# params_data = dm.data_params( datasets, root_data, loadflag, saveloc )


	# ## analysis 4: get number of egos with dynamics (t > a_0) for all datasets ##
	#
	# for dataname, eventname in datasets: #loop through considered datasets
	# 	print( '\t\tdataset name: ' + eventname[:-4] ) #print output
	#
	# 	#prepare ego network properties
	# 	egonet_props, egonet_acts = dm.egonet_props_acts( dataname, eventname, root_data, 'y', saveloc )
	# 	degrees, num_events, amins = egonet_props['degree'], egonet_props['strength'], egonet_props['act_min'] #unpack props
	#
	# 	num_egos = len(egonet_props) #total number of egos
	# 	#subset of egos with dynamics (t > a_0)
	# 	num_egos_dynamics = ( degrees * amins < num_events ).sum()
	#
	# 	print('total number of egos = {}'.format(num_egos))
	# 	print('number with t > a_0 = {}'.format(num_egos_dynamics))
	# 	print('fraction with t > a_0 = {:.2f}'.format( num_egos_dynamics / float(num_egos) ))


	## analysis 5: fit activity model to ego networks in all datasets ##

	bounds = (0, 1000) #bounds for alpha MLE fit
	nsims = 100 #number of syntethic datasets used to calculate p-value
	amax = 10000 #maximum activity for theoretical activity distribution

	for dataname, eventname in datasets: #loop through considered datasets
		print( 'dataset name: ' + eventname[:-4] ) #print output

		#fit activity model to all ego networks in dataset
		egonet_fits = dm.egonet_fits( dataname, eventname, root_data, loadflag, saveloc, bounds=bounds, nsims=nsims, amax=amax )
