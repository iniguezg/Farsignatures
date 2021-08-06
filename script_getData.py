#! /usr/bin/env python

### SCRIPT FOR GETTING DATA PROPERTIES IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
import pandas as pd
from os.path import expanduser

import data_misc as dm


## RUNNING DATA SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#input arguments
	loadflag = 'n' #load flag ( 'y', 'n' )

	# #SMALL DATASETS
	# root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	# root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	# saveloc = root_code+'files/data/' #location of output files
	# datasets = [ ('Copenhagen_nets', 'CNS_bt_symmetric.evt'), ('Copenhagen_nets', 'CNS_calls.evt'), ('Copenhagen_nets', 'CNS_sms.evt'), ('greedy_walk_nets', 'email.evt'), ('greedy_walk_nets', 'eml2.evt'), ('greedy_walk_nets', 'fb.evt'), ('greedy_walk_nets', 'forum.evt'), ('greedy_walk_nets', 'messages.evt'), ('MPC_UEu_net', 'MPC_UEu.evt'), ('SMS_net', 'MPC_Wu_SD01.evt'), ('SMS_net', 'MPC_Wu_SD02.evt'), ('SMS_net', 'MPC_Wu_SD03.evt'), ('greedy_walk_nets', 'pok.evt'), ('sex_contacts_net', 'sexcontact_events.evt') ]

	#LARGE DATASETS
	root_data = '/m/cs/scratch/networks-mobile/heydars1/set5_divided_to_small_files_for_gerardo_29_march_2021/'
	saveloc = '/m/cs/scratch/networks/inigueg1/prg/xocial/Farsignatures/files/data/'
	datasets = [ ( 'divided_to_roughly_40_mb_files_30_march', 'call' ), ( 'divided_to_roughly_40_mb_files_30_march', 'text' ) ]


	# ## analysis 1: format data (Bluetooth, Call, SMS) from Copenhagen Networks Study ##
	#
	# events_bt, events_call, events_sms = dm.format_data_CNS( root_data, loadflag )


	# ## analysis 2: get ego network properties for all datasets ##

	# #SMALL DATASETS
	# for dataname, eventname in datasets: #loop through considered datasets
	# 	print( 'dataset name: ' + eventname[:-4] ) #print output
	#
	# 	#prepare ego network properties / alter activities
	# 	egonet_props, egonet_acts = dm.egonet_props_acts( dataname, eventname, root_data, 'y', saveloc )
	#
	# #LARGE DATASETS
	# # datasets = [ ( 'divided_to_roughly_40_mb_files_30_march', sys.argv[1] ) ]
	# datasets = [ ('MPC_UEu_sample', 'text') ]
	# for dataname, eventname in datasets: #loop through datasets
	# 	print( 'dataset name: ' + eventname, flush=True ) #print output
	# 	fileloc = root_data + dataname +'/'+ eventname + '/'
	# 	filelist = os.listdir( fileloc )
	# 	# filelist = [ '1020407_1039726.txt' ]
	# 	filecount = 1 #initialise counter
	# 	for filename in filelist: #loop through files in data directory
	# 		print( 'progress: {:.2f}%, filename: {}'.format( 100*filecount/float(len(filelist)), filename ), flush=True )
	# 		filecount += 1 #update counter
	#
	# 		#prepare ego network properties / alter activities
	# 		egonet_props, egonet_acts = dm.egonet_props_acts_parallel( filename, fileloc, eventname, 'y', saveloc )


	# ## analysis 3: get parameters for all datasets ##

	# #SMALL DATASETS
	# params_data = dm.data_params( datasets, root_data, loadflag, saveloc )
	#
	# #LARGE DATASETS
	# # datasets = [ ('MPC_UEu_sample', 'text') ]
	# params_data = dm.data_params_parallel( datasets, root_data, loadflag, saveloc )


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

	#SMALL DATASETS
	# dataname = sys.argv[1]
	# eventname = sys.argv[2]
	#LARGE DATASETS
	dataname = '' #not needed for loading
	eventname = sys.argv[1][15:] #i.e. 'text_1000_1020405.pkl'

	print( 'event name: ' + eventname, flush=True ) #print output

	#fit activity model to all ego networks in dataset
	egonet_fits = dm.egonet_fits( dataname, eventname, root_data, loadflag, saveloc, nsims=300 )


	# ## analysis 8: join ego network properties and fits for large dataset separated into several files
	#
	# for dataname, eventname in datasets: #loop through datasets
	# 	print( 'dataset name: ' + eventname, flush=True ) #print output
	# 	egonet_props, egonet_fits = dm.egonet_props_fits_parallel( dataname, eventname, root_data, loadflag, saveloc )


	# ## analysis 7: fit gamma approx of activity model to all ego networks in all datasets ##
	#
	# for dataname, eventname in datasets: #loop through considered datasets
	# 	print( '\t\tdataset name: ' + eventname[:-4], flush=True ) #print output
	#
	# 	#fit gamma approx of activity model to all ego networks in dataset
	# 	egonet_gammas = dm.egonet_gammas( dataname, eventname, root_data, loadflag, saveloc )


# 	## analysis 8: build weighted graph from event list in all datasets ##
#
# 	# dataname = sys.argv[1] #considered dataset
# 	# eventname = sys.argv[2]
# #	datasets = [ ('Copenhagen_nets', 'CNS_bt_symmetric.evt') ]
# 	max_iter = 1000 #max number of iteration for centrality calculations
#
# 	for dataname, eventname in datasets: #loop through considered datasets
# 		print( 'dataset name: ' + eventname[:-4] ) #print output
#
# 		#build weighted graph from event list in dataset
# 		graph = dm.graph_weights( dataname, eventname, root_data, 'y', saveloc )
#
# 		#get graph properties for dataset
# 		graph_props = dm.graph_props( dataname, eventname, root_data, loadflag, saveloc, max_iter=max_iter )
