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

	#SMALL DATASETS
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files
	# datasets = [ ('Copenhagen_nets', 'CNS_bt_symmetric.evt'), ('Copenhagen_nets', 'CNS_calls.evt'), ('Copenhagen_nets', 'CNS_sms.evt'), ('greedy_walk_nets', 'email.evt'), ('greedy_walk_nets', 'eml2.evt'), ('greedy_walk_nets', 'fb.evt'), ('greedy_walk_nets', 'forum.evt'), ('greedy_walk_nets', 'messages.evt'), ('MPC_UEu_net', 'MPC_UEu.evt'), ('SMS_net', 'MPC_Wu_SD01.evt'), ('SMS_net', 'MPC_Wu_SD02.evt'), ('SMS_net', 'MPC_Wu_SD03.evt'), ('greedy_walk_nets', 'pok.evt'), ('sex_contacts_net', 'sexcontact_events.evt'), ('QA_nets', 'QA_askubuntu.evt'), ('QA_nets', 'QA_mathoverflow.evt'), ('QA_nets', 'QA_superuser.evt'), ('SNAP', 'email_Eu_core.evt'), ('SNAP', 'CollegeMsg.evt'), ('Enron', 'Enron.evt') ]
	datasets = [ ('SMS_net', 'MPC_Wu_SD01.evt'), ('SMS_net', 'MPC_Wu_SD02.evt'), ('SMS_net', 'MPC_Wu_SD03.evt'), ('SNAP', 'email_Eu_core.evt'), ('SNAP', 'CollegeMsg.evt'), ('Enron', 'Enron.evt') ]

	# #LARGE DATASETS
	# root_data = '/m/cs/scratch/networks-mobile/heydars1/set5_divided_to_small_files_for_gerardo_29_march_2021/'
	# saveloc = '/m/cs/scratch/networks/inigueg1/prg/xocial/Farsignatures/files/data/'
	# datasets = [ ( 'divided_to_roughly_40_mb_files_30_march', 'call' ), ( 'divided_to_roughly_40_mb_files_30_march', 'text' ) ]


	## analysis 1: format data ##

	#(Bluetooth, Call, SMS) from Copenhagen Networks Study
	# events_bt, events_call, events_sms = dm.format_data_CNS( root_data, loadflag )
	#SMS data (from Wu et al. study)
	# dm.format_data_SMS( root_data )
	#(AskUbuntu, MathOverflow, SuperUser) from Q&A websites
	# dm.format_data_QA( root_data )
	#txt-based datasets (email and college data from SNAP)
	# dm.format_data_txt( root_data )
	#Enron data (from KONECT repository)
	# dm.format_data_Enron( root_data )

	## analysis 2: get ego network properties for all datasets ##

	#SMALL DATASETS
	for dataname, eventname in datasets: #loop through considered datasets
		print( 'dataset name: ' + eventname[:-4] ) #print output

		#prepare ego network properties / alter activities
		egonet_props, egonet_acts = dm.egonet_props_acts( dataname, eventname, root_data, 'n', saveloc )
	#
	# #LARGE DATASETS
	# datasets = [ ( 'divided_to_roughly_40_mb_files_30_march', sys.argv[1] ) ]
	# # datasets = [ ('MPC_UEu_sample', 'text') ]
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


	## analysis 3: get parameters for all datasets ##

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


	# ## analysis 5: fit activity model to ego networks in all datasets ##
	#
	# #SMALL DATASETS
	# dataname = sys.argv[1]
	# eventname = sys.argv[2]
	# # #LARGE DATASETS
	# # dataname = '' #not needed for loading
	# # eventname = sys.argv[1][15:] #i.e. 'text_1000_1020405.pkl'
	#
	# print( 'event name: ' + eventname, flush=True ) #print output
	#
	# #fit activity model to all ego networks in dataset
	# egonet_fits = dm.egonet_fits( dataname, eventname, root_data, loadflag, saveloc, nsims=300 )


	# ## analysis 6: join ego network properties and fits for large dataset separated into several files
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


	# ## analysis 8: build weighted graph from event list in all datasets ##
	#
	# # dataname = sys.argv[1] #considered dataset
	# # eventname = sys.argv[2]
	# # datasets = [ ('Copenhagen_nets', 'CNS_bt_symmetric.evt') ]
	# max_iter = 1000 #max number of iteration for centrality calculations
	#
	# for dataname, eventname in datasets: #loop through considered datasets
	# 	print( 'dataset name: ' + eventname[:-4], flush=True ) #print output
	#
	# 	#build weighted graph from event list in dataset
	# 	graph = dm.graph_weights( dataname, eventname[:-4], root_data, 'y', saveloc )
	#
	# 	#get graph properties for dataset
	# 	graph_props = dm.graph_props( eventname[:-4], loadflag, saveloc, max_iter=max_iter )


	## analysis 9: compute connection kernel for all ego networks in all datasets

	# #SMALL DATASETS
	# for dataname, eventname in datasets: #loop through datasets
	# 	print( 'dataset name: ' + eventname[:-4], flush=True ) #print output
	# 	egonet_kernel = dm.egonet_kernel( dataname, eventname[:-4], root_data, loadflag, saveloc )

	# #LARGE DATASETS
	# dataname = 'divided_to_roughly_40_mb_files_30_march/'
	#
	# eventname = 'call' #considered dataset
	# filename = sys.argv[1][20:-4]+'.txt' #i.e. '1000_1020405.txt'
	# #filename = '2367333_2387126.txt'
	# print( 'eventname: {}, filename: {}'.format( eventname, filename ), flush=True ) #print output
	# #fit activity model to all ego networks in dataset
	# dm.egonet_kernel_parallel( filename, dataname, eventname, root_data, saveloc )
	#
	# eventname = sys.argv[1] #considered dataset
	# print( 'eventname: {}'.format( eventname ), flush=True ) #print output
	# #join connection kernels for all ego networks in dataset
	# egonet_kernel = dm.egonet_kernel_join( dataname, eventname, root_data, loadflag, saveloc )


	# # ## analysis 10: perform node percolation by property on all datasets
	#
	# prop_names = ['degree', 'strength', 'act_avg', 'act_min', 'act_max', 'alpha', 'gamma', 'beta'] #props to consider
	# alphamax = 1000 #maximum alpha for MLE fit
	# pval_thres = 0.1 #threshold above which alphas are considered
	# alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf
	# ntimes=100 #realizations for random percolation
	#
	# # datasets = [ ('Copenhagen_nets', 'CNS_calls.evt') ]
	# for dataname, eventname in datasets: #loop through datasets
	# 	print( 'dataset name: ' + eventname[:-4], flush=True ) #print output
	#
	# 	graph_percs = dm.graph_percs( eventname[:-4], loadflag, saveloc, prop_names=prop_names, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres, ntimes=ntimes )


	## analysis 11: get ego network properties per time period in all datasets ##

	# #SMALL DATASETS
	# # datasets = [ ('Copenhagen_nets', 'CNS_calls.evt') ]
	# for dataname, eventname in datasets: #loop through considered datasets
	# 	print( 'dataset name: ' + eventname[:-4] ) #print output
	#
	# 	#prepare ego network properties / alter activities
	# 	egonet_props_pieces, egonet_acts_pieces = dm.egonet_props_acts_pieces( dataname, eventname, root_data, loadflag, saveloc )
	#
	# #LARGE DATASETS
	# dataname = 'divided_to_roughly_40_mb_files_30_march/'
	# eventname = 'call' #considered dataset
	# filename = sys.argv[1][20:-4]+'.txt' #i.e. '1000_1020405.txt'
	# # filename = '2367333_2387126.txt'
	# print( 'eventname: {}, filename: {}'.format( eventname, filename ), flush=True ) #print output
	# #fit activity model to all ego networks in dataset
	# dm.egonet_props_acts_pieces_parallel( filename, dataname, eventname, root_data, saveloc )


	## analysis 12: fit activity model to ego networks per time period in all datasets ##

	# #SMALL DATASETS
	# dataname = sys.argv[1]
	# eventname = sys.argv[2]
	# piece = int( sys.argv[3] ) #chosen time period (=0,1)
	# nsims = int( sys.argv[4] ) #realizations for fit bootstrapping
	# print( 'event name: {}, time period (0/1): {}'.format( eventname, piece ), flush=True ) #print output
	# #fit activity model to all ego networks (for selected time period) in dataset
	# egonet_fits_piece = dm.egonet_fits_piece( dataname, eventname, piece, root_data, loadflag, saveloc, nsims=nsims )
	#
	# #LARGE DATASETS
	# dataname = '' #not needed for loading
	# eventname = sys.argv[1] + '_' + sys.argv[4][20:-4]+'.txt' #i.e. 'text_1000_1020405.txt'
	# # eventname = 'text_2367333_2387126.txt'
	# piece = int( sys.argv[2] ) #chosen time period (=0,1)
	# nsims = int( sys.argv[3] ) #realizations for fit bootstrapping
	# print( 'eventname: {}'.format(eventname), flush=True ) #print output
	# #fit activity model to all ego networks (for selected time period) in dataset
	# egonet_fits_piece = dm.egonet_fits_piece( dataname, eventname, piece, root_data, loadflag, saveloc, nsims=nsims )


	## analysis 13: join ego network properties / fits / Jaccard indices for periods in large dataset separated into several files

	# for dataname, eventname in datasets: #loop through datasets
	# 	print( 'dataset name: ' + eventname, flush=True ) #print output
	# 	dm.egonet_props_fits_pieces_parallel( dataname, eventname, root_data, saveloc )
	# 	dm.egonet_jaccard_parallel( dataname, eventname, root_data, saveloc )
