#! /usr/bin/env python

### MODULE FOR MISCELLANEOUS FUNCTIONS FOR DATA IN FARSIGNATURES PROJECT ###

#import modules
import numpy as np
import pandas as pd

import model_misc as mm


## FUNCTIONS ##

#function to get ego network properties and alter activities for dataset
def egonet_props_acts( dataname, eventname, root_data, loadflag, saveloc ):
	"""Get ego network properties and alter activities for dataset"""

	savenames = ( saveloc + 'egonet_props_' + eventname[:-4] + '.pkl',
				  saveloc + 'egonet_acts_' + eventname[:-4] + '.pkl' )

	if loadflag == 'y': #load files
		egonet_props = pd.read_pickle( savenames[0] )
		egonet_acts = pd.read_pickle( savenames[1] )

	elif loadflag == 'n': #or else, compute them

		names = ['nodei', 'nodej', 'tstamp'] #column names

		#load (unique) event list: node i, node j, timestamp
		filename = root_data + dataname + '/data_formatted/' + eventname
		events = pd.read_csv( filename, sep=';', header=None, names=names )

		#reverse nodes i, j in all events
		events_rev = events.rename( columns={ 'nodei':'nodej', 'nodej':'nodei' } )[ names ]
		#duplicate events (all nodes with events appear as node i)
		events_concat = pd.concat([ events, events_rev ])

		#ego degrees: get neighbor lists (grouping by node i and getting unique nodes j)
		neighbors = events_concat.groupby('nodei')['nodej'].unique()
		degrees = neighbors.apply( len )

		#mean alter activity: first get number of events per ego (tau)
		num_events = events_concat.groupby('nodei')['tstamp'].size()
		actmeans = num_events / degrees	#and then mean activity as avg number of events per alter

		#alter activity: count number of events per alter of each ego
		egonet_acts = events_concat.groupby(['nodei', 'nodej']).size()

		#min/max activity: get min/max activity across alters for each ego
		amins = egonet_acts.groupby('nodei').apply( min )
		amaxs = egonet_acts.groupby('nodei').apply( max )

		#dataframe with all ego network properties
		columns = { 'nodej' : 'degree', 'tstamp' : 'strength', 0 : 'act_avg', 1 : 'act_min', 2 : 'act_max' }
		egonet_props = pd.concat( [ degrees, num_events, actmeans, amins, amaxs ], axis=1 ).rename( columns=columns )

		#save everything
		egonet_props.to_pickle( savenames[0] )
		egonet_acts.to_pickle( savenames[1] )

	return egonet_props, egonet_acts


#function to get parameters for all datasets
def data_params( datasets, root_data, loadflag, saveloc ):
	"""Get parameters for all datasets"""

	#get dataframe of parameters for all datasets
	savename = saveloc + 'params_data.pkl' #savename
	if loadflag == 'y': #load file
		params_data = pd.read_pickle( savename )

	elif loadflag == 'n': #or else, compute parameters

		#initialise dataframe of parameters for datasets
		params_data = pd.DataFrame( np.zeros( ( len(datasets), 5 ) ), index=pd.Series( [ dset[1][:-4] for dset in datasets ], name='dataset') , columns=pd.Series( [ 'num_egos', 'num_events', 'avg_degree', 'avg_strength', 'avg_activity' ], name='parameter' ) )

		for dataname, eventname in datasets: #loop through considered datasets
			print( 'dataset name: ' + eventname[:-4] ) #print output

			#prepare ego network properties
			egonet_props, egonet_acts = egonet_props_acts( dataname, eventname, root_data, 'y', saveloc )
			degrees, actmeans = egonet_props['degree'], egonet_props['act_avg']

			#save dataset parameters
			params_data.at[ eventname[:-4], 'num_egos' ] = len( degrees )
			params_data.at[ eventname[:-4], 'num_events' ] = ( degrees * actmeans ).sum() / 2 #divide by 2 (since events are counted twice per ego/alter pair)
			params_data.at[ eventname[:-4], 'avg_degree' ] = degrees.mean()
			params_data.at[ eventname[:-4], 'avg_strength' ] = ( degrees * actmeans ).mean()
			params_data.at[ eventname[:-4], 'avg_activity' ] = actmeans.mean()

			#fix dtypes
			params_data = params_data.astype({ 'num_egos' : int, 'num_events' : int })

		params_data.to_pickle( savename ) #save dataframe to file

	return params_data


#function to fit activity model to all ego networks in dataset
def egonet_fits( dataname, eventname, root_data, loadflag, saveloc, bounds=(0, 1000), nsims=2500, amax=10000 ):
	"""Fit activity model to all ego networks in dataset"""

	savename = saveloc + 'egonet_fits_' + eventname[:-4] + '.pkl'

	if loadflag == 'y': #load files
		egonet_fits = pd.read_pickle( savename )

	elif loadflag == 'n': #or else, compute everything

		rng = np.random.default_rng() #initialise random number generator

		#prepare ego network properties
		egonet_props, egonet_acts = egonet_props_acts( dataname, eventname, root_data, 'y', saveloc )
		degrees, num_events, actmeans, amins = egonet_props['degree'], egonet_props['strength'], egonet_props['act_avg'], egonet_props['act_min'] #unpack props
		num_egos = len(egonet_props) #number of egos

		#initialise dataframe of alpha fits, KS statistics and p-values (to NaNs!)
		egonet_fits = pd.DataFrame( np.zeros( ( num_egos, 3 ) )*np.nan, index=egonet_props.index, columns=pd.Series( [ 'alpha', 'statistic', 'pvalue' ], name='measure' ) )

		for pos, nodei in enumerate( egonet_props.index ): #loop through egos
			if pos % 10 == 0: #to know where we stand
				print( 'ego {} out of {}'.format( pos, num_egos ) )

			#parameters in data
			k = degrees[nodei] #degree
			tau = num_events[nodei] #strength (number of events)
			t = actmeans[nodei] #mean alter activity
			a0 = amins[nodei] #minimum alter activity

			#only do fitting for egos with t > a_0
			if k * a0 < tau:
				#alter activity in data
				activity = egonet_acts[nodei]

				#alpha fit and KS statistic for data
				alpha, KSstat = mm.alpha_KSstat( activity, bounds=bounds )

				#theo activity dist in range a=[0, amax] (i.e. inclusive)
				act_dist_theo = [ mm.activity_dist( a, t, alpha, a0 ) for a in range(amax+1) ]

				#simulations of alter activity from data fit
				activity_sims = rng.choice( amax+1, (nsims, k), p=act_dist_theo )

				#KS statistics of alpha fit for simulated activity (get alpha but leave out!)
				KSstat_sims = np.array([ mm.alpha_KSstat( act, bounds=bounds )[1] for act in activity_sims ])

				#get p-values as fraction of sim KS stats LARGER than data KS stat
				pvalue = ( KSstat_sims > KSstat ).sum() / nsims

				#save results
				egonet_fits.loc[nodei] = alpha, KSstat, pvalue

		egonet_fits.to_pickle( savename ) #save dataframe to file

	return egonet_fits


#function to format data (Bluetooth, Call, SMS) from Copenhagen Networks Study
def format_data_CNS( root_data, loadflag ):
	"""Format data (Bluetooth, Call, SMS) from Copenhagen Networks Study"""

	folder = 'Copenhagen_nets' #folder for whole dataset
	saveloc = root_data + folder + 'data_formatted/' #location of output event files
	names = [ 'nodei', 'nodej', 'tstamp' ] #column names

#BLUETOOTH DATA

	eventname = 'bt_symmetric' #name of event: proximity by bluetooth signal
	savename = saveloc + 'CNS_' + eventname + '.evt' #event filename

	if loadflag == 'y': #load file
		events_bt = pd.read_csv( savename, sep=';', header=None, names=names )

	elif loadflag == 'n': #or else, format data

		#load raw data: timestamp, user_a, user_b, rssi
		filename = root_data + folder + '/data_original/' + eventname
		events_raw = pd.read_csv( filename+'.csv' )

		#remove empty scans & users from outside the experiment (user_b = -1, -2)
		events_exp = events_raw.drop( events_raw[ events_raw.user_b.isin([ -1, -2 ]) ].index )

		#reorder/rename columns and reset index
		events_bt = events_exp[[ 'user_a', 'user_b', '# timestamp' ]].rename( columns={ 'user_a' : names[0], 'user_b' : names[1], '# timestamp' : names[2] } ).reset_index( drop=True )

		#save event file (no header/index)
		events_bt.to_csv( savename, sep=';', header=False, index=False )

#CALL DATA

	eventname = 'calls' #name of event: phone calls
	savename = saveloc + 'CNS_' + eventname + '.evt' #event filename

	if loadflag == 'y': #load file
		events_call = pd.read_csv( savename, sep=';', header=None, names=names )

	elif loadflag == 'n': #or else, format data

		#load raw data: timestamp, caller, callee, duration
		filename = root_data + folder + '/data_original/' + eventname
		events_raw = pd.read_csv( filename+'.csv' )

		#remove missed calls
		events_exp = events_raw.drop( events_raw[ events_raw.duration.isin([ -1 ]) ].index )

		#reorder/rename columns and reset index
		events_call = events_exp[[ 'caller', 'callee', 'timestamp' ]].rename( columns={ 'caller' : names[0], 'callee' : names[1], 'timestamp' : names[2] } ).reset_index( drop=True )

		#save event file (no header/index)
		events_call.to_csv( savename, sep=';', header=False, index=False )

#SMS DATA

	eventname = 'sms' #name of event: SMS
	savename = saveloc + 'CNS_' + eventname + '.evt' #event filename

	if loadflag == 'y': #load file
		events_sms = pd.read_csv( savename, sep=';', header=None, names=names )

	elif loadflag == 'n': #or else, format data

		#load raw data: timestamp, sender, recipient
		filename = root_data + folder + '/data_original/' + eventname
		events_raw = pd.read_csv( filename+'.csv' )

		#reorder/rename columns and reset index
		events_sms = events_raw[[ 'sender', 'recipient', 'timestamp' ]].rename( columns={ 'sender' : names[0], 'recipient' : names[1], 'timestamp' : names[2] } ).reset_index( drop=True )

		#save event file (no header/index)
		events_sms.to_csv( savename, sep=';', header=False, index=False )


	return events_bt, events_call, events_sms


#DEBUGGIN'

		# nodei_list = ego_acts.index.get_level_values(0).unique() #get list of nodes i
		#
		# alphas = pd.Series( 0., index=nodei_list ) #initialise alpha array
		#
		# for nodei in nodei_list: #loop through egos
		# 	if nodei % 1000 == 0:
		# 		print( 'nodei = {}'.format( nodei ) ) #to know where we stand
		#
		# 	activity = ego_acts[ nodei ] #alter activity
		# 	alphas[ nodei ] = mm.alpha_MLE_fit( activity, bounds ) #get alpha for ego (within bounds) (go from mp.mpf to float)

# #function to get ego network degrees for dataset
# def egonet_degrees( dataname, eventname, root_data, loadflag, saveloc ):
# 	"""Get ego network degrees for dataset"""
#
# 	savename = saveloc + 'degrees_' + eventname[:-4] + '.pkl'
#
# 	if loadflag == 'y': #load file
# 		degrees = pd.read_pickle( savename )
#
# 	elif loadflag == 'n': #or else, compute degrees
#
# 		names = ['nodei', 'nodej', 'tstamp'] #column names
#
# 		#load (unique) event list: node i, node j, timestamp
# 		filename = root_data + dataname + '/data_formatted/' + eventname
# 		events = pd.read_csv( filename, sep=';', header=None, names=names )
#
# 		#reverse nodes i, j in all events
# 		events_rev = events.rename( columns={ 'nodei':'nodej', 'nodej':'nodei' } )[ names ]
#
# 		#duplicate events (all nodes with events appear as node i)
# 		events_concat = pd.concat([ events, events_rev ])
#
# 		#get neighbor lists (grouping by node i and getting unique nodes j)
# 		neighbors = events_concat.groupby('nodei')['nodej'].unique()
#
# 		degrees = neighbors.apply( len ) #get node degrees
#
# 		degrees.to_pickle( savename ) #save results
#
# 	return degrees
#
#
# #function to get ego network mean activities for dataset
# def egonet_actmeans( dataname, eventname, root_data, loadflag, saveloc ):
# 	"""Get ego network mean activities for dataset"""
#
# 	savename = saveloc + 'actmeans_' + eventname[:-4] + '.pkl'
#
# 	if loadflag == 'y': #load file
# 		actmeans = pd.read_pickle( savename )
#
# 	elif loadflag == 'n': #or else, compute mean activities
#
# 		names = ['nodei', 'nodej', 'tstamp'] #column names
#
# 		#load (unique) event list: node i, node j, timestamp
# 		filename = root_data + dataname + '/data_formatted/' + eventname
# 		events = pd.read_csv( filename, sep=';', header=None, names=names )
#
# 		#reverse nodes i, j in all events
# 		events_rev = events.rename( columns={ 'nodei':'nodej', 'nodej':'nodei' } )[ names ]
#
# 		#duplicate events (all nodes with events appear as node i)
# 		events_concat = pd.concat([ events, events_rev ])
#
# 		#get neighbor lists (grouping by node i and getting unique nodes j)
# 		neighbors = events_concat.groupby('nodei')['nodej'].unique()
#
# 		degrees = neighbors.apply( len ) #get node degrees
#
# 		#get number of events per ego (tau)
# 		num_events = events_concat.groupby('nodei')['tstamp'].size()
#
# 		#get mean activity as avg number of events per alter
# 		actmeans = num_events / degrees
#
# 		actmeans.to_pickle( savename ) #save results
#
# 	return actmeans
#
#
# #function to get ego network alpha MLEs for dataset
# def egonet_alphas( dataname, eventname, bounds, root_data, loadflag, saveloc ):
# 	"""Get ego network alpha MLEs for dataset"""
#
# 	savename = saveloc + 'alphas_' + eventname[:-4] + '.pkl'
#
# 	if loadflag == 'y': #load file
# 		alphas = pd.read_pickle( savename )
#
# 	elif loadflag == 'n': #or else, compute alpha MLEs
#
# 		names = ['nodei', 'nodej', 'tstamp'] #column names
#
# 		#load (unique) event list: node i, node j, timestamp
# 		filename = root_data + dataname + '/data_formatted/' + eventname
# 		events = pd.read_csv( filename, sep=';', header=None, names=names )
#
# 		#reverse nodes i, j in all events
# 		events_rev = events.rename( columns={ 'nodei':'nodej', 'nodej':'nodei' } )[ names ]
#
# 		#duplicate events (all nodes with events appear as node i)
# 		events_concat = pd.concat([ events, events_rev ])
#
# 		#count number of events per alter of each ego (alter activity)
# 		ego_acts = events_concat.groupby(['nodei', 'nodej']).size()
#
# 		#compute MLE optimal alpha per ego, within bounds
# 		alphas = ego_acts.groupby('nodei').apply( mm.alpha_MLE_fit, bounds=bounds )
#
# 		alphas.to_pickle( savename ) #save results
#
# 	return alphas

#		egonet_props = { 'degrees' : degrees, 'actmeans' : actmeans, 'ego_acts' : ego_acts, 'alphas' : alphas }
#		egonet_props.to_pickle( savename )

# egonet_props = egonet_props( dataname, eventname, bounds, root_data, 'y', saveloc )

# #prepare ego network properties
# egonet_props = egonet_props( dataname, eventname, bounds, root_data, 'y', saveloc )

#			KSstat_sims = []
#			for act in activity_sims:
#				print( act )
#				KSstat_sims.append( mm.alpha_KSstat( act, bounds=bounds )[1] )
#			KSstat_sims = np.array( KSstat_sims )
