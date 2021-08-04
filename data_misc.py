#! /usr/bin/env python

### MODULE FOR MISCELLANEOUS FUNCTIONS FOR DATA IN FARSIGNATURES PROJECT ###

#import modules
import os
import numpy as np
import pandas as pd
#import graph_tool.all as gt

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

def egonet_props_acts_parallel( filename, fileloc, eventname, loadflag, saveloc ):
	"""Get ego network properties and alter activities for large dataset separated into several files"""

	savenames = ( saveloc + 'egonet_props_' + eventname +'_'+ filename[:-4] + '.pkl',
				  saveloc + 'egonet_acts_' + eventname +'_'+ filename[:-4] + '.pkl' )

	if loadflag == 'y': #load files
		egonet_props = pd.read_pickle( savenames[0] )
		egonet_acts = pd.read_pickle( savenames[1] )

	elif loadflag == 'n': #or else, compute them

		#load event list:
		#ego_ID alter_ID timestamp comunication_type duration
		events = pd.read_csv( fileloc + filename, sep=' ' )

		#ego degrees: get neighbor lists (grouping by ego_ID and getting unique alter_ID values)
		neighbors = events.groupby('ego_ID')['alter_ID'].unique()
		degrees = neighbors.apply( len )

		#mean alter activity: first get number of events per ego_ID (tau)
		num_events = events.groupby('ego_ID')['timestamp'].size()
		actmeans = num_events / degrees	#and then mean activity as avg number of events per alter

		#alter activity: count number of events per alter of each ego
		egonet_acts = events.groupby(['ego_ID', 'alter_ID']).size()

		#min/max activity: get min/max activity across alters for each ego
		amins = egonet_acts.groupby('ego_ID').apply( min )
		amaxs = egonet_acts.groupby('ego_ID').apply( max )

		#dataframe with all ego network properties
		columns = { 'alter_ID' : 'degree', 'timestamp' : 'strength', 0 : 'act_avg', 1 : 'act_min', 2 : 'act_max' }
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
		params_data = pd.DataFrame( np.zeros( ( len(datasets), 7 ) ), index=pd.Series( [ dset[1][:-4] for dset in datasets ], name='dataset') , columns=pd.Series( [ 'num_egos', 'num_events', 'avg_degree', 'avg_strength', 'avg_activity', 'avg_actmin', 'avg_actmax' ], name='parameter' ) )

		for dataname, eventname in datasets: #loop through considered datasets
			print( 'dataset name: ' + eventname[:-4] ) #print output

			#prepare ego network properties
			egonet_props, egonet_acts = egonet_props_acts( dataname, eventname, root_data, 'y', saveloc )
#			degrees, actmeans = egonet_props['degree'], egonet_props['act_avg']

			#save dataset parameters
			params_data.at[ eventname[:-4], 'num_egos' ] = len( egonet_props )
			params_data.at[ eventname[:-4], 'num_events' ] = ( egonet_props.degree * egonet_props.act_avg ).sum() / 2 #divide by 2 (since events are counted twice per ego/alter pair)
			params_data.at[ eventname[:-4], 'avg_degree' ] = egonet_props.degree.mean()
			params_data.at[ eventname[:-4], 'avg_strength' ] = ( egonet_props.degree * egonet_props.act_avg ).mean()
			params_data.at[ eventname[:-4], 'avg_activity' ] = egonet_props.act_avg.mean()
			params_data.at[ eventname[:-4], 'avg_actmin' ] = egonet_props.act_min.mean()
			params_data.at[ eventname[:-4], 'avg_actmax' ] = egonet_props.act_max.mean()

			#fix dtypes
			params_data = params_data.astype({ 'num_egos' : int, 'num_events' : int })

		params_data.to_pickle( savename ) #save dataframe to file

	return params_data


#function to get parameters for all large datasets separated into several files
def data_params_parallel( datasets, root_data, loadflag, saveloc ):
	"""Get parameters for all large datasets separated into several files"""

	#get dataframe of parameters for all large datasets
	savename = saveloc + 'params_data_parallel.pkl' #savename
	if loadflag == 'y': #load file
		params_data = pd.read_pickle( savename )

	elif loadflag == 'n': #or else, compute parameters

		#initialise dataframe of parameters for large datasets
		params_data = pd.DataFrame( np.zeros( ( len(datasets), 7 ) ), index=pd.Series( [ dset[1] for dset in datasets ], name='dataset') , columns=pd.Series( [ 'num_egos', 'num_events', 'avg_degree', 'avg_strength', 'avg_activity', 'avg_actmin', 'avg_actmax' ], name='parameter' ) )

		for dataname, eventname in datasets: #loop through considered large datasets
			print( 'dataset name: ' + eventname, flush=True ) #print output
			fileloc = root_data + dataname +'/'+ eventname + '/'
			filelist = os.listdir( fileloc )

			for filepos, filename in enumerate( filelist ): #loop through files in data directory
				#prepare ego network properties (for piece of large dataset!)
				egonet_props_piece, not_used = egonet_props_acts_parallel( filename, fileloc, eventname, 'y', saveloc )

				if filepos: #accumulate pieces of large dataset
					egonet_props = pd.concat([ egonet_props, egonet_props_piece ])
				else: #and initialise dataframe
					egonet_props = egonet_props_piece
			egonet_props.sort_index() #sort ego indices

			#save dataset parameters
			params_data.at[ eventname, 'num_egos' ] = len( egonet_props )
			params_data.at[ eventname, 'num_events' ] = egonet_props.strength.sum() #events are only counted once per ego/alter pair!
			params_data.at[ eventname, 'avg_degree' ] = egonet_props.degree.mean()
			params_data.at[ eventname, 'avg_strength' ] = egonet_props.strength.mean()
			params_data.at[ eventname, 'avg_activity' ] = egonet_props.act_avg.mean()
			params_data.at[ eventname, 'avg_actmin' ] = egonet_props.act_min.mean()
			params_data.at[ eventname, 'avg_actmax' ] = egonet_props.act_max.mean()

			#fix dtypes
			params_data = params_data.astype({ 'num_egos' : int, 'num_events' : int })

		params_data.to_pickle( savename ) #save dataframe to file

	return params_data


#function to build weighted graph from event list in dataset
def graph_weights( dataname, eventname, root_data, loadflag, saveloc ):
	"""Build weighted graph from event list in dataset"""

	savename = saveloc + 'graph_' + eventname[:-4] + '.gt'

	if loadflag == 'y': #load file
		graph = gt.load_graph( savename )

	elif loadflag == 'n': #or else, compute

		names = ['nodei', 'nodej', 'tstamp'] #column names

		#load (unique) event list: node i, node j, timestamp
		filename = root_data + dataname + '/data_formatted/' + eventname
		events = pd.read_csv( filename, sep=';', header=None, names=names )

		#create edge list (as np array): ( node i, node j, weight )
		#weight as alter activity: count number of events per alter of each ego
		edge_list = events.groupby(['nodei', 'nodej']).size().reset_index( name='weight' ).to_numpy()

		#initialise graph stuff
		graph = gt.Graph( directed=False ) #initialise (undirected) graph
		eweight = graph.new_ep( 'long' ) #initialise weight as (int) edge property map

		#add (hashed) edge list (with weights) and get vertex indices as vertex property map
		vid = graph.add_edge_list( edge_list, hashed=True, eprops=[ eweight ] )

		#set internal property maps
		graph.vertex_properties[ 'id' ] = vid #vertex id
		graph.edge_properties[ 'weight' ] = eweight #edge weight

		graph.save( savename ) #save graph

	return graph


#function to get graph properties for dataset
def graph_props( dataname, eventname, root_data, loadflag, saveloc, max_iter=1000 ):
	"""Get graph properties for dataset"""

	savename = saveloc + 'graph_props_' + eventname[:-4] + '.pkl'

	if loadflag == 'y': #load file
		graph_props = pd.read_pickle( savename )

	elif loadflag == 'n': #or else, compute

		#prepare ego network properties & weighted graph
		egonet_props, egonet_acts = egonet_props_acts( dataname, eventname, root_data, 'y', saveloc )
		graph = graph_weights( dataname, eventname, root_data, 'y', saveloc )

		#calculate properties of all egos

		print('\tmeasure: clustering')
		vclustering = gt.local_clustering( graph, weight=graph.ep.weight ) #local clustering coefficient

		print('\tmeasure: pagerank')
		vpagerank, niter = gt.pagerank( graph, weight=graph.ep.weight, max_iter=max_iter, ret_iter=True ) #PageRank centrality
		print( '\t\titerations: {}'.format(niter) )

		print('\tmeasure: betweenness')
		vbetweenness, ebetweenness = gt.betweenness( graph, weight=graph.ep.weight ) #betweenness centrality

		print('\tmeasure: closeness')
		vcloseness = gt.closeness( graph, weight=graph.ep.weight ) #closeness centrality

		print('\tmeasure: eigenvector')
		eig_A, veigenvector = gt.eigenvector( graph, weight=graph.ep.weight, max_iter=max_iter ) #eigenvector centrality

		print('\tmeasure: Katz')
		vkatz = gt.katz( graph, alpha=1/eig_A, weight=graph.ep.weight, max_iter=max_iter ) #eigenvector centrality

		print('\tmeasure: HITS')
		eig_AAt, vauthority, vhub = gt.hits( graph, weight=graph.ep.weight, max_iter=max_iter ) #HITS centrality

		#accumulate property names/measures
		columns = [ 'clustering', 'pagerank', 'betweenness', 'closeness', 'eigenvector', 'katz', 'authority', 'hub' ]
		vprops = [ graph.vp.id, vclustering, vpagerank, vbetweenness, vcloseness, veigenvector, vkatz, vauthority, vhub ] #first one is nodei (vertex id)

		#get vertex (ego) list with properties as array
		varray = graph.get_vertices( vprops=vprops )

		#re-organize as dataframe
		graph_props = pd.DataFrame( varray[:, 2:], index=pd.Series( varray[:, 1].astype(int), name='nodei' ), columns=columns )
		graph_props = graph_props.reindex( egonet_props.index ) #re-index (just in case)

		graph_props.to_pickle( savename ) #save dataframe to file

	return graph_props


#function to fit activity model to all ego networks in dataset
def egonet_fits( dataname, eventname, root_data, loadflag, saveloc, alphamax=1000, nsims=2500, amax=10000 ):
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
				print( 'ego {} out of {}'.format( pos, num_egos ), flush=True )

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
				alpha, KSstat = mm.alpha_KSstat( activity, alphamax=alphamax )

				#theo activity dist in range a=[0, amax] (i.e. inclusive)
				act_dist_theo = np.array([ mm.activity_dist( a, t, alpha, a0 ) for a in range(amax+1) ])
				act_dist_theo = act_dist_theo / act_dist_theo.sum() #normalise if needed

				#simulations of alter activity from data fit
				activity_sims = rng.choice( amax+1, (nsims, k), p=act_dist_theo )

				#KS statistics of alpha fit for simulated activity (get alpha but leave out!)
				KSstat_sims = np.array([ mm.alpha_KSstat( act, alphamax=alphamax )[1] for act in activity_sims ])

				#get p-values as fraction of sim KS stats LARGER than data KS stat
				pvalue = ( KSstat_sims > KSstat ).sum() / nsims

				#save results
				egonet_fits.loc[nodei] = alpha, KSstat, pvalue

		egonet_fits.to_pickle( savename ) #save dataframe to file

	return egonet_fits


#function to join ego network properties and fits for large dataset separated into several files
def egonet_props_fits_parallel( dataname, eventname, root_data, loadflag, saveloc ):
	"""Join ego network properties and fits for large dataset separated into several files"""

	savenames = ( saveloc + 'egonet_props_' + eventname + '.pkl',
				  saveloc + 'egonet_fits_' + eventname + '.pkl' )

	if loadflag == 'y': #load files
		egonet_props = pd.read_pickle( savenames[0] )
		egonet_fits = pd.read_pickle( savenames[1] )

	elif loadflag == 'n': #or else, compute them

		fileloc = root_data + dataname +'/'+ eventname + '/'
		filelist = os.listdir( fileloc )

		for filepos, filename in enumerate( filelist ): #loop through files in data directory
			fnamend = eventname +'_'+ filename[:-4] + '.pkl' #end of filename

			#prepare ego network properties/fits (for piece of large dataset!)
			egonet_props_piece = pd.read_pickle( saveloc + 'egonet_props_' + fnamend )

			try: #handling missing fit data...
				egonet_fits_piece = pd.read_pickle( saveloc + 'egonet_fits_' + fnamend )
			except FileNotFoundError:
				print( 'file not found: {}'.format( fnamend ) )
			else:
				if filepos: #accumulate pieces of large dataset
					egonet_props = pd.concat([ egonet_props, egonet_props_piece ])
					egonet_fits = pd.concat([ egonet_fits, egonet_fits_piece ])
				else: #and initialise dataframes
					egonet_props = egonet_props_piece
					egonet_fits = egonet_fits_piece

		egonet_props.sort_index() #sort ego indices
		egonet_fits.sort_index()

		egonet_props.to_pickle( savenames[0] ) #save dataframes
		egonet_fits.to_pickle( savenames[1] )

	return egonet_props, egonet_fits


#function to filter egos according to fitting results
def egonet_filter( egonet_props, graph_props, egonet_fits, alphamax=1000, pval_thres=0.1, alph_thres=1 ):
	"""Filter egos according to fitting results"""

	#join (ego) properties and fits
	props_fits = pd.concat( [ egonet_props, graph_props, egonet_fits ], axis=1 )

	#egos with well-fitted parameters (alpha, gamma, beta)

	#step 1: egos with t > a_0
	egonet_filter = props_fits[ props_fits.degree * props_fits.act_min < props_fits.strength ]
	#step 2: egos with pvalue > threshold
	egonet_filter = egonet_filter[ egonet_filter.pvalue > pval_thres ]
	#step 3 egos with alpha < alphamax (within tolerance threshold)
	egonet_filter = egonet_filter[ egonet_filter.alpha < alphamax - alph_thres ]

	#gamma distribution quantities
	gammas = pd.Series( egonet_filter.alpha + egonet_filter.act_min, name='gamma' )
	betas = pd.Series( ( egonet_filter.act_avg - egonet_filter.act_min ) / ( egonet_filter.alpha + egonet_filter.act_min ), name='beta' )

	#add gamma quantities to [filtered] properties and fits
	egonet_filter = pd.concat( [ egonet_filter, gammas, betas ], axis=1 )

	#egos without paremeters or with alpha -> inf

	egonet_rest = props_fits.drop( egonet_filter.index )
	egonet_inf = egonet_rest[ egonet_rest.alpha > alphamax - alph_thres ]
	egonet_null = egonet_rest.drop( egonet_inf.index )

	return egonet_filter, egonet_inf, egonet_null


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


#function to fit gamma approx of activity model to all ego networks in dataset
def egonet_gammas( dataname, eventname, root_data, loadflag, saveloc ):
	"""Fit gamma approx of activity model to all ego networks in dataset"""

	savename = saveloc + 'egonet_gammas_' + eventname[:-4] + '.pkl'

	if loadflag == 'y': #load files
		egonet_gammas = pd.read_pickle( savename )

	elif loadflag == 'n': #or else, compute everything

		rng = np.random.default_rng() #initialise random number generator

		#prepare ego network properties
		egonet_props, egonet_acts = egonet_props_acts( dataname, eventname, root_data, 'y', saveloc )
		num_egos = len(egonet_props) #number of egos

		#initialise dataframe of (un-) biased gamma/beta fits, KS statistics & p-values (NaNs!)
		egonet_gammas = pd.DataFrame( np.zeros( ( num_egos, 6 ) )*np.nan, index=egonet_props.index, columns=pd.Series( [ 'gamma', 'gamma_bias', 'beta', 'beta_bias', 'statistic', 'pvalue' ], name='measure' ) )

		for pos, nodei in enumerate( egonet_props.index ): #loop through egos
			if pos % 1000 == 0: #to know where we stand
				print( 'ego {} out of {}'.format( pos, num_egos ), flush=True )

			#all alter activity
			activity = egonet_acts[nodei] #alter activity in data
			# k = activity.size #degree
			a0 = egonet_props.act_min[nodei] #minimum alter activity

			#filtered alter activity (a > a0)
			activity_noa0 = activity[ activity > a0 ] #alter activity a > a0
			k_noa0 = activity_noa0.size #degree
			tau_noa0 = activity_noa0.sum() #strength (number of events)
			amin_noa0 = activity_noa0.min() #min alter activity

			#only do fitting for egos with activity heterogeneties (after filtering!)
			if k_noa0 * amin_noa0 < tau_noa0:

				#gamma fit and KS statistic/p-value for data
				gamma, gamma_bias, beta, beta_bias, KSstat, KSpval = mm.gamma_KSstat( activity )

				# #simulations of alter activity from data fit (include a = a0)
				# activity_sims = a0 + rng.gamma( gamma, scale=beta, size=(nsims, k) ).round().astype(int)
				#
				# #KS statistics of gamma fit for simulated activity (ignore estimators!)
				# KSstat_sims = np.array([ mm.gamma_KSstat( act )[4] for act in activity_sims ])
				#
				# #get p-values as fraction of sim KS stats LARGER than data KS stat
				# pvalue = ( KSstat_sims > KSstat ).sum() / nsims

				#save results
				egonet_gammas.loc[nodei] = gamma, gamma_bias, beta, beta_bias, KSstat, KSpval

		egonet_gammas.to_pickle( savename ) #save dataframe to file

	return egonet_gammas


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
