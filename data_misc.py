#! /usr/bin/env python

### MODULE FOR MISCELLANEOUS FUNCTIONS FOR DATA IN FARSIGNATURES PROJECT ###

#import modules
import os
import numpy as np
import pandas as pd
import pickle as pk
# import graph_tool.all as gt

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

		#number of events per ego (tau)
		num_events = events_concat.groupby('nodei')['tstamp'].size()

		#alter activity: count number of events per alter of each ego
		egonet_acts = events_concat.groupby(['nodei', 'nodej']).size()
		egonet_acts_grouped = egonet_acts.groupby('nodei')

		#mean/variance of alter activity
		actmeans = egonet_acts_grouped.mean()
		actvars = egonet_acts_grouped.var() #NOTE: estimated variance (ddof=1)
		#min/max activity: get min/max activity across alters for each ego
		amins = egonet_acts_grouped.apply( min )
		amaxs = egonet_acts_grouped.apply( max )

		#dataframe with all ego network properties
		columns = { 'nodej' : 'degree', 'tstamp' : 'strength', 0 : 'act_avg', 1 : 'act_var', 2 : 'act_min', 3 : 'act_max' }
		egonet_props = pd.concat( [ degrees, num_events, actmeans, actvars, amins, amaxs ], axis=1 ).rename( columns=columns )

		#save everything
		egonet_props.to_pickle( savenames[0] )
		egonet_acts.to_pickle( savenames[1] )

	return egonet_props, egonet_acts


#function to get ego network properties and alter activities for large dataset separated into several files
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

		#number of events per ego_ID (tau)
		num_events = events.groupby('ego_ID')['timestamp'].size()

		#alter activity: count number of events per alter of each ego
		egonet_acts = events.groupby(['ego_ID', 'alter_ID']).size()
		egonet_acts_grouped = egonet_acts.groupby('ego_ID')

		#mean/variance of alter activity
		actmeans = egonet_acts_grouped.mean()
		actvars = egonet_acts_grouped.var() #NOTE: estimated variance (ddof=1)
		#min/max activity: get min/max activity across alters for each ego
		amins = egonet_acts_grouped.apply( min )
		amaxs = egonet_acts_grouped.apply( max )

		#dataframe with all ego network properties
		columns = { 'alter_ID' : 'degree', 'timestamp' : 'strength', 0 : 'act_avg', 1 : 'act_var', 2 : 'act_min', 3 : 'act_max' }
		egonet_props = pd.concat( [ degrees, num_events, actmeans, actvars, amins, amaxs ], axis=1 ).rename( columns=columns )

		#save everything
		egonet_props.to_pickle( savenames[0] )
		egonet_acts.to_pickle( savenames[1] )

	return egonet_props, egonet_acts


#function to get ego network properties and alter activities per time period in dataset
def egonet_props_acts_pieces( dataname, eventname, root_data, loadflag, saveloc ):
	"""Get ego network properties and alter activities per time period in dataset"""

	savenames = ( saveloc + 'egonet_props_pieces_' + eventname[:-4] + '.pkl',
				  saveloc + 'egonet_acts_pieces_' + eventname[:-4] + '.pkl' )

	if loadflag == 'y': #load files
		with open( savenames[0], 'rb' ) as file:
			egonet_props_pieces = pk.load( file )
		with open( savenames[1], 'rb' ) as file:
			egonet_acts_pieces = pk.load( file )

	elif loadflag == 'n': #or else, compute them

		#load (unique) event list: node i, node j, timestamp
		names = ['nodei', 'nodej', 'tstamp'] #column names
		filename = root_data + dataname + '/data_formatted/' + eventname
		events = pd.read_csv( filename, sep=';', header=None, names=names )

		#reverse nodes i, j in all events
		events_rev = events.rename( columns={ 'nodei':'nodej', 'nodej':'nodei' } )[ names ]
		#duplicate events (all nodes with events appear as node i), reset index, and sort by tstamp
		events_concat = pd.concat([ events, events_rev ]).reset_index(drop=True).sort_values( by='tstamp' )
		events_grouped = events_concat.groupby('nodei') #group events by ego

		#group events by ego and separate into two (equal-sized) time periods
		events_pieces = [
		events_grouped.apply( lambda x : x.iloc[ :int(np.ceil(len(x)/2)) ,:] ).reset_index(drop=True),
		events_grouped.apply( lambda x : x.iloc[ len(x) - int(np.floor(len(x)/2)): ,:] ).reset_index(drop=True)
						]

		#initialise dframes and columns
		egonet_props_pieces, egonet_acts_pieces = [], []
		columns = { 'nodej' : 'degree', 'tstamp' : 'strength', 0 : 'act_avg', 1 : 'act_min', 2 : 'act_max' }

		for pos, events_period in enumerate(events_pieces): #loop through time periods
			events_period_grouped = events_period.groupby('nodei') #group events by ego (in period)

			#ego degrees: get neighbor lists (grouping by node i and getting unique nodes j)
			neighbors = events_period_grouped['nodej'].unique()
			degrees = neighbors.apply( len )
			#mean alter activity: first get number of events per ego (tau)
			num_events = events_period_grouped['tstamp'].size()
			actmeans = num_events / degrees	#and then mean activity as avg number of events per alter

			#alter activity: count number of events per alter of each ego
			egonet_acts = events_period.groupby(['nodei', 'nodej']).size()

			#min/max activity: get min/max activity across alters for each ego
			amins = egonet_acts.groupby('nodei').apply( min )
			amaxs = egonet_acts.groupby('nodei').apply( max )

			#dataframe with all ego network properties
			egonet_props = pd.concat( [ degrees, num_events, actmeans, amins, amaxs ], axis=1 ).rename( columns=columns )

			egonet_props_pieces.append( egonet_props ) #store results
			egonet_acts_pieces.append( egonet_acts )

		#save all to file
		with open( savenames[0], 'wb' ) as file:
			pk.dump( egonet_props_pieces, file )
		with open( savenames[1], 'wb' ) as file:
			pk.dump( egonet_acts_pieces, file )

	return egonet_props_pieces, egonet_acts_pieces


#function to get ego network properties and alter activities per time period in large dataset separated into several files
def egonet_props_acts_pieces_parallel( filename, dataname, eventname, root_data, saveloc ):
	"""Get ego network properties and alter activities per time period in large dataset separated into several files"""

	#load event list: ego_ID alter_ID timestamp comunication_type duration
	events = pd.read_csv( root_data+dataname+eventname+'/'+filename, sep=' ' )
	events_grouped = events.groupby('ego_ID') #group events by ego

	#group events by ego and separate into two (equal-sized) time periods
	events_pieces = [
	events_grouped.apply( lambda x : x.iloc[ :int(np.ceil(len(x)/2)) ,:] ).reset_index(drop=True),
	events_grouped.apply( lambda x : x.iloc[ len(x) - int(np.floor(len(x)/2)): ,:] ).reset_index(drop=True)
					]

	#initialise dframes and columns
	egonet_props_pieces, egonet_acts_pieces = [], []
	columns = { 'alter_ID' : 'degree', 'timestamp' : 'strength', 0 : 'act_avg', 1 : 'act_var', 2 : 'act_min', 3 : 'act_max' }

	for pos, events_period in enumerate(events_pieces): #loop through time periods
		events_period_grouped = events_period.groupby('ego_ID') #group events by ego (in period)

		#ego degrees: get neighbor lists (grouping by node i and getting unique nodes j)
		neighbors = events_period_grouped['alter_ID'].unique()
		degrees = neighbors.apply( len )

		#mean alter activity: first get number of events per ego (tau)
		num_events = events_period_grouped['timestamp'].size()

		#alter activity: count number of events per alter of each ego
		egonet_acts = events_period.groupby(['ego_ID', 'alter_ID']).size()
		egonet_acts_grouped = egonet_acts.groupby('ego_ID')

		#mean/variance of alter activity
		actmeans = egonet_acts_grouped.mean()
		actvars = egonet_acts_grouped.var() #NOTE: estimated variance (ddof=1)
		#min/max activity: get min/max activity across alters for each ego
		amins = egonet_acts_grouped.apply( min )
		amaxs = egonet_acts_grouped.apply( max )

		#dataframe with all ego network properties
		egonet_props = pd.concat( [ degrees, num_events, actmeans, actvars, amins, amaxs ], axis=1 ).rename( columns=columns )

		egonet_props_pieces.append( egonet_props ) #store results
		egonet_acts_pieces.append( egonet_acts )

	#save all to file
	with open( saveloc+'egonet_props_pieces_'+eventname+'_'+filename[:-4]+'.pkl', 'wb' ) as file:
		pk.dump( egonet_props_pieces, file )
	with open( saveloc+'egonet_acts_pieces_'+eventname+'_'+filename[:-4]+'.pkl', 'wb' ) as file:
		pk.dump( egonet_acts_pieces, file )


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


#function to compute connection kernel for all ego networks in dataset
def egonet_kernel( dataname, eventname, root_data, loadflag, saveloc ):
	"""Compute connection kernel for all ego networks in dataset"""

	savename = saveloc + 'egonet_kernel_' + eventname + '.pkl'

	if loadflag == 'y': #load files
		egonet_kernel = pd.read_pickle( savename )

	elif loadflag == 'n': #or else, compute them
		egonet_kernel = pd.Series( dtype=float ) #initialise connection kernel for all egos

		#prepare ego network properties
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )

		#prepare raw events
		names = ['nodei', 'nodej', 'tstamp'] #column names
		#load (unique) event list: node i, node j, timestamp
		filename = root_data + dataname + '/data_formatted/' + eventname + '.evt'
		events = pd.read_csv( filename, sep=';', header=None, names=names )
		#reverse nodes i, j in all events
		events_rev = events.rename( columns={ 'nodei':'nodej', 'nodej':'nodei' } )[ names ]
		#duplicate events (all nodes with events appear as node i)
		events_concat = pd.concat([ events, events_rev ])

		#compute connection kernel for each ego network
		for pos, (nodei, events_nodei) in enumerate( events_concat.groupby('nodei') ): #loop through egos
			if pos % 100 == 0: #to know where we stand
				print( 'ego {} out of {}'.format( pos, len(egonet_props) ), flush=True )

			events_nodei = events_nodei.sort_values('tstamp') #sort by timestamp!

			#get alter list and initialise their activities
			neighs_nodei = events_nodei.nodej.unique()
			alter_acts = pd.Series( np.zeros(len(neighs_nodei)), index=neighs_nodei, dtype=int )

			#initialise kernel arrays
			act_max = egonet_props.loc[nodei, 'act_max'] #max alter activity
			act_counts = np.zeros( act_max+1, dtype=int ) #no. connections at given activity
			act_options = np.zeros( act_max+1, dtype=int ) #no. potential alters at given activity

			for nodej in events_nodei.nodej: #loop through (ordered) events
				act_options += np.bincount( alter_acts, minlength=act_max+1 ) #update no. potential alters at each activity
				act_counts[ alter_acts[nodej] ] += 1 #add connection to alter with given activity
				alter_acts[nodej] += 1 #increase activity of selected alter

			if act_options[-1] == 0: #remove last activity point if not neccesary...
				act_counts = act_counts[:-1]
				act_options = act_options[:-1]

			#store kernel as multi-index series
			index_arrs = [ nodei*np.ones(len(act_counts), dtype=int), np.arange(len(act_counts), dtype=int) ]
			egonet_kernel = egonet_kernel.append( pd.Series( act_counts / act_options.astype(float), index=index_arrs ) )

		egonet_kernel.index = pd.MultiIndex.from_tuples( egonet_kernel.index ) #format multi-index
		egonet_kernel.to_pickle( savename ) #save file

	return egonet_kernel


#function to compute connection kernel for all ego networks in large dataset separated into several files
def egonet_kernel_parallel( filename, dataname, eventname, root_data, saveloc ):
	"""Compute connection kernel for all ego networks in large dataset separated into several files"""

	egonet_kernel = pd.Series( dtype=float ) #initialise connection kernel
	egonet_props = pd.read_pickle( saveloc+'egonet_props_'+eventname+'_'+filename[:-4]+'.pkl' ) #prepare ego network properties
	#load event list: ego_ID alter_ID timestamp comunication_type duration
	events = pd.read_csv( root_data+dataname+eventname+'/'+filename, sep=' ' )

	#compute connection kernel for each ego network
	for pos, (nodei, events_nodei) in enumerate( events.groupby('ego_ID') ): #loop through egos
		if pos % 100 == 0: #to know where we stand
			print( 'ego {} out of {}'.format( pos, len(egonet_props) ), flush=True )

		events_nodei = events_nodei.sort_values('timestamp') #sort by timestamp!

		#get alter list and initialise their activities
		neighs_nodei = events_nodei.alter_ID.unique()
		alter_acts = pd.Series( np.zeros(len(neighs_nodei)), index=neighs_nodei, dtype=int )

		#initialise kernel arrays
		act_max = egonet_props.loc[nodei, 'act_max'] #max alter activity
		act_counts = np.zeros( act_max+1, dtype=int ) #no. connections at given activity
		act_options = np.zeros( act_max+1, dtype=int ) #no. potential alters at given activity

		for nodej in events_nodei.alter_ID: #loop through (ordered) events
			act_options += np.bincount( alter_acts, minlength=act_max+1 ) #update no. potential alters at each activity
			act_counts[ alter_acts[nodej] ] += 1 #add connection to alter with given activity
			alter_acts[nodej] += 1 #increase activity of selected alter

		if act_options[-1] == 0: #remove last activity point if not neccesary...
			act_counts = act_counts[:-1]
			act_options = act_options[:-1]

		#store kernel as multi-index series
		index_arrs = [ nodei*np.ones(len(act_counts), dtype=int), np.arange(len(act_counts), dtype=int) ]
		egonet_kernel = pd.concat([ egonet_kernel, pd.Series( act_counts / act_options.astype(float), index=index_arrs ) ])

	egonet_kernel.index = pd.MultiIndex.from_tuples( egonet_kernel.index ) #format multi-index
	egonet_kernel.to_pickle( saveloc+'egonet_kernel_'+eventname+'_'+filename[:-4]+'.pkl' ) #save file


#function to join connection kernel files for all ego networks in large dataset separated into several files
def egonet_kernel_join( dataname, eventname, root_data, loadflag, saveloc ):
	"""Join connection kernel files for all ego networks in large dataset separated into several files"""

	savename = saveloc + 'egonet_kernel_' + eventname + '.pkl'

	if loadflag == 'y': #load files
		egonet_kernel = pd.read_pickle( savename )

	elif loadflag == 'n': #or else, compute them
		egonet_kernel = pd.Series( dtype=float ) #initialise connection kernel

		#loop through files in data directory
		fileloc = root_data + dataname + eventname + '/'
		filelist = sorted( os.listdir( fileloc ) )
		for filepos, filename in enumerate( filelist ):
			print( 'file {} out of {}'.format( filepos, len(filelist) ), flush=True )
			fnamend = eventname +'_'+ filename[:-4] + '.pkl' #end of filename

			#prepare ego network kernels (for piece of large dataset!)
			try: #handling missing kernel data...
				egonet_kernel_piece = pd.read_pickle( saveloc + 'egonet_kernel_' + fnamend )
			except FileNotFoundError:
				print( 'file not found: {}'.format( fnamend ) )
			else: #accumulate pieces of large dataset
				egonet_kernel = pd.concat([ egonet_kernel, egonet_kernel_piece ])

		egonet_kernel.index = pd.MultiIndex.from_tuples( egonet_kernel.index ) #format multi-index
		egonet_kernel.to_pickle( savename ) #save dataframe

	return egonet_kernel


#function to get Jaccard index between neighbor sets in time periods for egos in all datasets
def egonet_jaccard( eventname, loadflag, saveloc ):
	"""Get Jaccard index between neighbor sets in time periods for egos in all datasets"""

	savename = saveloc + 'egonet_jaccard_' + eventname + '.pkl'

	if loadflag == 'y': #load file
		egonet_jaccard = pd.read_pickle( savename )

	elif loadflag == 'n': #or else, compute

		#load alter activities for all egos (for both time periods)
		egonet_acts_pieces = pd.read_pickle( saveloc + 'egonet_acts_pieces_' + eventname + '.pkl' )

		#get ego neighbor sets for both time periods and join
		egonet_neighs_0 = egonet_acts_pieces[0].groupby('nodei').apply( lambda x : set(x.index.get_level_values('nodej')) )
		egonet_neighs_1 = egonet_acts_pieces[1].groupby('nodei').apply( lambda x : set(x.index.get_level_values('nodej')) )
		egonet_neighs = pd.concat( [ egonet_neighs_0.rename('neighs_0'), egonet_neighs_1.rename('neighs_1') ], axis=1, join='inner' )

		#compute Jaccard index of neighbor sets
		jaccard_func = lambda x : len( x.neighs_0.intersection(x.neighs_1) ) / float(len( x.neighs_0.union(x.neighs_1) ))
		egonet_jaccard = egonet_neighs.apply( jaccard_func, axis=1 )

		egonet_jaccard.to_pickle( savename ) #save file

	return egonet_jaccard


#function to build weighted graph from event list in dataset
def graph_weights( dataname, eventname, root_data, loadflag, saveloc ):
	"""Build weighted graph from event list in dataset"""

	savename = saveloc + 'graph_' + eventname + '.gt'

	if loadflag == 'y': #load file
		graph = gt.load_graph( savename )

	elif loadflag == 'n': #or else, compute

		#load (unique) event list: node i, node j, timestamp
		filename = root_data + dataname + '/data_formatted/' + eventname + '.evt'
		events = pd.read_csv( filename, sep=';', header=None, names=['nodei', 'nodej', 'tstamp'] )

		#rearrange events symetrically
		events_sym = events.copy() #initialize array
		for row in events.itertuples(): #loop through events
			if row.nodei > row.nodej: #reverse links in lower-diagonal events
				events_sym.loc[ row.Index, ['nodei', 'nodej'] ] = row.nodej, row.nodei

		#create edge list (as np array): ( node i, node j, weight )
		#weight as alter activity: count number of (sym) events per alter of each ego
		edge_list = events_sym.groupby(['nodei', 'nodej']).size().reset_index( name='weight' ).to_numpy()

		#initialise graph stuff
		graph = gt.Graph( directed=False ) #initialise (undirected) graph
		eweight = graph.new_ep('int') #initialise weight as (int) edge property map

		#add (hashed) edge list (with weights) and get vertex indices as vertex property map
		vid = graph.add_edge_list( edge_list, hashed=True, eprops=[eweight] )

		#set internal property maps
		graph.vertex_properties['id'] = vid #vertex id
		graph.edge_properties['weight'] = eweight #edge weight

		graph.save( savename ) #save graph

	return graph


#function to get graph properties for dataset
def graph_props( eventname, loadflag, saveloc, max_iter=1000 ):
	"""Get graph properties for dataset"""

	savename = saveloc + 'graph_props_' + eventname + '.pkl'

	if loadflag == 'y': #load file
		graph_props = pd.read_pickle( savename )

	elif loadflag == 'n': #or else, compute

		#load ego network properties, and weighted graph
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
		graph = gt.load_graph( saveloc + 'graph_' + eventname + '.gt' )

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
def egonet_fits( dataname, eventname, root_data, loadflag, saveloc, egonet_tuple=None, alphamax=1000, nsims=2500, amax=10000 ):
	"""Fit activity model to all ego networks in dataset"""

	if egonet_tuple: #when analyzing selected time period
		savename = saveloc + 'egonet_fits_piece_{}_{}'.format( egonet_tuple[2], eventname[:-4] ) + '.pkl'
	else: #or the full dataset
		savename = saveloc + 'egonet_fits_' + eventname[:-4] + '.pkl'

	if loadflag == 'y': #load files
		egonet_fits = pd.read_pickle( savename )

	elif loadflag == 'n': #or else, compute everything

		rng = np.random.default_rng() #initialise random number generator

		#prepare ego network properties
		if egonet_tuple: #load directly from tuple
			egonet_props, egonet_acts, piece = egonet_tuple
		else: #or from function
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

			if pos % 10 == 0: #every once in a while...
				egonet_fits.to_pickle( savename ) #save dataframe to file
		egonet_fits.to_pickle( savename ) #and again (just in case)

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
		filelist = sorted( os.listdir( fileloc ) )

		for filepos, filename in enumerate( filelist ): #loop through files in data directory
			fnamend = eventname +'_'+ filename[:-4] + '.pkl' #end of filename

			#prepare ego network properties (for piece of large dataset!)
			egonet_props_piece = pd.read_pickle( saveloc + 'egonet_props_' + fnamend )
			if filepos: #accumulate pieces of large dataset
				egonet_props = pd.concat([ egonet_props, egonet_props_piece ])
			else: #and initialise dataframes
				egonet_props = egonet_props_piece

			#prepare ego network fits (for piece of large dataset!)
			try: #handling missing fit data...
				egonet_fits_piece = pd.read_pickle( saveloc + 'egonet_fits_' + fnamend )
			except FileNotFoundError:
				print( 'file not found: {}'.format( fnamend ) )
			else:
				if filepos: #accumulate pieces of large dataset
					egonet_fits = pd.concat([ egonet_fits, egonet_fits_piece ])
				else: #and initialise dataframes
					egonet_fits = egonet_fits_piece

		egonet_props.sort_index() #sort ego indices
		egonet_fits.sort_index()

		egonet_props.to_pickle( savenames[0] ) #save dataframes
		egonet_fits.to_pickle( savenames[1] )

	return egonet_props, egonet_fits


#function to join ego network properties and fits for all time periods in large dataset separated into several files
def egonet_props_fits_pieces_parallel( dataname, eventname, root_data, saveloc ):
	"""Join ego network properties and fits for all time periods in large dataset separated into several files"""

	#list of files in data directory
	fileloc = root_data + dataname +'/'+ eventname + '/'
	filelist = sorted( os.listdir( fileloc ) )

	#ego network properties

	# for filepos, filename in enumerate( filelist ): #loop through files in data directory
	# 	fnamend = eventname +'_'+ filename[:-4] + '.pkl' #end of filename
	# 	if filepos % 10 == 0: #to know where we stand
	# 		print( '\tfile {} out of {}'.format( filepos, len(filelist) ), flush=True )
	#
	# 	#prepare ego network properties in individual file
	# 	egonet_props_pieces_file = pd.read_pickle( saveloc + 'egonet_props_pieces_' + fnamend )
	# 	if filepos: #accumulate pieces of large dataset
	# 		for period in range(2): #loop through time periods
	# 			egonet_props_pieces[period] = pd.concat([ egonet_props_pieces[period], egonet_props_pieces_file[period] ])
	# 	else: #and initialise dataframes
	# 		egonet_props_pieces = egonet_props_pieces_file
	#
	# for period in range(2): #loop through time periods
	# 	egonet_props_pieces[period].sort_index() #sort ego indices
	# with open( saveloc + 'egonet_props_pieces_' + eventname + '.pkl', 'wb' ) as file: #save to file
	# 	pk.dump( egonet_props_pieces, file )

	#ego network fits

	for period in range(2): #loop through time periods
		for filepos, filename in enumerate( filelist ): #loop through files in data directory
			fnamend = 'piece_' + str(period) +'_'+ eventname +'_'+ filename[:-4] + '.pkl' #end of filename
			if filepos % 10 == 0: #to know where we stand
				print( '\tfile {} out of {}'.format( filepos, len(filelist) ), flush=True )

			#prepare ego network fits (for piece of large dataset!)
			try: #handling missing fit data...
				egonet_fits_piece_file = pd.read_pickle( saveloc + 'egonet_fits_' + fnamend )
			except FileNotFoundError:
				print( 'file not found: {}'.format( fnamend ) )
			else:
				if filepos: #accumulate pieces of large dataset
					egonet_fits_piece = pd.concat([ egonet_fits_piece, egonet_fits_piece_file ])
				else: #and initialise dataframes
					egonet_fits_piece = egonet_fits_piece_file

		egonet_fits_piece.sort_index() #sort ego indices
		egonet_fits_piece.to_pickle( saveloc + 'egonet_fits_piece_' + period + '_' + eventname + '.pkl' ) #save file


#function to fit activity model to all ego networks in selected time period in dataset
def egonet_fits_piece( dataname, eventname, piece, root_data, loadflag, saveloc, alphamax=1000, nsims=2500, amax=10000 ):
	"""Fit activity model to all ego networks per time period in dataset"""

	savename = saveloc + 'egonet_fits_piece_{}_{}'.format( piece, eventname[:-4] ) + '.pkl'

	if loadflag == 'y': #load file
		egonet_fits_piece = pd.read_pickle( savename )

	elif loadflag == 'n': #or else, compute everything

		#load ego network properties / alter activities (in selected time period)
		egonet_props_pieces, egonet_acts_pieces = egonet_props_acts_pieces( dataname, eventname, root_data, 'y', saveloc )
		egonet_tuple = ( egonet_props_pieces[piece], egonet_acts_pieces[piece], piece )

		#fit activity model to all ego networks (and save it)
		egonet_fits_piece = egonet_fits( dataname, eventname, root_data, 'n', saveloc, egonet_tuple=egonet_tuple, alphamax=alphamax, nsims=nsims, amax=amax )

	return egonet_fits_piece


#function to filter egos according to fitting results
def egonet_filter( egonet_props, egonet_fits, graph_props=None, alphamax=1000, pval_thres=0.1, alph_thres=1 ):
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


#function to get connected ego/alter activity ranks / properties for dataset
def egonet_ranks_props( eventname, loadflag, saveloc, prop_name='beta', alphamax=1000, pval_thres=0.1, alph_thres=1 ):
	"""Get connected ego/alter activity ranks / properties for dataset"""

	savename = saveloc + 'egonet_ranks_props_' + eventname + '.pkl'

	if loadflag == 'y': #load file
		egonet_ranks_props = pd.read_pickle( savename )

	elif loadflag == 'n': #or else, compute it

		## DATA ##

		#prepare ego network properties / alter activities
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
		egonet_acts = pd.read_pickle( saveloc + 'egonet_acts_' + eventname + '.pkl' )
		#fit activity model to all ego networks
		egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )

		#filter egos according to fitting results
		egonet_filt, egonet_inf, egonet_null = egonet_filter( egonet_props, egonet_fits, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )

		## HOMOPHILY ANALYSIS ##

		#initialise dataframe of connected ego/alter activity ranks / properties
		egonet_ranks_props = pd.DataFrame( np.ones(( egonet_acts.size, 4 ))*np.nan, index=egonet_acts.index, columns=pd.Series( ['ego_rank', 'alter_rank', 'ego_prop', 'alter_prop'], name='property' ) )

		for nodei in egonet_filt.index: #lopp through all egos

			#get ranks of alters of selected ego
			all_alter_ranks = egonet_acts[nodei].rank( ascending=False )

			for nodej in egonet_acts[nodei].index: #loop through alters
				if nodej in egonet_filt.index: #only for alters with alpha

					#get rank of selected ego/alter
					ego_rank = egonet_acts[nodej].rank( ascending=False )[nodei]
					alter_rank = all_alter_ranks[nodej]

					#get property of selected ego/alter
					ego_prop = egonet_filt.loc[nodei, prop_name]
					alter_prop = egonet_filt.loc[nodej, prop_name]

					#get connected ego/alter activity ranks / properties
					egonet_ranks_props.loc[nodei, nodej] = ego_rank, alter_rank, ego_prop, alter_prop
		egonet_ranks_props.dropna( inplace=True ) #keep only edges with alphas

		egonet_ranks_props.to_pickle( savename ) #save dataframe

	return egonet_ranks_props


#function to perform node percolation by property on dataset
def graph_percs( eventname, loadflag, saveloc, prop_names=['beta'], alphamax=1000, pval_thres=0.1, alph_thres=1, seed=None, ntimes=100, print_every=10 ):
	"""Perform node percolation by property on dataset"""

	savename = saveloc + 'graph_percs_' + eventname + '.pkl'

	if loadflag == 'y': #load file
		graph_percs = pd.read_pickle( savename )

	elif loadflag == 'n': #or else, compute

		## DATA ##

		#load ego network properties, and weighted graph
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
		graph = gt.load_graph( saveloc + 'graph_' + eventname + '.gt' )
		#fit activity model to all ego networks
		egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )

		#filter egos according to fitting results
		egonet_filt, egonet_inf, egonet_null = egonet_filter( egonet_props, egonet_fits, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )

		num_egos = len(egonet_props) #number of (un-)filtered egos
		num_filt = len(egonet_filt)


		## PERCOLATION ANALYSIS ##

		#initialise dataframe of percolation processes
		npercs = 2*len(prop_names)+1 #no. of percolations: large / small per prop plus random case
		index = pd.Series( range( num_egos-num_filt+1, num_egos+1 ), name='rem_nodes' ) #array of remaining (filtered) egos
		columns = pd.Series( [prop+'_large' for prop in prop_names] + [prop+'_small' for prop in prop_names] + ['random'], name='prop' ) #percolation cases
		graph_percs = pd.DataFrame( np.zeros(( num_filt, npercs ), dtype=int), index=index, columns=columns )

		#nodes NOT considered for percolation (inf+null beta classes)
		not_nodei = egonet_props.index.difference( egonet_filt.index )
		vertices_not = [ gt.find_vertex( graph, graph.vp.id, nodei )[0] for nodei in not_nodei ]

	#percolation by increasing/decreasing (non-random) property

		for prop in graph_percs.columns[:-1]: #loop through props
			print('\t'+prop, flush=True)
			ascending = True if prop[-6:] == '_large' else False #set order flag

			#delete nodes by largest/smallest prop first
			sorted_nodei = egonet_filt[ prop[:-6] ].sort_values( ascending=ascending ).index
			vertices = [ gt.find_vertex( graph, graph.vp.id, nodei )[0] for nodei in sorted_nodei ] #get nodes with increasing/decreassing prop
			sizes, comp = gt.vertex_percolation( graph, vertices_not + vertices )
			graph_percs[ prop ] = sizes[ num_egos-num_filt: ] #only store results for valid beta values

	#random node percolation

		rng = np.random.default_rng(seed) #initialise random number generator
		vertices = [ gt.find_vertex( graph, graph.vp.id, nodei )[0] for nodei in egonet_filt.index ] #get nodes with valid beta

		print('\trandom')
		for nt in range(ntimes): #loop through realizations
			if nt % print_every == 0:
				print('\t\tnt = {}'.format(nt), flush=True)

			rng.shuffle(vertices) #randomly shuffle nodes (in place)
			sizes, comp = gt.vertex_percolation( graph, vertices_not + vertices )
			graph_percs.random += sizes[ num_egos-num_filt: ] #accumulate results
		graph_percs.random /= float(ntimes) #and average

		graph_percs.to_pickle( savename ) #save dataframe

	return graph_percs


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


#function to format data (AskUbuntu, MathOverflow, SuperUser) from Q&A websites
def format_data_QA( root_data ):
	"""Format data (AskUbuntu, MathOverflow, SuperUser) from Q&A websites"""

	folder = 'QA_nets/' #folder for whole dataset

	#loop through datasets
	for eventname in [ 'askubuntu', 'mathoverflow', 'superuser' ]:
		eventname = eventname + '_all' #add bit to remember type of edge (all considered)

		#load raw data: timestamp, # source, target, timestamp
		events_raw = pd.read_csv( root_data + folder + 'data_original/' + eventname + '/edges.csv' )

		#remove self-loop events with the same user as source/target
		events = events_raw.drop( events_raw[ events_raw['# source']==events_raw[' target'] ].index )
		#rename columns and reset index
		events = events.rename( columns={ '# source' : 'nodei', ' target' : 'nodej', ' timestamp' : 'tstamp' } ).reset_index( drop=True )

		#save event file (no header/index)
		savename = root_data + folder + 'data_formatted/' + 'QA_' + eventname[:-4] + '.evt'
		events.to_csv( savename, sep=';', header=False, index=False )


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

# #function to build weighted graph from event list in dataset
# def graph_weights( dataname, eventname, root_data, loadflag, saveloc ):
# 	"""Build weighted graph from event list in dataset"""
#
# 	savename = saveloc + 'graph_' + eventname[:-4] + '.gt'
#
# 	if loadflag == 'y': #load file
# 		graph = gt.load_graph( savename )
#
# 	elif loadflag == 'n': #or else, compute
#
# 		names = ['nodei', 'nodej', 'tstamp'] #column names
#
# 		#load (unique) event list: node i, node j, timestamp
# 		filename = root_data + dataname + '/data_formatted/' + eventname
# 		events = pd.read_csv( filename, sep=';', header=None, names=names )
#
# 		#create edge list (as np array): ( node i, node j, weight )
# 		#weight as alter activity: count number of events per alter of each ego
# 		edge_list = events.groupby(['nodei', 'nodej']).size().reset_index( name='weight' ).to_numpy()
#
# 		#initialise graph stuff
# 		graph = gt.Graph( directed=False ) #initialise (undirected) graph
# 		eweight = graph.new_ep( 'long' ) #initialise weight as (int) edge property map
#
# 		#add (hashed) edge list (with weights) and get vertex indices as vertex property map
# 		vid = graph.add_edge_list( edge_list, hashed=True, eprops=[ eweight ] )
#
# 		#set internal property maps
# 		graph.vertex_properties[ 'id' ] = vid #vertex id
# 		graph.edge_properties[ 'weight' ] = eweight #edge weight
#
# 		graph.save( savename ) #save graph
#
# 	return graph

		# for prop in graph_percs.columns[:-2]: #loop through (non-random) properties
		# sorted_nodei = egonet_filt[ prop[:-4] ].sort_values().index

		# columns = pd.Series( [prop+'_LCC' for prop in prop_names] + [prop+'_2CC' for prop in prop_names] + ['rand_LCC', 'rand_2CC'], name='prop' ) #percolation cases

			# #remove SMALLEST VALUES first
			# sorted_nodei = egonet_filt[ prop[:-4] ].sort_values( ascending=False ).index
			# vertices_rev = [ gt.find_vertex( graph, graph.vp.id, nodei )[0] for nodei in sorted_nodei ]
			# sizes_rev, comp = gt.vertex_percolation( graph, vertices_rem + vertices_rev )

		# egonet_fits_pieces = [] #initialise dframe
		# for pos, egonet_tuple in enumerate(zip( egonet_props_pieces, egonet_acts_pieces )): #loop through time periods
		# 	print( '\ttime period: {} of {}'.format(pos+1, len(egonet_props_pieces)), flush=True )
		#
		# 	egonet_fits_pieces.append( egonet_fits( dataname, eventname, root_data, loadflag, saveloc, egonet_tuple=egonet_tuple, alphamax=alphamax, nsims=nsims, amax=amax ) ) #store results
		#
		# #save all to file
		# with open( savename, 'wb' ) as file:
		# 	pk.dump( egonet_fits_pieces, file )
