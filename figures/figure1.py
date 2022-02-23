#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE 1 IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_misc as dm
import plot_misc as pm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#locations
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files

	#dataset list: eventname, textname
	datasets = [ ( 'MPC_UEu', 'Mobile (call)'),
	# datasets = [ ( 'call', 'Mobile (call)'),
	# 			 ( 'text', 'Mobile (sms)'),
				 ( 'MPC_Wu_SD01', 'Mobile (Wu 1)'),
				 ( 'MPC_Wu_SD02', 'Mobile (Wu 2)'),
				 ( 'MPC_Wu_SD03', 'Mobile (Wu 3)'),
				 ( 'sexcontact_events', 'Contact'),
				 ( 'email', 'Email 1'),
				 ( 'eml2', 'Email 2'),
				 ( 'fb', 'Facebook'),
				 ( 'messages', 'Messages'),
				 ( 'forum', 'Forum'),
				 ( 'pok', 'Dating'),
				 ( 'CNS_bt_symmetric', 'CNS (bluetooth)'),
				 ( 'CNS_calls', 'CNS (call)'),
				 ( 'CNS_sms', 'CNS (sms)') ]

	#sizes/widths/coords
	plot_props = { 'xylabel' : 13,
	'figlabel' : 26,
	'ticklabel' : 15,
	'text_size' : 10,
	'marker_size' : 10,
	'linewidth' : 2,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':10 },
	'legend_hlen' : 1,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables
	fig_props = { 'fig_num' : 1,
	'fig_size' : (10, 8),
	'aspect_ratio' : (2, 2),
	'grid_params' : dict( left=0.08, bottom=0.065, right=0.98, top=0.98, wspace=0.3, hspace=0.4 ),
	'dpi' : 300,
	'savename' : 'figure1' }

	colors = sns.color_palette( 'Paired', n_colors=3 ) #colors to plot


	## PLOTTING ##

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )


# A: Example of event sequence and time evolution of ego network

	#subplot variables
	dataname, eventname, textname = 'Copenhagen_nets', 'CNS_calls', 'CNS (call)' #selected dataset
	nodei = 578 #selected ego
	linelen = 0.4 #(half-)length of event line to plot
	snaps = [ 0.3, 0.8 ] #snapshots (fractional time) to plot ego net
	edgewid = 5 #edge width

	print('DIAGRAM')
	print( 'dataset name: ' + eventname ) #print output

	## DATA ##

	#load ego network properties and alter activities
	egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
	egonet_acts = pd.read_pickle( saveloc + 'egonet_acts_' + eventname + '.pkl' )
	alters_nodei = egonet_acts.loc[nodei].sort_values(ascending=False).index #alters (from most to less active)
	k = len(alters_nodei) #degree

	#prepare raw events
	names = ['nodei', 'nodej', 'tstamp'] #column names
	#load (unique) event list: node i, node j, timestamp
	filename = root_data + dataname + '/data_formatted/' + eventname + '.evt'
	events = pd.read_csv( filename, sep=';', header=None, names=names )
	#reverse nodes i, j in all events
	events_rev = events.rename( columns={ 'nodei':'nodej', 'nodej':'nodei' } )[ names ]
	#duplicate events (all nodes with events appear as node i)
	events_concat = pd.concat([ events, events_rev ])

	#get events for ego and sort by time
	events_nodei = events_concat.groupby('nodei').get_group(nodei).sort_values('tstamp')
	#normalize times to [0,1] interval
	events_nodei['tstamp_norm'] = ( events_nodei.tstamp - min(events_nodei.tstamp) ) / ( max(events_nodei.tstamp) - min(events_nodei.tstamp) )


	# A1: Example of event sequence

	colors = sns.color_palette( 'Set2', n_colors=k ) #colors to plot

	#initialise subplot
	subgrid = grid[ 0,0 ].subgridspec( 2, 3, hspace=0, wspace=0, width_ratios=[0.2, 1, 1] )
	ax = plt.subplot( subgrid[ 0,1: ] )
	ax.set_axis_off()

	plt.text( -0.31, 0.88, 'a', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	for posj, nodej in enumerate( alters_nodei ): #loop through alters
		plt.axhline( y=k-posj, c=colors[posj], lw=plot_props['linewidth'] ) #plot event line
		plt.text( -0.03, k-posj, '(ego, alter {})'.format(posj+1), va='center', ha='right', fontsize=plot_props['text_size'] ) #plot link name

	for event_tuple in events_nodei.itertuples(): #loop through events
		posj = alters_nodei.get_loc( event_tuple.nodej ) #alter position (ranked by activity)
		y = k - posj #plot position
		plt.vlines( event_tuple.tstamp_norm, y-linelen, y+linelen, color=colors[posj], lw=plot_props['linewidth']-1 ) #plot events

	for t in snaps: #draw time snapshots
		plt.axvline( x=t, ls='--', c='0.5', lw=plot_props['linewidth'] )

	#plot time arrow
	arrow_str = 'time'
	bbox_props = dict( boxstyle="rarrow,pad=0.2", fc='None', ec='0.7', lw=1 )
	plt.text( 0.5, -0.1, arrow_str, ha='center', va='center', transform=ax.transAxes,
    size=plot_props['text_size'], bbox=bbox_props )

	#finalise subplot
	plt.axis([ -0.02, 1.02, 1-linelen, k+linelen ])


	# A2: Time evolution of ego network

	#set up ego network
	graph = nx.generators.classic.star_graph( k ) #ego net as star graph
	graph = nx.relabel_nodes( graph, {0:'ego'} ) #rename ego
	positions = nx.spring_layout( graph, seed=2 ) #seed layout for reproducibility

	for post, t in enumerate(snaps): #loop through snapshots of ego net
		#initialise subplot
		ax = plt.subplot( subgrid[ 1,post+1 ] )
		ax.set_axis_off()

		#plot plot nodes! ego, alters, and labels
		nx.draw_networkx_nodes( graph, positions, nodelist=['ego'], node_size=500, node_color='#ffd92f', margins=0.1 )
		nx.draw_networkx_nodes( graph, positions, nodelist=list(graph)[1:], node_size=200, node_color=colors, margins=0.1 )
		nx.draw_networkx_labels( graph, positions )

		#set up existing edges at time snapshot
		weights = events_nodei[ events_nodei.tstamp_norm < t ].groupby("nodej").size()
		edgelist = [ ('ego', alters_nodei.get_loc(nodej)+1 ) for nodej in weights.index ]
		width = [edgewid * np.log10(weights[nodej]) / np.log10(egonet_acts[nodei].max()) for nodej in weights.index] #log scaling!
		edge_color = [ colors[ alters_nodei.get_loc(nodej) ] for nodej in weights.index ]

		#plot plot edges!
		nx.draw_networkx_edges( graph, positions, edgelist=edgelist, width=width, edge_color=edge_color )


# B: Example of heterogeneous/homogeneous activity distributions and social signatures

	#alpha fit variables
	alphamax = 1000 #maximum alpha for MLE fit
	pval_thres = 0.1 #threshold above which alpha MLEs are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf

	#subplot variables
	dataname, eventname, textname = 'greedy_walk_nets', 'forum', 'Forum' #selected
	labels = ['heterogeneous ego', 'homogeneous ego'] #regime labels
	nodei_vals = [ 26, 1910 ] #selected egos (heterogeneous/homogeneous)

	colors = sns.color_palette( 'Set2', n_colors=len(labels) ) #colors to plot

	print('ACTIVITY')
	print( 'dataset name: ' + eventname ) #print output

	## DATA ##

	#load ego network properties, alter activities, and alpha fits
	egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
	egonet_acts = pd.read_pickle( saveloc + 'egonet_acts_' + eventname + '.pkl' )
	egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )

	#filter egos according to fitting results
	egonet_filter, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )

	print( '\theterogeneous ego:\n{}'.format( egonet_filter.loc[nodei_vals[0]] ) ) #print ego properties
	print( '\thomogeneous ego:\n{}'.format( egonet_filter.loc[nodei_vals[1]] ) )


	# B1: Activity distributions

	#initialise subplot
	subgrid = grid[ 0,1 ].subgridspec( 2, 1, hspace=0, height_ratios=[0.06, 1] )
	ax = plt.subplot( subgrid[ 1 ] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( r'activity', size=plot_props['xylabel'], labelpad=0 )
	plt.ylabel( r'CCDF of alters', size=plot_props['xylabel'], labelpad=0 )

	plt.text( -0.24, 0.99, 'b', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	for posi, nodei in enumerate(nodei_vals): #loop through selected egos
		activity = egonet_acts[nodei] #alter activities for selected ego

		#plot plot activity distribution!
		xplot, yplot = pm.plot_compcum_dist( activity ) #get alter activity CCDF: P[X >= x]
		plt.loglog( xplot, yplot, 'o', c=colors[posi], ms=plot_props['marker_size'], label=labels[posi] )

	#legend
	plt.legend( loc='lower center', bbox_to_anchor=(0.5, 1), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=len(labels) )

	#finalise subplot
	plt.axis([ 0.8e0, 1e3, 1e-2, 1.2e0 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=3 )


	#B2: Social signatures

	inax = ax.inset_axes([ 0.6, 0.6, 0.4, 0.4 ])
	sns.despine( ax=inax ) #take out spines
	inax.set_xlabel( r'alter rank', size=plot_props['text_size'], labelpad=0 )
	inax.set_ylabel( r'activity fraction', size=plot_props['text_size'], labelpad=0 )

	for posi, nodei in enumerate(nodei_vals): #loop through selected egos
		activity = egonet_acts[nodei] #alter activities for selected ego

		#plot plot social signature!
		xplot = np.arange( 1, len(activity)+1, dtype=int )
		yplot = activity.sort_values( ascending=False ) / activity.sum()
		inax.loglog( xplot, yplot, 'o', c=colors[posi], ms=plot_props['marker_size']-5 )

	#finalise inset
	inax.set_xlim( 0.8e0, 1e2)
	inax.set_ylim( 0.7e-3, 2.5e-1 )
	inax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['text_size'], length=2, pad=3 )


# C: Dispersion index for all datasets

	#subplot variables
	filter_prop = 'strength' #selected property and threshold to filter egos
	filter_thres = 10

	colors = sns.color_palette( 'Set2', n_colors=len(datasets) ) #colors to plot

	print('DISPERSION INDEX')

	#initialise subplot
	ax = plt.subplot( grid[ 1,0 ] )
	sns.despine( ax=ax )
	plt.xlabel( r'activity dispersion', size=plot_props['xylabel'] )
	plt.ylabel( r'CCDF of egos', size=plot_props['xylabel'] )

	plt.text( -0.19, 1.15, 'c', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	#loop through considered datasets
	for grid_pos, (eventname, textname) in enumerate(datasets):
		print( 'dataset name: ' + eventname ) #print output

		## DATA ##

		#prepare ego network properties and alter activities
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
		egonet_acts = pd.read_pickle( saveloc + 'egonet_acts_' + eventname + '.pkl' )

		#get activity means/variances/minimums per ego
		act_avgs = egonet_acts.groupby('nodei').mean()
		act_vars = egonet_acts.groupby('nodei').var() #NOTE: estimated variance (ddof=1)
		act_mins = egonet_props.act_min
		#filter by selected property
		act_avgs = act_avgs[ egonet_props[filter_prop] > filter_thres ]
		act_vars = act_vars[ egonet_props[filter_prop] > filter_thres ]
		act_mins = act_mins[ egonet_props[filter_prop] > filter_thres ]
		#get dispersion index measure per ego (use relative mean!)
		act_disps = ( act_vars - act_avgs + act_mins ) / ( act_vars + act_avgs - act_mins )
		act_disps = act_disps.dropna() #drop faulty egos

		print( '\tshown egos: {:.2f}%'.format( 100.*len(act_disps)/len(egonet_props) ) ) #filtered egos


		## PLOTTING ##

		#plot plot activity dispersion!
		xplot, yplot = pm.plot_CCDF_cont( act_disps ) #get dispersion CCDF: P[X >= x]
		plt.plot( xplot, yplot, '-', c=colors[grid_pos], label=textname, lw=plot_props['linewidth'] )

	#legend
	plt.legend( loc='lower left', bbox_to_anchor=(0.07, 0.99), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=7 )

	#finalise subplot
	plt.axis([ -0.05, 1.05, -0.05, 1.05 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
	ax.locator_params( nbins=6 )


# D: Connection kernel for all datasets

	#subplot variables
	min_degree = 2 #minimum degree of filtered egos
	min_negos = 50 #minimum number of egos in filtered activity group

	print('KERNEL')

	#initialise subplot
	ax = plt.subplot( grid[ 1,1 ] )
	sns.despine( ax=ax )
	plt.xlabel( r'activity', size=plot_props['xylabel'] )
	plt.ylabel( r'relative probability of event', size=plot_props['xylabel'] )

	plt.text( -0.24, 1.15, 'd', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	#loop through considered datasets
	for grid_pos, (eventname, textname) in enumerate(datasets):
		print( 'dataset name: ' + eventname ) #print output

		## DATA ##

		#prepare ego network properties and connection kernel
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
		egonet_kernel = pd.read_pickle( saveloc + 'egonet_kernel_' + eventname + '.pkl' )

		#prepare data: apply degree / negos filters, group and average
		filt_degree = egonet_kernel[ egonet_props[ egonet_props.degree >= min_degree ].index ]
		filt_negos = filt_degree.groupby( level=1 ).filter( lambda x : len(x) >= min_negos )
		data_grp = filt_negos.groupby( level=1 ) #group ego probs for each activity value
		data_avg = data_grp.mean() #average probs over egos

		#prepare baseline: prob = 1/k for random case
		bline_avg = ( 1 / egonet_props[ egonet_props.degree >= min_degree ].degree ).mean()

		print( '\t{:.2f}% egos after degree filter'.format( 100.*filt_degree.index.get_level_values(0).nunique() / len(egonet_props) ) ) #print filtering output


		## PLOTTING ##

		#plot plot connection kernel! (relative to baseline)
		label = 'data' if grid_pos == 0 else None #plot label
		plt.plot( data_avg - bline_avg, '-', c=colors[grid_pos], label=label, lw=plot_props['linewidth'], zorder=1 )

	#plot plot baseline!
	plt.axhline( 0, ls='--', c='0.7', label='random choice', lw=plot_props['linewidth'], zorder=0 )

	#legend
	plt.legend( loc='lower right', bbox_to_anchor=(1.02, 0.01), prop=plot_props['legend_prop'], handlelength=1.7, numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'] )

	#finalise subplot
	plt.axis([ 0, 100, -0.25, 0.65 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )


#DEBUGGIN'

	# nx.draw( graph, positions, node_color='#66c2a5', node_size=200, with_labels=True )
	#re-draw ego node
	# options = { 'node_size':500, 'node_color':'#fc8d62' }
	# nx.draw_networkx_nodes( graph, positions, nodelist=['ego'], **options )

	# xplot = np.arange( activity.min(), activity.max()+1, dtype=int )
	# yplot, not_used = np.histogram( activity, bins=len(xplot), range=( xplot[0]-0.5, xplot[-1]+0.5 ), density=True )
	# xplot_binned, yplot_binned = pm.plot_logbinned_dist( activity )

		# plt.loglog( xplot_binned, yplot_binned, '-', lw=plot_props['linewidth'] )

		# #bin data
		# yplot, bin_edges = np.histogram( act_disps, bins=bins, density=True )
		# xplot = [ ( bin_edges[i+1] + bin_edges[i] ) / 2 for i in range(len( bin_edges[:-1] )) ]
		# #mask data
		# yplot = np.ma.masked_where( yplot < 1e-2, yplot )
		#
		# #plot plot!
		# plt.semilogy( xplot, yplot, '-', c=colors[grid_pos], label=textname, lw=plot_props['linewidth'], ms=plot_props['marker_size']-5 )

		#only plot positive values after thrreshold (for log plotting)
		# xplot, yplot = xplot_raw[ xplot_raw > 0 ], yplot_raw[ xplot_raw > 0 ]

	# bins = np.linspace( -1, 1, 51 ) #bins for dispersion indes
	# plt.axis([ -1, 1, 1e-2, 2e1 ])
