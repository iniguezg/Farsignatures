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

	#root locations of data/code
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/'
	saveloc = expanduser('~') + '/prg/xocial/Farsignatures/files/data/'
	saveloc_fig = expanduser('~') + '/prg/xocial/Farsignatures/figures/figure1_data/'

	#dataset list: eventname, textname
	datasets = [ ( 'call', 'Mobile (call)'),
				 ( 'text', 'Mobile (sms)'),
				 ( 'MPC_Wu_SD01', 'Mobile (Wu 1)'),
				 ( 'MPC_Wu_SD02', 'Mobile (Wu 2)'),
				 ( 'MPC_Wu_SD03', 'Mobile (Wu 3)'),
				 ( 'Enron', 'Email (Enron)'),
				 ( 'email', 'Email (Kiel)'),
				 ( 'eml2', 'Email (Uni)'),
				 ( 'email_Eu_core', 'Email (EU)'),
				 ( 'fb', 'Facebook'),
				 ( 'messages', 'Messages'),
				 ( 'pok', 'Dating'),
				 ( 'forum', 'Forum'),
				 ( 'CollegeMsg', 'College'),
				 ( 'CNS_calls', 'CNS (call)'),
				 ( 'CNS_sms', 'CNS (sms)') ]

	#sizes/widths/coords
	plot_props = { 'xylabel' : 13,
	'figlabel' : 26,
	'ticklabel' : 15,
	'text_size' : 10,
	'marker_size' : 6,
	'linewidth' : 2,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':10 },
	'legend_hlen' : 1,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables
	fig_props = { 'fig_num' : 1,
	'fig_size' : (10, 10),
	'aspect_ratio' : (3, 2),
	'grid_params' : dict( left=0.085, bottom=0.05, right=0.98, top=0.975, wspace=0.3, hspace=0.48 ),
	'dpi' : 300,
	'savename' : 'figure1' }

	colors = sns.color_palette( 'Paired', n_colors=3 ) #colors to plot


	## PLOTTING ##

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'], height_ratios=[0.5, 0.9, 1] )
	grid.update( **fig_props['grid_params'] )


# A: Example of event sequence, alter ranking timeseries, and final ego network / activity distribution

	#diagram variables
	dataname, eventname, textname = 'Copenhagen_nets', 'CNS_calls', 'CNS (call)' #selected dataset
	nodei = 578 #selected ego
	linelen = 0.4 #(half-)length of event line to plot
	int_strs = [ 0.28, 0.78 ] #locations of interval strings
	edgewid = 5 #edge width

	print('DIAGRAM', flush=True)
	print( 'dataset name: ' + eventname, flush=True ) #print output

	## DATA ##

	#load ego network properties and alter activities
	egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
	egonet_acts = pd.read_pickle( saveloc + 'egonet_acts_' + eventname + '.pkl' )
	alters_nodei = egonet_acts.loc[nodei].sort_values(ascending=False).index #alters (from most to less active)
	k = len(alters_nodei) #degree
	activity = egonet_acts[nodei] #alter activities for selected ego

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

	## PLOTTING ##

	colors = sns.color_palette( 'Set2', n_colors=k ) #colors to plot


	# A1: Example of event sequence

	#initialise subplot
	subgrid = grid[ 0,: ].subgridspec( 1, 4, wspace=0.4 )
	ax = plt.subplot( subgrid[ 0 ] )
	ax.set_axis_off()

	plt.text( -0.46, 0.94, 'a', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	for posj, nodej in enumerate( alters_nodei ): #loop through alters
		plt.axhline( y=k-posj, c=colors[posj], lw=plot_props['linewidth'] ) #plot event line
		plt.text( -0.03, k-posj, '(ego, alter {})'.format(posj+1), va='center', ha='right', fontsize=plot_props['text_size'] ) #plot link name

	for event_tuple in events_nodei.itertuples(): #loop through events
		posj = alters_nodei.get_loc( event_tuple.nodej ) #alter position (ranked by activity)
		y = k - posj #plot position
		plt.vlines( event_tuple.tstamp_norm, y-linelen, y+linelen, color=colors[posj], lw=plot_props['linewidth']-1 ) #plot events

	#interval strings
	midpoint = events_nodei.iloc[int(np.ceil(len(events_nodei)/2))].tstamp_norm #same # events on both sides
	plt.axvline( x=midpoint, ls='--', c='0.5', lw=plot_props['linewidth'] )
	for pos, int_str in enumerate(int_strs): #loop through interval strings
		plt.text( int_str, 1.03, '$I_'+str(pos+1)+'$', va='bottom', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'] )

	#plot time arrow
	arrow_str = r'time'
	bbox_props = dict( boxstyle="rarrow,pad=0.2", fc='None', ec='0.7', lw=1 )
	plt.text( 0.5, -0.1, arrow_str, ha='center', va='center', transform=ax.transAxes,
    size=plot_props['text_size'], bbox=bbox_props )

	#finalise subplot
	plt.axis([ -0.02, 1.02, 1-linelen, k+linelen ])


	# A2: Alter ranking time series

	#initialise subplot
	ax = plt.subplot( subgrid[ 1 ] )
	sns.despine( ax=ax ) #take out spines
	# plt.xlabel( r'time', size=plot_props['xylabel'], labelpad=0 )
	plt.ylabel( r'$a$', size=plot_props['xylabel'], labelpad=0 )

	for posj, nodej in enumerate( alters_nodei ): #loop through alters, ranked by activity
		xplot = events_nodei[ events_nodei.nodej==nodej ].tstamp_norm #get normalised event times
		yplot = range( 1, len(xplot)+1 ) #and cumulative sum of number of events

		xplot = list(xplot) + [1.] #add endpoints
		yplot = list(yplot) + [yplot[-1]]

		plt.plot( xplot, yplot, '-', color=colors[posj], lw=plot_props['linewidth'] ) #plot plot!

	#interval strings
	midpoint = events_nodei.iloc[int(np.ceil(len(events_nodei)/2))].tstamp_norm #same # events on both sides
	plt.axvline( x=midpoint, ls='--', c='0.5', lw=plot_props['linewidth'] )
	for pos, int_str in enumerate(int_strs): #loop through interval strings
		plt.text( int_str, 1.03, '$I_'+str(pos+1)+'$', va='bottom', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'] )

	#plot time arrows

	arrow_str = r'time'
	bbox_props = dict( boxstyle="rarrow,pad=0.2", fc='None', ec='0.7', lw=1 )
	plt.text( 0.5, -0.1, arrow_str, ha='center', va='center', transform=ax.transAxes,
    size=plot_props['text_size'], bbox=bbox_props )

	arrow_str = 'end'
	bbox_props = dict( boxstyle="rarrow,pad=0.2", fc='None', ec='0.7', lw=1 )
	plt.text( 1.2, 0.5, arrow_str, ha='center', va='center', transform=ax.transAxes,
    size=plot_props['text_size'], bbox=bbox_props )

	#finalise subplot
	plt.axis([ -0.02, 1.02, 1e0, 1.5e2 ])
	ax.set_yscale('log')
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=3 )
	plt.xticks([])


# B: Example of final ego network and activity distribution

	#B1: Final ego network

	#set up ego network
	graph = nx.generators.classic.star_graph( k ) #ego net as star graph
	graph = nx.relabel_nodes( graph, {0:'ego'} ) #rename ego
	positions = nx.circular_layout( range(1, len(graph.nodes)), center=(0,0) )
	positions['ego'] = np.zeros(2) #ego at center

	#initialise subplot
	ax = plt.subplot( subgrid[ 2 ] )
	ax.set_axis_off()

	plt.text( -0.35, 0.94, 'b', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	#plot plot nodes! ego, alters, and labels
	nx.draw_networkx_nodes( graph, positions, nodelist=['ego'], node_size=500, node_color='#ffd92f', margins=0.1 )
	nx.draw_networkx_nodes( graph, positions, nodelist=list(graph)[1:], node_size=200, node_color=colors, margins=0.1 )
	nx.draw_networkx_labels( graph, positions )

	#set up existing edges at time snapshot
	weights = events_nodei.groupby("nodej").size()
	edgelist = [ ('ego', alters_nodei.get_loc(nodej)+1 ) for nodej in weights.index ]
	width = [edgewid * np.log10(weights[nodej]) / np.log10(egonet_acts[nodei].max()) for nodej in weights.index] #log scaling!
	edge_color = [ colors[ alters_nodei.get_loc(nodej) ] for nodej in weights.index ]

	#plot plot edges!
	nx.draw_networkx_edges( graph, positions, edgelist=edgelist, width=width, edge_color=edge_color )


	#B2: Final activity distribution

	#initialise subplot
	ax = plt.subplot( subgrid[ 3 ] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( r'$a$', size=plot_props['xylabel'] )
	plt.ylabel( r'$k p_a$', size=plot_props['xylabel'] )

	#plot plot!
	sns.histplot( activity, binrange=(activity.min()-1, activity.max()+1), binwidth=2, element='step', color='k', zorder=1 )

	#lines and notation
	plt.annotate( text='$a_0$', xy=( activity.min(), 2.8 ), ha='center', va='bottom' )
	plt.axvline( x=activity.min(), ls='--', c='0.5', lw=1, zorder=0 )
	plt.annotate( text='$t$', xy=( activity.mean(), 2.8 ), ha='center', va='bottom' )
	plt.axvline( x=activity.mean(), ls='--', c='0.5', lw=1, zorder=0 )
	pm.draw_brace( ax, ( activity.mean(), activity.mean() + activity.std() ), 2.3, '$\sigma$' )

	#finalise subplot
	plt.axis([ -10, 140, 0, 2.8 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=3 )
	ax.locator_params( nbins=3 )


# C-E: Activity distribution (CCDF), dispersion distribution, and connection kernel of heterogeneous/homogeneous egos

	#subplot variables
	eventname, textname = ( 'call', 'Mobile (call)')

	#filter variables
	min_degree = 10 #minimum degree of filtered egos
	min_negos = 30 #minimum number of egos in filtered activity group
	num_quants = 5 #number of quantiles (+1) of filtered egos

	#dispersion variables
	filter_prop = 'strength' #selected property and threshold to filter egos
	filter_thres = 0

	colors = sns.color_palette( 'GnBu', n_colors=num_quants-1 ) #colors to plot

	## DATA ##

	#load ego network properties, alter activities, alpha fits, and connection kernel
	egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )

	#prepare filter object (ego dispersions)
	#only consider egos with large enough degree!
	egonet_props_filt = egonet_props[ egonet_props.degree >= min_degree ]
	act_disps = dm.egonet_dispersion( egonet_props_filt, filter_prop, filter_thres )

	#get quantiles of filter parameter (dispersion)
	quantile_arr = np.linspace(0, 1, num_quants)
	quantile_vals = np.quantile( act_disps, quantile_arr )


	# C: Activity distribution (CCDF)

	print('ACTIVITY REGIMES', flush=True)
	print( 'dataset name: ' + eventname, flush=True ) #print output

	## PLOTTING ##

	# C: Activity distribution

	#initialise subplot
	subgrid = grid[ 1,: ].subgridspec( 1, 3, wspace=0.4, hspace=0.4 )
	ax = plt.subplot( subgrid[0] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( r'$a$', size=plot_props['xylabel'], labelpad=0 )
	plt.ylabel( r"$P[a' \geq a]$", size=plot_props['xylabel'] )

	plt.text( -0.34, 1.15, 'c', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	#loop through quantiles of filter parameter (inclusive!)
	for posval, (min_val, max_val) in enumerate( zip(quantile_vals[:-1], quantile_vals[1:]) ):
		#load alter activity CCDF and average social signature (not used!) according to filter
		ccdf, sign = pm.plot_activity_filter( 'divided_to_roughly_40_mb_files_30_march', eventname, filt_rule='dispersion', filt_obj=act_disps, filt_params={ 'min_val':min_val, 'max_val':max_val, 'min_negos':min_negos }, is_parallel=True, load=True, root_data=root_data, saveloc=saveloc, saveloc_fig=saveloc_fig )

		#label by filter property range
		label = '{:.2f} '.format(min_val)+'$\leq d <$'+' {:.2f}'.format(max_val)

		#plot plot alter activity CCDF!
		plt.loglog( ccdf.x, ccdf.y, '-', c=colors[posval], label=label, lw=plot_props['linewidth'] )

	#legend
	plt.legend( loc='lower right', bbox_to_anchor=(1.1, 1), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=2 )

	#finalise subplot
	plt.axis([ 1e0, 2e4, 1e-8, 2e0 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )


	# D: Dispersion distribution

	print('DISPERSION/KERNEL REGIMES', flush=True)
	print( 'dataset name: ' + eventname, flush=True ) #print output

	#subplot variables
	bins_disp = np.linspace(0, 1, 40) #bins to plot dispersion

	#initialise subplot
	ax = plt.subplot( subgrid[1] )
	sns.despine( ax=ax, top=False, bottom=True ) #take out spines
	plt.xlabel( r'$p_d$', size=plot_props['xylabel'] )
	plt.ylabel( r'$d$', size=plot_props['xylabel'], labelpad=0 )
	ax.xaxis.set_label_position('top')

	plt.text( -0.28, 1.15, 'd', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	#plot plot!
	cnts, vals, bars = plt.hist( act_disps, bins=bins_disp, density=True, orientation='horizontal', log=True )

	#loop through quantiles of filter parameter (inclusive!)
	for min_val, max_val in zip(quantile_vals[:-1], quantile_vals[1:]):
		plt.axhline( min_val, ls='--', c='0.7', lw=plot_props['linewidth'] )

	#color histogram according to quantile ranges!
	for patch in range( len(bars) ):
		for posval, (min_val, max_val) in enumerate( zip(quantile_vals[:-1], quantile_vals[1:]) ):
			if vals[patch] > min_val:
				bars[patch].set_facecolor( colors[posval] )

	#plot dispersion equation
	eq_str = r'$d = \frac{ \sigma^2 - t + a_0 }{ \sigma^2 + t - a_0 }$'
	plt.text( 0.95, 0.95, eq_str, va='top', ha='right', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

	#finalise plot
	plt.axis([ 5e-2, 1e1, bins_disp[0]-0.01, bins_disp[-1]+0.01 ])
	ax.invert_yaxis()
	ax.tick_params( labelbottom=False,labeltop=True )
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=3 )


	# E: Connection kernel

	#initialise subplot
	ax = plt.subplot( subgrid[2] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( r'$a / a_m$', size=plot_props['xylabel'] )
	plt.ylabel( r'$\pi_a - \langle 1/k \rangle$', size=plot_props['xylabel'] )

	plt.text( -0.3, 1.15, 'e', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	#loop through quantiles of filter parameter (inclusive!)
	for posval, (min_val, max_val) in enumerate( zip(quantile_vals[:-1], quantile_vals[1:]) ):

		#prepare kernel: apply degree / negos filters, group and average
		data_avg, filt_ind = pm.plot_kernel_filter( eventname, filt_rule='dispersion', filt_obj=act_disps, filt_params={ 'min_val':min_val, 'max_val':max_val, 'min_negos':min_negos }, load=True, saveloc=saveloc, saveloc_fig=saveloc_fig )

		#prepare baseline: prob = <1/k> for random case
		bline_avg = ( 1 / egonet_props_filt.degree[filt_ind] ).mean()

		#label by filter property range
		label = '{:.2f} '.format(min_val)+'$\leq d <$'+' {:.2f}'.format(max_val)

		#plot plot kernel mean (minus baseline)!
		xplot = data_avg.index / data_avg.index.max() #normalise by max activity
		line_data, = plt.plot( xplot, data_avg - bline_avg, '-', c=colors[posval], label=label, lw=plot_props['linewidth'], zorder=1 )

		#plot plot baseline!
		line_base = plt.axhline( 0, ls='--', c='0.7', label=None, lw=plot_props['linewidth'], zorder=0 )

	#legends
	leg1 = plt.legend( loc='lower right', bbox_to_anchor=(1.1, 1), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=2 )
	ax.add_artist(leg1)
	leg2 = plt.legend( (line_data, line_base), ('data', 'random choice'), loc='lower right', bbox_to_anchor=(1.1, 0.05), prop=plot_props['legend_prop'], handlelength=1.7, numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=2 )
	ax.add_artist(leg2)

	#finalise subplot
	plt.axis([ -0.02, 1.02, -0.05, 1.02 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )


# F: Dispersion index for all datasets

	#subplot variables
	filter_prop = 'strength' #selected property and threshold to filter egos
	filter_thres = 10

	colors = sns.color_palette( 'Set2', n_colors=len(datasets) ) #colors to plot

	print('DISPERSION INDEX', flush=True)

	#initialise subplot
	subgrid = grid[ 2,0 ].subgridspec( 2, 1, hspace=0, height_ratios=[0.1, 1] )
	ax = plt.subplot( subgrid[ 1 ] )
	sns.despine( ax=ax )
	plt.xlabel( r'$d$', size=plot_props['xylabel'] )
	plt.ylabel( r"$P[d' \geq d]$", size=plot_props['xylabel'] )

	plt.text( -0.2, 1.2, 'f', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	#loop through considered datasets
	total_egos_filter = 0 #init counter of all filtered egos
	for grid_pos, (eventname, textname) in enumerate(datasets):
		print( 'dataset name: ' + eventname, flush=True ) #print output

		## DATA ##

		#prepare ego network properties
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )

		#get activity dispersion for egos in dataset
		act_disps = dm.egonet_dispersion( egonet_props, filter_prop, filter_thres )

		#print output
		print( '\tshown egos: {:.2f}%'.format( 100.*len(act_disps)/len(egonet_props) ), flush=True ) #filtered egos
		total_egos_filter += len(act_disps) #filtered egos
		print( '\tavg disp: {:.2f}'.format( act_disps.mean() ), flush=True ) #mean dispersion
		if grid_pos == len(datasets)-1:
			print( '\t\ttotal number of filtered egos: {}'.format( total_egos_filter ), flush=True )

		## PLOTTING ##

		#plot plot activity dispersion!
		xplot, yplot = pm.plot_CCDF_cont( act_disps ) #get dispersion CCDF: P[X >= x]
		plt.plot( xplot, yplot, '-', c=colors[grid_pos], label=textname, lw=plot_props['linewidth'] )

	#legend
	plt.legend( loc='lower left', bbox_to_anchor=(0, 0.99), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=8 )

	#finalise subplot
	plt.axis([ -0.05, 1, 0, 1.05 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
	ax.locator_params( nbins=6 )


# G: Connection kernel for all datasets

	#subplot variables
	min_degree = 2 #minimum degree of filtered egos
	min_negos = 50 #minimum number of egos in filtered activity group

	colors = sns.color_palette( 'Set2', n_colors=len(datasets) ) #colors to plot

	print('KERNEL', flush=True)

	#initialise subplot
	subgrid = grid[ 2,1 ].subgridspec( 2, 1, hspace=0, height_ratios=[0.1, 1] )
	ax = plt.subplot( subgrid[ 1 ] )
	sns.despine( ax=ax )
	plt.xlabel( r'$a$', size=plot_props['xylabel'] )
	plt.ylabel( r'$\pi_a - \langle 1/k \rangle$', size=plot_props['xylabel'] )

	plt.text( -0.2, 1.2, 'g', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	#loop through considered datasets
	for grid_pos, (eventname, textname) in enumerate(datasets):
		print( 'dataset name: ' + eventname, flush=True ) #print output

		## DATA ##

		#prepare ego network properties
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )

		#only consider egos with large enough degree!
		egonet_props_filt = egonet_props[ egonet_props.degree >= min_degree ]

		#use single quantile range, i.e. all degrees!
		quantile_arr = np.linspace(0, 1, 2)
		min_val, max_val = np.quantile( egonet_props_filt.degree, quantile_arr )

		#prepare kernel: apply degree / negos filters, group and average
		data_avg, filt_ind = pm.plot_kernel_filter( eventname, filt_rule='degree', filt_obj=egonet_props.degree, filt_params={ 'min_val':min_val, 'max_val':max_val, 'min_negos':min_negos }, load=True, saveloc=saveloc, saveloc_fig=saveloc_fig )

		#prepare baseline: prob = <1/k> for random case
		bline_avg = ( 1 / egonet_props_filt.degree[filt_ind] ).mean()


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
