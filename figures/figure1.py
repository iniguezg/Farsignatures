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


## PLOTTING FUNCTION ##

#function to plot kernel curve according to filter
def plot_kernel_filter( egonet_kernel, eventname, filt_rule='', filt_obj=None, filt_params={}, load=False ):
	"""Plot kernel curve according to filter"""

	savename = 'figure1_data/kernel_{}_filter_rule_{}_params_{}.pkl'.format( eventname, filt_rule, ''.join([ k+'_'+str(v)+'_' for k, v in filt_params.items() ]) ) #filename to load/save

	if load:
		data_avg = pd.read_pickle(savename) #load file
	else:
		#filter egos by selected filter property
		if filt_rule == 'large_disp': #large dispersion
			filter = egonet_kernel[ filt_obj[ filt_obj > filt_obj.mean() ].index ]
		elif filt_rule == 'small_disp': #small dispersion
			filter = egonet_kernel[ filt_obj[ filt_obj < filt_obj.mean() ].index ]
		elif filt_rule == 'degree': #large enough degree
			filter = egonet_kernel[ filt_obj[ filt_obj.degree >= filt_params['min_degree'] ].index ]
		else: #no filter
			filter = egonet_kernel

		#get filtered activity groups
		filt_negos = filter.groupby( level=1 ).filter( lambda x : len(x) >= filt_params['min_negos'] )
		data_grp = filt_negos.groupby( level=1 ) #group ego probs for each activity value
		data_avg = data_grp.mean() #average probs over egos

		data_avg.to_pickle(savename) #save file

	return data_avg


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#root locations of data/code
	#LOCAL
	# root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/'
	# saveloc = expanduser('~') + '/prg/xocial/Farsignatures/files/data/'
	#TRITON
	root_data = '/m/cs/scratch/networks/inigueg1/prg/xocial/datasets/temporal_networks/'
	saveloc = '/m/cs/scratch/networks/inigueg1/prg/xocial/Farsignatures/files/data/'

	#dataset list: eventname, textname
	# datasets = [ ( 'MPC_UEu', 'Mobile (call)'),
	datasets = [ ( 'call', 'Mobile (call)'),
				 ( 'text', 'Mobile (sms)'),
				 ( 'MPC_Wu_SD01', 'Mobile (Wu 1)'),
				 ( 'MPC_Wu_SD02', 'Mobile (Wu 2)'),
				 ( 'MPC_Wu_SD03', 'Mobile (Wu 3)'),
				 # ( 'sexcontact_events', 'Contact'),
				 ( 'email', 'Email 1'),
				 ( 'eml2', 'Email 2'),
				 ( 'fb', 'Facebook'),
				 ( 'messages', 'Messages'),
				 ( 'forum', 'Forum'),
				 ( 'pok', 'Dating'),
				 # ( 'CNS_bt_symmetric', 'CNS (bluetooth)'),
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
	'grid_params' : dict( left=0.085, bottom=0.055, right=0.98, top=0.975, wspace=0.3, hspace=0.45 ),
	'dpi' : 300,
	'savename' : 'figure1' }

	colors = sns.color_palette( 'Paired', n_colors=3 ) #colors to plot


	## PLOTTING ##

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'], height_ratios=[0.5, 1, 1] )
	grid.update( **fig_props['grid_params'] )


# A: Example of event sequence, alter ranking timeseries, and final ego network / activity distribution

	#diagram variables
	dataname, eventname, textname = 'Copenhagen_nets', 'CNS_calls', 'CNS (call)' #selected dataset
	nodei = 578 #selected ego
	linelen = 0.4 #(half-)length of event line to plot
	int_strs = [ 0.25, 0.75 ] #locations of interval strings
	edgewid = 5 #edge width

	print('DIAGRAM')
	print( 'dataset name: ' + eventname ) #print output

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
	plt.axvline( x=0.5, ls='--', c='0.5', lw=plot_props['linewidth'] )
	for pos, int_str in enumerate(int_strs): #loop through interval strings
		plt.text( int_str, 1.03, '$I_'+str(pos+1)+'$', va='bottom', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'] )

	#plot time arrow
	arrow_str = r'event time $\tau$'
	bbox_props = dict( boxstyle="rarrow,pad=0.2", fc='None', ec='0.7', lw=1 )
	plt.text( 0.5, -0.1, arrow_str, ha='center', va='center', transform=ax.transAxes,
    size=plot_props['text_size'], bbox=bbox_props )

	#finalise subplot
	plt.axis([ -0.02, 1.02, 1-linelen, k+linelen ])


	# A2: Alter ranking time series

	#initialise subplot
	ax = plt.subplot( subgrid[ 1 ] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( r'$\tau/T$', size=plot_props['xylabel'], labelpad=0 )
	plt.ylabel( r'$a$', size=plot_props['xylabel'], labelpad=0 )

	for posj, nodej in enumerate( alters_nodei ): #loop through alters, ranked by activity
		xplot = events_nodei[ events_nodei.nodej==nodej ].tstamp_norm #get normalised event times
		yplot = range( 1, len(xplot)+1 ) #and cumulative sum of number of events

		xplot = [0.] + list(xplot) + [1.] #add endpoints
		yplot = [0.] + list(yplot) + [yplot[-1]]

		plt.plot( xplot, yplot, '-', color=colors[posj], lw=plot_props['linewidth'] ) #plot plot!

	#interval strings
	plt.axvline( x=0.5, ls='--', c='0.5', lw=plot_props['linewidth'] )
	for pos, int_str in enumerate(int_strs): #loop through interval strings
		plt.text( int_str, 1.03, '$I_'+str(pos+1)+'$', va='bottom', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'] )

	#plot time arrow
	arrow_str = r'$\tau=T$'
	bbox_props = dict( boxstyle="rarrow,pad=0.2", fc='None', ec='0.7', lw=1 )
	plt.text( 1.2, 0.5, arrow_str, ha='center', va='center', transform=ax.transAxes,
    size=plot_props['text_size'], bbox=bbox_props )

	#finalise subplot
	plt.axis([ -0.02, 1.02, 0.8e0, 1.5e2 ])
	ax.set_yscale('log')
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=3 )


	#A3: Final ego network

	#set up ego network
	graph = nx.generators.classic.star_graph( k ) #ego net as star graph
	graph = nx.relabel_nodes( graph, {0:'ego'} ) #rename ego
	positions = nx.spring_layout( graph, seed=2 ) #seed layout for reproducibility

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


	#A4: Final activity distribution

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
	plt.annotate( text='$\mu$', xy=( activity.mean(), 2.8 ), ha='center', va='bottom' )
	plt.axvline( x=activity.mean(), ls='--', c='0.5', lw=1, zorder=0 )
	pm.draw_brace( ax, ( activity.mean(), activity.mean() + activity.std() ), 2.3, '$\sigma$' )

	#finalise subplot
	plt.axis([ -10, 140, 0, 2.8 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=3 )
	ax.locator_params( nbins=3 )


# B: Activity distribution and social signatures of heterogeneous/homogeneous egos

	#dispersion variables
	filter_prop = 'strength' #selected property and threshold to filter egos
	filter_thres = 5

	#alpha fit variables
	alphamax = 1000 #maximum alpha for MLE fit
	pval_thres = 0.1 #threshold above which alpha MLEs are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf

	#subplot variables
	dataname, eventname, textname = 'greedy_walk_nets', 'forum', 'Forum' #selected dataset
	labels = ['heterogeneous ego', 'homogeneous ego'] #regime labels
	nodei_vals = [ 26, 1910 ] #selected egos (heterogeneous/homogeneous)

	colors = sns.color_palette( 'Paired', n_colors=2 ) #colors to plot
	symbols = ['o', 's'] #symbols to plot

	print('ACTIVITY REGIMES')
	print( 'dataset name: ' + eventname ) #print output

	## DATA ##

	#load ego network properties, alter activities, alpha fits, and connection kernel
	egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
	egonet_acts = pd.read_pickle( saveloc + 'egonet_acts_' + eventname + '.pkl' )
	egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )

	#get activity means/variances/minimums per ego
	act_avgs = egonet_props.act_avg
	act_vars = egonet_props.act_var
	act_mins = egonet_props.act_min
	#filter by selected property
	act_avgs = act_avgs[ egonet_props[filter_prop] > filter_thres ]
	act_vars = act_vars[ egonet_props[filter_prop] > filter_thres ]
	act_mins = act_mins[ egonet_props[filter_prop] > filter_thres ]
	#get dispersion index measure per ego (use relative mean!)
	act_disps = ( act_vars - act_avgs + act_mins ) / ( act_vars + act_avgs - act_mins )
	act_disps = act_disps.dropna() #drop faulty egos

	disp_vals = [ act_disps[nodei] for nodei in nodei_vals ] #dispersion values
	print( '\tshown egos: {:.2f}%'.format( 100.*len(act_disps)/len(egonet_props) ) ) #filtered egos

	#filter egos according to fitting results
	egonet_filter, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )

	print( '\theterogeneous ego:\n{}'.format( egonet_filter.loc[nodei_vals[0]] ) ) #print ego properties
	print( '\t\tdispersion: {:.2f}'.format( disp_vals[0] ) )
	print( '\thomogeneous ego:\n{}'.format( egonet_filter.loc[nodei_vals[1]] ) )
	print( '\t\tdispersion: {:.2f}'.format( disp_vals[1] ) )

	## PLOTTING ##

	# B1: Activity distribution

	#initialise subplot
	subgrid = grid[ 1,: ].subgridspec( 2, 3, wspace=0.4, hspace=0.4 )
	ax = plt.subplot( subgrid[ 0,0 ] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( r'$a$', size=plot_props['xylabel'], labelpad=0 )
	plt.ylabel( r"$P[a' \geq a]$", size=plot_props['xylabel'] )

	plt.text( -0.34, 1.25, 'c', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	for posi, nodei in enumerate(nodei_vals): #loop through selected egos
		activity = egonet_acts[nodei] #alter activities for selected ego

		#plot plot activity distribution!
		xplot, yplot = pm.plot_compcum_dist( activity ) #get alter activity CCDF: P[X >= x]
		plt.loglog( xplot, yplot, symbols[posi], c=colors[1-posi], ms=plot_props['marker_size'], label=labels[posi] )

	#legend
	plt.legend( loc='lower left', bbox_to_anchor=(-0.31, 1.05), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=len(labels) )

	#finalise subplot
	plt.axis([ 0.8e0, 2e2, 1e-2, 1.2e0 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=3 )


	#B2: Social signatures

	inax = plt.subplot( subgrid[ 1,0 ] )
	sns.despine( ax=inax ) #take out spines
	inax.set_xlabel( r'$r$', size=plot_props['xylabel'] )
	inax.set_ylabel( r'$f_a$', size=plot_props['xylabel'] )

	for posi, nodei in enumerate(nodei_vals): #loop through selected egos
		activity = egonet_acts[nodei] #alter activities for selected ego

		#plot plot social signature!
		xplot = np.arange( 1, len(activity)+1, dtype=int )
		yplot = activity.sort_values( ascending=False ) / activity.sum()
		inax.loglog( xplot, yplot, symbols[posi], c=colors[1-posi], ms=plot_props['marker_size'] )

	#finalise inset
	inax.set_xlim( 0.8e0, 1e2)
	inax.set_ylim( 0.7e-3, 2.5e-1 )
	inax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=3 )


# C: Dispersion distribution and connection kernel of heterogeneous/homogeneous egos

	#dispersion variables
	filter_prop = 'strength' #selected property and threshold to filter egos
	filter_thres = 5

	#kernel variables
	min_negos = 50 #minimum number of egos in filtered activity group

	#subplot variables
	eventname, textname = 'call', 'Mobile (call)' #selected dataset

	colors = sns.color_palette( 'Paired', n_colors=2 ) #colors to plot

	print('DISPERSION/KERNEL REGIMES')
	print( 'dataset name: ' + eventname ) #print output

	## DATA ##

	#load ego network properties, alter activities, alpha fits, and connection kernel
	egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
	egonet_kernel = pd.read_pickle( saveloc + 'egonet_kernel_' + eventname + '.pkl' )

	#get activity means/variances/minimums per ego
	act_avgs = egonet_props.act_avg
	act_vars = egonet_props.act_var
	act_mins = egonet_props.act_min
	#filter by selected property
	act_avgs = act_avgs[ egonet_props[filter_prop] > filter_thres ]
	act_vars = act_vars[ egonet_props[filter_prop] > filter_thres ]
	act_mins = act_mins[ egonet_props[filter_prop] > filter_thres ]
	#get dispersion index measure per ego (use relative mean!)
	act_disps = ( act_vars - act_avgs + act_mins ) / ( act_vars + act_avgs - act_mins )
	act_disps = act_disps.dropna() #drop faulty egos

	disp_vals = [ act_disps[nodei] for nodei in nodei_vals ] #dispersion values
	print( '\tshown egos: {:.2f}%'.format( 100.*len(act_disps)/len(egonet_props) ) ) #filtered egos

	## PLOTTING ##

	# C1: Dispersion distribution

	#initialise subplot
	ax = plt.subplot( subgrid[ :,1 ] )
	sns.despine( ax=ax, top=False, bottom=True ) #take out spines
	plt.xlabel( r'$p_d$', size=plot_props['xylabel'] )
	plt.ylabel( r'$d$', size=plot_props['xylabel'], labelpad=0 )
	ax.xaxis.set_label_position('top')

	plt.text( -0.32, 1.1, 'd', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	#plot plot!
	cnts, vals, bars = plt.hist( act_disps, bins=30, density=True, orientation='horizontal' )
	plt.axhline( act_disps.mean(), ls='--', c='0.5', lw=plot_props['linewidth'] )

	#color histogram!
	for patch in range( len(bars) ):
		if vals[patch] > act_disps.mean():
			bars[patch].set_facecolor( colors[1] )
		else:
			bars[patch].set_facecolor( colors[0] )

	#texts
	plt.text( 1.8, act_disps.mean(), r'$\langle d \rangle$', va='bottom', ha='center', fontsize=plot_props['ticklabel'] )

	#plot dispersion equation
	eq_str = r'$d = \frac{ \sigma^2 - \mu + a_0 }{ \sigma^2 + \mu - a_0 }$'
	plt.text( 0.95, 0.95, eq_str, va='top', ha='right', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

	#finalise plot
	plt.axis([ 0, 2, -0.4, 1 ])
	ax.invert_yaxis()
	ax.tick_params( labelbottom=False,labeltop=True )
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=3 )


	# C2: Connection kernel

	#initialise subplot
	ax = plt.subplot( subgrid[ :,2 ] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( r'$a / a_m$', size=plot_props['xylabel'] )
	plt.ylabel( r'$\pi_a$', size=plot_props['xylabel'] )

	plt.text( -0.3, 1.1, 'e', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	for posd in range(2): #loop through regimes
		print('\t{}:'.format(labels[posd])) #print regime

		#prepare data: apply dispersion / negos filters, group and average
		if posd == 0: #heterogeneous
			data_avg = plot_kernel_filter( egonet_kernel, eventname, filt_rule='large_disp', filt_obj=act_disps, filt_params={ 'min_negos':min_negos }, load=False )
			label=r'$d > \langle d \rangle$'
		else: #homogeneous
			data_avg = plot_kernel_filter( egonet_kernel, eventname, filt_rule='small_disp', filt_obj=act_disps, filt_params={ 'min_negos':min_negos }, load=False )
			label=r'$d < \langle d \rangle$'

		#prepare baseline: prob = 1/k for random case
		if posd == 0: #heterogeneous
			bline_avg = ( 1 / egonet_props.loc[ act_disps[ act_disps > act_disps.mean() ].index ].degree ).mean()
		else: #homogeneous
			bline_avg = ( 1 / egonet_props.loc[ act_disps[ act_disps < act_disps.mean() ].index ].degree ).mean()

		#plot plot!
		xplot = data_avg.index / data_avg.index.max() #normalise by max activity
		line_data, = plt.plot( xplot, data_avg, '-', c=colors[1-posd], label=label, lw=plot_props['linewidth'], zorder=1 )
		line_base = plt.hlines( y=bline_avg, xmin=0, xmax=1, ls='--', colors=[colors[1-posd]], label=None, lw=plot_props['linewidth'], zorder=0 )

	#legend
	leg1 = plt.legend( loc='lower left', bbox_to_anchor=(0,1), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=2 )
	ax.add_artist(leg1)
	leg2 = plt.legend( (line_data, line_base), ('data', r'$\langle 1/k \rangle$'), loc='upper left', bbox_to_anchor=(0, 1), prop=plot_props['legend_prop'], handlelength=1.7, numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ] )
	ax.add_artist(leg2)

	#finalise subplot
	plt.axis([ -0.05, 1, 0, 0.6 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )


# D: Dispersion index for all datasets

	#subplot variables
	filter_prop = 'strength' #selected property and threshold to filter egos
	filter_thres = 10

	colors = sns.color_palette( 'Set2', n_colors=len(datasets) ) #colors to plot

	print('DISPERSION INDEX')

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
		print( 'dataset name: ' + eventname ) #print output

		## DATA ##

		#prepare ego network properties
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )

		#get activity means/variances/minimums per ego
		act_avgs = egonet_props.act_avg
		act_vars = egonet_props.act_var
		act_mins = egonet_props.act_min
		#filter by selected property
		act_avgs = act_avgs[ egonet_props[filter_prop] > filter_thres ]
		act_vars = act_vars[ egonet_props[filter_prop] > filter_thres ]
		act_mins = act_mins[ egonet_props[filter_prop] > filter_thres ]
		#get dispersion index measure per ego (use relative mean!)
		act_disps = ( act_vars - act_avgs + act_mins ) / ( act_vars + act_avgs - act_mins )
		act_disps = act_disps.dropna() #drop faulty egos

		#print output
		print( '\tshown egos: {:.2f}%'.format( 100.*len(act_disps)/len(egonet_props) ) ) #filtered egos
		total_egos_filter += len(act_disps) #filtered egos
		print( '\tavg disp: {:.2f}'.format( act_disps.mean() ) ) #mean dispersion
		if grid_pos == len(datasets)-1:
			print( '\t\ttotal number of filtered egos: {}'.format( total_egos_filter ) )

		## PLOTTING ##

		#plot plot activity dispersion!
		xplot, yplot = pm.plot_CCDF_cont( act_disps ) #get dispersion CCDF: P[X >= x]
		plt.plot( xplot, yplot, '-', c=colors[grid_pos], label=textname, lw=plot_props['linewidth'] )

	#legend
	plt.legend( loc='lower left', bbox_to_anchor=(0.07, 0.99), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=7 )

	#finalise subplot
	plt.axis([ -0.05, 1, 0, 1.05 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
	ax.locator_params( nbins=6 )


# E: Connection kernel for all datasets

	#subplot variables
	min_degree = 2 #minimum degree of filtered egos
	min_negos = 50 #minimum number of egos in filtered activity group

	# #TEMPORARY: used datasets
	# datasets = [ ( 'MPC_UEu', 'Mobile (call)'),
	# 			 ( 'MPC_Wu_SD01', 'Mobile (Wu 1)'),
	# 			 ( 'MPC_Wu_SD02', 'Mobile (Wu 2)'),
	# 			 ( 'MPC_Wu_SD03', 'Mobile (Wu 3)'),
	# 			 # ( 'sexcontact_events', 'Contact'),
	# 			 ( 'email', 'Email 1'),
	# 			 ( 'eml2', 'Email 2'),
	# 			 ( 'fb', 'Facebook'),
	# 			 ( 'messages', 'Messages'),
	# 			 ( 'forum', 'Forum'),
	# 			 ( 'pok', 'Dating'),
	# 			 # ( 'CNS_bt_symmetric', 'CNS (bluetooth)'),
	# 			 ( 'CNS_calls', 'CNS (call)'),
	# 			 ( 'CNS_sms', 'CNS (sms)') ]

	colors = sns.color_palette( 'Set2', n_colors=len(datasets) ) #colors to plot

	print('KERNEL')

	#initialise subplot
	subgrid = grid[ 2,1 ].subgridspec( 2, 1, hspace=0, height_ratios=[0.1, 1] )
	ax = plt.subplot( subgrid[ 1 ] )
	sns.despine( ax=ax )
	plt.xlabel( r'$a$', size=plot_props['xylabel'] )
	plt.ylabel( r'$\pi_a - \langle 1/k \rangle$', size=plot_props['xylabel'] )

	plt.text( -0.2, 1.2, 'g', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	#loop through considered datasets
	for grid_pos, (eventname, textname) in enumerate(datasets):
		print( 'dataset name: ' + eventname ) #print output

		## DATA ##

		#prepare ego network properties and connection kernel
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
		egonet_kernel = pd.read_pickle( saveloc + 'egonet_kernel_' + eventname + '.pkl' )

		#prepare data: apply degree / negos filters, group and average
		data_avg = plot_kernel_filter( egonet_kernel, eventname, filt_rule='degree', filt_obj=egonet_props, filt_params={ 'min_degree':min_degree, 'min_negos':min_negos }, load=False )

		#prepare baseline: prob = 1/k for random case
		bline_avg = ( 1 / egonet_props[ egonet_props.degree >= min_degree ].degree ).mean()


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

	# plt.hlines( y=2.2, xmin=activity.mean(), xmax=activity.mean() + activity.std(), ls='--', colors=['0.5'], lw=1, zorder=0 )
	# plt.annotate( text='$\sigma$', xy=( activity.mean() + activity.std() / 2, 2.1 ), ha='center', va='top' )

	# plt.annotate( text='', xy=( 0, disp_vals[0] ), xytext=( -0.7, disp_vals[0] ), arrowprops=dict( headlength=12, headwidth=10, width=5, color=colors[1], alpha=0.5 ) ) #heterogeneous
	# plt.annotate( text='', xy=( 0, disp_vals[1] ), xytext=( -2.1, disp_vals[1] ), arrowprops=dict( headlength=12, headwidth=10, width=5, color=colors[0], alpha=0.5 ) ) #homogeneous

			# filt_disp = egonet_kernel[ act_disps[ act_disps > act_disps.mean() ].index ]
			# filt_disp = egonet_kernel[ act_disps[ act_disps < act_disps.mean() ].index ]
		# filt_negos = filt_disp.groupby( level=1 ).filter( lambda x : len(x) >= min_negos )
		# data_grp = filt_negos.groupby( level=1 ) #group ego probs for each activity value
		# data_avg = data_grp.mean() #average probs over egos
		# print( '\t{:.2f}% egos after dispersion filter'.format( 100.*filt_disp.index.get_level_values(0).nunique() / len(egonet_props) ) ) #print filtering output

		# filt_degree = egonet_kernel[ egonet_props[ egonet_props.degree >= min_degree ].index ]
		# filt_negos = filt_degree.groupby( level=1 ).filter( lambda x : len(x) >= min_negos )
		# data_grp = filt_negos.groupby( level=1 ) #group ego probs for each activity value
		# data_avg = data_grp.mean() #average probs over egos
		# print( '\t{:.2f}% egos after degree filter'.format( 100.*filt_degree.index.get_level_values(0).nunique() / len(egonet_props) ) ) #print filtering output
