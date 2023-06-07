#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (IC2S2 ABSTRACT) IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from os.path import expanduser

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_misc as dm
import model_misc as mm
import plot_misc as pm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#alpha fit variables
	stat = 'KS' #chosen test statistic
	alphamax = 1000 #maximum alpha for MLE fit
	pval_thres = 0.1 #threshold above which alpha MLEs are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf

	#locations
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc_data = root_code+'files/data/' #location of output files (data/model/fig)
	saveloc_model = root_code+'files/model/'
	saveloc_fig = root_code+'figures/figure1_data/'

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
	plot_props = { 'xylabel' : 15,
	'figlabel' : 26,
	'ticklabel' : 12,
	'text_size' : 10,
	'marker_size' : 5,
	'linewidth' : 2,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':10 },
	'legend_hlen' : 1,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables
	fig_props = { 'fig_num' : 1,
	'fig_size' : (11, 8),
	'aspect_ratio' : (2, 3),
	'grid_params' : dict( left=0.08, bottom=0.07, right=0.99, top=0.95, wspace=0.3, hspace=0.4 ),
	'dpi' : 300,
	'savename' : 'abstract_ic2s2' }


	## PLOTTING ##

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'], width_ratios=[1, 1, 1.2], height_ratios=[1.1, 1] )
	grid.update( **fig_props['grid_params'] )


# A: Example of event sequence and alter ranking timeseries

	#diagram variables
	dataname, eventname, textname = 'Copenhagen_nets', 'CNS_calls', 'CNS (call)' #selected dataset
	nodei = 578 #selected ego
	linelen = 0.4 #(half-)length of event line to plot
	int_strs = [ 0.28, 0.78 ] #locations of interval strings

	print('DIAGRAM', flush=True)
	print( 'dataset name: ' + eventname, flush=True ) #print output

	## DATA ##

	#load ego network properties and alter activities
	egonet_props = pd.read_pickle( saveloc_data + 'egonet_props_' + eventname + '.pkl' )
	egonet_acts = pd.read_pickle( saveloc_data + 'egonet_acts_' + eventname + '.pkl' )
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
	subgrid = grid[ 0,0 ].subgridspec( 2, 1, hspace=0.2 )
	ax = plt.subplot( subgrid[0] )
	ax.set_axis_off()

	plt.text( -0.3, 1, 'a', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

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
	plt.text( 0.5, 1.08, arrow_str, ha='center', va='bottom', transform=ax.transAxes,
    size=plot_props['text_size'], bbox=bbox_props )

	#finalise subplot
	plt.axis([ -0.02, 1.02, 1-linelen, k+linelen ])


	# A2: Alter ranking time series

	#initialise subplot
	ax = plt.subplot( subgrid[1] )
	sns.despine( ax=ax ) #take out spines
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

	#plot time arrows

	arrow_str = r'time'
	bbox_props = dict( boxstyle="rarrow,pad=0.2", fc='None', ec='0.7', lw=1 )
	plt.text( 0.5, -0.08, arrow_str, ha='center', va='top', transform=ax.transAxes,
    size=plot_props['text_size'], bbox=bbox_props )

	#finalise subplot
	plt.axis([ -0.02, 1.02, 1e0, 1.5e2 ])
	ax.set_yscale('log')
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=3 )
	plt.xticks([])


# B: Activity distribution and connection kernel of heterogeneous/homogeneous egos

	#dispersion variables
	filter_prop = 'strength' #selected property and threshold to filter egos
	filter_thres = 5
	#kernel variables
	min_negos = 50 #minimum number of egos in filtered activity group

	#flags
	load = True

	#subplot variables
	eventname, textname = 'forum', 'Forum'
	labels = ['heterogeneous egos', 'homogeneous egos'] #regime labels
	nodei_vals = [ 26, 1910 ] #selected egos (heterogeneous/homogeneous)

	colors = sns.color_palette( 'Paired', n_colors=2 ) #colors to plot
	symbols = ['o', 's'] #symbols to plot

	print('ACTIVITY REGIMES', flush=True)
	print( 'dataset name: ' + eventname, flush=True ) #print output

	## DATA ##

	#load ego network properties, alter activities, alpha fits, and connection kernel
	egonet_props = pd.read_pickle( saveloc_data + 'egonet_props_' + eventname + '.pkl' )
	egonet_acts = pd.read_pickle( saveloc_data + 'egonet_acts_' + eventname + '.pkl' )
	egonet_fits = pd.read_pickle( saveloc_data + 'egonet_fits_' + eventname + '.pkl' )

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
	print( '\tshown egos: {:.2f}%'.format( 100.*len(act_disps)/len(egonet_props) ), flush=True ) #filtered egos

	#filter egos according to fitting results
	egonet_filter, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, pval_thres=pval_thres, alphamax=alphamax, alph_thres=alph_thres )

	print( '\theterogeneous ego:\n{}'.format( egonet_filter.loc[nodei_vals[0]] ), flush=True ) #print ego properties
	print( '\t\tdispersion: {:.2f}'.format( disp_vals[0] ), flush=True )
	print( '\thomogeneous ego:\n{}'.format( egonet_filter.loc[nodei_vals[1]] ), flush=True )
	print( '\t\tdispersion: {:.2f}'.format( disp_vals[1] ), flush=True )


	## PLOTTING ##

	# B1: Activity distribution

	#initialise subplot
	subgrid = grid[ 0,1 ].subgridspec( 2, 1, hspace=0.4 )
	ax = plt.subplot( subgrid[0] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( r'$a$', size=plot_props['xylabel'], labelpad=0 )
	plt.ylabel( r"$P[a' \geq a]$", size=plot_props['xylabel'] )

	plt.text( -0.26, 1, 'b', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	for posi, nodei in enumerate(nodei_vals): #loop through selected egos
		activity = egonet_acts[nodei] #alter activities for selected ego

		#plot plot activity distribution!
		xplot, yplot = pm.plot_compcum_dist( activity ) #get alter activity CCDF: P[X >= x]
		plt.loglog( xplot, yplot, symbols[posi], c=colors[1-posi], ms=plot_props['marker_size'], label=labels[posi] )

	#legend
	plt.legend( loc='upper right', bbox_to_anchor=(1, 1.3), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=1 )

	#finalise subplot
	plt.axis([ 0.8e0, 2e2, 1e-2, 1.2e0 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=3 )


	# B2: Connection kernel

	#initialise subplot
	ax = plt.subplot( subgrid[1] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( r'$a / a_m$', size=plot_props['xylabel'] )
	plt.ylabel( r'$\pi_a$', size=plot_props['xylabel'] )

	for posd in range(2): #loop through regimes
		#prepare data: apply dispersion / negos filters, group and average
		if posd == 0: #heterogeneous
			data_avg = pm.plot_kernel_filter( eventname, filt_rule='large_disp', filt_obj=act_disps, filt_params={ 'min_negos':min_negos }, load=load, saveloc=saveloc_data, saveloc_fig=saveloc_fig )
			label=r'$d > \langle d \rangle$'
		else: #homogeneous
			data_avg = pm.plot_kernel_filter( eventname, filt_rule='small_disp', filt_obj=act_disps, filt_params={ 'min_negos':min_negos }, load=load, saveloc=saveloc_data, saveloc_fig=saveloc_fig )
			label=r'$d < \langle d \rangle$'
		print('\t{}:'.format(label), flush=True) #print regime

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
	leg2 = plt.legend( (line_data, line_base), ('data', r'$\langle 1/k \rangle$'), loc='upper left', bbox_to_anchor=(0, 1.1), prop=plot_props['legend_prop'], handlelength=1.7, numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ] )
	ax.add_artist(leg2)

	#finalise subplot
	plt.axis([ -0.05, 1, 0, 0.6 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )


# C: Diagram of alter activity model

	#subplot variables
	k = 5 #ego degree
	a0, am = 1, 10 #min/max alter activity
	alpha_vals = [ -0.9, 1000. ] #PA parameter
	t_vals = [ 3, 1000 ] #mean alter activity (max time in dynamics)
	labels = [ 'cumulative advantage\n'+r'($ \alpha \to -a_0 $)', 'random choice\n'+r'($ \alpha \to \infty $)' ]

	seed=2 #rng seed
	nsizes = [200, 70 ] #node sizes (ego, alters)
	edgewids = [ 4, 2 ] #edge widths
	colors = sns.color_palette( 'GnBu', n_colors=2 ) #colors to plot

	#set up ego network
	graph = nx.generators.classic.star_graph( k ) #ego net as star graph
	graph = nx.relabel_nodes( graph, {0:'ego'} ) #rename ego
	positions = nx.spring_layout( graph, seed=seed ) #seed layout for reproducibility

	print('DIAGRAM')

	# C1: initial ego network

	margins = (0.05, 0.7) #margins for plotting nets

	#initialise subplot
	subgrid = grid[ 0,2 ].subgridspec( 2, 3, hspace=0.6, wspace=0.5 )
	ax = plt.subplot( subgrid[ :, 0 ] )
	ax.set_axis_off()

	plt.text( -0.3, 1, 'c', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	#plot plot network! ego, alters, labels
	nx.draw_networkx_nodes( graph, positions, nodelist=['ego'], node_size=nsizes[0], node_color=[colors[0]], margins=margins )
	nx.draw_networkx_nodes( graph, positions, nodelist=list(graph)[1:], node_size=nsizes[1], node_color=[colors[1]], margins=margins )
	nx.draw_networkx_labels( graph, positions, labels={'ego':'ego'}, font_size=plot_props['text_size'] )

	#plot plot edges! edges, labels
	nx.draw_networkx_edges( graph, positions, width=edgewids[1], edge_color=colors[1] )
	nx.draw_networkx_edge_labels( graph, positions, edge_labels={ edge:'$a_0$' for edge in graph.edges }, rotate=False, font_size=plot_props['text_size'] )

	#plot connection kernel
	eq_str = r'$\pi_a = \frac{ a + \alpha }{ \tau + k \alpha }$'
	plt.text( 0.5, 0.9, eq_str, va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

	#plot time arrow
	arrow_str = '$t = a_0$'
	bbox_props = dict( boxstyle="rarrow,pad=0.2", fc='None', ec='0.7', lw=1 )
	plt.text( 0.5, -0.1, arrow_str, ha='center', va='center', transform=ax.transAxes, size=plot_props['text_size'], bbox=bbox_props )
	plt.text( 0.5, -0.03, 'initial activity', ha='center', va='center', transform=ax.transAxes, size=plot_props['text_size'] )

	# C2: shapes for connection probability

	for grid_pos in range(2): #loop through regimes
		#initialise subplot
		ax = plt.subplot( subgrid[ grid_pos, 1 ] )
		sns.despine( ax=ax )
		if grid_pos == 1:
			plt.xlabel( r'$a$', size=plot_props['xylabel'] )
		plt.ylabel( r'$\pi_a$', size=plot_props['xylabel'], rotation='horizontal', labelpad=10 )

		#get parameters and plot arrays
		alpha = alpha_vals[grid_pos]
		tau = k * t_vals[0]
		xplot = np.arange( a0, am+1, dtype=int )
		yplot = ( xplot + alpha ) / ( tau + k*alpha )

		#plot plot!
		plt.plot( xplot, yplot, '-k', lw=plot_props['linewidth'] )

		#finalise subplot
		plt.axis([ a0, am, -0.1, 1 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		plt.xticks([])
		plt.yticks([])

		#regime labels
		plt.text( 0.5, 1.03, labels[grid_pos], va='bottom', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )


	# C3: final ego network

	margins = (0.05, 0.4) #margins for plotting nets

	for grid_pos in range(2): #loop through regimes
		#initialise subplot
		ax = plt.subplot( subgrid[ grid_pos, 2 ] )
		ax.set_axis_off()

		#model parameters
		params = { 'alpha': alpha_vals[grid_pos], 'a0':a0, 'k' :k, 't':t_vals[1], 'ntimes':1 }
		#run model of alter activity
		activity = mm.model_activity( params, saveflag=False, seed=seed, print_every=1000 )

		#plot plot network! ego, alters, labels
		nx.draw_networkx_nodes( graph, positions, nodelist=['ego'], node_size=nsizes[0], node_color=[colors[0]], margins=margins )
		nx.draw_networkx_nodes( graph, positions, nodelist=list(graph)[1:], node_size=nsizes[1], node_color=[colors[1]], margins=margins )
		nx.draw_networkx_labels( graph, positions, labels={'ego':'ego'}, font_size=plot_props['text_size'] )

		#plot plot edges! edges, labels
		width = 1 + np.array([ edgewids[grid_pos] * np.log2(activity[ 0, nodej-1 ]) / np.log2(activity.max()) for nodej in graph['ego'] ]) #log scaling!
		nx.draw_networkx_edges( graph, positions, width=width, edge_color=colors[1] )

	#plot time arrow
	arrow_str =r'$t = \tau / k$'
	bbox_props = dict( boxstyle="rarrow,pad=0.2", fc='None', ec='0.7', lw=1 )
	plt.text( 0.5, -0.23, arrow_str, ha='center', va='center', transform=ax.transAxes,
    size=plot_props['text_size'], bbox=bbox_props )
	plt.text( 0.5, -0.07, 'mean activity', ha='center', va='center', transform=ax.transAxes, size=plot_props['text_size'] )


# D: Phase diagram of selected dataset

	#subplot variables
	eventname, textname = ( 'text', 'Mobile (sms)') #selected dset
	propx = ('gamma', r'\alpha_r')
	propy = ('act_avg_rel', 't_r')
	gridsize = 40 #grid size for hex bins
	vmax = 1e6 #max value in colorbar (larger than [filtered] N in any dataset!)

	print('PHASE DIAGRAM')

	#initialise subplot
	subgrid = grid[ 1,: ].subgridspec( 1, 3, wspace=0.5, width_ratios=[1.2, 1, 1] )
	ax = plt.subplot( subgrid[0] )
	sns.despine( ax=ax )
	plt.xlabel( '$'+propx[1]+'$', size=plot_props['xylabel'] )
	plt.ylabel( '$'+propy[1]+'$', size=plot_props['xylabel'] )

	plt.text( -0.3, 1, 'd', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	## DATA ##

	#prepare ego network properties
	egonet_props = pd.read_pickle( saveloc_data + 'egonet_props_' + eventname + '.pkl' )
	#fit activity model to all ego networks in dataset
	egonet_fits = pd.read_pickle( saveloc_data + 'egonet_fits_' + eventname + '.pkl' )

	#filter egos according to fitting results
	egonet_filter, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, stat=stat, pval_thres=pval_thres, alphamax=alphamax, alph_thres=alph_thres )

	#add relative quantities
	t_rels = pd.Series( egonet_filter.act_avg - egonet_filter.act_min, name='act_avg_rel' )
	egonet_filter = pd.concat( [ egonet_filter, t_rels ], axis=1 )

	#print output
	num_egos_filter = len( egonet_filter ) #statistically significant alpha
	frac_egos_random = ( egonet_filter.beta < 1 ).sum() / float( num_egos_filter ) #fraction of egos in random regime (beta < 1, i.e. t_r < alpha_r)
	frac_egos_cumadv = ( egonet_filter.beta > 1 ).sum() / float( num_egos_filter ) #fraction of egos in CA regime (beta > 1, i.e. t_r > alpha_r)
	print( '\thom regime: {:.2f}%, het regime: {:.2f}%'.format( frac_egos_random*100, frac_egos_cumadv*100 ) )

	## PLOTTING ##

	#plot plot!
	hexbin = plt.hexbin( propx[0], propy[0], data=egonet_filter, xscale='log', yscale='log', norm=LogNorm(vmin=1e0, vmax=vmax), mincnt=1, gridsize=gridsize, cmap='GnBu', zorder=0 )

	#colorbar
	cbar = plt.colorbar( hexbin, ax=ax, fraction=0.05 )
	cbar.ax.set_title( r'$N_{'+propx[1][:-2]+','+propy[1][:-2]+'}$' )
	cbar.ax.minorticks_off()

	#lines
	plt.plot( [1e-3, 1e4], [1e-3, 1e4], '--', c='0.6', lw=plot_props['linewidth'], zorder=1 )

	#texts
	beta_str = r'$\beta = \frac{t_r}{\alpha_r} = \frac{t - a_0}{\alpha + a_0}$'
	plt.text( 1, 1.1, beta_str, va='center', ha='right', transform=ax.transAxes, fontsize=plot_props['xylabel'] )
	plt.text( 0.8, 0.7, r'crossover ($\beta = 1$)', va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'], rotation=50 )
	het_str = 'heterogeneous\n'+r'($\beta > 1$)'
	plt.text( 0.3, 0.98, het_str, va='top', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'] )
	hom_str = 'homogeneous\n'+r'($\beta < 1$)'
	plt.text( 0.7, 0.05, hom_str, va='bottom', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'] )

	#finalise subplot
	plt.axis([ 1e-3, 1e3, 1e-2, 1e4 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
	ax.locator_params( numticks=5 )


# E: CCDF of estimated model parameter for all datasets

	#subplot variables
	prop_name = 'beta'
	prop_label = r'1 / \beta'

	colors = sns.color_palette( 'Set2', n_colors=len(datasets) ) #colors to plot

	print('PARAMETER CCDF')

	#initialise subplot
	ax = plt.subplot( subgrid[1] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( '${}$'.format( prop_label ), size=plot_props['xylabel'] )
	plt.ylabel( "$P[{}' \geq {}]$".format( prop_label,prop_label ), size=plot_props['xylabel'] )

	plt.text( -0.3, 1, 'e', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	#loop through considered datasets
	total_egos_filter = 0 #init counter of all filtered egos
	for grid_pos, (eventname, textname) in enumerate(datasets):
		print( 'dataset name: ' + eventname ) #print output

		## DATA ##

		#prepare ego network properties
		egonet_props = pd.read_pickle( saveloc_data + 'egonet_props_' + eventname + '.pkl' )
		#fit activity model to all ego networks in dataset
		egonet_fits = pd.read_pickle( saveloc_data + 'egonet_fits_' + eventname + '.pkl' )

		#filter egos according to fitting results
		egonet_filter, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, stat=stat, pval_thres=pval_thres, alphamax=alphamax, alph_thres=alph_thres )

		#print output
		num_egos_filter = len( egonet_filter ) #statistically significant alpha
		frac_egos_random = ( egonet_filter.beta < 1 ).sum() / float( num_egos_filter ) #fraction of egos in random regime (beta < 1, i.e. t_r < alpha_r)
		frac_egos_cumadv = ( egonet_filter.beta > 1 ).sum() / float( num_egos_filter ) #fraction of egos in CA regime (beta > 1, i.e. t_r > alpha_r)
		print( '\thom regime: {:.2f}%, het regime: {:.2f}%'.format( frac_egos_random*100, frac_egos_cumadv*100 ) )

		#print final output
		total_egos_filter += num_egos_filter #statistically significant alpha
		if grid_pos == len(datasets)-1:
			print( '\t\ttotal number of filtered egos: {}'.format( total_egos_filter ) )

		## PLOTTING ##

		#prepare data
		yplot_data = 1 / egonet_filter[ prop_name ] #filtered data!
		xplot, yplot = pm.plot_CCDF_cont( yplot_data ) #complementary cumulative dist

		#plot plot!
		plt.loglog( xplot, yplot, '-', label=textname, c=colors[grid_pos], lw=plot_props['linewidth'] )

	#lines
	plt.axvline( x=1, ls='--', c='0.6', lw=plot_props['linewidth'] )

	#texts
	plt.text( 0.25, 1, 'heterogeneous\n'+r'($\beta > 1$)', va='bottom', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'] )
	plt.text( 0.85, 1, 'homogeneous\n'+r'($\beta < 1$)', va='bottom', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'] )

	#legend
	plt.legend( loc='upper left', bbox_to_anchor=(0.4, -0.25), prop=plot_props['legend_prop'], handlelength=1, numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ] )

	#finalise subplot
	plt.axis([ 1e-5, 1e4, 3e-5, 5e0 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
	ax.locator_params( numticks=6 )


# F: Beta persistence in time for all datasets

	#subplot variables
	binrange = (-4, 4) #for histogram
	bins = 31

	colors = sns.color_palette( 'Set2', n_colors=len(datasets) ) #colors to plot

	print('PERSISTENCE')

	#initialise subplot
	ax = plt.subplot( subgrid[2] )
	sns.despine( ax=ax )

	plt.text( -0.3, 1, 'f', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	#loop through considered datasets
	for grid_pos, (eventname, textname) in enumerate(datasets):
		print( 'dataset name: ' + eventname ) #print output

		## DATA ##

		#load ego network properties / alter activities (all dataset and selected time periods)
		egonet_props = pd.read_pickle( saveloc_data + 'egonet_props_' + eventname + '.pkl' )
		egonet_props_pieces = pd.read_pickle( saveloc_data + 'egonet_props_pieces_' + eventname + '.pkl' )

		#fit activity model in all dataset and selected time periods
		egonet_fits = pd.read_pickle( saveloc_data + 'egonet_fits_' + eventname + '.pkl' )
		egonet_fits_piece_0 = pd.read_pickle( saveloc_data + 'egonet_fits_piece_0_' + eventname + '.pkl' )
		egonet_fits_piece_1 = pd.read_pickle( saveloc_data + 'egonet_fits_piece_1_' + eventname + '.pkl' )

		#filter egos according to fitting results
		egonet_filt, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, stat=stat, pval_thres=pval_thres, alphamax=alphamax, alph_thres=alph_thres )
		egonet_filt_piece_0, egonet_inf_piece_0, egonet_null_piece_0 = dm.egonet_filter( egonet_props_pieces[0], egonet_fits_piece_0, stat=stat, pval_thres=pval_thres, alphamax=alphamax, alph_thres=alph_thres )
		egonet_filt_piece_1, egonet_inf_piece_1, egonet_null_piece_1 = dm.egonet_filter( egonet_props_pieces[1], egonet_fits_piece_1, stat=stat, pval_thres=pval_thres, alphamax=alphamax, alph_thres=alph_thres )

		## PLOTTING ##

		#get property for all egos common to both time periods
		props_filt = pd.concat( [ egonet_filt.beta,
		egonet_filt_piece_0.beta.rename('beta_piece_0'),
		egonet_filt_piece_1.beta.rename('beta_piece_1')
		], axis=1, join='inner' )
		plot_data = ( props_filt.beta_piece_0 - props_filt.beta_piece_1 ) / props_filt.beta #relative diff in property

		#plot plot!
		sns.histplot( x=plot_data, binrange=binrange, bins=bins, stat='density', element='step', fill=False, color=colors[grid_pos], label=textname, zorder=0 )

		#plot perfect persistence line
		plt.axvline( x=0, ls='--', c='0.6', lw=plot_props['linewidth'], zorder=1 )

	#legend
	plt.legend( loc='upper right', bbox_to_anchor=(-0.2, 0.95), prop={ 'size':8 }, handlelength=1, numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ] )

	#finalise subplot
	plt.xlabel( r'$\Delta \beta / \beta$', size=plot_props['xylabel'] )
	plt.ylabel( r'$p_{\Delta \beta}$', size=plot_props['xylabel'] )
	plt.xlim( binrange )
	plt.ylim([ 0, 2 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
	ax.locator_params( axis='x', nbins=4 )
	ax.locator_params( axis='y', nbins=3 )


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
