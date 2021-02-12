#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (FIT RANKINGS) IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
import pandas as pd
import seaborn as sns
import scipy.ndimage as sn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser
from matplotlib.colors import LogNorm
from matplotlib.ticker import ( MultipleLocator, LogLocator )

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_misc as dm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#fit/data properties to rank
	propx = ('beta', r'r( \beta )')
	properties_y = [ ('pagerank', r'r( c_p )'),
	 				 ('betweenness', r'r( c_b )'),
					 ('closeness', r'r( c_c )'),
					 ('eigenvector', r'r( c_e )'),
					 ('katz', r'r( c_k )'),
					 ('authority', r'r( c_h )') ]

	max_iter = 1000 #max number of iteration for centrality calculations
	alphamax = 1000 #maximum alpha for MLE fit
	nsims = 1000 #number of syntethic datasets used to calculate p-value
	amax = 10000 #maximum activity for theoretical activity distribution

	pval_thres = 0.1 #threshold above which alphas are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf

	#plot variables
	gridsize = 40 #grid size for hex bins
	vmax = 1e4 #max value in colorbar (larger than [filtered] N in any dataset!)
	filter_factor = 80. #factor for uniform filter

	#locations
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files

	#dataset list: dataname, eventname, textname
	datasets = [ ('MPC_UEu_net', 'MPC_UEu.evt', 'Mobile (call)'),
				 ('SMS_net', 'MPC_Wu_SD01.evt', 'Mobile (Wu 1)'),
				 ('SMS_net', 'MPC_Wu_SD02.evt', 'Mobile (Wu 2)'),
				 ('SMS_net', 'MPC_Wu_SD03.evt', 'Mobile (Wu 3)'),
				 ('sex_contacts_net', 'sexcontact_events.evt', 'Contact'),
				 ('greedy_walk_nets', 'email.evt', 'Email 1'),
				 ('greedy_walk_nets', 'eml2.evt', 'Email 2'),
				 ('greedy_walk_nets', 'fb.evt', 'Facebook'),
				 ('greedy_walk_nets', 'messages.evt', 'Messages'),
				 ('greedy_walk_nets', 'forum.evt', 'Forum'),
				 ('greedy_walk_nets', 'pok.evt', 'Dating'),
				 ('Copenhagen_nets', 'CNS_bt_symmetric.evt', 'CNS (bluetooth)'),
				 ('Copenhagen_nets', 'CNS_calls.evt', 'CNS (call)'),
				 ('Copenhagen_nets', 'CNS_sms.evt', 'CNS (sms)') ]

	#sizes/widths/coords
	plot_props = { 'xylabel' : 15,
	'figlabel' : 26,
	'ticklabel' : 15,
	'text_size' : 15,
	'marker_size' : 6,
	'linewidth' : 2,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':12 },
	'legend_hlen' : 1.7,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#loop through properties to rank
	for prop_pos, propy in enumerate( properties_y ):
#	for prop_pos, propy in enumerate( [ ('katz', r'r( c_k )') ] ):

		print( 'propx = {}, propy = {}'.format( propx[0], propy[0] ) )

		#plot variables
		fig_props = { 'fig_num' : 1,
		'fig_size' : (10, 8),
		'aspect_ratio' : (4, 4),
		'grid_params' : dict( left=0.08, bottom=0.08, right=0.98, top=0.96, wspace=0.2, hspace=0.4 ),
		'width_ratios' : [1, 1, 1, 1.2],
		'dpi' : 300,
		'savename' : 'figure_rankings_{}'.format( propy[0] ) }

		#initialise plot
		sns.set( style='ticks' ) #set fancy fancy plot
		fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
		plt.clf()
		grid = gridspec.GridSpec( *fig_props['aspect_ratio'], width_ratios=fig_props['width_ratios'] )
		grid.update( **fig_props['grid_params'] )

		#loop through considered datasets
		for grid_pos, (dataname, eventname, textname) in enumerate(datasets):
#		for grid_pos, (dataname, eventname, textname) in enumerate([ ('MPC_UEu_net', 'MPC_UEu.evt', 'Mobile (call)') ]):
			print( 'dataset name: ' + eventname[:-4] ) #print output

			## DATA ##

			#prepare ego network / graph properties
			egonet_props, egonet_acts = dm.egonet_props_acts( dataname, eventname, root_data, 'y', saveloc )
			graph_props = dm.graph_props( dataname, eventname, root_data, 'y', saveloc, max_iter=max_iter )

			#fit activity model to all ego networks in dataset
			egonet_fits = dm.egonet_fits( dataname, eventname, root_data, 'y', saveloc, alphamax=alphamax, nsims=nsims, amax=amax )

			#filter egos according to fitting results
			egonet_filter, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, graph_props, egonet_fits, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )

			#some measures
			num_egos_filter = len( egonet_filter ) #filtered egos
			frac_egos_random = ( egonet_filter.beta < 1 ).sum() / float( num_egos_filter ) #fraction of egos in random regime (beta < 1, i.e. t_r < alpha_r)


			## PLOTTING ##

			#initialise subplot
			ax = plt.subplot( grid[ grid_pos] )
			sns.despine( ax=ax ) #take out spines
			if grid_pos in [10, 11, 12, 13]:
				plt.xlabel( '$'+propx[1]+'$', size=plot_props['xylabel'] )
			if grid_pos in [0, 4, 8, 12]:
				plt.ylabel( '$'+propy[1]+'$', size=plot_props['xylabel'] )

			#rank egos according to decreasing properties (and normalize ranks)
			data = egonet_filter[[ propx[0], propy[0] ]].rank( ascending=False, pct=True )

#			yplot_filter = sn.uniform_filter1d( yplot, int(num_egos_filter/filter_factor) )

			#plot plot!
			hexbin = plt.hexbin( propx[0], propy[0], data=data, xscale='log', yscale='log', norm=LogNorm(vmin=1e0, vmax=vmax), mincnt=1, gridsize=gridsize, cmap='copper_r' )
#			hexbin = plt.hexbin( propx[0], propy[0], data=data, norm=LogNorm(vmin=1e0, vmax=vmax), mincnt=1, gridsize=gridsize, cmap='copper_r' )


			#colorbar
			if grid_pos in [3, 7, 11]:
				cbar = plt.colorbar( hexbin, ax=ax )
				cbar.ax.set_title( r'$N_{\bullet, \bullet}$' )
				cbar.ax.minorticks_off()

			# #lines
			# plt.axvline( x=1-frac_egos_random, ls='--', c='k', lw=plot_props['linewidth'] )

			#texts
			plt.text( 0, 1.15, textname, va='top', ha='left', transform=ax.transAxes, fontsize=plot_props['text_size'] )

			#finalise subplot
			plt.axis([ 1e-4, 1, 1e-4, 1 ])
			ax.xaxis.set_major_locator( LogLocator( numticks=4 ) )
			ax.yaxis.set_major_locator( LogLocator( numticks=4 ) )
			ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
			if grid_pos not in [10, 11, 12, 13]:
				ax.tick_params(labelbottom=False)
			if grid_pos not in [0, 4, 8, 12]:
				ax.tick_params(labelleft=False)


		#finalise plot
		if fig_props['savename'] != '':
			plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
