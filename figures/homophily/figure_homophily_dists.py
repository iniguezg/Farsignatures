#! /usr/bin/env python

#Farsignatures - Exploring dynamics of egocentric communication networks
#Copyright (C) 2023 Gerardo Iñiguez

### SCRIPT FOR PLOTTING FIGURE (HOMOPHILY DISTRIBUTIONS) IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
# import graph_tool.all as gt #source activate gt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser
from matplotlib.colors import LogNorm
from matplotlib.ticker import ( MultipleLocator, LogLocator )

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_misc as dm
import plot_misc as pm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#property to correlate
	prop_name = 'beta'

	alphamax = 1000 #maximum alpha for MLE fit
	pval_thres = 0.1 #threshold above which alpha MLEs are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf

	nbins = 10 #number of bins
	max_logrank = 3 #maximum (log) rank of ego/alter

	bins = np.logspace(0, max_logrank, nbins+1) #bins/centers for homophily plot
	bin_centers = np.logspace(0, max_logrank, 2*nbins+1)[1:-1:2]

	#locations
	loadflag = 'y'
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files

	#dataset list: eventname, textname
	# datasets = [ ( 'fb', 'Facebook'),
	datasets = [ #( 'MPC_UEu', 'Mobile (call)'),
	# datasets = [ ( 'call', 'Mobile (call)'),
				 # ( 'text', 'Mobile (sms)'),
				 ( 'MPC_Wu_SD01', 'Mobile (Wu 1)'),
				 ( 'MPC_Wu_SD02', 'Mobile (Wu 2)'),
				 ( 'MPC_Wu_SD03', 'Mobile (Wu 3)'),
				 #( 'sexcontact_events', 'Contact'),
				 ( 'email', 'Email 1'),
				 ( 'eml2', 'Email 2'),
				 ( 'fb', 'Facebook'),
				 ( 'messages', 'Messages'),
				 ( 'forum', 'Forum'),
				 ( 'pok', 'Dating'),
				 #( 'CNS_bt_symmetric', 'CNS (bluetooth)'),
				 ( 'CNS_calls', 'CNS (call)'),
				 ( 'CNS_sms', 'CNS (sms)')
				]

	#sizes/widths/coords
	plot_props = { 'xylabel' : 15,
	'figlabel' : 26,
	'ticklabel' : 15,
	'text_size' : 15,
	'marker_size' : 5,
	'linewidth' : 3,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':15 },
	'legend_hlen' : 1,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables
	fig_props = { 'fig_num' : 1,
	'fig_size' : (10, 8),
	'aspect_ratio' : (4, 4),
	'grid_params' : dict( left=0.085, bottom=0.085, right=0.99, top=0.92, wspace=0.2, hspace=0.4 ),
	'width_ratios' : [1, 1, 1, 1.2],
	'dpi' : 300,
	'savename' : 'figure_homophily_dists' }

	colors = sns.color_palette( 'Paired', n_colors=1 )

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'], width_ratios=fig_props['width_ratios'] )
	grid.update( **fig_props['grid_params'] )

	#loop through considered datasets
	for grid_pos, (eventname, textname) in enumerate(datasets):
		print( 'dataset name: ' + eventname ) #print output

		## DATA ##

		#prepare ego network properties
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
		#fit activity model to all ego networks
		egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )

		#filter egos according to fitting results
		egonet_filt, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )

		#get connected ego/alter activity ranks / properties for dataset
		egonet_ranks_props = dm.egonet_ranks_props( eventname, loadflag, saveloc, prop_name=prop_name, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )

		#set up baseline (null model) for homophily analysis
		#absolute property difference over linked ego/alters
		diff_abs = np.abs( egonet_ranks_props.ego_prop - egonet_ranks_props.alter_prop )

		#get data restricted to rank tuple
		# data = egonet_ranks_props.loc[ ( egonet_ranks_props.ego_rank.astype(int)==1 ) & ( egonet_ranks_props.alter_rank.astype(int)==1 ) ]
		data = egonet_ranks_props.loc[ ( egonet_ranks_props.ego_rank.astype(int).isin( range(1, 2) ) ) & ( egonet_ranks_props.alter_rank.astype(int).isin( range(1, 2) ) ) ]

		data_diff_abs = np.abs( data.ego_prop - data.alter_prop )


		## PLOTTING ##

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos] )
		sns.despine( ax=ax ) #take out spines
		if grid_pos in [10, 11, 12, 13]:
			plt.xlabel( r'$|\Delta \alpha |$', size=plot_props['xylabel'] )
		if grid_pos in [0, 4, 8, 12]:
			plt.ylabel( 'CCDF', size=plot_props['xylabel'] )

		#print stats
		print( '\tno. alpha pairs = {}'.format( diff_abs.size ) )
		print( '\tno. alpha pairs (restricted) = {}'.format( data_diff_abs.size ) )
		# print( 'min/max abs alpha difference = {}, {}'.format( diff_abs.min(), diff_abs.max() ) )

		#plot plot!

		#plot reference baseline (null model)
		xplot_ref, yplot_ref = pm.plot_CCDF_cont( diff_abs ) #complementary cumulative dist
		plt.loglog( xplot_ref, yplot_ref, '--', c='k', label='reference', lw=plot_props['linewidth'] )
		plt.axvline( x=xplot_ref.mean(), label=None, ls='--', c='k', lw=1 )

		#plot restricted data
		xplot, yplot = pm.plot_CCDF_cont( data_diff_abs ) #complementary cumulative dist
		plt.loglog( xplot, yplot, '-', c=colors[0], label='restricted', lw=plot_props['linewidth'] )
		plt.axvline( x=xplot.mean(), label=None, ls='-', c=colors[0], lw=1 )

		#legend
		if grid_pos == 0:
			leg = plt.legend( loc='lower left', bbox_to_anchor=(1, 1.1), prop=plot_props['legend_prop'], handlelength=1.7, numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=2 )

		#texts
		plt.text( 0, 1.15, textname, va='top', ha='left', transform=ax.transAxes, fontsize=plot_props['text_size'] )

		#finalise subplot
		plt.axis([ 1e-1, 1e4, 1e-3, 1e0 ])
		ax.xaxis.set_major_locator( LogLocator( numticks=6 ) )
		ax.yaxis.set_major_locator( LogLocator( numticks=5 ) )
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		if grid_pos not in [10, 11, 12, 13]:
			ax.tick_params(labelbottom=False)
		if grid_pos not in [0, 4, 8, 12]:
			ax.tick_params(labelleft=False)


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
