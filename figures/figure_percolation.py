#! /usr/bin/env python

#Farsignatures - Exploring dynamics of egocentric communication networks
#Copyright (C) 2023 Gerardo Iñiguez

### SCRIPT FOR PLOTTING FIGURE (NODE PERCOLATION BY PROPERTY) IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from os.path import expanduser

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_misc as dm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#plotting variables
	prop = ( 'beta', r'\beta' ) #percolation property

	alphamax = 1000 #maximum alpha for MLE fit
	pval_thres = 0.1 #threshold above which alphas are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf

	#locations
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files

	#dataset list: eventname, textname
	datasets = [ #( 'MPC_UEu', 'Mobile (call)'),
	# datasets = [ ( 'call', 'Mobile (call)'),
	# 			 ( 'text', 'Mobile (sms)'),
				 ( 'MPC_Wu_SD01', 'Mobile (Wu 1)'),
				 ( 'MPC_Wu_SD02', 'Mobile (Wu 2)'),
				 ( 'MPC_Wu_SD03', 'Mobile (Wu 3)'),
				 #( 'sexcontact_events', 'Contact'),
				 #( 'email', 'Email 1'),
				 #( 'eml2', 'Email 2'),
				 ( 'fb', 'Facebook'),
				 #( 'messages', 'Messages'),
				 #( 'forum', 'Forum'),
				 #( 'pok', 'Dating'),
				 #( 'CNS_bt_symmetric', 'CNS (bluetooth)'),
				 ( 'CNS_calls', 'CNS (call)'),
				 ( 'CNS_sms', 'CNS (sms)') ]

	#sizes/widths/coords
	plot_props = { 'xylabel' : 15,
	'figlabel' : 26,
	'ticklabel' : 15,
	'text_size' : 15,
	'marker_size' : 1,
	'linewidth' : 2,
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
	'grid_params' : dict( left=0.07, bottom=0.07, right=0.98, top=0.93, wspace=0.3, hspace=0.3 ),
	'dpi' : 300,
	'savename' : 'figure_percolation' }

	colors = sns.color_palette( 'Paired', n_colors=3 ) #colors to plot

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )

	#loop through considered datasets
	for grid_pos, (eventname, textname) in enumerate(datasets):
	# for grid_pos, (eventname, textname) in enumerate([ ( 'CNS_calls', 'CNS (call)') ]):
		print( 'dataset name: ' + eventname ) #print output

		#load ego network properties / percolation classes
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
		graph_percs = pd.read_pickle( saveloc + 'graph_percs_' + eventname + '.pkl' )
		#fit activity model to all ego networks
		egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )

		#filter egos according to fitting results
		egonet_filt, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )

		num_egos = len(egonet_props) #number of (un-)filtered egos
		num_filt = len(egonet_filt)


		## PLOTTING ##

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos] )
		sns.despine( ax=ax ) #take out spines
		if grid_pos in [10, 11, 12, 13]:
			plt.xlabel( r'$f / n_{\alpha}$', size=plot_props['xylabel'] )
		if grid_pos in [0, 4, 8, 12]:
			plt.ylabel( r"$S_f / S_0$", size=plot_props['xylabel'] )

		#plot arrays!

		#relative fraction of nodes removed
		# xplot = 1 - graph_percs.index[::-1] / graph_percs.index[-1]
		xplot = ( num_egos - graph_percs.index[::-1] ) / float(num_filt)
		#relative LCC by first removing nodes with large/small prop
		yplot_large = graph_percs[ prop[0]+'_large' ].loc[::-1] / graph_percs[ prop[0]+'_large' ].iat[-1]
		yplot_small = graph_percs[ prop[0]+'_small' ].loc[::-1] / graph_percs[ prop[0]+'_small' ].iat[-1]
		yplot_random = graph_percs.random.loc[::-1] / graph_percs.random.iat[-1]

		#plot plot!
		plt.plot( xplot, yplot_large, '-', label='large $'+prop[1]+'$', c=colors[1], lw=plot_props['linewidth'] )
		plt.plot( xplot, yplot_small, '-', label='small $'+prop[1]+'$', c=colors[0], lw=plot_props['linewidth'] )
		plt.plot( xplot, yplot_random, '-', label='random', c=colors[2], lw=plot_props['linewidth'] )

		#texts
		plt.text( 1, 1, textname, va='top', ha='right', transform=ax.transAxes, fontsize=plot_props['text_size'] )

		#legend
		if grid_pos == 1:
			plt.legend( loc='upper left', bbox_to_anchor=(0.2, 1.4), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=3 )

		#finalise subplot
		plt.axis([ 0, 1, 0, 1 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		if grid_pos not in [10, 11, 12, 13]:
			ax.tick_params(labelbottom=False)
		if grid_pos not in [0, 4, 8, 12]:
			ax.tick_params(labelleft=False)


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
