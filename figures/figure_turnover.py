#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (ALTER TURNOVER) IN FARSIGNATURES PROJECT ###

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
	bins = 30 #number of bins in histogram
	gridsize = 40 #grid size for hex bins
	vmax = 2e4 #max value in colorbar (larger than [filtered] N in any dataset!)

	alphamax = 1000 #maximum alpha for MLE fit
	pval_thres = 0.1 #threshold above which alphas are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf

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
	'grid_params' : dict( left=0.075, bottom=0.07, right=0.96, top=0.96, wspace=0.2, hspace=0.3 ),
	'dpi' : 300,
	'savename' : 'figure_turnover' }

	colors = sns.color_palette( 'Paired', n_colors=1 ) #colors to plot

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

		## DATA ##

		#load ego network properties / alter activities (all dataset and selected time periods)
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
		egonet_props_pieces = pd.read_pickle( saveloc + 'egonet_props_pieces_' + eventname + '.pkl' )

		#compute Jaccard index of neighbor sets
		egonet_jaccard = dm.egonet_jaccard( eventname, 'y', saveloc )

		#fit activity model in all dataset and selected time periods
		egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )
		egonet_fits_piece_0 = pd.read_pickle( saveloc + 'egonet_fits_piece_0_' + eventname + '.pkl' )
		egonet_fits_piece_1 = pd.read_pickle( saveloc + 'egonet_fits_piece_1_' + eventname + '.pkl' )

		#filter egos according to fitting results
		egonet_filt, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )
		egonet_filt_piece_0, egonet_inf_piece_0, egonet_null_piece_0 = dm.egonet_filter( egonet_props_pieces[0], egonet_fits_piece_0, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )
		egonet_filt_piece_1, egonet_inf_piece_1, egonet_null_piece_1 = dm.egonet_filter( egonet_props_pieces[1], egonet_fits_piece_1, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )

		#get property for all egos common to both time periods
		props_filt = pd.concat( [ egonet_filt.beta,
		egonet_filt_piece_0.beta.rename('beta_piece_0'),
		egonet_filt_piece_1.beta.rename('beta_piece_1')
		], axis=1, join='inner' )


		# ## PLOTTING DIST ##
		#
		# #initialise subplot
		# ax = plt.subplot( grid[ grid_pos] )
		# sns.despine( ax=ax ) #take out spines
		#
		# #plot plot!
		# sns.histplot( x=egonet_jaccard, bins=bins, stat='density', element='step', color=colors[0] )
		#
		# #texts
		# plt.text( 1, 1, textname, va='bottom', ha='right', transform=ax.transAxes, fontsize=plot_props['text_size'] )
		#
		# if grid_pos in [10, 11, 12, 13]:
		# 	plt.xlabel( r'$J$', size=plot_props['xylabel'] )
		# else:
		# 	plt.xlabel('')
		# if grid_pos in [0, 4, 8, 12]:
		# 	plt.ylabel( 'PDF', size=plot_props['xylabel'] )
		# else:
		# 	plt.ylabel('')
		#
		# #finalise subplot
		# ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )


		## PLOTTINMG CORRS ##

		plot_data = pd.concat( [
		( ( props_filt.beta_piece_0 - props_filt.beta_piece_1 ) / props_filt.beta ).rename('beta_diff'),
		egonet_jaccard.rename('jaccard')
		], axis=1, join='inner' )

		print('\tfiltered N = {}'.format(len(plot_data.index)))

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos] )
		sns.despine( ax=ax ) #take out spines

		#plot plot!
		hexbin = plt.hexbin( 'jaccard', 'beta_diff', data=plot_data, xscale='lin', yscale='lin', norm=LogNorm(vmin=1e0, vmax=vmax), mincnt=1, gridsize=(gridsize,gridsize), cmap='copper_r' )

		#colorbar
		if grid_pos in [3, 7, 11, 13]:
			cbar = plt.colorbar( hexbin, ax=ax, fraction=0.05 )
			cbar.ax.set_title( r'$N_{J, \beta}$' )
			cbar.ax.minorticks_off()

		#texts
		plt.text( 0, 1.15, textname, va='top', ha='left', transform=ax.transAxes, fontsize=plot_props['text_size'] )

		if grid_pos in [10, 11, 12, 13]:
			plt.xlabel( '$J$', size=plot_props['xylabel'] )
		else:
			plt.xlabel('')
			plt.xticks([])
		if grid_pos in [0, 4, 8, 12]:
			plt.ylabel( r'$\Delta \beta / \beta$', size=plot_props['xylabel'] )
		else:
			plt.ylabel('')
			plt.yticks([])

		#finalise subplot
		plt.axis([ 0, 1, -10, 10 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		# ax.locator_params( numticks=5 )

	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
