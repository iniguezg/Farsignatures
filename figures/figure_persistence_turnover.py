#! /usr/bin/env python

#Farsignatures - Exploring dynamics of egocentric communication networks
#Copyright (C) 2023 Gerardo Iñiguez

### SCRIPT FOR PLOTTING FIGURE (ALTER PERSISTENCE-TURNOVER CORR) IN FARSIGNATURES PROJECT ###

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

	#alpha fit variables
	stat = 'KS' #chosen test statistic
	alphamax = 1000 #maximum alpha for MLE fit
	pval_thres = 0.1 #threshold above which alphas are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf

	#plotting variables
	gridsize = 41 #grid size for hex bin
	bins = 31 #number of bins in histograms
	range_beta_diff = (-35, 35) #ranges for data
	range_jaccard = (0, 1)

	#locations
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files

	#dataset list: eventname, textname
	datasets = [ #( 'call', 'Mobile (call)'),
				 #( 'text', 'Mobile (sms)'),
				 ( 'MPC_Wu_SD01', 'Mobile (Wu 1)'),
				 ( 'MPC_Wu_SD02', 'Mobile (Wu 2)'),
				 ( 'MPC_Wu_SD03', 'Mobile (Wu 3)'),
				 ( 'Enron', 'Email (Enron)'),
				 #( 'email', 'Email (Kiel)'),
				 #( 'eml2', 'Email (Uni)'),
				 ( 'email_Eu_core', 'Email (EU)'),
				 ( 'fb', 'Facebook'),
				 #( 'messages', 'Messages'),
				 #( 'pok', 'Dating'),
				 #( 'forum', 'Forum'),
				 ( 'CollegeMsg', 'College'),
				 ( 'CNS_calls', 'CNS (call)'),
				 ( 'CNS_sms', 'CNS (sms)') ]

	#sizes/widths/coords
	plot_props = { 'xylabel' : 15,
	'figlabel' : 26,
	'ticklabel' : 13,
	'text_size' : 8,
	'marker_size' : 6,
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
	'grid_params' : dict( left=0.07, bottom=0.07, right=0.965, top=0.965, wspace=0.5, hspace=0.45 ),
	'dpi' : 300,
	'savename' : 'figure_persistence_turnover' }

	colors = sns.color_palette( 'GnBu', n_colors=3 ) #colors to plot

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )

	#loop through considered datasets
	for grid_pos, (eventname, textname) in enumerate(datasets):
		print( 'dataset name: ' + eventname ) #print output

		#initialise subplot
		plotgrid = grid[grid_pos].subgridspec( 2, 3, wspace=0.3, hspace=0.15, height_ratios=(0.3, 1), width_ratios=(1, 0.3,0.1) )

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
		egonet_filt, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, stat=stat, pval_thres=pval_thres, alphamax=alphamax, alph_thres=alph_thres )
		egonet_filt_piece_0, egonet_inf_piece_0, egonet_null_piece_0 = dm.egonet_filter( egonet_props_pieces[0], egonet_fits_piece_0, stat=stat, pval_thres=pval_thres, alphamax=alphamax, alph_thres=alph_thres )
		egonet_filt_piece_1, egonet_inf_piece_1, egonet_null_piece_1 = dm.egonet_filter( egonet_props_pieces[1], egonet_fits_piece_1, stat=stat, pval_thres=pval_thres, alphamax=alphamax, alph_thres=alph_thres )

		#get property for all egos common to both time periods
		props_filt = pd.concat( [ egonet_filt.beta,
		egonet_filt_piece_0.beta.rename('beta_piece_0'),
		egonet_filt_piece_1.beta.rename('beta_piece_1')
		], axis=1, join='inner' )

		## PLOTTINMG ##

		plot_data = pd.concat( [
		( ( props_filt.beta_piece_0 - props_filt.beta_piece_1 ) / props_filt.beta ).rename('beta_diff'),
		egonet_jaccard.rename('jaccard')
		], axis=1, join='inner' )

		print('\tfiltered N = {}'.format(len(plot_data.index)))


		## PLOTTING ##

		#main subplot: correlation between turnover (x) and persistence (y)

		#initialise subplot
		ax = plt.subplot( plotgrid[1,0] )
		sns.despine( ax=ax )
		if grid_pos in [12, 13, 14, 15]:
			plt.xlabel( '$J$', size=plot_props['xylabel'] )
		if grid_pos in [0, 4, 8, 12]:
			plt.ylabel( r'$\Delta \beta / \beta$', size=plot_props['xylabel'] )

		#plot plot!
		vmax = len(egonet_filt) #max value in colorbar (total number of egos in filtered dataset)
		hexbin = plt.hexbin( 'jaccard', 'beta_diff', data=plot_data, norm=LogNorm(vmin=1e0, vmax=vmax), mincnt=1, gridsize=gridsize, extent=[*range_jaccard, *range_beta_diff], cmap='GnBu' )

		#finalise plot
		plt.axis([ *range_jaccard, *range_beta_diff ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )

		#colorbar
		ax=plt.subplot( plotgrid[1,2] ).set_axis_off()
		cbar = plt.colorbar( hexbin, ax=ax, fraction=1 )
		cbar.ax.set_title( r'$N_{J, \Delta \beta}$' )
		cbar.ax.minorticks_off()

		#x marginal: turnover histogram

		#initialise subplot
		ax = plt.subplot( plotgrid[0,0] )
		sns.despine( ax=ax )
		plt.ylabel( r'$N_J$', size=plot_props['text_size'] )

		plt.text( 0, 1.1, textname, va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		#plot plot!
		plt.hist( plot_data.jaccard, bins=bins, range=range_jaccard, log=True, histtype='stepfilled', color=colors[0] )

		#finalise subplot
		plt.axis([ *range_jaccard, 1e0, vmax ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['text_size'], length=2, pad=4 )
		ax.locator_params( axis='y', numticks=2 )
		plt.xticks([])

		#y marginal: persistence histogram

		#initialise subplot
		ax = plt.subplot( plotgrid[1,1] )
		sns.despine( ax=ax )
		plt.xlabel( r'$N_{\Delta \beta}$', size=plot_props['text_size'] )

		#plot plot!
		plt.hist( plot_data.beta_diff, bins=bins, range=range_beta_diff, log=True, histtype='stepfilled', color=colors[0], orientation='horizontal' )

		#finalise subplot
		plt.axis([ 1e0, vmax, *range_beta_diff ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['text_size'], length=2, pad=4 )
		ax.locator_params( axis='x', numticks=2 )
		plt.yticks([])

	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
