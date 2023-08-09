#! /usr/bin/env python

#Farsignatures - Exploring dynamics of egocentric communication networks
#Copyright (C) 2023 Gerardo Iñiguez

### SCRIPT FOR PLOTTING FIGURE 3 IN FARSIGNATURES PROJECT ###

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
import plot_misc as pm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#alpha fit variables
	stat = 'KS' #chosen test statistic
	alphamax = 1000 #maximum alpha for MLE fit
	pval_thres = 0.1 #threshold above which alpha MLEs are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf

	#root locations of data/code
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/'
	saveloc = expanduser('~') + '/prg/xocial/Farsignatures/files/data/'
	saveloc_fig = expanduser('~') + '/prg/xocial/Farsignatures/figures/figure1_data/'

	#dataset list: eventname, textname
	datasets = [ #( 'call', 'Mobile (call)'),
				 #( 'text', 'Mobile (sms)'),
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
	'ticklabel' : 15,
	'text_size' : 10,
	'marker_size' : 8,
	'linewidth' : 2,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':10 },
	'legend_hlen' : 1,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables
	fig_props = { 'fig_num' : 3,
	'fig_size' : (11, 8),
	'aspect_ratio' : (2, 3),
	'grid_params' : dict( left=0.07, bottom=0.07, right=0.985, top=0.96, wspace=0.5, hspace=0.4 ),
	'dpi' : 300,
	'savename' : 'figure3' }


	## PLOTTING ##

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'], width_ratios=[1.2, 1, 1] )
	grid.update( **fig_props['grid_params'] )


# A: Phase diagram of selected dataset

	#subplot variables
	eventname, textname = ( 'call', 'Mobile (call)') #selected dataset
	propx = ('gamma', r'\alpha_r')
	propy = ('act_avg_rel', 't_r')
	gridsize = 40 #grid size for hex bins
	vmax = 1e6 #max value in colorbar (larger than [filtered] N in any dataset!)

	print('PHASE DIAGRAM')

	#initialise subplot
	ax = plt.subplot( grid[0,0] )
	sns.despine( ax=ax )
	plt.xlabel( '$'+propx[1]+'$', size=plot_props['xylabel'] )
	plt.ylabel( '$'+propy[1]+'$', size=plot_props['xylabel'] )

	plt.text( -0.28, 0.98, 'a', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	## DATA ##

	#prepare ego network properties
	egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
	#fit activity model to all ego networks in dataset
	egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )

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
	plt.text( 0.8, 0.75, r'crossover ($\beta = 1$)', va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'], rotation=50 )
	het_str = 'heterogeneous\n'+r'($\beta > 1$)'
	plt.text( 0.3, 1, het_str, va='top', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'] )
	hom_str = 'homogeneous\n'+r'($\beta < 1$)'
	plt.text( 0.7, 0.05, hom_str, va='bottom', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'] )

	#finalise subplot
	plt.axis([ 2e-4, 1e3, 1e-3, 5e3 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
	ax.locator_params( numticks=5 )


# B: Aggregated activity distribution in regimes of selected dataset

	#subplot variables
	eventname, textname = ( 'call', 'Mobile (call)') #selected dset
	beta_thres = 1 #beta threshold for activity regimes

	#filter variables
	min_negos = 30 #minimum number of egos in filtered activity group (only for social signature; not used!)

	colors = sns.color_palette( 'Set2', n_colors=2 ) #colors to plot

	print('ACT DIST PER REGIME')

	#initialise subplot
	ax = plt.subplot( grid[0, 1] )
	sns.despine( ax=ax )
	plt.xlabel( '$a$', size=plot_props['xylabel'] )
	plt.ylabel( r"$P[a' \geq a]$", size=plot_props['xylabel'] )

	plt.text( -0.32, 0.98, 'b', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	## DATA ##

	#load ego network properties, alter activities, and alpha fits
	egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
	egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )

	#filter egos according to fitting results
	egonet_filter, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, stat=stat, pval_thres=pval_thres, alphamax=alphamax, alph_thres=alph_thres )

	## PLOTTING ##

	#define beta intervals (het, hom) to aggregate egos:
	beta_lims = ( [ beta_thres, egonet_filter.beta.max() ],
				  [ egonet_filter.beta.min(), beta_thres ] )

	#loop through regimes
	for beta_pos, (beta_min, beta_max) in enumerate(beta_lims):

		#load alter activity CCDF and average social signature (not used!) according to filter
		ccdf, sign = pm.plot_activity_filter( 'divided_to_roughly_40_mb_files_30_march', eventname, filt_rule='beta', filt_obj=egonet_filter.beta, filt_params={ 'min_val':beta_min, 'max_val':beta_max, 'min_negos':min_negos }, is_parallel=True, load=True, root_data=root_data, saveloc=saveloc, saveloc_fig=saveloc_fig )

		symbol = '-' if beta_pos == 0 else '--' #symbols and labels
		label = r'$\beta > 1$' if beta_pos == 0 else r'$\beta < 1$'

		#plot plot alter activity CCDF!
		plt.loglog( ccdf.x, ccdf.y, symbol, c=colors[beta_pos], label=label, lw=plot_props['linewidth']+1, zorder=0 )

	#legend
	plt.legend( loc='upper right', bbox_to_anchor=(1,1), prop=plot_props['legend_prop'], handlelength=2.5, numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'] )

	#finalise subplot
	plt.axis([ 1e0, 1e4, 5e-9, 1.1e0 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
	ax.locator_params( numticks=5 )


# C: CCDF of estimated model parameter for all datasets

	#subplot variables
	prop_name = 'beta'
	prop_label = r'1 / \beta'

	colors = sns.color_palette( 'Set2', n_colors=len(datasets) ) #colors to plot

	print('PARAMETER CCDF')

	#initialise subplot
	ax = plt.subplot( grid[0, 2] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( '${}$'.format( prop_label ), size=plot_props['xylabel'] )
	plt.ylabel( "$P[{}' \geq {}]$".format( prop_label,prop_label ), size=plot_props['xylabel'] )

	plt.text( -0.3, 0.98, 'c', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	#loop through considered datasets
	total_egos_filter = 0 #init counter of all filtered egos
	for grid_pos, (eventname, textname) in enumerate(datasets):
		print( 'dataset name: ' + eventname ) #print output

		## DATA ##

		#prepare ego network properties
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
		#fit activity model to all ego networks in dataset
		egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )

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
	plt.text( 0.25, 1, 'heterogeneous\n'+r'($\beta > 1$)', va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'] )
	plt.text( 0.85, 1, 'homogeneous\n'+r'($\beta < 1$)', va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'] )

	#legend
	plt.legend( loc='upper left', bbox_to_anchor=(0.4, -0.25), prop=plot_props['legend_prop'], handlelength=1, numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ] )

	#finalise subplot
	plt.axis([ 1e-5, 1e3, 5e-5, 2e0 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
	ax.locator_params( numticks=6 )


# D-E: Beta persistence and alter turnover

	#initialise subplot
	subgrid = grid[1,:].subgridspec( 1, 3, wspace=0.5, width_ratios=[1.2,1,0.1] )


# D: Correlation between persistence and turnover in selected dataset

	#subplot variables
	eventname, textname = ( 'call', 'Mobile (call)')

	gridsize = 41 #grid size for hex bin
	vmax = 1e6 #max value in colorbar
	bins = 31 #number of bins in histograms
	range_beta_diff = (-35, 35) #ranges for data
	range_jaccard = (0, 1)

	colors = sns.color_palette( 'GnBu', n_colors=3 ) #colors to plot

	print('CORRELATION')

	#initialise subplot
	plotgrid = subgrid[0].subgridspec( 2, 3, wspace=0.3, hspace=0.15, height_ratios=(0.3, 1), width_ratios=(1, 0.3,0.05) )

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

	## PLOTTING ##

	plot_data = pd.concat( [
	( ( props_filt.beta_piece_0 - props_filt.beta_piece_1 ) / props_filt.beta ).rename('beta_diff'),
	egonet_jaccard.rename('jaccard')
	], axis=1, join='inner' )

	print('\tfiltered N = {}'.format(len(plot_data.index)))

	# D main plot: correlation between turnover (x) and persistence (y)

	#initialise subplot
	ax = plt.subplot( plotgrid[1,0] )
	sns.despine( ax=ax )
	plt.xlabel( '$J$', size=plot_props['xylabel'] )
	plt.ylabel( r'$\Delta \beta / \beta$', size=plot_props['xylabel'] )

	#plot plot!
	hexbin = plt.hexbin( 'jaccard', 'beta_diff', data=plot_data, norm=LogNorm(vmin=1e0, vmax=vmax), mincnt=1, gridsize=gridsize, extent=[*range_jaccard, *range_beta_diff], cmap='GnBu' )

	#finalise plot
	plt.axis([ *range_jaccard, *range_beta_diff ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )

	#colorbar
	cbar = plt.colorbar( hexbin, cax=plt.subplot( plotgrid[1,2] ) )
	cbar.ax.set_title( r'$N_{J, \Delta \beta}$' )
	cbar.ax.minorticks_off()

	# D x-marginal: turnover histogram

	#initialise subplot
	ax = plt.subplot( plotgrid[0,0] )
	sns.despine( ax=ax )
	plt.ylabel( r'$N_J$', size=plot_props['text_size'] )

	plt.text( -0.28, 1.18, 'd', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	#plot plot!
	plt.hist( plot_data.jaccard, bins=bins, range=range_jaccard, log=True, histtype='stepfilled', color=colors[0] )

	#finalise subplot
	plt.axis([ *range_jaccard, 1e0, vmax ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['text_size'], length=2, pad=4 )
	plt.xticks([])

	# D y-marginal: persistence histogram

	#initialise subplot
	ax = plt.subplot( plotgrid[1,1] )
	sns.despine( ax=ax )
	plt.xlabel( r'$N_{\Delta \beta}$', size=plot_props['text_size'] )

	#plot plot!
	plt.hist( plot_data.beta_diff, bins=bins, range=range_beta_diff, log=True, histtype='stepfilled', color=colors[0], orientation='horizontal' )

	#finalise subplot
	plt.axis([ 1e0, vmax, *range_beta_diff ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['text_size'], length=2, pad=4 )
	plt.yticks([])


# E: Beta persistence in time for all datasets

	#subplot variables
	binrange = (-4, 4) #for histogram
	bins = 31

	colors = sns.color_palette( 'Set2', n_colors=len(datasets) ) #colors to plot

	print('PERSISTENCE')

	#initialise subplot
	ax = plt.subplot( subgrid[ 1 ] )
	sns.despine( ax=ax )

	plt.text( -0.21, 1.04, 'e', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	#loop through considered datasets
	for grid_pos, (eventname, textname) in enumerate(datasets):
		print( 'dataset name: ' + eventname ) #print output

		## DATA ##

		#load ego network properties / alter activities (all dataset and selected time periods)
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
		egonet_props_pieces = pd.read_pickle( saveloc + 'egonet_props_pieces_' + eventname + '.pkl' )

		#fit activity model in all dataset and selected time periods
		egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )
		egonet_fits_piece_0 = pd.read_pickle( saveloc + 'egonet_fits_piece_0_' + eventname + '.pkl' )
		egonet_fits_piece_1 = pd.read_pickle( saveloc + 'egonet_fits_piece_1_' + eventname + '.pkl' )

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
		sns.histplot( x=plot_data, binrange=binrange, bins=bins, stat='density', element='step', fill=False, color=colors[grid_pos], zorder=0 )

		#plot perfect persistence line
		plt.axvline( x=0, ls='--', c='0.6', lw=plot_props['linewidth'], zorder=1 )

	#finalise subplot
	plt.xlabel( r'$\Delta \beta / \beta$', size=plot_props['xylabel'] )
	plt.ylabel( r'$p_{\Delta \beta}$', size=plot_props['xylabel'] )
	plt.axis([ binrange[0], binrange[1], -0.05, 1.5 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
	ax.locator_params( axis='x', nbins=4 )
	ax.locator_params( axis='y', nbins=4 )


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
