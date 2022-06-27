#! /usr/bin/env python

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
	alphamax = 1000 #maximum alpha for MLE fit
	pval_thres = 0.1 #threshold above which alpha MLEs are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf

	#locations
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files

	#dataset list: eventname, textname
	# datasets = [ ( 'MPC_UEu', 'Mobile (call)'),
	datasets = [ ( 'call', 'Mobile (call)'),
				 ( 'text', 'Mobile (sms)'),
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
	'fig_size' : (10, 8),
	'aspect_ratio' : (2, 2),
	'grid_params' : dict( left=0.08, bottom=0.07, right=0.85, top=0.96, wspace=0.4, hspace=0.4 ),
	'dpi' : 300,
	'savename' : 'figure3' }


	## PLOTTING ##

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )


# A: Phase diagram of selected dataset

	#subplot variables
	eventname, textname = ( 'text', 'Mobile (sms)') #selected dset
	propx = ('gamma', r'\alpha_r')
	propy = ('act_avg_rel', 't_r')
	gridsize = 40 #grid size for hex bins
	vmax = 3e6 #max value in colorbar (larger than [filtered] N in any dataset!)

	print('PHASE DIAGRAM')

	#initialise subplot
	ax = plt.subplot( grid[ 0,0 ] )
	sns.despine( ax=ax )
	plt.xlabel( '$'+propx[1]+'$', size=plot_props['xylabel'] )
	plt.ylabel( '$'+propy[1]+'$', size=plot_props['xylabel'] )

	plt.text( -0.26, 0.98, 'a', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )
	plt.text( 0, 1.03, textname, va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

	## DATA ##

	#load ego network properties, alter activities, and alpha fits
	egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
	egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )

	#filter egos according to fitting results
	egonet_filter, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )

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
	hexbin = plt.hexbin( propx[0], propy[0], data=egonet_filter, xscale='log', yscale='log', norm=LogNorm(vmin=1e0), mincnt=1, gridsize=gridsize, cmap='GnBu', zorder=0 )

	#colorbar
	cbar = plt.colorbar( hexbin, ax=ax, fraction=0.05 )
	cbar.ax.set_title( r'$N_{'+propx[1][:-2]+','+propy[1][:-2]+'}$' )
	cbar.ax.minorticks_off()

	#lines
	plt.plot( [1e-3, 1e4], [1e-3, 1e4], '--', c='0.6', lw=plot_props['linewidth'], zorder=1 )

	#texts
	plt.text( 0.8, 0.7, r'crossover ($\beta = 1$)', va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'], rotation=45 )

	#finalise subplot
	plt.axis([ 1e-3, 1e3, 1e-2, 1e4 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
	ax.locator_params( numticks=5 )


# B: CCDF of estimated model parameter for all datasets

	#subplot variables
	prop_name = 'beta'
	prop_label = r'1 / \beta'

	colors = sns.color_palette( 'Set2', n_colors=len(datasets) ) #colors to plot

	print('PARAMETER CCDF')

	#initialise subplot
	ax = plt.subplot( grid[ 0, 1 ] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( '${}$'.format( prop_label ), size=plot_props['xylabel'] )
	plt.ylabel( 'CCDF', size=plot_props['xylabel'] )

	plt.text( -0.21, 0.98, 'b', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

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
		egonet_filter, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )

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
	plt.text( 0.25, 1, 'heterogeneous\nregime', va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'], weight='bold' )
	plt.text( 0.85, 1, 'homogeneous\nregime', va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'], weight='bold' )

	#legend
	plt.legend( loc='lower left', bbox_to_anchor=(1, -0.55), prop=plot_props['legend_prop'], handlelength=1, numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ] )

	#finalise subplot
	plt.axis([ 1e-5, 1e4, 3e-5, 5e0 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
	ax.locator_params( numticks=6 )


# C: Aggregated activity distribution in regimes

	#subplot variables
	eventname, textname = ( 'fb', 'Facebook') #selected dset
	beta_thres = 1 #beta threshold for activity regimes

	colors = sns.color_palette( 'Set2', n_colors=2 ) #colors to plot

	print('ACT DIST PER REGIME')

	#initialise subplot
	ax = plt.subplot( grid[ 1,0 ] )
	sns.despine( ax=ax )
	plt.xlabel( '$a$', size=plot_props['xylabel'] )
	plt.ylabel( r'CCDF', size=plot_props['xylabel'] )

	plt.text( -0.24, 0.98, 'c', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )
	plt.text( 0, 1.03, textname, va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

	## DATA ##

	#load ego network properties, alter activities, and alpha fits
	egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
	egonet_acts = pd.read_pickle( saveloc + 'egonet_acts_' + eventname + '.pkl' )
	egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )

	#filter egos according to fitting results
	egonet_filter, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )


	## PLOTTING ##

	#define beta intervals (het, hom) to aggregate egos:
	beta_lims = ( [ beta_thres, egonet_filter.beta.max() ],
				  [ egonet_filter.beta.min(), beta_thres ] )


	# C1: Activity distributions

	#loop through regimes
	for beta_pos, (beta_min, beta_max) in enumerate(beta_lims):

		#betas between extreme values in selected interval
		egonet_filter_beta = egonet_filter[ egonet_filter.beta.between( beta_min, beta_max ) ]
		acts_filter = egonet_acts.loc[ egonet_filter_beta.index ] #alter activities (only filtered egos with beta in interval)

		xplot, yplot_data = pm.plot_compcum_dist( acts_filter ) #get alter activity CCDF: P[X >= x]

		symbol = '-' if beta_pos == 0 else '--'
		label = r'$\beta > 1$' if beta_pos == 0 else r'$\beta < 1$'
		plt.loglog( xplot, yplot_data, symbol, c=colors[beta_pos], label=label, lw=plot_props['linewidth']+1 )

	#legend
	plt.legend( loc='lower left', bbox_to_anchor=(0, 0), prop=plot_props['legend_prop'], handlelength=2.5, numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'] )

	#finalise subplot
	plt.axis([ 1e0, 1e4, 1e-6, 1e0 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
	ax.locator_params( numticks=5 )


	#C2: Social signatures

	inax = ax.inset_axes([ 0.6, 0.6, 0.4, 0.4 ])
	sns.despine( ax=inax ) #take out spines
	inax.set_xlabel( r'alter rank', size=plot_props['text_size'], labelpad=0 )
	inax.set_ylabel( r'activity fraction', size=plot_props['text_size'], labelpad=0 )

	#loop through regimes
	for beta_pos, (beta_min, beta_max) in enumerate(beta_lims):

		#betas between extreme values in selected interval
		egonet_filter_beta = egonet_filter[ egonet_filter.beta.between( beta_min, beta_max ) ]
		acts_filter = egonet_acts.loc[ egonet_filter_beta.index ] #alter activities (only filtered egos with beta in interval)

		xplot = np.arange( 1, len(acts_filter)+1, dtype=int )
		yplot = acts_filter.sort_values( ascending=False ) / acts_filter.sum()

		symbol = '-' if beta_pos == 0 else '--'
		inax.loglog( xplot, yplot, symbol, c=colors[beta_pos], ms=plot_props['marker_size']-5, lw=plot_props['linewidth'] )

	#finalise inset
	inax.set_xlim( 1e0, 1e6)
	inax.set_ylim( 5e-7, 2e-3 )
	inax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['text_size'], length=2, pad=3 )


# D: Beta persistence in time

	#subplot variables
	binrange = (-4, 4) #for histogram
	bins = 31

	#TEMPORARY: used datasets
	datasets = [ ( 'MPC_UEu', 'Mobile (call)'),
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

	colors = sns.color_palette( 'Set2', n_colors=len(datasets) ) #colors to plot

	print('PERSISTENCE')

	#initialise subplot
	ax = plt.subplot( grid[ 1,1 ] )
	sns.despine( ax=ax )

	plt.text( -0.21, 0.98, 'd', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	#loop through considered datasets
	for grid_pos, (eventname, textname) in enumerate(datasets):
		print( 'dataset name: ' + eventname ) #print output

		## DATA ##

		#load ego network properties
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
		#fit activity model in all dataset and selected time periods
		egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )
		egonet_fits_piece_0 = pd.read_pickle( saveloc + 'egonet_fits_piece_0_' + eventname + '.pkl' )
		egonet_fits_piece_1 = pd.read_pickle( saveloc + 'egonet_fits_piece_1_' + eventname + '.pkl' )

		#filter egos according to fitting results
		egonet_filt, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )
		egonet_filt_piece_0, egonet_inf_piece_0, egonet_null_piece_0 = dm.egonet_filter( egonet_props, egonet_fits_piece_0, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )
		egonet_filt_piece_1, egonet_inf_piece_1, egonet_null_piece_1 = dm.egonet_filter( egonet_props, egonet_fits_piece_1, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )


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
	plt.ylabel( 'PDF', size=plot_props['xylabel'] )
	plt.xlim( binrange )
	plt.ylim([ 0, 2 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
	ax.locator_params( axis='x', nbins=4 )
	ax.locator_params( axis='y', nbins=3 )


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
