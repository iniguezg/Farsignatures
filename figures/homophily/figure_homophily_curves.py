#! /usr/bin/env python

#Farsignatures - Exploring dynamics of egocentric communication networks
#Copyright (C) 2023 Gerardo Iñiguez

### SCRIPT FOR PLOTTING FIGURE (HOMOPHILY CURVES) IN FARSIGNATURES PROJECT ###

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
				 #( 'email', 'Email 1'),
				 #( 'eml2', 'Email 2'),
				 ( 'fb', 'Facebook'),
				 #( 'messages', 'Messages'),
				 #( 'forum', 'Forum'),
				 #( 'pok', 'Dating'),
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
	'grid_params' : dict( left=0.07, bottom=0.08, right=0.98, top=0.965, wspace=0.2, hspace=0.4 ),
	'width_ratios' : [1, 1, 1, 1.2],
	'dpi' : 300,
	'savename' : 'figure_homophily_curves' }

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

		#set up null model for z-score calculation
		#mean/sdev of absolute property difference over linked ego/alters
		diff_avg = np.abs( egonet_ranks_props.ego_prop - egonet_ranks_props.alter_prop ).mean()
		diff_std = np.abs( egonet_ranks_props.ego_prop - egonet_ranks_props.alter_prop ).std()
		#mean/sdev of absolute property difference (in filtered system)
		# diff_avg = egonet_filt.beta.mean()
		# diff_std = egonet_filt.beta.std()

		#classify ego/alter ranks into intervals
		egonet_ranks_props['ego_interval'] = pd.cut( egonet_ranks_props.ego_rank, bins, labels=bin_centers )
		egonet_ranks_props['alter_interval'] = pd.cut( egonet_ranks_props.alter_rank, bins, labels=bin_centers )
		# egonet_ranks_props['prop_diff'] = np.abs( egonet_ranks_props.ego_prop - egonet_ranks_props.alter_prop )
		egonet_ranks_props['prop_zscore'] = ( np.abs( egonet_ranks_props.ego_prop - egonet_ranks_props.alter_prop ) - diff_avg ) / diff_std

		#group by ego/alter rank intervals
		# interval_groups = egonet_ranks_props[['ego_interval', 'alter_interval', 'prop_diff' ]].groupby(['ego_interval', 'alter_interval'])
		interval_groups = egonet_ranks_props[['ego_interval', 'alter_interval', 'prop_zscore' ]].groupby(['ego_interval', 'alter_interval'])

		#calculate average of absolute property difference in interval pair and re-organize
		# prop_diffs = interval_groups.mean().reset_index()
		#calculate avg/sdev of property z-score in interval pair and re-organize
		prop_zscore_means = interval_groups.mean().reset_index()
		prop_zscore_sdevs = interval_groups.std().reset_index()


		## PLOTTING ##

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos] )
		sns.despine( ax=ax ) #take out spines
		if grid_pos in [10, 11, 12, 13]:
			plt.xlabel( r'$r_{\bullet}$', size=plot_props['xylabel'] )
		if grid_pos in [0, 4, 8, 12]:
			plt.ylabel( r'$\langle z \rangle \pm \sigma_z$', size=plot_props['xylabel'] )

		#plot plot!

		# plt.errorbar( prop_zscore_means[::nbins+1].ego_interval, prop_zscore_means[::nbins+1].prop_zscore, yerr=prop_zscore_sdevs[::nbins+1].prop_zscore, fmt='-', c=colors[0], label='restricted', lw=plot_props['linewidth'], zorder=1 )
		# sns.boxplot( x='ego_interval', y='prop_zscore', data=egonet_ranks_props, zorder=1 )
		sns.violinplot( x='ego_interval', y='prop_zscore', data=egonet_ranks_props, zorder=1 )

		plt.axhline( ls='--', c='k', lw=1, zorder=0 )

		#texts

		plt.text( 0, 1.15, textname, va='top', ha='left', transform=ax.transAxes, fontsize=plot_props['text_size'] )

		#finalise subplot
		# plt.axis([ 1, np.power(10, max_logrank), -1, 1 ])
		plt.axis([ 1, 10, -1, 1 ])
		# plt.xscale('log')
		# ax.xaxis.set_major_locator( LogLocator( numticks=4 ) )
		ax.tick_params( axis='both', which='both', direction='out', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		if grid_pos not in [10, 11, 12, 13]:
			ax.tick_params(labelbottom=False)
		if grid_pos not in [0, 4, 8, 12]:
			ax.tick_params(labelleft=False)


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
