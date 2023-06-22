#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (ACTIVITY DIST AGG BY BETA) IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_misc as dm
import plot_misc as pm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#beta variables
	beta_thres = 1 #beta threshold for activity regimes

	#alpha fit variables
	stat = 'KS' #chosen test statistic
	alphamax = 1000 #maximum alpha for MLE fit
	pval_thres = 0.1 #threshold above which alphas are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf

	#filter variables
	min_negos = 30 #minimum number of egos in filtered activity group (only for social signature)

	#root locations of data/code
	# #LOCAL
	# root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	# root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	# saveloc = root_code+'files/data/' #location of output files
	# saveloc_fig = expanduser('~') + '/prg/xocial/Farsignatures/figures/figure1_data/'
	# # saveloc_fig = ''
	#TRITON
	root_data = '/m/cs/scratch/networks-mobile/heydars1/set5_divided_to_small_files_for_gerardo_29_march_2021/'
	saveloc = '/m/cs/scratch/networks/inigueg1/prg/xocial/Farsignatures/files/data/'
	saveloc_fig = '/m/cs/scratch/networks/inigueg1/prg/xocial/Farsignatures/figures/figure1_data/'

	#flags
	load = False

	#dataset list: eventname, textname
	datasets = [ # ( 'call', 'Mobile (call)'),
				 ( 'text', 'Mobile (sms)'),
				 # ( 'MPC_Wu_SD01', 'Mobile (Wu 1)'),
				 # ( 'MPC_Wu_SD02', 'Mobile (Wu 2)'),
				 # ( 'MPC_Wu_SD03', 'Mobile (Wu 3)'),
				 # ( 'Enron', 'Email (Enron)'),
				 # ( 'email', 'Email (Kiel)'),
				 # ( 'eml2', 'Email (Uni)'),
				 # ( 'email_Eu_core', 'Email (EU)'),
				 # ( 'fb', 'Facebook'),
				 # ( 'messages', 'Messages'),
				 # ( 'pok', 'Dating'),
				 # ( 'forum', 'Forum'),
				 # ( 'CollegeMsg', 'College'),
				 # ( 'CNS_calls', 'CNS (call)'),
				 # ( 'CNS_sms', 'CNS (sms)')
				]

	#sizes/widths/coords
	plot_props = { 'xylabel' : 15,
	'figlabel' : 26,
	'ticklabel' : 15,
	'text_size' : 15,
	'marker_size' : 3,
	'linewidth' : 3,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':7 },
	'legend_hlen' : 1,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables
	fig_props = { 'fig_num' : 1,
	'fig_size' : (10, 8),
	'aspect_ratio' : (4, 4),
	'grid_params' : dict( left=0.08, bottom=0.07, right=0.98, top=0.97, wspace=0.3, hspace=0.4 ),
	'dpi' : 300,
	'savename' : 'figure_activity_beta' }

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )

	#loop through considered datasets
	for grid_pos, (eventname, textname) in enumerate(datasets):
	# for grid_pos, (eventname, textname) in enumerate([ ( 'fb', 'Facebook') ]):
		print( 'dataset name: ' + eventname, flush=True ) #print output

		## PLOTTING ##

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos] )
		sns.despine( ax=ax ) #take out spines
		if grid_pos in [10, 11, 12, 13]:
			plt.xlabel( '$a$', size=plot_props['xylabel'] )
		if grid_pos in [0, 4, 8, 12]:
			plt.ylabel( r"CCDF $P[a' \geq a]$", size=plot_props['xylabel'] )

		colors = sns.color_palette( 'Set2', n_colors=2 ) #colors to plot


		## DATA ##

		#load ego network properties, alter activities, and alpha fits
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
		egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )

		#filter egos according to fitting results
		egonet_filter, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, stat=stat, pval_thres=pval_thres, alphamax=alphamax, alph_thres=alph_thres )

		#define beta intervals (het, hom) to aggregate egos:
		beta_lims = ( [ beta_thres, egonet_filter.beta.max() ],
					  [ egonet_filter.beta.min(), beta_thres ] )

		#dataset naming (relevant for large datasets)
		dataname = 'divided_to_roughly_40_mb_files_30_march' if eventname in ['call', 'text'] else ''
		is_parallel = True if eventname in ['call', 'text'] else False

		#loop through regimes
		for beta_pos, (beta_min, beta_max) in enumerate(beta_lims):
			print('\tbeta_min = {:.2f}, beta_max = {:.2f}'.format(beta_min, beta_max), flush=True) #filter range

			#plot alter activity CCDF and average social signature according to filter
			ccdf, sign = pm.plot_activity_filter( dataname, eventname, filt_rule='beta', filt_obj=egonet_filter.beta, filt_params={ 'min_val':beta_min, 'max_val':beta_max, 'min_negos':min_negos }, is_parallel=is_parallel, load=load, root_data=root_data, saveloc=saveloc, saveloc_fig=saveloc_fig )


			## PLOTTING ##

			symbol = '-' if beta_pos == 0 else '--' #symbols and labels
			label = r'$\beta > 1$' if beta_pos == 0 else r'$\beta < 1$'

			#plot plot alter activity CCDF!
			plt.loglog( ccdf.x, ccdf.y, symbol, c=colors[beta_pos], label=label, lw=plot_props['linewidth'], zorder=0 )

		#texts
		plt.text( 1, 1, textname, va='bottom', ha='right', transform=ax.transAxes, fontsize=plot_props['text_size'] )

		#legend
		plt.legend( loc='upper right', bbox_to_anchor=(1,1), prop=plot_props['legend_prop'], handlelength=3, numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'] )

		#finalise subplot
		plt.axis([ 1e0, 1e4, 1e-6, 1e0 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		ax.locator_params( numticks=5 )
		if grid_pos not in [10, 11, 12, 13]:
			ax.tick_params(labelbottom=False)
		if grid_pos not in [0, 4, 8, 12]:
			ax.tick_params(labelleft=False)

	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )


#DEBUGGIN'

			# plt.loglog( sign.index+1, sign, 'o', c=colors[beta_pos], label=label, ms=plot_props['marker_size'], zorder=0 )
