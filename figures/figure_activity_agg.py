#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (ACTIVITY DIST [AGG]) IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_misc as dm
import plot_misc as pm
import model_misc as mm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#plotting variables
	beta_thres = 1 #beta threshold for activity regimes

	stat = 'KS' #chosen test statistic
	alphamax = 1000 #maximum alpha for MLE fit
	pval_thres = 0.1 #threshold above which alphas are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf

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
	'text_size' : 11,
	'marker_size' : 3,
	'linewidth' : 2,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':12 },
	'legend_hlen' : 1,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables
	fig_props = { 'fig_num' : 1,
	'fig_size' : (10, 8),
	'aspect_ratio' : (4, 4),
	'grid_params' : dict( left=0.08, bottom=0.08, right=0.98, top=0.97, wspace=0.3, hspace=0.5 ),
	'dpi' : 300,
	'savename' : 'figure_activity_agg' }

	colors = sns.color_palette( 'Set2', n_colors=2 ) #colors to plot

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )

	#loop through considered datasets
	for grid_pos, (eventname, textname) in enumerate(datasets):
	# for grid_pos, (eventname, textname) in enumerate([ ( 'MPC_UEu', 'Mobile (call)') ]):
		print( 'dataset name: ' + eventname ) #print output

		## DATA ##

		#load ego network properties, alter activities, and alpha fits
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
		egonet_acts = pd.read_pickle( saveloc + 'egonet_acts_' + eventname + '.pkl' )
		egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )

		#filter egos according to fitting results
		egonet_filter, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, stat=stat, pval_thres=pval_thres, alphamax=alphamax, alph_thres=alph_thres )


		## PLOTTING ##

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos] )
		sns.despine( ax=ax ) #take out spines
		if grid_pos in [10, 11, 12, 13]:
			plt.xlabel( '$a$', size=plot_props['xylabel'] )
		if grid_pos in [0, 4, 8, 12]:
			plt.ylabel( r'CCDF', size=plot_props['xylabel'] )

		#loop through beta intervals (het, hom) to aggregate egos:
		beta_lims = ( [ beta_thres, egonet_filter.beta.max() ],
					  [ egonet_filter.beta.min(), beta_thres ] )
		for beta_pos, (beta_min, beta_max) in enumerate(beta_lims):

			#filtering process (continued)
			#betas between extreme values in selected interval
			egonet_filter_beta = egonet_filter[ egonet_filter.beta.between( beta_min, beta_max ) ]

			#get [aggregated] activity distribution
			acts_filter = egonet_acts.loc[ egonet_filter_beta.index ] #alter activities (only filtered egos with beta in interval)

			if len(acts_filter) > 0: #if there's anything to plot
				#plot data!

				xplot, yplot_data = pm.plot_compcum_dist( acts_filter ) #get alter activity CCDF: P[X >= x]

				symbol = 'o' if beta_pos == 0 else 's'
				label = r'$\beta > 1$' if beta_pos == 0 else r'$\beta < 1$'

				plt.loglog( xplot, yplot_data, symbol, c=colors[beta_pos], label=label, ms=plot_props['marker_size'], zorder=0 )

		#text
		plt.text( 1, 1.15, textname, va='top', ha='right', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		#legend
		if grid_pos == 0:
			plt.legend( loc='upper right', bbox_to_anchor=(1.1, 1), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'] )

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

				# xplot = np.arange( egonet_filter_beta.act_min.min(), egonet_filter_beta.act_max.max()+1, dtype=int ) #activity range
				# yplot_data, not_used = np.histogram( acts_filter, bins=len(xplot), range=( xplot.min()-0.5, xplot.max()+0.5 ), density=True )

				# #plot model!
				#
				# #avg values of model parameters (over filtered egos)
				# t = egonet_filter_beta.act_avg.mean()
				# a0 = int( egonet_filter_beta.act_min.mean() )
				# alpha = egonet_filter_beta.alpha.mean()
				#
				# xplot = np.arange( acts_filter.max()+1, dtype=int ) #activity range
				# yplot_model = 1 - np.cumsum([ mm.activity_dist( a, t, alpha, a0 ) for a in xplot ]) #CCDF of theo activity dist in range a=[0, amax] (i.e. inclusive)
				#
				# plt.loglog( xplot, yplot_model, '-', c=colors[beta_pos], label=None, lw=plot_props['linewidth'], zorder=1 )
