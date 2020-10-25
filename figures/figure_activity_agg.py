#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (ACTIVITY DIST [AGG]) IN FARSIGNATURES PROJECT ###

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
import model_misc as mm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	bounds = (0, 1000) #bounds for alpha MLE fit
	nsims = 100 #number of syntethic datasets used to calculate p-value
	amax = 10000 #maximum activity for theoretical activity distribution

	pval_thres = 0.1 #threshold above which alphas are considered
	alpha_lims = [ (1e-4, 1e0), (1e0, 1e2) ] #alpha intervals to aggregate egos
#	alpha_lims = [ (1e-4, 1e2) ]

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
	'text_size' : 11,
	'marker_size' : 3,
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
	'grid_params' : dict( left=0.08, bottom=0.08, right=0.98, top=0.97, wspace=0.3, hspace=0.5 ),
	'dpi' : 300,
	'savename' : 'figure_activity_agg' }

	colors = sns.color_palette( 'Set2', n_colors=len(alpha_lims) ) #colors to plot

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )

	#loop through considered datasets
	for grid_pos, (dataname, eventname, textname) in enumerate(datasets):
#	for grid_pos, (dataname, eventname, textname) in enumerate([ ('MPC_UEu_net', 'MPC_UEu.evt', 'Mobile (call)') ]):
		print( 'dataset name: ' + eventname[:-4] ) #print output

		## DATA ##

		#prepare ego network properties
		egonet_props, egonet_acts = dm.egonet_props_acts( dataname, eventname, root_data, 'y', saveloc )

		#fit activity model to all ego networks in dataset
		egonet_fits = dm.egonet_fits( dataname, eventname, root_data, 'y', saveloc, bounds=bounds, nsims=nsims, amax=amax )

		#join (ego) properties and fits
		props_fits = pd.concat( [ egonet_props, egonet_fits ], axis=1 )

		#filtering process
		#step 1: egos with t > a_0
		props_fits_filter = props_fits[ props_fits.degree * props_fits.act_min < props_fits.strength ]
		#step 2: egos with pvalue > threshold
		props_fits_filter = props_fits_filter[ props_fits_filter.pvalue > pval_thres ]


		## PLOTTING ##

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos] )
		sns.despine( ax=ax ) #take out spines
		if grid_pos in [10, 11, 12, 13]:
			plt.xlabel( 'activity $a$', size=plot_props['xylabel'] )
		if grid_pos in [0, 4, 8, 12]:
			plt.ylabel( r"PDF $p_a(t)$", size=plot_props['xylabel'] )

		#loop through alpha intervals to aggregate egos
		for alpha_pos, (alpha_min, alpha_max) in enumerate( alpha_lims ):

			#filtering process (continued)
			#step 3: alphas between extreme values in selected interval
			props_fits_alpha = props_fits_filter[ props_fits_filter.alpha.between( alpha_min, alpha_max ) ]

			#get [aggregated] activity distribution
			acts_filter = egonet_acts.loc[ props_fits_alpha.index ] #alter activities (filtered egos only)

			if len(acts_filter) > 0: #if there's anything to plot
				#plot data!

				xplot = np.arange( props_fits_alpha.act_min.min(), props_fits_alpha.act_max.max()+1, dtype=int ) #activity range
				yplot_data, not_used = np.histogram( acts_filter, bins=len(xplot), range=( xplot.min()-0.5, xplot.max()+0.5 ), density=True )

#				label = 'data' if alpha_pos == 0 else None
				symbol = 'o' if alpha_pos == 0 else 's'

				plt.loglog( xplot, yplot_data, symbol, c=colors[alpha_pos], label=None, ms=plot_props['marker_size'], zorder=0 )

				#plot model!

				#average values of model parameters (over filtered egos)
				t = props_fits_alpha.act_avg.mean()
				alpha = props_fits_alpha.alpha.mean()
				a0 = int( props_fits_alpha.act_min.mean() )

				xplot = np.arange( a0, 1e2, dtype=int ) #activity range
				yplot_model = np.array([ mm.activity_dist( a, t, alpha, a0 ) for a in xplot ])

				label = r'$10^{'+str(int(np.log10(alpha_min)))+r'} \leq \alpha \leq 10^{'+str(int(np.log10(alpha_max)))+'}$'

				plt.loglog( xplot, yplot_model, '-', c=colors[alpha_pos], label=label, lw=plot_props['linewidth'], zorder=1 )

		#text
		plt.text( 1, 1.15, textname, va='top', ha='right', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		#legend
		if grid_pos == 13:
			plt.legend( loc='upper left', bbox_to_anchor=(1.1, 1), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'] )

		#finalise subplot
		plt.axis([ 5e-1, 1e2, 1e-5, 1e0 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		ax.locator_params( numticks=6 )
		if grid_pos not in [10, 11, 12, 13]:
			ax.tick_params(labelbottom=False)
		if grid_pos not in [0, 4, 8, 12]:
			ax.tick_params(labelleft=False)

	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )

#DEBUGGIN'

		# xplot = np.arange( egonet_props.act_min.min(), egonet_props.act_max.max()+1, dtype=int ) #activity range
		# yplot_data, not_used = np.histogram( egonet_acts, bins=len(xplot), range=( xplot.min()-0.5, xplot.max()+0.5 ), density=True )

#				xplot = np.arange( a0, props_fits_alpha.act_max.max()+1, dtype=int ) #activity range
