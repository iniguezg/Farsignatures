#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (ACTIVITY DIST [AGG]) IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
import numpy as np
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_misc as dm
import model_misc as mm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	max_iter = 1000 #max number of iteration for centrality calculations
	alphamax = 1000 #maximum alpha for MLE fit
	nsims = 100 #number of syntethic datasets used to calculate p-value
	amax = 10000 #maximum activity for theoretical activity distribution

	pval_thres = 0.1 #threshold above which alphas are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf
	gamma_lims = [ (1e-3, 1e0), (1e0, 1e2) ] #alpha intervals to aggregate egos

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

	colors = sns.color_palette( 'Set2', n_colors=len(gamma_lims) ) #colors to plot

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

		#prepare ego network / graph properties
		egonet_props, egonet_acts = dm.egonet_props_acts( dataname, eventname, root_data, 'y', saveloc )
		graph_props = dm.graph_props( dataname, eventname, root_data, 'y', saveloc, max_iter=max_iter )

		#fit activity model to all ego networks in dataset
		egonet_fits = dm.egonet_fits( dataname, eventname, root_data, 'y', saveloc, alphamax=alphamax, nsims=nsims, amax=amax )

		#filter egos according to fitting results
		egonet_filter, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, graph_props, egonet_fits, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )


		## PLOTTING ##

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos] )
		sns.despine( ax=ax ) #take out spines
		if grid_pos in [10, 11, 12, 13]:
			plt.xlabel( 'activity $a$', size=plot_props['xylabel'] )
		if grid_pos in [0, 4, 8, 12]:
			plt.ylabel( r"PDF $p_a(t)$", size=plot_props['xylabel'] )

		#loop through gamma intervals to aggregate egos
		for gamma_pos, (gamma_min, gamma_max) in enumerate( gamma_lims ):

			#filtering process (continued)
			#gammas between extreme values in selected interval
			egonet_filter_gamma = egonet_filter[ egonet_filter.gamma.between( gamma_min, gamma_max ) ]

			#get [aggregated] activity distribution
			acts_filter = egonet_acts.loc[ egonet_filter_gamma.index ] #alter activities (only filtered egos with gamma in interval)

			if len(acts_filter) > 0: #if there's anything to plot
				#plot data!

				xplot = np.arange( egonet_filter_gamma.act_min.min(), egonet_filter_gamma.act_max.max()+1, dtype=int ) #activity range
				yplot_data, not_used = np.histogram( acts_filter, bins=len(xplot), range=( xplot.min()-0.5, xplot.max()+0.5 ), density=True )

#				label = 'data' if gamma_pos == 0 else None
				symbol = 'o' if gamma_pos == 0 else 's'

				plt.loglog( xplot, yplot_data, symbol, c=colors[gamma_pos], label=None, ms=plot_props['marker_size'], zorder=0 )

				#plot model!

				#avg values of model parameters (over filtered egos)
				gamma = egonet_filter_gamma.gamma.mean()
				a0 = int( egonet_filter_gamma.act_min.mean() )
				beta = egonet_filter_gamma.beta.mean()

				xplot = np.arange( a0+1, 1e4, dtype=int ) #activity range
				yplot_model = ss.gamma.pdf( xplot, gamma, a0, beta ) #gamma approx

				label = '$\gamma < 1$' if gamma_pos == 0 else '$\gamma > 1$'

				plt.loglog( xplot, yplot_model, '-', c=colors[gamma_pos], label=label, lw=plot_props['linewidth'], zorder=1 )

		#text
		plt.text( 1, 1.15, textname, va='top', ha='right', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		#legend
		if grid_pos == 0:
			plt.legend( loc='upper right', bbox_to_anchor=(1.1, 1), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'] )

		#finalise subplot
		plt.axis([ 5e-1, 1e4, 1e-6, 1e0 ])
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

				# label = r'$10^{'+str(int(np.log10(gamma_min)))+r'} \leq \gamma \leq 10^{'+str(int(np.log10(gamma_max)))+'}$'
