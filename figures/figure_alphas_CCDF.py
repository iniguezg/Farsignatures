#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (ALPHAS CCDF) IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
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

	#property to plot
	# prop_names = [ 'gamma' ]
	# prop_labels = [ r'\hat{\alpha}_r' ]
	prop_names = [ 'beta' ]
	prop_labels = [ r'1 / \hat{\beta}' ]

	alphamax = 1000 #maximum alpha for MLE fit
	pval_thres = 0.1 #threshold above which alpha MLEs are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf
	# max_iter = 1000 #max number of iteration for centrality calculations
	# nsims = 1000 #number of syntethic datasets used to calculate p-value
	# amax = 10000 #maximum activity for theoretical activity distribution

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
	'text_size' : 11,
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
	'grid_params' : dict( left=0.085, bottom=0.085, right=0.98, top=0.965, wspace=0.3, hspace=0.5 ),
	'dpi' : 300,
	'savename' : 'figure_alphas_CCDF' }

	colors = sns.color_palette( 'Set2', n_colors=len(prop_names) ) #colors to plot

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )

	#loop through considered datasets
	for grid_pos, (eventname, textname) in enumerate(datasets):
		print( 'dataset name: ' + eventname ) #print output

		## DATA ##

		#prepare ego network properties
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
		#fit activity model to all ego networks in dataset
		egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )

		#filter egos according to fitting results
		egonet_filter, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )

		#some measures
		num_egos = len( egonet_props ) #all egos
		num_egos_filter = len( egonet_filter ) #filtered egos
#		frac_egos_random = ( egonet_filter.gamma > 1 ).sum() / float( num_egos_filter )
		frac_egos_random = ( egonet_filter.beta < 1 ).sum() / float( num_egos_filter ) #fraction of egos in random regime (beta < 1, i.e. t_r < alpha_r)


		## PRINTING ##

		print( 'fraction filtered egos = {:.2f}'.format( num_egos_filter / float(num_egos) ) )
		print( 'fraction inf egos = {:.2f}'.format( len(egonet_inf) / float(num_egos) ) )
		print( 'fraction null egos = {:.2f}'.format( len(egonet_null) / float(num_egos) ) )
		print( '\n' )


		## PLOTTING ##

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos] )
		sns.despine( ax=ax ) #take out spines

		#loop through considered properties
		for prop_pos, (prop_name, prop_label) in enumerate(zip( prop_names, prop_labels )):

			#initialise subplot
			if grid_pos in [11, 12, 13, 14]:
				plt.xlabel( '${}$'.format( prop_label ), size=plot_props['xylabel'] )
			if grid_pos in [0, 4, 8, 12]:
				plt.ylabel( r"$P[ {}' \geq {} ]$".format( prop_label, prop_label ), size=plot_props['xylabel'] )

			#prepare data
			yplot_data = 1 / egonet_filter[ prop_name ] #filtered data!
			xplot, yplot = pm.plot_CCDF_cont( yplot_data ) #complementary cumulative dist

			#plot plot!
			plt.loglog( xplot, yplot, '-', label=prop_label, c=colors[prop_pos], lw=plot_props['linewidth'] )

		#lines
		plt.vlines( x=1, ymin = 1e-5, ymax=frac_egos_random, linestyles='--', colors='0.6', lw=plot_props['linewidth']-1 )
		plt.hlines( y=frac_egos_random, xmin = 1e-4, xmax=1, linestyles='--', colors='0.6', lw=plot_props['linewidth']-1 )

		#texts

		plt.text( 1, 1.15, textname, va='top', ha='right', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		txt_str = r'$N_{\alpha}=$ '+'{}'.format(num_egos_filter)+'\n'+r'$n_{RN} =$'+'{:.2f}'.format(frac_egos_random)
		plt.text( 0.05, 0.05, txt_str, va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['text_size'] )

		#finalise subplot
		plt.axis([ 1e-4, 1e3, 1e-4, 2e0 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		ax.locator_params( numticks=5 )
		if grid_pos not in [11, 12, 13, 14]:
			ax.tick_params(labelbottom=False)
		if grid_pos not in [0, 4, 8, 12]:
			ax.tick_params(labelleft=False)

	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )


#DEBUGGIN'

#			plt.loglog( xplot, yplot, '-', label=prop_label, mec=colors[prop_pos], mfc='w', mew=plot_props['linewidth'], ms=plot_props['marker_size'] )
