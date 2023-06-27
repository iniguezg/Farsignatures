#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (ALPHAS DIST) IN FARSIGNATURES PROJECT ###

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


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#property to plot
	prop_name = 'beta'
	prop_label = r'\beta'
	# prop_label = r'1 / \beta'

	stat = 'KS' #chosen test statistic
	alphamax = 1000 #maximum alpha for MLE fit
	pval_thres = 0.1 #threshold above which alpha MLEs are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf

	nbins = 30 #number of bins to (log-)plot raw property distribution

	#locations
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files

	#dataset list: eventname, textname
	datasets = [ ( 'call', 'Mobile (call)'),
				 ( 'text', 'Mobile (sms)'),
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
	'grid_params' : dict( left=0.08, bottom=0.08, right=0.99, top=0.965, wspace=0.3, hspace=0.5 ),
	'dpi' : 300,
	'savename' : 'figure_alphas_dist' }

	colors = sns.color_palette( 'Paired', n_colors=1 ) #colors to plot

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
		#filter egos by t > a0 condition
		egonet_props_filter = egonet_props[ egonet_props.degree * egonet_props.act_min < egonet_props.strength ]
		#fit activity model to all ego networks in dataset
		egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )

		#filter egos according to fitting results
		egonet_filter, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, stat=stat, pval_thres=pval_thres, alphamax=alphamax, alph_thres=alph_thres )

		#some measures

		num_egos = len( egonet_props_filter ) #all (filtered) egos

		num_egos_filter = len( egonet_filter ) #statistically significant alpha
		frac_egos_filter = num_egos_filter / float( num_egos )
		frac_egos_inf = len( egonet_inf ) / float( num_egos ) #infinite alpha
		frac_egos_null = len( egonet_null ) / float( num_egos ) #undefined alpha

		frac_egos_random = ( egonet_filter.beta < 1 ).sum() / float( num_egos_filter ) #fraction of egos in random regime (beta < 1, i.e. t_r < alpha_r)


		## PRINTING ##

		print( 'fraction filtered egos = {:.2f}'.format( frac_egos_filter ) )
		print( 'fraction inf egos = {:.2f}'.format( frac_egos_inf ) )
		print( 'fraction null egos = {:.2f}'.format( frac_egos_null ) )
		print( '\n' )


		## PLOTTING ##

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos] )
		sns.despine( ax=ax ) #take out spines
		if grid_pos in [12, 13, 14, 15]:
			plt.xlabel( '${}$'.format( prop_label ), size=plot_props['xylabel'] )
		if grid_pos in [0, 4, 8, 12]:
			plt.ylabel( r"$P[ {} ]$".format( prop_label ), size=plot_props['xylabel'] )

		#prepare data
		yplot_data = egonet_filter[ prop_name ] #filtered data!
		# yplot_data = 1 / egonet_filter[ prop_name ]
		bins = np.logspace( np.log10( yplot_data.min() ), np.log10( yplot_data.max() ), num=nbins+1 ) #log bins
		yplot, bin_edges = np.histogram( yplot_data, bins=bins, density=True )
		xplot = [ ( bin_edges[i+1] + bin_edges[i] ) / 2 for i in range(len( bin_edges[:-1] )) ]

		#plot plot!
		ax.loglog( xplot, yplot, 'o', label=prop_label, c=colors[0], ms=plot_props['marker_size'], zorder=1 )

		#lines
		plt.axvline( x=1, ls='--', c='0.6', lw=plot_props['linewidth'], zorder=0 )

		#texts
		ax.text( 1, 1.15, textname, va='top', ha='right', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		#finalise subplot
		ax.axis([ 1e-3, 2e4, 8e-9, 1e0 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		ax.locator_params( numticks=5 )
		if grid_pos not in [12, 13, 14, 15]:
			ax.tick_params(labelbottom=False)
		if grid_pos not in [0, 4, 8, 12]:
			ax.tick_params(labelleft=False)

	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
