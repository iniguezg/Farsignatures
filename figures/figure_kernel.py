#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (CONNECTION KERNEL) IN FARSIGNATURES PROJECT ###

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

	#plotting variables
	min_degree = 2 #minimum degree of filtered egos
	min_negos = 50 #minimum number of egos in filtered activity group

	#locations
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files

	#dataset list: eventname, textname
	datasets = [ ( 'MPC_UEu', 'Mobile (call)'),
	# datasets = [ ( 'call', 'Mobile (call)'),
	# 			 ( 'text', 'Mobile (sms)'),
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
	'text_size' : 15,
	'marker_size' : 1,
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
	'grid_params' : dict( left=0.08, bottom=0.08, right=0.99, top=0.92, wspace=0.1, hspace=0.4 ),
	'dpi' : 300,
	'savename' : 'figure_kernel' }

	colors = sns.color_palette( 'Paired', n_colors=3 ) #colors to plot

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

		#prepare ego network properties
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
		#prepare ego network connection kernel
		egonet_kernel = pd.read_pickle( saveloc + 'egonet_kernel_' + eventname + '.pkl' )


		## PLOTTING ##

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos] )
		sns.despine( ax=ax ) #take out spines
		if grid_pos in [10, 11, 12, 13]:
			plt.xlabel( r'$a$', size=plot_props['xylabel'] )
		if grid_pos in [0, 4, 8, 12]:
			plt.ylabel( r"$\langle \pi_a \rangle$", size=plot_props['xylabel'] )

		#prepare data: apply degree / negos filters, group and average
		filt_degree = egonet_kernel[ egonet_props[ egonet_props.degree >= min_degree ].index ]
		filt_negos = filt_degree.groupby( level=1 ).filter( lambda x : len(x) >= min_negos )
		data_grp = filt_negos.groupby( level=1 ) #group ego probs for each activity value
		data_avg = data_grp.mean() #average probs over egos

		#prepare baseline: prob = 1/k for random case
		bline_avg = ( 1 / egonet_props[ egonet_props.degree >= min_degree ].degree ).mean()

		print( '\t{:.2f}% egos after degree filter'.format( 100.*filt_degree.index.get_level_values(0).nunique() / len(egonet_props) ) ) #print filtering output

		#plot plot mean!
		plt.semilogx( data_avg, '-', c=colors[1], lw=plot_props['linewidth'], zorder=2 )

		#plot plot baseline!
		plt.axhline( bline_avg, ls='--', c='0.7', lw=plot_props['linewidth'], zorder=0 )

		#texts
		plt.text( 1, 1.05, textname, va='bottom', ha='right', transform=ax.transAxes, fontsize=plot_props['text_size'] )

		#finalise subplot
		plt.axis([ 9e-1, 5e3, -0.05, 1.05 ])
		ax.set_xticks([ 1e0, 1e1, 1e2, 1e3 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		if grid_pos not in [10, 11, 12, 13]:
			ax.tick_params(labelbottom=False)
		if grid_pos not in [0, 4, 8, 12]:
			ax.tick_params(labelleft=False)


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )


#DEBUGGIN'

		# data_CI = 0.99 * data_grp.std() / np.sqrt( data_grp.size() ) #(Gaussian) confidence interval
		# plt.fill_between( data_avg.index, data_avg - data_CI, data_avg + data_CI, color=colors[0], alpha=0.5, zorder=1 )

	# gridsize = 40 #grid size for hex bins
	# vmax = 8e4 #max value in colorbar (larger than N in any dataset!)
	# 	data_sct = filt_negos[ filt_negos.index.get_level_values(1) != 0 ]
	# 	#plot plot scatterplot of probabilities!
	# 	plt.hexbin( data_sct.index.get_level_values(1), data_sct.values, xscale='log', norm=LogNorm(vmin=1e0, vmax=vmax), mincnt=10, gridsize=gridsize, cmap='copper_r', zorder=1 )
