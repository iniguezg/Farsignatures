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
import plot_misc as pm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#plotting variables
	min_degree = 2 #minimum degree of filtered egos
	min_negos = 50 #minimum number of egos in filtered activity group

	#locations
	#LOCAL
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files
	saveloc_fig = expanduser('~') + '/prg/xocial/Farsignatures/figures/figure1_data/'
	# saveloc_fig = ''
	# #TRITON
	# root_data = '/m/cs/scratch/networks/inigueg1/prg/xocial/datasets/temporal_networks/'
	# saveloc = '/m/cs/scratch/networks/inigueg1/prg/xocial/Farsignatures/files/data/'
	# saveloc_fig = '/m/cs/scratch/networks/inigueg1/prg/xocial/Farsignatures/figures/figure1_data/'

	#flags
	load = True

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
				 ( 'CNS_sms', 'CNS (sms)')
				]

	#sizes/widths/coords
	plot_props = { 'xylabel' : 15,
	'figlabel' : 26,
	'ticklabel' : 15,
	'text_size' : 15,
	'marker_size' : 1,
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
	'grid_params' : dict( left=0.07, bottom=0.06, right=0.98, top=0.97, wspace=0.25, hspace=0.3 ),
	'dpi' : 300,
	'savename' : 'figure_kernel' }

	colors = sns.color_palette( 'Paired', n_colors=1 ) #colors to plot

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )

	#loop through considered datasets
	for grid_pos, (eventname, textname) in enumerate(datasets):
	# for grid_pos, (eventname, textname) in enumerate([ ( 'MPC_UEu', 'Mobile (call)') ]):
		print( 'dataset name: ' + eventname, flush=True ) #print output

		## DATA ##

		#prepare ego network properties
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )

		#only consider egos with large enough degree!
		egonet_props_filt = egonet_props[ egonet_props.degree >= min_degree ]

		#use single quantile range, i.e. all degrees!
		quantile_arr = np.linspace(0, 1, 2)
		min_val, max_val = np.quantile( egonet_props_filt.degree, quantile_arr )

		#prepare kernel: apply degree / negos filters, group and average
		data_avg, filt_ind = pm.plot_kernel_filter( eventname, filt_rule='degree', filt_obj=egonet_props.degree, filt_params={ 'min_val':min_val, 'max_val':max_val, 'min_negos':min_negos }, load=load, saveloc=saveloc, saveloc_fig=saveloc_fig )

		#prepare baseline: prob = <1/k> for random case
		bline_avg = ( 1 / egonet_props_filt.degree[filt_ind] ).mean()


		## PLOTTING ##

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos] )
		sns.despine( ax=ax ) #take out spines
		if grid_pos in [12, 13, 14, 15]:
			plt.xlabel( r'$a$', size=plot_props['xylabel'] )
		if grid_pos in [0, 4, 8, 12]:
			plt.ylabel( r"$\pi_a$", size=plot_props['xylabel'] )

		#plot plot kernel mean!
		plt.plot( data_avg, '-', c=colors[0], lw=plot_props['linewidth'], zorder=1 )
		#plot plot baseline!
		plt.axhline( bline_avg, xmax=0.75, ls='--', c='0.7', lw=plot_props['linewidth']-1, zorder=0 )

		#texts
		plt.text( 1, 1.1, textname, va='top', ha='right', transform=ax.transAxes, fontsize=plot_props['text_size'] )
		plt.text( 80, bline_avg, r'$\langle 1/k \rangle$', va='center', ha='left', fontsize=plot_props['ticklabel'] )

		#finalise subplot
		plt.axis([ 0, 100, -0.05, 1.05 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		if grid_pos not in [12, 13, 14, 15]:
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

		# print( '\t{:.2f}% egos after degree filter'.format( 100.*filt_degree.index.get_level_values(0).nunique() / len(egonet_props) ) ) #print filtering output
