#! /usr/bin/env python

#Farsignatures - Exploring dynamics of egocentric communication networks
#Copyright (C) 2023 Gerardo Iñiguez

### SCRIPT FOR PLOTTING FIGURE (CONNECTION KERNEL BY DISPERSION) IN FARSIGNATURES PROJECT ###

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

	#filter variables
	filt_rule = 'dispersion' #chosen filter parameter ('degree', 'dispersion')
	min_negos = 30 #minimum number of egos in filtered activity group

	#dset-specific params: minimum degree / number of quantiles (+1) of filtered egos
	dset_params = { 'call': {'min_degree':10, 'num_quants':5},
					'text': {'min_degree':10, 'num_quants':5},
					'MPC_Wu_SD01': {'min_degree':10, 'num_quants':5},
					'MPC_Wu_SD02': {'min_degree':10, 'num_quants':5},
					'MPC_Wu_SD03': {'min_degree':10, 'num_quants':5},
					'Enron': {'min_degree':10, 'num_quants':5},
					'email': {'min_degree':10, 'num_quants':5},
					'eml2': {'min_degree':10, 'num_quants':5},
					'email_Eu_core': {'min_degree':10, 'num_quants':5},
					'fb': {'min_degree':10, 'num_quants':5},
					'messages': {'min_degree':10, 'num_quants':5},
					'pok': {'min_degree':10, 'num_quants':5},
					'forum': {'min_degree':10, 'num_quants':5},
					'CollegeMsg': {'min_degree':10, 'num_quants':5},
					'CNS_calls': {'min_degree':2, 'num_quants':5},
					'CNS_sms': {'min_degree':2, 'num_quants':5},}

	#dispersion variables
	filter_prop = 'strength' #selected property and threshold to filter egos
	filter_thres = 0

	#root locations of data/code
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
	'legend_prop' : { 'size':7 },
	'legend_hlen' : 1,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables
	fig_props = { 'fig_num' : 1,
	'fig_size' : (10, 8),
	'aspect_ratio' : (4, 4),
	'grid_params' : dict( left=0.07, bottom=0.065, right=0.975, top=0.97, wspace=0.25, hspace=0.3 ),
	'dpi' : 300,
	'savename' : 'figure_kernel_{}'.format(filt_rule) }

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )

	print('\t\tfilter rule: {}'.format(filt_rule), flush=True) #filter rule

	#loop through considered datasets
	for grid_pos, (eventname, textname) in enumerate(datasets):
	# for grid_pos, (eventname, textname) in enumerate([ ( 'fb', 'Facebook') ]):
		print( 'dataset name: ' + eventname, flush=True ) #print output

		## PLOTTING ##

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos] )
		sns.despine( ax=ax ) #take out spines
		if grid_pos in [12, 13, 14, 15]:
			plt.xlabel( r'$a / a_m$', size=plot_props['xylabel'] )
		if grid_pos in [0, 4, 8, 12]:
			plt.ylabel( r'$\pi_a - \langle 1/k \rangle$', size=plot_props['xylabel'] )

		colors = sns.color_palette( 'GnBu', n_colors=dset_params[eventname]['num_quants']-1 ) #colors to plot


		## DATA ##

		#prepare ego network properties
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )

		#prepare filter object (ego degrees or dispersions)
		#only consider egos with large enough degree!
		egonet_props_filt = egonet_props[ egonet_props.degree >= dset_params[eventname]['min_degree'] ]
		if filt_rule == 'degree':
			filt_obj = egonet_props_filt.degree
		elif filt_rule == 'dispersion':
			filt_obj = dm.egonet_dispersion( egonet_props_filt, filter_prop, filter_thres )

		#get quantiles of filter parameter
		quantile_arr = np.linspace(0, 1, dset_params[eventname]['num_quants'])
		quantile_vals = np.quantile( filt_obj, quantile_arr )

		#loop through quantiles of filter parameter (inclusive!)
		for posval, (min_val, max_val) in enumerate( zip(quantile_vals[:-1], quantile_vals[1:]) ):
			print('\tmin_val = {:.2f}, max_val = {:.2f}'.format(min_val, max_val), flush=True) #filter range

			#prepare kernel: apply degree / negos filters, group and average
			data_avg, filt_ind = pm.plot_kernel_filter( eventname, filt_rule=filt_rule, filt_obj=filt_obj, filt_params={ 'min_val':min_val, 'max_val':max_val, 'min_negos':min_negos }, load=load, saveloc=saveloc, saveloc_fig=saveloc_fig )

			#prepare baseline: prob = <1/k> for random case
			bline_avg = ( 1 / egonet_props_filt.degree[filt_ind] ).mean()

			print('\t\tno. filtered egos: {}'.format(len(filt_ind)), flush=True) #filter set


			## PLOTTING ##

			#label by filter property range
			label = '{:.2f} '.format(min_val)+'$\leq d <$'+' {:.2f}'.format(max_val)

			#plot plot kernel mean (minus baseline)!
			xplot = data_avg.index / data_avg.index.max() #normalise by max activity
			line_data, = plt.plot( xplot, data_avg - bline_avg, '-', c=colors[posval], label=label, lw=plot_props['linewidth'], zorder=1 )

		#texts
		plt.text( 1, 1.13, textname, va='top', ha='right', transform=ax.transAxes, fontsize=plot_props['text_size'] )

		#legend
		plt.legend( loc='upper left', bbox_to_anchor=(0, 1), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=2 )

		#finalise subplot
		plt.axis([ 0, 1, -0.1, 1.3 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		if grid_pos not in [12, 13, 14, 15]:
			ax.tick_params(labelbottom=False)
		if grid_pos not in [0, 4, 8, 12]:
			ax.tick_params(labelleft=False)


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
