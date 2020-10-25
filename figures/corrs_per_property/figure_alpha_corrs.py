#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (ALPHA CORRS) IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
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

	#properties to correlate
	properties_x = [ ('alpha', r'\alpha'),
	 				 ('alpha', r'\alpha'),
					 ('alpha', r'\alpha'),
					 ('alpha', r'\alpha'),
					 ('alpha', r'\alpha'),
					 ('gamma', r'\gamma') ]
	properties_y = [ ('degree', 'k'),
					 ('strength', r'\tau'),
					 ('act_avg', 't'),
					 ('act_min', r'a_0'),
					 ('act_max', r'a_m'),
					 ('beta', r'\beta') ]

	bounds = (0, 1000) #bounds for alpha MLE fit
	nsims = 100 #number of syntethic datasets used to calculate p-value
	amax = 10000 #maximum activity for theoretical activity distribution

	pval_thres = 0.1 #threshold above which alphas are considered
	alpha_min, alpha_max = 1e-4, 1e2 #extreme values for alpha (to avoid num errors)

	#plotting variables
	gridsize = 40 #grid size for hex bins
	vmax = 1e4 #max value in colorbar (larger than [filtered] N in any dataset!)

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
	'text_size' : 15,
	'marker_size' : 6,
	'linewidth' : 2,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':15 },
	'legend_hlen' : 1,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#loop through properties to correlate
	for prop_pos, (propx, propy) in enumerate(zip( properties_x, properties_y )):
#	for prop_pos, (propx, propy) in enumerate(zip( [('gamma', r'\gamma')], [('beta', r'\beta')] )):
		print( 'propx = {}, propy = {}'.format( propx[0], propy[0] ) )

		#plot variables
		fig_props = { 'fig_num' : 1,
		'fig_size' : (10, 8),
		'aspect_ratio' : (4, 4),
		'grid_params' : dict( left=0.08, bottom=0.08, right=0.98, top=0.97, wspace=0.2, hspace=0.4 ),
		'width_ratios' : [1, 1, 1, 1.2],
		'dpi' : 300,
		'savename' : 'figure_data_corrs_{}_{}'.format( propx[0], propy[0] ) }

		#initialise plot
		sns.set( style='ticks' ) #set fancy fancy plot
		fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
		plt.clf()
		grid = gridspec.GridSpec( *fig_props['aspect_ratio'], width_ratios=fig_props['width_ratios'] )
		grid.update( **fig_props['grid_params'] )

		#loop through considered datasets
		for grid_pos, (dataname, eventname, textname) in enumerate(datasets):
#		for grid_pos, (dataname, eventname, textname) in enumerate([ ('MPC_UEu_net', 'MPC_UEu.evt', 'Mobile (call)') ]):
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
			#step 3: alphas between extreme values (numerical errors)
			props_fits_filter = props_fits_filter[ props_fits_filter.alpha.between( alpha_min, alpha_max ) ]

			#gamma distribution quantities
			gammas = pd.Series( props_fits_filter.alpha + props_fits_filter.act_min, name='gamma' )
			betas = pd.Series( ( props_fits_filter.act_avg - props_fits_filter.act_min ) / ( props_fits_filter.alpha + props_fits_filter.act_min ), name='beta' )

			#add gamma quantities to [filtered] properties and fits
			props_fits_filter = pd.concat( [ props_fits_filter, gammas, betas ], axis=1 )


			## PLOTTING ##

			#initialise subplot
			ax = plt.subplot( grid[ grid_pos] )
			sns.despine( ax=ax ) #take out spines
			if grid_pos in [10, 11, 12, 13]:
				plt.xlabel( '$'+propx[1]+'$', size=plot_props['xylabel'] )
			if grid_pos in [0, 4, 8, 12]:
				plt.ylabel( '$'+propy[1]+'$', size=plot_props['xylabel'] )

			#plot plot!
			hexbin = plt.hexbin( propx[0], propy[0], data=props_fits_filter, xscale='log', yscale='log', norm=LogNorm(vmin=1e0, vmax=vmax), mincnt=1, gridsize=gridsize, cmap='copper_r' )

			#colorbar
			if grid_pos in [3, 7, 11]:
				cbar = plt.colorbar( hexbin, ax=ax )
				cbar.ax.set_title( r'$N_{'+propx[1]+','+propy[1]+'}$' )
				cbar.ax.minorticks_off()

			#texts
			plt.text( 0, 1.15, textname, va='top', ha='left', transform=ax.transAxes, fontsize=plot_props['text_size'] )

			#finalise subplot
			if prop_pos in [0, 1]:
				plt.axis([ 1e-4, 1e2, 8e-1, 1e3 ])
			if prop_pos in [2, 4]:
				plt.axis([ 1e-4, 1e2, 8e-1, 1e2 ])
			if prop_pos == 3:
				plt.axis([ 1e-4, 1e2, 8e-1, 1e1 ])
			if prop_pos == 5:
				plt.axis([ 8e-1, 2e2, 5e-3, 1e2 ])
			ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
			ax.locator_params( numticks=6 )
			if grid_pos not in [10, 11, 12, 13]:
				ax.tick_params(labelbottom=False)
			if grid_pos not in [0, 4, 8, 12]:
				ax.tick_params(labelleft=False)

		#finalise plot
		if fig_props['savename'] != '':
			plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
