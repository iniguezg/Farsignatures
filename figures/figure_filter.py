#! /usr/bin/env python

#Farsignatures - Exploring dynamics of egocentric communication networks
#Copyright (C) 2023 Gerardo Iñiguez

### SCRIPT FOR PLOTTING FIGURE (FILTER) IN FARSIGNATURES PROJECT ###

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
	propx = ('degree', 'k')
	propy = ('act_avg', 't')

	#alpha fit variables
	alphamax = 1000 #maximum alpha for MLE fit
	pval_thres = 0.1 #threshold above which alpha MLEs are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf

	#plotting variables
	gridsize = (40, 30) #grid size for hex bins

	#locations
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files

	#dataset list: eventname, textname
	# datasets = [ ( 'MPC_UEu', 'Mobile (call)'),
	datasets = [ #( 'call', 'Mobile (call)'),
				 #( 'text', 'Mobile (sms)'),
				 ( 'MPC_Wu_SD01', 'Mobile (Wu 1)'),
				 ( 'MPC_Wu_SD02', 'Mobile (Wu 2)'),
				 ( 'MPC_Wu_SD03', 'Mobile (Wu 3)'),
				 # ( 'sexcontact_events', 'Contact'),
				 ( 'email', 'Email 1'),
				 ( 'eml2', 'Email 2'),
				 ( 'fb', 'Facebook'),
				 ( 'messages', 'Messages'),
				 ( 'forum', 'Forum'),
				 ( 'pok', 'Dating'),
				 # ( 'CNS_bt_symmetric', 'CNS (bluetooth)'),
				 ( 'CNS_calls', 'CNS (call)'),
				 ( 'CNS_sms', 'CNS (sms)') ]

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

	#plot variables
	fig_props = { 'fig_num' : 1,
	'fig_size' : (12, 8),
	'aspect_ratio' : (4, 4),
	'grid_params' : dict( left=0.06, bottom=0.07, right=0.99, top=0.97, wspace=0.1, hspace=0.4 ),
	'dpi' : 300,
	'savename' : 'figure_filter' }

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )


	#loop through considered datasets
	for grid_pos, (eventname, textname) in enumerate(datasets):
	# for grid_pos, (eventname, textname) in enumerate([( 'MPC_UEu', 'Mobile (call)')]):
		print( 'dataset name: ' + eventname ) #print output

		## DATA ##

		#prepare ego network properties
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
		egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )

		#filter egos according to fitting results
		egonet_filter, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )


		## PLOTTING ##

		#initialise subplot
		ax = plt.subplot( grid[grid_pos] )
		sns.despine( ax=ax ) #take out spines
		if grid_pos in [9, 10, 11, 12]:
			plt.xlabel( '$'+propx[1]+'$', size=plot_props['xylabel'] )
		if grid_pos in [0, 4, 8, 12]:
			plt.ylabel( '$'+propy[1]+'$', size=plot_props['xylabel'] )

		#plot plot!
		vmax = len(egonet_filter) #max value in colorbar (total number of egos in filtered dataset)
		hexbin_null = ax.hexbin( propx[0], propy[0], data=egonet_null, xscale='log', yscale='log', norm=LogNorm(vmin=1e0, vmax=vmax), mincnt=1, gridsize=gridsize, cmap='OrRd', zorder=0 )
		hexbin_filter = ax.hexbin( propx[0], propy[0], data=egonet_filter, xscale='log', yscale='log', norm=LogNorm(vmin=1e0, vmax=vmax), mincnt=1, gridsize=gridsize, cmap='GnBu', zorder=1 )

		#colorbars
		cbar_null = plt.colorbar( hexbin_null, ax=ax )
		cbar_null.ax.set_title( r'$N_{\emptyset}$' )
		cbar_null.ax.minorticks_off()
		cbar_filter = plt.colorbar( hexbin_filter, ax=ax )
		cbar_filter.ax.set_title( r'$N_{\beta}$' )
		cbar_filter.ax.minorticks_off()
		cbar_filter.set_ticks([])

		#texts
		plt.text( 0, 1.15, textname, va='top', ha='left', transform=ax.transAxes, fontsize=plot_props['text_size'] )

		#finalise subplot
		plt.axis([ 0.5e0, 1e4, 0.5e0, 1e4 ])
		ax.tick_params( axis='both', which='major', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		ax.locator_params( numticks=6 )
		if grid_pos not in [9, 10, 11, 12]:
			ax.tick_params(labelbottom=False)
		if grid_pos not in [0, 4, 8, 12]:
			ax.tick_params(labelleft=False)


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
