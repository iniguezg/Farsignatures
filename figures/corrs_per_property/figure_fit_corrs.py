#! /usr/bin/env python

#Farsignatures - Exploring dynamics of egocentric communication networks
#Copyright (C) 2023 Gerardo Iñiguez

### SCRIPT FOR PLOTTING FIGURE (FIT CORRS) IN FARSIGNATURES PROJECT ###

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

#	fit/data properties to correlate
	# properties_x = [ ('gamma', r'\hat{\alpha}_r'),
	#  				 ('gamma', r'\hat{\alpha}_r'),
	# 				 ('act_avg_rel', 't_r'),
	# 				 ('beta', r'\beta'),
	# 				 ('beta', r'\beta'),
	# 				 ('beta', r'\beta'),
	# 				 ('beta', r'\beta')
	# 				]
	# properties_y = [ ('act_avg_rel', 't_r'),
	# 				 ('beta', r'\beta'),
	# 				 ('beta', r'\beta'),
	# 				 ('degree', 'k'),
	# 				 ('str_rel', r'\tau_r'),
	# 				 ('act_min', r'a_0'),
	# 				 ('act_max', r'a_m'),
	# 				]
	properties_x = [ ('gamma', r'\hat{\alpha}_r') ]
	properties_y = [ ('act_avg_rel', 't_r') ]

	alphamax = 1000 #maximum alpha for MLE fit
	pval_thres = 0.1 #threshold above which alpha MLEs are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf

	#plotting variables
	gridsize = 40 #grid size for hex bins
	vmax = 1e6 #max value in colorbar (larger than [filtered] N in any dataset!)

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
				 ( 'email', 'Email (Kiel)'),
				 ( 'eml2', 'Email (Uni)'),
				 ( 'email_Eu_core', 'Email (EU)'),
				 ( 'Enron', 'Email (Enron)'),
				 ( 'fb', 'Facebook'),
				 ( 'messages', 'Messages'),
				 ( 'forum', 'Forum'),
				 ( 'pok', 'Dating'),
				 ( 'CollegeMsg', 'College'),
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

	#loop through properties to correlate
	for prop_pos, (propx, propy) in enumerate(zip( properties_x, properties_y )):
#	for prop_pos, (propx, propy) in enumerate(zip( [('gamma', r'\gamma')], [('beta', r'\beta')] )):
		print( 'propx = {}, propy = {}'.format( propx[0], propy[0] ) )

		#plot variables
		fig_props = { 'fig_num' : 1,
		'fig_size' : (10, 8),
		'aspect_ratio' : (4, 4),
		'grid_params' : dict( left=0.075, bottom=0.08, right=0.98, top=0.97, wspace=0.2, hspace=0.4 ),
		'dpi' : 300,
		'savename' : 'figure_fit_corrs_{}_{}'.format( propx[0], propy[0] ) }

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

			#add relative quantities
			tau_rels = pd.Series( egonet_filter.strength - egonet_filter.degree * egonet_filter.act_min, name='str_rel' )
			t_rels = pd.Series( egonet_filter.act_avg - egonet_filter.act_min, name='act_avg_rel' )
			egonet_filter = pd.concat( [ egonet_filter, tau_rels, t_rels ], axis=1 )


			## PLOTTING ##

			#initialise subplot
			ax = plt.subplot( grid[ grid_pos] )
			sns.despine( ax=ax ) #take out spines
			if grid_pos in [12, 13, 14, 15]:
				plt.xlabel( '$'+propx[1]+'$', size=plot_props['xylabel'] )
			if grid_pos in [0, 4, 8, 12]:
				plt.ylabel( '$'+propy[1]+'$', size=plot_props['xylabel'] )

			#plot plot!
			vmax = len(egonet_filter) #max value in colorbar (total number of egos in filtered dataset)
			hexbin = plt.hexbin( propx[0], propy[0], data=egonet_filter, xscale='log', yscale='log', norm=LogNorm(vmin=1e0, vmax=vmax), mincnt=1, gridsize=gridsize, cmap='GnBu', zorder=0 )

			#colorbar
			cbar = plt.colorbar( hexbin, ax=ax )
			cbar.ax.set_title( r'$N_{'+propx[1]+','+propy[1]+'}$' )
			cbar.ax.minorticks_off()

			#lines

			if prop_pos == 0:
				plt.plot( [1e-4, 1e4], [1e-4, 1e4], '-', c='0.6', lw=plot_props['linewidth'], zorder=1 )
				plt.plot( [1, 1], [1, 1e4], '--', c='0.6', lw=plot_props['linewidth'], zorder=1 )

			if prop_pos in [ 1, 2 ]:
				plt.axhline( y=1, ls='--', c='0.6', lw=plot_props['linewidth'] )

			if prop_pos in [ 3, 4, 5, 6 ]:
				plt.axvline( x=1, ls='--', c='0.6', lw=plot_props['linewidth'] )

			#texts
			plt.text( 0, 1.15, textname, va='top', ha='left', transform=ax.transAxes, fontsize=plot_props['text_size'] )

			#finalise subplot
			if prop_pos == 0:
				plt.axis([ 1e-4, 1e3, 1e-3, 1e4 ])
			if prop_pos == 1:
				plt.axis([ 1e-3, 1e3, 1e-4, 1e5 ])
			if prop_pos == 2:
				plt.axis([ 1e-2, 1e3, 1e-4, 1e5 ])
			if prop_pos == 3:
				plt.axis([ 1e-4, 1e5, 5e-1, 1e3 ])
			if prop_pos == 4:
				plt.axis([ 1e-4, 1e5, 1e0, 1e5 ])
			if prop_pos == 5:
				plt.axis([ 1e-4, 1e5, 5e-1, 1e3 ])
			if prop_pos == 6:
				plt.axis([ 1e-4, 1e5, 1e0, 1e5 ])
			ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
			ax.locator_params( numticks=5 )
			if grid_pos not in [12, 13, 14, 15]:
				ax.tick_params(labelbottom=False)
			if grid_pos not in [0, 4, 8, 12]:
				ax.tick_params(labelleft=False)

		#finalise plot
		if fig_props['savename'] != '':
			plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
