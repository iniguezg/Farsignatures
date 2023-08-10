#! /usr/bin/env python

#Farsignatures - Exploring dynamics of egocentric communication networks
#Copyright (C) 2023 Gerardo Iñiguez

### SCRIPT FOR PLOTTING FIGURE (DATA CORRS) IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
import pandas as pd
import seaborn as sns
import itertools as it
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
	properties = [ ('degree', 'k'),
				   ('strength', r'\tau'),
				   ('act_avg', 't'),
				   ('act_min', r'a_0'),
				   ('act_max', r'a_m') ]

	#dispersion filtering
	filter_prop = 'strength' #selected property and threshold to filter egos
	filter_thres = 10

	#plotting variables
	gridsize = 40 #grid size for hex bins
	# vmax = 3e6 #max value in colorbar (larger than [filtered] N in any dataset!)
	vmax = 1e5

	#locations
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files

	#dataset list: eventname, textname
	datasets = [ #( 'MPC_UEu', 'Mobile (call)'),
	# datasets = [ ( 'call', 'Mobile (call)'),
	# 			 ( 'text', 'Mobile (sms)'),
				 ( 'MPC_Wu_SD01', 'Mobile (Wu 1)'),
				 ( 'MPC_Wu_SD02', 'Mobile (Wu 2)'),
				 ( 'MPC_Wu_SD03', 'Mobile (Wu 3)'),
				 #( 'sexcontact_events', 'Contact'),
				 #( 'email', 'Email 1'),
				 #( 'eml2', 'Email 2'),
				 ( 'fb', 'Facebook'),
				 #( 'messages', 'Messages'),
				 #( 'forum', 'Forum'),
				 #( 'pok', 'Dating'),
				 #( 'CNS_bt_symmetric', 'CNS (bluetooth)'),
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

	#loop through combinations of properties to correlate
	# for propx, propy in it.combinations( properties, 2 ):
	for propx, propy in [ ( ('degree', 'k'), ('dispersion', 'd') ) ]:
	# for propx, propy in [ ( ('strength', r'\tau'), ('dispersion', 'd') ) ]:
	# for propx, propy in [ ( ('act_avg', 't'), ('dispersion', 'd') ) ]:
		print( 'propx = {}, propy = {}'.format( propx[0], propy[0] ) )

		#plot variables
		fig_props = { 'fig_num' : 1,
		'fig_size' : (10, 8),
		'aspect_ratio' : (4, 4),
		'grid_params' : dict( left=0.07, bottom=0.08, right=0.98, top=0.97, wspace=0.1, hspace=0.4 ),
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
		for grid_pos, (eventname, textname) in enumerate(datasets):
			print( 'dataset name: ' + eventname ) #print output

			## DATA ##

			#prepare ego network properties
			egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
			egonet_acts = pd.read_pickle( saveloc + 'egonet_acts_' + eventname + '.pkl' )

			if propy[0] == 'dispersion':
				#get activity means/variances/minimums per ego
				act_avgs = egonet_acts.groupby('nodei').mean()
				act_vars = egonet_acts.groupby('nodei').var() #NOTE: estimated variance (ddof=1)
				act_mins = egonet_props.act_min
				#filter by selected property
				act_avgs = act_avgs[ egonet_props[filter_prop] > filter_thres ]
				act_vars = act_vars[ egonet_props[filter_prop] > filter_thres ]
				act_mins = act_mins[ egonet_props[filter_prop] > filter_thres ]
				#get dispersion index measure per ego (use relative mean!)
				act_disps = ( act_vars - act_avgs + act_mins ) / ( act_vars + act_avgs - act_mins )
				act_disps = act_disps.dropna() #drop faulty egos
				#add dispersion to props dataframe
				egonet_props = pd.concat( [ egonet_props, act_disps.rename('dispersion') ], axis=1, join='inner' )
				#print out correlation
				print( '\tPearson corr = {:.2f}'.format( egonet_props.corr().loc[propx[0], propy[0]] ) )


			## PLOTTING ##

			#initialise subplot
			ax = plt.subplot( grid[ grid_pos] )
			sns.despine( ax=ax ) #take out spines
			if grid_pos in [10, 11, 12, 13]:
				plt.xlabel( '$'+propx[1]+'$', size=plot_props['xylabel'] )
			if grid_pos in [0, 4, 8, 12]:
				plt.ylabel( '$'+propy[1]+'$', size=plot_props['xylabel'] )

			#plot plot!
			yscale = 'lin'
			hexbin = plt.hexbin( propx[0], propy[0], data=egonet_props, xscale='log', yscale=yscale, norm=LogNorm(vmin=1e0, vmax=vmax), mincnt=1, gridsize=gridsize, cmap='copper_r' )

			#colorbar
			if grid_pos in [3, 7, 11]:
				cbar = plt.colorbar( hexbin, ax=ax )
				cbar.ax.set_title( r'$N_{'+propx[1]+','+propy[1]+'}$' )
				cbar.ax.minorticks_off()

			#texts
			plt.text( 0, 1.15, textname, va='top', ha='left', transform=ax.transAxes, fontsize=plot_props['text_size'] )

			#finalise subplot
			# plt.axis([ 1e0, 1e5, 1e0, 1e5 ])
			plt.axis([ 1e0, 1e5, 0, 1 ])
			ax.tick_params( axis='both', which='major', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
			# ax.locator_params( numticks=6 )
			if grid_pos not in [10, 11, 12, 13]:
				ax.tick_params(labelbottom=False)
			if grid_pos not in [0, 4, 8, 12]:
				ax.tick_params(labelleft=False)

		#finalise plot
		if fig_props['savename'] != '':
			plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
