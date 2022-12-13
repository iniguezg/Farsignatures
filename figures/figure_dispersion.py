#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (ACTIVITY DISPERSION) IN FARSIGNATURES PROJECT ###

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

	#subplot variables
	filter_prop = 'strength' #selected property and threshold to filter egos
	filter_thres = 10

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
	'savename' : 'figure_dispersion' }

	colors = sns.color_palette( 'Paired', n_colors=1 ) #colors to plot

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )

	#loop through considered datasets
	total_egos_filter = 0 #init counter of all filtered egos
	for grid_pos, (eventname, textname) in enumerate(datasets):
	# for grid_pos, (eventname, textname) in enumerate([ ( 'CNS_calls', 'CNS (call)') ]):
		print( 'dataset name: ' + eventname ) #print output

		## DATA ##

		#prepare ego network properties
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
		#filter egos by t > a0 condition
		egonet_props_filter = egonet_props[ egonet_props.degree * egonet_props.act_min < egonet_props.strength ]

		#get activity means/variances/minimums per ego
		act_avgs = egonet_props_filter.act_avg
		act_vars = egonet_props_filter.act_var
		act_mins = egonet_props_filter.act_min
		#filter by selected property
		act_avgs = act_avgs[ egonet_props_filter[filter_prop] > filter_thres ]
		act_vars = act_vars[ egonet_props_filter[filter_prop] > filter_thres ]
		act_mins = act_mins[ egonet_props_filter[filter_prop] > filter_thres ]
		#get dispersion index measure per ego (use relative mean!)
		act_disps = ( act_vars - act_avgs + act_mins ) / ( act_vars + act_avgs - act_mins )
		act_disps = act_disps.dropna() #drop faulty egos

		#print output
		print( '\tshown egos: {:.2f}%'.format( 100.*len(act_disps)/len(egonet_props) ), flush=True ) #filtered egos
		total_egos_filter += len(act_disps) #filtered egos
		print( '\tavg disp: {:.2f}'.format( act_disps.mean() ), flush=True ) #mean dispersion
		if grid_pos == len(datasets)-1:
			print( '\t\ttotal number of filtered egos: {}'.format( total_egos_filter ), flush=True )


		## PLOTTING ##

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos] )
		sns.despine( ax=ax ) #take out spines
		if grid_pos in [12, 13, 14, 15]:
			plt.xlabel( r'$d$', size=plot_props['xylabel'] )
		if grid_pos in [0, 4, 8, 12]:
			plt.ylabel( r"CCDF $P[ d' \geq d ]$", size=plot_props['xylabel'] )

		#plot plot activity dispersion!
		xplot, yplot = pm.plot_CCDF_cont( act_disps ) #get dispersion CCDF: P[X >= x]
		plt.plot( xplot, yplot, '-', c=colors[0], lw=plot_props['linewidth'], zorder=1 )
		#plot plot average dispersion!
		plt.axvline( act_disps.mean(), ls='--', c='0.5', lw=plot_props['linewidth']-1, zorder=0 )

		#texts
		plt.text( 1, 1.1, textname, va='top', ha='right', transform=ax.transAxes, fontsize=plot_props['text_size'] )
		plt.text( act_disps.mean()-0.05, 0.05, r'$\langle d \rangle$', va='bottom', ha='right', fontsize=plot_props['ticklabel'] )

		#finalise subplot
		plt.axis([ -0.05, 1, 0, 1.05 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		ax.locator_params( nbins=4 )
		if grid_pos not in [12, 13, 14, 15]:
			ax.tick_params(labelbottom=False)
		if grid_pos not in [0, 4, 8, 12]:
			ax.tick_params(labelleft=False)


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )