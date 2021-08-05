#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (DATA CCDF) IN FARSIGNATURES PROJECT ###

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

	#properties to plot
	prop_names = [ 'degree', 'strength', 'act_avg', 'act_min', 'act_max' ]
	prop_labels = [ '$k$', r'$\tau$', '$t$', r'$a_0$', r'$a_m$' ]

	#locations
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files

	#dataset list: eventname, textname
#	datasets = [ ( 'MPC_UEu', 'Mobile (call)'),
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
	'fig_size' : (10, 8),
	'aspect_ratio' : (4, 4),
	'grid_params' : dict( left=0.08, bottom=0.08, right=0.99, top=0.92, wspace=0.1, hspace=0.4 ),
	'dpi' : 300,
	'savename' : 'figure_data_CCDF' }

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

		## PLOTTING ##

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos] )
		sns.despine( ax=ax ) #take out spines
		if grid_pos in [11, 12, 13, 14]:
			plt.xlabel( r'property $\bullet$', size=plot_props['xylabel'] )
		if grid_pos in [0, 4, 8, 12]:
			plt.ylabel( r"CCDF $P[ \bullet' \geq \bullet ]$", size=plot_props['xylabel'] )

		#loop through considered properties
		for prop_pos, (prop_name, prop_label) in enumerate(zip( prop_names, prop_labels )):

			#prepare data
			yplot_data = egonet_props[ prop_name ]
			xplot, yplot = pm.plot_compcum_dist( yplot_data ) #complementary cumulative dist

			#plot plot!
			plt.loglog( xplot, yplot, '-', label=prop_label, c=colors[prop_pos], lw=plot_props['linewidth'], ms=plot_props['marker_size'] )


		#texts
		plt.text( 1, 1.1, textname, va='top', ha='right', transform=ax.transAxes, fontsize=plot_props['text_size'] )

		#legend
		if grid_pos == 1:
			plt.legend( loc='upper left', bbox_to_anchor=(0.1, 1.55), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'], ncol=len(prop_names) )

		#finalise subplot
		plt.axis([ 1e0, 1e5, 1e-5, 2e0 ])
		ax.tick_params( axis='both', which='major', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		ax.locator_params( numticks=6 )
		if grid_pos not in [11, 12, 13, 14]:
			ax.tick_params(labelbottom=False)
		if grid_pos not in [0, 4, 8, 12]:
			ax.tick_params(labelleft=False)

	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
