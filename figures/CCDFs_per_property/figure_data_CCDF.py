#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (DATA CCDF) IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
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

	#property to plot
#	prop_name, prop_xlabel, prop_ylabel = 'degree', '$k$', "$P[ k' \geq k ]$"
#	prop_name, prop_xlabel, prop_ylabel = 'strength', r'$\tau$', r"$P[ \tau' \geq \tau ]$"
#	prop_name, prop_xlabel, prop_ylabel = 'act_avg', '$t$', "$P[ t' \geq t ]$"
#	prop_name, prop_xlabel, prop_ylabel = 'act_min', r'$a_0$', r"$P[ a_0' \geq a_0 ]$"
	prop_name, prop_xlabel, prop_ylabel = 'act_max', r'$a_m$', r"$P[ a_m' \geq a_m ]$"

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
	'ticklabel' : 11,
	'text_size' : 15,
	'marker_size' : 6,
	'linewidth' : 3,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':10 },
	'legend_hlen' : 1,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables
	fig_props = { 'fig_num' : 1,
	'fig_size' : (10, 8),
	'aspect_ratio' : (4, 4),
	'grid_params' : dict( left=0.07, bottom=0.07, right=0.98, top=0.97, wspace=0.4, hspace=0.5 ),
	'dpi' : 300,
	'savename' : 'figure_data_CCDF_' + prop_name }

	colors = sns.color_palette( 'Set2', n_colors=1 ) #colors to plot

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )

	#loop through considered datasets
	for grid_pos, data_tuple in enumerate(datasets):
		dataname, eventname, textname = data_tuple
		print( 'dataset name: ' + eventname[:-4] ) #print output

		## DATA ##

		#prepare ego network properties
		egonet_props, egonet_acts = dm.egonet_props_acts( dataname, eventname, root_data, 'y', saveloc )


		## PLOTTING ##

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos] )
		sns.despine( ax=ax ) #take out spines
		if grid_pos in [10, 11, 12, 13]:
			plt.xlabel( prop_xlabel, size=plot_props['xylabel'] )
		if grid_pos in [0, 4, 8, 12]:
			plt.ylabel( prop_ylabel, size=plot_props['xylabel'] )

		#prepare data
		yplot_data = egonet_props[ prop_name ]
		xplot, yplot = pm.plot_compcum_dist( yplot_data ) #complementary cumulative dist

		#plot plot!
		plt.loglog( xplot, yplot, '-', c=colors[0], lw=plot_props['linewidth'], ms=plot_props['marker_size'] )

		#texts
		plt.text( 1, 1.1, textname, va='top', ha='right', transform=ax.transAxes, fontsize=plot_props['text_size'] )

		#finalise subplot
		if prop_name == 'degree':
			plt.axis([ 5e-1, 1e4, 5e-6, 5e0 ])
		if prop_name == 'strength':
			plt.axis([ 9e-2, 1e5, 5e-6, 5e0 ])
		if prop_name == 'act_avg':
			plt.axis([ 5e-1, 1e4, 5e-6, 5e0 ])
		if prop_name == 'act_min':
			plt.axis([ 5e-1, 1e4, 5e-6, 5e0 ])
		if prop_name == 'act_max':
			plt.axis([ 5e-1, 1e4, 5e-6, 5e0 ])
		ax.tick_params( axis='both', which='major', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		ax.locator_params( axis='x', numticks=6 )
		ax.locator_params( axis='y', numticks=6 )


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
