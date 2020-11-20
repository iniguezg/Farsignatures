#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (ALPHAS CCDF) IN FARSIGNATURES PROJECT ###

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

	#property to plot
	prop_names = [ 'gamma' ]
	prop_labels = [ r'$\alpha_r$' ]

	alphamax = 1000 #maximum alpha for MLE fit
	nsims = 1000 #number of syntethic datasets used to calculate p-value
	amax = 10000 #maximum activity for theoretical activity distribution

	pval_thres = 0.1 #threshold above which alpha MLEs are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf

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
	'text_size' : 11,
	'marker_size' : 5,
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
	'grid_params' : dict( left=0.08, bottom=0.08, right=0.98, top=0.97, wspace=0.3, hspace=0.5 ),
	'dpi' : 300,
	'savename' : 'figure_alphas_CCDF' }

	colors = sns.color_palette( 'Set2', n_colors=len(prop_names) ) #colors to plot

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )

	#loop through considered datasets
	for grid_pos, (dataname, eventname, textname) in enumerate(datasets):
#		for grid_pos, (dataname, eventname, textname) in enumerate([ ('MPC_UEu_net', 'MPC_UEu.evt', 'Mobile (call)') ]):
		print( 'dataset name: ' + eventname[:-4] ) #print output

		## DATA ##

		#prepare ego network properties
		egonet_props, egonet_acts = dm.egonet_props_acts( dataname, eventname, root_data, 'y', saveloc )

		#fit activity model to all ego networks in dataset
		egonet_fits = dm.egonet_fits( dataname, eventname, root_data, 'y', saveloc, alphamax=alphamax, nsims=nsims, amax=amax )

		#filter egos according to fitting results
		egonet_filter, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )

		#some measures
		num_egos = len( egonet_props ) #all egos
		num_egos_filter = len( egonet_filter ) #filtered egos
		frac_egos_random = ( egonet_filter.gamma > 1 ).sum() / float( num_egos_filter ) #fraction of egos in random regime (gamma > 1, i.e. alpha > 1 - a0)


		## PRINTING ##

		print( 'fraction filtered egos = {:.2f}'.format( num_egos_filter / float(num_egos) ) )
		print( 'fraction inf egos = {:.2f}'.format( len(egonet_inf) / float(num_egos) ) )
		print( 'fraction null egos = {:.2f}'.format( len(egonet_null) / float(num_egos) ) )
		print( '\n' )


		## PLOTTING ##

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos] )
		sns.despine( ax=ax ) #take out spines
		if grid_pos in [10, 11, 12, 13]:
			plt.xlabel( r'$\alpha_r$', size=plot_props['xylabel'] )
		if grid_pos in [0, 4, 8, 12]:
			plt.ylabel( r"CCDF $P[ \alpha_r' \geq \alpha_r ]$", size=plot_props['xylabel'] )

		#loop through considered properties
		for prop_pos, (prop_name, prop_label) in enumerate(zip( prop_names, prop_labels )):

			#prepare data
			yplot_data = egonet_filter[ prop_name ] #filtered data!
			xplot, yplot = pm.plot_CCDF_cont( yplot_data ) #complementary cumulative dist

			#plot plot!
			plt.loglog( xplot, yplot, '-', label=prop_label, c=colors[prop_pos], lw=plot_props['linewidth'] )

		#lines
		plt.vlines( x=1, ymin = 1e-5, ymax=frac_egos_random, linestyles='--', colors='0.6', lw=plot_props['linewidth'] )
		plt.hlines( y=frac_egos_random, xmin = 1e-3, xmax=1, linestyles='--', colors='0.6', lw=plot_props['linewidth'] )

		#texts

		plt.text( 1, 1.15, textname, va='top', ha='right', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		txt_str = r'$N_{\alpha}=$ '+'{}'.format(num_egos_filter)+'\n'+r'$n_{RN} =$'+'{:.2f}'.format(frac_egos_random)
		plt.text( 0.05, 0.05, txt_str, va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['text_size'] )

		#finalise subplot
		plt.axis([ 1e-3, 1e3, 1e-5, 2e0 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		ax.locator_params( numticks=5 )
		if grid_pos not in [10, 11, 12, 13]:
			ax.tick_params(labelbottom=False)
		if grid_pos not in [0, 4, 8, 12]:
			ax.tick_params(labelleft=False)

	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )


#DEBUGGIN'

#			plt.loglog( xplot, yplot, '-', label=prop_label, mec=colors[prop_pos], mfc='w', mew=plot_props['linewidth'], ms=plot_props['marker_size'] )
