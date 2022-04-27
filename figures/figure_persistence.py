#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (BETA PERSISTENCE BY PROPERTY) IN FARSIGNATURES PROJECT ###

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


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#plotting variables
	binrange = (-4, 4) #for histogram
	bins = 31

	alphamax = 1000 #maximum alpha for MLE fit
	pval_thres = 0.1 #threshold above which alphas are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf

	#locations
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files

	#dataset list: eventname, textname
	datasets = [ ( 'MPC_UEu', 'Mobile (call)'),
	# datasets = [ ( 'call', 'Mobile (call)'),
	# 			 ( 'text', 'Mobile (sms)'),
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
	'marker_size' : 1,
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
	'grid_params' : dict( left=0.05, bottom=0.07, right=0.99, top=0.97, wspace=0.25, hspace=0.3 ),
	'dpi' : 300,
	'savename' : 'figure_persistence' }

	colors = sns.color_palette( 'Paired', n_colors=1 ) #colors to plot

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )

	#loop through considered datasets
	for grid_pos, (eventname, textname) in enumerate(datasets):
	# for grid_pos, (eventname, textname) in enumerate([ ( 'CNS_calls', 'CNS (call)') ]):
		print( 'dataset name: ' + eventname ) #print output

		#load ego network properties
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
		#fit activity model in all dataset and selected time periods
		egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )
		egonet_fits_piece_0 = pd.read_pickle( saveloc + 'egonet_fits_piece_0_' + eventname + '.pkl' )
		egonet_fits_piece_1 = pd.read_pickle( saveloc + 'egonet_fits_piece_1_' + eventname + '.pkl' )

		#filter egos according to fitting results
		egonet_filt, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )
		egonet_filt_piece_0, egonet_inf_piece_0, egonet_null_piece_0 = dm.egonet_filter( egonet_props, egonet_fits_piece_0, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )
		egonet_filt_piece_1, egonet_inf_piece_1, egonet_null_piece_1 = dm.egonet_filter( egonet_props, egonet_fits_piece_1, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )

		print( 'N = {}'.format( len(egonet_props) ) )
		print( 'full net -> N_alpha = {}, N_inf = {}, N_null = {}'.format( len(egonet_filt), len(egonet_inf), len(egonet_null) ) )
		print( 'piece 0 -> N_alpha = {}, N_inf = {}, N_null = {}'.format( len(egonet_filt_piece_0), len(egonet_inf_piece_0), len(egonet_null_piece_0) ) )
		print( 'piece 1 -> N_alpha = {}, N_inf = {}, N_null = {}'.format( len(egonet_filt_piece_1), len(egonet_inf_piece_1), len(egonet_null_piece_1) ) )


		## PLOTTING ##

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos] )
		sns.despine( ax=ax ) #take out spines

		#plot arrays!

		#get property for all egos common to both time periods
		props_filt = pd.concat( [ egonet_filt.beta,
		egonet_filt_piece_0.beta.rename('beta_piece_0'),
		egonet_filt_piece_1.beta.rename('beta_piece_1')
		], axis=1, join='inner' )
		plot_data = ( props_filt.beta_piece_0 - props_filt.beta_piece_1 ) / props_filt.beta #relative diff in property

		print( '\thet regime = {}, hom regime = {}'.format( len(props_filt[ props_filt.beta > 1 ]), len(props_filt[ props_filt.beta < 1 ]) ) )

		#plot plot!
		sns.histplot( x=plot_data, binrange=binrange, bins=bins, stat='density', element='step', color=colors[0], label='all', zorder=0 )

		#plot perfect persistence line
		plt.axvline( x=0, ls='--', c='0.6', lw=plot_props['linewidth'], zorder=1 )

		#texts
		plt.text( 1, 1, textname, va='bottom', ha='right', transform=ax.transAxes, fontsize=plot_props['text_size'] )

		if grid_pos in [10, 11, 12, 13]:
			plt.xlabel( r'$\Delta \beta / \beta$', size=plot_props['xylabel'] )
		else:
			plt.xlabel('')
			plt.xticks([])
		if grid_pos in [0, 4, 8, 12]:
			plt.ylabel( 'PDF', size=plot_props['xylabel'] )
		else:
			plt.ylabel('')
			plt.yticks([])

		#finalise subplot
		plt.xlim( binrange )
		plt.ylim([ 0, 2 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		ax.locator_params( axis='x', nbins=4 )
		ax.locator_params( axis='y', nbins=3 )


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )

#DEBUGGIN'

		# props_filt_CA = props_filt[ props_filt.beta > 1 ]
		# plot_data_CA = ( props_filt_CA.beta_piece_0 - props_filt_CA.beta_piece_1 ) / props_filt_CA.beta
		# props_filt_RN = props_filt[ props_filt.beta < 1 ]
		# plot_data_RN = ( props_filt_RN.beta_piece_0 - props_filt_RN.beta_piece_1 ) / props_filt_RN.beta

		# sns.histplot( x=plot_data_CA, binrange=binrange, bins=bins, stat='density', element='step', color=colors[1], label='CA', zorder=0 )
		# sns.histplot( x=plot_data_RN, binrange=binrange, bins=bins, stat='density', element='step', color=colors[2], label='RN', zorder=0 )

		# #legend
		# if grid_pos == 0:
		# 	leg = plt.legend( loc='upper left', bbox_to_anchor=(0,1), prop=plot_props['legend_prop'], handlelength=1.7, numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'] )
