#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (ACTIVITY CCDF) IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_misc as dm
import model_misc as mm
import plot_misc as pm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	bounds = (0, 1000) #bounds for alpha MLE fit
	nsims = 100 #number of syntethic datasets used to calculate p-value
	amax = 10000 #maximum activity for theoretical activity distribution

	pval_thres = 0.1 #threshold above which alphas are considered
	alpha_min, alpha_max = 1e-4, 1e2 #extreme values for alpha (to avoid num errors)

	#locations
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files

	#selected dataset: dataname, eventname, textname
	dataname, eventname, textname = ('MPC_UEu_net', 'MPC_UEu.evt', 'Mobile (call)')

	#sizes/widths/coords
	plot_props = { 'xylabel' : 15,
	'figlabel' : 26,
	'ticklabel' : 15,
	'text_size' : 15,
	'marker_size' : 8,
	'linewidth' : 3,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':15 },
	'legend_hlen' : 1.7,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables
	fig_props = { 'fig_num' : 1,
	'fig_size' : (10, 5),
	'aspect_ratio' : (1, 2),
	'grid_params' : dict( left=0.08, bottom=0.13, right=0.99, top=0.9, wspace=0.2 ),
	'dpi' : 100 }

	colors = sns.color_palette( 'Set2', n_colors=2 ) #colors to plot


	## DATA ##

	print( 'dataset name: ' + eventname[:-4] ) #print output

	#prepare ego network properties
	egonet_props, egonet_acts = dm.egonet_props_acts( dataname, eventname, root_data, 'y', saveloc )

	#fit activity model to all ego networks in dataset
	egonet_fits = dm.egonet_fits( dataname, eventname, root_data, 'y', saveloc, bounds=bounds, nsims=nsims, amax=amax )

	#get activity CCDFs for all egos
	CCDFs = egonet_acts.groupby('nodei').apply( pm.plot_compcum_dist )

	#filtering process
	#step 1: egos with t > a_0
	egonet_fits_filter = egonet_fits[ egonet_props.degree * egonet_props.act_min < egonet_props.strength ]
	#step 2: egos with pvalue > threshold
	egonet_fits_filter = egonet_fits_filter[ egonet_fits_filter.pvalue > pval_thres ]
	#step 3: alphas between extreme values (numerical errors)
	egonet_fits_filter = egonet_fits_filter[ egonet_fits_filter.alpha.between( alpha_min, alpha_max ) ]
	egonet_fits_filter.sort_values( by='alpha', inplace=True ) #sort by alpha (increasing)

	#filter CCDFs
	CCDFs_filter = CCDFs[ egonet_fits_filter.index ]


	## PLOTTING ##

	with PdfPages( 'figure_activity_CCDF_{}.pdf'.format( eventname[:-4] ) ) as pdf:

		#loop through selected egos
		for pos_ego, nodei in enumerate( CCDFs_filter.index ):
			if pos_ego % 100 == 0:
				print( 'pos_ego = {}, nodei = {}'.format( pos_ego, nodei ) )

			#initialise plot
			sns.set( style='ticks' ) #set fancy fancy plot
			fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
			plt.clf()
			grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
			grid.update( **fig_props['grid_params'] )

			#prepare model

			#ego parameters, fitted alpha, and activity array
			t, a0, amax = egonet_props.act_avg[nodei], egonet_props.act_min[nodei], egonet_props.act_max[nodei] #ego parameters
			alpha = egonet_fits.alpha[nodei] #fitted alpha
			a_vals = np.arange( a0, amax+1, dtype=int ) #activity range

			#model activity dist and cumulative
			act_dist_theo = np.array([ mm.activity_dist( a, t, alpha, a0 ) for a in a_vals ])
			act_cumdist_theo = np.cumsum( act_dist_theo )


			#plot activity dist (0) and associated CCDF (1)
			for grid_pos in range(2):

				#initialise subplot
				ax = plt.subplot( grid[grid_pos] )
				sns.despine( ax=ax ) #take out spines
				plt.xlabel( r'activity $a$', size=plot_props['xylabel'] )
				if grid_pos == 0:
					plt.ylabel( r"PDF $p_a(t)$", size=plot_props['xylabel'] )
				else:
					plt.ylabel( r"CCDF $P[ a' \geq a ]$", size=plot_props['xylabel'] )

				#plot activity CCDF

				if grid_pos == 0:
					xplot = a_vals
					yplot, not_used = np.histogram( egonet_acts[ nodei ], bins=len(xplot), range=( xplot.min()-0.5, xplot.max()+0.5 ), density=True )
					symbol = 'o'
				else:
					xplot, yplot = CCDFs_filter[ nodei ]
					symbol = '--'

				plt.loglog( xplot, yplot, symbol, c=colors[0], label='data', lw=plot_props['linewidth']-1, ms=plot_props['marker_size'], zorder=0 )

				#plot model CCDF

				xplot = a_vals
				if grid_pos == 0:
					yplot = act_dist_theo #activity dist
				else:
					yplot = np.ones(len( xplot ))
					yplot[1:] = 1 - act_cumdist_theo[:-1] #get activity CCDF P[X >= x]

				plt.loglog( xplot, yplot, '-', c=colors[1], label='model', lw=plot_props['linewidth'], zorder=1 )

				#texts

				if grid_pos == 0:
					plt.text( 1.1, 1.05, 'nodei = {}'.format( nodei ), va='bottom', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'] )

				if grid_pos == 1:
					param_str = '$k =$ {}\n'.format( egonet_props.degree[nodei] )+r'$\tau =$ {}'.format( egonet_props.strength[nodei] )+'\n$t =$ {:.2f}\n$a_0 =$ {}\n$a_m =$ {}\n\n'.format( t, a0, amax )+r'$\alpha =$ {:.4f}'.format( alpha )+'\nKS stat = {:.2f}\np-val = {:.2f}'.format( egonet_fits.statistic[nodei], egonet_fits.pvalue[nodei] )

					plt.text( 1, 0.7, param_str, va='top', ha='right', transform=ax.transAxes, fontsize=plot_props['text_size'] )

					#legend
					plt.legend( loc='upper right', bbox_to_anchor=(1, 1), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'] )

				#finalise subplot
				plt.axis([ 1e0, egonet_props.act_max[ egonet_fits_filter.index ].max()+1, 1./( egonet_props.degree[ egonet_fits_filter.index ].max()+1 ), 1e0 ])
				ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )

			#finalise plot
			pdf.savefig( dpi=fig_props['dpi'] )
			plt.close()


#DEBUGGIN'

#	degrees, num_events, actmeans, amins, amaxs = egonet_props['degree'], egonet_props['strength'], egonet_props['act_avg'], egonet_props['act_min'], egonet_props['act_max'] #unpack props

#			if fig_props['savename'] != '':
#				plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )

#	#join files at the end and delete temp files
#	os.system( 'pdfunite temp_*.pdf figure_activity_CCDF_{}.pdf'.format( eventname[:-4] ) )
#	os.system( 'rm temp_*.pdf' )
