#! /usr/bin/env python

#Farsignatures - Exploring dynamics of egocentric communication networks
#Copyright (C) 2023 Gerardo Iñiguez

### SCRIPT FOR PLOTTING FIGURE (ACTIVITY CCDF) IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
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

	#properties to correlate
	propx = ('degree', 'k')
	propy = ('act_avg', 't')

	#alpha fit variables
	alphamax = 1000 #maximum alpha for MLE fit
	pval_thres = 0.1 #threshold above which alpha MLEs are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf

	frac_sel = 0.1 #fraction of egos selected to plot

	#locations
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files

	#selected dataset: eventname, textname
	eventname, textname = ( 'forum', 'Forum')

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

	print( 'dataset name: ' + eventname ) #print output

	#prepare ego network properties, alter activties, and alpha fits
	egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
	egonet_acts = pd.read_pickle( saveloc + 'egonet_acts_' + eventname + '.pkl' )
	egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )

	#filter egos according to fitting results
	egonet_filter, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )

	#select only egos with largest values of chosen properties:
	#statistically significant betas (filter)
	# egonet_selected = egonet_filter.sort_values( by=[propx[0], propy[0]], ascending=False ).iloc[:int(frac_sel*len(egonet_filter)),:]
	#rejected betas (null)
	egonet_selected = egonet_null.sort_values( by=[propx[0], propy[0]], ascending=False ).iloc[:int(frac_sel*len(egonet_null)),:]

	#get activity CCDFs for selected egos (keeping ranked order!)
	CCDFs = egonet_acts.loc[egonet_selected.index,:].groupby('nodei').apply( pm.plot_compcum_dist )[egonet_selected.index]


	# PLOTTING ##

	# with PdfPages( 'figure_activity_CCDF_{}_filter.pdf'.format( eventname ) ) as pdf:
	with PdfPages( 'figure_activity_CCDF_{}_null.pdf'.format( eventname ) ) as pdf:

		#loop through selected egos
		for pos_ego, nodei in enumerate( CCDFs.index ):
			if pos_ego % 100 == 0:
				print( 'pos_ego = {}, nodei = {}'.format( pos_ego, nodei ) )

			#initialise plot
			sns.set( style='ticks' ) #set fancy fancy plot
			fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
			plt.clf()
			grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
			grid.update( **fig_props['grid_params'] )

			#prepare data

			#alter activities of selected ego
			activity = egonet_acts.loc[nodei,:]

			#ego parameters
			k = activity.size #degree
			t = activity.mean() #mean alter activity
			a0 = activity.min() #min/max alter activity
			amax = activity.max()

			#fitted alpha/beta, and activity array
			alpha = egonet_selected.alpha[nodei] #fitted alpha/beta
			beta = (t - a0) / (alpha + a0)
			a_vals = np.arange( a0, amax+1, dtype=int ) #activity range

			#cumulative dist of alter activity in range a=[a0, amax] (i.e. inclusive)
			act_cumdist = ss.cumfreq( activity, defaultreallimits=( a_vals[0]-0.5, a_vals[-1]+0.5 ), numbins=len(a_vals) ).cumcount / k

			#prepare model

			#theo activity dist in range a=[a0, amax] (i.e. inclusive)
			act_dist_theo = np.array([ mm.activity_dist( a, t, alpha, a0 ) for a in a_vals ])
			act_dist_theo /= act_dist_theo.sum() #normalise (due to finite activity range in data)
			#theo cumulative dist
			act_cumdist_theo = np.cumsum( act_dist_theo )
			#difference between cum dists
			cumdist_diff = act_cumdist - act_cumdist_theo

			#Kolmogorov-Smirnov statistic
			KS = np.abs( cumdist_diff ).max()
			KS_aval = a_vals[ np.abs( cumdist_diff ).argmax() ] #activity value where KS statistic appears

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

				#plot activity DF/CCDF

				if grid_pos == 0:
					xplot = a_vals
					yplot, not_used = np.histogram( egonet_acts[ nodei ], bins=len(xplot), range=( xplot.min()-0.5, xplot.max()+0.5 ), density=True )
					symbol = 'o'
				else:
					xplot, yplot = CCDFs[ nodei ]
					symbol = '--'

				plt.loglog( xplot, yplot, symbol, c=colors[0], label='data', lw=plot_props['linewidth']-1, ms=plot_props['marker_size'], zorder=1 )

				#plot model DF/CCDF

				xplot = a_vals
				if grid_pos == 0:
					yplot = act_dist_theo #activity dist
				else:
					yplot = np.ones(len( xplot ))
					yplot[1:] = 1 - act_cumdist_theo[:-1] #get activity CCDF P[X >= x]

				plt.loglog( xplot, yplot, '-', c=colors[1], label='model', lw=plot_props['linewidth'], zorder=2 )

				#plot KS line
				if grid_pos == 1:
					plt.axvline( x=KS_aval, ls='--', c='0.5', lw=plot_props['linewidth']-1, zorder=0 )

				#texts

				if grid_pos == 0:
					plt.text( 1.1, 1.05, 'nodei = {}'.format( nodei ), va='bottom', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'] )

				if grid_pos == 1:
					param_str = '$k =$ {}\n'.format( egonet_selected.degree[nodei] )+r'$\tau =$ {}'.format( egonet_selected.strength[nodei] )+'\n$t =$ {:.2f}\n$a_0 =$ {}\n$a_m =$ {}\n\n'.format( t, a0, amax )+r'$\beta =$ {:.4f}'.format( beta )+'\nKS stat = {:.3f}\np-val = {:.3f}'.format( egonet_selected.statistic[nodei], egonet_selected.pvalue[nodei] )

					plt.text( 1, 0.7, param_str, va='top', ha='right', transform=ax.transAxes, fontsize=plot_props['text_size'] )

					#legend
					plt.legend( loc='upper right', bbox_to_anchor=(1, 1), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props['legend_colsp'] )

				#finalise subplot
				plt.axis([ 1e0, egonet_props.act_max[ egonet_selected.index ].max()+1, 1./( egonet_props.degree[ egonet_selected.index ].max()+1 ), 1e0 ])
				ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )

			#finalise plot
			pdf.savefig( dpi=fig_props['dpi'] )
			plt.close()
