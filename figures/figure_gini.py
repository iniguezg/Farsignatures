#! /usr/bin/env python

#Farsignatures - Exploring dynamics of egocentric communication networks
#Copyright (C) 2023 Gerardo Iñiguez

### SCRIPT FOR PLOTTING FIGURE (GINI COEFFICIENT AND BETA) IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
import numpy as np
import seaborn as sns
import itertools as it
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpmath import mp
from os.path import expanduser

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import model_misc as mm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#parameters
	a0, am = 1, 100000 #min/max alter activity
	a_vals = np.arange(a0, am+1, dtype=int)

	#parameter arrays
	t_vals = [ 2., 10., 100., 1000. ] #mean alter activity
	beta_vals = np.logspace(-2, 4, 20) #crossover parameter
	beta_vals_approx = np.logspace(0, 4, 20)

	#locations
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc_data = root_code+'files/data/' #location of output files (data/model)
	saveloc_model = root_code+'files/model/'

	#sizes/widths/coords
	plot_props = { 'xylabel' : 15,
	'figlabel' : 26,
	'ticklabel' : 13,
	'text_size' : 12,
	'marker_size' : 6,
	'linewidth' : 2,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':12 },
	'legend_hlen' : 1,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables
	fig_props = { 'fig_num' : 1,
	'fig_size' : (8, 6),
	'aspect_ratio' : (1, 1),
	'grid_params' : dict( left=0.08, bottom=0.1, right=0.98, top=0.95, wspace=0.1 ),
	'dpi' : 300,
	'savename' : 'figure_gini' }

	colors = sns.color_palette( 'GnBu', n_colors=len(t_vals) ) #colors to plot

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )

	#initialise subplot
	ax = plt.subplot( grid[0] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( r'$\beta$', size=plot_props['xylabel'] )
	plt.ylabel( r'$G$', size=plot_props['xylabel'] )


	# Gini coefficient as a function of beta parameter in activity model

	lines_theo = [] #initialise list of plot handles / labels
	labels_theo = []

	for post, t in enumerate(t_vals): #loop through times
		trel = t - a0 #relative time

		gini_theo = np.zeros(len(beta_vals)) #initialise Gini array (theo)
		for posb, beta in enumerate(beta_vals): #loop through betas
			alpha = trel / beta - a0 #alpha

			## THEO ##

			#theo activity dist in range a=[a0, amax] (i.e. inclusive)
			act_dist_theo = np.array([ mm.activity_dist( a, t, alpha, a0 ) for a in a_vals ])
			act_dist_theo /= act_dist_theo.sum() #normalise (due to finite activity range in data)
			#theo cumulative dist
			act_cumdist_theo = np.cumsum( act_dist_theo )

			# #Gini coefficient
			gini_theo[posb] = 1 - (( 1 - act_cumdist_theo )**2).sum() / t

			#check for expected behaviour
			print('t = {}, avg of dist = {}'.format(t, np.sum( act_dist_theo * a_vals )))

			# for pos_ai, pos_aj in it.product(range(len(a_vals)), repeat=2):
			# 	gini_theo[posb] += act_dist_theo[pos_ai] * act_dist_theo[pos_aj] * np.abs( a_vals[pos_ai] - a_vals[pos_aj] )
			# gini_theo[posb] /= 2*t


		## GAMMA APPROX ##

		gini_approx = np.zeros(len(beta_vals_approx)) #initialise Gini array (gamma approx)
		for posb, beta in enumerate(beta_vals_approx): #loop through betas
			#gamma parameter (relative alpha)
			gamma_half = mp.mpf(str( trel / beta + 1/2 ))
			gamma_one = mp.mpf(str( trel / beta + 1 ))

			gini_approx[posb] =  mp.gammaprod( [gamma_half, mp.mpf('1')], [gamma_one, mp.mpf('1/2')] )


		## PLOTTING ##

		#theo
		curve_theo, = plt.semilogx( beta_vals, gini_theo, 'o', c=colors[post], ms=plot_props['marker_size'], zorder=2 )

		lines_theo.append( curve_theo ) #append curve handles/labels
		labels_theo.append( '{:.0f}'.format(t) )

		#gamma approx
		curve_approx, = plt.semilogx( beta_vals_approx, gini_approx, '-', c=colors[post], lw=plot_props['linewidth'], zorder=1 )

		#lines and texts
		plt.axvline( 1, ls='--', c='0.8', lw=plot_props['linewidth'], zorder=0 )
		plt.text( 0.1, 1.01, 'homogeneous regime', va='bottom', ha='center', fontsize=plot_props['text_size'] )
		plt.text( 100, 1.01, 'heterogeneous regime', va='bottom', ha='center', fontsize=plot_props['text_size'] )

	#legends
	leg1 = plt.legend( lines_theo, labels_theo, title='$t$', loc='lower right', bbox_to_anchor=(1, 0.1), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=2 )
	ax.add_artist(leg1)
	leg2 = plt.legend( (lines_theo[0], curve_approx), ('theo', 'gamma'), loc='lower right', bbox_to_anchor=(1, 0), prop=plot_props['legend_prop'], handlelength=1.6, numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=2 )
	ax.add_artist(leg2)

	#finalise subplot
	plt.axis([ beta_vals[0], beta_vals[-1], 0, 1 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
