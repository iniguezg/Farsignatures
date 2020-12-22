#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (APROXs & SIMS FOR ACT DIST) IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
import numpy as np
import seaborn as sns
import matplotlib as mpl
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import model_misc as mm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#parameters
	a0 = 1 #minimum alter activity
	k = 100 #number of alters (ego's degree)
	ntimes = 10000 #number of realizations for averages

	#parameter arrays
	beta_vals = [ 100., 1., 0.01 ] #crossover parameter
	trel_vals = [ 0.1, 1., 10., 100., 1000. ] #relative mean alter activity

	#parameter dict
	params = { 'a0' : a0, 'k' : k, 'ntimes' : ntimes }

	#locations
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc_data = root_code+'files/data/' #location of output files (data/model)
	saveloc_model = root_code+'files/model/'

	#sizes/widths/coords
	plot_props = { 'xylabel' : 15,
	'figlabel' : 26,
	'ticklabel' : 13,
	'text_size' : 8,
	'marker_size' : 3,
	'linewidth' : 2,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':12 },
	'legend_hlen' : 1,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables
	fig_props = { 'fig_num' : 1,
	'fig_size' : (12, 5),
	'aspect_ratio' : (1, 3),
	'grid_params' : dict( left=0.06, bottom=0.18, right=0.99, top=0.98, wspace=0.1 ),
	'dpi' : 300,
	'savename' : 'figure_activity_regimes' }

	colors = sns.color_palette( 'Set2', n_colors=len(trel_vals) ) #colors to plot


	## PLOTTING ##

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )


# Time dependence of activity distribution for varying beta

	for betapos, beta in enumerate( beta_vals ): #loop through beta values
		print( 'beta = '+str( beta ) )

		#initialise subplot
		ax = plt.subplot( grid[ betapos ] )
		sns.despine( ax=ax ) #take out spines
		plt.xlabel( r'$a_r$', size=plot_props['xylabel'] )
		if betapos == 0:
			plt.ylabel( r'$p_a$', size=plot_props['xylabel'] )

		#plot plot!

#SIMS

		line_sims = [] #initialise list of plot handles / labels
		labels = []

		for post, trel in enumerate( trel_vals ): #loop through relative times
			t = trel + a0 #get mean alter activity
			gamma = trel / beta #gamma parameter (relative alpha)
			alpha = gamma - a0 #get alpha

			params['alpha'] = alpha #PA parameter
			params['t'] = t #mean alter activity (max time in dynamics)

			#load model of alter activity, according to parameters
			activity = mm.model_activity( params, loadflag='y', saveloc=saveloc_model )

			a_vals = np.arange( activity.min(), activity.max()+1, dtype=int ) #activity range

			#plot arrays for unbinned distribution
			xplot = a_vals - a0
			yplot, not_used = np.histogram( activity, bins=len(a_vals), range=( a_vals[0]-0.5, a_vals[-1]+0.5 ), density=True )

			labels.append( r'$t_r =$ {}'.format( trel ) )

			curve, = plt.loglog( xplot, yplot, 'o', c=colors[post], ms=plot_props['marker_size'], zorder=0 )
			line_sims.append( curve ) #append handle

#THEO

		a_vals = np.unique( np.logspace( np.log10(a0+1), 4, num=100, dtype=int ) ) #alter activities

		for post, trel in enumerate( trel_vals ): #loop through relative times
			t = trel + a0 #get mean alter activity
			gamma = trel / beta #gamma parameter (relative alpha)
			alpha = gamma - a0 #get alpha

			print( '\t\talpha = {}, t = {}'.format( alpha, t ) ) #to know what we plot

			#gamma approximation (heterogeneous/crossover regimes)
			if betapos in [0, 1]:

				xplot = a_vals - a0 #relative activity
				yplot_gamma = np.array([ ss.gamma.pdf( a, gamma, a0, beta ) for a in a_vals ])

				line_gamma, = plt.loglog( xplot, yplot_gamma, '-', label=None, c=colors[post], lw=plot_props['linewidth'], zorder=0 )

			#Poisson approximation (crossover/homogeneous regimes)
			if betapos in [1, 2]:

				xplot = a_vals - a0 #relative activity
				yplot_poisson = np.array([ ss.poisson.pmf( a, trel ) for a in a_vals ])

				line_poisson, = plt.loglog( xplot, yplot_poisson, '--', label=None, c=colors[post], lw=plot_props['linewidth'], zorder=0 )


		#text
		if betapos == 0:
			reg_str = 'heterogeneous regime\n'
		if betapos == 1:
			reg_str = 'crossover\n'
		if betapos == 2:
			reg_str = 'homogeneous regime\n'
		plt.text( 0.5, 1, reg_str+r'($\beta=$ {}'.format(beta)+')', va='top', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		#legends

		if betapos == 1:
			leg1 = plt.legend( line_sims, labels, loc='upper center', bbox_to_anchor=(0.5, -0.12), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=len(trel_vals) )
			ax.add_artist(leg1)

		if betapos == 0:
			leg2 = plt.legend( (line_sims[0], line_gamma), ('num', 'gamma'), loc='upper right', bbox_to_anchor=(1, 0.9), prop=plot_props['legend_prop'], handlelength=1.6, numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ] )
		if betapos == 1:
			leg2 = plt.legend( (line_sims[0], line_gamma, line_poisson), ('num', 'gamma', 'Poisson'), loc='upper right', bbox_to_anchor=(1, 0.9), prop=plot_props['legend_prop'], handlelength=1.6, numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ] )
		if betapos == 2:
			leg2 = plt.legend( (line_sims[0], line_poisson), ('num', 'Poisson'), loc='upper right', bbox_to_anchor=(1, 0.9), prop=plot_props['legend_prop'], handlelength=1.6, numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ] )
		ax.add_artist(leg2)

		#finalise subplot
		plt.axis([ 8e-1, 5e3, 5e-7, 1e0 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		ax.locator_params( numticks=6 )
		if betapos > 0:
			plt.yticks([])


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )

#DEBUGGIN'

			# #PGF expression
			#
			# xplot = a_vals - a0 #relative activity
			# yplot_model = np.array([ mm.activity_dist( a, t, alpha, a0 ) for a in a_vals ])
			#
			# line_theo, = plt.loglog( xplot, yplot_model, '-', label=label, c=colors[post], lw=plot_props['linewidth'], zorder=0 )
