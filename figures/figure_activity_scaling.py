#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (SCALING FOR ACT DIST) IN FARSIGNATURES PROJECT ###

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
import plot_misc as pm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#parameters
	a0 = 1 #minimum alter activity
	k = 100 #number of alters (ego's degree)
	ntimes = 10000 #number of realizations for averages

	#parameter arrays
	t_vals = [ 2., 10., 100., 1000. ] #mean alter activity (max time in dynamics)

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
	'fig_size' : (12, 8),
	'aspect_ratio' : (2, 3),
	'grid_params' : dict( left=0.065, bottom=0.08, right=0.985, top=0.94, wspace=0.1, hspace=0.5 ),
	'dpi' : 300,
	'savename' : 'figure_activity_scaling' }

	colors = sns.color_palette( 'Set2', n_colors=len(t_vals) ) #colors to plot


	## PLOTTING ##

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )


# A-C: Gamma scaling (with time t) of activity distribution for varying alpha

	#parameter arrays
	alpha_vals = [ -0.7, 0., 9. ] #PA parameter
	a_vals = np.unique( np.logspace( 0, 5, num=100, dtype=int ) ) #alter activities

	for alphapos, alpha in enumerate( alpha_vals ): #loop through alpha values
		params['alpha'] = alpha #PA parameter
		gamma = alpha + a0 #gamma dist shape parameter

		print( 'alpha = '+str( alpha ) )

		#initialise subplot
		ax = plt.subplot( grid[ alphapos ] ) #upper row
		sns.despine( ax=ax ) #take out spines
		plt.xlabel( r'$a_r / \beta_r$', size=plot_props['xylabel'] )
		if alphapos == 0:
			plt.ylabel( r'$\beta_r p_a$', size=plot_props['xylabel'] )

		#plot plot!

#SIMS

		for post, t in enumerate( t_vals ): #loop through times
			params['t'] = t #mean alter activity (max time in dynamics)
			beta = ( t - a0 ) / ( alpha + a0 ) #gamma dist scale parameter

			#load model of alter activity, according to parameters
			activity = mm.model_activity( params, loadflag='y', saveloc=saveloc_model )

			#plot arrays for unbinned distribution
			xplot = np.arange( activity.min(), activity.max()+1, dtype=int )
			yplot, not_used = np.histogram( activity, bins=len(xplot), range=( xplot[0]-0.5, xplot[-1]+0.5 ), density=True )

			#rescale plot arrays
			xplot_resc = ( xplot - a0 ) / beta #rescaled activity ( a - a0 ) / beta
			yplot_resc = beta * yplot #rescaled probability beta * p_d(t)

			line_sims, = plt.loglog( xplot_resc, yplot_resc, 'o', label=None, c=colors[post], ms=plot_props['marker_size'], zorder=0 )

#THEO

		for post, t in enumerate( t_vals ): #loop through times
			beta = ( t - a0 ) / ( alpha + a0 ) #gamma dist scale parameter

			#PGF expression

			xplot = a_vals
			yplot_model = np.array([ mm.activity_dist( a, t, alpha, a0 ) for a in xplot ])

			#rescale plot arrays
			xplot_resc = ( xplot - a0 ) / beta #rescaled activity ( a - a0 ) / beta
			yplot_resc = beta * yplot_model #rescaled probability beta * p_d(t)

			label = '$t = $ {:.0f}'.format(t) if post == 0 else '{:.0f}'.format(t)

			line_theo, = plt.loglog( xplot_resc, yplot_resc, '-', label=label, c=colors[post], lw=plot_props['linewidth'], zorder=0 )

		#gamma approximation (w/ last t value only!)

		xplot = a_vals[1:] #disregard a = a_0 for gamma approx
		xplot_resc = ( xplot - a0 ) / beta #rescaled activity ( a - a0 ) / beta

		#after scaling, gamma dist is standard form!
		yplot_model = np.array([ ss.gamma.pdf( db, gamma ) for db in xplot_resc ])

		line_gamma, = plt.loglog( xplot_resc, yplot_model, '--', label=None, c='k', lw=plot_props['linewidth']+1, zorder=1 )

		#texts

		if alphapos == 0:
			reg_str = 'preferential attachment regime\n'
		if alphapos == 1:
			reg_str = 'crossover regime\n'
		if alphapos == 2:
			reg_str = 'random regime\n'
		plt.text( 0.5, 1.14, reg_str+r'($\alpha_r=$ {:.1f}'.format(gamma)+')', va='top', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		if alphapos == 0:
			plt.text( -0.2, 1, 'GAMMA\nSCALING', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['ticklabel'], weight='bold' )

		#legends
		if alphapos == 0:
			leg = plt.legend( loc='upper left', bbox_to_anchor=(1.15, -0.19), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=len(t_vals) )
		if alphapos == 1:
			leg = plt.legend( (line_sims, line_theo, line_gamma), ('num', 'Eq. (S9)', 'Eq. ()'), loc='upper right', bbox_to_anchor=(1, 1.03), prop=plot_props['legend_prop'], handlelength=2.2, numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ] )

		#finalise subplot
		plt.axis([ 8e-1, 1e2, 5e-7, 1e0 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		ax.locator_params( numticks=6 )
		if alphapos > 0:
			plt.yticks([])


# D-F: Gaussian scaling (with time t) of activity distribution for varying alpha

	#parameter arrays
	alpha_vals = [ 9., 99., 999. ] #PA parameter
	a_vals = np.concatenate(( np.linspace( 1, 9, num=9 ), np.unique( np.logspace( 1, 5, num=1000, dtype=int ) ) )) #alter activities

	for alphapos, alpha in enumerate( alpha_vals ): #loop through alpha values
		params['alpha'] = alpha #PA parameter
		gamma = alpha + a0 #gamma dist shape parameter

		print( 'alpha = '+str( alpha ) )

		#initialise subplot
		ax = plt.subplot( grid[ alphapos + 3 ] ) #lower row
		sns.despine( ax=ax ) #take out spines
		plt.xlabel( r'$(a_r - t_r) / \sqrt{ t_r }$', size=plot_props['xylabel'] )
		if alphapos == 0:
			plt.ylabel( r'$\sqrt{ t_r } p_a$', size=plot_props['xylabel'] )

		#plot plot!

#SIMS

		for post, t in enumerate( t_vals ): #loop through times
			params['t'] = t #mean alter activity (max time in dynamics)

			#load model of alter activity, according to parameters
			activity = mm.model_activity( params, loadflag='y', saveloc=saveloc_model )

			#plot arrays for unbinned distribution
			xplot = np.arange( activity.min(), activity.max()+1, dtype=int )
			yplot, not_used = np.histogram( activity, bins=len(xplot), range=( xplot[0]-0.5, xplot[-1]+0.5 ), density=True )

			#rescale plot arrays
			xplot_resc = ( xplot - t ) / np.sqrt( t - a0 ) #rescaled activity
			yplot_resc = np.sqrt( t - a0 ) * yplot #rescaled probability

			line_sims, = plt.semilogy( xplot_resc, yplot_resc, 'o', label=None, c=colors[post], ms=plot_props['marker_size'], zorder=0 )

#THEO

		for post, t in enumerate( t_vals ): #loop through times

			#PGF expression

			xplot = a_vals
			yplot_model = np.array([ mm.activity_dist( a, t, alpha, a0 ) for a in xplot ])

			#rescale plot arrays
			xplot_resc = ( xplot - t ) / np.sqrt( t - a0 ) #rescaled activity
			yplot_resc = np.sqrt( t - a0 ) * yplot_model #rescaled probability

			line_theo, = plt.semilogy( xplot_resc, yplot_resc, '-', label=None, c=colors[post], lw=plot_props['linewidth'], zorder=0 )

		#Gaussian approximation (w/ last t value only!)

		xplot = a_vals
		xplot_resc = ( xplot - t ) / np.sqrt( t - a0 ) #rescaled activity

		#after scaling, Gaussian dist is standard form!
		yplot_model = np.array([ ss.norm.pdf( db ) for db in xplot_resc ])

		line_gauss, = plt.semilogy( xplot_resc, yplot_model, '--', label=None, c='k', lw=plot_props['linewidth']+1, zorder=1 )

		#texts

		if alphapos == 0:
			reg_str = 'random regime\n'
		if alphapos == 1:
			reg_str = 'random regime\n'
		if alphapos == 2:
			reg_str = 'random regime\n'
		plt.text( 0.5, 1.16, reg_str+r'($\alpha_r=$ {:.1f}'.format(gamma)+')', va='top', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		if alphapos == 0:
			plt.text( -0.2, 1.02, 'GAUSSIAN\nSCALING', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['ticklabel'], weight='bold' )

		#legends
		if alphapos == 1:
			leg = plt.legend( (line_sims, line_theo, line_gauss), ('num', 'Eq. (S9)', 'Eq. ()'), loc='upper right', bbox_to_anchor=(1, 1.05), prop=plot_props['legend_prop'], handlelength=2.2, numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ] )


		#finalise subplot
		plt.axis([ -20, 20, 5e-7, 5e0 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
#		ax.locator_params( numticks=6 )
		if alphapos > 0:
			plt.yticks([])


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
