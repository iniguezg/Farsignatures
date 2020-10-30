#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (APROXs & SIMS FOR ACT DIST) IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
import numpy as np
import seaborn as sns
import matplotlib as mpl
#import scipy.stats as ss
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
	# alpha_vals = [ -0.7, 0., 9. ] #PA parameter
	# t_vals = [ 2., 10., 100., 1000. ] #mean alter activity (max time in dynamics)
	alpha_vals = [ -0.7 ]
	t_vals = [ 2., 10., 100. ]
	a_vals = np.unique( np.logspace( np.log10( a0 ), np.log10(max( t_vals ))+1, num=100, dtype=int ) ) #alter activities

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
	'marker_size' : 4,
	'linewidth' : 2,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':9 },
	'legend_hlen' : 3,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables
	fig_props = { 'fig_num' : 1,
	'fig_size' : (12, 5),
	'aspect_ratio' : (1, 3),
	'grid_params' : dict( left=0.07, bottom=0.125, right=0.985, top=0.95, wspace=0.1 ),
	'dpi' : 300,
	'savename' : 'figure_activity_num_approx' }

	colors = sns.color_palette( 'Set2', n_colors=len(t_vals) ) #colors to plot


	## PLOTTING ##

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )


# A-C: time dependence of activity distribution for varying alpha

	for alphapos, alpha in enumerate( alpha_vals ): #loop through alpha values
		params['alpha'] = alpha #PA parameter
		gamma = alpha + a0 #rescaled parameter

		print( 'alpha = '+str( alpha ) )

		#initialise subplot
		ax = plt.subplot( grid[ alphapos ] )
		sns.despine( ax=ax ) #take out spines
		plt.xlabel( r'activity $a$', size=plot_props['xylabel'] )
		if alphapos == 0:
			plt.ylabel( r'PDF $p_a(t)$', size=plot_props['xylabel'] )

		#plot plot!

#SIMS

		for post, t in enumerate( t_vals ): #loop through times
			params['t'] = t #mean alter activity (max time in dynamics)

			#load model of alter activity, according to parameters
			activity = mm.model_activity( params, loadflag='y', saveloc=saveloc_model )

			bins = np.logspace( 0, np.log10( activity.max().max() ), num=50 )
			yplot, bin_edges = np.histogram( activity, bins=bins, density=True )
			xplot = [ ( bin_edges[i+1] + bin_edges[i] ) / 2 for i in range(len( bin_edges[:-1] )) ]

			plt.loglog( xplot, yplot, 'o', label=None, c=colors[post], ms=plot_props['marker_size'], zorder=0 )

#THEO

		for post, t in enumerate( t_vals ): #loop through times
			#PGF expression

			xplot = a_vals
			yplot_model = np.array([ mm.activity_dist( a, t, alpha, a0 ) for a in xplot ])

			plt.loglog( xplot, yplot_model, '-', label='{:.0f}'.format(t), c=colors[post], lw=plot_props['linewidth'], zorder=0 )

			#gamma approximation

			xplot = a_vals[1:] #disregard a = a_0 for gamma approx
			yplot_model = np.array([ mm.activity_dist_gamma( a, t, alpha, a0 ) for a in xplot ])

			label = 'Eq. ()' if post == 3 else None

			plt.loglog( xplot, yplot_model, '--', label=label, c=colors[post], lw=plot_props['linewidth'], zorder=1 )

		#text
		if alphapos == 0:
			reg_str = 'preferential attachment regime\n'
		if alphapos == 1:
			reg_str = 'crossover regime\n'
		if alphapos == 2:
			reg_str = 'random regime\n'
		plt.text( 0.5, 1.05, reg_str+r'($\gamma=$ {:.1f}'.format(gamma)+')', va='top', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		#legend
		if alphapos == 0:
			leg = plt.legend( loc='upper right', bbox_to_anchor=(1, 0.95), title=r'$t=$', prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=1 )

		#finalise subplot
		plt.axis([ 1e0, max(a_vals), 1e-4, 1e0 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		if alphapos > 0:
			plt.yticks([])


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )

#DEBUGGIN'

		# #low alpha: power-law limit
		# if alphapos == 0:
		# 	label = 'Power law'
		# 	act_dist = lambda a1, alpha1: mp.power( a1, -( 1 - alpha1 ) ) / mp.gamma( alpha1 )
		#
		# 	yplot = [ float( act_dist( a, alpha ) ) for a in a_vals ]
		# 	plt.loglog( xplot, yplot, '--', label='Eq. (S11)', c='0.5', lw=plot_props['linewidth'], zorder=2 )
		#
		# #intermediate alpha: power law with exponential cutoff
		# if alphapos == 1:
		# 	label = 'Power law with exponential cutoff'
		# 	t = t_vals[-1] #consider only large t
		#
		# 	lt = lambda alpha1: mp.ln( 1 + alpha1 / t )
		# 	exp_decay = lambda a1, alpha1: mp.exp( -lt(alpha1) * ( a1 + alpha1 ) + alpha1 * mp.ln( alpha1 / t ) )
		# 	act_dist = lambda a1, alpha1: mp.power( a1, -( 1 - alpha1 ) ) * exp_decay( a1, alpha1 ) / mp.gamma( alpha1 )
		#
		# 	yplot = [ float( act_dist( a, alpha ) ) for a in a_vals ]
		# 	plt.loglog( xplot, yplot, '--', label='Eq. (S12)', c='0.5', lw=plot_props['linewidth'], zorder=2 )
		#
		# #high alpha: Poisson limit
		# if alphapos == 2:
		# 	label = 'Poisson'
		# 	mu = float( t_vals[-1] ) #Poisson mean is time
		#
		# 	xplot = np.arange( ss.poisson.ppf( ppf, mu ), ss.poisson.ppf( 1 - ppf, mu ) )
		# 	yplot = ss.poisson.pmf( xplot, mu )
		# 	plt.loglog( xplot, yplot, '--', label='Eq. (S10)', c='0.5', lw=plot_props['linewidth'], zorder=2 )