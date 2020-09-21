#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (ACTIVITY DISTRIBUTION, FIXED t) IN FARSIGNATURES PROJECT ###

#import modules
import numpy as np
import seaborn as sns
import matplotlib as mpl
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpmath import mp

import model_misc as mm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#parameters
	ppf = 1e-5 #percent point function (to plot Poisson limit)

	#parameter arrays
	alpha_vals = [ mp.mpf( alpha ) for alpha in [ '0.01', 1, 100 ] ]
	t_vals = [ mp.mpf( t ) for t in [ 1, 10, 100, 1000 ] ]
	a_vals = [ mp.mpf(str( a )) for a in np.logspace( 0, np.log10( float(max(t_vals)) )+1, num=100 ) ]

	#parameter dict
	params = {}
	params['k'] = 100 #number of alters (ego's degree)
	params['ntimes'] = 10000 #number of realizations for averages

	#flags and locations
	loadflag = 'y'
	saveloc = 'files/model/fixed_t/' #location of output files

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
	'savename' : 'figure_activity_dist_fixed_t' }

#	colors = sns.color_palette( 'GnBu', n_colors=len(t_vals)+2 ) #colors to plot
	colors = sns.color_palette( 'Set2', n_colors=len(t_vals) ) #colors to plot


	## PLOTTING ##

	#initialise plot
	sns.set( style="white" ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )


# A-C: time dependence of activity distribution for varying alpha

	for alphapos, alpha in enumerate( alpha_vals ): #loop through alpha values
		params['alpha'] = float( alpha ) #CA parameter

		print( 'alpha = '+str( alpha ) )

		#initialise subplot
		ax = plt.subplot( grid[ alphapos ] )
		sns.despine( ax=ax ) #take out spines
		plt.xlabel( r'activity $a$', size=plot_props['xylabel'] )
		if alphapos == 0:
			plt.ylabel( r'distribution $p_a(t)$', size=plot_props['xylabel'] )

		#plot plot!

#SIMS

		for post, t in enumerate( t_vals ): #loop through times
			params['T'] = float( t ) #time to run dynamics (t=T to compare sims/theo)

			#run model of alter activity, according to parameters
			activity = mm.model_activity( params, loadflag=loadflag, saveloc=saveloc )

			bins = np.logspace( 0, np.log10( activity.max().max() ), num=50 )
			yplot, bin_edges = np.histogram( activity, bins=bins, density=True )
			xplot = [ ( bin_edges[i+1] + bin_edges[i] ) / 2 for i in range(len( bin_edges[:-1] )) ]

			plt.loglog( xplot, yplot, 'o', label=None, c=colors[post], ms=plot_props['marker_size'], zorder=0 )

#THEO

		xplot = [ float(a) for a in a_vals ]

		for post, t in enumerate( t_vals ): #loop through times
			yplot = [ float( mm.activity_dist_fixed_t( a, t, alpha ) ) for a in a_vals ]

			plt.loglog( xplot, yplot, '-', label=str(t), c=colors[post], lw=plot_props['linewidth'], zorder=1 )

#			print( 'activity integral = {}'.format( sum(yplot) ) )

		#low alpha: power-law limit
		if alphapos == 0:
			label = 'Power law'
			act_dist = lambda a1, alpha1: mp.power( a1, -( 1 - alpha1 ) ) / mp.gamma( alpha1 )

			yplot = [ float( act_dist( a, alpha ) ) for a in a_vals ]
			plt.loglog( xplot, yplot, '--', label='Eq. (S11)', c='0.5', lw=plot_props['linewidth'], zorder=2 )

		#intermediate alpha: power law with exponential cutoff
		if alphapos == 1:
			label = 'Power law with exponential cutoff'
			t = t_vals[-1] #consider only large t

			lt = lambda alpha1: mp.ln( 1 + alpha1 / t )
			exp_decay = lambda a1, alpha1: mp.exp( -lt(alpha1) * ( a1 + alpha1 ) + alpha1 * mp.ln( alpha1 / t ) )
			act_dist = lambda a1, alpha1: mp.power( a1, -( 1 - alpha1 ) ) * exp_decay( a1, alpha1 ) / mp.gamma( alpha1 )

			yplot = [ float( act_dist( a, alpha ) ) for a in a_vals ]
			plt.loglog( xplot, yplot, '--', label='Eq. (S12)', c='0.5', lw=plot_props['linewidth'], zorder=2 )

		#high alpha: Poisson limit
		if alphapos == 2:
			label = 'Poisson'
			mu = float( t_vals[-1] ) #Poisson mean is time

			xplot = np.arange( ss.poisson.ppf( ppf, mu ), ss.poisson.ppf( 1 - ppf, mu ) )
			yplot = ss.poisson.pmf( xplot, mu )
			plt.loglog( xplot, yplot, '--', label='Eq. (S10)', c='0.5', lw=plot_props['linewidth'], zorder=2 )

		#text
		plt.text( 0.5, 1.05, label+'\n'+r'($\alpha=$ '+str(alpha)+')', va='top', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		#legend
		leg = plt.legend( loc='upper right', bbox_to_anchor=(1, 0.95), title=r'$t=$', prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=1 )

		#finalise subplot
		plt.axis([ 1e0, float(max(a_vals)), 1e-4, 1e0 ])
		ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )
		if alphapos > 0:
			plt.yticks([])


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
