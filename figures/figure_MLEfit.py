#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (MLE FITTING FOR MODEL) IN FARSIGNATURES PROJECT ###

#import modules
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpmath import mp

import model_misc as mm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#parameter dict
	params = {}
	params['alpha'] = 1.0 #cumulative advantage parameter
	params['k'] = 1000 #number of alters (ego's degree)
	params['T'] = 100 #max time (mean alter activity) in dynamics
	params['ntimes'] = 10000 #number of realizations for averages

	#empirical/simulation data variables
	sel_nt = 0 #selected realization for simulated empirical data
	bounds = (0, 100) #bounds within which to look for alpha

	#plotting variables
	num = 50 #number of log bins to plot distributions
	alpha_vals = [ mp.mpf(str( alpha )) for alpha in np.logspace( -1, 2, num=100 ) ] #alpha values for plotting graphical solution

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
	'fig_size' : (10, 5),
	'aspect_ratio' : (1, 2),
	'grid_params' : dict( left=0.08, bottom=0.13, right=0.985, top=0.97, wspace=0.3 ),
	'dpi' : 300,
	'savename' : 'figure_MLEfit' }

	colors = sns.color_palette( 'Set2', n_colors=1 ) #colors to plot


	## DATA ##

	#run model of alter activity (one realization as simulated empirical data!)
	activity = mm.model_activity( params, loadflag=loadflag, saveloc=saveloc )[0]

	#get activity range (for theo distribution)
	a_vals = [ mp.mpf(str( a )) for a in range( min(activity), max(activity)+1 ) ]

	#get mean activity from simulated data (in simulation, t=T with given T)
	t = int( np.mean(activity) ) #mean alter activity


	## FITTING ##

	#solving alpha trascendental equation
	digamma_avg = lambda alpha, activity : mp.fsum([ mp.digamma( alpha + a ) - mp.digamma( alpha ) for a in activity ]) / len( activity )
	alpha_func = lambda alpha, t, activity : t / ( mp.exp( digamma_avg( alpha, activity ) ) - 1 )

	#find MLE optimal alpha for given activity
	alpha_hat = mm.alpha_MLE_fit( activity, bounds )


	## PLOTTING ##

	#initialise plot
	sns.set( style="white" ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )


# A: empirical/simmulated a ctivity distribution and fit

	#initialise subplot
	ax = plt.subplot( grid[0] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( r'activity $a$', size=plot_props['xylabel'] )
	plt.ylabel( r'distribution $p_a(t)$', size=plot_props['xylabel'] )

	#plot plot!

	#plot data

	bins = np.logspace( 0, np.log10( activity.max() ), num=num )
	yplot, bin_edges = np.histogram( activity, bins=bins, density=True )
	xplot = [ ( bin_edges[i+1] + bin_edges[i] ) / 2 for i in range(len( bin_edges[:-1] )) ]

	label = r'Model data ($\alpha =$ '+'{:.2f})'.format( params['alpha'] )
	plt.loglog( xplot, yplot, 'o', label=label, c=colors[0], ms=plot_props['marker_size'], zorder=0 )

	#plot theo distribution (with MLE optimal alpha)

	xplot = [ float(a) for a in a_vals ]
	yplot = [ float( mm.activity_dist_fixed_t( a, t, alpha_hat ) ) for a in a_vals ]

	label = r'MLE fit ($\hat{\alpha} =$ '+'{:.2f})'.format( float(alpha_hat) )
	plt.loglog( xplot, yplot, '-', label=label, c=colors[0], lw=plot_props['linewidth'], zorder=0 )

	#legend
	leg = plt.legend( loc='upper right', bbox_to_anchor=(1, 1), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=1 )

	#finalise subplot
	plt.axis([ 1e0, 1e4, 1e-4, 1e0 ])
	ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )


# B : Graphical solution of trascendental equation for optimal alpha

	#initialise subplot
	ax = plt.subplot( grid[1] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( r'parameter $\alpha$', size=plot_props['xylabel'] )
	plt.ylabel( r'parameter $\alpha$', size=plot_props['xylabel'] )

	#plot plot!

	yplot = [ alpha_func( alpha, params['T'], activity ) for alpha in alpha_vals ]
	plt.loglog( alpha_vals, yplot, 'o', c=colors[0], ms=plot_props['marker_size'], label=r'rhs Eq. (S19)', zorder=0 )

	plt.loglog( alpha_vals, alpha_vals, '--', c=colors[0], lw=plot_props['linewidth'], label=r'lhs Eq. (S19)', zorder=0 )

	label = r'$\alpha = \hat{\alpha}$'
	plt.axvline( x=alpha_hat, ls='-.', c='0.5', lw=plot_props['linewidth'], label=label, zorder=0 )

	#legend
	leg = plt.legend( loc='lower right', bbox_to_anchor=(1, 0), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=1 )

	#finalise subplot
	plt.axis([ float(min(alpha_vals)), float(max(alpha_vals)), float(min(alpha_vals)), float(max(alpha_vals)) ])
	ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )

	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
