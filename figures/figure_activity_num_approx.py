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
import plot_misc as pm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#parameters
	a0 = 1 #minimum alter activity
	ntimes = 10000 #number of realizations for averages

	#parameter arrays
	k_vals = [ 100, 10 ] #number of alters (ego's degree)
	alpha_vals = [ -0.7, 0., 9. ] #PA parameter
	t_vals = [ 2., 10., 100., 1000. ] #mean alter activity (max time in dynamics)
	a_vals = np.unique( np.logspace( np.log10(a0), 4, num=100, dtype=int ) ) #alter activities

	#parameter dict
	params = { 'a0' : a0, 'ntimes' : ntimes }

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
	'grid_params' : dict( left=0.06, bottom=0.075, right=0.99, top=0.96, wspace=0.1, hspace=0.4 ),
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


# Time dependence of activity distribution for varying alpha
# A-C and D-F: 2 values of degree

	for kpos, k in enumerate( k_vals ): #loop through k values
		params['k'] = k #set degree

		print( 'k = '+str( k ) )

		for alphapos, alpha in enumerate( alpha_vals ): #loop through alpha values
			params['alpha'] = alpha #PA parameter
			gamma = alpha + a0 #rescaled parameter

			print( '\talpha = '+str( alpha ) )

			#initialise subplot
			ax = plt.subplot( grid[ 3*kpos + alphapos ] )
			sns.despine( ax=ax ) #take out spines
			plt.xlabel( r'$a$', size=plot_props['xylabel'] )
			if alphapos == 0:
				plt.ylabel( r'$p_a$', size=plot_props['xylabel'] )

			#plot plot!

	#SIMS

			for post, t in enumerate( t_vals ): #loop through times
				params['t'] = t #mean alter activity (max time in dynamics)

				#load model of alter activity, according to parameters
				activity = mm.model_activity( params, loadflag='y', saveloc=saveloc_model )

				#plot arrays for unbinned distribution
				xplot = np.arange( activity.min(), activity.max()+1, dtype=int )
				yplot, not_used = np.histogram( activity, bins=len(xplot), range=( xplot[0]-0.5, xplot[-1]+0.5 ), density=True )

				line_sims, = plt.loglog( xplot, yplot, 'o', label=None, c=colors[post], ms=plot_props['marker_size'], zorder=0 )

	#THEO

			for post, t in enumerate( t_vals ): #loop through times
				#PGF expression

				xplot = a_vals
				yplot_model = np.array([ mm.activity_dist( a, t, alpha, a0 ) for a in xplot ])

				label = '$t = $ {:.0f}'.format(t) if post == 0 else '{:.0f}'.format(t)

				line_theo, = plt.loglog( xplot, yplot_model, '-', label=label, c=colors[post], lw=plot_props['linewidth'], zorder=0 )

			#texts

			plt.text( 0.5, 1, r'$\alpha_r=$ {:.1f}'.format(gamma), va='top', ha='center', transform=ax.transAxes, fontsize=plot_props['xylabel'] )

			if alphapos == 0:
				plt.text( -0.18, 1.02, '$k =$ {}'.format(k), va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['xylabel'] )

			#legends
			if kpos == 0 and alphapos == 0:
				leg = plt.legend( loc='upper left', bbox_to_anchor=(1.15, -0.19), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=len(t_vals) )
			if alphapos == 1:
				leg = plt.legend( (line_sims, line_theo), ('num', 'theo'), loc='upper right', bbox_to_anchor=(1, 0.9), prop=plot_props['legend_prop'], handlelength=1.6, numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ] )

			#finalise subplot
			plt.axis([ 8e-1, 5e4, 5e-7, 1e0 ])
			ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
			ax.locator_params( numticks=6 )
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

			# bins = np.logspace( 0, np.log10( activity.max() ), num=25 )
			# yplot, bin_edges = np.histogram( activity, bins=bins, density=True )
			# xplot = [ ( bin_edges[i+1] + bin_edges[i] ) / 2 for i in range(len( bin_edges[:-1] )) ]
