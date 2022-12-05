#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (MLE FITTING FOR MODEL) IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
import numpy as np
import seaborn as sns
import scipy.special as sps
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import model_misc as mm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#parameter dict (target parameters for MLE)
	params = {}
	params['a0'] = 1 #minimum alter activity
	params['k'] = 1000 #number of alters (ego's degree)
	params['ntimes'] = 1000 #number of realizations for averages

	#parameter arrays
	alpha_vals = [ -0.7, 0., 9. ] #target PA parameter
#	alpha_vals = [ 0. ]
	t_vals = [ 2., 10., 100., 1000. ] #target mean alter activity (max time in dynamics)
#	t_vals = [ 100. ]

	#fitting parameters
	alpha_bounds=(1e-4, 1e3) #bounds for alpha search

	#plot arrays
	sim_vals = range(params['ntimes']) #range of simulations to consider
	# sim_vals = range(50)
	gamma_vals = np.logspace( -1, 2, num=100 ) #range of estimated relative alpha to plot

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
	'grid_params' : dict( left=0.05, bottom=0.19, right=0.985, top=0.94, wspace=0.1 ),
	'dpi' : 300,
	'savename' : 'figure_MLEfit' }

	colors = sns.color_palette( 'Set2', n_colors=len(t_vals) ) #colors to plot


	## MLE ##

	#MLE functions
	digamma_avg = lambda alpha, activity, a0 : sps.digamma( alpha+activity ).mean() - sps.digamma( alpha+a0 )
	beta = lambda alpha, a0, t : ( t - a0 ) / ( alpha + a0 )
	alpha_func = lambda alpha, activity, a0, t : digamma_avg(alpha, activity, a0) - np.log( 1 + beta(alpha, a0, t) )


	## PLOTTING ##

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )


# MLE equations for varying beta

	for alphapos in range(len( alpha_vals )): #loop through target alpha values
		params['alpha'] = alpha_vals[ alphapos ] #PA parameter
		params['gamma'] = params['alpha'] + params['a0'] #rescaled parameter
		print( 'alpha_r = {:.2f}'.format( params['gamma'] ) )

		#initialise subplot
		ax = plt.subplot( grid[ alphapos ] )
		sns.despine( ax=ax ) #take out spines
		ax.set_xlabel( r'$\alpha_r$', size=plot_props['xylabel'] )
		if alphapos == 0:
			ax.set_ylabel( r'$ \langle f_{ \alpha } - \ln ( 1 + \beta ) \rangle $', size=plot_props['xylabel'] )

		#initialise inset
		loc=3 if alphapos == 2 else 1 #inset location and pad
		borderpad=4 if alphapos == 2 else 0
		inax = inset_axes(ax, width="30%", height="30%", borderpad=borderpad, loc=loc)
		sns.despine( ax=inax ) #take out spines
		inax.set_xlabel( r'$\hat{\alpha}_r$', size=plot_props['xylabel'], labelpad=0 )
		inax.set_ylabel( r'$P[ \hat{\alpha}_r ]$', size=plot_props['xylabel'], labelpad=2 )

		line_sims, labels_t = [], [] #initialise list of plot handles / labels

		for post in range(len( t_vals )): #loop through target times
			params['t'] = t_vals[ post ] #mean alter activity (max time in dynamics)
			print( '\tt = {}'.format( params['t'] ) )

			yplot = np.zeros(len(gamma_vals)) #initialise average MLE results
			gammahat_vals = np.zeros(len(sim_vals))
			for simpos in sim_vals: #loop through model realizations

				#load model of alter activity, according to parameters (only selected realization!)
				activity = mm.model_activity( params, loadflag='y', saveloc=saveloc_model )[ simpos ]
				#measure a0 and t (according to activity array!)
				a0 = min( activity ) #minimum alter activity
				t = np.mean( activity ) #mean alter activity

				#estimated alpha value
				bracket = ( -a0+alpha_bounds[0], alpha_bounds[1] ) #bounds for alpha search
				alphahat = mm.alpha_MLE_fit( activity, bracket )

				#accumulate MLE results
				gammahat_vals[simpos] = alphahat + a0
				yplot += np.array([ alpha_func( gamma-a0, activity, a0, t ) for gamma in gamma_vals ])
			yplot /= len(sim_vals)

			#plot plot!
			xplot = gamma_vals
			curve, = ax.semilogx( xplot, yplot, '-', c=colors[post], lw=plot_props['linewidth'], ms=plot_props['marker_size'], label=None, zorder=0 )

			#line at target relative alpha (to be recovered by MLE as a root)
			if post == 0:
				ax.axvline( x=params['gamma'], ls='--', c='0.5', lw=plot_props['linewidth'], zorder=1 )
				ax.axhline( y=0, ls='--', c='0.5', lw=plot_props['linewidth'], zorder=1 )

			#plot plot inset!
			sns.kdeplot( gammahat_vals, ax=inax, log_scale=[True, False], color=colors[post], lw=plot_props['linewidth'], zorder=0 )
			inax.axvline( x=params['gamma'], ls='--', c='0.5', lw=plot_props['linewidth']-1, zorder=1 )

			#legend stuff
			line_sims.append( curve ) #append handle
			if post == 0:
				labels_t.append( r'$t=$ {:.0f}'.format( params['t'] ) )
			else:
				labels_t.append( r'{:.0f}'.format( params['t'] ) )

		#text
		ax.text( params['gamma'], 2, r'$\alpha^*_r=$ {:.1f}'.format( params['gamma'] ), va='bottom', ha='center', fontsize=plot_props['xylabel'] )

		#legends

		if alphapos == 1:
			leg1 = ax.legend( line_sims, labels_t, loc='upper center', bbox_to_anchor=(0.5, -0.14), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=len(t_vals) )
			# ax.add_artist(leg1)

		#finalise subplot
		ax.axis([ gamma_vals[0], gamma_vals[-1], -2, 2 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		ax.locator_params( axis='x', numticks=4 )
		ax.locator_params( axis='y', nbins=4 )
		if alphapos > 0:
			ax.set_yticks([])

		#finalise inset
		if alphapos == 0:
			inax.set_xlim(1e-2, 1e0)
		elif alphapos == 1:
			inax.set_xlim(1e-1, 1e1)
		else:
			inax.set_xlim(1e0, 1e2)
		inax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )


#DEBUGGIN'

	#MLE equations
	# MLE_eq = lambda activity, gamma : float( mp.fsum([ mp.digamma( arel + gamma ) - mp.digamma( gamma ) for arel in activity - min(activity) ]) / len( activity ) )

			# #MLE equations
			# digamma_func = np.array([ MLE_eq( activity, gamma ) for gamma in gamma_vals ])
			# log_func = np.log( 1 + beta_vals )

				# #measure extra parameters/arrays
				# trel = t - a0 #relative mean alter activity
				# beta_vals = trel / gamma_vals #range of estimated scale factors

				# yplot += np.abs( np.array([ alpha_func( gamma-a0, activity, a0, t ) for gamma in gamma_vals ]) )

				# print( '\t\testimated alpha_r = {:.2f}'.format( gammahat ) )

			# gammahat = gamma_vals.mean()
			# labels_alpha.append( r'{:.2f}'.format( gammahat ) )

		# leg2 = plt.legend( line_sims, labels_alpha, title=r'$\hat{ \alpha }_r =$', loc='lower left', bbox_to_anchor=(0, 0), prop=plot_props['legend_prop'], handlelength=1.6, numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ] )
