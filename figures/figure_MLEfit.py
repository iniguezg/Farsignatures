#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (MLE FITTING FOR MODEL) IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
import numpy as np
import seaborn as sns
import matplotlib as mpl
import scipy.special as ss
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpmath import mp
from os.path import expanduser

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
	alphamax = 1000 #maximum alpha for MLE fit

	#plot arrays
	simpos = 0 #model realization to analyze
	gamma_vals = np.logspace( -2, 2, num=100 ) #range of estimated relative alpha to plot

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
	'grid_params' : dict( left=0.065, bottom=0.2, right=0.985, top=0.95, wspace=0.1 ),
	'dpi' : 300,
	'savename' : 'figure_MLEfit' }

	colors = sns.color_palette( 'Set2', n_colors=len(t_vals) ) #colors to plot


	## MLE ##

	#MLE equations
	MLE_eq = lambda activity, gamma : float( mp.fsum([ mp.digamma( arel + gamma ) - mp.digamma( gamma ) for arel in activity - min(activity) ]) / len( activity ) )


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
		plt.xlabel( r'$\alpha_r$', size=plot_props['xylabel'] )
		if alphapos == 0:
			plt.ylabel( r'$ | f_{ \alpha } - \ln ( 1 + \beta ) | $', size=plot_props['xylabel'] )

		line_sims = [] #initialise list of plot handles / labels
		labels_t, labels_alpha = [], []

		for post in range(len( t_vals )): #loop through target times
			params['t'] = t_vals[ post ] #mean alter activity (max time in dynamics)
			print( '\tt = {}'.format( params['t'] ) )

			#load model of alter activity, according to parameters (only selected realization!)
			activity = mm.model_activity( params, loadflag='y', saveloc=saveloc_model )[ simpos ]

			#measure a0 and t (according to activity array!)
			a0 = min( activity ) #minimum alter activity
			t = np.mean( activity ) #mean alter activity

			#estimated alpha value
			bounds = ( -a0, alphamax ) #bounds for alpha search
			alphahat = mm.alpha_MLE_fit( activity, bounds )
			gammahat = alphahat + a0
			print( '\t\testimated alpha_r = {:.2f}'.format( gammahat ) )
#			print( '\t\t{}'.format( mm.gamma_KSstat( activity ) ) )

			#measure extra parameters/arrays
			trel = t - a0 #relative mean alter activity
			beta_vals = trel / gamma_vals #range of estimated scale factors

			#MLE equations
			digamma_func = np.array([ MLE_eq( activity, gamma ) for gamma in gamma_vals ])
			log_func = np.log( 1 + beta_vals )

			#plot arrays
			xplot = gamma_vals
			yplot = np.abs( digamma_func - log_func )

			#digamma function
			curve, = plt.loglog( xplot, yplot, 'o', c=colors[post], lw=plot_props['linewidth'], ms=plot_props['marker_size'], label=None, zorder=1 )

			#line at target relative alpha (to be recovered by MLE)
			if post == 0:
				plt.axvline( x=params['gamma'], ls='--', c='0.5', lw=plot_props['linewidth'], label=None, zorder=0 )

			#legend stuff
			line_sims.append( curve ) #append handle
			if post == 0:
				labels_t.append( r'$t=$ {:.0f}'.format( params['t'] ) )
			else:
				labels_t.append( r'{:.0f}'.format( params['t'] ) )
			labels_alpha.append( r'{:.2f}'.format( gammahat ) )

		#text
		plt.text( 0.5, 1.05, r'$\alpha^*_r=$ {:.1f}'.format( params['gamma'] ), va='top', ha='center', transform=ax.transAxes, fontsize=plot_props['xylabel'] )

		#legends

		if alphapos == 1:
			leg1 = plt.legend( line_sims, labels_t, loc='upper center', bbox_to_anchor=(0.5, -0.15), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=len(t_vals) )
			ax.add_artist(leg1)

		leg2 = plt.legend( line_sims, labels_alpha, title=r'$\hat{ \alpha }_r =$', loc='lower left', bbox_to_anchor=(0, 0), prop=plot_props['legend_prop'], handlelength=1.6, numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ] )

		#finalise subplot
		plt.axis([ 1e-2, 1e2, 1e-6, 1e2 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		ax.locator_params( numticks=6 )
		if alphapos > 0:
			plt.yticks([])


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )


#DEBUGGIN'

	# ntimes = 10000 #number of realizations for averages
	# #parameter dict
	# params = { 'a0' : a0, 'k' : k, 'ntimes' : ntimes }

	# alphamax = 100000 #maximum alpha for MLE fit
	# bounds = ( -a0, alphamax ) #bounds for alpha search

			# params['alpha'] = alpha #PA parameter
			# params['t'] = t #mean alter activity (max time in dynamics)

			# #load model of alter activity, according to parameters (1 realization only!)
			# activity = mm.model_activity( params, loadflag='y', saveloc=saveloc_model )[0]

			# #find MLE optimal alpha for given activity
			# alphahat = mm.alpha_MLE_fit( activity, bounds )
			# print( '\t{}, {}'.format( alpha, alphahat ) )

			# #plot arrays
			# xplot = betahat_vals #range of estimated beta
			# yplot_MLE = MLE_eq #MLE equation

#	beta_vals = [ 100., 1., 0.01 ] #crossover parameter
#	beta_vals = [ 100. ]
#	trel_vals = [ 0.1, 1., 10., 100., 1000. ] #relative mean alter activity
#	trel_vals = [ 10. ] #relative mean alter activity

#	amax = 10000 #maximum activity for theoretical activity distribution
#	betahat_vals = np.logspace( -3, 3, num=100 ) #range of estimated beta to plot

#	rng = np.random.default_rng() #initialise random number generator
#
#	# MLE_eq = lambda activity, gammahat : float( mp.exp( mp.fsum([ mp.digamma( arel + gammahat ) - mp.digamma( gammahat ) for arel in activity - a0 ]) / len( activity ) ) ) - 1

#	for betapos, beta in enumerate( beta_vals ): #loop through beta values
#		print( 'beta = '+str( beta ) )

#		for post, trel in enumerate( trel_vals ): #loop through relative times

			# t = trel + a0 #get mean alter activity
			# gamma = trel / beta #gamma parameter (relative alpha)
			# alpha = gamma - a0 #get alpha
			# gammahat_vals = trel / betahat_vals #range of estimated gammas

			# params['alpha'] = alpha #PA parameter
			# params['t'] = t #mean alter activity (max time in dynamics)

			# #theo activity dist in range a=[0, amax] (i.e. inclusive)
			# act_dist_theo = np.array([ mm.activity_dist( a, t, alpha, a0 ) for a in range(amax+1) ])
			# act_dist_theo = act_dist_theo / act_dist_theo.sum() #normalise if needed
			# activity = rng.choice( amax+1, k, p=act_dist_theo ) #simulations of alter activity from data fit

			# act_tile = np.tile( activity, ( len(gammahat_vals), 1 ) )
			# gamma_tile = np.tile( gammahat_vals, ( len(activity), 1 ) ).transpose()
			# digamma_func = ( ss.digamma( act_tile + gamma_tile ) - ss.digamma( gamma_tile ) ).mean( axis=1 )

			# #estimated beta value
			# betahat = betahat_vals[ np.abs( digamma_func - log_func ).argmin() ]
			# print( 'estimated beta = {}'.format( betahat ) )

#			gammahat = alphahat + a0 #rescaled parameter

			# #log function
			# if post == 0:
			# 	line_log, = plt.loglog( betahat_vals, log_func, '--', c='0.5', lw=plot_props['linewidth'], label=None, zorder=0 )

		# if betapos == 0:
		# 	reg_str = 'heterogeneous regime\n'
		# if betapos == 1:
		# 	reg_str = 'crossover\n'
		# if betapos == 2:
		# 	reg_str = 'homogeneous regime\n'
		# plt.text( 0.5, 1, reg_str+r'($\beta=$ {}'.format(beta)+')', va='top', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

	# #parameter dict
	# params = { 'a0' : a0, 'k' : k, 'ntimes' : ntimes }
