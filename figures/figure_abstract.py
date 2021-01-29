#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (NETWORKS ABSTRACT) IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from os.path import expanduser

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_misc as dm
import model_misc as mm
import plot_misc as pm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

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
	'legend_prop' : { 'size':10 },
	'legend_hlen' : 1,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables
	fig_props = { 'fig_num' : 1,
	'fig_size' : (12, 4),
	'aspect_ratio' : (2, 3),
	'grid_params' : dict( left=0.065, bottom=0.16, right=0.875, top=0.94, wspace=0.5, hspace=0.6 ),
	'dpi' : 300,
	'savename' : 'figure_abstract' }


	## PLOTTING ##

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )


# A-B: Scaling (with time t) of activity distribution in both regimes

	#parameters
	a0 = 1 #minimum alter activity
	k = 100 #number of alters (ego's degree)
	ntimes = 10000 #number of realizations for averages

	#parameter arrays
	alpha_vals = [ -0.7, 999. ] #PA parameter (Gamma regime, random regime)
	t_vals = [ 2., 10., 100., 1000. ] #mean alter activity (max time in dynamics)

	#parameter dict
	params = { 'a0' : a0, 'k' : k, 'ntimes' : ntimes }

	colors = sns.color_palette( 'GnBu', n_colors=len(t_vals) ) #colors to plot

	for alphapos, alpha in enumerate( alpha_vals ): #loop through alpha values (i.e. regimes)
		params['alpha'] = alpha #PA parameter
		gamma = alpha + a0 #gamma dist shape parameter

		print( 'alpha = '+str( alpha ) )

		#initialise subplot
		ax = plt.subplot( grid[ alphapos, 0 ] ) #upper row
		sns.despine( ax=ax ) #take out spines
		if alphapos == 0:
			plt.xlabel( r'$a_r / \beta$', size=plot_props['xylabel'] )
		else:
			plt.xlabel( r'$(a_r - t_r) / \sqrt{ t_r }$', size=plot_props['xylabel'] )
		if alphapos == 0:
			plt.ylabel( r'$\beta p_a$', size=plot_props['xylabel'] )
		else:
			plt.ylabel( r'$\sqrt{ t_r } p_a$', size=plot_props['xylabel'] )

		#plot plot!

#SIMS

		for post, t in enumerate( t_vals ): #loop through times
			params['t'] = t #mean alter activity (max time in dynamics)
			beta = ( t - a0 ) / ( alpha + a0 ) #gamma dist scale parameter

			#load model of alter activity, according to parameters
			activity = mm.model_activity( params, loadflag='y', saveloc=saveloc_model )

			a_vals = np.arange( activity.min(), activity.max()+1, dtype=int )

			#plot arrays for unbinned distribution
			xplot = a_vals
			yplot, not_used = np.histogram( activity, bins=len(a_vals), range=( a_vals[0]-0.5, a_vals[-1]+0.5 ), density=True )

			#rescale plot arrays
			if alphapos == 0:
				xplot_resc = ( xplot - a0 ) / beta #rescaled activity ( a - a0 ) / beta
				yplot_resc = beta * yplot #rescaled probability beta * p_d(t)
			else:
				xplot_resc = ( xplot - t ) / np.sqrt( t - a0 ) #rescaled activity
				yplot_resc = np.sqrt( t - a0 ) * yplot #rescaled probability

			if alphapos == 0:
				line_sims, = plt.loglog( xplot_resc, yplot_resc, 'o', c=colors[post], ms=plot_props['marker_size'], zorder=0 )
			else:
				line_sims, = plt.semilogy( xplot_resc, yplot_resc, 'o', c=colors[post], ms=plot_props['marker_size'], zorder=0 )

#THEO

		if alphapos == 0:
			a_vals = np.unique( np.logspace( np.log10(a0+1), 5, num=100, dtype=int ) ) #alter activities
		else:
			a_vals = np.concatenate(( np.linspace( 1, 9, num=9 ), np.unique( np.logspace( 1, 5, num=1000, dtype=int ) ) )) #alter activities

		lines_theo = [] #initialise list of plot handles / labels
		labels = []

		for post, t in enumerate( t_vals ): #loop through times
			beta = ( t - a0 ) / ( alpha + a0 ) #gamma dist scale parameter

			#PGF expression

			xplot = a_vals
			yplot_model = np.array([ mm.activity_dist( a, t, alpha, a0 ) for a in xplot ])

			#rescale plot arrays
			if alphapos == 0:
				xplot_resc = ( xplot - a0 ) / beta #rescaled activity ( a - a0 ) / beta
				yplot_resc = beta * yplot_model #rescaled probability beta * p_d(t)
			else:
				xplot_resc = ( xplot - t ) / np.sqrt( t - a0 ) #rescaled activity
				yplot_resc = np.sqrt( t - a0 ) * yplot_model #rescaled probability

			labels.append( '{:.0f}'.format(t) )

			if alphapos == 0:
				curve, = plt.loglog( xplot_resc, yplot_resc, '-', c=colors[post], lw=plot_props['linewidth'], zorder=0 )
			else:
				curve, = plt.semilogy( xplot_resc, yplot_resc, '-', c=colors[post], lw=plot_props['linewidth'], zorder=0 )
			lines_theo.append( curve ) #append handle


		#gamma approximation (w/ last t value only!)
		if alphapos == 0:
			xplot = a_vals
			xplot_resc = ( xplot - a0 ) / beta #rescaled activity ( a - a0 ) / beta
			#after scaling, gamma dist is standard form!
			yplot_model = np.array([ ss.gamma.pdf( db, gamma ) for db in xplot_resc ])
			line_gamma, = plt.loglog( xplot_resc, yplot_model, '--', c='k', lw=plot_props['linewidth']+1, zorder=1 )

		#Gaussian approximation (w/ last t value only!)
		else:
			xplot = a_vals
			xplot_resc = ( xplot - t ) / np.sqrt( t - a0 ) #rescaled activity
			#after scaling, Gaussian dist is standard form!
			yplot_model = np.array([ ss.norm.pdf( db ) for db in xplot_resc ])
			line_gauss, = plt.semilogy( xplot_resc, yplot_model, '--', c='k', lw=plot_props['linewidth']+1, zorder=1 )

		#texts

		if alphapos == 0:
			plt.text( 1.1, 1, 'heterogeneous\nregime', va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'], weight='bold' )
			plt.text( -0.3, 1.1, r'$\alpha_r=$ {:.1f}'.format(gamma), va='center', ha='left', transform=ax.transAxes, fontsize=plot_props['xylabel'] )

		else:
			plt.text( 1.1, -0.3, 'homogeneous\nregime', va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'], weight='bold' )
			plt.text( -0.3, -0.4, r'$\alpha_r=$ {:.0f}'.format(gamma), va='center', ha='left', transform=ax.transAxes, fontsize=plot_props['xylabel'] )

		#legends

		if alphapos == 1:
			leg1 = plt.legend( lines_theo, labels, title='$t=$', loc='upper left', bbox_to_anchor=(0.65, 1.35), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=2 )
			ax.add_artist(leg1)

		if alphapos == 0:
			leg2 = plt.legend( (line_sims, curve, line_gamma), ('num', 'theo', 'gamma'), loc='lower left', bbox_to_anchor=(0, 0), prop=plot_props['legend_prop'], handlelength=1, numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ] )
		else:
			leg2 = plt.legend( (line_sims, curve, line_gauss), ('num', 'theo', 'Gauss'), loc='lower center', bbox_to_anchor=(0.5, 0), prop=plot_props['legend_prop'], handlelength=1, numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ] )
		ax.add_artist(leg2)

		#finalise subplot
		if alphapos == 0:
			plt.axis([ 2e-4, 2e1, 5e-7, 2e2 ])
		else:
			plt.axis([ -9, 9, 5e-7, 2e0 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		if alphapos == 0:
			ax.locator_params( numticks=6 )


# C: Phase diagram (gamma, trel) for all ego nets in given dataset

	propx = ('gamma', r'\alpha_r')
	propy = ('act_avg_rel', 't_r')
	dataname, eventname, textname = ('MPC_UEu_net', 'MPC_UEu.evt', 'Mobile (call)')

	print( 'dataset name: ' + eventname[:-4] ) #print output

	alphamax = 1000 #maximum alpha for MLE fit
	nsims = 1000 #number of syntethic datasets used to calculate p-value
	amax = 10000 #maximum activity for theoretical activity distribution

	pval_thres = 0.1 #threshold above which alphas are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf

	#plotting variables
	gridsize = 40 #grid size for hex bins
	vmax = 1e4 #max value in colorbar (larger than [filtered] N in any dataset!)


	## DATA ##

	#prepare ego network properties
	egonet_props, egonet_acts = dm.egonet_props_acts( dataname, eventname, root_data, 'y', saveloc_data )

	#fit activity model to all ego networks in dataset
	egonet_fits = dm.egonet_fits( dataname, eventname, root_data, 'y', saveloc_data, alphamax=alphamax, nsims=nsims, amax=amax )

	#filter egos according to fitting results
	egonet_filter, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )

	#add relative quantities
	t_rels = pd.Series( egonet_filter.act_avg - egonet_filter.act_min, name='act_avg_rel' )
	egonet_filter = pd.concat( [ egonet_filter, t_rels ], axis=1 )


	## PLOTTING ##

	#initialise subplot
	ax = plt.subplot( grid[ :, 1 ] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( '$'+propx[1]+'$', size=plot_props['xylabel'] )
	plt.ylabel( '$'+propy[1]+'$', size=plot_props['xylabel'] )

	#plot plot!
	hexbin = plt.hexbin( propx[0], propy[0], data=egonet_filter, xscale='log', yscale='log', norm=LogNorm(vmin=1e0, vmax=vmax), mincnt=1, gridsize=gridsize, cmap='GnBu', zorder=0 )

	#colorbar
	cbar = plt.colorbar( hexbin, ax=ax, fraction=0.05 )
	cbar.ax.set_title( r'$N_{'+propx[1][:-2]+','+propy[1][:-2]+'}$' )
	cbar.ax.minorticks_off()

	#lines
	plt.plot( [1e-2, 1e4], [1e-2, 1e4], '--', c='0.6', lw=plot_props['linewidth'], zorder=1 )

	#texts
	plt.text( 0.5, 1, textname, va='bottom', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )
	plt.text( 0.7, 0.6, r'crossover ($\beta = 1$)', va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'], rotation=60 )

	#arrows
	plt.annotate( text='', xy=( 0.2, 0.5 ), xytext=( -0.45, 0.9 ), arrowprops=dict( headlength=12, headwidth=10, width=5, color=colors[-1], alpha=0.5 ), xycoords=ax.transAxes, textcoords=ax.transAxes )
	plt.annotate( text='', xy=( 0.5, 0.1 ), xytext=( -0.45, 0 ), arrowprops=dict( headlength=12, headwidth=10, width=5, color=colors[-1], alpha=0.5 ), xycoords=ax.transAxes, textcoords=ax.transAxes )

	#finalise subplot
	plt.axis([ 3e-2, 2e2, 3e-1, 5e2 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
	ax.locator_params( numticks=5 )


# D: CCDF of estimated alpha for all datasets

	prop_name = 'beta'
	prop_label = r'1 / \beta'

	alphamax = 1000 #maximum alpha for MLE fit
	nsims = 1000 #number of syntethic datasets used to calculate p-value
	amax = 10000 #maximum activity for theoretical activity distribution

	pval_thres = 0.1 #threshold above which alpha MLEs are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf

	#dataset list: dataname, eventname, textname
	datasets = [ ('MPC_UEu_net', 'MPC_UEu.evt', 'Mobile (call)'),
				 ('SMS_net', 'MPC_Wu_SD01.evt', 'Mobile (Wu 1)'),
				 ('SMS_net', 'MPC_Wu_SD02.evt', 'Mobile (Wu 2)'),
				 ('SMS_net', 'MPC_Wu_SD03.evt', 'Mobile (Wu 3)'),
				 ('sex_contacts_net', 'sexcontact_events.evt', 'Contact'),
				 ('greedy_walk_nets', 'email.evt', 'Email 1'),
				 ('greedy_walk_nets', 'eml2.evt', 'Email 2'),
				 ('greedy_walk_nets', 'fb.evt', 'Facebook'),
				 ('greedy_walk_nets', 'messages.evt', 'Messages'),
				 ('greedy_walk_nets', 'forum.evt', 'Forum'),
				 ('greedy_walk_nets', 'pok.evt', 'Dating'),
				 ('Copenhagen_nets', 'CNS_bt_symmetric.evt', 'CNS (bluetooth)'),
				 ('Copenhagen_nets', 'CNS_calls.evt', 'CNS (call)'),
				 ('Copenhagen_nets', 'CNS_sms.evt', 'CNS (sms)') ]

	colors = sns.color_palette( 'Set2', n_colors=len(datasets) ) #colors to plot

	#initialise subplot
	ax = plt.subplot( grid[ :, 2 ] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( '${}$'.format( prop_label ), size=plot_props['xylabel'] )
	plt.ylabel( 'CCDF', size=plot_props['xylabel'] )

	#loop through considered datasets
	for dset_pos, (dataname, eventname, textname) in enumerate(datasets):
		print( 'dataset name: ' + eventname[:-4] ) #print output

		## DATA ##

		#prepare ego network properties
		egonet_props, egonet_acts = dm.egonet_props_acts( dataname, eventname, root_data, 'y', saveloc_data )

		#fit activity model to all ego networks in dataset
		egonet_fits = dm.egonet_fits( dataname, eventname, root_data, 'y', saveloc_data, alphamax=alphamax, nsims=nsims, amax=amax )

		#filter egos according to fitting results
		egonet_filter, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )

		## PLOTTING ##

		#prepare data
		yplot_data = 1 / egonet_filter[ prop_name ] #filtered data!
		xplot, yplot = pm.plot_CCDF_cont( yplot_data ) #complementary cumulative dist

		#plot plot!
		plt.loglog( xplot, yplot, '-', label=textname, c=colors[dset_pos], lw=plot_props['linewidth'] )

	#lines
	plt.axvline( x=1, ls='--', c='0.6', lw=plot_props['linewidth'] )

	#texts
	plt.text( 0.25, 1, 'heterogeneous\nregime', va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'], weight='bold' )
	plt.text( 0.85, 1, 'homogeneous\nregime', va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'], weight='bold' )

	#legend
	plt.legend( loc='lower left', bbox_to_anchor=(1, 0), prop=plot_props['legend_prop'], handlelength=1, numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ] )

	#finalise subplot
	plt.axis([ 1e-5, 1e4, 3e-5, 5e0 ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
	ax.locator_params( numticks=6 )


	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )


#DEBUGGIN'

				# print(np.log10(xplot_resc.min()))
