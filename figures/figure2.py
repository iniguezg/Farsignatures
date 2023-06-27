#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE 2 IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
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
	'ticklabel' : 15,
	'text_size' : 10,
	'marker_size' : 2,
	'linewidth' : 2,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':10 },
	'legend_hlen' : 1,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables
	fig_props = { 'fig_num' : 2,
	'fig_size' : (10, 8),
	'aspect_ratio' : (2, 2),
	'grid_params' : dict( left=0.01, bottom=0.08, right=0.995, top=0.96, wspace=0.3, hspace=0.4 ),
	'dpi' : 300,
	'savename' : 'figure2' }


	## PLOTTING ##

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )


# A: Diagram of alter activity model

	#subplot variables
	k = 5 #ego degree
	a0, am = 1, 10 #min/max alter activity
	alpha_vals = [ -0.9, 1000. ] #PA parameter
	t_vals = [ 3, 1000 ] #mean alter activity (max time in dynamics)
	labels = [ 'cumulative advantage\n'+r'($ \alpha \to -a_0 $)', 'random choice\n'+r'($ \alpha \to \infty $)' ]

	seed=2 #rng seed
	nsizes = [350, 100 ] #node sizes (ego, alters)
	edgewids = [ 4, 2 ] #edge widths
	colors = sns.color_palette( 'GnBu', n_colors=2 ) #colors to plot

	#set up ego network
	graph = nx.generators.classic.star_graph( k ) #ego net as star graph
	graph = nx.relabel_nodes( graph, {0:'ego'} ) #rename ego
	positions = nx.spring_layout( graph, seed=seed ) #seed layout for reproducibility

	print('DIAGRAM')

	# A1: initial ego network

	#initialise subplot
	subgrid = grid[ 0,0 ].subgridspec( 2, 3, hspace=0.4, wspace=0.3 )
	ax = plt.subplot( subgrid[ :, 0 ] )
	ax.set_axis_off()

	plt.text( -0.06, 0.99, 'a', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	#plot plot network! ego, alters, labels
	nx.draw_networkx_nodes( graph, positions, nodelist=['ego'], node_size=nsizes[0], node_color=[colors[0]], margins=(0.07, 0.6) )
	nx.draw_networkx_nodes( graph, positions, nodelist=list(graph)[1:], node_size=nsizes[1], node_color=[colors[1]], margins=(0.07, 0.6) )
	nx.draw_networkx_labels( graph, positions, labels={'ego':'ego'} )

	#plot plot edges! edges, labels
	nx.draw_networkx_edges( graph, positions, width=edgewids[1], edge_color=colors[1] )
	nx.draw_networkx_edge_labels( graph, positions, edge_labels={ edge:'$a_0$' for edge in graph.edges }, rotate=False )

	#plot connection kernel
	eq_str = r'$\pi_a = \frac{ a + \alpha }{ \tau + k \alpha }$'
	plt.text( 0.5, 0.9, eq_str, va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

	#plot time arrow
	arrow_str = '$t = a_0$'
	bbox_props = dict( boxstyle="rarrow,pad=0.2", fc='None', ec='0.7', lw=1 )
	plt.text( 0.5, -0.1, arrow_str, ha='center', va='center', transform=ax.transAxes, size=plot_props['text_size'], bbox=bbox_props )
	plt.text( 0.5, -0.03, 'initial activity', ha='center', va='center', transform=ax.transAxes, size=plot_props['text_size'] )

	# A2: shapes for connection probability

	for grid_pos in range(2): #loop through regimes
		#initialise subplot
		ax = plt.subplot( subgrid[ grid_pos, 1 ] )
		sns.despine( ax=ax )
		if grid_pos == 1:
			plt.xlabel( r'$a$', size=plot_props['xylabel'] )
		plt.ylabel( r'$\pi_a$', size=plot_props['xylabel'], rotation='horizontal', labelpad=10 )

		#get parameters and plot arrays
		alpha = alpha_vals[grid_pos]
		tau = k * t_vals[0]
		xplot = np.arange( a0, am+1, dtype=int )
		yplot = ( xplot + alpha ) / ( tau + k*alpha )

		#plot plot!
		plt.plot( xplot, yplot, '-k', lw=plot_props['linewidth'] )

		#finalise subplot
		plt.axis([ a0, am, -0.1, 1 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		plt.xticks([])
		plt.yticks([])

		#regime labels
		plt.text( 0.5, 1.05, labels[grid_pos], va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )


	# A3: final ego network

	for grid_pos in range(2): #loop through regimes
		#initialise subplot
		ax = plt.subplot( subgrid[ grid_pos, 2 ] )
		ax.set_axis_off()

		#model parameters
		params = { 'alpha': alpha_vals[grid_pos], 'a0':a0, 'k' :k, 't':t_vals[1], 'ntimes':1 }
		#run model of alter activity
		activity = mm.model_activity( params, saveflag=False, seed=seed, print_every=1000 )

		#plot plot network! ego, alters, labels
		nx.draw_networkx_nodes( graph, positions, nodelist=['ego'], node_size=nsizes[0], node_color=[colors[0]], margins=(0.07, 0.1) )
		nx.draw_networkx_nodes( graph, positions, nodelist=list(graph)[1:], node_size=nsizes[1], node_color=[colors[1]], margins=(0.07, 0.1) )
		nx.draw_networkx_labels( graph, positions, labels={'ego':'ego'} )

		#plot plot edges! edges, labels
		width = 1 + np.array([ edgewids[grid_pos] * np.log2(activity[ 0, nodej-1 ]) / np.log2(activity.max()) for nodej in graph['ego'] ]) #log scaling!
		nx.draw_networkx_edges( graph, positions, width=width, edge_color=colors[1] )

	#plot time arrow
	arrow_str =r'$t = \tau / k$'
	bbox_props = dict( boxstyle="rarrow,pad=0.2", fc='None', ec='0.7', lw=1 )
	plt.text( 0.5, -0.23, arrow_str, ha='center', va='center', transform=ax.transAxes,
    size=plot_props['text_size'], bbox=bbox_props )
	plt.text( 0.5, -0.07, 'mean activity', ha='center', va='center', transform=ax.transAxes, size=plot_props['text_size'] )


# B: Parameter dependence of activity distribution

	#subplot variables
	k = 100 #number of alters (ego degree)
	a0 = 1 #minimum alter activity
	alpha_vals = [ -0.7, 9. ] #PA parameter
	t_vals = [ 2., 10., 100., 1000. ] #mean alter activity (max time in dynamics)
	ntimes = 10000 #number of realizations for averages
	a_vals = np.unique( np.logspace( np.log10(a0), 4, num=100, dtype=int ) ) #alter activities
	params = { 'k' : k, 'a0' : a0, 'ntimes' : ntimes } #initialise param dict

	labels = [ 'cumulative advantage', 'random choice' ]
	titley = [1.05, 0.9]
	colors = sns.color_palette( 'GnBu', n_colors=len(t_vals) ) #colors to plot

	print('ACTIVITY DIST')

	#initialise subplot
	subgrid = grid[ 0,1 ].subgridspec( 2, 1, hspace=0.2 )

	for grid_pos, alpha in enumerate( alpha_vals ): #loop through regimes
		params['alpha'] = alpha #PA parameter
		gamma = alpha + a0 #rescaled parameter

		#initialise subplot
		ax = plt.subplot( subgrid[grid_pos] )
		sns.despine( ax=ax ) #take out spines
		if grid_pos == 1:
			plt.xlabel( r'$a$', size=plot_props['xylabel'] )
		plt.ylabel( r'$p_a$', size=plot_props['xylabel'] )

		if grid_pos == 0:
			plt.text( -0.25, 0.97, 'b', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

		#plot regime alpha
		plt.text( 0.5, titley[grid_pos], labels[grid_pos], va='bottom', ha='center', transform=ax.transAxes, fontsize=plot_props['xylabel'] )

		for post, t in enumerate( t_vals ): #loop through times

			#SIMS plot!

			params['t'] = t #mean alter activity (max time in dynamics)
			#load model of alter activity, according to parameters
			activity = mm.model_activity( params, loadflag='y', saveloc=saveloc_model )

			#plot arrays for unbinned distribution
			xplot = np.arange( activity.min(), activity.max()+1, dtype=int )
			yplot, not_used = np.histogram( activity, bins=len(xplot), range=( xplot[0]-0.5, xplot[-1]+0.5 ), density=True )

			line_sims, = plt.loglog( xplot, yplot, 'o', label=None, c=colors[post], ms=plot_props['marker_size'], zorder=0 )

			#THEO plot!

			#plot arrays
			xplot = a_vals
			yplot_model = np.array([ mm.activity_dist( a, t, alpha, a0 ) for a in xplot ])
			label = '$t = $ {:.0f}'.format(t) if post == 0 else '{:.0f}'.format(t)

			line_theo, = plt.loglog( xplot, yplot_model, '-', label=label, c=colors[post], lw=plot_props['linewidth'], zorder=1 )

		#legends
		if grid_pos == 0:
			leg1 = plt.legend( loc='upper right', bbox_to_anchor=(1, 1), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=len(t_vals) )
			ax.add_artist(leg1)
			leg2 = plt.legend( (line_sims, line_theo), ('num', 'theo'), loc='upper right', bbox_to_anchor=(1, 0.8), prop=plot_props['legend_prop'], handlelength=1.6, numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ] )
			ax.add_artist(leg2)

		#finalise subplot
		plt.axis([ 8e-1, 4e4, 2e-7, 1e0 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		ax.locator_params( numticks=6 )
		if grid_pos == 0:
			plt.xticks([])


# C: Phase diagram of activity model

	#subplot variables
	gamma_vals = np.logspace(-1, 3.5, 50) #relative CA parameter
	trel_vals = np.logspace(-1, 3.5, 50) #relative mean alter activity

	#same values as in plot D, for comparison
	a0 = 1 #minimum alter activity
	alpha_vals = [ -0.7, 999. ] #PA parameter (Gamma regime, random regime)
	t_vals = [ 2., 10., 100., 1000. ] #mean alter activity (max time in dynamics)

	print('PHASE DIAGRAM')

	#initialise subplot
	subgrid = grid[ 1,0 ].subgridspec( 1, 2, wspace=0, width_ratios=[0.2, 1] )
	ax = plt.subplot( subgrid[1] )
	sns.despine( ax=ax )
	plt.xlabel( r'$\alpha_r$', size=plot_props['xylabel'] )
	plt.ylabel( r'$t_r$', size=plot_props['xylabel'] )

	plt.text( -0.27, 1.1, 'c', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

	#set up (theo) data
	gamma_mesh, trel_mesh = np.meshgrid( gamma_vals, trel_vals ) #relative PA parameter and mean alter activity
	beta_vals = trel_mesh / gamma_mesh #beta parameter
	disp_vals = beta_vals / ( 2 + beta_vals ) #activity dispersion (theo)

	#plot plot!
	norm = TwoSlopeNorm( 1/3., vmin=0, vmax=1 ) #colormap center at beta=1 (disp=1/3)
	plt.pcolormesh( gamma_mesh, trel_mesh, disp_vals, norm=norm, cmap='coolwarm', alpha=0.5, zorder=0 )

	plt.colorbar( label='dispersion $d$' ) #colorbar
	plt.plot( gamma_vals, gamma_vals, '--k', lw=plot_props['linewidth'], zorder=1 ) #identity line

	#parameter values for plot D
	for alpha in alpha_vals:
		xplot = ( alpha+a0 )*np.ones(len(t_vals))
		yplot = np.array(t_vals) - a0
		plt.plot( xplot, yplot, marker='o', ls='--', c='0.5', ms=6, lw=1, zorder=2 )

	#texts
	het_str = 'heterogeneous\nregime\n'+r'($\beta > 1$)'
	plt.text( 0.4, 0.85, het_str, va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['xylabel'] )
	hom_str = 'homogeneous\nregime\n'+r'($\beta < 1$)'
	plt.text( 0.6, 0.15, hom_str, va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['xylabel'] )
	beta_str = r'$\beta = \frac{t_r}{\alpha_r} = \frac{t - a_0}{\alpha + a_0}$'
	plt.text( 1, 1.1, beta_str, va='center', ha='right', transform=ax.transAxes, fontsize=plot_props['xylabel'] )

	#finalise subplot
	ax.set_xscale('log')
	ax.set_yscale('log')
	plt.axis([ gamma_vals[0], gamma_vals[-1], trel_vals[0], trel_vals[-1] ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
	ax.locator_params( numticks=6 )


# D: Scaling properties of activity regimes

	#subplot variables
	k = 100 #number of alters (ego degree)
	a0 = 1 #minimum alter activity
	alpha_vals = [ -0.7, 999. ] #PA parameter (Gamma regime, random regime)
	t_vals = [ 2., 10., 100., 1000. ] #mean alter activity (max time in dynamics)
	ntimes = 10000 #number of realizations for averages
	params = { 'k' : k, 'a0' : a0, 'ntimes' : ntimes } #initialise param dict

	xlabels = [ r'$a_r / \beta$', r'$(a_r - t_r) / \sqrt{ t_r }$' ]
	ylabels = [ r'$\beta p_a$', r'$\sqrt{ t_r } p_a$' ]
	reg_labels = [ 'heterogeneous\nregime', 'homogeneous\nregime' ]
	colors = sns.color_palette( 'GnBu', n_colors=len(t_vals) ) #colors to plot

	print('SCALING REGIMES')

	#initialise subplot
	subgrid = grid[ 1,1 ].subgridspec( 2, 1, hspace=0.5 )

	for alphapos, alpha in enumerate( alpha_vals ): #loop through alpha values (i.e. regimes)
		params['alpha'] = alpha #PA parameter
		gamma = alpha + a0 #gamma dist shape parameter

		#initialise subplot
		ax = plt.subplot( subgrid[alphapos] ) #upper row
		sns.despine( ax=ax ) #take out spines
		plt.xlabel( xlabels[alphapos], size=plot_props['xylabel'], labelpad=0 )
		plt.ylabel( ylabels[alphapos], size=plot_props['xylabel'] )

		if alphapos == 0:
			plt.text( -0.24, 1.22, 'd', va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['figlabel'], fontweight='bold' )

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

			labels.append( '$t = $ {:.0f}'.format(t) if post == 0 else '{:.0f}'.format(t) )

			if alphapos == 0:
				curve, = plt.loglog( xplot_resc, yplot_resc, '-', c=colors[post], lw=plot_props['linewidth'], zorder=0 )
			else:
				curve, = plt.semilogy( xplot_resc, yplot_resc, '-', c=colors[post], lw=plot_props['linewidth'], zorder=0 )
			lines_theo.append( curve ) #append handle


		#gamma approximation (w/ last t value only!)
		xplot = a_vals
		if alphapos == 0:
			xplot_resc = ( xplot - a0 ) / beta #rescaled activity ( a - a0 ) / beta
			#after scaling, gamma dist is standard form!
			yplot_model = np.array([ ss.gamma.pdf( db, gamma ) for db in xplot_resc ])
			line_gamma, = plt.loglog( xplot_resc, yplot_model, '--', c='k', lw=plot_props['linewidth']+1, zorder=1 )
		#Gaussian approximation (w/ last t value only!)
		else:
			xplot_resc = ( xplot - t ) / np.sqrt( t - a0 ) #rescaled activity
			#after scaling, Gaussian dist is standard form!
			yplot_model = np.array([ ss.norm.pdf( db ) for db in xplot_resc ])
			line_gauss, = plt.semilogy( xplot_resc, yplot_model, '--', c='k', lw=plot_props['linewidth']+1, zorder=1 )

		#regime labels
		plt.text( 1, 0.87, reg_labels[alphapos], va='center', ha='right', transform=ax.transAxes, fontsize=plot_props['ticklabel'] )

		#legends
		if alphapos == 0:
			leg1 = plt.legend( lines_theo, labels, loc='upper left', bbox_to_anchor=(0, 1.3), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=len(t_vals) )
			ax.add_artist(leg1)
		if alphapos == 0:
			leg2 = plt.legend( (line_sims, curve, line_gamma), ('num', 'theo', 'gamma'), loc='lower left', bbox_to_anchor=(0, 0.0), prop=plot_props['legend_prop'], handlelength=1, numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ] )
		else:
			leg2 = plt.legend( (line_sims, curve, line_gauss), ('num', 'theo', 'Gauss'), loc='upper left', bbox_to_anchor=(0, 1.08), prop=plot_props['legend_prop'], handlelength=1, numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ] )
		ax.add_artist(leg2)

		#finalise subplot
		if alphapos == 0:
			plt.axis([ 2e-4, 2e1, 5e-7, 2e2 ])
		else:
			plt.axis([ -9, 9, 5e-7, 2e0 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		if alphapos == 0:
			ax.xaxis.set_major_locator( plt.LogLocator(numticks=6) )
		else:
			ax.xaxis.set_major_locator( plt.MaxNLocator(5) )
		ax.yaxis.set_major_locator( plt.LogLocator(numticks=5) )

	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )


#DEBUGGIN'

		# width = [ edgewid * ( activity[ 0, nodej-1 ] - activity.min() ) / ( activity.max() - activity.min() ) for nodej in graph['ego'] ]

# import cairosvg as cs
# from PIL import Image
# from io import BytesIO
	# ax.set_axis_off()
	#
	# #plot plot!
	# img = cs.svg2png( url='diagrams/phase_diagram.svg' )
	# img = Image.open( BytesIO(img) )
	# plt.imshow(img)

	# norm = LogNorm( vmin=disp_vals.min(), vmax=disp_vals.max() )
