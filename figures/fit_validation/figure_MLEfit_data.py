#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (MLE FITTING FOR DATA) IN FARSIGNATURES PROJECT ###

#import modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpmath import mp
from os.path import expanduser

import data_misc as dm
import model_misc as mm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#dataset
	dataname = 'SMS_net'
	eventname = 'MPC_Wu_SD03_htnet.evt'
	bounds = (0, 1000) #bounds for alpha MLE fit

	#flags and locations
	loadflag = 'y'
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files

	#plotting variables
#	alpha_min, alpha_max = 2., 5. #alpha interval where to choose ego
	num = 30 #number of log bins to plot distributions
	alpha_vals = [ mp.mpf(str( alpha )) for alpha in np.logspace( -1, 3, num=100 ) ] #alpha values for plotting graphical solution

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
	'savename' : 'figure_MLEfit_data_' + eventname[:-10] }

	colors = sns.color_palette( 'Set2', n_colors=1 ) #colors to plot


	## DATA ##

	#prepare ego network properties
	egonet_props = dm.egonet_props( dataname, eventname, bounds, root_data, loadflag, saveloc )
	degrees, actmeans, alphas = egonet_props['degrees'], egonet_props['actmeans'], egonet_props['alphas']

	#get alter activity per ego
	names = ['nodei', 'nodej', 'tstamp'] #column names
	filename = root_data + dataname + '/data_formatted/' + eventname
	events = pd.read_csv( filename, sep=';', header=None, names=names )
	events_rev = events.rename( columns={ 'nodei':'nodej', 'nodej':'nodei' } )[ names ]
	events_concat = pd.concat([ events, events_rev ])
	ego_acts = events_concat.groupby(['nodei', 'nodej']).size()

	#select arbitrary ego (with given condition) and get properties
#	nodei = alphas[ alphas.between( alpha_min, alpha_max ) ].index[0]
	nodei = ( degrees ).sort_values( ascending=False ).index[3]
#	nodei = ( ego_acts.groupby('nodei').std() ).sort_values( ascending=False ).index[1]
	activity = ego_acts[ nodei ] #alters activity
	alpha_hat = alphas[ nodei ] #optimal alpha
	t = np.mean(activity) #mean alter activity


	## FITTING ##

	#solving alpha trascendental equation
	digamma_avg = lambda alpha, activity : mp.fsum([ mp.digamma( alpha + a ) - mp.digamma( alpha ) for a in activity ]) / len( activity )
	alpha_func = lambda alpha, t, activity : t / ( mp.exp( digamma_avg( alpha, activity ) ) - 1 )


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

#	bins = np.concatenate(( np.linspace( 1, 9, num=9 ), np.logspace( 1, np.log10( activity.max() ), num=num ) ))
	bins = np.linspace( 1, activity.max(), num=activity.max() )
	yplot, bin_edges = np.histogram( activity, bins=bins, density=True )
	xplot = [ ( bin_edges[i+1] + bin_edges[i] ) / 2 for i in range(len( bin_edges[:-1] )) ]

	label = 'Data'
	plt.loglog( xplot, yplot, 'o', label=label, c=colors[0], ms=plot_props['marker_size'], zorder=0 )

	#plot theo distribution (with MLE optimal alpha)

	#get activity range (for theo distribution)
	a_vals = [ mp.mpf(str( a )) for a in range( min(activity), max(activity)+1 ) ]

	xplot = [ float(a) for a in a_vals ]
	yplot = [ float( mm.activity_dist_fixed_t( a, t, alpha_hat ) ) for a in a_vals ]

	label = r'MLE fit ($\hat{\alpha} =$ '+'{:.2f})'.format( float(alpha_hat) )
	plt.loglog( xplot, yplot, '-', label=label, c=colors[0], lw=plot_props['linewidth'], zorder=0 )

	#legend
	leg = plt.legend( loc='lower left', bbox_to_anchor=(0, 0), prop=plot_props['legend_prop'], handlelength=plot_props['legend_hlen'], numpoints=plot_props['legend_np'], columnspacing=plot_props[ 'legend_colsp' ], ncol=1 )

	#finalise subplot
#	plt.axis([ 1e0, 1e4, 1e-4, 1e0 ])
	ax.tick_params( axis='both', which='major', labelsize=plot_props['ticklabel'], pad=2 )


# B : Graphical solution of trascendental equation for optimal alpha

	#initialise subplot
	ax = plt.subplot( grid[1] )
	sns.despine( ax=ax ) #take out spines
	plt.xlabel( r'parameter $\alpha$', size=plot_props['xylabel'] )
	plt.ylabel( r'parameter $\alpha$', size=plot_props['xylabel'] )

	#plot plot!

	yplot = [ alpha_func( alpha, t, activity ) for alpha in alpha_vals ]
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
