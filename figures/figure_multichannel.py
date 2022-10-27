#! /usr/bin/env python

### SCRIPT FOR PLOTTING MULTICHANNEL FIGURE IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from os.path import expanduser

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_misc as dm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#alpha fit variables
	alphamax = 1000 #maximum alpha for MLE fit
	pval_thres = 0.1 #threshold above which alpha MLEs are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf

	#plot variables
	gridsize = 41 #grid size for hex bin
	vmax = 1e5 #max value in colorbar
	range_call = [1e-3, 1e5] #ranges for data
	range_text = [1e-3, 1e5]

	#locations
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files

	#sizes/widths/coords
	plot_props = { 'xylabel' : 15,
	'figlabel' : 26,
	'ticklabel' : 15,
	'text_size' : 10,
	'marker_size' : 8,
	'linewidth' : 2,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':10 },
	'legend_hlen' : 1,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables
	fig_props = { 'fig_num' : 1,
	'fig_size' : (8, 6),
	'aspect_ratio' : (1, 1),
	'grid_params' : dict( left=0.1, bottom=0.11, right=0.96, top=0.96 ),
	'dpi' : 300,
	'savename' : 'figure_multichannel' }


	## DATA ##

	#load ego network properties for call/text datasets
	egonet_props_call = pd.read_pickle( saveloc + 'egonet_props_call.pkl' )
	egonet_props_text = pd.read_pickle( saveloc + 'egonet_props_text.pkl' )

	#fit activity model to call/text datasets
	egonet_fits_call = pd.read_pickle( saveloc + 'egonet_fits_call.pkl' )
	egonet_fits_text = pd.read_pickle( saveloc + 'egonet_fits_text.pkl' )

	#filter egos according to fitting results
	egonet_filter_call, egonet_inf_call, egonet_null_call = dm.egonet_filter( egonet_props_call, egonet_fits_call, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )
	egonet_filter_text, egonet_inf_text, egonet_null_text = dm.egonet_filter( egonet_props_text, egonet_fits_text, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )


	## PLOTTING ##

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )

	ax = plt.subplot( grid[0] )
	sns.despine( ax=ax )
	plt.xlabel( r'$\beta_{\mathrm{call}}$', size=plot_props['xylabel'] )
	plt.ylabel( r'$\beta_{\mathrm{text}}$', size=plot_props['xylabel'] )

	#get plot data
	plot_data = pd.concat( [ egonet_filter_call.beta.rename('beta_call'), egonet_filter_text.beta.rename('beta_text') ], axis=1, join='inner' )

	#plot plot!
	hexbin = plt.hexbin( 'beta_call', 'beta_text', data=plot_data, xscale='log', yscale='log', norm=LogNorm(vmin=1e0, vmax=vmax), mincnt=1, gridsize=gridsize, cmap='GnBu', zorder=0 )

	#colorbar
	cbar = plt.colorbar( hexbin, ax=ax, fraction=0.05 )
	cbar.ax.set_title( r'$N_{c,t}$' )
	cbar.ax.minorticks_off()

	#lines
	plt.plot( range_call, range_text, '--', c='0.6', lw=plot_props['linewidth'], zorder=1 )
	plt.plot( [1, 1], range_text, '--', c='0.6', lw=plot_props['linewidth'], zorder=1 )
	plt.plot( range_call, [1, 1], '--', c='0.6', lw=plot_props['linewidth'], zorder=1 )

	#texts
	plt.text( 0.34, 1, r'crossover ($\beta_{\mathrm{call}} = 1$)', va='top', ha='left', transform=ax.transAxes, fontsize=plot_props['text_size'], rotation=90 )
	plt.text( 1, 0.34, r'crossover ($\beta_{\mathrm{text}} = 1$)', va='top', ha='right', transform=ax.transAxes, fontsize=plot_props['text_size'] )
	plt.text( 0.9, 0.95, r'$\beta_{\mathrm{call}} = \beta_{\mathrm{text}}$', va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'], rotation=45 )
	plt.text( 0.95, 0.8, 'heterogeneous\n'+r'($\beta_i > 1$)', va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'] )
	plt.text( 0.25, 0.1, 'homogeneous\n'+r'($\beta_i < 1$)', va='center', ha='center', transform=ax.transAxes, fontsize=plot_props['text_size'] )

	#finalise subplot
	plt.axis([ *range_call, *range_text ])
	ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
	ax.locator_params( numticks=5 )

	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
