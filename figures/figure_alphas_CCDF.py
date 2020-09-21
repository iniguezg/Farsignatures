#! /usr/bin/env python

### SCRIPT FOR PLOTTING FIGURE (ALPHAS CCDF) IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_misc as dm
import plot_misc as pm


## RUNNING FIGURE SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#properties to plot
	prop_names = [ 'alpha' ]
	prop_labels = [ r'$\alpha$' ]

	bounds = (0, 1000) #bounds for alpha MLE fit
	nsims = 100 #number of syntethic datasets used to calculate p-value
	amax = 10000 #maximum activity for theoretical activity distribution

	pval_thres = 0.05 #threshold above which alphas are considered
	alpha_min, alpha_max = 1e-4, 1e2 #extreme values for alpha (to avoid num errors)

	#locations
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files

	#dataset list: dataname, eventname, textname
	datasets = [ ('greedy_walk_nets', 'eml2.evt', 'Email 2'),
				 ('Copenhagen_nets', 'CNS_calls.evt', 'CNS (call)'),
				 ('Copenhagen_nets', 'CNS_sms.evt', 'CNS (sms)') ]

	#sizes/widths/coords
	plot_props = { 'xylabel' : 15,
	'figlabel' : 26,
	'ticklabel' : 15,
	'text_size' : 15,
	'marker_size' : 8,
	'linewidth' : 1.5,
	'tickwidth' : 1,
	'barwidth' : 0.8,
	'legend_prop' : { 'size':15 },
	'legend_hlen' : 1,
	'legend_np' : 1,
	'legend_colsp' : 1.1 }

	#plot variables
	fig_props = { 'fig_num' : 1,
	'fig_size' : (10, 4),
	'aspect_ratio' : (1, 3),
	'grid_params' : dict( left=0.08, bottom=0.16, right=0.98, top=0.91, wspace=0.2, hspace=0.4 ),
	'dpi' : 300,
	'savename' : 'figure_alphas_CCDF' }

	colors = sns.color_palette( 'Set2', n_colors=len(prop_names) ) #colors to plot

	#initialise plot
	sns.set( style='ticks' ) #set fancy fancy plot
	fig = plt.figure( fig_props['fig_num'], figsize=fig_props['fig_size'] )
	plt.clf()
	grid = gridspec.GridSpec( *fig_props['aspect_ratio'] )
	grid.update( **fig_props['grid_params'] )

	#loop through considered datasets
	for grid_pos, (dataname, eventname, textname) in enumerate(datasets):
		print( 'dataset name: ' + eventname[:-4] ) #print output

		## DATA ##

		#prepare ego network properties
		egonet_props, egonet_acts = dm.egonet_props_acts( dataname, eventname, root_data, 'y', saveloc )
		degrees, num_events, amins = egonet_props['degree'], egonet_props['strength'], egonet_props['act_min'] #unpack props

		#fit activity model to all ego networks in dataset
		egonet_fits = dm.egonet_fits( dataname, eventname, root_data, 'y', saveloc, bounds=bounds, nsims=nsims, amax=amax )

		#filtering process
		#step 1: egos with t > a_0
		egonet_fits_filter = egonet_fits[ degrees * amins < num_events ]
		#step 2: egos with pvalue > threshold
		egonet_fits_filter = egonet_fits_filter[ egonet_fits_filter.pvalue > pval_thres ]
		#step 3: alphas between extreme values (numerical errors)
		egonet_fits_filter = egonet_fits_filter[ egonet_fits_filter.alpha.between( alpha_min, alpha_max ) ]

		#some measures
		num_egos_filter = len( egonet_fits_filter ) #filtered egos
		frac_egos_random = ( egonet_fits_filter.alpha > 1 ).sum() / float( num_egos_filter ) #fraction of egos in random regime (alpha > 1)

		## PLOTTING ##

		#initialise subplot
		ax = plt.subplot( grid[ grid_pos] )
		sns.despine( ax=ax ) #take out spines
		plt.xlabel( r'parameter $\alpha$', size=plot_props['xylabel'] )
		if grid_pos in [0]:
			plt.ylabel( r"CCDF $P[ \alpha' \geq \alpha ]$", size=plot_props['xylabel'] )

		#loop through considered properties
		for prop_pos, (prop_name, prop_label) in enumerate(zip( prop_names, prop_labels )):

			#prepare data
			yplot_data = egonet_fits_filter[ prop_name ] #filtered data!
			xplot, yplot = pm.plot_CCDF_cont( yplot_data ) #complementary cumulative dist

			#plot plot!
			plt.loglog( xplot, yplot, 'o', label=prop_label, mec=colors[prop_pos], mfc='w', mew=plot_props['linewidth'], ms=plot_props['marker_size'] )

		#lines
		plt.vlines( x=1, ymin = 5e-3, ymax=frac_egos_random, linestyles='--', colors='0.6', lw=plot_props['linewidth'] )
		plt.hlines( y=frac_egos_random, xmin = 1e-4, xmax=1, linestyles='--', colors='0.6', lw=plot_props['linewidth'] )

		#texts

		plt.text( 1, 1.1, textname, va='top', ha='right', transform=ax.transAxes, fontsize=plot_props['text_size'] )

		txt_str = '$N_{\mathrm{filtered}} =$ '+'{}'.format(num_egos_filter)+'\n'+r'$n_{\mathrm{random}} =$'+'{:.2f}'.format(frac_egos_random)
		plt.text( 0.05, 0.05, txt_str, va='bottom', ha='left', transform=ax.transAxes, fontsize=plot_props['text_size'] )

		#finalise subplot
		plt.axis([ 1e-4, 1e2, 5e-3, 1.2e0 ])
		ax.tick_params( axis='both', which='both', direction='in', labelsize=plot_props['ticklabel'], length=2, pad=4 )
		ax.locator_params( numticks=6 )
		# if grid_pos not in [10, 11, 12, 13]:
		# 	ax.tick_params(labelbottom=False)
		if grid_pos not in [0]:
			ax.tick_params(labelleft=False)

	#finalise plot
	if fig_props['savename'] != '':
		plt.savefig( fig_props['savename']+'.pdf', format='pdf', dpi=fig_props['dpi'] )
