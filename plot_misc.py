#! /usr/bin/env python

### MODULE FOR MISCELLANEOUS FUNCTIONS FOR PLOTTING IN FARSIGNATURES PROJECT ###

#import modules
import os
import numpy as np
import pandas as pd
import scipy.stats as ss

import data_misc as dm


## FUNCTIONS ##

#function to plot complementary cumulative distribution P[X >= x]
def plot_compcum_dist( yplot_data ):
	"""Plot complementary cumulative distribution P[X >= x]"""

	defaultreallimits = ( yplot_data.min() - 1.5, yplot_data.max() + 0.5 ) #dist limits, counting 0
	numbins = int( yplot_data.max() - yplot_data.min() + 2 ) #number of bins
	#cumulative frequency histogram
	cumfreq = ss.cumfreq( yplot_data, defaultreallimits=defaultreallimits, numbins=numbins )

	xplot = np.linspace( yplot_data.min(), yplot_data.max(), numbins-1 ) #data points
	yplot = 1 - cumfreq.cumcount[:-1] / yplot_data.size #P[X >= x] (without last bin!)

	return xplot, yplot


#function to plot complementary cumulative distribution P[X >= x]
def plot_CCDF_cont( yplot_data ):
	"""Plot complementary cumulative distribution P[X >= x]"""

	xplot = yplot_data.sort_values() #sort data as x-axis
	#get (complementary) proportional values of samples as y-axis
	yplot = 1 - np.linspace( 0, 1, len( yplot_data ), endpoint=False )

	return xplot, yplot


#function to plot logbinned distribution
def plot_logbinned_dist( yplot_data, num=30 ):
	"""Plot logbinned distribution"""

	#linear bins and log bins
	bins = np.concatenate(( np.linspace( 1, 9, num=9 ), np.logspace( 1, np.log10( yplot_data.max() ), num=num ) ))

	yplot, bin_edges = np.histogram( yplot_data, bins=bins, density=True )
	xplot = [ ( bin_edges[i+1] + bin_edges[i] ) / 2 for i in range(len( bin_edges[:-1] )) ]

	return xplot, yplot


#function to plot kernel curve according to filter
def plot_kernel_filter( eventname, filt_rule='', filt_obj=None, filt_params={}, load=False, saveloc='', saveloc_fig='' ):
	"""Plot kernel curve according to filter"""

	savename = saveloc_fig+'kernel_{}_filter_rule_{}_params_{}.pkl'.format( eventname, filt_rule, ''.join([ k+'_'+'{:.2f}'.format(v)+'_' for k, v in filt_params.items() ]) ) #filename to load/save

	#set filter condition
	cond = (filt_params['min_val'] <= filt_obj) & (filt_obj < filt_params['max_val']) #filter condition
	filt_ind = filt_obj[cond].index #filtered indices

	if load:
		data_avg = pd.read_pickle(savename) #load file
	else:
		#prepare connection kernel
		egonet_kernel = pd.read_pickle( saveloc+'egonet_kernel_'+eventname+'.pkl' )

		#filter egos by selected filter property
		filter = egonet_kernel[filt_ind]

		#get filtered activity groups
		filt_negos = filter.groupby( level=1 ).filter( lambda x : len(x) >= filt_params['min_negos'] )
		data_grp = filt_negos.groupby( level=1 ) #group ego probs for each activity value
		data_avg = data_grp.mean() #average probs over egos

		if saveloc_fig:
			data_avg.to_pickle(savename) #save file

	return data_avg, filt_ind


#function to plot alter activity CCDF and average social signature according to filter
def plot_activity_filter( dataname, eventname, filt_rule='', filt_obj=None, filt_params={}, is_parallel=False, load=False, root_data='', saveloc='', saveloc_fig='' ):
	"""Plot alter activity CCDF and average social signature according to filter"""

	savename = '_{}_filter_rule_{}_params_{}.pkl'.format( eventname, filt_rule, ''.join([ k+'_'+'{:.2f}'.format(v)+'_' for k, v in filt_params.items() ]) ) #filename to load/save

	if load: #load aggregated activity files
		ccdf = pd.read_pickle(saveloc_fig+'ccdf'+savename)
		sign = pd.read_pickle(saveloc_fig+'sign'+savename)
	else: #or compute them from scratch

		#set filter condition
		cond = (filt_params['min_val'] <= filt_obj) & (filt_obj < filt_params['max_val']) #filter condition
		filt_ind = filt_obj[cond].index #filtered indices

		#load and filter alter activities
		if is_parallel == False: #for small datasets
			egonet_acts = pd.read_pickle( saveloc + 'egonet_acts_' + eventname + '.pkl' )

			#filter egos by selected filter property
			acts_filter = egonet_acts[ egonet_acts.index.isin(filt_ind, level=0) ]

		else: #for large datasets (separated into several files)
			fileloc = root_data + dataname +'/'+ eventname + '/'
			# filelist = os.listdir( fileloc )
			filelist = [ '1020407_1039726.txt' ]

			for filepos, filename in enumerate( filelist ): #loop through files in data directory
				if filepos % 100 == 0: #to know where we stand
					print( '\t\tfile {} out of {}'.format( filepos, len(filelist) ), flush=True )

				#load alter activities (for piece of large dataset!)
				not_used, egonet_acts_piece = dm.egonet_props_acts_parallel( filename, fileloc, eventname, 'y', saveloc )

				#filter egos by selected filter property
				acts_filter_piece = egonet_acts_piece[ egonet_acts_piece.index.isin(filt_ind, level=0) ]

				if filepos: #accumulate pieces of large dataset
					acts_filter = pd.concat([ acts_filter, acts_filter_piece ])
				else: #and initialise dataframe
					acts_filter = acts_filter_piece

		#get average social signature and alter activity CCDF

		#sort alter activities (across all egos) and normalise (by max activity for each ego)
		acts_filter_sorted = acts_filter.sort_values(ascending=False) / acts_filter.groupby(level=0).sum()
		#reset inner index (alter, nodej) as activity rank, for ego aggregation
		acts_filter_sorted.index = pd.MultiIndex.from_arrays( [acts_filter_sorted.index.get_level_values(0), acts_filter_sorted.groupby(level=0).cumcount()], names=['nodei', 'arank'] )

		#get filtered activity rank groups
		acts_filter_sorted_negos = acts_filter_sorted[ acts_filter_sorted.groupby(level=1).transform('count') >= filt_params['min_negos'] ]
		#group ego probs for each activity rank and average probs over egos
		sign = acts_filter_sorted_negos.groupby(level=1).mean()

		#get alter activity CCDF: P[X >= x]
		ccdf_x, ccdf_y = plot_compcum_dist( acts_filter )

		#save alter activity CCDF and average social signature
		ccdf = pd.DataFrame(data={ 'x':ccdf_x, 'y':ccdf_y })
		if saveloc_fig:
			ccdf.to_pickle(saveloc_fig+'ccdf'+savename)
			sign.to_pickle(saveloc_fig+'sign'+savename)

	return ccdf, sign


def draw_brace(ax, xspan, yy, text):
    """Draws an annotated brace on the axes"""

    xmin, xmax = xspan
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan/xax_span*100)*2+1 # guaranteed uneven
    beta = 300./xax_span # the higher this is, the smaller the radius

    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:int(resolution/2)+1]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                    + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = yy + (.05*y - .01)*yspan # adjust vertical position

    ax.autoscale(False)
    ax.plot(x, y, color='black', lw=1)

    ax.text((xmax+xmin)/2., yy+.07*yspan, text, ha='center', va='bottom')


#DEBUGGIN'

# 		#filter egos by selected filter property
# 		if filt_rule == 'large_disp': #large dispersion
# 			filter = egonet_kernel[ filt_obj[ filt_obj > filt_obj.mean() ].index ]
# 		elif filt_rule == 'small_disp': #small dispersion
# 			filter = egonet_kernel[ filt_obj[ filt_obj < filt_obj.mean() ].index ]
# 		elif filt_rule == 'degree': #large enough degree
# 			filter = egonet_kernel[ filt_obj[ filt_obj.degree >= filt_params['min_degree'] ].index ]
# 		else: #no filter
# 			filter = egonet_kernel

		# #filter egos by selected filter property
		# if filt_rule == 'dispersion': #dispersion within bounds
		# 	cond = (filt_params['min_val'] <= filt_obj) & (filt_obj <= filt_params['max_val'])
		# elif filt_rule == 'degree': #degree within bounds
		# 	cond = (filt_params['min_val'] <= filt_obj.degree) & (filt_obj.degree <= filt_params['max_val'])

		# acts_filter_sorted_negos = acts_filter_sorted.groupby(level=1).filter( lambda x : len(x) >= filt_params['min_negos'] )
