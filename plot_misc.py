#! /usr/bin/env python

### MODULE FOR MISCELLANEOUS FUNCTIONS FOR PLOTTING IN FARSIGNATURES PROJECT ###

#import modules
import numpy as np
import scipy.stats as ss

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
