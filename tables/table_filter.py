#! /usr/bin/env python

### SCRIPT FOR CREATING FILTER CLASSES TABLE IN FARSIGNATURES PROJECT ###

#import modules
import os, sys
import numpy as np
import pandas as pd
from os.path import expanduser

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_misc as dm


## RUNNING DATA SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#parameters
	alphamax = 1000 #maximum alpha for MLE fit
	pval_thres = 0.1 #threshold above which alpha MLEs are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf
	# nsims = 1000 #number of syntethic datasets used to calculate p-value
	# amax = 10000 #maximum activity for theoretical activity distribution

	#locations
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files

	#dataset list: eventname, textname
	# datasets = [ ( 'MPC_UEu', 'Mobile (call)'),
	datasets = [ ( 'call', 'Mobile (call)'),
				 ( 'text', 'Mobile (sms)'),
				 ( 'MPC_Wu_SD01', 'Mobile (Wu 1)'),
				 ( 'MPC_Wu_SD02', 'Mobile (Wu 2)'),
				 ( 'MPC_Wu_SD03', 'Mobile (Wu 3)'),
				 ( 'sexcontact_events', 'Contact'),
				 ( 'email', 'Email 1'),
				 ( 'eml2', 'Email 2'),
				 ( 'fb', 'Facebook'),
				 ( 'messages', 'Messages'),
				 ( 'forum', 'Forum'),
				 ( 'pok', 'Dating'),
				 ( 'CNS_bt_symmetric', 'CNS (bluetooth)'),
				 ( 'CNS_calls', 'CNS (call)'),
				 ( 'CNS_sms', 'CNS (sms)') ]


	## DATA ##

	#initialise dataframe of filter properties
	params_filter = pd.DataFrame( np.zeros( ( len(datasets), 7 ) ), index=pd.Series( [ dset[1][:-4] for dset in datasets ], name='dataset') , columns=pd.Series( [ 'num_egos', 'frac_egos_filter', 'frac_egos_inf', 'frac_egos_null', 'num_egos_filter', 'frac_egos_cumadv', 'frac_egos_random' ], name='parameter' ) )

	#loop through considered datasets
	for grid_pos, (eventname, textname) in enumerate(datasets):
		print( 'dataset name: ' + eventname ) #print output

		#prepare ego network properties
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
		#fit activity model to all ego networks in dataset
		egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )

		#filter egos according to fitting results
		egonet_filter, egonet_inf, egonet_null = dm.egonet_filter( egonet_props, egonet_fits, alphamax=alphamax, pval_thres=pval_thres, alph_thres=alph_thres )

		#some measures

		num_egos = len( egonet_props ) #all egos

		num_egos_filter = len( egonet_filter ) #statistically significant alpha
		frac_egos_filter = num_egos_filter / float( num_egos )
		frac_egos_inf = len( egonet_inf ) / float( num_egos ) #infinite alpha
		frac_egos_null = len( egonet_null ) / float( num_egos ) #undefined alpha

		frac_egos_random = ( egonet_filter.beta < 1 ).sum() / float( num_egos_filter ) #fraction of egos in random regime (beta < 1, i.e. t_r < alpha_r)
		frac_egos_cumadv = ( egonet_filter.beta > 1 ).sum() / float( num_egos_filter ) #fraction of egos in CA regime (beta > 1, i.e. t_r > alpha_r)

		#store in dframe
		params_filter.loc[ eventname ] = ( num_egos, frac_egos_filter, frac_egos_inf, frac_egos_null, num_egos_filter, frac_egos_cumadv, frac_egos_random )


	## PRINTING ##

	print(
r"""
\begin{table}[t]
\small
\noindent\makebox[\textwidth]{ \begin{tabular}{l | r r r r | r r r}
\toprule
Dataset & $N$ & $n_{\alpha}$ & $n_{\infty}$ & $n_{\emptyset}$ & $N_{\alpha}$ & $n_{CA}$ & $n_{RN}$ \\
\midrule"""+'\n'
+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[0][1], *params_filter.loc[ 'call' ] )+'\n'
+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[1][1], *params_filter.loc[ 'text' ] )+'\n'
+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[2][1], *params_filter.loc[ 'MPC_Wu_SD01' ] )+'\n'
+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[3][1], *params_filter.loc[ 'MPC_Wu_SD02' ] )+'\n'+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[4][1], *params_filter.loc[ 'MPC_Wu_SD03' ] )+'\n'+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[5][1], *params_filter.loc[ 'sexcontact_events' ] )+'\n'+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[6][1], *params_filter.loc[ 'email' ] )+'\n'+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[7][1], *params_filter.loc[ 'eml2' ] )+'\n'+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[8][1], *params_filter.loc[ 'fb' ] )+'\n'+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[9][1], *params_filter.loc[ 'messages' ] )+'\n'+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[10][1], *params_filter.loc[ 'forum' ] )+'\n'+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[11][1], *params_filter.loc[ 'pok' ] )+'\n'+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[12][1], *params_filter.loc[ 'CNS_bt_symmetric' ] )+'\n'+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[13][1], *params_filter.loc[ 'CNS_calls' ] )+'\n'+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[14][1], *params_filter.loc[ 'CNS_sms' ] )
+r"""
\bottomrule
\end{tabular}}
\caption{
\small {\bf Ego classes based on maximum likelihood estimation (MLE)}.
We classify the $N$ ego networks in each studied dataset into a fraction $n_{\alpha} = N_{\alpha} / N$ with statistically significant MLE $\hat{\alpha}$ (relative mean activity $t_r > 0$, p-value $p > 0.1$, and $\hat{\alpha} < \alpha_b$ with $\alpha_b = 10^3$), a fraction $n_{\infty} = N_{\infty} / N$ with infinite $\hat{\alpha}$ [\eref{eq:logDeriv} does not converge to zero below $\alpha_b$], and the remaining fraction $n_{\emptyset} = N_{\emptyset} / N$ with undefined $\hat{\alpha}$. The $N_{\alpha}$ egos with statistically significant $\hat{\alpha}$ are separated into a fraction $n_{RN} = N_{RN} / N_{\alpha}$ in the homogenous regime ($\beta < 1$), and a fraction $n_{CA} = N_{CA} / N_{\alpha}$ in the heterogeneous regime ($\beta > 1$).
}
\label{tab:filterClasses}
\end{table}
"""
	)
