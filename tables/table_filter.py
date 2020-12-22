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
	nsims = 1000 #number of syntethic datasets used to calculate p-value
	amax = 10000 #maximum activity for theoretical activity distribution

	pval_thres = 0.1 #threshold above which alpha MLEs are considered
	alph_thres = 1 #threshold below alphamax to define alpha MLE -> inf

	#locations
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files

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


	## DATA ##

	#initialise dataframe of filter properties
	params_filter = pd.DataFrame( np.zeros( ( len(datasets), 7 ) ), index=pd.Series( [ dset[1][:-4] for dset in datasets ], name='dataset') , columns=pd.Series( [ 'num_egos', 'frac_egos_filter', 'frac_egos_inf', 'frac_egos_null', 'num_egos_filter', 'frac_egos_cumadv', 'frac_egos_random' ], name='parameter' ) )

	#loop through considered datasets
	for grid_pos, (dataname, eventname, textname) in enumerate(datasets):
#		for grid_pos, (dataname, eventname, textname) in enumerate([ ('MPC_UEu_net', 'MPC_UEu.evt', 'Mobile (call)') ]):
		print( 'dataset name: ' + eventname[:-4] ) #print output

		#prepare ego network properties
		egonet_props, egonet_acts = dm.egonet_props_acts( dataname, eventname, root_data, 'y', saveloc )

		#fit activity model to all ego networks in dataset
		egonet_fits = dm.egonet_fits( dataname, eventname, root_data, 'y', saveloc, alphamax=alphamax, nsims=nsims, amax=amax )

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
		params_filter.loc[ eventname[:-4] ] = ( num_egos, frac_egos_filter, frac_egos_inf, frac_egos_null, num_egos_filter, frac_egos_cumadv, frac_egos_random )


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
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[0][2], *params_filter.loc[ 'MPC_UEu' ] )+'\n'
+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[1][2], *params_filter.loc[ 'MPC_Wu_SD01' ] )+'\n'
+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[2][2], *params_filter.loc[ 'MPC_Wu_SD02' ] )+'\n'+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[3][2], *params_filter.loc[ 'MPC_Wu_SD03' ] )+'\n'+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[4][2], *params_filter.loc[ 'sexcontact_events' ] )+'\n'+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[5][2], *params_filter.loc[ 'email' ] )+'\n'+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[6][2], *params_filter.loc[ 'eml2' ] )+'\n'+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[7][2], *params_filter.loc[ 'fb' ] )+'\n'+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[8][2], *params_filter.loc[ 'messages' ] )+'\n'+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[9][2], *params_filter.loc[ 'forum' ] )+'\n'+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[10][2], *params_filter.loc[ 'pok' ] )+'\n'+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[11][2], *params_filter.loc[ 'CNS_bt_symmetric' ] )+'\n'+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[12][2], *params_filter.loc[ 'CNS_calls' ] )+'\n'+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} \\'.format( datasets[13][2], *params_filter.loc[ 'CNS_sms' ] )
+r"""
\bottomrule
\end{tabular}}
\caption{
\small {\bf }.
.
}
\label{tab:filterClasses}
\end{table}
"""
	)
