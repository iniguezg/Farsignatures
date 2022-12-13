#! /usr/bin/env python

### SCRIPT FOR CREATING TEST STATISTICS TABLE IN FARSIGNATURES PROJECT ###

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

	#locations
	root_data = expanduser('~') + '/prg/xocial/datasets/temporal_networks/' #root location of data/code
	root_code = expanduser('~') + '/prg/xocial/Farsignatures/'
	saveloc = root_code+'files/data/' #location of output files

	#dataset list: eventname, textname
	datasets = [ ( 'call', 'Mobile (call)'),
				 ( 'text', 'Mobile (sms)'),
				 ( 'MPC_Wu_SD01', 'Mobile (Wu 1)'),
				 ( 'MPC_Wu_SD02', 'Mobile (Wu 2)'),
				 ( 'MPC_Wu_SD03', 'Mobile (Wu 3)'),
				 ( 'Enron', 'Email (Enron)'),
				 ( 'email', 'Email (Kiel)'),
				 ( 'eml2', 'Email (Uni)'),
				 ( 'email_Eu_core', 'Email (EU)'),
				 ( 'fb', 'Facebook'),
				 ( 'messages', 'Messages'),
				 ( 'pok', 'Dating'),
				 ( 'forum', 'Forum'),
				 ( 'CollegeMsg', 'College'),
				 ( 'CNS_calls', 'CNS (call)'),
				 ( 'CNS_sms', 'CNS (sms)') ]


	## DATA ##

	#initialise dataframe of filter properties
	columns = [ 'num_egos', 'frac_egos_KS', 'frac_egos_W2', 'frac_egos_U2', 'frac_egos_A2' ]
	params_stats = pd.DataFrame( np.zeros( ( len(datasets), len(columns) ) ), index=pd.Series( [ dset[0] for dset in datasets ], name='dataset') , columns=pd.Series( columns, name='parameter' ) )

	#loop through considered datasets
	for eventname, textname in datasets:
	# for eventname, textname in datasets[2:]:
		print( 'dataset name: ' + eventname ) #print output

		#prepare ego network properties
		egonet_props = pd.read_pickle( saveloc + 'egonet_props_' + eventname + '.pkl' )
		#filter egos by t > a0 condition
		egonet_props_filter = egonet_props[ egonet_props.degree * egonet_props.act_min < egonet_props.strength ]
		#fit activity model to all ego networks in dataset
		egonet_fits = pd.read_pickle( saveloc + 'egonet_fits_' + eventname + '.pkl' )

		#filter egos according to fitting results
		#Kolmogorov-Smirnov
		egonet_filter_KS, not_used, not_used = dm.egonet_filter( egonet_props, egonet_fits, stat='KS', pval_thres=pval_thres, alphamax=alphamax, alph_thres=alph_thres )
		#Cramer-von Mises
		egonet_filter_W2, not_used, not_used = dm.egonet_filter( egonet_props, egonet_fits, stat='W2', pval_thres=pval_thres, alphamax=alphamax, alph_thres=alph_thres )
		egonet_filter_U2, not_used, not_used = dm.egonet_filter( egonet_props, egonet_fits, stat='U2', pval_thres=pval_thres, alphamax=alphamax, alph_thres=alph_thres )
		egonet_filter_A2, not_used, not_used = dm.egonet_filter( egonet_props, egonet_fits, stat='A2', pval_thres=pval_thres, alphamax=alphamax, alph_thres=alph_thres )

		#fractions of statistically significant egos
		num_egos = len(egonet_props_filter) #all (filtered) egos!
		frac_egos_KS = len(egonet_filter_KS) / float(num_egos)
		frac_egos_W2 = len(egonet_filter_W2) / float(num_egos)
		frac_egos_U2 = len(egonet_filter_U2) / float(num_egos)
		frac_egos_A2 = len(egonet_filter_A2) / float(num_egos)

		#store in dframe
		params_stats.loc[ eventname ] = ( num_egos, frac_egos_KS, frac_egos_W2, frac_egos_U2, frac_egos_A2 )
		#fix dtypes
		params_stats.num_egos = params_stats.num_egos.astype(int)


	## PRINTING ##

	print(
r"""
\begin{table}[t]
\small
\noindent\makebox[\textwidth]{ \begin{tabular}{l r | r r r r}
\toprule
Dataset & $N$ & $n_{D}$ & $n_{W^2}$ & $n_{U^2}$ & $n_{A^2}$ \\
\midrule"""+'\n'
+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( datasets[0][1], *params_stats.loc[ 'call' ] )+'\n'
+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( datasets[1][1], *params_stats.loc[ 'text' ] )+'\n'
# +
# r'{} & - & - & - & - & - \\'.format( datasets[0][1], *params_stats.loc[ 'call' ] )+'\n'
# +
# r'{} & - & - & - & - & - \\'.format( datasets[1][1], *params_stats.loc[ 'text' ] )+'\n'
+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( datasets[2][1], *params_stats.loc[ 'MPC_Wu_SD01' ] )+'\n'
+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( datasets[3][1], *params_stats.loc[ 'MPC_Wu_SD02' ] )+'\n'
+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( datasets[4][1], *params_stats.loc[ 'MPC_Wu_SD03' ] )+'\n'
+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( datasets[5][1], *params_stats.loc[ 'Enron' ] )+'\n'
+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( datasets[6][1], *params_stats.loc[ 'email' ] )+'\n'
+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( datasets[7][1], *params_stats.loc[ 'eml2' ] )+'\n'
+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( datasets[8][1], *params_stats.loc[ 'email_Eu_core' ] )+'\n'
+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( datasets[9][1], *params_stats.loc[ 'fb' ] )+'\n'
+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( datasets[10][1], *params_stats.loc[ 'messages' ] )+'\n'
+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( datasets[11][1], *params_stats.loc[ 'pok' ] )+'\n'
+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( datasets[12][1], *params_stats.loc[ 'forum' ] )+'\n'
+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( datasets[13][1], *params_stats.loc[ 'CollegeMsg' ] )+'\n'
+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( datasets[14][1], *params_stats.loc[ 'CNS_calls' ] )+'\n'
+
r'{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( datasets[15][1], *params_stats.loc[ 'CNS_sms' ] )
+r"""
\bottomrule
\end{tabular}}
\caption{
\small {\bf Statistical significance of maximum likelihood estimation}. Fraction $n_{\bullet}$ of ego networks satisfying the condition $p_{\bullet} > 0.1$ on the $p$-value $p_{\bullet}$ associated to the test statistics of Kolmogorov-Smirnov, Cramér-von Mises, Watson, and Anderson-Darling [$\bullet = D, W^2, U^2, A^2$, respectively; see \esref{eq:KSstat}{eq:A2stat}]. Fractions $n_{\bullet}$ are calculated relative to the number $N$ of egos in each dataset under the condition $t > a_0$ (i.e. with any level of heterogeneity on their communication signatures). The model is able to reproduce observed data for most egos, at least according to some statistic. For large datasets, statistical significance is robust to the choice of statistic.
}
\label{tab:filterStats}
\end{table}
"""
	)
