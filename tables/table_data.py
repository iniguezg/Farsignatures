#! /usr/bin/env python

### SCRIPT FOR CREATING DATA TABLE IN FARSIGNATURES PROJECT ###

#import modules
import pandas as pd
from os.path import expanduser


## RUNNING DATA SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	root_loc = expanduser('~') + '/prg/xocial/Farsignatures/' #root location of code
	saveloc_data = root_loc + 'files/data/' #location of data files

	#get parameters for all datasets
	params_data = pd.read_pickle( saveloc_data + 'params_data.pkl' )

	## PRINTING ##

	print(
r"""
\begin{table}[!ht]
\small
\noindent\makebox[\textwidth]{ \begin{tabular}{l l r r r r r r r}
\toprule
Dataset & Event description & $N$ & $V$ & $\langle k \rangle$ & $\langle \tau \rangle$ & $\langle t \rangle$ & $\langle a_0 \rangle$ & $\langle a_m \rangle$ \\
\midrule
Mobile (call)~\cite{onnela2007analysis,onnela2007structure,karsai2011small,kivela2012multiscale,kovanen2013temporal,unicomb2018threshold,heydari2018multichannel} & Phone call records &"""
+r' {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'MPC_UEu', 'num_egos' ], params_data.at[ 'MPC_UEu', 'num_events' ], params_data.at[ 'MPC_UEu', 'avg_degree' ], params_data.at[ 'MPC_UEu', 'avg_strength' ], params_data.at[ 'MPC_UEu', 'avg_activity' ], params_data.at[ 'MPC_UEu', 'avg_actmin' ], params_data.at[ 'MPC_UEu', 'avg_actmax' ] )+
r"""
Mobile (Wu 1)~\cite{wu2010evidence} & Short messages &"""
+r' {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'MPC_Wu_SD01', 'num_egos' ], params_data.at[ 'MPC_Wu_SD01', 'num_events' ], params_data.at[ 'MPC_Wu_SD01', 'avg_degree' ], params_data.at[ 'MPC_Wu_SD01', 'avg_strength' ], params_data.at[ 'MPC_Wu_SD01', 'avg_activity' ], params_data.at[ 'MPC_Wu_SD01', 'avg_actmin' ], params_data.at[ 'MPC_Wu_SD01', 'avg_actmax' ] )+
r"""
Mobile (Wu 2)~\cite{wu2010evidence} & Short messages &"""
+r' {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'MPC_Wu_SD02', 'num_egos' ], params_data.at[ 'MPC_Wu_SD02', 'num_events' ], params_data.at[ 'MPC_Wu_SD02', 'avg_degree' ], params_data.at[ 'MPC_Wu_SD02', 'avg_strength' ], params_data.at[ 'MPC_Wu_SD02', 'avg_activity' ], params_data.at[ 'MPC_Wu_SD02', 'avg_actmin' ], params_data.at[ 'MPC_Wu_SD02', 'avg_actmax' ] )+
r"""
Mobile (Wu 3)~\cite{wu2010evidence} & Short messages &"""
+r' {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'MPC_Wu_SD03', 'num_egos' ], params_data.at[ 'MPC_Wu_SD03', 'num_events' ], params_data.at[ 'MPC_Wu_SD03', 'avg_degree' ], params_data.at[ 'MPC_Wu_SD03', 'avg_strength' ], params_data.at[ 'MPC_Wu_SD03', 'avg_activity' ], params_data.at[ 'MPC_Wu_SD03', 'avg_actmin' ], params_data.at[ 'MPC_Wu_SD03', 'avg_actmax' ] )+
r"""
Contact~\cite{rocha2010information,rocha2011simulated} & Reported sexual contacts &"""
+r' {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'sexcontact_events', 'num_egos' ], params_data.at[ 'sexcontact_events', 'num_events' ], params_data.at[ 'sexcontact_events', 'avg_degree' ], params_data.at[ 'sexcontact_events', 'avg_strength' ], params_data.at[ 'sexcontact_events', 'avg_activity' ], params_data.at[ 'sexcontact_events', 'avg_actmin' ], params_data.at[ 'sexcontact_events', 'avg_actmax' ] )+
r"""
Email 1~\cite{ebel2002scale,saramaki2015exploring} & Email communication &"""
+r' {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'email', 'num_egos' ], params_data.at[ 'email', 'num_events' ], params_data.at[ 'email', 'avg_degree' ], params_data.at[ 'email', 'avg_strength' ], params_data.at[ 'email', 'avg_activity' ], params_data.at[ 'email', 'avg_actmin' ], params_data.at[ 'email', 'avg_actmax' ] )+
r"""
Email 2~\cite{eckmann2004entropy,saramaki2015exploring} & Email communication &"""
+r' {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'eml2', 'num_egos' ], params_data.at[ 'eml2', 'num_events' ], params_data.at[ 'eml2', 'avg_degree' ], params_data.at[ 'eml2', 'avg_strength' ], params_data.at[ 'eml2', 'avg_activity' ], params_data.at[ 'eml2', 'avg_actmin' ], params_data.at[ 'eml2', 'avg_actmax' ] )+
r"""
Facebook~\cite{viswanath2009evolution,saramaki2015exploring} & Online messages &"""
+r' {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'fb', 'num_egos' ], params_data.at[ 'fb', 'num_events' ], params_data.at[ 'fb', 'avg_degree' ], params_data.at[ 'fb', 'avg_strength' ], params_data.at[ 'fb', 'avg_activity' ], params_data.at[ 'fb', 'avg_actmin' ], params_data.at[ 'fb', 'avg_actmax' ] )+
r"""
Messages~\cite{said2010social,karimi2014structural,saramaki2015exploring} & Online messages &"""
+r' {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'messages', 'num_egos' ], params_data.at[ 'messages', 'num_events' ], params_data.at[ 'messages', 'avg_degree' ], params_data.at[ 'messages', 'avg_strength' ], params_data.at[ 'messages', 'avg_activity' ], params_data.at[ 'messages', 'avg_actmin' ], params_data.at[ 'messages', 'avg_actmax' ] )+
r"""
Forum~\cite{said2010social,karimi2014structural,saramaki2015exploring} & Online messages &"""
+r' {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'forum', 'num_egos' ], params_data.at[ 'forum', 'num_events' ], params_data.at[ 'forum', 'avg_degree' ], params_data.at[ 'forum', 'avg_strength' ], params_data.at[ 'forum', 'avg_activity' ], params_data.at[ 'forum', 'avg_actmin' ], params_data.at[ 'forum', 'avg_actmax' ] )+
r"""
Dating~\cite{holme2004structure,saramaki2015exploring} & Online messages &"""
+r' {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'pok', 'num_egos' ], params_data.at[ 'pok', 'num_events' ], params_data.at[ 'pok', 'avg_degree' ], params_data.at[ 'pok', 'avg_strength' ], params_data.at[ 'pok', 'avg_activity' ], params_data.at[ 'pok', 'avg_actmin' ], params_data.at[ 'pok', 'avg_actmax' ] )+
r"""
CNS (bluetooth)~\cite{stopczynski2014measuring,sapiezynski2019interaction} & Proximity via Bluetooth &"""
+r' {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'CNS_bt_symmetric', 'num_egos' ], params_data.at[ 'CNS_bt_symmetric', 'num_events' ], params_data.at[ 'CNS_bt_symmetric', 'avg_degree' ], params_data.at[ 'CNS_bt_symmetric', 'avg_strength' ], params_data.at[ 'CNS_bt_symmetric', 'avg_activity' ], params_data.at[ 'CNS_bt_symmetric', 'avg_actmin' ], params_data.at[ 'CNS_bt_symmetric', 'avg_actmax' ] )+
r"""
CNS (call)~\cite{stopczynski2014measuring,sapiezynski2019interaction} & Phone call records &"""
+r' {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'CNS_calls', 'num_egos' ], params_data.at[ 'CNS_calls', 'num_events' ], params_data.at[ 'CNS_calls', 'avg_degree' ], params_data.at[ 'CNS_calls', 'avg_strength' ], params_data.at[ 'CNS_calls', 'avg_activity' ], params_data.at[ 'CNS_calls', 'avg_actmin' ], params_data.at[ 'CNS_calls', 'avg_actmax' ] )+
r"""
CNS (sms)~\cite{stopczynski2014measuring,sapiezynski2019interaction} & Short messages &"""
+r' {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'CNS_sms', 'num_egos' ], params_data.at[ 'CNS_sms', 'num_events' ], params_data.at[ 'CNS_sms', 'avg_degree' ], params_data.at[ 'CNS_sms', 'avg_strength' ], params_data.at[ 'CNS_sms', 'avg_activity' ], params_data.at[ 'CNS_sms', 'avg_actmin' ], params_data.at[ 'CNS_sms', 'avg_actmax' ] )+
r"""
\bottomrule
\end{tabular}}
\caption{
\small {\bf Datasets used in this study}.
Characteristics of the available datasets, including system size $N$ (number of egos), number of events $V$ (all communication events between egos and alters), average degree $\langle k \rangle$ (mean number of alters per ego), average strength $\langle \tau \rangle$ (mean number of events per ego), average mean alter activity $\langle t \rangle$ (mean number of events per alter per ego), and average minimum/maximum alter activity $\langle a_0 \rangle$ and $\langle a_m \rangle$ (mean of lowest/highest alter activity per ego). The table includes references to detailed studies of each dataset and locations of publicly available data.
}
\label{tab:datasets}
\end{table}
"""
	)
