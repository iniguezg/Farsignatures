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

	#get parameters for all datasets (single and parallel)
	params_data = pd.read_pickle( saveloc_data + 'params_data.pkl' )
	params_data_parallel = pd.read_pickle( saveloc_data + 'params_data_parallel.pkl' )

	## PRINTING ##

	print(
r"""
\begin{table}[!ht]
\small
\noindent\makebox[\textwidth]{ \begin{tabular}{l l r r | r r r r r r}
\toprule
Dataset & Event & $N_u$ & $V$ & $N$ & $\langle k \rangle$ & $\langle \tau \rangle$ & $\langle t \rangle$ & $\langle a_0 \rangle$ & $\langle a_m \rangle$ \\
\midrule
Mobile (call)~\cite{onnela2007analysis,onnela2007structure,karsai2011small,kivela2012multiscale,kovanen2013temporal,unicomb2018threshold,heydari2018multichannel} & Phone calls &"""
+r' {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data_parallel.at[ 'call', 'num_all' ], params_data_parallel.at[ 'call', 'num_events' ], params_data_parallel.at[ 'call', 'num_egos' ], params_data_parallel.at[ 'call', 'avg_degree' ], params_data_parallel.at[ 'call', 'avg_strength' ], params_data_parallel.at[ 'call', 'avg_actavg' ], params_data_parallel.at[ 'call', 'avg_actmin' ], params_data_parallel.at[ 'call', 'avg_actmax' ] )+
r"""
Mobile (sms)~\cite{onnela2007analysis,onnela2007structure,karsai2011small,kivela2012multiscale,kovanen2013temporal,unicomb2018threshold,heydari2018multichannel} & Short messages &"""
+r' {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data_parallel.at[ 'text', 'num_all' ], params_data_parallel.at[ 'text', 'num_events' ], params_data_parallel.at[ 'text', 'num_egos' ], params_data_parallel.at[ 'text', 'avg_degree' ], params_data_parallel.at[ 'text', 'avg_strength' ], params_data_parallel.at[ 'text', 'avg_actavg' ], params_data_parallel.at[ 'text', 'avg_actmin' ], params_data_parallel.at[ 'text', 'avg_actmax' ] )+
r"""
Mobile (Wu 1)~\cite{wu2010evidence} & Short messages &"""
+r' {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'MPC_Wu_SD01', 'num_all' ], params_data.at[ 'MPC_Wu_SD01', 'num_events' ], params_data.at[ 'MPC_Wu_SD01', 'num_egos' ], params_data.at[ 'MPC_Wu_SD01', 'avg_degree' ], params_data.at[ 'MPC_Wu_SD01', 'avg_strength' ], params_data.at[ 'MPC_Wu_SD01', 'avg_actavg' ], params_data.at[ 'MPC_Wu_SD01', 'avg_actmin' ], params_data.at[ 'MPC_Wu_SD01', 'avg_actmax' ] )+
r"""
Mobile (Wu 2)~\cite{wu2010evidence} & Short messages &"""
+r' {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'MPC_Wu_SD02', 'num_all' ], params_data.at[ 'MPC_Wu_SD02', 'num_events' ], params_data.at[ 'MPC_Wu_SD02', 'num_egos' ], params_data.at[ 'MPC_Wu_SD02', 'avg_degree' ], params_data.at[ 'MPC_Wu_SD02', 'avg_strength' ], params_data.at[ 'MPC_Wu_SD02', 'avg_actavg' ], params_data.at[ 'MPC_Wu_SD02', 'avg_actmin' ], params_data.at[ 'MPC_Wu_SD02', 'avg_actmax' ] )+
r"""
Mobile (Wu 3)~\cite{wu2010evidence} & Short messages &"""
+r' {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'MPC_Wu_SD03', 'num_all' ], params_data.at[ 'MPC_Wu_SD03', 'num_events' ], params_data.at[ 'MPC_Wu_SD03', 'num_egos' ], params_data.at[ 'MPC_Wu_SD03', 'avg_degree' ], params_data.at[ 'MPC_Wu_SD03', 'avg_strength' ], params_data.at[ 'MPC_Wu_SD03', 'avg_actavg' ], params_data.at[ 'MPC_Wu_SD03', 'avg_actmin' ], params_data.at[ 'MPC_Wu_SD03', 'avg_actmax' ] )+
# r"""
# Contact~\cite{rocha2010information,rocha2011simulated} & Reported sexual contacts &"""
# +r' {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'sexcontact_events', 'num_all' ], params_data.at[ 'sexcontact_events', 'num_events' ], params_data.at[ 'sexcontact_events', 'num_egos' ], params_data.at[ 'sexcontact_events', 'avg_degree' ], params_data.at[ 'sexcontact_events', 'avg_strength' ], params_data.at[ 'sexcontact_events', 'avg_actavg' ], params_data.at[ 'sexcontact_events', 'avg_actmin' ], params_data.at[ 'sexcontact_events', 'avg_actmax' ] )+
r"""
Email (Enron)~\cite{} & Emails &"""
+r' {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'Enron', 'num_all' ], params_data.at[ 'Enron', 'num_events' ], params_data.at[ 'Enron', 'num_egos' ], params_data.at[ 'Enron', 'avg_degree' ], params_data.at[ 'Enron', 'avg_strength' ], params_data.at[ 'Enron', 'avg_actavg' ], params_data.at[ 'Enron', 'avg_actmin' ], params_data.at[ 'Enron', 'avg_actmax' ] )+
r"""
Email (Kiel)~\cite{ebel2002scale,saramaki2015exploring} & Emails &"""
+r' {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'email', 'num_all' ], params_data.at[ 'email', 'num_events' ], params_data.at[ 'email', 'num_egos' ], params_data.at[ 'email', 'avg_degree' ], params_data.at[ 'email', 'avg_strength' ], params_data.at[ 'email', 'avg_actavg' ], params_data.at[ 'email', 'avg_actmin' ], params_data.at[ 'email', 'avg_actmax' ] )+
r"""
Email (Uni)~\cite{eckmann2004entropy,saramaki2015exploring} & Emails &"""
+r' {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'eml2', 'num_all' ], params_data.at[ 'eml2', 'num_events' ], params_data.at[ 'eml2', 'num_egos' ], params_data.at[ 'eml2', 'avg_degree' ], params_data.at[ 'eml2', 'avg_strength' ], params_data.at[ 'eml2', 'avg_actavg' ], params_data.at[ 'eml2', 'avg_actmin' ], params_data.at[ 'eml2', 'avg_actmax' ] )+
r"""
Email (EU)~\cite{} & Emails &"""
+r' {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'email_Eu_core', 'num_all' ], params_data.at[ 'email_Eu_core', 'num_events' ], params_data.at[ 'email_Eu_core', 'num_egos' ], params_data.at[ 'email_Eu_core', 'avg_degree' ], params_data.at[ 'email_Eu_core', 'avg_strength' ], params_data.at[ 'email_Eu_core', 'avg_actavg' ], params_data.at[ 'email_Eu_core', 'avg_actmin' ], params_data.at[ 'email_Eu_core', 'avg_actmax' ] )+
r"""
Facebook~\cite{viswanath2009evolution,saramaki2015exploring} & Online messages &"""
+r' {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'fb', 'num_all' ], params_data.at[ 'fb', 'num_events' ], params_data.at[ 'fb', 'num_egos' ], params_data.at[ 'fb', 'avg_degree' ], params_data.at[ 'fb', 'avg_strength' ], params_data.at[ 'fb', 'avg_actavg' ], params_data.at[ 'fb', 'avg_actmin' ], params_data.at[ 'fb', 'avg_actmax' ] )+
r"""
Messages~\cite{said2010social,karimi2014structural,saramaki2015exploring} & Online messages &"""
+r' {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'messages', 'num_all' ], params_data.at[ 'messages', 'num_events' ], params_data.at[ 'messages', 'num_egos' ], params_data.at[ 'messages', 'avg_degree' ], params_data.at[ 'messages', 'avg_strength' ], params_data.at[ 'messages', 'avg_actavg' ], params_data.at[ 'messages', 'avg_actmin' ], params_data.at[ 'messages', 'avg_actmax' ] )+
r"""
Dating~\cite{holme2004structure,saramaki2015exploring} & Online messages &"""
+r' {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'pok', 'num_all' ], params_data.at[ 'pok', 'num_events' ], params_data.at[ 'pok', 'num_egos' ], params_data.at[ 'pok', 'avg_degree' ], params_data.at[ 'pok', 'avg_strength' ], params_data.at[ 'pok', 'avg_actavg' ], params_data.at[ 'pok', 'avg_actmin' ], params_data.at[ 'pok', 'avg_actmax' ] )+
r"""
Forum~\cite{said2010social,karimi2014structural,saramaki2015exploring} & Online messages &"""
+r' {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'forum', 'num_all' ], params_data.at[ 'forum', 'num_events' ], params_data.at[ 'forum', 'num_egos' ], params_data.at[ 'forum', 'avg_degree' ], params_data.at[ 'forum', 'avg_strength' ], params_data.at[ 'forum', 'avg_actavg' ], params_data.at[ 'forum', 'avg_actmin' ], params_data.at[ 'forum', 'avg_actmax' ] )+
r"""
College~\cite{} & Online messages &"""
+r' {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'CollegeMsg', 'num_all' ], params_data.at[ 'CollegeMsg', 'num_events' ], params_data.at[ 'CollegeMsg', 'num_egos' ], params_data.at[ 'CollegeMsg', 'avg_degree' ], params_data.at[ 'CollegeMsg', 'avg_strength' ], params_data.at[ 'CollegeMsg', 'avg_actavg' ], params_data.at[ 'CollegeMsg', 'avg_actmin' ], params_data.at[ 'CollegeMsg', 'avg_actmax' ] )+
# r"""
# CNS (bluetooth)~\cite{stopczynski2014measuring,sapiezynski2019interaction} & Proximity via Bluetooth &"""
# +r' {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'CNS_bt_symmetric', 'num_all' ], params_data.at[ 'CNS_bt_symmetric', 'num_events' ], params_data.at[ 'CNS_bt_symmetric', 'num_egos' ], params_data.at[ 'CNS_bt_symmetric', 'avg_degree' ], params_data.at[ 'CNS_bt_symmetric', 'avg_strength' ], params_data.at[ 'CNS_bt_symmetric', 'avg_actavg' ], params_data.at[ 'CNS_bt_symmetric', 'avg_actmin' ], params_data.at[ 'CNS_bt_symmetric', 'avg_actmax' ] )+
r"""
CNS (call)~\cite{stopczynski2014measuring,sapiezynski2019interaction} & Phone calls &"""
+r' {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'CNS_calls', 'num_all' ], params_data.at[ 'CNS_calls', 'num_events' ], params_data.at[ 'CNS_calls', 'num_egos' ], params_data.at[ 'CNS_calls', 'avg_degree' ], params_data.at[ 'CNS_calls', 'avg_strength' ], params_data.at[ 'CNS_calls', 'avg_actavg' ], params_data.at[ 'CNS_calls', 'avg_actmin' ], params_data.at[ 'CNS_calls', 'avg_actmax' ] )+
r"""
CNS (sms)~\cite{stopczynski2014measuring,sapiezynski2019interaction} & Short messages &"""
+r' {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format( params_data.at[ 'CNS_sms', 'num_all' ], params_data.at[ 'CNS_sms', 'num_events' ], params_data.at[ 'CNS_sms', 'num_egos' ], params_data.at[ 'CNS_sms', 'avg_degree' ], params_data.at[ 'CNS_sms', 'avg_strength' ], params_data.at[ 'CNS_sms', 'avg_actavg' ], params_data.at[ 'CNS_sms', 'avg_actmin' ], params_data.at[ 'CNS_sms', 'avg_actmax' ] )+
r"""
\bottomrule
\end{tabular}}
\caption{
\small {\bf Datasets used in this study}.
Characteristics of the available datasets, starting with system size $N_u$ (unfiltered number of egos) and number of events $V$ (all communication events between egos and alters). We only consider egos with mean alter activity larger than its minimum ($t > a_0$), leading to a system of size $N$ (filtered number of egos) with the following properties: average degree $\langle k \rangle$ (mean number of alters per ego), average strength $\langle \tau \rangle$ (mean number of events per ego), average mean alter activity $\langle t \rangle$ (mean number of events per alter per ego), and average minimum/maximum alter activity $\langle a_0 \rangle$ and $\langle a_m \rangle$ (mean of lowest/highest alter activity per ego). We include references to detailed studies of each dataset and locations of publicly available data.
}
\label{tab:datasets}
\end{table}
"""
	)
