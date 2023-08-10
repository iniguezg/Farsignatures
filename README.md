# Farsignatures
[![DOI](https://zenodo.org/badge/297413241.svg)](https://zenodo.org/badge/latestdoi/297413241)

## Farsignatures - Exploring dynamics of egocentric communication networks

This library includes code to build, analyze, and visualize empirical datasets and model simulations of dynamics of egocentric communication networks. The description and results of communication data and ego network evolution models can be found in the following publication. If you use this code, please cite it as:

G. Iñiguez, S. Heydari, J. Kertész, J. Saramäki  
Universal patterns in egocentric communication networks  
To appear in *Nature Communications* (2023)  
[arXiv: 2302.13972](https://arxiv.org/abs/2302.13972)


### CONTENTS

#### Main folder

Core library:
- **data_misc.py** (functions to process and analyze empirical communication data)
- **model_misc.py** (functions to run and analyze model of ego network evolution)
- **plot_misc.py** (functions to plot statistical properties of data/model)

Sample analysis scripts:
- **script_getData.py** (script for getting data properties)
- **script_runModel.py** (script for running ego network model)

#### Figures folder

Each script corresponds to a figure in the main text and supplementary information.

#### Files folder

- **egonet_acts_\*.pkl** (# events per alter of each ego in dataset)
- **egonet_acts_pieces_\*.pkl** (same as egonet_acts_\*.pkl, but for each time period in persistence analysis)
- **egonet_props_\*.pkl** (ego network properties for each ego in dataset: degree, # events, mean alter activity, alter activity variance, and min/max alter activity)
- **egonet_props_pieces_\*.pkl** (same as egonet_props_\*.pkl, but for each time period in persistence analysis)
- **params_data.pkl** (basic statistics per dataset: # egos (unfiltered), # events, # egos (filtered), avg degree, avg strength, avg mean alter activity, avg alter activity variance, and avg min/max alter activity)
- **params_data_parallel.pkl** (same as params_data.pkl, but for large datasets)

#### Tables folder

Each script corresponds to a table in the supplementary information.

#### Data

For reproducibility purposes, this library includes processed versions of openly accesible datasets of temporal communication networks (more info on the supplementary information of our paper). Original datasets can be found in the following locations:

- **Short messages (Wu 1, 2, & 3)** Y. Wu, C. Zhou, J. Xiao, J. Kurths, and H. J. Schellnhuber. Evidence for a bimodal distribution in human communication, PNAS 107, 44, 18803–18808, 2010. https://doi.org/10.1073/pnas.1013140107

- **Emails (Enron)** J. Kunegis. KONECT: The Koblenz network collection, in Proc. Int. Conf. on World Wide Web Companion, 1343–1350, 2013. http://konect.cc/networks/enron/

- **Emails (EU)** A. Paranjape, A. R. Benson, and J. Leskovec. Motifs in temporal networks, in Proc. 10th ACM Int. Conf. on Web Search and Data Mining, 601–610, 2017. https://snap.stanford.edu/data/email-Eu-core-temporal.html

- **Online messages (Facebook)** B. Viswanath, A. Mislove, M. Cha, and K. P. Gummadi. On the evolution of user interaction in Facebook, in Proceedings of the 2nd ACM workshop on Online social networks, 37–42, 2009. https://socialnetworks.mpi-sws.org/data-wosn2009.html

- **Online messages (College)** T. Opsahl and P. Panzarasa. Clustering in weighted networks, Soc. Net., 31, 2, 155–163, 2009. https://toreopsahl.com/datasets/#online_social_network

- **Copenhagen Networks Study (CNS call & sms)** P. Sapiezynski, A. Stopczynski, D. D. Lassen, and S. Lehmann. Interaction data from the Copenhagen Networks Study, Sci. Data, 6, 1, 1–10, 2019. https://doi.org/10.6084/m9.figshare.7267433
