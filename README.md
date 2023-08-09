# Farsignatures

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
- **egonet_acts_pieces_\*.pkl** (same as egonet_acts_\*.pkl, but for big datasets separated into pieces)
- **egonet_props_\*.pkl** (ego network properties for each ego in dataset: degree, # events, mean alter activity, alter activity variance, and min/max alter activity)
- **egonet_props_pieces_\*.pkl** (same as egonet_props_\*.pkl, but for big datasets separated into pieces)
- **params_data.pkl** (basic statistics per dataset: # egos (unfiltered), # events, # egos (filtered), avg degree, avg strength, avg mean alter activity, avg alter activity variance, and avg min/max alter activity)
- **params_data_parallel.pkl** (same as params_data.pkl, but for big datasets separated into pieces)

#### Tables folder

Each script corresponds to a table in the supplementary information.
