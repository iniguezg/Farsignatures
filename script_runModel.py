#! /usr/bin/env python

### SCRIPT FOR RUNNING MODEL IN FARSIGNATURES PROJECT ###

#import modules
import sys

import model_misc as mm


## RUNNING MODEL SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#parameters
	a0 = 1 #minimum alter activity
	k = 100 #number of alters (ego's degree)
	ntimes = 10000 #number of realizations for averages

	#parameter arrays
	# alpha_vals = [ 0., 99., 999. ] #PA parameter
	# t_vals = [ 2., 10., 100., 1000. ] #mean alter activity (max time in dynamics)
	alpha_vals = [ float( sys.argv[1] ) ]
	t_vals = [ float( sys.argv[2] ) ]

	#parameter dict
	params = { 'a0' : a0, 'k' : k, 'ntimes' : ntimes }

	#flags and locations
	loadflag = 'n'
	saveloc = 'files/model/' #location of output files

	## MODEL ##

	for alpha in alpha_vals: #loop through alpha values
		params['alpha'] = alpha #PA parameter

		print( 'alpha = {:.2f}'.format( alpha ), flush=True ) #to know where we stand

		for t in t_vals: #loop through times
			params['t'] = t #mean alter activity (max time in dynamics)

			print( '\tt = {:.2f}'.format( t ), flush=True ) #to know where we stand

			#run model of alter activity, according to parameters
			activity = mm.model_activity( params, loadflag=loadflag, saveloc=saveloc )
