#! /usr/bin/env python

### SCRIPT FOR RUNNING MODEL IN FARSIGNATURES PROJECT ###

#import modules
import model_misc as mm


## RUNNING MODEL SCRIPT ##

if __name__ == "__main__":

	## CONF ##

	#parameter arrays
	alpha_vals = [ 0.01, 1, 100 ] #cumulative advantage parameter
	T_vals = [ 1000 ] #max time (mean alter activity) in dynamics

	#parameter dict
	params = {}
	params['k'] = 100 #number of alters (ego's degree)
	params['ntimes'] = 10000 #number of realizations for averages

	#flags and locations
	loadflag = 'n'
	saveloc = 'files/model/' #location of output files

	## MODEL ##

	for alpha in alpha_vals: #loop through alpha values
		params['alpha'] = alpha #CA parameter

		print( '\talpha = {:.2f}'.format( alpha ) ) #to know where we stand

		for T in T_vals: #loop through times
			params['T'] = T #time to run dynamics

			print( '\t\tT = {:.2f}'.format( T ) ) #to know where we stand

			#run model of alter activity, according to parameters
			activity = mm.model_activity( params, loadflag=loadflag, saveloc=saveloc )
