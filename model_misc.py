#! /usr/bin/env python

### MODULE FOR MISCELLANEOUS FUNCTIONS FOR MODEL IN FARSIGNATURES PROJECT ###

#import modules
import numpy as np
import pandas as pd
import scipy.stats as ss
import scipy.special as sps
import scipy.optimize as spo
from mpmath import mp


## FUNCTIONS ##

#function to compute activity distribution (theo)
def activity_dist( a, t, alpha, a0 ):
	"""Compute activity distribution (theo)"""

	if t > a0: #non-trivial case
		#variables as high-accuracy input
		t = mp.mpf(str( t ))
		alpha = mp.mpf(str( alpha ))

		#Beta factor
		if a > a0: #large activity
			ca = 1 / ( ( a-a0 ) * mp.power( alpha+a0, a-a0 ) * mp.beta( a-a0, alpha+a0 ) )
		elif a == a0: #minimum activity
			ca = 1.
		else: #zero activity
			ca = 0.

		#activity distribution
		pa = float(mp.re( ca * mp.power( t-a0, a-a0 ) * mp.power( ( t+alpha ) / ( a0+alpha ), -(a + alpha) ) ))

	else: #trivial case
		pa = 1. if a == a0 else 0. #delta function at initial condition

	return pa

#function to compute activity distribution (gamma approx)
def activity_dist_gamma( a, t, alpha, a0 ):
	"""Compute activity distribution (gamma approx)"""

	gamma = alpha + a0 #shape parameter
	beta = ( t - a0 ) / ( alpha + a0 ) #scale parameter

	pa = ss.gamma.pdf( a, gamma, a0, beta ) #rescaled gamma distribution

	return pa

#function to run model of alter activity, according to parameters
def model_activity( params, loadflag='n', saveloc='files/model/', saveflag=True, seed=None, print_every=10 ):
	"""Run model of alter activity, according to parameters"""

	#get model parameters
	alpha, a0, k, t, ntimes = params['alpha'], params['a0'], params['k'], params['t'], params['ntimes']

	#filename for output file
	savename = saveloc+'activity_alpha_{:.2f}_a0_{}_k_{}_t_{:.2f}_ntimes_{}.npy'.format( alpha, a0, k, t, ntimes )
	# savename = saveloc+'activity_alpha_{:.3f}_a0_{}_k_{}_t_{:.2f}_ntimes_{}.npy'.format( alpha, a0, k, t, ntimes )

	if loadflag == 'y': #load activity
		with open( savename, 'rb' ) as act_file:
			activity = np.load( act_file )

	elif loadflag == 'n': #or else, compute activity

		rng = np.random.default_rng(seed) #initialise random number generator

		time = np.arange( a0*k, k*t, dtype=int ) #time step array (without final time tau = k*t)

		#initialise array of activity per realization (row) per alter (column) (accumulated until tau = k*t)
		#run = 0, ..., ntimes-1, alter = 0, ..., k-1
		#initial condition: a_i(0) = a0 for all i
		activity = np.ones( ( ntimes, k ), dtype=int ) * a0

		#prob that alter w/ activity a_i(tau) gets active, over activity array of realisation
		prob_func = lambda act_rel, alpha, k, tau : ( act_rel + alpha ) / ( tau + k * alpha )
		#random alter chosen with previous probability
		alter_func = lambda act_rel, alpha, k, tau : rng.choice( k, p=prob_func( act_rel, alpha, k, tau ) )

		#activity dynamics
		for tau in time: #loop through time step tau = a0*k, ..., k*t-1

			t_inter = tau / float( k ) #get intermediate MC time (mean alter activity)
			if t_inter % print_every == 0:
				print( "\t\tt' = {:.2f}".format( t_inter ), flush=True ) #to know where we stand

			#get active alters for all realizations
			alters = np.apply_along_axis( alter_func, 1, activity, alpha, k, tau )

			#increase activity of chosen alters
			activity[ range( ntimes ), alters ] += 1

		if saveflag:
			with open( savename, 'wb' ) as act_file:
				np.save( act_file, activity ) #save activity

	return activity


#function to compute MLE optimal alpha from activity array
def alpha_MLE_fit( activity, bracket ):
	"""Compute MLE optimal alpha from activity array"""

	k = activity.size #degree
	tau = activity.sum() #strength (number of events)
	t = activity.mean() #mean alter activity
	a0 = activity.min() #min alter activity

	if k * a0 < tau: #solving alpha trascendental equation in non-trivial case
		#MLE functions
		digamma_avg = lambda alpha : sps.digamma( alpha+activity ).mean() - sps.digamma( alpha+a0 )
		beta = lambda alpha : ( t - a0 ) / ( alpha + a0 )
		alpha_func = lambda alpha : digamma_avg(alpha) - np.log( 1 + beta(alpha) )

		#look for alpha root numerically (within given bounds via bracket method)
		if np.sign(alpha_func(bracket[0])) != np.sign(alpha_func(bracket[1])):
			#if bracket limits have opposite signs
			alpha_res = spo.root_scalar( alpha_func, bracket=bracket )
			alpha = alpha_res.root
		else:
			alpha = bracket[1] #else, root diverges to infinity

	else: #if we cannot do MLE
		alpha = np.nan

	return alpha


#function to get MLE optimal alpha (optional) and test statistics from activity array
def alpha_stats( activity, alpha=None, alpha_bounds=(1e-4, 1e3) ):
	"""Get MLE optimal alpha (optional) and test statistics from activity array"""

	k = activity.size #degree
	t = activity.mean() #mean alter activity
	a0 = activity.min() #min/max alter activity
	amax = activity.max()

	if alpha == None:
		alpha = alpha_MLE_fit( activity, ( -a0+alpha_bounds[0], alpha_bounds[1] ) ) #alpha fit

	#only consider alphas in non-trivial case t > a_0
	if np.isnan( alpha ) == False:

		a_vals = range(a0, amax+1) #activity values
		#cumulative dist of alter activity in range a=[a0, amax] (i.e. inclusive)
		act_cumdist = ss.cumfreq( activity, defaultreallimits=( a_vals[0]-0.5, a_vals[-1]+0.5 ), numbins=len(a_vals) ).cumcount / k

		#theo activity dist in range a=[a0, amax] (i.e. inclusive)
		act_dist_theo = np.array([ activity_dist( a, t, alpha, a0 ) for a in a_vals ])
		act_dist_theo /= act_dist_theo.sum() #normalise (due to finite activity range in data)
		#theo cumulative dist
		act_cumdist_theo = np.cumsum( act_dist_theo )
		#related quantities: difference between cum dists, its average, and theo cum dist product
		cumdist_diff = act_cumdist - act_cumdist_theo
		cumdist_diff_avg = ( cumdist_diff * act_dist_theo ).sum()
		cumdist_theo_prod = act_cumdist_theo * ( 1 - act_cumdist_theo )
		#fix (small) numerical errors
		cumdist_diff[-1], cumdist_theo_prod[-1] = 0., 0.

		#Kolmogorov-Smirnov and Cramer-von Mises statistics
		KS = np.abs( cumdist_diff ).max() #Kolmogorov-Smirnov
		W2 = k*( cumdist_diff**2 * act_dist_theo ).sum() #Cramer-von Mises
		U2 = k*( ( cumdist_diff - cumdist_diff_avg )**2 * act_dist_theo ).sum() #Watson
		A2 = k*( cumdist_diff[:-1]**2 * act_dist_theo[:-1] / cumdist_theo_prod[:-1] ).sum() #Anderson-Darling (last term is 0/0=0)

	else:
		KS, W2, U2, A2 = np.nan, np.nan, np.nan, np.nan

	return alpha, [KS, W2, U2, A2]


#function to compute MLE optimal alpha from activity array (MP implementation)
def alpha_MLE_fit_MP( activity, bounds ):
	"""Compute MLE optimal alpha from activity array (MP implementation)"""

	a0 = min( activity ) #minimum alter activity
	t = np.mean( activity ) #mean alter activity

	if t > a0: #solving alpha trascendental equation in non-trivial case
		t = mp.mpf(str( t )) #turn to high precision

		digamma_avg = lambda alpha, a0, activity : mp.exp( mp.fsum([ mp.digamma( alpha + a ) - mp.digamma( alpha+a0 ) for a in activity ]) / len( activity ) )

		alpha_func = lambda alpha, a0, t, activity : ( t - a0 * digamma_avg( alpha, a0, activity ) ) / ( digamma_avg( alpha, a0, activity ) - 1 )

		alpha_min = lambda alpha, a0, t, activity : float( mp.fabs( alpha - alpha_func( alpha, a0, t, activity ) ) )

		#look for optimal alpha numerically (within given bounds)
		alpha_res = spo.minimize_scalar( alpha_min, args=( a0, t, activity ), bounds=bounds, method='bounded' )

		alpha = alpha_res.x

	else: #if we cannot do MLE
		alpha = np.nan

	return alpha


#function to get MLE optimal alpha and KS statistic from activity array
def alpha_KSstat( activity, alphamax=1000 ):
	"""Get MLE optimal alpha and KS statistic from activity array"""

	k = len(activity) #degree
	t = np.mean(activity) #mean alter activity
	a0 = min(activity) #min/max alter activity
	amax = max(activity)

	#cumulative dist of alter activity in range a=[0, amax] (i.e. inclusive)
	act_cumdist = ss.cumfreq( activity, defaultreallimits=( -0.5, amax+0.5 ), numbins=amax+1 ).cumcount / k

	#bounds for alpha search
	bounds = ( -a0, alphamax )

	alpha = alpha_MLE_fit( activity, bounds ) #alpha fit

	#only consider alphas in non-trivial case t > a_0
	if np.isnan( alpha ) == False:
		#theo activity dist in range a=[0, amax] (i.e. inclusive)
		act_dist_theo = np.array([ activity_dist( a, t, alpha, a0 ) for a in range(amax+1) ])
		act_dist_theo /= act_dist_theo.sum() #normalise (due to finite activity range in data)
		#cumulative dist
		act_cumdist_theo = np.cumsum( act_dist_theo )

		#KS statistic
		KSstat = np.abs( act_cumdist - act_cumdist_theo ).max()

	else:
		KSstat = np.nan

	return alpha, KSstat


#function to get optimal gamma and KS statistic from activity array
def gamma_KSstat( activity ):
	"""Get optimal gamma and KS statistic from activity array"""

	a0 = min(activity) #minimum alter activity

	#filtered alter activity (a > a0)
	activity_noa0 = activity[ activity > a0 ] #alter activity a > a0
	k_noa0 = activity_noa0.size #degree
	tau_noa0 = activity_noa0.sum() #strength (number of events)
	t_noa0 = activity_noa0.mean() #mean alter activity
	amin_noa0 = activity_noa0.min() #min alter activity

	#closed-form (biased) gamma/beta estimators
	gamma_bias = k_noa0 * ( t_noa0 - a0 ) / ( ( activity_noa0 - t_noa0 ) * np.log( activity_noa0 - a0 ) ).sum()
	beta_bias = ( t_noa0 - a0 ) / gamma_bias

	#small-sample bias corrections
	gamma = gamma_bias - ( 3*gamma_bias - (2/3.)*( gamma_bias / (1 + gamma_bias) ) - (4/5.) * gamma_bias / (1 + gamma_bias)**2 ) / k_noa0
	beta = beta_bias * k_noa0 / ( k_noa0 - 1 )

	#KS statistic
	KSstat, KSpval = ss.ks_1samp( activity_noa0, ss.gamma.cdf, args=( gamma, a0, beta ) )

	return gamma, gamma_bias, beta, beta_bias, KSstat, KSpval


#DEBUGGIN'

# #function to run (slow) model of alter activity, according to parameters
# def model_activity_slow( params, loadflag='n', saveloc='files/' ):
# 	"""Run (slow) model of alter activity, according to parameters"""
#
# 	#get model parameters
# 	alpha, k, T, ntimes = params['alpha'], params['k'], params['T'], params['ntimes']
#
# 	#filename for output file
# 	savename = saveloc+'activity_alpha_{:.2f}_k_{}_T_{:.2f}_ntimes_{}.pkl'.format( alpha, k, T, ntimes )
#
# 	if loadflag == 'y': #load activity
# 		activity = pd.read_pickle( savename )
#
# 	elif loadflag == 'n': #or else, compute activity
#
# 		rng = np.random.default_rng() #initialise random number generator
#
# 		time = np.arange( 0, k*T, dtype=int ) #time step array (without final time tau = kT)
#
# 		#initialise dataframe of activity per realization (row) per alter (column) (accumulated until tau = kT)
# 		#run = 0, ..., ntimes-1, alter = 0, ..., k-1
# 		#initial condition: a_i(0) = 0 for all i
# 		activity = pd.DataFrame( np.zeros(( ntimes, k )), index=pd.Series( range(ntimes), name='run' ), columns=pd.Series( range(k), name='alter' ), dtype=int )
#
# 		#activity dynamics
# 		for tau in time: #loop through time step tau = 0, ..., kT-1
# 			t = tau / float( k ) #get MC time (mean alter activity)
#
# 			if t % 1 == 0:
# 				print( 't = {:.2f}'.format( t ) ) #to know where we stand
#
# 			p_rand = alpha / ( alpha + t ) #probability of choosing random alter
#
# 			for nt in range(ntimes): #loop through relaizations
#
# 				#compute probability of picking alters
# 				if rng.random() < p_rand: #random alter
# 					alter_probs = np.ones( k ) / k
# 				else: #cumulative advantage
# 					alter_probs = activity.iloc[ nt ] / float( tau )
#
# 				alter = rng.choice( range(k), p=alter_probs ) #choose alter with uniform/CA probability
#
# 				activity.iloc[ nt, alter ] += 1 #add activity event!
#
# 		activity.to_pickle( savename ) #save activity
#
# 	return activity

# #function to compute activity distribution for large tau (theo)
# def activity_dist_large_tau( a, alpha ):
# 	"""Compute activity distribution for large_tau (theo)"""
#
# 	return mp.exp( -alpha ) * alpha / ( a + alpha )

#		alter_pool = {}
#		alter_pool[ 'rand' ] = np.tile( np.arange(k, dtype=int), (ntimes, 1) )
#		alter_pool[ 'CA' ] = - np.ones( ( ntimes, len(time) ), dtype=int )

# #function to compute activity distribution for large tau (theo)
# def activity_dist_large_tau( a, tau, alpha, k ):
# 	"""Compute activity distribution for large_tau (theo)"""
#
# 	return mp.power( 1 - alpha / ( tau + k * alpha ), tau ) * alpha / ( a + alpha )
#
# #function to compute activity distribution for fixed t (theo)
# def activity_dist_fixed_t( a, t, alpha ):
# 	"""Compute activity distribution for fixed t (theo)"""
#
# 	##normalization constant
# 	if a > 0: #nonzero activity
# 		ca = 1 / ( a * mp.power( alpha, a ) * mp.beta( a, alpha ) )
# 	else: #zero activity
# 		ca = mp.mpf( 1 )
#
# 	return ca * mp.power( t, a ) * mp.power( 1 + t/alpha, -(a + alpha) )

# #function to compute MLE optimal alpha from empirical activity array
# def alpha_MLE_fit( activity, bounds=(0, 1000) ):
# 	"""Compute MLE optimal alpha from empirical activity array"""
#
# 	t = mp.mpf(str(np.mean( activity ))) #mean alter activity
#
# 	#solving alpha trascendental equation
# 	digamma_avg = lambda alpha, activity : mp.fsum([ mp.digamma( alpha + a ) - mp.digamma( alpha ) for a in activity ]) / len( activity )
# 	alpha_func = lambda alpha, t, activity : t / ( mp.exp( digamma_avg( alpha, activity ) ) - 1 )
# 	alpha_min = lambda alpha, t, activity : float( mp.fabs( alpha - alpha_func( alpha, t, activity ) ) )
#
# 	#look for optimal alpha numerically (within given bounds)
# 	alpha_res = spo.minimize_scalar( alpha_min, args=( t, activity ), bounds=bounds, method='bounded' )
#
# 	return alpha_res.x

		# #initialise pool from where to select alters at each time step (default value = -1)
		# #axis 0: 0 (random choice) and 1 (cumulative advantage [CA])
		# #axis 1: run = 0, ..., ntimes-1
		# #axis 2: alter = 0, ..., max between degree k and max tau = kT
		# alter_pool = - np.ones( ( 2, ntimes, max([ k, len(time) ]) ), dtype=int )
		# #initialise random choice as all alters
		# alter_pool[ 0, :, :k ] = np.tile( np.arange( k, dtype=int ), ( ntimes, 1 ) )
		# #initialise CA as all alters with a0 events each
		# alter_pool[ 1, :, :a0*k ] = np.tile( np.repeat( np.arange( k, dtype=int ), a0 ), ( ntimes, 1 ) )

		# time = np.arange( 0, k*T, dtype=int ) #time step array (without final time tau = kT)
		#
		# #initialise array of activity per realization (row) per alter (column) (accumulated until tau = kT)
		# #run = 0, ..., ntimes-1, alter = 0, ..., k-1
		# #initial condition: a_i(0) = 0 for all i
		# activity = np.zeros( ( ntimes, k ), dtype=int )
		#
		# #initialise pool from where to select alters at each time step (default value = -1)
		# #axis 0: 0 (random choice) and 1 (cumulative advantage [CA])
		# #axis 1: run = 0, ..., ntimes-1
		# #axis 2: alter = 0, ..., max between degree k and max tau = kT
		# alter_pool = - np.ones( ( 2, ntimes, max([ k, len(time) ]) ), dtype=int )
		# alter_pool[ 0, :, :k ] = np.tile( np.arange( k, dtype=int ), ( ntimes, 1 ) )
		#
		# #activity dynamics
		# for tau in time: #loop through time step tau = 0, ..., kT-1
		# 	t = tau / float( k ) #get MC time (mean alter activity)
		#
		# 	if t % 100 == 0:
		# 		print( 't = {:.2f}'.format( t ) ) #to know where we stand
		#
		# 	#probability of choosing random alter
		# 	p_rand = alpha / ( alpha + t )
		#
		# 	#maximum length of pools (random and CA)
		# 	pool_lens = np.array([ k, tau ])
		#
		# 	#mechanisms with which we'll choose alters ( random [0] or CA[1] )
		# 	rand_mechs = rng.choice( 2, size=ntimes, p=[ p_rand, 1 - p_rand ] )
		#
		# 	#positions of alters (chosen with given raandom mechanisms)
		# 	alter_pos = rng.integers( pool_lens[ rand_mechs ] )
		#
		# 	#alters chosen
		# 	alters = alter_pool[ rand_mechs, range(ntimes), alter_pos ]
		#
		# 	#increase activity of chosen alters
		# 	activity[ range( ntimes ), alters ] += 1
		#
		# 	#save alters with nonzero activity in CA alter pool
		# 	alter_pool[ 1, range(ntimes), tau ] = alters
