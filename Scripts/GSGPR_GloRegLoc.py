import os
import sys
import time
from time import gmtime, strftime
import numpy
import scipy 
from scipy import optimize

from numpy import matrix
from numpy import genfromtxt
from numpy import linalg

import matplotlib
import matplotlib.pyplot as plt

print (time.strftime("%H:%M:%S"))

backslash = '\\'

wkspc = str(os.getcwd()).replace(backslash,"/") + "/"

##Covariance functions

def matern(sigma,numer,denom,data):
	
	if numer == 5:
		ret_val = (sigma**2.) * (1.+ ((numpy.sqrt(5.) * (data))/denom) + ((5. * (data**2))/(3.*(denom**2.)))) *  numpy.exp(-1.*((numpy.sqrt(5.) * (data))/denom)) 
	if numer == 3:
		ret_val = (sigma**2.) * (1.+((numpy.sqrt(3.) * (data))/denom)) *  numpy.exp(-1.*((numpy.sqrt(3.) * (data))/denom)) 
	if numer == 1:
		ret_val = (sigma**2.) * numpy.exp(-1.*((1. * (data))/denom))
	return ret_val 
	
def linear_G(sigma1,sigma2,data):
	
	ret_val = (sigma1**2.) + ((sigma2**2.) * data)
	
	return ret_val 		
	
def linear_RL(sigma1,sigma2,data,s_mat1):
	
	ret_val = ((sigma1**2.) * s_mat1) + (((sigma2**2.) * s_mat1) * data)
	
	return ret_val 		

def WHE_NSE(hyp1_in,hyp2_in,t_matrix_i,noise_mat_in,s_mat1,s_mat2):
	
	return (noise_mat_in) + ((s_mat1*(numpy.identity(len(t_matrix_i[:,0])))) * (hyp1_in **2.)) +  ((s_mat2*(numpy.identity(len(t_matrix_i[:,0])))) * (hyp2_in **2.))

#data_series is input data [time,value,error,ID1 Regional or Global ID,ID2 Local ID]
#reg_names is the location names relative to ID1 [ID1,region name]
#loc_names is the location names relative to ID2 [ID2,location name]
#glob_ID1 is the 'regional' identifier of the global dataseries
#glob_ID2 is the 'local' identifier of the global dataseries
#guess_orig is 14 item vector - original guess for parameters
#stepsize_divisor is a single positive floating point number, 10-100 is reasonable.
#f_ev_iters is number of function evaluations for Nelder-Mead optimization process
#MCMC_iters is number of  iterations for MCMC process, 2000-10000 is reasonable
#new_dt is sample rate of plotted functions
#fit indicates whether to refit parameters: 1 is fit, 2 is use guess_orig

def GP_GloRegLoc(data_series,reg_names,loc_names,glob_ID1,glob_ID2,guess_orig,f_ev_iters,MCMC_iters,stepsize_divisor,new_dt,fit):
	global count
	xes1_t_1 = data_series[:,0]
	xes2_sl_1 = data_series[:,1]
	xes4_error_1 = data_series[:,2]
	xes3_type_1 = data_series[:,3]
	xes5_type_2 = data_series[:,4]

	n_SL_1 = len(xes1_t_1)

	#distances - preallocated memory

	y_1 = numpy.reshape(xes2_sl_1,(-1,1))	
	y_transpose_1 = numpy.transpose(y_1)

	noise_mat_1 = (1.*(numpy.identity(n_SL_1)))

	for index1 in range(0,n_SL_1):
		noise_mat_1[index1,index1] = noise_mat_1[index1,index1] * (xes4_error_1[index1]**2.)

	t_matrix_1 = numpy.sqrt((numpy.repeat(numpy.reshape(xes1_t_1,(1,-1)),n_SL_1,axis=0) - numpy.repeat(numpy.reshape(xes1_t_1,(-1,1)),n_SL_1,axis=1))**2.)	

	#Regional matern

	s_matrix_1_tmp = numpy.repeat(numpy.reshape(xes3_type_1,(1,-1)),n_SL_1,axis=0)
	s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes3_type_1,(-1,1)),n_SL_1,axis=1)

	w1_tmp = (s_matrix_1_tmp != glob_ID1) & (s_matrix_2_tmp != glob_ID1)
	w1 = numpy.where((s_matrix_1_tmp == s_matrix_2_tmp) & (w1_tmp))

	s_matrix_1 = numpy.zeros(numpy.shape(s_matrix_1_tmp))
	s_matrix_1[w1[0],w1[1]] = s_matrix_1[w1[0],w1[1]] + 1.0

	#Local matern
	
	s_matrix_1_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(1,-1)),n_SL_1,axis=0)
	s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(-1,1)),n_SL_1,axis=1)

	w1_tmp = (s_matrix_1_tmp != glob_ID2) & (s_matrix_2_tmp != glob_ID2)
	w1 = numpy.where((s_matrix_1_tmp == s_matrix_2_tmp) & (w1_tmp))

	s_matrix_2 = numpy.zeros(numpy.shape(s_matrix_1_tmp))
	s_matrix_2[w1[0],w1[1]] = s_matrix_2[w1[0],w1[1]] + 1.0

	###Error indicators

	s_matrix_1_tmp = numpy.repeat(numpy.reshape(xes3_type_1,(1,-1)),n_SL_1,axis=0)
	s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes3_type_1,(-1,1)),n_SL_1,axis=1)

	w1_tmp = (s_matrix_1_tmp==glob_ID1) & (s_matrix_2_tmp==glob_ID1)
	w1 = numpy.where((s_matrix_1_tmp == s_matrix_2_tmp) & (w1_tmp))

	s_matrix_3 = numpy.zeros(numpy.shape(s_matrix_1_tmp))
	s_matrix_3[w1[0],w1[1]] = s_matrix_3[w1[0],w1[1]] + 1.0

	w1_tmp = (s_matrix_1_tmp!=glob_ID1) & (s_matrix_2_tmp!=glob_ID1)
	w1 = numpy.where((s_matrix_1_tmp == s_matrix_2_tmp) & (w1_tmp))

	s_matrix_4 = numpy.zeros(numpy.shape(s_matrix_1_tmp))
	s_matrix_4[w1[0],w1[1]] = s_matrix_4[w1[0],w1[1]] + 1.0

	new_mults_t_1 = numpy.repeat(numpy.reshape(xes1_t_1,(1,-1)),n_SL_1,axis=0) * numpy.repeat(numpy.reshape(xes1_t_1,(-1,1)),n_SL_1,axis=1)
	count = 0
	if fit == 1:
		t1 = float(time.time())
		def optimize_MLE_merge_guess_f(guess1):
			global count
			K1 =  matern(guess1[0],3.,guess1[1],t_matrix_1)
				
			K2 =  matern(guess1[2],3.,guess1[3],t_matrix_1) * s_matrix_1
			
			K3 =  matern(guess1[4],1.,guess1[5],t_matrix_1) * s_matrix_2
			
			#K4 =  matern(guess1[6],1.,guess1[7],t_matrix_1) * s_matrix_2
			
			K5 =  linear_G(guess1[6],guess1[7],new_mults_t_1)
			
			K6 =  linear_RL(guess1[8],guess1[9],new_mults_t_1,s_matrix_1)
			
			K7 =  linear_RL(guess1[10],guess1[11],new_mults_t_1,s_matrix_2)
			
			WN = WHE_NSE(guess1[12],guess1[13],t_matrix_1,noise_mat_1,s_matrix_3,s_matrix_4)
			
			K_1 = K1 + K2 + K3 + K5 + K6 + K7 + WN  #+ K4
			
			K_inv = numpy.linalg.inv(K_1)
			matmul_tmp = numpy.matmul(y_transpose_1,K_inv)
			term1 = (-1./2.) * numpy.matmul(matmul_tmp,y_1)
			det_k = numpy.linalg.slogdet(K_1)[1]
			term2 = (1./2.) * det_k
			term3 = (float(len(y_1))/2.) * numpy.log(2.*numpy.pi)
			opt_outs_GPR_1 = term1 - term2 - term3
			
			opt_outs_GPR = opt_outs_GPR_1[0]
			
			if opt_outs_GPR > 0:
				opt_outs_GPR = numpy.nan
			
			opt_outs_GPR = opt_outs_GPR * -1.
			
			count = count + 1
			if count % 500 == 0 or count==10:
				t2 = float(time.time())
				time_per = (t2 - t1) / (float(count))
				print ("")
				print ("Count:",count)
				#print("Diff. evo. params tmp:",guess1)
				print("Nelder-Mead params:",guess1)
				#print("Diff. evo. loglik tmp:",opt_outs_GPR)
				print("Nelder-Mead loglik:",opt_outs_GPR)
				print("Time (s) per iteration:",round(time_per,2))
				print("Max time (m) remaining:",round(((time_per*(f_ev_iters-count))+(time_per*MCMC_iters))/60.,2))
				print ("")
			return opt_outs_GPR
			
		def optimize_MLE_merge_guess(guess1):

			K1 =  matern(guess1[0],3.,guess1[1],t_matrix_1)
				
			K2 =  matern(guess1[2],3.,guess1[3],t_matrix_1) * s_matrix_1
			
			K3 =  matern(guess1[4],1.,guess1[5],t_matrix_1) * s_matrix_2
			
			#K4 =  matern(guess1[6],1.,guess1[7],t_matrix_1) * s_matrix_2
			
			K5 =  linear_G(guess1[6],guess1[7],new_mults_t_1)
			
			K6 =  linear_RL(guess1[8],guess1[9],new_mults_t_1,s_matrix_1)
			
			K7 =  linear_RL(guess1[10],guess1[11],new_mults_t_1,s_matrix_2)
			
			WN = WHE_NSE(guess1[12],guess1[13],t_matrix_1,noise_mat_1,s_matrix_3,s_matrix_4)
			
			K_1 = K1 + K2 + K3 + K5 + K6 + K7 + WN #+ K4
			
			K_inv = numpy.linalg.inv(K_1)
			matmul_tmp = numpy.matmul(y_transpose_1,K_inv)
			term1 = (-1./2.) * numpy.matmul(matmul_tmp,y_1)
			det_k = numpy.linalg.slogdet(K_1)[1]
			term2 = (1./2.) * det_k
			term3 = (float(len(y_1))/2.) * numpy.log(2.*numpy.pi)
			opt_outs_GPR_1 = term1 - term2 - term3
			
			opt_outs_GPR = opt_outs_GPR_1[0]
			
			if opt_outs_GPR > 0:
				opt_outs_GPR = numpy.nan
			
			opt_outs_GPR = opt_outs_GPR * -1.

			return opt_outs_GPR		

		'''
		bounds1 = scipy.optimize.Bounds(lb=0., ub=numpy.inf, keep_feasible=False)
		'''
		stepsizes = guess_orig/stepsize_divisor
		
		unique_l1 = numpy.unique(xes5_type_2)
		
		out_tmp = numpy.zeros(len(unique_l1))
		for n in range(0,len(unique_l1)):
			
			w1 = numpy.where(xes5_type_2==unique_l1[n])[0]
			w2 = numpy.argsort(xes1_t_1[w1])
			out_tmp[n] = numpy.quantile(xes1_t_1[w1[w2[1:]]] - xes1_t_1[w1[w2[0:-1]]],0.5)
		median_dt = numpy.quantile(out_tmp,0.5)
		
		lb1 = guess_orig*0.
		ub1= lb1 + numpy.inf
		
		bounds2 = []
		for n in range(0,len(ub1)):
			if n == 1 or n == 3:	
				bounds2.append([lb1[n],numpy.max(xes1_t_1)-numpy.min(xes1_t_1)],)
			elif n == 5:
				bounds2.append([lb1[n],5.*median_dt],)
			else:
				bounds2.append([lb1[n],ub1[n]],)
		
		bounds2 = tuple(bounds2)
		
		print("Original:",guess_orig)
		
		res = scipy.optimize.minimize(optimize_MLE_merge_guess_f,guess_orig,method='Nelder-Mead',bounds=bounds2,tol=10e-6,options={'maxfev':f_ev_iters})
		#res = scipy.optimize.differential_evolution(optimize_MLE_merge_guess_f,bounds2,x0=guess_orig,maxiter=1000)
		
		guess_orig2 = res.x
		#guess_orig2 = guess_orig*1.
		
		print ("Nelder-Mead:",guess_orig2)
		
		old_alpha = numpy.absolute(guess_orig2)
		new_alpha = old_alpha * 1.0
		
		output_matrix_A = numpy.zeros((MCMC_iters,len(guess_orig2)))* numpy.nan

		loglik_output = numpy.zeros((MCMC_iters,2))
		accept_output = numpy.zeros((MCMC_iters,2)) * numpy.nan

		# Metropolis-Hastings

		t1 = float(time.time())
		index_to_change = 0
		for n in range(MCMC_iters):

			if (n+1) % 500 == 0 or n+1 == 10:
			
				t2 = float(time.time())
				time_per = (t2 - t1) / (float(n+1))
				accept_output[n,1] = time_per
				print ("")
				print ("MCMC count: ", n+1)
				print ("MCMC accept rate: ", numpy.mean(accept_output[0:n,0]))
				print ("MCMC params:",old_alpha)
				print ("MCMC loglik: ",old_loglik)
				print ("Time (s) per iteration: ",round(time_per,2))
				print ("Approx. time (m) remaining:",round((time_per*(MCMC_iters-float(n+1)))/60.,2))					
				print ("")

			if n > 0:
				old_alpha  = output_matrix_A[n-1,:]
				old_loglik = loglik_output[n-1,0]

			new_alpha[index_to_change] = numpy.absolute(numpy.random.normal(loc = old_alpha[index_to_change], scale = stepsizes[index_to_change]))
			index_to_change = index_to_change + 1

			if index_to_change == len(new_alpha):
				index_to_change = 0		
			
			new_loglik = optimize_MLE_merge_guess(new_alpha)
			
			if n == 0:
				old_loglik = new_loglik * .9999999999999
				
			if numpy.isnan(new_loglik) == False:
				if (new_loglik < old_loglik):
					output_matrix_A[n,:] = new_alpha
					loglik_output[n,0] = new_loglik
					accept_output[n,0] = 1.0

				else:			
					u = numpy.random.uniform(0.0,1.0)
					
					if (u < numpy.exp(old_loglik - new_loglik)):
						output_matrix_A[n,:] = new_alpha				
						loglik_output[n,0] = new_loglik
						accept_output[n,0] = 1.0

					else:
						output_matrix_A[n,:] = old_alpha
						loglik_output[n,0] = old_loglik
						accept_output[n,0] = 0.0

			else:
				output_matrix_A[n,:] = old_alpha
				loglik_output[n,0] = old_loglik
				accept_output[n,0] = 0.0

		w1 = numpy.argsort(loglik_output[:,0])[int(len(loglik_output[:,0])/2)]
		
		print(loglik_output[w1,0])
		
		MAP_est = output_matrix_A[w1,:]
		
		print ("MAP:",MAP_est)
	
	if fit == 2:
		MAP_est = guess_orig*1.
		
	########################################################################
	########################################################################
	########################################################################
	########################################################################
	########################################################################
	
	unique_values1 = numpy.unique(xes3_type_1)
	unique_values2 = numpy.unique(xes5_type_2)
	
	t_vals_tmp = numpy.arange(numpy.min(xes1_t_1),numpy.max(xes1_t_1)+ new_dt,new_dt)
	s_vals1_tmp = t_vals_tmp * 0.0
	s_vals2_tmp = t_vals_tmp * 0.0
	
	for n in range(0,len(unique_values2)):
		
		w_ind2 = numpy.where(xes5_type_2==unique_values2[n])[0]
		ind1_tmp = xes3_type_1[w_ind2[0]]
		ind2_tmp = unique_values2[n]
		s_vals1_tmp2 = s_vals1_tmp + ind1_tmp
		s_vals2_tmp2 = s_vals2_tmp + ind2_tmp	
		
		if n==0:
			
			t_vals = t_vals_tmp * 1.0
			s_vals1 = s_vals1_tmp2 * 1.0
			s_vals2 = s_vals2_tmp2 * 1.0
		
		if n>0:
			
			t_vals = numpy.append(t_vals,t_vals_tmp)
			s_vals1 = numpy.append(s_vals1,s_vals1_tmp2)
			s_vals2 = numpy.append(s_vals2,s_vals2_tmp2) 
	

	###
	
	m1 =len(t_vals)
	
	t_new_1 = numpy.sqrt((numpy.repeat(numpy.reshape(t_vals,(1,-1)),len(xes1_t_1),axis=0) - numpy.repeat(numpy.reshape(xes1_t_1,(-1,1)),len(t_vals),axis=1))**2.)
	t_mults_new_1 = numpy.repeat(numpy.reshape(t_vals,(1,-1)),len(xes1_t_1),axis=0) * numpy.repeat(numpy.reshape(xes1_t_1,(-1,1)),len(t_vals),axis=1)

	#Regional matern

	s_matrix_1_tmp = numpy.repeat(numpy.reshape(s_vals1,(1,-1)),len(xes1_t_1),axis=0)
	s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes3_type_1,(-1,1)),len(s_vals1),axis=1)

	w1_tmp = (s_matrix_1_tmp != glob_ID1) & (s_matrix_2_tmp != glob_ID1)
	w1 = numpy.where((s_matrix_1_tmp == s_matrix_2_tmp) & (w1_tmp))

	s_matrix_1_b = numpy.zeros(numpy.shape(s_matrix_1_tmp))
	s_matrix_1_b[w1[0],w1[1]] = s_matrix_1_b[w1[0],w1[1]] + 1.0
	
	#Local matern
	
	s_matrix_1_tmp = numpy.repeat(numpy.reshape(s_vals2,(1,-1)),len(xes1_t_1),axis=0)
	s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(-1,1)),len(s_vals1),axis=1)

	w1_tmp = (s_matrix_1_tmp != glob_ID2) & (s_matrix_2_tmp != glob_ID2)
	w1 = numpy.where((s_matrix_1_tmp == s_matrix_2_tmp) & (w1_tmp))

	s_matrix_2_b = numpy.zeros(numpy.shape(s_matrix_1_tmp))
	s_matrix_2_b[w1[0],w1[1]] = s_matrix_2_b[w1[0],w1[1]] + 1.0

	###

	t_matrix2_1 = numpy.sqrt((numpy.repeat(numpy.reshape(t_vals,(1,-1)),m1,axis=0) - numpy.repeat(numpy.reshape(t_vals,(-1,1)),m1,axis=1))**2.)
	t_matrix2_mult_1 = numpy.repeat(numpy.reshape(t_vals,(1,-1)),m1,axis=0) * numpy.repeat(numpy.reshape(t_vals,(-1,1)),m1,axis=1)

	#Regional matern

	s_matrix_1_tmp = numpy.repeat(numpy.reshape(s_vals1,(1,-1)),m1,axis=0)
	s_matrix_2_tmp = numpy.repeat(numpy.reshape(s_vals1,(-1,1)),m1,axis=1)

	w1_tmp = (s_matrix_1_tmp != glob_ID1) &  (s_matrix_2_tmp != glob_ID1)
	w1 = numpy.where((s_matrix_1_tmp == s_matrix_2_tmp) & (w1_tmp))

	s_matrix_1_c = numpy.zeros(numpy.shape(s_matrix_1_tmp))
	s_matrix_1_c[w1[0],w1[1]] = s_matrix_1_c[w1[0],w1[1]] + 1.0

	#Local matern

	s_matrix_1_tmp = numpy.repeat(numpy.reshape(s_vals2,(1,-1)),m1,axis=0)
	s_matrix_2_tmp = numpy.repeat(numpy.reshape(s_vals2,(-1,1)),m1,axis=1)

	w1_tmp = (s_matrix_1_tmp != glob_ID2) & (s_matrix_2_tmp != glob_ID2) 
	w1 = numpy.where((s_matrix_1_tmp == s_matrix_2_tmp) & (w1_tmp))

	s_matrix_2_c = numpy.zeros(numpy.shape(s_matrix_1_tmp))
	s_matrix_2_c[w1[0],w1[1]] = s_matrix_2_c[w1[0],w1[1]] + 1.0

	output_matrix = numpy.zeros((len(t_vals),5))
	
	output_matrix[:,0] = t_vals
	output_matrix[:,1] = s_vals1
	output_matrix[:,2] = s_vals2

	hyp1,hyp2,hyp3,hyp4,hyp5,hyp6,hyp7,hyp8,hyp9,hyp10,hyp11,hyp12,hyp13,hyp14 =  MAP_est

	################1################
	
	################ (x,x) ################
	
	K1_1 = matern(hyp1,3.,hyp2,t_matrix_1)
	
	K2_1 = matern(hyp3,3.,hyp4,t_matrix_1) * s_matrix_1
	
	K3_1 = matern(hyp5,1.,hyp6,t_matrix_1) * s_matrix_2	
	
	#K4_1 = matern(hyp7,1.,hyp8,t_matrix_1) * s_matrix_2

	K5_1 = linear_G(hyp7,hyp8,new_mults_t_1)
	
	K6_1 = linear_RL(hyp9,hyp10,new_mults_t_1,s_matrix_1)

	K7_1 = linear_RL(hyp11,hyp12,new_mults_t_1,s_matrix_2)	
	
	WN = WHE_NSE(hyp13,hyp14,t_matrix_1,noise_mat_1,s_matrix_3,s_matrix_4)

	K = K1_1 + K2_1 + K3_1 + K5_1 + K6_1 + K7_1 + WN #+ K4_1 
		
	K_inv = numpy.linalg.inv(K)
		
	################ (x,*) ################
	
	K1_2 =  matern(hyp1,3.,hyp2,t_new_1)
	
	K2_2 =  matern(hyp3,3.,hyp4,t_new_1) * s_matrix_1_b

	K3_2 =  matern(hyp5,1.,hyp6,t_new_1) * s_matrix_2_b	

	#K4_2 =  matern(hyp7,1.,hyp8,t_new_1) * s_matrix_2_b	
	
	K5_2 =  linear_G(hyp7,hyp8,t_mults_new_1)
	
	K6_2 =  linear_RL(hyp9,hyp10,t_mults_new_1,s_matrix_1_b)
	
	K7_2 =  linear_RL(hyp11,hyp12,t_mults_new_1,s_matrix_2_b)
	
	K2_f = K1_2 + K2_2 + K3_2 + K5_2 + K6_2 + K7_2 # + K4_2

	new_y_p = numpy.matmul(K_inv,K2_f) 
	new_y_p_2 = numpy.matmul(numpy.transpose(K2_f),new_y_p)

	################ (*,*) ################

	K1_3 =  matern(hyp1,3.,hyp2,t_matrix2_1)
	
	K2_3 =  matern(hyp3,3.,hyp4,t_matrix2_1) *  s_matrix_1_c

	K3_3 =  matern(hyp5,1.,hyp6,t_matrix2_1) *  s_matrix_2_c	
	
	#K4_3 =  matern(hyp7,1.,hyp8,t_matrix2_1) *  s_matrix_2_c		
	
	K5_3 =  linear_G(hyp7,hyp8,t_matrix2_mult_1)

	K6_3 =  linear_RL(hyp9,hyp10,t_matrix2_mult_1,s_matrix_1_c)
	
	K7_3 =  linear_RL(hyp11,hyp12,t_matrix2_mult_1,s_matrix_2_c)	

	K_2 = K1_3 + K2_3 + K3_3 + K5_3 + K6_3 + K7_3 # + K4_3 

	new_y = numpy.matmul(numpy.transpose(K2_f),numpy.matmul(K_inv,numpy.reshape(y_1,(-1,1))))

	output_GRL = numpy.ndarray.flatten(new_y) * 1.0
	output_GRL_stdev = numpy.sqrt(numpy.diag(K_2 - new_y_p_2)) * 1.0
	
	output_matrix[:,3] = output_GRL * 1.0
	output_matrix[:,4] = output_GRL_stdev * 1.0
	
	filename = "output_GRL.txt"
	numpy.savetxt(filename, output_matrix, fmt='%0.4f', delimiter=',')
	'''
	################2################

	################ (x,*) ################
	
	K2_f = K1_2 + K2_2 + K3_2 + K5_2 + K6_2 + K7_2

	new_y_p = numpy.matmul(K_inv,K2_f) 
	new_y_p_2 = numpy.matmul(numpy.transpose(K2_f),new_y_p)

	################ (*,*) ################
	
	K_2 =  K1_3 + K2_3 + K3_3 + K5_3 + K6_3 + K7_3

	new_y = numpy.matmul(numpy.transpose(K2_f),numpy.matmul(K_inv,numpy.reshape(y_1,(-1,1))))

	output_GRL = numpy.ndarray.flatten(new_y) * 1.0
	output_GRL_stdev = numpy.sqrt(numpy.diag(K_2 - new_y_p_2)) * 1.0

	output_matrix[:,3] = output_GRL * 1.0
	output_matrix[:,4] = output_GRL_stdev * 1.0	
	
	filename = "output_GRL.txt"
	numpy.savetxt(filename, output_matrix, fmt='%0.4f', delimiter=',')
	'''
	################3################
	
	################ (x,*) ################

	K2_f = K1_2 + K2_2 + K5_2 + K6_2

	new_y_p = numpy.matmul(K_inv,K2_f) 
	new_y_p_2 = numpy.matmul(numpy.transpose(K2_f),new_y_p)

	################ (*,*) ################
	
	K_2 = K1_3 + K2_3 + K5_3 + K6_3

	new_y = numpy.matmul(numpy.transpose(K2_f),numpy.matmul(K_inv,numpy.reshape(y_1,(-1,1))))

	output_GR = numpy.ndarray.flatten(new_y) * 1.0
	output_GR_stdev = numpy.sqrt(numpy.diag(K_2 - new_y_p_2)) * 1.0
	
	output_matrix[:,3] = output_GR * 1.0
	output_matrix[:,4] = output_GR_stdev * 1.0	 
		
	filename = "output_GR.txt"
	numpy.savetxt(filename, output_matrix, fmt='%0.4f', delimiter=',')		
	
	################4################

	#4 regression	
	
	#####
	
	K2_f = K1_2 + K5_2

	new_y_p = numpy.matmul(K_inv,K2_f) 
	new_y_p_2 = numpy.matmul(numpy.transpose(K2_f),new_y_p)

	#####
	
	K_2 = K1_3 + K5_3

	new_y = numpy.matmul(numpy.transpose(K2_f),numpy.matmul(K_inv,numpy.reshape(y_1,(-1,1))))

	output_G = numpy.ndarray.flatten(new_y)
	output_G_stdev = numpy.sqrt(numpy.diag(K_2 - new_y_p_2))
	
	output_matrix[:,3] = output_G
	output_matrix[:,4] = output_G_stdev	
		
	filename = "output_G.txt"
	numpy.savetxt(filename, output_matrix, fmt='%0.4f', delimiter=',')		
		
	################5################
	
	#5
	
	#####
	
	K2_f = K2_2 + K6_2

	new_y_p = numpy.matmul(K_inv,K2_f) 
	new_y_p_2 = numpy.matmul(numpy.transpose(K2_f),new_y_p)

	#####
	
	K_2 = K2_3 + K6_3

	new_y = numpy.matmul(numpy.transpose(K2_f),numpy.matmul(K_inv,numpy.reshape(y_1,(-1,1))))

	output_R = numpy.ndarray.flatten(new_y)
	output_R_stdev = numpy.sqrt(numpy.diag(K_2 - new_y_p_2))
	
	output_matrix[:,3] = output_R
	output_matrix[:,4] = output_R_stdev	
		
	filename = "output_R.txt"
	numpy.savetxt(filename, output_matrix, fmt='%0.4f', delimiter=',')		
			
	################6################
		
	#6
	
	#####

	K2_f = K2_2 + K3_2 + K6_2 + K7_2

	new_y_p = numpy.matmul(K_inv,K2_f) 
	new_y_p_2 = numpy.matmul(numpy.transpose(K2_f),new_y_p)

	#####

	K_2 = K2_3 + K3_3 + K6_3 + K7_3
	
	new_y = numpy.matmul(numpy.transpose(K2_f),numpy.matmul(K_inv,numpy.reshape(y_1,(-1,1))))

	output_RL = numpy.ndarray.flatten(new_y)
	output_RL_stdev = numpy.sqrt(numpy.diag(K_2 - new_y_p_2))
	
	output_matrix[:,3] = output_RL
	output_matrix[:,4] = output_RL_stdev	
		
	filename = "output_RL.txt"
	numpy.savetxt(filename, output_matrix, fmt='%0.4f', delimiter=',')		
				
	########################################################################
	########################################################################
	########################################################################
	########################################################################
	########################################################################
		
	fig= plt.figure(1,figsize=(9.,6.5))

	ax1 = plt.subplot(311)
	
	w_name = numpy.where(reg_names[:,0]==glob_ID1)[0]
	name_str_g = str(reg_names[w_name,1][0])
	
	w_g2 = numpy.where(s_vals1==glob_ID1)[0]
	
	ax1.fill_between(t_vals[w_g2],output_G[w_g2] - 2.*output_G_stdev[w_g2]	,output_G[w_g2] + 2.*output_G_stdev[w_g2]	,alpha=0.1,color="black")
	ax1.fill_between(t_vals[w_g2],output_G[w_g2] - output_G_stdev[w_g2]		,output_G[w_g2] + output_G_stdev[w_g2]		,alpha=0.2,color="black")
	
	w_g1 = numpy.where(xes3_type_1==glob_ID1)[0]
	ax1.plot(xes1_t_1[w_g1],xes2_sl_1[w_g1],color='k',marker='x',markersize=1.0,linewidth=0.0,label="Global data")
	
	label_g = str(name_str_g) + ": G(t)"
	
	ax1.plot(t_vals[w_g2],output_G[w_g2],color='k',linewidth=1.0,label=label_g)
	ax1.grid()
	ax1.set_xlim(numpy.min(t_vals),numpy.max(t_vals))
	ax1.legend(fontsize=7)
	
	ax1 = plt.subplot(312)

	w_g2 = numpy.where(s_vals1==glob_ID1)[0]

	ax1.fill_between(t_vals[w_g2],output_G[w_g2] - 2.*output_G_stdev[w_g2]	,output_G[w_g2] + 2.*output_G_stdev[w_g2]	,alpha=0.1,color="black")
	ax1.fill_between(t_vals[w_g2],output_G[w_g2] - output_G_stdev[w_g2]		,output_G[w_g2] + output_G_stdev[w_g2]		,alpha=0.2,color="black")
	ax1.plot(t_vals[w_g2],output_G[w_g2],color='k',linewidth=1.0,label=label_g)
	
	unique_r = numpy.unique(s_vals1)
	r_colors = ['C1','C2','C3','C4','C5','C6','C7','C8','C9']
	for n in range(0,len(unique_r)):
		if n !=glob_ID1:
			r_val = unique_r[n]
			w_name = numpy.where(reg_names[:,0]==r_val)[0]
			name_str_r = str(reg_names[w_name,1][0])
			w_r2 = numpy.where(s_vals1==r_val)[0][0:len(t_vals_tmp)]
			label1 = name_str_r+": G(t) + R$_" + str(int(r_val)) + "$(t)" 
			
			ax1.fill_between(t_vals[w_r2],output_GR[w_r2] - 2.*output_GR_stdev[w_r2]	,output_GR[w_r2] + 2.*output_GR_stdev[w_r2]	,alpha=0.1,color=r_colors[n])
			ax1.fill_between(t_vals[w_r2],output_GR[w_r2] - output_GR_stdev[w_r2]		,output_GR[w_r2] + output_GR_stdev[w_r2]	,alpha=0.2,color=r_colors[n])
			ax1.plot(t_vals[w_r2],output_GR[w_r2],linewidth=1.0,color=r_colors[n],label=label1)
	ax1.legend(fontsize=7)
	ax1.grid()
	ax1.set_xlim(numpy.min(t_vals),numpy.max(t_vals))	
	
	ax1 = plt.subplot(313)
	
	unique_r = numpy.unique(s_vals1)
	r_colors = ['C1','C2','C3','C4','C5','C6','C7','C8','C9']
	for n in range(0,len(unique_r)):
		if n !=glob_ID1:
			r_val = unique_r[n]
			w_r2 = numpy.where(s_vals1==r_val)[0][0:len(t_vals_tmp)]
			w_name = numpy.where(reg_names[:,0]==r_val)[0]
			name_str_r = str(reg_names[w_name,1][0])	
			label1 = name_str_r +" - " +name_str_g + ": R$_" + str(int(r_val)) + "$(t)"
			
			ax1.fill_between(t_vals[w_r2],output_R[w_r2] - 2.*output_R_stdev[w_r2]	,output_R[w_r2] + 2.*output_R_stdev[w_r2]	,alpha=0.1,color=r_colors[n])
			ax1.fill_between(t_vals[w_r2],output_R[w_r2] - output_R_stdev[w_r2]		,output_R[w_r2] + output_R_stdev[w_r2]		,alpha=0.2,color=r_colors[n])
			ax1.plot(t_vals[w_r2],output_R[w_r2],linewidth=1.0,color=r_colors[n],label=label1)
		
	ax1.legend(fontsize=7)
	ax1.grid()
	ax1.set_xlim(numpy.min(t_vals),numpy.max(t_vals))

	plt.tight_layout()

	pltname = wkspc +  '00_Gt_GtRt_Rt.png'

	plt.savefig(pltname, dpi = 300)

	plt.close()
	
	unique_L = numpy.unique(s_vals2)
	
	for n in range(0,len(unique_L)):
		if n != glob_ID2:
			l_val = unique_L[n]

			w_name = numpy.where(loc_names[:,0]==l_val)[0]
			name_str_l = str(loc_names[w_name,1][0])		

			fig= plt.figure(1,figsize=(9.,6.5))

			ax1 = plt.subplot(311)

			w_g2 = numpy.where(s_vals1==glob_ID1)[0]
			
			ax1.fill_between(t_vals[w_g2],output_G[w_g2] - 2.*output_G_stdev[w_g2]	,output_G[w_g2] + 2.*output_G_stdev[w_g2]	,alpha=0.1,color="black")
			ax1.fill_between(t_vals[w_g2],output_G[w_g2] - output_G_stdev[w_g2]		,output_G[w_g2] + output_G_stdev[w_g2]		,alpha=0.2,color="black")
			
			w_g1 = numpy.where(xes3_type_1==glob_ID1)[0]
			ax1.plot(xes1_t_1[w_g1],xes2_sl_1[w_g1],color='k',marker='x',markersize=1.0,linewidth=0.0,label="Global data")
			
			ax1.plot(t_vals[w_g2],output_G[w_g2],color='k',linewidth=1.0,label=label_g)
			ax1.legend(fontsize=7)
			ax1.grid()
			ax1.set_xlim(numpy.min(t_vals),numpy.max(t_vals))
			
			ax1 = plt.subplot(312)

			w_g2 = numpy.where(s_vals1==glob_ID1)[0]

			ax1.fill_between(t_vals[w_g2],output_G[w_g2] - 2.*output_G_stdev[w_g2]	,output_G[w_g2] + 2.*output_G_stdev[w_g2]	,alpha=0.1,color="black")
			ax1.fill_between(t_vals[w_g2],output_G[w_g2] - output_G_stdev[w_g2]		,output_G[w_g2] + output_G_stdev[w_g2]		,alpha=0.2,color="black")


			w_l2 = numpy.where(s_vals2==l_val)[0]
			r_val = int(s_vals1[w_l2][0])
			label1 = name_str_l+": G(t) + R$_" + str(r_val) + "$(t) + L$_{" +str(r_val) + "," +  str(int(l_val)) + "}$(t)"
			
			ax1.fill_between(t_vals[w_l2],output_GRL[w_l2] - 2.*output_GRL_stdev[w_l2]	,output_GRL[w_l2] + 2.*output_GRL_stdev[w_l2]	,alpha=0.1,color="blue")
			ax1.fill_between(t_vals[w_l2],output_GRL[w_l2] - output_GRL_stdev[w_l2]	,output_GRL[w_l2] + output_GRL_stdev[w_l2]	,alpha=0.2,color="blue")

			ax1.plot(t_vals[w_g2],output_G[w_g2],color='k',linewidth=1.0,label=label_g)	
			ax1.plot(t_vals[w_l2],output_GRL[w_l2],linewidth=1.0,color="blue",label=label1)			
			w_l1 = numpy.where(xes5_type_2==l_val)[0]		
			label_2 = name_str_l + " data"
			ax1.plot(xes1_t_1[w_l1],xes2_sl_1[w_l1],color='dodgerblue',marker='x',markersize=1.0,linewidth=0.0,label=label_2)	

			ax1.legend(fontsize=7)
			ax1.grid()
			ax1.set_xlim(numpy.min(t_vals),numpy.max(t_vals))	
			
			ax1 = plt.subplot(313)
			
			w_l2 = numpy.where(s_vals2==l_val)[0]
			r_val = int(s_vals1[w_l2][0])
			label1 = name_str_l+" - "+str(name_str_g)+": R$_" + str(r_val) + "$(t) + L$_{" +str(r_val) + "," +  str(int(l_val)) + "}$(t)"# - mL2$_{" +str(r_val) + "," +  str(int(l_val)) + "}(t)"
			
			ax1.fill_between(t_vals[w_l2],output_RL[w_l2] - 2.*output_RL_stdev[w_l2]	,output_RL[w_l2] + 2.*output_RL_stdev[w_l2]	,alpha=0.1,color="red")
			ax1.fill_between(t_vals[w_l2],output_RL[w_l2] - output_RL_stdev[w_l2]		,output_RL[w_l2] + output_RL_stdev[w_l2]	,alpha=0.2,color="red")
			ax1.plot(t_vals[w_l2],output_RL[w_l2],linewidth=1.0,color="red",label=label1)
			
			ax1.legend(fontsize=7)
			ax1.grid()
			ax1.set_xlim(numpy.min(t_vals),numpy.max(t_vals))

			plt.tight_layout()

			pltname = wkspc +  '01_Gt_GtRtLt_RtLt_' + str(int(l_val)).zfill(2) + '.png'

			plt.savefig(pltname, dpi = 300)

			plt.close()		
