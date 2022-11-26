import os
import sys
import datetime
import re
import shutil
import time
from time import gmtime, strftime
import pandas
import glob
import requests
import numpy
from numpy import matrix
from numpy import genfromtxt
from numpy import linalg

backslash = '\\'

wkspc = str(os.getcwd()).replace(backslash,"/") + "/"

print (str(__file__))

print (time.strftime("%H:%M:%S"))

###THESE LINES DOWNLOAD A PYTHON SCRIPT FROM GITHUB###
######################################################

##dowload generalized psuedo-spatial GP regression 
##algoritm as a python script from Github
##remove these lines if modifying the script locally
##or they will overwrite any changes made

url = 'https://raw.githubusercontent.com/wschmelz/GSGPR_GloRegLoc/main/Scripts/GSGPR_GloRegLoc.py'
GSGPR_GloRegLoc = requests.get(url)  

with open('GSGPR_GloRegLoc_py.py', 'w') as f:
    f.write(GSGPR_GloRegLoc.text)

import GSGPR_GloRegLoc_py

######################################################
######################################################

####THESE LINES DOWNLOAD SAMPLE DATA FROM GITHUB######
######################################################

#tide gauge data from:
##1 AC, NJ; 2 SH, NJ; 3 NYC, NY;
##4 MTK, NY; 5 PTL, ME; and 6 BH, ME
##and the Church and White (2016) GMSL record

#remove these lines and  a different data series 
#using format specified below to apply this tool generally

#Download datafile and save to local directory as 'US_NorthEast_MSLrel2000.csv'
url = 'https://raw.githubusercontent.com/wschmelz/GSGPR_GloRegLoc/main/Data/US_NorthEast_MSLrel2000.csv'
US_NorthEast_MSLrel2000 = requests.get(url)  

with open('US_NorthEast_MSLrel2000.csv', 'w') as f:
    f.write(US_NorthEast_MSLrel2000.text)
	
#Download datafile and save to local directory as 'US_NorthEast_LocNames.csv'
url = 'https://raw.githubusercontent.com/wschmelz/GSGPR_GloRegLoc/main/Data/US_NorthEast_LocNames.csv'
US_NorthEast_LocNames = requests.get(url)  

with open('US_NorthEast_LocNames.csv', 'w') as f:
    f.write(US_NorthEast_LocNames.text)
	
#Download datafile and save to local directory as 'US_NorthEast_RegNames.csv'
url = 'https://raw.githubusercontent.com/wschmelz/GSGPR_GloRegLoc/main/Data/US_NorthEast_RegNames.csv'
US_NorthEast_RegNames = requests.get(url)  

with open('US_NorthEast_RegNames.csv', 'w') as f:
    f.write(US_NorthEast_RegNames.text)

#read datafiles

filename = "US_NorthEast_MSLrel2000.csv"
data_series = numpy.genfromtxt(filename, delimiter=',')	

filename = "US_NorthEast_LocNames.csv"
loc_names_tmp = pandas.read_csv(filename, delimiter=',',header=None)			
loc_names = numpy.array(loc_names_tmp)

filename = "US_NorthEast_RegNames.csv"
reg_names_tmp = pandas.read_csv(filename, delimiter=',',header=None)	
reg_names = numpy.array(reg_names_tmp)

######################################################
######################################################

#GP function is:
#GSGPR_GloRegLoc_py.GP_GloRegLoc(data_series,reg_names,loc_names,glob_ID1,glob_ID2,guess_orig,stepsize_divisor,MCMC_iters,new_dt,fit)

#data_series is input data [time,value,error,ID1 Regional or Global ID,ID2 Local ID]
#reg_names is the location names relative to ID1 [ID1,region name]
#loc_names is the location names relative to ID2 [ID2,location name]
#glob_ID1 is the 'regional' identifier of the global dataseries
#glob_ID2 is the 'local' identifier of the global dataseries
#guess_orig is 16 item vector - original guess for parameters

#t is time; m is region; n is site

##x_i = G(t_i) + E_xi + w_x
##y_j,m,n = G(t_j,m_j,n_j) + R_m(t_j,m_j,n_j) + L_n(t_j,m_j,n_j) + E_y(t_j,m_j,n_j) + w_y

##G(t) = matern_G(t) + linear_G(t) + offset_G
##R(t,m) = G(t) + maternR(t,m) + linear_R_m(t,m) + offset_R(m)
##L(t,m,n) = R(t,m) + matern_L1(t,m,n) + matern_L2(t,m,n) + linear_L(t,m,n) + offset_L(m,n)

##G(t) ~ GP[0,K_G(t,t^')]
##R(t,m) ~ GP[0,K_R(t,t^';m,m^')]
##L(t,m,n) ~ GP[0,K_L(t,t^';m,m^';n,n^')]

##K_G(t,t^') = k_matern_G(t,t^') + σ_linear_G^2(t-t_0)(t^'-t_0 ) + σ_offset_G^2
##K_R(t,t^';m,m^') = K_G(t,t^') + matern_R(t,t^')*δ(m-m^') + σ_linear_R^2(t-t_0)(t^'-t_0)*δ(m-m^') + σ_offset_R^2*δ(m-m^')
##K_L_m,n(t,t^';m,m^';n,n^') = K_G(t,t^') + K_R_m(t,t^';m,m^') + matern_L1(t,t^')*δ(n-n^') + matern_L2(t,t^')*δ(n-n^') + σ_linear_L^2(t-t_0)(t^'-t_0)*δ(n-n^') + σ_offset_L^2*δ(n-n^')

###Nonlinear functions
###guess_orig[0] is amplitude parameter for global nonlinear component(matern_G; v=3/2)
###guess_orig[1] is timescale parameter for global nonlinear component(matern_G; v=3/2)
###guess_orig[2] is amplitude parameter for regional nonlinear component(matern_R; v=3/2)
###guess_orig[3] is timescale parameter for regional nonlinear component(matern_R; v=3/2)
###guess_orig[4] is amplitude parameter for local nonlinear component(matern_L1; v=3/2)
###guess_orig[5] is timescale parameter for local nonlinear component(matern_L1; v=3/2)
###guess_orig[6] is amplitude parameter for local nonlinear component(matern_L2; v=3/2)
###guess_orig[7] is timescale parameter for local nonlinear component(matern_L2; v=3/2; < 3x dt; local short-term variation)

###Linear functions
###guess_orig[8] is offset parameter for global linear component (offset_G)
###guess_orig[9] is slope parameter for global linear component (linear_G)
###guess_orig[10] is offset parameter for regional linear component (offset_R)
###guess_orig[11] is slope parameter for regional linear component (linear_R)
###guess_orig[12] is offset parameter for local linear component (offset_L)
###guess_orig[13] is slope parameter for local linear component (linear_L)

###Error components
###guess_orig[14] is error term for data in global set (w_x)
###guess_orig[15] is error term for all data not in global set (w_y)


#f_ev_iters is number of function evaluations for Nelder-Mead optimization process
#MCMC_iters is number of  iterations for MCMC process, 2000-10000 is reasonable
#stepsize_divisor is a single positive floating point number, 10-100 is reasonable.
#new_dt is sample rate of plotted functions
#fit indicates whether to refit parameters: 1 is fit, 2 is use guess_orig

glob_ID1 = 0
glob_ID2 = 0
guess_orig = numpy.array([50.,25.,50.,25.,50.,25.,25.,1.,5.,1.,5.,1.,1.,1.,10.,10.])
f_ev_iters = 20000
MCMC_iters = 5000
stepsize_divisor=20
new_dt =  0.25

GSGPR_GloRegLoc_py.GP_GloRegLoc(data_series,reg_names,loc_names,glob_ID1,glob_ID2,guess_orig,f_ev_iters,MCMC_iters,stepsize_divisor,new_dt,1)