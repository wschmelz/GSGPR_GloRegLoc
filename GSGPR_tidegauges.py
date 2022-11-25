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

##x_i = G(t_i) + E_xi + w_x
##y_i = G(t_i) + R_m(t_i) + L_n(t_i) + E_yi + w_y; m=1,...,M and n=1,...,N

##G(t) = matern_g(t) + linear_g(t) + offset_g
##R_m(t) = G(t) + matern_r(t) + linear_r(t) + offset_r; m=1,...,M
##L_m,n(t) = R_m(t) + matern_l1(t) + matern_l2(t) + linear_l(t) + offset_l; m=1,...,M and n=1,...,N

##G(t) ~ GP[0,K_G(t,t^')]
##R_m(t) ~ GP[0,K_R_m(t_m,t_m^')]; m=1,...,M
##L_m,n(t) ~ GP[0,K_L_m,n(t_m,n,t_m,n^')]; m=1,...,M and n=1,...,N

##K_G(t,t^') = k_matern_g(t,t^') + σ_linear_g^2(t-t_0)(t^'-t_0 ) + σ_offset_g^2
##K_R_m(t_m,t_m^') = K_G(t,t^') + matern_r(t_m,t_m^') + σ_linear_r^2(t_m-t_0,m)(t_m^'-t_0,m ) + σ_offset_r^2; m=1,...,M
##K_L_m,n(t_m,n,t_m,n^') = K_G(t,t^') + K_R_m(t_m,t_m^') + matern_l1(t_n,t_n^') + matern_l2(t_n,t_n^') + σ_linear_l^2(t_n-t_0,n)(t_n^'-t_0,n) + σ_offset_l^2;m=1,...,M and n=1,...,N

###Nonlinear
###guess_orig[0] is amplitude parameter for global nonlinear component(matern_g; v=3/2)
###guess_orig[1] is timescale parameter for global nonlinear component(matern_g; v=3/2)
###guess_orig[2] is amplitude parameter for regional nonlinear component(matern_r; v=3/2)
###guess_orig[3] is timescale parameter for regional nonlinear component(matern_r; v=3/2)
###guess_orig[4] is amplitude parameter for local nonlinear component(matern_l1; v=3/2)
###guess_orig[5] is timescale parameter for local nonlinear component(matern_l1; v=3/2)
###guess_orig[6] is amplitude parameter for local nonlinear component(matern_l2; v=1/2)
###guess_orig[7] is timescale parameter for local nonlinear component(matern_l2; v=1/2)

###Linear
###guess_orig[8] is offset parameter for global linear component (offset_g)
###guess_orig[9] is slope parameter for global linear component (linear_g)
###guess_orig[10] is offset parameter for regional linear component (offset_r)
###guess_orig[11] is slope parameter for regional linear component (linear_r)
###guess_orig[12] is offset parameter for local linear component (offset_l)
###guess_orig[13] is slope parameter for local linear component (linear_l)

###Error
###guess_orig[14] is error term for data in global set (w_x)
###guess_orig[15] is error term for all data not in global set (w_y)

#stepsize_divisor is a single positive floating point number, 10-100 is reasonable.
#MCMC_iters is number of  iterations for MCMC process, 2000-10000 is reasonable
#new_dt is sample rate of plotted functions
#fit indicates whether to refit parameters: 1 is fit, 2 is use guess_orig

glob_ID1 = 0
glob_ID2 = 0
guess_orig = numpy.array([100.,100.,100.,100.,100.,50.,1.,1.,5.,1.,5.,1.,1.,1.,10.,10.])
stepsize_divisor=10
MCMC_iters = 5000
new_dt =  0.25

GSGPR_GloRegLoc_py.GP_GloRegLoc(data_series[::3,:],reg_names,loc_names,glob_ID1,glob_ID2,guess_orig,stepsize_divisor,MCMC_iters,new_dt,2)