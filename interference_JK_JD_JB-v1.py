# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 13:49:09 2019

@author: xzou
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.integrate import*
from scipy.interpolate import*
from statistics import mean 
from Tkinter import *
import Tkinter as ttk 
from ttk import *
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm



df = pd.read_csv(r'C:\Xingquan\Interference_script_test\5015-JBDS5024_RMA122205\8_3d_N2O-H2O-CO2\20181227valid\Combined_20181227_095952.dat', header=0, sep='\s\s+', engine='python' )

def avg(myArray, N=4):            #Average every four points
    cum = np.cumsum(myArray,0)
    result = cum[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]

    remainder = myArray.shape[0] % N
    if remainder != 0:
        if remainder < myArray.shape[0]:
            lastAvg = (cum[-1]-cum[-1-remainder])/float(remainder)
        else:
            lastAvg = cum[-1]/float(remainder)
        result = np.vstack([result, lastAvg])

    return result
    
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

n = 5      #choose how many points to do linearRegression
p = 5      #choose how many linear regressions to compare
    
######--------------------------------------------
print (30 * '-')
print ("   MidIR Analyzer Interference test and validation")
print (30 * '-')
print ("1.  JKADS_1D_CO")
print ("2.  JKADS_1D_CO2")
print ("3.  JKADS_1D_N2O")
print ("4.  JKADS_2D_N2O-CO2_test")
print ("5.  JKADS_2D_N2O-CO2_validation")
print ("6.  JKADS_2D_CO-CO2_validation")
print ("7.  JDDS_1D_N2O")
print ("8.  JDDS_2D_N2O-CO2_test")
print ("9.  JDDS_2D_N2O-CO2_validation")
print ("10. JBDS_1D_N2O")
print ("11. JBDS_2D_N2O-CO2_test")
print ("12. JBDS_2D_N2O-CO2_validation")
print ("13. JBDS_2D_N2O-H2O_test")
print ("14. JBDS_2D_N2O-H2O_validation")
print ("15. JBDS_3D_N2O-H2O-CO2_validation")
print (30 * '-')
is_valid=0
while not is_valid :
        try :
                choice = int ( raw_input('Enter your choice [1-16] : ') )
                is_valid = 1 ## set it to 1 to validate input and to terminate the while..not loop
               
        except ValueError, e :
                print ("'%s' is not a valid integer." % e.args[0].split(": ")[1])
 
### Take action as per selected menu-option ###

if choice == 1:
        print ("You have chosen JKADS_1D_CO_test")
        
        Temp = np.zeros((len(df.N2O), 3))
        Temp[:, 0]=df.str1
        Temp[:, 1]=df.str31
        Temp[:, 2]=df.str50
        
        str1_str50_fit = np.poly1d(np.polyfit(Temp[:,2], Temp[:,0], 2))
        str31_str50_fit = np.poly1d(np.polyfit(Temp[:,2], Temp[:,1], 1))
        t = np.linspace(min(Temp[:,2]), max(Temp[:,2]), 100)
        
        plt.figure(1)
        plt.subplot(221)
        plt.plot(Temp[:,2], Temp[:,0],'o')
        plt.plot(t, str1_str50_fit(t), 'r-')
        plt.ylabel('str1_str50', fontsize=8)
        plt.subplot(222)
        plt.plot(Temp[:,2], Temp[:,0]-str1_str50_fit(Temp[:,2]),'o')
        plt.axhline(y=0.0, color='r', linestyle='-')
        plt.ylabel('str1_str50_residual', fontsize=8)
        plt.subplot(223)
        plt.plot(Temp[:,2], Temp[:,1],'o')
        plt.plot(t, str31_str50_fit(t), 'r-')
        plt.xlabel('str50', fontsize=8)
        plt.ylabel('str31_str50', fontsize=8)
        plt.subplot(224)
        plt.plot(Temp[:,2], Temp[:,1]-str31_str50_fit(Temp[:,2]),'o')
        plt.axhline(y=0.0, color='r', linestyle='-')
        plt.xlabel('str50', fontsize=8)
        plt.ylabel('str31_str50_residual', fontsize=8)
        plt.savefig("JKADS_1D_CO_test.png")
        
        with open('JKADS_1D_CO_test.txt', 'w') as out_file:
            out_string='Fitting parameters (sign reversed)\n'
            out_string+='S1_S50      = ' +str(-1*str1_str50_fit[1])
            out_string+='\n'   
            out_string+='S1_S50_qua  = ' +str(-1*str1_str50_fit[2])
            out_string+='\n'     
            out_string+='S31_S50_lin = ' +str(-1*str31_str50_fit[1])
            out_string+='\n'
            out_file.write(out_string)
            
            
if choice == 2:
        print ("You have chosen JKADS_1D_CO2_test")
        
        Temp = np.zeros((len(df.N2O), 3))
        Temp[:, 0]=df.str1
        Temp[:, 1]=df.str40
        Temp[:, 2]=df.str50
        
        str1_str40_fit = np.poly1d(np.polyfit(Temp[:,1], Temp[:,0], 1))
        t = np.linspace(min(Temp[:,1]), max(Temp[:,1]), 100)
        
        plt.figure(1)
        plt.subplot(211)
        plt.plot(Temp[:,1], Temp[:,0],'o')
        plt.plot(t, str1_str40_fit(t), 'r-')
        plt.xlabel('str40', fontsize=8)
        plt.ylabel('str1_str40', fontsize=8)
        plt.subplot(212)
        plt.plot(Temp[:,1], Temp[:,0]-str1_str40_fit(Temp[:,1]),'o')
        plt.axhline(y=0.0, color='r', linestyle='-')
        plt.xlabel('str40', fontsize=8)
        plt.ylabel('str1_str40_residual', fontsize=8)
        plt.savefig("JKADS_1D_CO2_test.png")
        
        with open('JKADS_1D_CO2_test.txt', 'w') as out_file:
            out_string='Fitting parameters (sign reversed)\n'
            out_string+='S1_S40      = ' +str(-1*str1_str40_fit[1])
            out_string+='\n'   
            out_file.write(out_string)
            
            
if choice == 3:
        print ("You have chosen JKADS_1D_N2O_test")
        
        Temp = np.zeros((len(df.N2O), 3))
        Temp[:, 0]=df.str1
        Temp[:, 1]=df.str31
        Temp[:, 2]=df.str50
        
        str31_str1_fit = np.poly1d(np.polyfit(Temp[:,0], Temp[:,1], 2))
        t = np.linspace(min(Temp[:,0]), max(Temp[:,0]), 100)
        
        plt.figure(1)
        plt.subplot(211)
        plt.plot(Temp[:,0], Temp[:,1],'o')
        plt.plot(t, str31_str1_fit(t), 'r-')
        plt.xlabel('str1', fontsize=8)
        plt.ylabel('str31_str1', fontsize=8)
        plt.subplot(212)
        plt.plot(Temp[:,0], Temp[:,1]-str31_str1_fit(Temp[:,0]),'o')
        plt.axhline(y=0.0, color='r', linestyle='-')
        plt.xlabel('str1', fontsize=8)
        plt.ylabel('str31_str1_residual', fontsize=8)
        plt.savefig("JKADS_1D_N2O_test.png")
        
        with open('JKADS_1D_N2O_test.txt', 'w') as out_file:
            out_string='Fitting parameters (sign reversed)\n'
            out_string+='S31_S1_lin      = ' +str(-1*str31_str1_fit[1])
            out_string+='\n' 
            out_string+='S31_S1_qua      = ' +str(-1*str31_str1_fit[2])
            out_string+='\n'   
            out_file.write(out_string)
            
elif choice == 4:
        print ("You have chosen JKADS_2D_N2O-CO2_test.")
        Temp = np.zeros((len(df.N2O), 7))
        Temp[:, 0]=df.timestamp
        Temp[:, 1]=df.N2O
        Temp[:, 2]=df.str1
        Temp[:, 3]=df.str31
        Temp[:, 4]=df.str40
        Temp[:, 5]=df.str50
        Temp[:, 6]=df.c13o2

        data=avg(Temp)

        data_x = np.array(data[:,0])
        data_y = np.array(data[:,1])
        data_str1   = np.array(data[:,2])
        data_str31   = np.array(data[:,3])
        data_str40   = np.array(data[:,4])
        data_str50   = np.array(data[:,5])
        data_c13o2         = np.array(data[:,6])

  
        x = list(chunks(data_x, n)) 
        y = list(chunks(data_y, n)) 
        str1 = list(chunks(data_str1, n))
        str31 = list(chunks(data_str31, n))
        str40 = list(chunks(data_str40, n))
        str50 = list(chunks(data_str50, n))
        co2_corr       = list(chunks(data_c13o2, n))

        M=np.zeros((int(len(data_y)/n), 9))
        choose=np.zeros((int(len(data_y)/n/p), 10))

#str1_spec  = str1  + Offset1  + S1_S50*str50 + S1_S50_qua*str50**2  + S1_S40 *str40
#str50_spec = str50 + Offset50 + S50_S1*str1  + S50_S40*str40
#str31_spec = str31 + Offset31 + S31_S1_lin*str1 + S31_S1_qua*str1**2 + S31_S50_lin*str50

        for i in range(0, int(len(data_y)/n)):
            xx_i=np.array(x[i])
            yy_i=np.array(y[i])
            xx=xx_i.reshape(len(x[i]), 1)
            yy=yy_i.reshape(len(y[i]), 1)
            regr = linear_model.LinearRegression()
            regr.fit(xx, yy)
            plt.scatter(xx_i, yy_i, color='black')
            plt.plot(xx, regr.predict(xx), color='blue', linewidth=3)
            M[i][0]=abs(regr.coef_)
            M[i][1]=regr.intercept_
            M[i][2]=x[i][0]                  #time
            M[i][3]=y[i][0]                  #N2O
            M[i][4]=str1[i][0]     #
            M[i][5]=str31[i][0]
            M[i][6]=str40[i][0]
            M[i][7]=str50[i][0]
            M[i][8]=co2_corr[i][0]
    
        N=list(chunks(M, p))
        for j in range(0, int(len(data_y)/n/p)):
            slope_index=N[j][:,0].argmin()
            choose[j][0]=slope_index      # choose data index position 
            choose[j][1]=min(N[j][:,0])   #choose data slope
            choose[j][2]=N[j][slope_index][2]      #choose data first x value
            choose[j][3]=N[j][slope_index][3]      #choose data first y value
            choose[j][4]=N[j][slope_index][1]      #choose data interception
            choose[j][5]=N[j][slope_index][4]      #str1
            choose[j][6]=N[j][slope_index][5]      #str31
            choose[j][7]=N[j][slope_index][6]      #str40
            choose[j][8]=N[j][slope_index][7]      #str50
            choose[j][9]=N[j][slope_index][8]      #co2_corr

        plt.figure(1) 
        plt.subplot(211)   
        plt.scatter(choose[:,2], choose[:,3], color='red')
        plt.xlabel('time', fontsize=8)
        plt.ylabel('N2O', fontsize=8)
        plt.subplot(212)
        plt.scatter(choose[:,2], choose[:,9], color='blue')
        plt.xlabel('time', fontsize=8)
        plt.ylabel('co2_corr', fontsize=8)
        
#        choose_transpose = choose.transpose()
        y = choose[:,8]
        x = np.array([choose[:,5],choose[:,7]])
        
        
        def reg_m(y, x):
            ones = np.ones(len(x[0]))
            X = sm.add_constant(np.column_stack((x[0], ones)))
            for ele in x[1:]:
                X = sm.add_constant(np.column_stack((ele, X)))
            results = sm.OLS(y, X).fit()
            return results
#        print reg_m(y, x).summary()  #for test use
        
        
        clf = linear_model.LinearRegression()
        
        x=x.transpose()    #x[0]:str1;   x[1]: str40
        clf.fit(x, y)
        print(clf.coef_)  
        print(clf.intercept_)
#        plt.figure(4) 
#        plt.plot(choose[:,2], x[:,0])
#        plt.ylabel('x0', fontsize=8)
#        plt.figure(5) 
#        plt.plot(choose[:,2], x[:,1])
#        plt.ylabel('x1', fontsize=8)

        CO_cr = 0.05512*(choose[:,8]+(-1*clf.coef_[0]*choose[:,5])+(-1*clf.coef_[1]*choose[:,7]))
        
        
        str50_str1_fit = np.poly1d(np.polyfit(choose[:,5], CO_cr, 1))    #linear fit of CO_cr Vs str1 
        str50_str40_fit = np.poly1d(np.polyfit(choose[:,7], CO_cr, 1))    #linear fit of CO Vs str40

        t_str1 = np.linspace(min(choose[:,5]), max(choose[:,5]), 100)
        t_str40  = np.linspace(min(choose[:,7]), max(choose[:,7]), 100)

        plt.figure(2)
        plt.subplot(211)
        plt.plot(choose[:,5], CO_cr, 'o')
        plt.plot(t_str1, str50_str1_fit(t_str1), 'r-', label= 'fit slope='+str(round(str50_str1_fit[1],6)))
        plt.legend()
        plt.xlabel('str1', fontsize=8)
        plt.ylabel('str50_str1', fontsize=8)
        plt.subplot(212)
        plt.plot(choose[:,7], CO_cr, 'o')
        plt.plot(t_str40, str50_str40_fit(t_str40), 'r-', label= 'fit slope='+str(round(str50_str40_fit[1],6)))
        plt.legend()
        plt.ylabel('str50_str40', fontsize=8)
        plt.xlabel('str40', fontsize=8)
        plt.ylabel('str50_str40', fontsize=8)
        plt.savefig('JKADS_N2O-CO2_2Dtest.png')


        with open('JKADS_N2O-CO2_2Dtest.txt', 'w') as out_file:
            out_string='Fitting parameters (sign reversed).Restart analyzer after updating config file for validation test\n'
            out_string+='S50_S1  =     ' +str(-1*clf.coef_[0])
            out_string+='\n'   
            out_string+='S50_S40  =     ' +str(-1*clf.coef_[1])
            out_string+='\n'        
            out_file.write(out_string)

elif choice == 5:
        print ("You have chosen JKADS_2D_N2O-CO2_validation.")
        Temp = np.zeros((len(df.N2O), 8))
        Temp[:, 0]=df.timestamp
        Temp[:, 1]=df.N2O
        Temp[:, 2]=df.str1
        Temp[:, 3]=df.str31
        Temp[:, 4]=df.str40
        Temp[:, 5]=df.str50
        Temp[:, 6]=df.c13o2
        Temp[:, 7]=df.CO

        data=avg(Temp)

        data_x = np.array(data[:,0])
        data_y = np.array(data[:,1])
        data_str1   = np.array(data[:,2])
        data_str31   = np.array(data[:,3])
        data_str40   = np.array(data[:,4])
        data_str50   = np.array(data[:,5])
        data_c13o2   = np.array(data[:,6])
        data_CO   = np.array(data[:,7])
  
        x = list(chunks(data_x, n)) 
        y = list(chunks(data_y, n)) 
        str1 = list(chunks(data_str1, n))
        str31 = list(chunks(data_str31, n))
        str40 = list(chunks(data_str40, n))
        str50 = list(chunks(data_str50, n))
        co2_corr = list(chunks(data_c13o2, n))
        CO       = list(chunks(data_CO, n))

        M=np.zeros((int(len(data_y)/n), 10))
        choose=np.zeros((int(len(data_y)/n/p), 11))

        for i in range(0, int(len(data_y)/n)):
            xx_i=np.array(x[i])
            yy_i=np.array(y[i])
            xx=xx_i.reshape(len(x[i]), 1)
            yy=yy_i.reshape(len(y[i]), 1)
            regr = linear_model.LinearRegression()
            regr.fit(xx, yy)
            plt.scatter(xx_i, yy_i, color='black')
            plt.plot(xx, regr.predict(xx), color='blue', linewidth=3)
            M[i][0]=abs(regr.coef_)
            M[i][1]=regr.intercept_
            M[i][2]=x[i][0]                  #time
            M[i][3]=y[i][0]                  #N2O
            M[i][4]=str1[i][0]     #
            M[i][5]=str31[i][0]
            M[i][6]=str40[i][0]
            M[i][7]=str50[i][0]
            M[i][8]=co2_corr[i][0]
            M[i][9]=CO[i][0]
    
        N=list(chunks(M, p))
        for j in range(0, int(len(data_y)/n/p)):
            slope_index=N[j][:,0].argmin()
            choose[j][0]=slope_index      # choose data index position 
            choose[j][1]=min(N[j][:,0])   #choose data slope
            choose[j][2]=N[j][slope_index][2]      #choose data first x value   #time
            choose[j][3]=N[j][slope_index][3]      #choose data first y value   #N2O
            choose[j][4]=N[j][slope_index][1]      #choose data interception
            choose[j][5]=N[j][slope_index][4]      #str1
            choose[j][6]=N[j][slope_index][5]      #str31
            choose[j][7]=N[j][slope_index][6]      #str40
            choose[j][8]=N[j][slope_index][7]      #str50
            choose[j][9]=N[j][slope_index][8]      #co2_corr
            choose[j][10]=N[j][slope_index][9]     #CO
                        
        CO_N2O_fit = np.poly1d(np.polyfit(choose[:,3], choose[:,10], 1))    #linear fit of CO Vs N2O
        CO_c13o2_fit = np.poly1d(np.polyfit(choose[:,9], choose[:,10], 1))    #linear fit of CO Vs c13o2

        t_N2O = np.linspace(min(choose[:,3]), max(choose[:,3]), 100)
        t_c13o2  = np.linspace(min(choose[:,9]), max(choose[:,9]), 100)

        plt.figure(1)
        plt.subplot(211)
        plt.plot(choose[:,3], choose[:,10], 'o')
        plt.plot(t_N2O, CO_N2O_fit(t_N2O), 'r-', label= 'fit slope='+str(round(CO_N2O_fit[1],6)))
        plt.legend()
        plt.xlabel('N2O', fontsize=8)
        plt.ylabel('CO_N2O', fontsize=8)
        plt.subplot(212)
        plt.plot(choose[:,9], choose[:,10], 'o')
        plt.plot(t_c13o2, CO_c13o2_fit(t_c13o2), 'r-', label= 'fit slope='+str(round(CO_c13o2_fit[1],6)))
        plt.legend()
        plt.ylabel('CO_c13o2', fontsize=8)
        plt.xlabel('c13o2', fontsize=8)
        plt.savefig('JKADS_N2O-CO2_2Dvalidation.png')

elif choice == 6:
        print ("You have chosen JKADS_2D_CO-CO2_validation.")
        Temp = np.zeros((len(df.N2O), 8))
        Temp[:, 0]=df.timestamp
        Temp[:, 1]=df.N2O
        Temp[:, 2]=df.str1
        Temp[:, 3]=df.str31
        Temp[:, 4]=df.str40
        Temp[:, 5]=df.str50
        Temp[:, 6]=df.c13o2
        Temp[:, 7]=df.CO

        data=avg(Temp)

        data_x = np.array(data[:,0])
        data_y = np.array(data[:,1])
        data_str1   = np.array(data[:,2])
        data_str31   = np.array(data[:,3])
        data_str40   = np.array(data[:,4])
        data_str50   = np.array(data[:,5])
        data_c13o2   = np.array(data[:,6])
        data_CO   = np.array(data[:,7])
  
        x = list(chunks(data_x, n)) 
        y = list(chunks(data_y, n)) 
        str1 = list(chunks(data_str1, n))
        str31 = list(chunks(data_str31, n))
        str40 = list(chunks(data_str40, n))
        str50 = list(chunks(data_str50, n))
        co2_corr = list(chunks(data_c13o2, n))
        CO       = list(chunks(data_CO, n))

        M=np.zeros((int(len(data_y)/n), 10))
        choose=np.zeros((int(len(data_y)/n/p), 11))

        for i in range(0, int(len(data_y)/n)):
            xx_i=np.array(x[i])
            yy_i=np.array(y[i])
            xx=xx_i.reshape(len(x[i]), 1)
            yy=yy_i.reshape(len(y[i]), 1)
            regr = linear_model.LinearRegression()
            regr.fit(xx, yy)
            plt.scatter(xx_i, yy_i, color='black')
            plt.plot(xx, regr.predict(xx), color='blue', linewidth=3)
            M[i][0]=abs(regr.coef_)
            M[i][1]=regr.intercept_
            M[i][2]=x[i][0]                  #time
            M[i][3]=y[i][0]                  #N2O
            M[i][4]=str1[i][0]     #
            M[i][5]=str31[i][0]
            M[i][6]=str40[i][0]
            M[i][7]=str50[i][0]
            M[i][8]=co2_corr[i][0]
            M[i][9]=CO[i][0]
    
        N=list(chunks(M, p))
        for j in range(0, int(len(data_y)/n/p)):
            slope_index=N[j][:,0].argmin()
            choose[j][0]=slope_index      # choose data index position 
            choose[j][1]=min(N[j][:,0])   #choose data slope
            choose[j][2]=N[j][slope_index][2]      #choose data first x value   #time
            choose[j][3]=N[j][slope_index][3]      #choose data first y value   #N2O
            choose[j][4]=N[j][slope_index][1]      #choose data interception
            choose[j][5]=N[j][slope_index][4]      #str1
            choose[j][6]=N[j][slope_index][5]      #str31
            choose[j][7]=N[j][slope_index][6]      #str40
            choose[j][8]=N[j][slope_index][7]      #str50
            choose[j][9]=N[j][slope_index][8]      #co2_corr
            choose[j][10]=N[j][slope_index][9]     #CO
                        
        N2O_CO_fit = np.poly1d(np.polyfit(choose[:,10], choose[:,3], 1))    #linear fit of N2O Vs CO
        N2O_c13o2_fit = np.poly1d(np.polyfit(choose[:,9], choose[:,3], 1))    #linear fit of N2O Vs c13o2

        t_CO = np.linspace(min(choose[:,10]), max(choose[:,10]), 100)
        t_c13o2  = np.linspace(min(choose[:,9]), max(choose[:,9]), 100)

        plt.figure(1)
        plt.subplot(211)
        plt.plot(choose[:,10], choose[:,3], 'o')
        plt.plot(t_CO, N2O_CO_fit(t_CO), 'r-', label= 'fit slope='+str(round(N2O_CO_fit[1],6)))
        plt.legend()
        plt.xlabel('CO', fontsize=8)
        plt.ylabel('N2O_CO', fontsize=8)
        plt.subplot(212)
        plt.plot(choose[:,9], choose[:,3], 'o')
        plt.plot(t_c13o2, N2O_c13o2_fit(t_c13o2), 'r-', label= 'fit slope='+str(round(N2O_c13o2_fit[1],6)))
        plt.legend()
        plt.ylabel('N2O_c13o2', fontsize=8)
        plt.xlabel('c13o2', fontsize=8)
        plt.savefig('JKADS_CO-CO2_2Dvalidation.png')       
        
elif choice == 7:
        print ("You have chosen JDDS_1D_N2O.")

        Temp = np.zeros((len(df.N2O), 7))
        Temp[:, 0]=df.timestamp
        Temp[:, 1]=df.N2O
        Temp[:, 2]=df.fsr_peak1_spec
        Temp[:, 3]=df.fsr_peak4_TC
        Temp[:, 4]=df.fsr_peak5_TC
        Temp[:, 5]=df.fsr_peak6_TC
        Temp[:, 6]=df.co2_TC

        data=avg(Temp)

        data_x = np.array(data[:,0])
        data_y = np.array(data[:,1])
        data_fsr_peak1_spec = np.array(data[:,2])
        data_fsr_peak4_TC   = np.array(data[:,3])
        data_fsr_peak5_TC   = np.array(data[:,4])
        data_fsr_peak6_TC   = np.array(data[:,5])
        data_co2_TC         = np.array(data[:,6])
  
        x = list(chunks(data_x, n)) 
        y = list(chunks(data_y, n)) 
        fsr_peak1_spec = list(chunks(data_fsr_peak1_spec, n))
        fsr_peak4_TC = list(chunks(data_fsr_peak4_TC, n))
        fsr_peak5_TC = list(chunks(data_fsr_peak5_TC, n))
        fsr_peak6_TC = list(chunks(data_fsr_peak6_TC, n))
        co2_TC       = list(chunks(data_co2_TC, n))



        M=np.zeros((int(len(data_y)/n), 9))
        choose=np.zeros((int(len(data_y)/n/p), 10))

        for i in range(0, int(len(data_y)/n)):
            xx_i=np.array(x[i])
            yy_i=np.array(y[i])
            xx=xx_i.reshape(len(x[i]), 1)
            yy=yy_i.reshape(len(y[i]), 1)
            regr = linear_model.LinearRegression()
            regr.fit(xx, yy)
            plt.scatter(xx_i, yy_i, color='black')
            plt.plot(xx, regr.predict(xx), color='blue', linewidth=3)
            M[i][0]=abs(regr.coef_)
            M[i][1]=regr.intercept_
            M[i][2]=x[i][0]                  #time
            M[i][3]=y[i][0]                  #N2O
            M[i][4]=fsr_peak1_spec[i][0]     #
            M[i][5]=fsr_peak4_TC[i][0]
            M[i][6]=fsr_peak5_TC[i][0]
            M[i][7]=fsr_peak6_TC[i][0]
            M[i][8]=co2_TC[i][0]
    
        N=list(chunks(M, p))
        for j in range(0, int(len(data_y)/n/p)):
            slope_index=N[j][:,0].argmin()
            choose[j][0]=slope_index      # choose data index position 
            choose[j][1]=min(N[j][:,0])   #choose data slope
            choose[j][2]=N[j][slope_index][2]      #choose data first x value
            choose[j][3]=N[j][slope_index][3]      #choose data first y value
            choose[j][4]=N[j][slope_index][1]      #choose data interception
            choose[j][5]=N[j][slope_index][4]      #fsr_peak1_spec
            choose[j][6]=N[j][slope_index][5]      #fsr_peak4_TC
            choose[j][7]=N[j][slope_index][6]      #fsr_peak5_TC
            choose[j][8]=N[j][slope_index][7]      #fsr_peak6_TC
            choose[j][9]=N[j][slope_index][8]      #co2_TC
    
        plt.scatter(choose[:,2], choose[:,3], color='red')

        plt.figure(2)
        plt.plot(choose[:,2], choose[:,3],'r+')
        #plt.plot(t, fsr_peak1_avg_fit(t), 'r-')
        plt.ylabel('choose_y', fontsize=8)   

        fsr_peak4_fit = np.poly1d(np.polyfit(choose[:,5], choose[:,6], 3))
        fsr_peak5_fit = np.poly1d(np.polyfit(choose[:,5], choose[:,7], 3))
        fsr_peak6_fit = np.poly1d(np.polyfit(choose[:,5], choose[:,8], 3))
        co2_raw_fit = np.poly1d(np.polyfit(choose[:,5], choose[:,9], 2))    

        t = np.linspace(min(choose[:,5]), max(choose[:,5]), 100)

        plt.figure(3)
        plt.subplot(421)
        plt.plot(choose[:,5], choose[:,6],'o')
        plt.plot(t, fsr_peak4_fit(t), 'r-')
        plt.ylabel('fsr_peak4', fontsize=8)
        plt.subplot(422)
        plt.plot(choose[:,5], choose[:,6]-fsr_peak4_fit(choose[:,5]),'o')
        plt.axhline(y=0.0, color='r', linestyle='-')
        plt.ylabel('peak4_residual', fontsize=8)
        plt.subplot(423)
        plt.plot(choose[:,5], choose[:,7],'o')
        plt.plot(t, fsr_peak5_fit(t), 'r-')
        plt.ylabel('fsr_peak5', fontsize=8)
        plt.subplot(424)
        plt.plot(choose[:,5], choose[:,7]-fsr_peak5_fit(choose[:,5]),'o')
        plt.axhline(y=0.0, color='r', linestyle='-')
        plt.ylabel('peak5_residual', fontsize=8)
        plt.subplot(425)
        plt.plot(choose[:,5], choose[:,8],'o')
        plt.plot(t, fsr_peak6_fit(t), 'r-')
        plt.ylabel('fsr_peak6', fontsize=8)
        plt.subplot(426)
        plt.plot(choose[:,5], choose[:,8]-fsr_peak6_fit(choose[:,5]),'o')
        plt.axhline(y=0.0, color='r', linestyle='-')
        plt.ylabel('peak6_residual', fontsize=8)
        plt.subplot(427)
        plt.plot(choose[:,5], choose[:,9],'o')
        plt.plot(t, co2_raw_fit(t), 'r-')
        plt.ylabel('co2_raw', fontsize=8)
        plt.subplot(428)
        plt.plot(choose[:,5], choose[:,9]-co2_raw_fit(choose[:,5]),'o')
        plt.axhline(y=0.0, color='r', linestyle='-')
        plt.ylabel('co2_raw_residual', fontsize=8)
        plt.show()
        plt.savefig("Fitting.png")

        with open('JDDS_1D_output_parameters.txt', 'w') as out_file:
            out_string='Fitting parameters (sign reversed), RESTART analyzer after updating the FitterConfig file\n'
            out_string+='Offset4  =         '+str(-1*fsr_peak4_fit[0])
            out_string+='\n'   
            out_string+='Peak4_N2O_bilin  = '+str(-1*fsr_peak4_fit[2])
            out_string+='\n'  
            out_string+='Peak4_N2O_cubic  = '+str(-1*fsr_peak4_fit[3])
            out_string+='\n'  
            out_string+='Offset5  =         '+str(-1*fsr_peak5_fit[0])
            out_string+='\n'  
            out_string+='Peak5_N2O_bilin  = '+str(-1*fsr_peak5_fit[2])
            out_string+='\n'      
            out_string+='Peak5_N2O_cubic  = ' +str(-1*fsr_peak5_fit[3])
            out_string+='\n'   
            out_string+='Offset6  =         ' +str(-1*fsr_peak6_fit[0])
            out_string+='\n'  
            out_string+='Peak6_N2O_bilin  = ' +str(-1*fsr_peak6_fit[2])
            out_string+='\n'  
            out_string+='Peak6_N2O_cubic  = ' +str(-1*fsr_peak6_fit[3])
            out_string+='\n'  
            out_string+='CO2_offset  =      ' +str(-1*co2_raw_fit[0])
            out_string+='\n'   
            out_string+='CO2_N2O_lin  =     ' +str(-1*co2_raw_fit[1])
            out_string+='\n'   
            out_string+='CO2_N2O_bilin  =   ' +str(-1*co2_raw_fit[2])
            out_string+='\n'     
            out_file.write(out_string)
            
elif choice == 8:
        print ("You have chosen JDDS_2D_N2O-CO2_test.")
        Temp = np.zeros((len(df.N2O), 7))
        Temp[:, 0]=df.timestamp
        Temp[:, 1]=df.N2O
        Temp[:, 2]=df.fsr_peak1_spec
        Temp[:, 3]=df.fsr_peak4_spec
        Temp[:, 4]=df.fsr_peak5_spec
        Temp[:, 5]=df.fsr_peak6_spec
        Temp[:, 6]=df.co2_corr

        data=avg(Temp)

        data_x = np.array(data[:,0])
        data_y = np.array(data[:,1])
        data_fsr_peak1_spec   = np.array(data[:,2])
        data_fsr_peak4_spec   = np.array(data[:,3])
        data_fsr_peak5_spec   = np.array(data[:,4])
        data_fsr_peak6_spec   = np.array(data[:,5])
        data_co2_corr         = np.array(data[:,6])

  
        x = list(chunks(data_x, n)) 
        y = list(chunks(data_y, n)) 
        fsr_peak1_spec = list(chunks(data_fsr_peak1_spec, n))
        fsr_peak4_spec = list(chunks(data_fsr_peak4_spec, n))
        fsr_peak5_spec = list(chunks(data_fsr_peak5_spec, n))
        fsr_peak6_spec = list(chunks(data_fsr_peak6_spec, n))
        co2_corr       = list(chunks(data_co2_corr, n))



        M=np.zeros((int(len(data_y)/n), 9))
        choose=np.zeros((int(len(data_y)/n/p), 10))

#co2_corr   = co2_TC  + co2_offset + co2_N1 * fsr_peak1_spec +  co2_N2 *fsr_peak1_spec**2
#fsr_peak4_TC = fsr_peak4_scaled + P4_TC1*PTemp_Offseted + P4_TC2*PTemp_Offseted**2 + P4_TC3*PTemp_Offseted**3 + P4_TC4*PTemp_Offseted**4  + P4_TC5*PTemp_Offseted**5   # temperature correction
#fsr_peak4_spec = fsr_peak4_TC + offset4 + P4_N2*fsr_peak1_spec**2 + P4_N3*fsr_peak1_spec**3          # N2O correction
#fsr_peak4_spec += P4_A1*co2_corr + P4_A2*co2_corr**2 + P4_A1N2*fsr_peak1_spec*co2_corr   # CO2 correction
#23222*(col(fsrpeak4tc)+(FitAll_Pk4[1]+FitAll_Pk4[3]*col(fsrpeak1spec)^2+FitAll_Pk4[4]*col(fsrpeak1spec)^3+FitAll_Pk4[5]*col(co2cr)))/col(fsrpeak1spec)-1000
        for i in range(0, int(len(data_y)/n)):
            xx_i=np.array(x[i])
            yy_i=np.array(y[i])
            xx=xx_i.reshape(len(x[i]), 1)
            yy=yy_i.reshape(len(y[i]), 1)
            regr = linear_model.LinearRegression()
            regr.fit(xx, yy)
            plt.scatter(xx_i, yy_i, color='black')
            plt.plot(xx, regr.predict(xx), color='blue', linewidth=3)
            M[i][0]=abs(regr.coef_)
            M[i][1]=regr.intercept_
            M[i][2]=x[i][0]                  #time
            M[i][3]=y[i][0]                  #N2O
            M[i][4]=fsr_peak1_spec[i][0]     #
            M[i][5]=fsr_peak4_spec[i][0]
            M[i][6]=fsr_peak5_spec[i][0]
            M[i][7]=fsr_peak6_spec[i][0]
            M[i][8]=co2_corr[i][0]
    
        N=list(chunks(M, p))
        for j in range(0, int(len(data_y)/n/p)):
            slope_index=N[j][:,0].argmin()
            choose[j][0]=slope_index      # choose data index position 
            choose[j][1]=min(N[j][:,0])   #choose data slope
            choose[j][2]=N[j][slope_index][2]      #choose data first x value
            choose[j][3]=N[j][slope_index][3]      #choose data first y value
            choose[j][4]=N[j][slope_index][1]      #choose data interception
            choose[j][5]=N[j][slope_index][4]      #fsr_peak1_spec
            choose[j][6]=N[j][slope_index][5]      #fsr_peak4_spec
            choose[j][7]=N[j][slope_index][6]      #fsr_peak5_spec
            choose[j][8]=N[j][slope_index][7]      #fsr_peak6_spec
            choose[j][9]=N[j][slope_index][8]      #co2_corr

        plt.figure(1) 
        plt.subplot(211)   
        plt.scatter(choose[:,2], choose[:,3], color='red')
        plt.xlabel('time', fontsize=8)
        plt.ylabel('N2O', fontsize=8)
        plt.subplot(212)
        plt.scatter(choose[:,2], choose[:,9], color='blue')
        plt.xlabel('time', fontsize=8)
        plt.ylabel('co2_corr', fontsize=8)

        fsr_peak4_corr_fit = np.poly1d(np.polyfit(choose[:,9], choose[:,6], 1))    #linear fit of fsr_peak4 Vs co2_corr 
        fsr_peak5_corr_fit = np.poly1d(np.polyfit(choose[:,9], choose[:,7], 1))    #linear fit of fsr_peak5 Vs co2_corr
        fsr_peak6_corr_fit = np.poly1d(np.polyfit(choose[:,9], choose[:,8], 1))    #linear fit of fsr_peak6 Vs co2_corr

        fsr_peak4_cr_fit = np.poly1d(np.polyfit(choose[:,9], 23222*choose[:,6]/choose[:,5]-1000, 1))
        fsr_peak5_cr_fit = np.poly1d(np.polyfit(choose[:,9], 23222*choose[:,7]/choose[:,5]-1000, 1))
        fsr_peak6_cr_fit = np.poly1d(np.polyfit(choose[:,9], 23222*choose[:,8]/choose[:,5]-1000, 1))

        fsr_peak4_N2O_fit = np.poly1d(np.polyfit(choose[:,3], 23222*choose[:,6]/choose[:,5]-1000, 1))
        fsr_peak5_N2O_fit = np.poly1d(np.polyfit(choose[:,3], 23222*choose[:,7]/choose[:,5]-1000, 1))
        fsr_peak6_N2O_fit = np.poly1d(np.polyfit(choose[:,3], 23222*choose[:,8]/choose[:,5]-1000, 1))

        print('fsr_peak4_co2_valid=' + str(fsr_peak4_cr_fit[1]))
        print('fsr_peak5_c02_valid=' + str(fsr_peak5_cr_fit[1]))
        print('fsr_peak6_co2_valid=' + str(fsr_peak6_cr_fit[1]))
        print('fsr_peak4_N2O_valid=' +str(fsr_peak4_N2O_fit[1]))
        print('fsr_peak5_N2O_valid=' +str(fsr_peak5_N2O_fit[1]))
        print('fsr_peak6_N2O_valid=' +str(fsr_peak6_N2O_fit[1]))

        t_corr = np.linspace(min(choose[:,9]), max(choose[:,9]), 100)
        t_N2O  = np.linspace(min(choose[:,3]), max(choose[:,3]), 100)

        plt.figure(2)
        plt.subplot(311)
        plt.plot(choose[:,3], 23222*choose[:,6]/choose[:,5]-1000, 'o')
        plt.plot(t_N2O, fsr_peak4_N2O_fit(t_N2O), 'r-', label= 'fit slope='+str(round(fsr_peak4_N2O_fit[1],6)))
        plt.legend()
        plt.ylabel('fsr_peak4_N2O', fontsize=8)
        plt.subplot(312)
        plt.plot(choose[:,3], 23222*choose[:,7]/choose[:,5]-1000, 'o')
        plt.plot(t_N2O, fsr_peak5_N2O_fit(t_N2O), 'r-', label= 'fit slope='+str(round(fsr_peak5_N2O_fit[1],6)))
        plt.legend()
        plt.ylabel('fsr_peak5_N2O', fontsize=8)
        plt.subplot(313)
        plt.plot(choose[:,3], 23222*choose[:,8]/choose[:,5]-1000, 'o')
        plt.plot(t_N2O, fsr_peak6_N2O_fit(t_N2O), 'r-', label= 'fit slope='+str(round(fsr_peak6_N2O_fit[1],6)))
        plt.legend()
        plt.xlabel('N2O', fontsize=8)
        plt.ylabel('fsr_peak6_N2O', fontsize=8)
        plt.savefig('fsr_peak-N2O_2Dtest.png')

        plt.figure(3)
        plt.subplot(321)
        plt.plot(choose[:,9], choose[:,6], 'o')
        plt.plot(choose[:,9], fsr_peak4_corr_fit(choose[:,9]), 'r-')
        plt.ylabel('fsr_peak4', fontsize=8)
        plt.subplot(322)
        plt.plot(choose[:,9], 23222*choose[:,6]/choose[:,5]-1000, 'o')
        plt.plot(t_corr, fsr_peak4_cr_fit(t_corr), 'r-', label= 'fit slope=' + str(round(fsr_peak4_cr_fit[1],6)))
        plt.legend()
        plt.ylabel('fsr_peak4_cr', fontsize=8)
        plt.subplot(323)
        plt.plot(choose[:,9], choose[:,7], 'o')
        plt.plot(choose[:,9], fsr_peak5_corr_fit(choose[:,9]), 'r-')
        plt.ylabel('fsr_peak5', fontsize=8)
        plt.subplot(324)
        plt.plot(choose[:,9], 23222*choose[:,7]/choose[:,5]-1000, 'o')
        plt.plot(t_corr, fsr_peak5_cr_fit(t_corr), 'r-', label= 'fit slope=' + str(round(fsr_peak5_cr_fit[1],6)))
        plt.legend()
        plt.ylabel('fsr_peak5_cr', fontsize=8)
        plt.subplot(325)
        plt.plot(choose[:,9], choose[:,8], 'o')
        plt.plot(choose[:,9], fsr_peak6_corr_fit(choose[:,9]), 'r-')
        plt.xlabel('co2_corr', fontsize=8)
        plt.ylabel('fsr_peak6', fontsize=8)
        plt.subplot(326)
        plt.plot(choose[:,9], 23222*choose[:,8]/choose[:,5]-1000, 'o')
        plt.plot(t_corr, fsr_peak6_cr_fit(t_corr), 'r-', label= 'fit slope=' + str(round(fsr_peak6_cr_fit[1],6)))
        plt.legend()
        plt.xlabel('co2_corr', fontsize=8)
        plt.ylabel('fsr_peak6_cr', fontsize=8)
        plt.savefig('fsr_peak-co2_corr_2Dtest.png')

        with open('JDDS_2D_output_parameters.txt', 'w') as out_file:
            out_string='Fitting parameters (sign reversed)\n'
            out_string+='Peak4_CO2_lin  =     ' +str(-1*fsr_peak4_corr_fit[1])
            out_string+='\n'   
            out_string+='Peak5_CO2_lin  =     ' +str(-1*fsr_peak5_corr_fit[1])
            out_string+='\n'  
            out_string+='Peak6_CO2_lin  =     ' +str(-1*fsr_peak6_corr_fit[1])
            out_string+='\n'      
            out_file.write(out_string)

elif choice == 9:
        print ("You have chosen JDDS_2D_N2O-CO2_validation.")
        Temp = np.zeros((len(df.N2O), 10))
        Temp[:, 0]=df.timestamp
        Temp[:, 1]=df.N2O
        Temp[:, 2]=df.fsr_peak1_spec
        Temp[:, 3]=df.fsr_peak4_spec
        Temp[:, 4]=df.fsr_peak5_spec
        Temp[:, 5]=df.fsr_peak6_spec
        Temp[:, 6]=df.co2_corr
        Temp[:, 7]=df.d15Nalpha
        Temp[:, 8]=df.d15Nbeta
        Temp[:, 9]=df.d18O

        data=avg(Temp)

        data_x = np.array(data[:,0])
        data_y = np.array(data[:,1])
        data_fsr_peak1_spec   = np.array(data[:,2])
        data_fsr_peak4_spec   = np.array(data[:,3])
        data_fsr_peak5_spec   = np.array(data[:,4])
        data_fsr_peak6_spec   = np.array(data[:,5])
        data_co2_corr         = np.array(data[:,6])
        data_d15Nalpha        = np.array(data[:,7])
        data_d15Nbeta         = np.array(data[:,8])
        data_d18O             = np.array(data[:,9])

        x = list(chunks(data_x, n)) 
        y = list(chunks(data_y, n)) 
        fsr_peak1_spec = list(chunks(data_fsr_peak1_spec, n))
        fsr_peak4_spec = list(chunks(data_fsr_peak4_spec, n))
        fsr_peak5_spec = list(chunks(data_fsr_peak5_spec, n))
        fsr_peak6_spec = list(chunks(data_fsr_peak6_spec, n))
        co2_corr       = list(chunks(data_co2_corr, n))
        d15Nalpha      = list(chunks(data_d15Nalpha, n))
        d15Nbeta       = list(chunks(data_d15Nbeta, n))
        d18O           = list(chunks(data_d18O, n))

        M=np.zeros((int(len(data_y)/n), 12))
        choose=np.zeros((int(len(data_y)/n/p), 13))

        for i in range(0, int(len(data_y)/n)):
            xx_i=np.array(x[i])
            yy_i=np.array(y[i])
            xx=xx_i.reshape(len(x[i]), 1)
            yy=yy_i.reshape(len(y[i]), 1)
            regr = linear_model.LinearRegression()
            regr.fit(xx, yy)
            plt.scatter(xx_i, yy_i, color='black')
            plt.plot(xx, regr.predict(xx), color='blue', linewidth=3)
            M[i][0]=abs(regr.coef_)
            M[i][1]=regr.intercept_
            M[i][2]=x[i][0]                  #time
            M[i][3]=y[i][0]                  #N2O
            M[i][4]=fsr_peak1_spec[i][0]     #
            M[i][5]=fsr_peak4_spec[i][0]
            M[i][6]=fsr_peak5_spec[i][0]
            M[i][7]=fsr_peak6_spec[i][0]
            M[i][8]=co2_corr[i][0]
            M[i][9]=d15Nalpha[i][0]
            M[i][10]=d15Nbeta[i][0]
            M[i][11]=d18O[i][0]
    
        N=list(chunks(M, p))
        for j in range(0, int(len(data_y)/n/p)):
            slope_index=N[j][:,0].argmin()
            choose[j][0]=slope_index      # choose data index position 
            choose[j][1]=min(N[j][:,0])   #choose data slope
            choose[j][2]=N[j][slope_index][2]      #choose data first x value
            choose[j][3]=N[j][slope_index][3]      #choose data first y value
            choose[j][4]=N[j][slope_index][1]      #choose data interception
            choose[j][5]=N[j][slope_index][4]      #fsr_peak1_spec
            choose[j][6]=N[j][slope_index][5]      #fsr_peak4_spec
            choose[j][7]=N[j][slope_index][6]      #fsr_peak5_spec
            choose[j][8]=N[j][slope_index][7]      #fsr_peak6_spec
            choose[j][9]=N[j][slope_index][8]      #co2_corr
            choose[j][10]=N[j][slope_index][9]     #d15Nalpha
            choose[j][11]=N[j][slope_index][10]     #d15Nbeta
            choose[j][12]=N[j][slope_index][11]     #d18O
    

        plt.figure(1) 
        plt.subplot(211)   
        plt.scatter(choose[:,2], choose[:,3], color='red')
        plt.xlabel('time', fontsize=8)
        plt.ylabel('N2O', fontsize=8)
        plt.subplot(212)
        plt.scatter(choose[:,2], choose[:,9], color='blue')
        plt.xlabel('time', fontsize=8)
        plt.ylabel('co2_corr', fontsize=8)

        d15Nalpha_corr_fit = np.poly1d(np.polyfit(choose[:,9], choose[:,10], 1))         #linear fit of d15Nalpha Vs co2_corr 
        d15Nbeta_corr_fit  = np.poly1d(np.polyfit(choose[:,9], choose[:,11], 1))         #linear fit of d15Nbeta Vs co2_corr
        d18O_corr_fit      = np.poly1d(np.polyfit(choose[:,9], choose[:,12], 1))         #linear fit of d18O Vs co2_corr

        d15Nalpha_N2O_fit = np.poly1d(np.polyfit(choose[:,3], choose[:,10], 1))          #linear fit of d15Nalpha Vs N2O 
        d15Nbeta_N2O_fit  = np.poly1d(np.polyfit(choose[:,3], choose[:,11], 1))          #linear fit of d15Nbeta Vs N2O
        d18O_N2O_fit      = np.poly1d(np.polyfit(choose[:,3], choose[:,12], 1))          #linear fit of d18O Vs N2O

        print('d15Nalpha_corr_valid=' + str(d15Nalpha_corr_fit[1]))
        print('d15Nbeta_corr_valid=' + str(d15Nbeta_corr_fit[1]))
        print('d18O_corr_valid=' + str(d18O_corr_fit[1]))
        print('d15Nalpha_N2O_valid=' +str(d15Nalpha_N2O_fit[1]))
        print('d15Nbeta_N2O_valid=' +str(d15Nbeta_N2O_fit[1]))
        print('d18O_N2O_valid=' +str(d18O_N2O_fit[1]))

        t_corr = np.linspace(min(choose[:,9]), max(choose[:,9]), 100)
        t_N2O  = np.linspace(min(choose[:,3]), max(choose[:,3]), 100)


        plt.figure(2)
        plt.subplot(321)
        plt.plot(choose[:,3], choose[:,10], 'o')
        plt.plot(t_N2O, d15Nalpha_N2O_fit(t_N2O), 'r-', label= 'fit slope='+str(round(d15Nalpha_N2O_fit[1],6)))
        plt.legend()
        plt.ylabel('d15Nalpha_N2O', fontsize=8)
        plt.subplot(322)
        plt.plot(choose[:,9], choose[:,10], 'o')
        plt.plot(t_corr, d15Nalpha_corr_fit(t_corr), 'r-', label= 'fit slope='+str(round(d15Nalpha_corr_fit[1],6)))
        plt.legend()
        plt.ylabel('d15Nalpha_corr', fontsize=8)
        plt.subplot(323)
        plt.plot(choose[:,3], choose[:,11], 'o')
        plt.plot(t_N2O, d15Nbeta_N2O_fit(t_N2O), 'r-', label= 'fit slope='+str(round(d15Nbeta_N2O_fit[1],6)))
        plt.legend()
        plt.ylabel('d15Nbeta_N2O', fontsize=8)
        plt.subplot(324)
        plt.plot(choose[:,9], choose[:,11], 'o')
        plt.plot(t_corr, d15Nbeta_corr_fit(t_corr), 'r-', label= 'fit slope='+str(round(d15Nbeta_corr_fit[1],6)))
        plt.legend()
        plt.ylabel('d15Nbeta_corr', fontsize=8)
        plt.subplot(325)
        plt.plot(choose[:,3], choose[:,12], 'o')
        plt.plot(t_N2O, d18O_N2O_fit(t_N2O), 'r-', label= 'fit slope='+str(round(d18O_N2O_fit[1],6)))
        plt.legend()
        plt.xlabel('N2O', fontsize=8)
        plt.ylabel('d18O_N2O', fontsize=8)
        plt.subplot(326)
        plt.plot(choose[:,9], choose[:,12], 'o')
        plt.plot(t_corr, d18O_corr_fit(t_corr), 'r-', label= 'fit slope='+str(round(d18O_corr_fit[1],6)))
        plt.legend()
        plt.xlabel('co2_corr', fontsize=8)
        plt.ylabel('d18O_corr', fontsize=8)
        plt.savefig('JDDS_2D_valid.png')
        
elif choice == 10:
        print ("You have chosen JBDS_1D_N2O.")

        Temp = np.zeros((len(df.N2O), 7))
        Temp[:, 0]=df.timestamp
        Temp[:, 1]=df.N2O
        Temp[:, 2]=df.fsr_peak1_TC
        Temp[:, 3]=df.fsr_peak10_TC
        Temp[:, 4]=df.fsr_peak11_TC
        Temp[:, 5]=df.fsr_str20
        Temp[:, 6]=df.co2_TC

        data=avg(Temp)

        data_x = np.array(data[:,0])
        data_y = np.array(data[:,1])
        data_fsr_peak1_TC  = np.array(data[:,2])
        data_fsr_peak10_TC = np.array(data[:,3])
        data_fsr_peak11_TC = np.array(data[:,4])
        data_fsr_str20     = np.array(data[:,5])
        data_co2_TC        = np.array(data[:,6])
  
        x = list(chunks(data_x, n)) 
        y = list(chunks(data_y, n)) 
        fsr_peak1_TC = list(chunks(data_fsr_peak1_TC, n))
        fsr_peak10_TC = list(chunks(data_fsr_peak10_TC, n))
        fsr_peak11_TC = list(chunks(data_fsr_peak11_TC, n))
        fsr_str20    = list(chunks(data_fsr_str20, n))
        co2_TC       = list(chunks(data_co2_TC, n))



        M=np.zeros((int(len(data_y)/n), 9))
        choose=np.zeros((int(len(data_y)/n/p), 10))

        for i in range(0, int(len(data_y)/n)):
            xx_i=np.array(x[i])
            yy_i=np.array(y[i])
            xx=xx_i.reshape(len(x[i]), 1)
            yy=yy_i.reshape(len(y[i]), 1)
            regr = linear_model.LinearRegression()
            regr.fit(xx, yy)
            plt.scatter(xx_i, yy_i, color='black')
            plt.plot(xx, regr.predict(xx), color='blue', linewidth=3)
            M[i][0]=abs(regr.coef_)
            M[i][1]=regr.intercept_
            M[i][2]=x[i][0]                  #time
            M[i][3]=y[i][0]                  #N2O
            M[i][4]=fsr_peak1_TC[i][0]     #
            M[i][5]=fsr_peak10_TC[i][0]
            M[i][6]=fsr_peak11_TC[i][0]
            M[i][7]=fsr_str20[i][0]
            M[i][8]=co2_TC[i][0]
    
        N=list(chunks(M, p))
        for j in range(0, int(len(data_y)/n/p)):
            slope_index=N[j][:,0].argmin()
            choose[j][0]=slope_index      # choose data index position 
            choose[j][1]=min(N[j][:,0])   #choose data slope
            choose[j][2]=N[j][slope_index][2]      #choose data first x value   #time
            choose[j][3]=N[j][slope_index][3]      #choose data first y value   #N2O
            choose[j][4]=N[j][slope_index][1]      #choose data interception
            choose[j][5]=N[j][slope_index][4]      #fsr_peak1_TC
            choose[j][6]=N[j][slope_index][5]      #fsr_peak10_TC
            choose[j][7]=N[j][slope_index][6]      #fsr_peak11_TC
            choose[j][8]=N[j][slope_index][7]      #fsr_str20
            choose[j][9]=N[j][slope_index][8]      #co2_TC
    
        plt.scatter(choose[:,2], choose[:,3], color='red')

        plt.figure(2)
        plt.plot(choose[:,2], choose[:,3],'r+')
        plt.ylabel('choose_y(N2O)', fontsize=8)   

        fsr_peak10_fit = np.poly1d(np.polyfit(choose[:,5], choose[:,6], 3))
        fsr_peak11_fit = np.poly1d(np.polyfit(choose[:,5], choose[:,7], 3))
        co2_TC_fit = np.poly1d(np.polyfit(choose[:,5], choose[:,9], 2))    

        t = np.linspace(min(choose[:,5]), max(choose[:,5]), 100)

        plt.figure(3)
        plt.subplot(321)
        plt.plot(choose[:,5], choose[:,6],'o')
        plt.plot(t, fsr_peak10_fit(t), 'r-')
        plt.ylabel('fsr_peak10', fontsize=8)
        plt.subplot(322)
        plt.plot(choose[:,5], choose[:,6]-fsr_peak10_fit(choose[:,5]),'o')
        plt.axhline(y=0.0, color='r', linestyle='-')
        plt.ylabel('peak10_residual', fontsize=8)
        plt.subplot(323)
        plt.plot(choose[:,5], choose[:,7],'o')
        plt.plot(t, fsr_peak11_fit(t), 'r-')
        plt.ylabel('fsr_peak11', fontsize=8)
        plt.subplot(324)
        plt.plot(choose[:,5], choose[:,7]-fsr_peak11_fit(choose[:,5]),'o')
        plt.axhline(y=0.0, color='r', linestyle='-')
        plt.ylabel('peak11_residual', fontsize=8)
        plt.subplot(325)
        plt.plot(choose[:,5], choose[:,9],'o')
        plt.plot(t, co2_TC_fit(t), 'r-')
        plt.ylabel('co2_TC', fontsize=8)
        plt.subplot(326)
        plt.plot(choose[:,5], choose[:,9]-co2_TC_fit(choose[:,5]),'o')
        plt.axhline(y=0.0, color='r', linestyle='-')
        plt.ylabel('co2_TC_residual', fontsize=8)
        plt.show()
        plt.savefig("JBDS_1D_N2O.png")

        with open('JBDS_1D_N2O.txt', 'w') as out_file:
            out_string='Fitting parameters (sign reversed), RESTART analyzer after updating the FitterConfig file\n'
            out_string+='Offset10         = '+str(-1*fsr_peak10_fit[0])
            out_string+='\n'   
            out_string+='Peak10_N2O_bilin = '+str(-1*fsr_peak10_fit[2])
            out_string+='\n'  
            out_string+='Peak10_N2O_cubic  = '+str(-1*fsr_peak10_fit[3])
            out_string+='\n'  
            out_string+='Offset11          = '+str(-1*fsr_peak11_fit[0])
            out_string+='\n'  
            out_string+='Peak11_N2O_bilin  = '+str(-1*fsr_peak11_fit[2])
            out_string+='\n'      
            out_string+='Peak11_N2O_cubic  = '+str(-1*fsr_peak11_fit[3])
            out_string+='\n'    
            out_string+='CO2_offset  =      ' +str(-1*co2_TC_fit[0])
            out_string+='\n'   
            out_string+='CO2_N2O_lin  =     ' +str(-1*co2_TC_fit[1])
            out_string+='\n'   
            out_string+='CO2_N2O_bilin  =   ' +str(-1*co2_TC_fit[2])
            out_string+='\n'     
            out_file.write(out_string)
            
elif choice == 11:
        print ("You have chosen JBDS_2D_N2O-CO2_test.")
        Temp = np.zeros((len(df.N2O), 8))
        Temp[:, 0]=df.timestamp
        Temp[:, 1]=df.N2O
        Temp[:, 2]=df.fsr_peak1_dry
        Temp[:, 3]=df.fsr_peak10_spec
        Temp[:, 4]=df.fsr_peak11_spec
        Temp[:, 5]=df.fsr_str20
        Temp[:, 6]=df.co2_corr
        Temp[:, 7]=df.fsr_peak1_TC

        data=avg(Temp)

        data_x = np.array(data[:,0])
        data_y = np.array(data[:,1])
        data_fsr_peak1_dry    = np.array(data[:,2])
        data_fsr_peak10_spec   = np.array(data[:,3])
        data_fsr_peak11_spec   = np.array(data[:,4])
        data_fsr_str20         = np.array(data[:,5])
        data_co2_corr          = np.array(data[:,6])
        data_fsr_peak1_TC    = np.array(data[:,7])
  
        x = list(chunks(data_x, n)) 
        y = list(chunks(data_y, n)) 
        fsr_peak1_dry  = list(chunks(data_fsr_peak1_dry, n))
        fsr_peak10_spec = list(chunks(data_fsr_peak10_spec, n))
        fsr_peak11_spec = list(chunks(data_fsr_peak11_spec, n))
        fsr_str20       = list(chunks(data_fsr_str20, n))
        co2_corr        = list(chunks(data_co2_corr, n))
        fsr_peak1_TC  = list(chunks(data_fsr_peak1_TC, n))



        M=np.zeros((int(len(data_y)/n), 10))
        choose=np.zeros((int(len(data_y)/n/p), 12))

# Cross-talk corrections 
#co2_corr   = co2_TC  + A_offset + A_N1 * fsr_peak1_TC +  A_N2 *fsr_peak1_TC**2                          # after 1D N2O test.  Fit of "co2_TC vs fsr_peak1_TC"
#co2_corr  += A_H1*fsr_str20 + A_H2*fsr_str20*fsr_str20 + A_N1H1*fsr_peak1_TC*fsr_str20                  # after 2D N2O-H2O test.
#fsr_peak10_spec =  fsr_peak10_TC + Offset10 + P10_N2*fsr_peak1_TC**2 + P10_N3*fsr_peak1_TC**3           # after 1D N2O test.  Fit of "fsr_peak10_TC vs fsr_peak1_TC"
#fsr_peak10_spec += P10_A1*co2_corr + P10_A1N1*co2_corr*fsr_peak1_TC                                     # after 2D N2O-CO2 test.  Fit of "fsr_peak10_TC vs co2_corr"
#fsr_peak10_spec += P10_H1*fsr_str20 + P10_H2*fsr_str20*fsr_str20 + P10_N1H1*fsr_peak10_TC*fsr_str20 + P10_N1H2*fsr_peak10_TC*fsr_str20*fsr_str20 # after 2D N2O-H2O test.  Fit of "fsr_peak10_TC vs fsr_str20"
    
#fsr_peak11_spec =  fsr_peak11_TC + Offset11 + P11_N2*fsr_peak1_TC**2 + P11_N3*fsr_peak1_TC**3           # after 1D N2O test.  Fit of "fsr_peak10_TC vs fsr_peak1_TC"
#fsr_peak11_spec += P11_A1*co2_corr + P11_A1N1*co2_corr*fsr_peak1_TC                                     # after 2D N2O-CO2 test.  Fit of "fsr_peak10_TC vs co2_corr"
#fsr_peak11_spec += P11_H1*fsr_str20 + P11_H2*fsr_str20*fsr_str20 + P11_N1H1*fsr_peak11_TC*fsr_str20 + P11_N1H2*fsr_peak11_TC*fsr_str20*fsr_str20 # after 2D N2O-H2O test.  Fit of "fsr_peak10_TC vs fsr_str20"
    
        for i in range(0, int(len(data_y)/n)):
            xx_i=np.array(x[i])
            yy_i=np.array(y[i])
            xx=xx_i.reshape(len(x[i]), 1)
            yy=yy_i.reshape(len(y[i]), 1)
            regr = linear_model.LinearRegression()
            regr.fit(xx, yy)
            plt.scatter(xx_i, yy_i, color='black')
            plt.plot(xx, regr.predict(xx), color='blue', linewidth=3)
            M[i][0]=abs(regr.coef_)
            M[i][1]=regr.intercept_
            M[i][2]=x[i][0]                  #time
            M[i][3]=y[i][0]                  #N2O
            M[i][4]=fsr_peak1_dry[i][0]     #
            M[i][5]=fsr_peak10_spec[i][0]
            M[i][6]=fsr_peak11_spec[i][0]
            M[i][7]=fsr_str20[i][0]
            M[i][8]=co2_corr[i][0]
            M[i][9]=fsr_peak1_TC[i][0]
    
        N=list(chunks(M, p))
        for j in range(0, int(len(data_y)/n/p)):
            slope_index=N[j][:,0].argmin()
            choose[j][0]=slope_index      # choose data index position 
            choose[j][1]=min(N[j][:,0])   #choose data slope
            choose[j][2]=N[j][slope_index][2]      #choose data first x value
            choose[j][3]=N[j][slope_index][3]      #choose data first y value
            choose[j][4]=N[j][slope_index][1]      #choose data interception
            choose[j][5]=N[j][slope_index][4]      #fsr_peak1_spec
            choose[j][6]=N[j][slope_index][5]      #fsr_peak10_spec
            choose[j][7]=N[j][slope_index][6]      #fsr_peak11_spec
            choose[j][8]=N[j][slope_index][7]      #fsr_str20
            choose[j][9]=N[j][slope_index][8]      #co2_corr
            choose[j][10]=N[j][slope_index][9]     ##fsr_peak1_TC
            choose[j][11]=N[j][slope_index][8]*N[j][slope_index][9]   ##co2_corr*fsr_peak1_TC

        plt.figure(1) 
        plt.subplot(211)   
        plt.scatter(choose[:,2], choose[:,3], color='red')
        plt.xlabel('time', fontsize=8)
        plt.ylabel('N2O', fontsize=8)
        plt.subplot(212)
        plt.scatter(choose[:,2], choose[:,9], color='blue')
        plt.xlabel('time', fontsize=8)
        plt.ylabel('co2_corr', fontsize=8)

        y_pk10 = choose[:,6]
        y_pk11 = choose[:,7] 
        x = np.array([choose[:,10],choose[:,9],choose[:,11]])
       
        
        clf_pk10 = linear_model.LinearRegression()
        clf_pk11 = linear_model.LinearRegression()
        
        x=x.transpose()    #x[0]:fsr_peak1_TC;   x[1]:co2_corr;   x[2]: co2_corr*fsr_peak1_TC
        clf_pk10.fit(x, y_pk10)
        clf_pk11.fit(x, y_pk11)
        
        print(clf_pk10.coef_)  
        print(clf_pk10.intercept_)
        print(clf_pk11.coef_)  
        print(clf_pk11.intercept_)

        Alpha_cr = 17120*(choose[:,6]+(-1)*clf_pk10.coef_[1]*choose[:,9]+(-1)*clf_pk10.coef_[2]*choose[:,11] )/choose[:,10]-1000    # use only when analyzer is restarted after 1D test
        Beta_cr = 17570*(choose[:,7]+(-1)*clf_pk11.coef_[1]*choose[:,9]+(-1)*clf_pk11.coef_[2]*choose[:,11] )/choose[:,10]-1000     # use only when analyzer is restarted after 1D test
 
        Alpha_cr_N2O_fit        = np.poly1d(np.polyfit(choose[:,3], Alpha_cr, 1))    #linear fit of Alpha_cr Vs N2O
        Beta_cr_co2_corr_fit    = np.poly1d(np.polyfit(choose[:,9], Beta_cr, 1))    #linear fit of Beta_cr Vs co2_corr
        Alpha_cr_co2_corr_fit   = np.poly1d(np.polyfit(choose[:,9], Alpha_cr, 1))    #linear fit of Alpha_cr Vs N2O
        Beta_cr_N2O_fit         = np.poly1d(np.polyfit(choose[:,3], Beta_cr, 1))    #linear fit of Beta_cr Vs co2_corr

        t_N2O      = np.linspace(min(choose[:,3]), max(choose[:,3]), 100)
        t_co2_corr = np.linspace(min(choose[:,9]), max(choose[:,9]), 100)

        plt.figure(2)
        plt.subplot(221)
        plt.plot(choose[:,3], Alpha_cr, 'o')
        plt.plot(t_N2O, Alpha_cr_N2O_fit(t_N2O), 'r-', label= 'fit slope='+str(round(Alpha_cr_N2O_fit[1],6)))
        plt.legend()
        plt.xlabel('N2O', fontsize=8)
        plt.ylabel('Alpha_cr_N2O', fontsize=8)
        plt.subplot(222)
        plt.plot(choose[:,9], Beta_cr, 'o')
        plt.plot(t_co2_corr, Beta_cr_co2_corr_fit(t_co2_corr), 'r-', label= 'fit slope='+str(round(Beta_cr_co2_corr_fit[1],6)))
        plt.legend()
        plt.ylabel('Beta_cr_co2_corr', fontsize=8)
        plt.xlabel('co2_corr', fontsize=8)
        plt.ylabel('Beta_cr_co2_corr', fontsize=8)
        plt.subplot(223)
        plt.plot(choose[:,9], Alpha_cr, 'o')
        plt.plot(t_co2_corr, Alpha_cr_co2_corr_fit(t_co2_corr), 'r-', label= 'fit slope='+str(round(Alpha_cr_co2_corr_fit[1],6)))
        plt.legend()
        plt.xlabel('co2_corr', fontsize=8)
        plt.ylabel('Alpha_cr_co2_corr', fontsize=8)
        plt.subplot(224)
        plt.plot(choose[:,3], Beta_cr, 'o')
        plt.plot(t_N2O, Beta_cr_N2O_fit(t_N2O), 'r-', label= 'fit slope='+str(round(Beta_cr_N2O_fit[1],6)))
        plt.legend()
        plt.ylabel('Beta_cr_N2O', fontsize=8)
        plt.xlabel('N2O', fontsize=8)
        plt.savefig('JBDS_N2O-CO2_2Dtest.png')
        
        with open('JBDS_2D_N2O-CO2_test.txt', 'w') as out_file:
            out_string='Fitting parameters (sign reversed).RESTART analyzer after updating the FitterConfig file for validation\n'
            out_string+='Peak10_CO2_lin        = ' +str(-1*clf_pk10.coef_[1])
            out_string+='\n'   
            out_string+='Peak10_CO2_N2O_bilin  = ' +str(-1*clf_pk10.coef_[2])
            out_string+='\n'  
            out_string+='Peak11_CO2_lin        = ' +str(-1*clf_pk11.coef_[1])
            out_string+='\n'   
            out_string+='Peak11_CO2_N2O_bilin  = ' +str(-1*clf_pk11.coef_[2])
            out_string+='\n'      
            out_file.write(out_string)

elif choice == 12:
        print ("You have chosen JBDS_2D_N2O-CO2_validation.")
        Temp = np.zeros((len(df.N2O), 10))
        Temp[:, 0]=df.timestamp
        Temp[:, 1]=df.N2O
        Temp[:, 2]=df.fsr_peak1_dry
        Temp[:, 3]=df.fsr_peak10_spec
        Temp[:, 4]=df.fsr_peak11_spec
        Temp[:, 5]=df.fsr_str20
        Temp[:, 6]=df.co2_corr
        Temp[:, 7]=df.fsr_peak1_TC
        Temp[:, 8]=df.d15Nalpha
        Temp[:, 9]=df.d15Nbeta
        
        data=avg(Temp)

        data_x = np.array(data[:,0])
        data_y = np.array(data[:,1])
        data_fsr_peak1_dry     = np.array(data[:,2])
        data_fsr_peak10_spec   = np.array(data[:,3])
        data_fsr_peak11_spec   = np.array(data[:,4])
        data_fsr_str20         = np.array(data[:,5])
        data_co2_corr          = np.array(data[:,6])
        data_fsr_peak1_TC      = np.array(data[:,7])
        data_d15Nalpha         = np.array(data[:,8])
        data_d15Nbeta          = np.array(data[:,9])
  
        x = list(chunks(data_x, n)) 
        y = list(chunks(data_y, n)) 
        fsr_peak1_dry   = list(chunks(data_fsr_peak1_dry, n))
        fsr_peak10_spec = list(chunks(data_fsr_peak10_spec, n))
        fsr_peak11_spec = list(chunks(data_fsr_peak11_spec, n))
        fsr_str20       = list(chunks(data_fsr_str20, n))
        co2_corr        = list(chunks(data_co2_corr, n))
        fsr_peak1_TC    = list(chunks(data_fsr_peak1_TC, n))
        d15Nalpha       = list(chunks(data_d15Nalpha, n))
        d15Nbeta        = list(chunks(data_d15Nbeta, n))


        M=np.zeros((int(len(data_y)/n), 12))
        choose=np.zeros((int(len(data_y)/n/p), 14))
    
        for i in range(0, int(len(data_y)/n)):
            xx_i=np.array(x[i])
            yy_i=np.array(y[i])
            xx=xx_i.reshape(len(x[i]), 1)
            yy=yy_i.reshape(len(y[i]), 1)
            regr = linear_model.LinearRegression()
            regr.fit(xx, yy)
            plt.scatter(xx_i, yy_i, color='black')
            plt.plot(xx, regr.predict(xx), color='blue', linewidth=3)
            M[i][0]=abs(regr.coef_)
            M[i][1]=regr.intercept_
            M[i][2]=x[i][0]                  #time
            M[i][3]=y[i][0]                  #N2O
            M[i][4]=fsr_peak1_dry[i][0]     #
            M[i][5]=fsr_peak10_spec[i][0]
            M[i][6]=fsr_peak11_spec[i][0]
            M[i][7]=fsr_str20[i][0]
            M[i][8]=co2_corr[i][0]
            M[i][9]=fsr_peak1_TC[i][0]
            M[i][10]=d15Nalpha[i][0]
            M[i][11]=d15Nbeta[i][0]
    
        N=list(chunks(M, p))
        for j in range(0, int(len(data_y)/n/p)):
            slope_index=N[j][:,0].argmin()
            choose[j][0]=slope_index      # choose data index position 
            choose[j][1]=min(N[j][:,0])   #choose data slope
            choose[j][2]=N[j][slope_index][2]      #choose data first x value
            choose[j][3]=N[j][slope_index][3]      #choose data first y value
            choose[j][4]=N[j][slope_index][1]      #choose data interception
            choose[j][5]=N[j][slope_index][4]      #fsr_peak1_spec
            choose[j][6]=N[j][slope_index][5]      #fsr_peak10_spec
            choose[j][7]=N[j][slope_index][6]      #fsr_peak11_spec
            choose[j][8]=N[j][slope_index][7]      #fsr_str20
            choose[j][9]=N[j][slope_index][8]      #co2_corr
            choose[j][10]=N[j][slope_index][9]     ##fsr_peak1_TC
            choose[j][11]=N[j][slope_index][8]*N[j][slope_index][9]   ##co2_corr*fsr_peak1_TC
            choose[j][12]=N[j][slope_index][10]      #d15Nalpha
            choose[j][13]=N[j][slope_index][11]     ##d15Nbeta

        plt.figure(1) 
        plt.subplot(211)   
        plt.scatter(choose[:,2], choose[:,3], color='red')
        plt.xlabel('time', fontsize=8)
        plt.ylabel('N2O', fontsize=8)
        plt.subplot(212)
        plt.scatter(choose[:,2], choose[:,9], color='blue')
        plt.xlabel('time', fontsize=8)
        plt.ylabel('co2_corr', fontsize=8)

 
        d15Nalpha_N2O_fit        = np.poly1d(np.polyfit(choose[:,3], choose[:,12], 1))    #linear fit of Alpha_cr Vs N2O
        d15Nbeta_co2_corr_fit    = np.poly1d(np.polyfit(choose[:,9], choose[:,13], 1))    #linear fit of Beta_cr Vs co2_corr
        d15Nalpha_co2_corr_fit   = np.poly1d(np.polyfit(choose[:,9], choose[:,12], 1))    #linear fit of Alpha_cr Vs N2O
        d15Nbeta_N2O_fit         = np.poly1d(np.polyfit(choose[:,3], choose[:,13], 1))    #linear fit of Beta_cr Vs co2_corr

        t_N2O      = np.linspace(min(choose[:,3]), max(choose[:,3]), 100)
        t_co2_corr = np.linspace(min(choose[:,9]), max(choose[:,9]), 100)

        plt.figure(2)
        plt.subplot(221)
        plt.plot(choose[:,3], choose[:,12], 'o')
        plt.plot(t_N2O, d15Nalpha_N2O_fit(t_N2O), 'r-', label= 'fit slope='+str(round(d15Nalpha_N2O_fit[1],6)))
        plt.legend()
        plt.xlabel('N2O', fontsize=8)
        plt.ylabel('d15Nalpha_N2O', fontsize=8)
        plt.subplot(224)
        plt.plot(choose[:,9], choose[:,13], 'o')
        plt.plot(t_co2_corr, d15Nbeta_co2_corr_fit(t_co2_corr), 'r-', label= 'fit slope='+str(round(d15Nbeta_co2_corr_fit[1],6)))
        plt.legend()
        plt.ylabel('d15Nbeta_co2_corr', fontsize=8)
        plt.xlabel('co2_corr', fontsize=8)
        plt.subplot(222)
        plt.plot(choose[:,9], choose[:,12], 'o')
        plt.plot(t_co2_corr, d15Nalpha_co2_corr_fit(t_co2_corr), 'r-', label= 'fit slope='+str(round(d15Nalpha_co2_corr_fit[1],6)))
        plt.legend()
        plt.xlabel('co2_corr', fontsize=8)
        plt.ylabel('Alpha_cr_co2_corr', fontsize=8)
        plt.subplot(223)
        plt.plot(choose[:,3], choose[:,13], 'o')
        plt.plot(t_N2O, d15Nbeta_N2O_fit(t_N2O), 'r-', label= 'fit slope='+str(round(d15Nbeta_N2O_fit[1],6)))
        plt.legend()
        plt.ylabel('d15Nbeta_N2O', fontsize=8)
        plt.xlabel('N2O', fontsize=8)
        plt.savefig('JBDS_2D_N2O-CO2_validation.png')
        
elif choice == 13:
        print ("You have chosen JBDS_2D_N2O-H2O_test.")
        Temp = np.zeros((len(df.N2O), 11))
        Temp[:, 0]=df.timestamp
        Temp[:, 1]=df.N2O
        Temp[:, 2]=df.fsr_peak1_dry
        Temp[:, 3]=df.fsr_peak10_spec
        Temp[:, 4]=df.fsr_peak11_spec
        Temp[:, 5]=df.fsr_str20
        Temp[:, 6]=df.co2_corr
        Temp[:, 7]=df.fsr_peak1_TC
        Temp[:, 8]=df.fsr_peak10_TC
        Temp[:, 9]=df.fsr_peak10_TC
        Temp[:, 10]=df.H2O

        data=avg(Temp)

        data_x = np.array(data[:,0])
        data_y = np.array(data[:,1])
        data_fsr_peak1_dry     = np.array(data[:,2])
        data_fsr_peak10_spec   = np.array(data[:,3])
        data_fsr_peak11_spec   = np.array(data[:,4])
        data_fsr_str20         = np.array(data[:,5])
        data_co2_corr          = np.array(data[:,6])
        data_fsr_peak1_TC      = np.array(data[:,7])
        data_fsr_peak10_TC     = np.array(data[:,8])
        data_fsr_peak11_TC     = np.array(data[:,9])
        data_H2O               = np.array(data[:,10])
  
        x = list(chunks(data_x, n)) 
        y = list(chunks(data_y, n)) 
        fsr_peak1_dry   = list(chunks(data_fsr_peak1_dry, n))
        fsr_peak10_spec = list(chunks(data_fsr_peak10_spec, n))
        fsr_peak11_spec = list(chunks(data_fsr_peak11_spec, n))
        fsr_str20       = list(chunks(data_fsr_str20, n))
        co2_corr        = list(chunks(data_co2_corr, n))
        fsr_peak1_TC    = list(chunks(data_fsr_peak1_TC, n))
        fsr_peak10_TC   = list(chunks(data_fsr_peak10_TC, n))
        fsr_peak11_TC   = list(chunks(data_fsr_peak11_TC, n))
        H2O             = list(chunks(data_H2O, n))

        M=np.zeros((int(len(data_y)/n), 13))
        choose=np.zeros((int(len(data_y)/n/p), 21))

# Cross-talk corrections 
#co2_corr   = co2_TC  + A_offset + A_N1 * fsr_peak1_TC +  A_N2 *fsr_peak1_TC**2                          # after 1D N2O test.  Fit of "co2_TC vs fsr_peak1_TC"
#co2_corr  += A_H1*fsr_str20 + A_H2*fsr_str20*fsr_str20 + A_N1H1*fsr_peak1_TC*fsr_str20                  # after 2D N2O-H2O test.
#fsr_peak10_spec =  fsr_peak10_TC + Offset10 + P10_N2*fsr_peak1_TC**2 + P10_N3*fsr_peak1_TC**3           # after 1D N2O test.  Fit of "fsr_peak10_TC vs fsr_peak1_TC"
#fsr_peak10_spec += P10_A1*co2_corr + P10_A1N1*co2_corr*fsr_peak1_TC                                     # after 2D N2O-CO2 test.  Fit of "fsr_peak10_TC vs co2_corr"
#fsr_peak10_spec += P10_H1*fsr_str20 + P10_H2*fsr_str20*fsr_str20 + P10_N1H1*fsr_peak10_TC*fsr_str20 + P10_N1H2*fsr_peak10_TC*fsr_str20*fsr_str20 # after 2D N2O-H2O test.  Fit of "fsr_peak10_TC vs fsr_str20"
    
#fsr_peak11_spec =  fsr_peak11_TC + Offset11 + P11_N2*fsr_peak1_TC**2 + P11_N3*fsr_peak1_TC**3           # after 1D N2O test.  Fit of "fsr_peak10_TC vs fsr_peak1_TC"
#fsr_peak11_spec += P11_A1*co2_corr + P11_A1N1*co2_corr*fsr_peak1_TC                                     # after 2D N2O-CO2 test.  Fit of "fsr_peak10_TC vs co2_corr"
#fsr_peak11_spec += P11_H1*fsr_str20 + P11_H2*fsr_str20*fsr_str20 + P11_N1H1*fsr_peak11_TC*fsr_str20 + P11_N1H2*fsr_peak11_TC*fsr_str20*fsr_str20 # after 2D N2O-H2O test.  Fit of "fsr_peak10_TC vs fsr_str20"
    
        for i in range(0, int(len(data_y)/n)):
            xx_i=np.array(x[i])
            yy_i=np.array(y[i])
            xx=xx_i.reshape(len(x[i]), 1)
            yy=yy_i.reshape(len(y[i]), 1)
            regr = linear_model.LinearRegression()
            regr.fit(xx, yy)
            plt.scatter(xx_i, yy_i, color='black')
            plt.plot(xx, regr.predict(xx), color='blue', linewidth=3)
            M[i][0]=abs(regr.coef_)
            M[i][1]=regr.intercept_
            M[i][2]=x[i][0]                  #time
            M[i][3]=y[i][0]                  #N2O
            M[i][4]=fsr_peak1_dry[i][0]     #
            M[i][5]=fsr_peak10_spec[i][0]
            M[i][6]=fsr_peak11_spec[i][0]
            M[i][7]=fsr_str20[i][0]
            M[i][8]=co2_corr[i][0]
            M[i][9]=fsr_peak1_TC[i][0]
            M[i][10]=fsr_peak10_TC[i][0]
            M[i][11]=fsr_peak11_TC[i][0]
            M[i][12]=H2O[i][0]
    
        N=list(chunks(M, p))
        for j in range(0, int(len(data_y)/n/p)):
            slope_index=N[j][:,0].argmin()
            choose[j][0]=slope_index      # choose data index position 
            choose[j][1]=min(N[j][:,0])   #choose data slope
            choose[j][2]=N[j][slope_index][2]      #choose data first x value
            choose[j][3]=N[j][slope_index][3]      #choose data first y value
            choose[j][4]=N[j][slope_index][1]      #choose data interception
            choose[j][5]=N[j][slope_index][4]      #fsr_peak1_spec
            choose[j][6]=N[j][slope_index][5]      #fsr_peak10_spec
            choose[j][7]=N[j][slope_index][6]      #fsr_peak11_spec
            choose[j][8]=N[j][slope_index][7]      #fsr_str20
            choose[j][9]=N[j][slope_index][8]      #co2_corr
            choose[j][10]=N[j][slope_index][9]     ##fsr_peak1_TC
            choose[j][11]=N[j][slope_index][8]*N[j][slope_index][9]   ##co2_corr*fsr_peak1_TC
            choose[j][12]=N[j][slope_index][10]      #fsr_peak10_TC
            choose[j][13]=N[j][slope_index][11]     ##fsr_peak11_TC
            choose[j][14]=N[j][slope_index][7]*N[j][slope_index][7]   ##fsr_str20*fsr_str20 
            choose[j][15]=N[j][slope_index][10]*N[j][slope_index][7]   ##fsr_peak10_TC*fsr_str20 
            choose[j][16]=N[j][slope_index][10]*N[j][slope_index][7]*N[j][slope_index][7]   ##fsr_peak10_TC*fsr_str20*fsr_str20  
            choose[j][17]=N[j][slope_index][9]*N[j][slope_index][7]   ##fsr_peak1_TC*fsr_str20 
            choose[j][18]=N[j][slope_index][11]*N[j][slope_index][7]   ##fsr_peak11_TC*fsr_str20 
            choose[j][19]=N[j][slope_index][11]*N[j][slope_index][7]*N[j][slope_index][7]   ##fsr_peak11_TC*fsr_str20*fsr_str20  
            choose[j][20]=N[j][slope_index][12]     ##H2O


        plt.figure(1) 
        plt.subplot(211)   
        plt.scatter(choose[:,2], choose[:,3], color='red')
        plt.xlabel('time', fontsize=8)
        plt.ylabel('N2O', fontsize=8)
        plt.subplot(212)
        plt.scatter(choose[:,2], choose[:,9], color='blue')
        plt.xlabel('time', fontsize=8)
        plt.ylabel('co2_corr', fontsize=8)

        y_pk10 = choose[:,6]
        y_pk11 = choose[:,7]
        y_co2  = choose[:,9]
        x_pk10 = np.array([choose[:,10],choose[:,8],choose[:,14],choose[:,15],choose[:,16]])
        x_pk11 = np.array([choose[:,10],choose[:,8],choose[:,14],choose[:,18],choose[:,19]])
        x_co2  = np.array([choose[:,8],choose[:,14],choose[:,17]])
        
        clf_pk10 = linear_model.LinearRegression()
        clf_pk11 = linear_model.LinearRegression()
        clf_co2 = linear_model.LinearRegression()
        
        
        x_pk10=x_pk10.transpose()    #x[0]:fsr_peak1_TC;   x[1]:fsr_str20;   x[2]: fsr_str20*fsr_str20;  x[3]: fsr_peak10_TC*fsr_str20; x[4]: fsr_peak10_TC*fsr_str20*fsr_str20
        x_pk11=x_pk11.transpose()    #x[0]:fsr_peak1_TC;   x[1]:fsr_str20;   x[2]: fsr_str20*fsr_str20;  x[3]: fsr_peak11_TC*fsr_str20; x[4]: fsr_peak11_TC*fsr_str20*fsr_str20
        x_co2=x_co2.transpose()      #x[0]:fsr_str20;   x[1]:fsr_str20*fsr_str20;   x[3]: fsr_peak1_TC*fsr_str20
        clf_pk10.fit(x_pk10, y_pk10)
        clf_pk11.fit(x_pk11, y_pk11)
        clf_co2.fit(x_co2, y_co2)
        
        print(clf_pk10.coef_)  
        print(clf_pk10.intercept_)
        print(clf_pk11.coef_)  
        print(clf_pk11.intercept_)
        print(clf_co2.coef_)  
        print(clf_co2.intercept_)

        Alpha_cr = 17120*(choose[:,6]+(-1)*clf_pk10.coef_[1]*choose[:,8]+(-1)*clf_pk10.coef_[2]*choose[:,14]+(-1)*clf_pk10.coef_[3]*choose[:,15]+(-1)*clf_pk10.coef_[4]*choose[:,16] )/choose[:,10]-1000  
        Beta_cr  = 17570*(choose[:,7]+(-1)*clf_pk11.coef_[1]*choose[:,8]+(-1)*clf_pk11.coef_[2]*choose[:,14]+(-1)*clf_pk11.coef_[3]*choose[:,18]+(-1)*clf_pk11.coef_[4]*choose[:,19] )/choose[:,10]-1000 
        co2_cr   = choose[:,9]+(-1)*clf_co2.coef_[0]*choose[:,8]+(-1)*clf_co2.coef_[1]*choose[:,14]+(-1)*clf_co2.coef_[2]*choose[:,17]
 
        Alpha_cr_N2O_fit   = np.poly1d(np.polyfit(choose[:,3], Alpha_cr, 1))   #linear fit of Alpha_cr Vs N2O
        Beta_cr_H2O_fit    = np.poly1d(np.polyfit(choose[:,20], Beta_cr, 1))    #linear fit of Beta_cr Vs H2O
        Alpha_cr_H2O_fit   = np.poly1d(np.polyfit(choose[:,20], Alpha_cr, 1))   #linear fit of Alpha_cr Vs N2O
        Beta_cr_N2O_fit    = np.poly1d(np.polyfit(choose[:,3], Beta_cr, 1))    #linear fit of Beta_cr Vs H2O

        t_N2O = np.linspace(min(choose[:,3]), max(choose[:,3]), 100)
        t_H2O = np.linspace(min(choose[:,20]), max(choose[:,20]), 100)

        plt.figure(2)
        plt.subplot(221)
        plt.plot(choose[:,3], Alpha_cr, 'o')
        plt.plot(t_N2O, Alpha_cr_N2O_fit(t_N2O), 'r-', label= 'fit slope='+str(round(Alpha_cr_N2O_fit[1],6)))
        plt.legend()
        plt.xlabel('N2O', fontsize=8)
        plt.ylabel('Alpha_cr_N2O', fontsize=8)
        plt.subplot(222)
        plt.plot(choose[:,20], Beta_cr, 'o')
        plt.plot(t_H2O, Beta_cr_H2O_fit(t_H2O), 'r-', label= 'fit slope='+str(round(Beta_cr_H2O_fit[1],6)))
        plt.legend()
        plt.ylabel('Beta_cr_H2O', fontsize=8)
        plt.xlabel('H2O', fontsize=8)
        plt.subplot(223)
        plt.plot(choose[:,20], Alpha_cr, 'o')
        plt.plot(t_H2O, Alpha_cr_H2O_fit(t_H2O), 'r-', label= 'fit slope='+str(round(Alpha_cr_H2O_fit[1],6)))
        plt.legend()
        plt.xlabel('H2O', fontsize=8)
        plt.ylabel('Alpha_cr_H2O', fontsize=8)
        plt.subplot(224)
        plt.plot(choose[:,3], Beta_cr, 'o')
        plt.plot(t_N2O, Beta_cr_N2O_fit(t_N2O), 'r-', label= 'fit slope='+str(round(Beta_cr_N2O_fit[1],6)))
        plt.legend()
        plt.ylabel('Beta_cr_N2O', fontsize=8)
        plt.xlabel('N2O', fontsize=8)
        plt.savefig('JBDS_2D_N2O-H2O_test.png')
        
        with open('JBDS_2D_N2O-H2O_test.txt', 'w') as out_file:
            out_string='Fitting parameters (sign reversed).RESTART analyzer after updating the FitterConfig file for validation\n'
            out_string+='Peak10_water_lin            = ' +str(-1*clf_pk10.coef_[1])
            out_string+='\n'   
            out_string+='Peak10_water_bilin          = ' +str(-1*clf_pk10.coef_[2])
            out_string+='\n'  
            out_string+='Peak10_water_N2O_bilin      = ' +str(-1*clf_pk10.coef_[3])
            out_string+='\n'   
            out_string+='Peak10_water_N2O_bilin_lin  = ' +str(-1*clf_pk10.coef_[4])
            out_string+='\n' 
            out_string+='Peak11_water_lin            = ' +str(-1*clf_pk11.coef_[1])
            out_string+='\n'   
            out_string+='Peak11_water_bilin          = ' +str(-1*clf_pk11.coef_[2])
            out_string+='\n'  
            out_string+='Peak11_water_N2O_bilin      = ' +str(-1*clf_pk11.coef_[3])
            out_string+='\n'   
            out_string+='Peak11_water_N2O_bilin_lin  = ' +str(-1*clf_pk11.coef_[4])
            out_string+='\n'   
            out_string+='CO2_H2O_lin                 = ' +str(-1*clf_co2.coef_[0])
            out_string+='\n'   
            out_string+='CO2_H2O_bilin               = ' +str(-1*clf_co2.coef_[1])
            out_string+='\n'  
            out_string+='CO2_N2O_H2O_bilin           = ' +str(-1*clf_co2.coef_[2])
            out_string+='\n' 
            out_file.write(out_string)
            
elif choice == 14:
        print ("You have chosen JBDS_2D_N2O-H2O_validation.")
        Temp = np.zeros((len(df.N2O), 13))
        Temp[:, 0]=df.timestamp
        Temp[:, 1]=df.N2O
        Temp[:, 2]=df.fsr_peak1_dry
        Temp[:, 3]=df.fsr_peak10_spec
        Temp[:, 4]=df.fsr_peak11_spec
        Temp[:, 5]=df.fsr_str20
        Temp[:, 6]=df.co2_corr
        Temp[:, 7]=df.fsr_peak1_TC
        Temp[:, 8]=df.fsr_peak10_TC
        Temp[:, 9]=df.fsr_peak10_TC
        Temp[:, 10]=df.H2O
        Temp[:, 11]=df.d15Nalpha
        Temp[:, 12]=df.d15Nbeta

        data=avg(Temp)

        data_x = np.array(data[:,0])
        data_y = np.array(data[:,1])
        data_fsr_peak1_dry     = np.array(data[:,2])
        data_fsr_peak10_spec   = np.array(data[:,3])
        data_fsr_peak11_spec   = np.array(data[:,4])
        data_fsr_str20         = np.array(data[:,5])
        data_co2_corr          = np.array(data[:,6])
        data_fsr_peak1_TC      = np.array(data[:,7])
        data_fsr_peak10_TC     = np.array(data[:,8])
        data_fsr_peak11_TC     = np.array(data[:,9])
        data_H2O               = np.array(data[:,10])
        data_d15Nalpha         = np.array(data[:,11])
        data_d15Nbeta          = np.array(data[:,12])
 
        x = list(chunks(data_x, n)) 
        y = list(chunks(data_y, n)) 
        fsr_peak1_dry   = list(chunks(data_fsr_peak1_dry, n))
        fsr_peak10_spec = list(chunks(data_fsr_peak10_spec, n))
        fsr_peak11_spec = list(chunks(data_fsr_peak11_spec, n))
        fsr_str20       = list(chunks(data_fsr_str20, n))
        co2_corr        = list(chunks(data_co2_corr, n))
        fsr_peak1_TC    = list(chunks(data_fsr_peak1_TC, n))
        fsr_peak10_TC   = list(chunks(data_fsr_peak10_TC, n))
        fsr_peak11_TC   = list(chunks(data_fsr_peak11_TC, n))
        H2O             = list(chunks(data_H2O, n))
        d15Nalpha       = list(chunks(data_d15Nalpha, n))
        d15Nbeta        = list(chunks(data_d15Nbeta, n))

        M=np.zeros((int(len(data_y)/n), 15))
        choose=np.zeros((int(len(data_y)/n/p), 23))
    
        for i in range(0, int(len(data_y)/n)):
            xx_i=np.array(x[i])
            yy_i=np.array(y[i])
            xx=xx_i.reshape(len(x[i]), 1)
            yy=yy_i.reshape(len(y[i]), 1)
            regr = linear_model.LinearRegression()
            regr.fit(xx, yy)
            plt.scatter(xx_i, yy_i, color='black')
            plt.plot(xx, regr.predict(xx), color='blue', linewidth=3)
            M[i][0]=abs(regr.coef_)
            M[i][1]=regr.intercept_
            M[i][2]=x[i][0]                  #time
            M[i][3]=y[i][0]                  #N2O
            M[i][4]=fsr_peak1_dry[i][0]     #
            M[i][5]=fsr_peak10_spec[i][0]
            M[i][6]=fsr_peak11_spec[i][0]
            M[i][7]=fsr_str20[i][0]
            M[i][8]=co2_corr[i][0]
            M[i][9]=fsr_peak1_TC[i][0]
            M[i][10]=fsr_peak10_TC[i][0]
            M[i][11]=fsr_peak11_TC[i][0]
            M[i][12]=H2O[i][0]
            M[i][13]=d15Nalpha[i][0]
            M[i][14]=d15Nbeta[i][0]
    
        N=list(chunks(M, p))
        for j in range(0, int(len(data_y)/n/p)):
            slope_index=N[j][:,0].argmin()
            choose[j][0]=slope_index      # choose data index position 
            choose[j][1]=min(N[j][:,0])   #choose data slope
            choose[j][2]=N[j][slope_index][2]      #choose data first x value
            choose[j][3]=N[j][slope_index][3]      #choose data first y value
            choose[j][4]=N[j][slope_index][1]      #choose data interception
            choose[j][5]=N[j][slope_index][4]      #fsr_peak1_spec
            choose[j][6]=N[j][slope_index][5]      #fsr_peak10_spec
            choose[j][7]=N[j][slope_index][6]      #fsr_peak11_spec
            choose[j][8]=N[j][slope_index][7]      #fsr_str20
            choose[j][9]=N[j][slope_index][8]      #co2_corr
            choose[j][10]=N[j][slope_index][9]     ##fsr_peak1_TC
            choose[j][11]=N[j][slope_index][8]*N[j][slope_index][9]   ##co2_corr*fsr_peak1_TC
            choose[j][12]=N[j][slope_index][10]      #fsr_peak10_TC
            choose[j][13]=N[j][slope_index][11]     ##fsr_peak11_TC
            choose[j][14]=N[j][slope_index][7]*N[j][slope_index][7]   ##fsr_str20*fsr_str20 
            choose[j][15]=N[j][slope_index][10]*N[j][slope_index][7]   ##fsr_peak10_TC*fsr_str20 
            choose[j][16]=N[j][slope_index][10]*N[j][slope_index][7]*N[j][slope_index][7]   ##fsr_peak10_TC*fsr_str20*fsr_str20  
            choose[j][17]=N[j][slope_index][9]*N[j][slope_index][7]   ##fsr_peak1_TC*fsr_str20 
            choose[j][18]=N[j][slope_index][11]*N[j][slope_index][7]   ##fsr_peak11_TC*fsr_str20 
            choose[j][19]=N[j][slope_index][11]*N[j][slope_index][7]*N[j][slope_index][7]   ##fsr_peak11_TC*fsr_str20*fsr_str20  
            choose[j][20]=N[j][slope_index][12]     ##H2O
            choose[j][21]=N[j][slope_index][13]      #d15Nalpha
            choose[j][22]=N[j][slope_index][14]     ##d15Nbeta

        plt.figure(1) 
        plt.subplot(211)   
        plt.scatter(choose[:,2], choose[:,3], color='red')
        plt.xlabel('time', fontsize=8)
        plt.ylabel('N2O', fontsize=8)
        plt.subplot(212)
        plt.scatter(choose[:,2], choose[:,9], color='blue')
        plt.xlabel('time', fontsize=8)
        plt.ylabel('co2_corr', fontsize=8)
 
        d15Nalpha_N2O_fit   = np.poly1d(np.polyfit(choose[:,3], choose[:,21], 1))   #linear fit of d15Nalpha Vs N2O
        d15Nbeta_H2O_fit    = np.poly1d(np.polyfit(choose[:,20], choose[:,22], 1))    #linear fit of d15Nbeta Vs H2O
        d15Nalpha_H2O_fit   = np.poly1d(np.polyfit(choose[:,20], choose[:,21], 1))   #linear fit of d15Nalpha Vs H2O
        d15Nbeta_N2O_fit    = np.poly1d(np.polyfit(choose[:,3], choose[:,22], 1))    #linear fit of d15Nbeta Vs N2O
        co2_corr_N2O_fit    = np.poly1d(np.polyfit(choose[:,3], choose[:,9], 1))    #linear fit of co2_corr Vs N2O
        co2_corr_H2O_fit    = np.poly1d(np.polyfit(choose[:,20], choose[:,9], 1))    #linear fit of co2_corr Vs H2O

        t_N2O = np.linspace(min(choose[:,3]), max(choose[:,3]), 100)
        t_H2O = np.linspace(min(choose[:,20]), max(choose[:,20]), 100)

        plt.figure(2)
        plt.subplot(321)
        plt.plot(choose[:,3], choose[:,21], 'o')
        plt.plot(t_N2O, d15Nalpha_N2O_fit(t_N2O), 'r-', label= 'fit slope='+str(round(d15Nalpha_N2O_fit[1],6)))
        plt.legend()
        plt.xlabel('N2O', fontsize=8)
        plt.ylabel('d15Nalpha_N2O', fontsize=8)
        plt.subplot(322)
        plt.plot(choose[:,20], choose[:,21], 'o')
        plt.plot(t_H2O, d15Nalpha_H2O_fit(t_H2O), 'r-', label= 'fit slope='+str(round(d15Nalpha_H2O_fit[1],6)))
        plt.legend()
        plt.xlabel('H2O', fontsize=8)
        plt.ylabel('d15Nalpha_H2O', fontsize=8)
        plt.subplot(323)
        plt.plot(choose[:,3], choose[:,22], 'o')
        plt.plot(t_N2O, d15Nbeta_N2O_fit(t_N2O), 'r-', label= 'fit slope='+str(round(d15Nbeta_N2O_fit[1],6)))
        plt.legend()
        plt.ylabel('d15Nbeta_N2O', fontsize=8)
        plt.xlabel('N2O', fontsize=8)
        plt.subplot(324)
        plt.plot(choose[:,20], choose[:,22], 'o')
        plt.plot(t_H2O, d15Nbeta_H2O_fit(t_H2O), 'r-', label= 'fit slope='+str(round(d15Nbeta_H2O_fit[1],6)))
        plt.legend()
        plt.ylabel('d15Nbeta_H2O', fontsize=8)
        plt.xlabel('H2O', fontsize=8)
        plt.subplot(325)
        plt.plot(choose[:,3], choose[:,9], 'o')
        plt.plot(t_N2O, co2_corr_N2O_fit(t_N2O), 'r-', label= 'fit slope='+str(round(co2_corr_N2O_fit[1],6)))
        plt.legend()
        plt.ylabel('co2_corr_N2O', fontsize=8)
        plt.xlabel('N2O', fontsize=8)
        plt.subplot(326)
        plt.plot(choose[:,20], choose[:,9], 'o')
        plt.plot(t_H2O, co2_corr_H2O_fit(t_H2O), 'r-', label= 'fit slope='+str(round(co2_corr_H2O_fit[1],6)))
        plt.legend()
        plt.ylabel('co2_corr_H2O', fontsize=8)
        plt.xlabel('H2O', fontsize=8)
        plt.savefig('JBDS_2D_N2O-H2O_validation.png')
        
elif choice == 15:
        print ("You have chosen JBDS_3D_N2O-H2O-CO2_validation.")
        Temp = np.zeros((len(df.N2O), 13))
        Temp[:, 0]=df.timestamp
        Temp[:, 1]=df.N2O
        Temp[:, 2]=df.fsr_peak1_dry
        Temp[:, 3]=df.fsr_peak10_spec
        Temp[:, 4]=df.fsr_peak11_spec
        Temp[:, 5]=df.fsr_str20
        Temp[:, 6]=df.co2_corr
        Temp[:, 7]=df.fsr_peak1_TC
        Temp[:, 8]=df.fsr_peak10_TC
        Temp[:, 9]=df.fsr_peak10_TC
        Temp[:, 10]=df.H2O
        Temp[:, 11]=df.d15Nalpha
        Temp[:, 12]=df.d15Nbeta

        data=avg(Temp)

        data_x = np.array(data[:,0])
        data_y = np.array(data[:,1])
        data_fsr_peak1_dry     = np.array(data[:,2])
        data_fsr_peak10_spec   = np.array(data[:,3])
        data_fsr_peak11_spec   = np.array(data[:,4])
        data_fsr_str20         = np.array(data[:,5])
        data_co2_corr          = np.array(data[:,6])
        data_fsr_peak1_TC      = np.array(data[:,7])
        data_fsr_peak10_TC     = np.array(data[:,8])
        data_fsr_peak11_TC     = np.array(data[:,9])
        data_H2O               = np.array(data[:,10])
        data_d15Nalpha         = np.array(data[:,11])
        data_d15Nbeta          = np.array(data[:,12])
 
        x = list(chunks(data_x, n)) 
        y = list(chunks(data_y, n)) 
        fsr_peak1_dry   = list(chunks(data_fsr_peak1_dry, n))
        fsr_peak10_spec = list(chunks(data_fsr_peak10_spec, n))
        fsr_peak11_spec = list(chunks(data_fsr_peak11_spec, n))
        fsr_str20       = list(chunks(data_fsr_str20, n))
        co2_corr        = list(chunks(data_co2_corr, n))
        fsr_peak1_TC    = list(chunks(data_fsr_peak1_TC, n))
        fsr_peak10_TC   = list(chunks(data_fsr_peak10_TC, n))
        fsr_peak11_TC   = list(chunks(data_fsr_peak11_TC, n))
        H2O             = list(chunks(data_H2O, n))
        d15Nalpha       = list(chunks(data_d15Nalpha, n))
        d15Nbeta        = list(chunks(data_d15Nbeta, n))

        M=np.zeros((int(len(data_y)/n), 15))
        choose=np.zeros((int(len(data_y)/n/p), 23))
    
        for i in range(0, int(len(data_y)/n)):
            xx_i=np.array(x[i])
            yy_i=np.array(y[i])
            xx=xx_i.reshape(len(x[i]), 1)
            yy=yy_i.reshape(len(y[i]), 1)
            regr = linear_model.LinearRegression()
            regr.fit(xx, yy)
            plt.scatter(xx_i, yy_i, color='black')
            plt.plot(xx, regr.predict(xx), color='blue', linewidth=3)
            M[i][0]=abs(regr.coef_)
            M[i][1]=regr.intercept_
            M[i][2]=x[i][0]                  #time
            M[i][3]=y[i][0]                  #N2O
            M[i][4]=fsr_peak1_dry[i][0]     #
            M[i][5]=fsr_peak10_spec[i][0]
            M[i][6]=fsr_peak11_spec[i][0]
            M[i][7]=fsr_str20[i][0]
            M[i][8]=co2_corr[i][0]
            M[i][9]=fsr_peak1_TC[i][0]
            M[i][10]=fsr_peak10_TC[i][0]
            M[i][11]=fsr_peak11_TC[i][0]
            M[i][12]=H2O[i][0]
            M[i][13]=d15Nalpha[i][0]
            M[i][14]=d15Nbeta[i][0]
    
        N=list(chunks(M, p))
        for j in range(0, int(len(data_y)/n/p)):
            slope_index=N[j][:,0].argmin()
            choose[j][0]=slope_index      # choose data index position 
            choose[j][1]=min(N[j][:,0])   #choose data slope
            choose[j][2]=N[j][slope_index][2]      #choose data first x value
            choose[j][3]=N[j][slope_index][3]      #choose data first y value
            choose[j][4]=N[j][slope_index][1]      #choose data interception
            choose[j][5]=N[j][slope_index][4]      #fsr_peak1_spec
            choose[j][6]=N[j][slope_index][5]      #fsr_peak10_spec
            choose[j][7]=N[j][slope_index][6]      #fsr_peak11_spec
            choose[j][8]=N[j][slope_index][7]      #fsr_str20
            choose[j][9]=N[j][slope_index][8]      #co2_corr
            choose[j][10]=N[j][slope_index][9]     ##fsr_peak1_TC
            choose[j][11]=N[j][slope_index][8]*N[j][slope_index][9]   ##co2_corr*fsr_peak1_TC
            choose[j][12]=N[j][slope_index][10]      #fsr_peak10_TC
            choose[j][13]=N[j][slope_index][11]     ##fsr_peak11_TC
            choose[j][14]=N[j][slope_index][7]*N[j][slope_index][7]   ##fsr_str20*fsr_str20 
            choose[j][15]=N[j][slope_index][10]*N[j][slope_index][7]   ##fsr_peak10_TC*fsr_str20 
            choose[j][16]=N[j][slope_index][10]*N[j][slope_index][7]*N[j][slope_index][7]   ##fsr_peak10_TC*fsr_str20*fsr_str20  
            choose[j][17]=N[j][slope_index][9]*N[j][slope_index][7]   ##fsr_peak1_TC*fsr_str20 
            choose[j][18]=N[j][slope_index][11]*N[j][slope_index][7]   ##fsr_peak11_TC*fsr_str20 
            choose[j][19]=N[j][slope_index][11]*N[j][slope_index][7]*N[j][slope_index][7]   ##fsr_peak11_TC*fsr_str20*fsr_str20  
            choose[j][20]=N[j][slope_index][12]     ##H2O
            choose[j][21]=N[j][slope_index][13]      #d15Nalpha
            choose[j][22]=N[j][slope_index][14]     ##d15Nbeta

        plt.figure(1) 
        plt.subplot(211)   
        plt.scatter(choose[:,2], choose[:,3], color='red')
        plt.xlabel('time', fontsize=8)
        plt.ylabel('N2O', fontsize=8)
        plt.subplot(212)
        plt.scatter(choose[:,2], choose[:,9], color='blue')
        plt.xlabel('time', fontsize=8)
        plt.ylabel('co2_corr', fontsize=8)
 
        d15Nalpha_N2O_fit   = np.poly1d(np.polyfit(choose[:,3], choose[:,21], 1))   #linear fit of d15Nalpha Vs N2O
        d15Nbeta_H2O_fit    = np.poly1d(np.polyfit(choose[:,20], choose[:,22], 1))    #linear fit of d15Nbeta Vs H2O
        d15Nalpha_H2O_fit   = np.poly1d(np.polyfit(choose[:,20], choose[:,21], 1))   #linear fit of d15Nalpha Vs H2O
        d15Nbeta_N2O_fit    = np.poly1d(np.polyfit(choose[:,3], choose[:,22], 1))    #linear fit of d15Nbeta Vs N2O
        d15Nalpha_co2_corr_fit    = np.poly1d(np.polyfit(choose[:,9], choose[:,21], 1))    #linear fit of d15Nalpha Vs co2_corr
        d15Nbeta_co2_corr_fit     = np.poly1d(np.polyfit(choose[:,9], choose[:,22], 1))    #linear fit of d15Nbeta Vs co2_corr

        t_N2O = np.linspace(min(choose[:,3]), max(choose[:,3]), 100)
        t_H2O = np.linspace(min(choose[:,20]), max(choose[:,20]), 100)
        t_co2_corr = np.linspace(min(choose[:,9]), max(choose[:,9]), 100)

        plt.figure(2)
        plt.subplot(231)
        plt.plot(choose[:,3], choose[:,21], 'o')
        plt.plot(t_N2O, d15Nalpha_N2O_fit(t_N2O), 'r-', label= 'fit slope='+str(round(d15Nalpha_N2O_fit[1],6)))
        plt.legend()
        plt.xlabel('N2O', fontsize=8)
        plt.ylabel('d15Nalpha_N2O', fontsize=8)
        plt.subplot(232)
        plt.plot(choose[:,20], choose[:,21], 'o')
        plt.plot(t_H2O, d15Nalpha_H2O_fit(t_H2O), 'r-', label= 'fit slope='+str(round(d15Nalpha_H2O_fit[1],6)))
        plt.legend()
        plt.xlabel('H2O', fontsize=8)
        plt.ylabel('d15Nalpha_H2O', fontsize=8)
        plt.subplot(233)
        plt.plot(choose[:,9], choose[:,21], 'o')
        plt.plot(t_co2_corr, d15Nalpha_co2_corr_fit(t_co2_corr), 'r-', label= 'fit slope='+str(round(d15Nalpha_co2_corr_fit[1],6)))
        plt.legend()
        plt.xlabel('co2_corr', fontsize=8)
        plt.ylabel('d15Nalpha_co2_corr', fontsize=8)
        plt.subplot(234)
        plt.plot(choose[:,3], choose[:,22], 'o')
        plt.plot(t_N2O, d15Nbeta_N2O_fit(t_N2O), 'r-', label= 'fit slope='+str(round(d15Nbeta_N2O_fit[1],6)))
        plt.legend()
        plt.ylabel('d15Nbeta_N2O', fontsize=8)
        plt.xlabel('N2O', fontsize=8)
        plt.subplot(235)
        plt.plot(choose[:,20], choose[:,22], 'o')
        plt.plot(t_H2O, d15Nbeta_H2O_fit(t_H2O), 'r-', label= 'fit slope='+str(round(d15Nbeta_H2O_fit[1],6)))
        plt.legend()
        plt.ylabel('d15Nbeta_H2O', fontsize=8)
        plt.xlabel('H2O', fontsize=8)
        plt.subplot(236)
        plt.plot(choose[:,9], choose[:,22], 'o')
        plt.plot(t_co2_corr, d15Nbeta_co2_corr_fit(t_co2_corr), 'r-', label= 'fit slope='+str(round(d15Nbeta_co2_corr_fit[1],6)))
        plt.legend()
        plt.ylabel('d15Nbeta_co2_corr', fontsize=8)
        plt.xlabel('co2_corr', fontsize=8)
        plt.savefig('JBDS_3D_N2O-H2O-CO2_validation.png')
        
else:
        print ("Invalid number. Try again...")


        
    

