# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:28:06 2020

@author: chris
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 09:16:40 2020
@author: Gebruiker
"""


import os
import pandas as pd

import seaborn as sns
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

loc2 =  os.getcwd().replace("\\","/") + "/predator-prey-results_final.csv"
df = pd.read_csv(loc2) #get data frame
df["Score"] = df["Score"].apply(lambda x: x**2/100) #adjust for the difference in error



def main():
    
    """
    Calculates the statistical values for all the simulations
    """
    
    print(df)
    final_df = df.loc[df["It"] == 11507].copy() #get the final values of df
#    final_df = final_df.drop("It", axis = 1).reset_index(drop = True).set_index("Method").stack()
#    print(final_df.index)
    print(final_df)
    prey10_df = final_df.loc[final_df["Method"] == "prey10"]
    pred10_df = final_df.loc[final_df["Method"] == "pred10"]
    for i in range(10,100,10):
        for k in ["pred","prey"]:
            variable_df = final_df.loc[final_df["Method"] == f"{k}{i}"]
            
            print(f"{k}{i}")
            print(np.mean(variable_df["Score"]), np.std(variable_df["Score"]))
            print(stats.shapiro(variable_df["Score"]))
            if k == "pred":
                print(stats.mannwhitneyu(variable_df["Score"], pred10_df["Score"]))
            else:
                print(stats.mannwhitneyu(variable_df["Score"], prey10_df["Score"]))
                
            print()
            print()
    

    
    its = (len(df))/(20)/18
    its2 = int(its)
    x = np.linspace(0, its2, num =int(its2))
    
    
    
    """
    Plots the mean and standard deviation of the selected simulations. 
    Can be adjusted by adding prey or pred to the first loop and number of removed points to the second loop.
    """
    
    for method in ["pred"]:
        for number in [10, 90]:
            mse = df.loc[df['Method'] == f"{method}{number}"]
            averages = []
            std = []
            
            for i in range(int(its)):
                d = mse.loc[mse['It'] == i]
                sc = np.array(d['Score'])
                averages.append(np.mean(sc))
                std.append(np.std(sc))
            
            #print(x, averages, d)
            plt.plot(x, averages, label = f"{method}{number}")
            plt.xlabel("Iterations")
            plt.ylabel("Error")
            plt.fill_between(x,np.array(averages)+np.array(std), np.array(averages)-np.array(std), alpha = 0.5)
            plt.ylim(0,20)
    plt.legend()
    plt.show()
      


if __name__ == '__main__':
    main()