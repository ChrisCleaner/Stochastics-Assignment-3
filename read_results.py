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

loc2 =  os.getcwd().replace("\\","/") + "/predator-prey-results.csv"



def main():
    df = pd.read_csv(loc2)
    print(df)


    
    its = (len(df))/(50)/2
    x = np.linspace(0, its, num =int(its))
    
    
    averages = []
    std = []
    
    mse = df.loc[df['Method'] == 'mse']
    
    for i in range(int(its)):
        d = mse.loc[mse['It'] == i]
        sc = np.array(d['Score'])
        averages.append(np.mean(sc))
        std.append(np.std(sc))
    
    plt.plot(x, averages)
    plt.fill_between(x,np.array(averages)+np.array(std), np.array(averages)-np.array(std), alpha = 0.5)
        
        
        

    newdf = df.loc[df['It'] == its -1] 
    newdf1 = newdf.loc[newdf['Method'] == 'mse']
    newdf2 = newdf.loc[newdf['Method'] == 'rmse']
    
    
 
    
    print(stats.shapiro(newdf1['Score']))
    print(stats.shapiro(newdf2['Score']))
    
    
    scores1 = (list(newdf1['Score']))
    
    scores2 = (list(newdf2['Score']))
    
    
    """
    For plotting with seaborn use the following: will take some time
    #sns.histplot(data = newdf1, x = 'Score')
    #sns.histplot(data = newdf2, x = 'Score')
    

    
    #sns.lineplot(data = df, x = 'It', y = 'Score', hue='Method')

    """


if __name__ == '__main__':
    main()