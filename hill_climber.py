# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 11:54:12 2020
@author: Gebruiker
"""
import numpy as np
import pandas as pd
import random

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os
from math import sqrt
import csv



import math


n_sim = 50
loc = os.getcwd().replace("\\","/") + "/predator-prey-data.csv"
loc2 =  os.getcwd().replace("\\","/") + "/predator-prey-results.csv"

df = pd.read_csv(loc)
    
pred = (df['x'])
prey = (df['y'])


def main():
    
    with open(loc2, 'w', newline='') as file:
        writer = csv.writer(file)
    
        writer.writerow(["It", "Score", "Method"])

    df = pd.read_csv(loc)
    
    pred = (df['x'])
    prey = (df['y'])
    
    sns.lineplot(data =df, x = 't', y='x', label = 'predator')
    sns.lineplot(data =df, x = 't', y = 'y', label = 'prey')
    
    plt.show()
    alpha = 0.25
    beta = 0.25
    
    t = list(df['t'])
    
    
    y0 = [pred[0], prey[0]]
    
   
    print('y0', y0)
    best_list = []
    for i in range(n_sim):
        parameters = []
        for i in range(4):
            parameters.append(random.uniform(0,2))
        print(f"parameters: {parameters}")
                
        it = []
        score = []
        meth = []
        
            
        
          


   
        parameters, it, meth, score = hill_climbing(parameters,y0, t, pred, prey, it, meth, score)
        best_list.append(parameters)
    
          
       
        dat = [it, score, meth]
        data = pd.DataFrame(np.transpose(dat))

   

        with open(loc2,'a', newline="\n") as fd:
            for i in range(len(data)):
               writer = csv.writer(fd)
               writer.writerow((list(data.iloc[i]))) 



    

def get_cost(parameters, y0, t, pred, prey, method):
    """
    Used pend and Odeint to find a list of values of Preys and Predators with the given parameters
    Parameters
    ----------
    parameters : list of length 4, parameters for the lotka volterera equations ( in order alpha, beta, gamma, delta)
    y0 :list of 2,  start values
    t : list, times as in the csv doc
    pred : list, predator numbers as in the csv doc
    prey : list, prey numbers as in the csv doc
    Returns
    -------
    The total of the mean squared error of the predator and prey resulting from solving the ODE with the given parameters
    """
    
    
    sol = odeint(pend, y0, t, args=(parameters[0], parameters[1], parameters[2], parameters[3]))
    
    
    preds = sol[:,0]
    preys = sol[:,1]
    
    """
    plt.plot(t, preds, label ='predator')
    plt.plot(t, preys, label = 'prey')
    plt.legend()
    plt.show()
    """            
    
    if method == "mse":
        mse1 =  np.mean((pred - preds)**2)
        mse2 = np.mean((prey - preys)**2)
        
        return mse1 + mse2
    elif method == "rmse":
        mse1 =  np.mean((pred - preds)**2)
        mse2 = np.mean((prey - preys)**2)
        
        return sqrt(mse1 + mse2)


def pend(pred_prey,t, alpha, beta, gamma, delta):
    """
    First order differential equations of Lotka-Volterra
    x()= prey population
    y()= predator population
    
    alpha, beta, gamma, delta are model parameters
    Returns
    -------
    vector dydt with : CHange in preys dx, change in predators dy 
    """
    #print(pred_prey)
    

    x = pred_prey[0]
    y = pred_prey[1]

    
    
    dydt = [alpha * x - beta * x * y, -gamma * y + delta * x * y]
    
    return dydt
    
def create_eight_neighbors(parameters, scale):
    """
    Creates eight neighbors at the difference of scale.
    Each parameter has a +scale and -scale neighbor
    """
    neighbors = []
    
    for i in range(4):
        new_para = parameters.copy()
        new_para[i] += scale
        neighbors.append(new_para)
        
        new_para = parameters.copy()
        new_para[i] -= scale
        neighbors.append(new_para)
    
    return neighbors
                
        
def hill_climbing(parameters,y0, t, pred, prey, it, meth, score):
    
    scale = 0.05
    counter = 0

    new_param = parameters.copy()
    costs = get_cost(new_param,y0, t, pred, prey, "rmse")
    
    #do until peak is reached
    while True:
        #create the eight neighbors
        neighbors = create_eight_neighbors(new_param, scale)
        cost_list = []
        
        #check what the errors are
        for para in neighbors:
            cost_list.append(get_cost(para,y0, t, pred, prey, "rmse"))
        
        #if one neighbors is better then the current position, move to best
        if min(cost_list) < costs:
            costs = min(cost_list)
            it.append(counter)
            meth.append("hill_climb")
            score.append(costs)
            best_para_index = cost_list.index(min(cost_list))
            new_param = neighbors[best_para_index]
        else: #stop algorithm
            break
        
        
        counter += 1
    return new_param, it, meth, score

if __name__ == '__main__':
    main()