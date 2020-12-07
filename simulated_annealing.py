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

import math
import os


loc = os.getcwd().replace("\\","/") + "/predator-prey-data.csv"

def main():
    df = pd.read_csv(loc)
    
    pred = (df['x'])
    prey = (df['y'])
    
    sns.lineplot(data =df, x = 't', y='x', label = 'predator')
    sns.lineplot(data =df, x = 't', y = 'y', label = 'prey')
    
    plt.show()
    alpha = 0.25
    beta = 0.25
    
    t = list(df['t'])
    y0 = [df['y'][0], df['x'][0]]
    
    
    alpha = 0.9
    beta = 0.7
    gamma = 0.9
    delta = 0.5
    
    

    

    parameters = [alpha, beta, gamma, delta]

                  
    """
    parameters['alpha'] = 0.5
    parameters['beta'] = 0.1
    parameters['gamma'] = 0.4
    parameters['delta'] = 0.09]
    """
    parameters, sol2 = sim_annealing(parameters, y0, t, pred, prey)

    sol = odeint(pend, y0, t, args=(parameters[0], parameters[1], parameters[2], parameters[3]))
    
    
    preds = sol[:,1]
    preys = sol[:,0]
    
    
    
    plt.plot(t, preds, label ='predator')
    plt.plot(t, preys, label = 'prey')
    plt.legend()
    
   

def get_cost(parameters, y0, t, pred, prey):
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
    preds = sol[:,1]
    preys = sol[:,0]
    
    mse1 =  (sum((pred - preds)**2))
    mse2 = (sum((prey - preys)**2))
    
    return mse1 + mse2

def get_neighbors(parameters):
    """

    Parameters
    ----------
    parameters : the parameters for the lotka volterra equations
    Returns
    -------
    neighbors : returnsset of neigbors, each neighbor has one of the parametres changed with a value
        

    """
    neighbors= []
    value = 0.005
    
#    neighbor = parameters.copy()
#    neighbor[0] =  neighbor[0] + (random.uniform(-value,value))
#    neighbor[1] =  neighbor[1] + (random.uniform(-value,value))
#    neighbor[2] =  neighbor[2] + (random.uniform(-value,value))
#    neighbor[3] =  neighbor[3] + (random.uniform(-value,value))
#    neighbors.append(neighbor)
    
    
    #old code
    for i in range(10):
        neighbor = parameters.copy()
        neighbor[0] =  neighbor[0] + (random.uniform(-value,value) * i)
        neighbor[1] =  neighbor[1] + (random.uniform(-value,value) * i)
        neighbor[2] =  neighbor[2] + (random.uniform(-value,value) * i)
        neighbor[3] =  neighbor[3] + (random.uniform(-value,value) * i)
        neighbors.append(neighbor)

    
        
    return neighbors
    
    
    
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

    x, y = pred_prey
    
    dydt = alpha * x - beta * x * y
    dxdt = -gamma * y + delta * x * y
    
    return dydt, dxdt
    

def sim_annealing(parameters, y0, t, pred, prey):
    """
    Algorithm of simmulated annealing according to this example:
        https://medium.com/swlh/how-to-implement-simulated-annealing-algorithm-in-python-ab196c2f56a0
        
    """
    init_temp =100
    final_temp = 0.1
    alpha = .05
    
    #parameters = [0.1,0.1,0.1,0.1]
    current_temp = init_temp
    
    solution = parameters.copy()
    while current_temp > final_temp:
        #generate a set of neigbbors
        neighbors = get_neighbors(solution)

        neighbor = random.choice(neighbors)
        
        #print(neighbor)
        cost_diff = get_cost(solution, y0, t, pred, prey) - get_cost(neighbor, y0, t, pred, prey)

        if cost_diff >= 0:
            solution = neighbor.copy()
        else: 
            if random.uniform(0,1) < math.exp(-(cost_diff / current_temp)):
                
                solution = neighbor.copy()
        current_temp -= alpha
        
    print(solution)
    return solution, get_cost(neighbor, y0, t, pred, prey)
                
        
def hill_climbing(parameters, y0, t, pred, prey):
    
    """
    find neighbor, check if it is better, if it's better, go there repeat, otherwise stop
    """
    new_param = parameters.copy()
    cost = get_cost(new_param)
    neighbor_cost = 0
    change = 0.05
    
    while (cost > neighbor_cost) :
        
        neighbor_1 = new_param.copy()
        neighbor_2 = new_param.copy()
        for i in range(4):
            neighbor_1[i] = neighbor_1[i] + random.uniform(-change,change)
            neighbor_2[i] = neighbor_2[i] + random.uniform(-change,change)
            
        neighbor_cost_1 = get_cost(neighbor_1)
        neighbor_cost_2 = get_cost(neighbor_2)
    
    if neighbor_cost_1 < neighbor_cost_2:
        new_params = neighbor_1
        neighbor_cost = neighbor_cost_1
        
    else:
        new_param = neighbor_2
        neighbor_cost = neighbor_cost_2
    
    
    return neighbor

if __name__ == '__main__':
    main()