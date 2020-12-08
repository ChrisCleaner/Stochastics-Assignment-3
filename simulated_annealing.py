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


loc = r'C:\Users\Gebruiker\OneDrive\Computational_Science\Year1_Semester1_Block2\Stochastic_Simulations\Assignment3\predator-prey-data.csv'
df = pd.read_csv(loc)
    
pred = (df['x'])
prey = (df['y'])
    

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
    
    
    y0 = [pred[0], prey[0]]
    
    
    alpha = float(0.8)
    beta = float( 0.5)
    gamma = float(1.2)
    delta = float(1.0)
    
        
    alpha = float(0.7)
    beta = float( 0.5)
    gamma = float(1.4)
    delta = float(1.0)
    
    print('y0', y0)
    sol = odeint(pend, y0, t, args=(alpha, beta, gamma, delta))

    preys = sol[:,1]
    preds = sol[:,0]

    parameters = [alpha, beta, gamma, delta]
    
    plt.plot(t, preds, label ='predator')
    plt.plot(t, preys, label = 'prey')
    plt.legend()

    plt.show()
                 
   
    parameters, sol2 = sim_annealing(parameters, y0, t, pred, prey)

    sol = odeint(pend, y0, t, args=(parameters[0], parameters[1], parameters[2], parameters[3]))
    
    
    preys = sol[:,1]
    preds = sol[:,0]

    
    
    plt.plot(t, preds, label ='predator')
    plt.plot(t, preys, label = 'prey')
    plt.legend()
    plt.show()
    print(parameters)
    
   
    

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
    
    
    preds = sol[:,0]
    preys = sol[:,1]
    
    """
    plt.plot(t, preds, label ='predator')
    plt.plot(t, preys, label = 'prey')
    plt.legend()

    plt.show()
    """            
    

    mse1 =  (sum(abs(pred - preds)))
    mse2 = (sum(abs(prey - preys)))
    
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
    value = float(0.01)

    
    for i in range(len(parameters)*2):
        new_p = np.copy(parameters)
        
        if i < len(parameters):
            new_p[i] = new_p[i] + value
        else: 
            if new_p[i-len(parameters)] >= value:
                new_p[i-len(parameters)] = new_p[i-len(parameters)] - value
            
            
     
        neighbors.append(new_p)
        
        
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
    #print(pred_prey)
    

    x = pred_prey[0]
    y = pred_prey[1]

    
    
    dydt = [alpha * x - beta * x * y, -gamma * y + delta * x * y]
    
    return dydt
    

def sim_annealing(parameters, y0, t, pred, prey):
    """
    Algorithm of simmulated annealing according to this example:
        https://medium.com/swlh/how-to-implement-simulated-annealing-algorithm-in-python-ab196c2f56a0
        
    """
    init_temp = 100
    final_temp = 0.01
    alpha = 0.01
    
    current_temp = init_temp
    y0 = [pred[0], prey[0]]
    

    current_state = parameters
    solution = current_state
    
    n = 0
    
    while current_temp > final_temp:
        #generate a set of neigbbors
        neighbors = get_neighbors(solution)
        
        
        neighbor = random.choice(neighbors)
        #print(neighbor)
        
        cost_diff = get_cost(solution, y0, t, pred, prey) - get_cost(neighbor, y0, t, pred, prey)
  
        
        if cost_diff > 0:
            solution = neighbor
        else: 
            if random.uniform(0,1) < math.exp(cost_diff / current_temp):
                solution = neighbor
                
                
        current_temp -= alpha
        
    return solution, get_cost(neighbor, y0, t, pred, prey)
                
        
            
    

if __name__ == '__main__':
    main()