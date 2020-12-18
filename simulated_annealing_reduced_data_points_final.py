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

it = []
score = []
meth = []

    

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
    for var in ["pred", "prey"]:
        for remove_points in range(10,100,10):
            for k in range(1):
                parameters = []
                for i in range(4):
                    parameters.append(1)
                it = []
                score = []
                meth = []

                params, error, points, it, score, meth =remove_points_sim_an(parameters, y0, t, pred, prey, remove_points, var, it, score, meth)
    
                
                dat = [it, score, meth]
                data = pd.DataFrame(np.transpose(dat))
        


        
                with open(loc2,'a', newline="\n") as fd:
                    for i in range(len(data)):
                       writer = csv.writer(fd)
                       writer.writerow((list(data.iloc[i]))) 
        
        
    
    
    
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
    

def get_neighbors(parameters):
    """
    Parameters
    ----------
    parameters : the parameters for the lotka volterra equations
    Returns
    -------
    neighbors : returnsset of neigbors, each neighbor has one of the parametres changed with a value
        
    """
    value = 0.1
    
    param_num = random.randint(0,3)
    neighbor = parameters.copy()
    neighbor[param_num] = neighbor[param_num] + random.uniform(-value,value)
    if neighbor[param_num] <= 0:
        for i in range(4):
            neighbor[i] = random.random()

    
    
        
        
    return neighbor
    
    
          
        

def get_cost_removed(parameters, y0, t, pred, prey, list_removed_points, source_of_removal, real_pred, real_prey):
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
    The total of the error of the predator and prey resulting from solving the ODE with the given parameters
    Additionally, also returns the error of the whole set of data points.
    """
    
    
    sol = odeint(pend, y0, t, args=(parameters[0], parameters[1], parameters[2], parameters[3]))
    
    
    preds = sol[:,0]
    preys = sol[:,1]
    
    mse3 =  (sum(abs(preds - real_pred)))
    mse4 = (sum(abs(preys - real_prey)))
    
    
            
    if source_of_removal == "prey":
        preys = np.delete(preys, list_removed_points)
    elif source_of_removal == "pred":
        preds = np.delete(preds, list_removed_points)
        
    
    mse1 =  (sum(abs(pred - preds)))
    mse2 = (sum(abs(prey - preys)))
    

    
    return sqrt(mse1 + mse2), sqrt(mse3 + mse4) #returns first the error with the less data points and the real error to compare later


def remove_points_sim_an(parameters, y0, t, pred, prey, data_points_removed, source_of_removal, it, score, meth):
    """
    Algorithm of simmulated annealing according to this example:
        https://medium.com/swlh/how-to-implement-simulated-annealing-algorithm-in-python-ab196c2f56a0
        
    Deletes points of the data set to check simulate less data. Amount is given by data_points_removed.
    If the predator or prey data is removed is decided by soure_of_removal.
    it, score and method are for tracking and saving of the data.
    Parameters provides the initial parameters.
    y0 are the inital values.
    t is the time of the data points.
    pred and prey are the data points.
        
    """
    init_temp = 100
    final_temp = 0.001
    alpha = 0.1
    
    #removes data from the list of points
    preds_removed = list(pred.copy())
    preys_removed = list(prey.copy())
    
    current_temp = init_temp
    y0 = [pred[0], prey[0]]
    
    points = random.sample(range(100), data_points_removed)
    val_points = []
    if source_of_removal == "prey": #remove of prey
        for point in points:
            val_points.append(preys_removed[point])
        
        for val in val_points:
            preys_removed.remove(val)
            
    elif source_of_removal == "pred": #remove of pred
        for point in points:
            val_points.append(preds_removed[point])
        
        for val in val_points:
            preds_removed.remove(val)

    current_state = parameters
    solution = current_state
    counter = 0
    
    
    #starts simulated annealing
    while current_temp > final_temp:
        #generate a set of neigbbors
        neighbor = get_neighbors(solution)
        
        it.append(counter)
        meth.append(f"{source_of_removal}{data_points_removed}")
      
        
        #gets the the error of the current solution and the neighbor
        cost_sol, real_cost_sol =  get_cost_removed(solution, y0, t, preds_removed, preys_removed, points, source_of_removal, pred, prey) 
        cost_neighbor, real_cost_neighbor = get_cost_removed(neighbor, y0, t, preds_removed, preys_removed, points, source_of_removal, pred, prey)
  
        #appends the score
        score.append(real_cost_sol)
        cost_diff = cost_sol-cost_neighbor
        
        #checks which solution has a smaller error
        if cost_diff > 0:
            solution = neighbor
        else: #if the neighbor has a larger error the neighbor is selected by the simulated annealing formula
            if random.uniform(0,1) < math.exp(cost_diff / current_temp):
                solution = neighbor
                
        
        current_temp = current_temp * alpha #reduces temperature
        counter += 1
#       
    return solution, get_cost_removed(solution, y0, t, preds_removed, preys_removed, points, source_of_removal, pred, prey), points, it, score, meth
    
    

if __name__ == '__main__':
    main()