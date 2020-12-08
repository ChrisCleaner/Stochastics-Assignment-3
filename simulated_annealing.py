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



import math


n_sim = 4
loc = os.getcwd().replace("\\","/") + "/predator-prey-data.csv"
df = pd.read_csv(loc)
    
pred = (df['x'])
prey = (df['y'])

it = []
score = []
meth = []

    

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
    best_list = []
    for i in range(n_sim):
        parameters = []
        for i in range(4):
            parameters.append(random.uniform(0,2))
        print(f"parameters: {parameters}")
        
            
        parameters, sol2 = sim_annealing(parameters, y0, t, pred, prey, "mse")
        parameters, sol2 = sim_annealing(parameters, y0, t, pred, prey, "rmse")
#        parameters = hill_climbing(parameters,y0, t, pred, prey)
        best_list.append(parameters)
    
        sol = odeint(pend, y0, t, args=(parameters[0], parameters[1], parameters[2], parameters[3]))
        
        
        preys = sol[:,1]
        preds = sol[:,0]
    
        
        
        plt.plot(t, preds, label ='predator')
        plt.plot(t, preys, label = 'prey')
        plt.plot(t, pred, "x", label="actual predator", alpha = 0.5)
        plt.plot(t, prey, "o", label="actual prey", alpha = 0.5)
        plt.legend()
        plt.show()
        print(parameters)
    print(best_list)
    
    parameters = []
    for i in range(4):
        parameters.append(random.uniform(0,2))
    print(f"parameters: {parameters}")
    
    parameters, sol2, del_points = remove_points_sim_an(parameters, y0, t, pred, prey, 30)
    sol = odeint(pend, y0, t, args=(parameters[0], parameters[1], parameters[2], parameters[3]))

    t_removed = t 
    val_points = []
    val_t_points = []
    for point in del_points:
        val_points.append(prey[point])
        val_t_points.append(t_removed[point])
    
    prey = list(prey)
    t_removed = list(t_removed)
    for val in val_points:
        prey.remove(val)
    for val in val_t_points:
        t_removed.remove(val)
        
    
    preys = sol[:,1]
    preds = sol[:,0]

    
    
    plt.plot(t, preds, label ='predator')
    plt.plot(t, preys, label = 'prey')
    plt.plot(t, pred, "x", label="actual predator", alpha = 0.5)
    plt.plot(t_removed, prey, "o", label="actual prey", alpha = 0.5)
    plt.legend()
    plt.show()
    print(parameters)
    
    
    d = {"it":it, "score":score, "method":meth}
    df1 = pd.DataFrame(data=d)
    sns.lineplot(data=df1, x="it",y="score", hue="method")
    
    
    

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
    

def sim_annealing(parameters, y0, t, pred, prey, method):
    """
    Algorithm of simmulated annealing according to this example:
        https://medium.com/swlh/how-to-implement-simulated-annealing-algorithm-in-python-ab196c2f56a0
        
    """
    init_temp = 10
    final_temp = 0.1
    alpha = 0.999
    
    current_temp = init_temp
    y0 = [pred[0], prey[0]]
    

    current_state = parameters
    solution = current_state
    counter = 0
    
    
    while current_temp > final_temp:
        #generate a set of neigbbors
        neighbor = get_neighbors(solution)
        
        
        
        #print(neighbor)
        
        sol_cost = get_cost(solution, y0, t, pred, prey, method) 
        neighbor_cost = get_cost(neighbor, y0, t, pred, prey, method)
  
        cost_diff = sol_cost-neighbor_cost
        
        it.append(counter)
        if method == "mse":
            score.append(sqrt(sol_cost))
        else:
            score.append(sol_cost)
        meth.append(method)
        
        if cost_diff > 0:
            solution = neighbor
        else: 
            if random.uniform(0,1) < math.exp(cost_diff / current_temp):
                solution = neighbor
                
                
        current_temp = current_temp * alpha
        counter += 1
#        if (counter % 10) == 0:
#            print(current_temp)
#            sol = odeint(pend, y0, t, args=(solution[0],solution[1],solution[2],solution[3]))
#            print(solution)
#    
#    
#            preys = sol[:,1]
#            preds = sol[:,0]
#        
#            
#            
#            plt.plot(t, preds, label ='predator')
#            plt.plot(t, preys, label = 'prey')
#            plt.plot(t, pred, "x", label="actual predator", alpha = 0.5)
#            plt.plot(t, prey, "o", label="actual prey", alpha = 0.5)
#            plt.legend()
#            plt.show()
    print(counter)
    return solution, get_cost(neighbor, y0, t, pred, prey, method)

def create_eight_neighbors(parameters, scale):
    neighbors = []
    
    for i in range(4):
        new_para = parameters.copy()
        new_para[i] += scale
        neighbors.append(new_para)
        
        new_para = parameters.copy()
        new_para[i] -= scale
        neighbors.append(new_para)
    
    return neighbors
                
        
def hill_climbing(parameters,y0, t, pred, prey):
    
    scale = 0.05

    new_param = parameters.copy()
    costs = get_cost(new_param,y0, t, pred, prey)
    
    while True:
        
        neighbors = create_eight_neighbors(new_param, scale)
        cost_list = []
        
        for para in neighbors:
            cost_list.append(get_cost(para,y0, t, pred, prey))
        
        if min(cost_list) < costs:
            costs = min(cost_list)
            best_para_index = cost_list.index(min(cost_list))
            new_param = neighbors[best_para_index]
        else:
            break
    return new_param

def get_cost_removed(parameters, y0, t, pred, prey, list_removed_points):
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

    preys = np.delete(preys, list_removed_points)
        
    
    mse1 =  (sum(abs(pred - preds)))
    mse2 = (sum(abs(prey - preys)))
    
    return sqrt(mse1 + mse2) #newly added since last run 2.47pm 8dec


def remove_points_sim_an(parameters, y0, t, pred, prey, data_points_removed):
    """
    Algorithm of simmulated annealing according to this example:
        https://medium.com/swlh/how-to-implement-simulated-annealing-algorithm-in-python-ab196c2f56a0
        
    """
    init_temp = 100
    final_temp = 0.01
    alpha = 0.999
    
    preds_removed = list(pred.copy())
    preys_removed = list(prey.copy())
    
    current_temp = init_temp
    y0 = [pred[0], prey[0]]
    
    points = random.sample(range(100), 10)
    val_points = []
    for point in points:
        val_points.append(preys_removed[point])
    
    for val in val_points:
        preys_removed.remove(val)

    current_state = parameters
    solution = current_state
    counter = 0
    
    
    while current_temp > final_temp:
        #generate a set of neigbbors
        neighbor = get_neighbors(solution)
        
        
        
        #print(neighbor)
        
        cost_diff = get_cost_removed(solution, y0, t, preds_removed, preys_removed, points) - get_cost_removed(neighbor, y0, t, preds_removed, preys_removed, points)
  
        
        if cost_diff > 0:
            solution = neighbor
        else: 
            if random.uniform(0,1) < math.exp(cost_diff / current_temp):
                solution = neighbor
                
                
        current_temp = current_temp * alpha
        counter += 1
#        if (counter % 10) == 0:
#            print(current_temp)
#            sol = odeint(pend, y0, t, args=(solution[0],solution[1],solution[2],solution[3]))
#            print(solution)
#    
#    
#            preys = sol[:,1]
#            preds = sol[:,0]
#        
#            
#            
#            plt.plot(t, preds, label ='predator')
#            plt.plot(t, preys, label = 'prey')
#            plt.plot(t, pred, "x", label="actual predator", alpha = 0.5)
#            plt.plot(t, prey, "o", label="actual prey", alpha = 0.5)
#            plt.legend()
#            plt.show()
    print(counter)
    return solution, get_cost(neighbor, y0, t, pred, prey), points
    
    

if __name__ == '__main__':
    main()