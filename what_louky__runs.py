# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:00:48 2020

@author: Gebruiker
"""
# -- coding: utf-8 --
"""
Created on Sat Dec  5 11:54:12 2020
@author: Gebruiker
"""
import numpy as np
import pandas as pd
import random
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os
from math import sqrt
import scipy.stats as stats

import csv

import math
import time

n_sim = 50

init_temp = 50
final_temp = 0.001
alpha = 0.6





best_score = 100
best_parms = 0
best_method = 0
best_it = 0


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
        it = []
        score = []
        meth = []
        
        random.seed(i)
        parameters = []
        for i in range(4):
            parameters.append(1)
        print(f"parameters: {parameters}")
        

        start = time.time()
  
        parameters, sol2,  it, score, meth= sim_annealing(parameters, y0, t, pred, prey, 'mse', it, score, meth)
        end = time.time()
        print('elapsed time', end - start)
        parameters, sol2,  it, score, meth = sim_annealing(parameters, y0, t, pred, prey, 'rmse', it, score, meth)
        
        dat = [it, score, meth]
        data = pd.DataFrame(np.transpose(dat))
        
        
        
       
        
        with open(loc2,'a', newline="\n") as fd:
            for i in range(len(data)):
               writer = csv.writer(fd)
               writer.writerow((list(data.iloc[i]))) 
                
         
            


        #parameters = hill_climbing(parameters,y0, t, pred, prey)
        #best_list.append(parameters)
    
        sol = odeint(pend, y0, t, args=(parameters[0], parameters[1], parameters[2], parameters[3]))
        
        
        preys = sol[:,1]
        preds = sol[:,0]
    
        
        """
        plt.plot(t, preds, label ='predator')
        plt.plot(t, preys, label = 'prey')
        plt.plot(t, pred, "x", label="actual predator", alpha = 0.5)
        plt.plot(t, prey, "o", label="actual prey", alpha = 0.5)
        plt.legend()
        plt.show()
        print(parameters)
        """
    #print(best_list)
    
    #print('iterations', it)
    #print('score', score)
    


    
    """
    its = (len(it))/(n_sim)/2
    print(its)
    
    newdf = df.loc[df['it'] == its -1] 
    newdf1 = newdf.loc[newdf['method'] == 'mse']
    newdf2 = newdf.loc[newdf['method'] == 'rmse']
    
 
    
    print(stats.shapiro(newdf1['score']))
    print(stats.shapiro(newdf2['score']))
    
    
    print(list(newdf1['score']))
    
    print(list(newdf2['score']))
    
    print('best results')
    print(best_score, best_parms, best_method, best_it)
    """
    
    print('best results')
    print(best_score, best_parms, best_method, best_it)
    

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
    

def sim_annealing(parameters, y0, t, pred, prey, method, it, score, meth):
    """
    Algorithm of simmulated annealing according to this example:
        https://medium.com/swlh/how-to-implement-simulated-annealing-algorithm-in-python-ab196c2f56a0
        
    """

    
    
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
        neigh_cost =  get_cost(neighbor, y0, t, pred, prey, method)
        
        it.append(counter)
        #score.append(sol_cost)
         
        if method == "mse":
            score.append(sqrt(sol_cost))
        else:
            score.append(sol_cost)
        meth.append(method)
        
        cost_diff = sol_cost - neigh_cost
        
        global best_score, best_parms, best_method, best_it

        if sol_cost < best_score:
            best_score = sol_cost
            best_parms = solution
            best_method = method
            best_it = counter
  
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
    print()  
    data = [it, score, meth]
    #df = pd.DataFrame(data=d)
    return solution, get_cost(solution, y0, t, pred, prey, method), it, score, meth

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
    

if __name__ == '__main__':
    main()