# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 15:31:32 2020

@author: chris
simulated annealing
"""
import numpy as np
import pandas as pd
import random

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import odeint

import math
from math import sqrt
import os



loc = os.getcwd().replace("\\","/") + "/predator-prey-data.csv"
df = pd.read_csv(loc)
pred = df["x"]
prey = df["y"]
t = df["t"]
y0 = [pred[0], prey[0]]

plt.plot(t,prey)
plt.plot(t,pred)
plt.legend(["Prey", "Predator"])
plt.show()

alpha = 0.9
beta = 0.7
gamma = 0.9
delta = 0.5
parameters = [alpha, beta, gamma, delta]




def predprey(y0, t, alpha, beta, gamma, delta):
    prey, pred = y0
    
    dpreddt = alpha * pred - beta * prey * pred
    dpreydt = delta * prey * pred - gamma * prey
    
    return dpreydt, dpreddt

def create_neighbors(params):
    
    neighbors = []
    range_ = 0.1
    for i in range(4):
        new_params = params.copy()
        new_params[i] += np.random.uniform(-range_,range_)
        neighbors.append(new_params)
        
    return neighbors

def cost(param):
    #get data points
    sol = odeint(predprey, y0, t, args =  (param[0],param[1],param[2],param[3]))
    preds = sol[:,1]
    preys = sol[:,0]
    error1 = sum((np.array(preds) - list(pred))**2)
    error2 = sum((np.array(preys) - list(prey))**2)
    
    return error1 + error2
    

solution = odeint(predprey, y0, t, args =  (parameters[0],parameters[1],parameters[2],parameters[3]))
preds = solution[:,1]
preys = solution[:,0]
plt.plot(t,preys)
plt.plot(t,preds)
plt.legend(["Prey", "Predator"])
plt.show()

def sim_annealing():
    """
    Simmulated Annealing
    """
    
    start_temp = 100000 #molten
    end_temp = 0.005 #room temp in K 
    cooling_rate = .99
    
    
    current_temp = start_temp
    new_parameters = parameters.copy()
    
    while current_temp > end_temp:
        #create neighbors
        neighbors = create_neighbors(new_parameters)
    
        neighbor = random.choice(neighbors)
    
    
        #check if they are good. 
        diff = cost(neighbor) - cost(new_parameters)
        
    
        if diff < 0:
            new_parameters = neighbor.copy()
        else:
            try:
                if np.random.random() > math.exp((diff)/current_temp):
                    new_parameters = neighbor.copy()
            except:
                print(f"fail at difference {diff}")
        
        if current_temp > 100:
            current_temp -= 1
        else:
            current_temp = current_temp * cooling_rate
        
    print(new_parameters)
    solution = odeint(predprey, y0, t, args =  (new_parameters[0],new_parameters[1],new_parameters[2],new_parameters[3]))
    preds = solution[:,1]
    preys = solution[:,0]
    plt.plot(t,preys)
    plt.plot(t,preds)
    plt.legend(["Prey", "Predator"])
    plt.show()
    

def hill_climbing():
    """
    find neighbor, check if it is better, if it's better, go there repeat, otherwise stop
    """
    new_param = parameters.copy()
    new_param = [1,1,1,1]
    costs = cost(new_param)
    neighbor_cost = 0
    change = 0.05
    
    while (costs > neighbor_cost) :
        
        neighbor_1 = new_param.copy()
        neighbor_2 = new_param.copy()
        for i in range(4):
            neighbor_1[i] = neighbor_1[i] + random.uniform(-change,change)
            neighbor_2[i] = neighbor_2[i] + random.uniform(-change,change)
            
        neighbor_cost_1 = cost(neighbor_1)
        neighbor_cost_2 = cost(neighbor_2)
    
        if neighbor_cost_1 < costs:
            new_param = neighbor_1
            neighbor_cost = neighbor_cost_1
            
        elif neighbor_cost_2 < costs:
            new_param = neighbor_2
            neighbor_cost = neighbor_cost_2
        else: break

    print(new_param)
    new_parameters = new_param
    solution = odeint(predprey, y0, t, args =  (new_parameters[0],new_parameters[1],new_parameters[2],new_parameters[3]))
    preds = solution[:,1]
    preys = solution[:,0]
    plt.plot(t,preys)
    plt.plot(t,preds)
    plt.legend(["Prey", "Predator"])
    plt.show()
    
def hill_climbing2():
    """
    find neighbor, check if it is better, if it's better, go there repeat, otherwise stop
    """
    new_param = parameters.copy()
    costs = cost(new_param)
    neighbor_cost = 0
    change = 0.05
    counter = 0
    
    while (costs > neighbor_cost) :
        
        neighbor_1 = new_param.copy()
        neighbor_2 = new_param.copy()
        for i in range(4):
            neighbor_1[i] = neighbor_1[i] - random.uniform(0,change)
            neighbor_2[i] = neighbor_2[i] + random.uniform(0,change)
            
            neighbor_cost_1 = cost(neighbor_1)
            neighbor_cost_2 = cost(neighbor_2)
        
            if neighbor_cost_1 < costs:
                new_param = neighbor_1
                neighbor_cost = neighbor_cost_1
                counter += 1
                
            elif neighbor_cost_2 < costs:
                new_param = neighbor_2
                neighbor_cost = neighbor_cost_2
                counter += 1
        if counter == 4:
            break
            
            

    print(new_param)
    new_parameters = new_param
    solution = odeint(predprey, y0, t, args =  (new_parameters[0],new_parameters[1],new_parameters[2],new_parameters[3]))
    preds = solution[:,1]
    preys = solution[:,0]
    plt.plot(t,preys)
    plt.plot(t,preds)
    plt.legend(["Prey", "Predator"])
    plt.show()
    
sim_annealing()