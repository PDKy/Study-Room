# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 16:15:01 2024

@author: chiang
"""

import numpy as np 
from Dynamics import simulateDyn,EvaluateDynamic
import pydrake.math
from pydrake.autodiffutils import AutoDiffXd

def CollocationConstraintEvaluator(dt, x_i, u_i, x_ip1, u_ip1):
    n_x = 4
    h_i = np.zeros(n_x,dtype=AutoDiffXd)
    # TODO: Add a dynamics constraint using x_i, u_i, x_ip1, u_ip1, dt
    # You should make use of the EvaluateDynamics() function to compute f(x,u)
    f_i = EvaluateDynamic(x_i,u_i)
    f_ip1 = EvaluateDynamic(x_ip1,u_ip1)
    x_ip05 = 0.5*(x_i + x_ip1) - (dt/8) * (f_ip1-f_i)
    u_ip05 = 0.5 * (u_ip1 + u_i)
    #hi = AutoDiffXd(3/(2*dt))*(x_ip1 - x_i) - AutoDiffXd(1/4)*(f_i+f_ip1) - EvaluateDynamics(planar_arm,context,x_ip05,u_ip05)
    h_i = (3/(2*dt))*(x_ip1 - x_i) - (0.25)*(f_i+f_ip1) - EvaluateDynamic(x_ip05, u_ip05)
    
    return h_i

def AddCollocationConstraints(prog, N, x, u, timesteps):
    n_x = 4
    n_u = 1
    epsilon = 10**(-2)
    lower_b = np.zeros(n_x) - epsilon
    upper_b = np.zeros(n_x) + epsilon
    
    for i in range(N-1):
        def CollocationConstraintHelper(vars):
            x_i = vars[:n_x]
            u_i = vars[n_x:n_x+n_u]
            x_ip1 = vars[n_x+n_u:2*n_x+n_u]
            u_ip1 = vars[-n_u:]
            return CollocationConstraintEvaluator(timesteps[i+1] - timesteps[i], x_i, u_i, x_ip1, u_ip1)
        
        prog.AddConstraint(CollocationConstraintEvaluator,lower_b,upper_b,np.hstack(x[i],u[i],x[i+1],u[i+1]))
        
        
        
        
    
    