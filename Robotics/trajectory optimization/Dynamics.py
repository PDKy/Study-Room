# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:58:25 2024

@author: chiang
"""

import numpy as np 
from scipy.integrate import solve_ivp

def EvaluateDynamic(xk,uk):
    
    m1 = 1 #kg
    m2 = 1 #kg
    l1 = 1 #m
    l2 = 1 #m
    g = 9.81
    
    I1 = (1/3) *m1* l1**2
    I2 = (1/3) * m2 * l2 **2
    q1 = xk[0]
    q2 = xk[1]
    q = xk[:2]
    qdot = xk[-2:]
    
    q1dot = xk[2]
    q2dot = xk[3]
    M = np.array([[I1 + I2 + m2*l1**2 + 2 * m2*l1* (1/2)*l2 * np.cos(q2), I2 + m2*l1*(1/2)*l2*np.cos(q2)],
                 [I2 + m2*l1*(1/2)*l2*np.cos(q2)  ,   I2]])
    
    C = np.array([[-2*m2*l1*(1/2)*l2*np.sin(q2)*q2dot , -m2*l1* (1/2)*l2 * np.sin(q2) *q2dot], 
                  [m2*l1* (1/2)*l2 *np.sin(q2) * q1dot,  0]])

    G = np.array([-m1 * g * (1/2)*l1 * np.sin(q1) - m2 * g * (l1 * np.sin(q1) + (1/2)*l2 * np.sin(q1+q2)),
                 -m2 * g * (1/2)*l2 * np.sin(q1 + q2)])
    
    B = np.array([0,1])
    
    qdd = np.linalg.inv(M) @ (G + B*uk - C @ qdot)
    
    return np.hstack((xk[-2:],qdd))


    
def simulateDyn(xk,uk,dt):
    xdot = EvaluateDynamic(xk, uk)
    
    
    sol = solve_ivp(lambda t, x: EvaluateDynamic(x, uk), (0, dt), xk, method='RK45', t_eval=[dt])
    return sol.y[:,-1]
    