# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 16:31:18 2024

@author: chiang
"""

from Dynamics import simulateDyn,EvaluateDynamic
from dynamics_constraints import CollocationConstraintEvaluator,AddCollocationConstraints
import numpy as np
from pydrake.all import (
    DiagramBuilder, Simulator, FindResourceOrThrow, MultibodyPlant, PiecewisePolynomial, SceneGraph,
    Parser, JointActuatorIndex, MathematicalProgram, Solve
)
from pydrake.autodiffutils import AutoDiffXd
def find_trajectory(N,initial_state,final_state,tf):
    n_x = 4
    n_u = 1
    
    
    prog = MathematicalProgram()
    x = np.zeros((N,n_x) , dtype = "object")
    u = np.zeros((N,n_u), dtype = "object")
    for i in range(N):
        x[i] = prog.NewContinuousVariables(n_x, "x_"+str(i))
        u[i] = prog.NewContinuousVariables(n_u,"u_"+str(i))
        
    timestep = np.linspace(0,tf,N)
    x0 = x[0]
    xf = x[-1]
    prog.AddLinearEqualityConstraint(np.eye(len(x0), dtype=AutoDiffXd),initial_state,x[0])
    prog.AddLinearEqualityConstraint(np.eye(len(xf), dtype=AutoDiffXd),final_state,x[-1])
    
    u_limit = np.zeros(N) + 20
    prog.AddBoundingBoxConstraint(-u_limit,u_limit,u)
    
    dh = (timestep[1]-timestep[0])
    b = np.zeros((n_u))
    Q = np.eye(n_u)
    for i in range(N-1):
        prog.AddQuadraticCost(dh*Q,b,u[i].flatten())
        prog.AddQuadraticCost(dh*Q,b,u[i+1].flatten())
    
    AddCollocationConstraints(prog, N, x, u, timestep)
    
    u_guess = np.zeros((N, n_u))
    for i in range(len(u_guess)):
        for j in range(len(u_guess[i])):
            u_guess[i][j] = np.random.uniform(-u_limit[j],u_limit[j])
    prog.SetInitialGuess(u,u_guess)
    
    x_guess = np.zeros((N,n_x))
    q0_sequence = np.linspace(initial_state[0],np.pi,N)
    q1_sequence = np.linspace(initial_state[1],0,N)
    for i in range(len(x_guess)):
        x_guess[i][0] = q0_sequence[i]
        x_guess[i][1] = q1_sequence[i]
        
    result = Solve(prog)
    
    x_sol = result.GetSolution(x)
    u_sol = result.GetSolution(u)
    
    u_traj = PiecewisePolynomial.ZeroOrderHold(timestep, u_sol.T)
    
    return u_traj