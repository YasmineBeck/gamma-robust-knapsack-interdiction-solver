# This file is part of the code used for the computational study in the paper
#
#     "Exact Methods for Discrete Gamma-Robust Interdiction Problems
#     with an Application to the Bilevel Knapsack Problem"
#
# by Yasmine Beck, Ivana Ljubic, and Martin Schmidt (2021)
# (https://optimization-online.org/2021/11/8678/).

# Global imports
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def get_dominance(leader_weights, follower_weights, profits, deviations=None):
    # Determines the set of items that satisfy the dominance properties
    # stated in Theorem 3 of the paper.
    size = len(leader_weights)
    if deviations is None:
        deviations = np.zeros(size)

    idx_set_1 = []
    idx_set_2 = []
    for item_1 in range(size):
        for item_2 in range(size):
            if item_1 == item_2:
                pass
            else:
                if ((leader_weights[item_1] <= leader_weights[item_2])
                    and (follower_weights[item_1] <= follower_weights[item_2])
                    and (profits[item_1] >= profits[item_2])
                    and (profits[item_1] - deviations[item_1] >=
                         profits[item_2] - deviations[item_2])):
                    idx_set_1.append(item_1)
                    idx_set_2.append(item_2)
    return idx_set_1, idx_set_2

def lifted_cut_separation(follower_var, leader_var, profits,
                          follower_weights, deviations=None):
    # Determines the set of items that satify the requirements of
    # Theorems 4 or 5 in the paper.
    if deviations is None:
        deviations = [0]*len(profits)
        
    set_a = []
    set_b = []
    zeros = np.where(follower_var < 0.5)[0]
    ones = np.where(follower_var > 0.5)[0]
    for one in ones:
        candidates = []
        for zero in zeros:
            if zero not in set_b:
                if ((follower_weights[one] >= follower_weights[zero])
                    and (profits[one] - deviations[one]
                         < profits[zero] - deviations[zero])
                    and (deviations[one] <= deviations[zero])):
                    candidates.append(zero)
        all_candidates = len(candidates)
        if all_candidates > 0:
            coefs = []
            for candidate in range(all_candidates):
                coef = ((profits[candidates[candidate]]
                         - deviations[candidates[candidate]]
                         - profits[one]
                         + deviations[candidates[candidate]])\
                        *(1 - leader_var[candidates[candidate]]))
                coefs.append(coef)
            max_coef = np.argmax(coefs)
            set_a.append(one)
            set_b.append(candidates[max_coef])
    return set_a, set_b

def make_maximal(array_var, profits, weights, budget,
                 single_var=0, deviations=None):
    # Completes a feasible decision to a maximal packing.
    size = len(array_var)
    if deviations is None:
        deviations = np.zeros(size)
    residual = budget - sum(weights[idx]*array_var[idx] for idx in range(size))
    
    # Order items in decreasing order according to profit-to-weight ratio. 
    order = np.argsort(-np.divide(profits, weights))
    array_var = array_var[order]
    weights = weights[order]
    deviations = deviations[order]
    
    idx = 0
    while ((idx < size) and (residual > 0)):
        if ((array_var[idx] < 0.5)
            and (residual - weights[idx] >= 0)
            and (single_var >= deviations[idx])):
            residual -= weights[idx]
            array_var[idx] = 1
        idx += 1
        
    # Revert ordering.
    revert_order = np.argsort(order)
    array_var = array_var[revert_order]
    return array_var 

def solve_lower_level(leader_var, profits,
                      follower_weights, follower_budget):
    # Solves the parameterized lower-level problem in its nominal form.
    size = len(profits)
    
    # Build model.
    model = gp.Model()
    
    # Add variables of the follower.
    var = model.addVars(size, vtype=GRB.BINARY)
    
    # Set lower-level objective.
    model.setObjective(gp.quicksum(profits[idx]*var[idx]
                                   for idx in range(size)),
                       GRB.MAXIMIZE)
    
    # Add budget constraint of the follower.
    model.addConstr(gp.quicksum(follower_weights[idx]*var[idx]
                                for idx in range(size))
                    <= follower_budget)
    
    # Add interdiction constraints.
    for idx in range(size):
        model.addConstr(var[idx] <= 1 - leader_var[idx])
        
    model.Params.OutputFlag = False

    # Optimize.
    model.optimize()
    
    # Extract solution.
    var = np.array([var.x for var in model.getVars()])
    obj = model.objVal
    return var, obj

def solve_extended_lower_level(leader_var, gamma, profits, deviations,
                               follower_weights, follower_budget):
    # Solves the parameterized lower-level problem in its extended form.
    size = len(profits)
    
    # Build model.
    model = gp.Model()

    # Add variables of the follower.
    var_y = model.addVars(size, vtype=GRB.BINARY)
    var_z = model.addVars(size, vtype=GRB.CONTINUOUS, lb=0.0)
    var_t = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
    
    # Set lower-level objective.
    model.setObjective(-gamma*var_t\
                       + gp.quicksum(profits[idx]*var_y[idx] - var_z[idx]
                                     for idx in range(size)),
                       GRB.MAXIMIZE)

    # Add budget constraint of the follower.
    model.addConstr(gp.quicksum(follower_weights[idx]*var_y[idx]
                                for idx in range(size))
                                <= follower_budget)

    for idx in range(size):
        # Add interdiction constraints.
        model.addConstr(var_y[idx] <= 1 - leader_var[idx])
        
        # Add robustification constraints.
        model.addConstr(var_t + var_z[idx] >= deviations[idx]*var_y[idx])

    model.Params.OutputFlag = False

    # Optimize.
    model.optimize()
    
    # Extract solution.
    var_y = np.array([var_y[idx].X for idx in range(size)])
    var_z = np.array([var_z[idx].X for idx in range(size)])
    var_t = var_t.X
    obj = model.objVal
    return var_y, var_t, var_z, obj
