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

# Local imports
from help_functions import *
    
def interdiction_cuts_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            tol = 1e-06
            
            leader_var = [None]*model._size
            for idx in range(model._size):
                leader_var[idx] = model.cbGetSolution(model._leader_var[idx])
            aux_var = model.cbGetSolution(model._aux_var)

            follower_var, follower_obj = solve_lower_level(
                    leader_var,
                    model._profits,
                    model._follower_weights,
                    model._follower_budget
            )
            
            follower_var = make_maximal(
                    follower_var,
                    model._profits,
                    model._follower_weights,
                    model._follower_budget
            )

            coef = np.multiply(model._profits, follower_var)
            set_a, set_b = lifted_cut_separation(
                    follower_var,
                    leader_var,
                    model._profits,
                    model._follower_weights
            )
            
            for item in range(model._size):
                for idx in range(len(set_a)):
                    if item == set_b[idx]:
                        coef[item] += (model._profits[item]
                                       - model._profits[set_a[idx]])
                        
            if aux_var + tol < follower_obj:
                model.cbLazy(model._aux_var
                             >= gp.quicksum(coef[idx]*(1 - model._leader_var[idx])
                                            for idx in range(model._size)))

class NominalWarmstart(object):
    """
    Class to warmstart the methods (multi-follower approach,
    extended formulation) using the nominal problem.
    """
    def __init__(self,
                 size,
                 leader_budget,
                 leader_weights,
                 follower_budget,
                 follower_weights,
                 profits):
        self.size = size
        self.leader_budget = leader_budget
        self.leader_weights = leader_weights
        self.follower_budget = follower_budget
        self.follower_weights = follower_weights
        self.profits = profits
                
    def solve(self, sol=None, obj=None):
        # Build model.
        model = gp.Model()

        # Construct upper-level variables.
        leader_var = model.addVars(self.size, vtype=GRB.BINARY)
        aux_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)

        model.setObjective(aux_var, GRB.MINIMIZE)

        # Add leader's budget constraint.
        model.addConstr(gp.quicksum(self.leader_weights[idx]*leader_var[idx]
                                    for idx in range(self.size))
                        <= self.leader_budget)

        # Add dominance inequalities.
        idx_set_1, idx_set_2 = get_dominance(
                self.leader_weights,
                self.follower_weights,
                self.profits
        )
        
        for dominance in range(len(idx_set_1)):
            model.addConstr(leader_var[idx_set_2[dominance]]
                            <= leader_var[idx_set_1[dominance]])

        model._leader_var = leader_var
        model._aux_var = aux_var
        model._size = self.size
        model._profits = self.profits
        model._follower_weights = self.follower_weights
        model._follower_budget = self.follower_budget

        model.Params.LazyConstraints = 1
        model.Params.TimeLimit = 3600
        model.Params.OutputFlag = False
        
        model.optimize(interdiction_cuts_callback)

        if model.status == GRB.OPTIMAL:
                sol = np.asarray([var.x for var in model.getVars()][:-1])
                obj = model.objVal
        return sol, obj
