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
from help_functions import get_dominance

class HeuristicWarmstart(object):
    """
    Class to warmstart the methods (multi-follower approach,
    extended formulation) using a modified upper-level problem.
    """
    def __init__(self,
                 size,
                 leader_budget,
                 leader_weights,
                 follower_budget,
                 follower_weights,
                 profits,
                 deviations,
                 gamma,
                 modified_profits=None):
        self.size = size
        self.leader_budget = leader_budget
        self.leader_weights = leader_weights
        self.follower_budget = follower_budget
        self.follower_weights = follower_weights
        self.profits = profits
        self.modified_profits = modified_profits
        self.deviations = deviations
        self.gamma = gamma

    def solve_multi_follower(self, sol=None, obj=None):
        model = self._build_feasibility_problem()

        model.Params.OutputFlag = False

        # Check whether the modified leader's problem is feasible.
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            followers = [f for f in range(self.gamma, self.size + 2)]
            
            # Account for Python indexing starting at 0.
            followers = [followers[idx] - 1\
                         for idx in range(len(followers))]
            
            if ((self.gamma == 0) or ((self.deviations <= 0).all())):
                sol, obj = self._solve_subprob(model, followers[0])
                return sol, obj
            
            follower_cnt = len(followers)
            
            sols = [None]*follower_cnt
            objs = [None]*follower_cnt
            for idx, follower in enumerate(followers):
                sol, obj = self._solve_subprob(model, follower)
                sols[idx] = sol
                objs[idx] = obj
                
            max_follower = np.argmax(objs)
            sol = sols[max_follower]
            obj = objs[max_follower]
        return sol, obj

    def solve_extended_form(self, sol=None, obj=None):
        model = gp.Model()

        leader_var = model.addVars(self.size, vtype=GRB.BINARY)
        var_z = model.addVars(self.size, vtype=GRB.CONTINUOUS, lb=0.0)
        var_t = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)

        model.setObjective(gp.quicksum(self.profits[idx]*(1 - leader_var[idx]) + var_z[idx]
                                       for idx in range(self.size))\
                           + self.gamma*var_t,
                           GRB.MINIMIZE)

        model.addConstr(gp.quicksum(self.leader_weights[idx]*leader_var[idx]
                                    for idx in range(self.size))
                        <= self.leader_budget)
        
        model.addConstr(gp.quicksum(self.follower_weights[idx]*(1 - leader_var[idx])
                                    for idx in range(self.size))
                        <= self.follower_budget)

        for idx in range(self.size):
            model.addConstr(var_z[idx] + var_t
                            >= self.deviations[idx]*(1 - leader_var[idx]))

        model.Params.Outputflag = False
        
        model.optimize()
        if model.status == GRB.OPTIMAL:
            sol = [leader_var[idx].X for idx in range(self.size)]
            obj = model.objVal
        return sol, obj
        
    def _build_feasibility_problem(self):
        # Build model.
        model = gp.Model()

        # Construct variables of the leader.
        var = [None]*self.size
        for idx in range(self.size):
            var[idx] = model.addVar(vtype=GRB.BINARY, name="var_%s" % idx)

        # Add leader's budget constraint.
        model.addConstr(gp.quicksum(self.leader_weights[idx]*var[idx]
                                    for idx in range(self.size))
                        <= self.leader_budget)

        # Add follower's budget constraint.
        model.addConstr(gp.quicksum(self.follower_weights[idx]*(1 - var[idx])
                                    for idx in range(self.size))
                        <= self.follower_budget)

        # Add dominance inequalities.
        idx_set_1, idx_set_2 = get_dominance(
            self.leader_weights,
            self.follower_weights,
            self.profits,
            self.deviations
        )
        
        for dominance in range(len(idx_set_1)):
            model.addConstr(var[idx_set_2[dominance]]
                            <= var[idx_set_1[dominance]])

        model.update()
        return model
   
    def _solve_subprob(self, model, follower):
        var = [None]*self.size
        for idx in range(self.size):
            var[idx] = model.getVarByName("var_%s" %idx)

        model.setObjective(-self.gamma*self.deviations[follower]\
                           + gp.quicksum(self.modified_profits[follower][idx]*(1 - var[idx])
                                         for idx in range(self.size)),
                           GRB.MINIMIZE)

        model.Params.OutputFlag = False
        model.optimize()

        sol = [var.x for var in model.getVars()]
        obj = model.objVal
        return sol, obj
