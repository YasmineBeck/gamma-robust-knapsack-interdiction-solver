# This file is part of the code used for the computational study in the paper
#
#     "Exact Methods for Discrete Gamma-Robust Interdiction Problems
#     with an Application to the Bilevel Knapsack Problem"
#
# by Yasmine Beck, Ivana Ljubic, and Martin Schmidt (2023)
# (https://optimization-online.org/2021/11/8678/).

# Global imports
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from time import time

# Local imports
from help_functions import *
from heuristic_warmstart import HeuristicWarmstart
from nominal_warmstart import NominalWarmstart

def interdiction_cuts_callback(model, where):
    tol = 1e-06
    
    # Get root node gap and root node relaxation.
    if where == GRB.Callback.MIPNODE:
        nodcnt = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
        if nodcnt < 1:
            best_obj = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
            bound = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
            model._root_gap = abs(bound - best_obj)/(abs(best_obj) + tol)

            if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.Status.OPTIMAL:
                model._root_relax = model.cbGetNodeRel(model._aux_var)
    
    if where == GRB.Callback.MIPSOL:        
        leader_var = [None]*model._size
        for idx in range(model._size):
            leader_var[idx] = model.cbGetSolution(model._var[idx])
        aux_var = model.cbGetSolution(model._aux_var)

        # Generate cut.
        start_time = time()
        var_y, var_t, var_z, follower_obj = solve_extended_lower_level(
            leader_var,
            model._gamma,
            model._profits,
            model._deviations,
            model._follower_weights,
            model._follower_budget
        )
        model._solved_subproblems += 1
        
        # Complete follower's decision to a maximal packing.
        if model._max_pack > 0:
            var_y = make_maximal(
                var_y,
                model._follower_weights,
                model._profits,
                model._follower_budget,
                var_t,
                model._deviations
            )
        
        coef = np.multiply(model._profits, var_y)
        const = -model._gamma*var_t - sum(var_z)
        
        if model._cut_type > 0:
            set_a, set_b = lifted_cut_separation(
                var_y,
                leader_var,
                model._profits,
                model._follower_weights,
                model._deviations
            )
            
            for item in range(model._size):
                for idx in range(len(set_b)):
                    if item == set_b[idx]:
                        coef[item] += (model._profits[item]
                                       - model._deviations[item]
                                       - model._profits[set_a[idx]]
                                       + model._deviations[set_a[idx]])
                        
        if aux_var + tol < follower_obj:
            model.cbLazy(
                model._aux_var
                >= const + gp.quicksum(coef[idx]*(1 - model._var[idx])
                                       for idx in range(model._size))
            )
            model._generated_cuts += 1
        end_time = time() - start_time
        model._cut_generation_times.append(end_time)

class ExtendedFormModel(object):
    """
    Class for the extended formulation approach to solve the Gamma-robust
    knapsack interdiction problem.
    """
    def __init__(self,
                 instance_data_dict,
                 cut_type,
                 dominance_ineq,
                 max_pack,
                 warmstart):
        self.size = instance_data_dict["size"]
        self.profits = instance_data_dict["profits"]
        self.leader_weights = instance_data_dict["leader weights"]
        self.follower_weights = instance_data_dict["follower weights"]
        self.leader_budget = instance_data_dict["leader budget"]
        self.follower_budget = instance_data_dict["follower budget"]
        self.gamma = instance_data_dict["gamma"]
        self.deviations = instance_data_dict["deviations"]
        self.time_limit = 3600
        self.cut_type = cut_type
        self.dominance_ineq = dominance_ineq
        self.max_pack = max_pack
        self.warmstart = warmstart

    def run(self):
        start_time = time()

        model, var, aux_var = self._build_master_problem()
        self._warmstart_method(var, aux_var)

        model.Params.LazyConstraints = 1
        model.Params.TimeLimit = self.time_limit
        
        model.optimize(interdiction_cuts_callback)

        runtime = time() - start_time
        sol = [var.x for var in model.getVars()][:-1]
        
        result_dict = {
            "objective": model.objVal,
            "leader decision": sol,
            "runtime": runtime,
            "node count": model.getAttr("NodeCount"),
            "optimality gap": model.MIPGap,
            "root node gap": model._root_gap,
            "root node relaxation": model._root_relax,
            "solved subproblems": model._solved_subproblems,
            "generated cuts": model._generated_cuts,
            "times for cut generation": model._cut_generation_times,
            "cut generation time": sum(model._cut_generation_times)
        }      
        return result_dict

    def _build_master_problem(self):
        # Build model.
        model = gp.Model()
    
        # Construct variables of the leader.
        var = [None]*self.size
        for idx in range(self.size):
            var[idx] = model.addVar(vtype=GRB.BINARY, name="var_%s" % idx)
        aux_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="aux_var")
    
        # Add budget constraint of the leader.
        model.addConstr(
            gp.quicksum(self.leader_weights[idx]*var[idx]
                        for idx in range(self.size))
            <= self.leader_budget,
            name="interdiction_budget_constr"
        )
    
        # Add dominance inequalities.
        if self.dominance_ineq > 0:
            idx_set_1, idx_set_2 = get_dominance(
                self.leader_weights,
                self.follower_weights,
                self.profits,
                self.deviations
            )
            
            for idx in range(len(idx_set_1)):
                model.addConstr(
                    var[idx_set_2[idx]] <= var[idx_set_1[idx]],
                    name="dominance_inequality_%s_%s"
                    % (str(idx_set_1[idx]), str(idx_set_2[idx]))
                )
                
        # Set objective function.
        model.setObjective(aux_var, GRB.MINIMIZE)

        # Pass data to callback.
        model._var = var
        model._aux_var = aux_var
        
        model._size = self.size
        model._profits = self.profits
        model._follower_weights = self.follower_weights
        model._follower_budget = self.follower_budget
        model._gamma = self.gamma
        model._deviations = self.deviations
        
        model._cut_type = self.cut_type
        model._max_pack = self.max_pack

        model._root_gap = None
        model._root_relax = None
        model._solved_subproblems = 0
        model._generated_cuts = 0
        model._cut_generation_times = []
        
        model.update()
        return model, var, aux_var

    def _warmstart_method(self, var, aux_var,
                          var_start=None, aux_var_start=None):
        if self.warmstart == 1:
            starter = HeuristicWarmstart(
                self.size,
                self.leader_budget,
                self.leader_weights,
                self.follower_budget,
                self.follower_weights,
                self.profits,
                self.deviations,
                self.gamma
            )
            var_start, aux_var_start = starter.solve_extended_form()
            
        if self.warmstart == 2:
            starter = NominalWarmstart(
                self.size,
                self.leader_budget,
                self.leader_weights,
                self.follower_budget,
                self.follower_weights,
                self.profits
            )
            var_start, aux_var_start = starter.solve()
            
        if aux_var_start is not None:
            aux_var.start = aux_var_start
            for idx in range(self.size):
                var[idx].start = var_start[idx]
