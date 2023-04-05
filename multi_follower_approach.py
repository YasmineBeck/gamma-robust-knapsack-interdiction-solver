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
        cnt = 0
        added_cuts = 0
        cut_gen_times = []
        
        leader_var = [None]*model._size
        for idx in range(model._size):
            leader_var[idx] = model.cbGetSolution(model._var[idx])
        aux_var = model.cbGetSolution(model._aux_var)

        cut_sel_start_time = time()

        # Create lists for coefficients, constant terms, and violations of
        # cuts to determine most violated cut.
        if model._cut_strategy == 1:
            coefs = []
            consts = []
            violations = []
            
        # Sort followers.
        if model._cut_strategy == 2:
            order = np.argsort(
                -np.divide(model._violation, model._frequency, where=False)
            )
            model._followers = np.asarray(model._followers)[order]
            model._frequency = model._frequency[order]
            model._violation = model._violation[order]
            
        # Shuffle follower sub-problems randomly.
        if model._cut_strategy == 4:
            np.random.seed(cnt)
            np.random.shuffle(model._followers)
            
        # Generate cuts.
        for follower in model._followers:
            cut_gen_start_time = time()
            follower_var, follower_obj = solve_lower_level(
                leader_var,
                model._modified_profits[follower],
                model._follower_weights,
                model._follower_budget
            )
            
            const = -model._gamma*model._deviations[follower]
            follower_obj += const
            model._solved_subproblems += 1
            
            # Complete follower's decision to a maximal packing.
            if model._max_pack > 0:
                follower_var = make_maximal(
                    follower_var,
                    model._modified_profits[follower],
                    model._follower_weights,
                    model._follower_budget
                )
                
            # Update frequency and violation for sorting.
            if model._cut_strategy == 2:
                model._frequency[cnt] += 1
                model._violation[cnt] += follower_obj - aux_var

            if aux_var + tol < follower_obj:
                coef = np.multiply(
                    model._modified_profits[follower],
                    follower_var
                )
                
                if model._cut_type > 0:
                    set_a, set_b = lifted_cut_separation(
                        follower_var,
                        leader_var,
                        model._modified_profits[follower],
                        model._follower_weights
                    )
                    
                    for item in range(model._size):
                        for idx in range(len(set_b)):
                            if item == set_b[idx]:
                                coef[item] += (model._modified_profits[follower][item]
                                               - model._modified_profits[follower][set_a[idx]])

                if model._cut_strategy == 1:
                    coefs.append(coef)
                    consts.append(const)
                    violations.append(follower_obj - aux_var)
                else:
                    model.cbLazy(model._aux_var
                                 >= const\
                                 + gp.quicksum(coef[idx]*(1 - model._var[idx])
                                               for idx in range(model._size)))
                    added_cuts += 1
                    model._generated_cuts += 1

            cut_gen_time = time() - cut_gen_start_time
            cut_gen_times.append(cut_gen_time)

            # Add only one cut for cut separation strategies
            # sorting, first-in, or random.
            if ((model._cut_strategy > 1) and (added_cuts > 0)):
                break

            # Only one follower is considered if no uncertainties arise.
            if ((model._gamma == 0) or (model._deviations <= 0).all()):
                break

        start_time = time()
        # Add most violated cut.
        if ((model._cut_strategy == 1) and (len(violations) > 0)):
            max_viol = np.argmax(violations)
            model.cbLazy(model._aux_var
                         >= consts[max_viol]\
                         + gp.quicksum(coefs[max_viol][idx]*(1 - model._var[idx])
                                       for idx in range(model._size)))
        end_time = time() - start_time
        cut_sel_time = time() - cut_sel_start_time
        
        model._cut_gen_real_times.append(sum(cut_gen_times) + end_time)
        model._cut_gen_ideal_times.append(max(cut_gen_times) + end_time)
        model._cut_selection_times.append(cut_sel_time)
        cnt += 1

class MultiFollowerModel(object):
    """
    Class for the multi-follower approach to solve the Gamma-robust
    knapsack interdiction problem.
    """
    def __init__(self,
                 instance_data_dict,
                 cut_type,
                 dominance_ineq,
                 max_pack,
                 warmstart,
                 cut_strategy):
        self.size = instance_data_dict["size"]
        self.profits = instance_data_dict["profits"]
        self.modified_profits = instance_data_dict["modified profits"]
        self.leader_weights = instance_data_dict["leader weights"]
        self.follower_weights = instance_data_dict["follower weights"]
        self.leader_budget = instance_data_dict["leader budget"]
        self.follower_budget = instance_data_dict["follower budget"]
        self.gamma = instance_data_dict["gamma"]
        self.deviations = instance_data_dict["deviations"]

        self.cut_type = cut_type
        self.dominance_ineq = dominance_ineq
        self.max_pack = max_pack
        self.warmstart = warmstart
        self.cut_strategy = cut_strategy

        # Initialize follower sub-problems.
        if self.gamma < 1:
            self.followers = [0]
        else:
            # Reduction of sub-problems as in Alvarez-Miranda et al. (2013).
            self.followers = [s for s in range(self.gamma, self.size + 2)]
            
            # Account for Python indexing starting at 0.
            self.followers = [self.followers[idx] - 1\
                              for idx in range(len(self.followers))]

        self.follower_cnt = len(self.followers)
        
    def run(self):
        start_time = time()
        
        model, var, aux_var = self._build_master_problem()
        self._warmstart_method(var, aux_var)
        
        model.Params.LazyConstraints = 1
        model.Params.TimeLimit = 3600
        
        model.optimize(interdiction_cuts_callback)

        runtime = time() - start_time
        ideal_runtime = (runtime
                         - sum(model._cut_gen_real_times)
                         + sum(model._cut_gen_ideal_times))
        ideal_cut_selection_runtime = (sum(model._cut_selection_times)
                                       - sum(model._cut_gen_real_times)
                                       + sum(model._cut_gen_ideal_times))
        sol = [var.x for var in model.getVars()][:-1]
        
        result_dict = {
            "objective": model.objVal,
            "leader decision": sol,
            "total runtime": runtime,
            "ideal runtime": ideal_runtime,
            "node count": model.getAttr("NodeCount"),
            "optimality gap": model.MIPGap,
            "root node gap": model._root_gap,
            "root node relaxation": model._root_relax,
            "solved subproblems": model._solved_subproblems,
            "generated cuts": model._generated_cuts,
            "times for cut selection": model._cut_selection_times,
            "real cut selection time": sum(model._cut_selection_times),
            "ideal cut selection time": ideal_cut_selection_runtime,
            "real times for cut generation": model._cut_gen_real_times,
            "real cut generation time": sum(model._cut_gen_real_times),
            "ideal times for cut generation": model._cut_gen_ideal_times,
            "ideal cut generation time": sum(model._cut_gen_ideal_times)
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
        model.addConstr(gp.quicksum(self.leader_weights[idx]*var[idx]
                                    for idx in range(self.size))
                        <= self.leader_budget,
                        name="interdiction_budget_constr")
    
        # Add dominance inequalities.
        if self.dominance_ineq > 0:
            idx_set_1, idx_set_2 = get_dominance(
                self.leader_weights,
                self.follower_weights,
                self.profits,
                self.deviations
            )

            for idx in range(len(idx_set_1)):
                model.addConstr(var[idx_set_2[idx]] <= var[idx_set_1[idx]],
                                name="dominance_inequality_%s_%s"
                                % (str(idx_set_1[idx]), str(idx_set_2[idx])))
                
        # Set objective function.
        model.setObjective(aux_var, GRB.MINIMIZE)

        # Pass data to callback.
        model._var = var
        model._aux_var = aux_var
        
        model._size = self.size
        model._profits = self.profits
        model._modified_profits = self.modified_profits
        model._follower_weights = self.follower_weights
        model._follower_budget = self.follower_budget
        model._gamma = self.gamma
        model._deviations = self.deviations
        
        model._cut_type = self.cut_type
        model._max_pack = self.max_pack
        model._cut_strategy = self.cut_strategy
        
        model._followers = self.followers
        model._frequency = np.zeros(self.follower_cnt)
        model._violation = np.zeros(self.follower_cnt)
        
        model._cut_gen_real_times = []
        model._cut_gen_ideal_times = []
        model._cut_selection_times = []
        model._root_gap = None
        model._root_relax = None
        model._solved_subproblems = 0
        model._generated_cuts = 0
    
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
                self.gamma,
                self.modified_profits
            )
            var_start, aux_var_start = starter.solve_multi_follower()
            
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
