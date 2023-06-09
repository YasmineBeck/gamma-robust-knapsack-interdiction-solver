# This file is part of the code used for the computational study in the paper
#
#     "Exact Methods for Discrete Gamma-Robust Interdiction Problems
#     with an Application to the Bilevel Knapsack Problem"
#
# by Yasmine Beck, Ivana Ljubic, and Martin Schmidt (2023)
# (https://optimization-online.org/2021/11/8678/).

# Global imports
import json
import numpy as np

class InstanceDataBuilder(object):
    """
    Takes an instance (nominal data) of a bilevel knapsack interdiction 
    problem and delivers the necessary data to apply either the 
    multi-follower approach or the extended formulation on the 
    Gamma-robust variant of the problem.
    Nominal instance data must be given as dictionary.
    """
    def __init__(self,
                 instance_data_file,
                 conservatism,
                 uncertainty=None,
                 deviations=None,
                 only_positive_profits=True):
        self.instance_data_file = instance_data_file
        self.conservatism = conservatism
        if ((conservatism < 0) or (conservatism > 1)):
            raise ValueError("Level of conservatism must be between 0 and 1.")
        self.uncertainty = uncertainty
        self.deviations = deviations
        self.only_positive_profits = only_positive_profits

    def build_multi_follower_instance(self):
        # Read (nominal) instance data.
        instance_data = self._read_instance()

        # Add level of conservatism and deviations for robustification.
        instance_data["gamma"] = self._add_conservatism_level(instance_data)
        instance_data["deviations"] = self._add_deviations(instance_data)

        # Sort indices such that the deviations are non-increasing.
        deviations, order = self._sort_indices(instance_data)
        instance_data["deviations"] = deviations
        
        # Update instance data using new order of indices.
        keys = ["profits", "leader weights", "follower weights"]
        for key in keys:
            if key in instance_data:
                instance_data[key] = instance_data[key][order]
    
        if (instance_data["follower weights"] < 0).any():
            raise ValueError("Follower weights must be positive.")
        
        # Construct modified profits.
        instance_data["modified profits"]\
            = self._add_modified_profits(instance_data)
        return instance_data

    def build_extended_form_instance(self):
        # Read (nominal) instance data.
        instance_data = self._read_instance()

        # Add level of conservatism and deviations for robustification.
        instance_data["gamma"] = self._add_conservatism_level(instance_data)
        instance_data["deviations"] = self._add_deviations(instance_data)
        return instance_data        
        
    def _read_instance(self):
        with open(self.instance_data_file, "r") as file:
            data_from_file = file.read()
        instance_data = json.loads(data_from_file)
        keys = ["profits", "leader weights", "follower weights"]
        for key in keys:
            if key in instance_data:
                instance_data[key] = np.asarray(instance_data[key])

        if len(instance_data["follower weights"]) != instance_data["size"]:
            raise ValueError("Dimensions do not match (follower weights).")

        if len(instance_data["leader weights"]) != instance_data["size"]:
            raise ValueError("Dimensions do not match (leader weights).")

        if len(instance_data["profits"]) != instance_data["size"]:
            raise ValueError("Dimensions do not match (profits).")

        return instance_data

    def _add_conservatism_level(self, instance_data):
        size = instance_data["size"]
        gamma = int(np.round(self.conservatism*size))
        return gamma

    def _add_deviations(self, instance_data):
        if self.deviations is not None:
            self.deviations = np.asarray(self.deviations)
            if (self.deviations < 0).any():
                raise ValueError("Deviations must be non-negative.")
            if len(self.deviations) != instance_data["size"]:
                raise ValueError("Dimensions do not match (deviations).")
            if self.uncertainty is not None:
                raise ValueError("Either specify uncertainty or deviations.")
            return self.deviations

        if self.uncertainty is not None:
            if ((self.uncertainty < 0) or (self.uncertainty > 1)):
                raise ValueError("Uncertainty must be between 0 and 1.")
            profits = instance_data["profits"]
            deviations = self.uncertainty*profits
            return deviations
        raise ValueError("Either specify uncertainty or deviations.")

    def _sort_indices(self, instance_data):
        deviations = instance_data["deviations"]
        order = np.argsort(-deviations)
        deviations = deviations[order]
        deviations = np.append(deviations, 0)
        return deviations, order
    
    def _add_modified_profits(self, instance_data):
        size = instance_data["size"]
        profits = instance_data["profits"]
        deviations = instance_data["deviations"]
        
        modified_profits = np.zeros((size + 1, size))
        for follower in range(size + 1):
            for idx in range(follower):
                modified_profits[follower, idx]\
                    = profits[idx] - deviations[idx] + deviations[follower]
            for idx in range(follower, size):
                modified_profits[follower, idx] = profits[idx]

        if self.only_positive_profits:
            modified_profits\
                = np.where(modified_profits > 0, modified_profits, 0)
        return modified_profits
