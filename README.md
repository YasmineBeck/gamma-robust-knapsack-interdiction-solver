# Introduction
This file is part of the code used for the computational study in the paper

    "Exact Methods for Discrete Gamma-Robust Interdiction Problems 
    with an Application to the Bilevel Knapsack Problem"

by Yasmine Beck, Ivana Ljubic, and Martin Schmidt (2021)
(https://optimization-online.org/2021/11/8678/).

# Dependencies
The methods are implemented in Python 3.6.9 and Gurobi 9.1.2 
is used to solve all arising sub-problems.
Further, the following Python packages and modules are required:

   argparse
   json
   numpy
   time

# Contents
extended_formulation_approach.py    Solves a bilevel knapsack interdiction
				    problem using the extended formulation.

help_functions.py		    Contains the following functions:

				    'get_dominance'
				    Determines the set of items that
				    satisfy the dominance properties in
				    Theorem 3 in the paper.

				    'lifted_cut_separation'
				    Determines the set of items that
				    satify the requirements of Theorems 4
				    and 5 in the paper.

				    'make_maximal'
    				    Completes a feasible decision to a
				    maximal packing.

				    'solve_(extended_)lower_level'
				    Solves the parameterized lower-level
				    problem in its nominal (extended)
				    form.

heuristic_warmstart.py		    Solves a modified upper-level problem.
				    The solution can be used to warmstart
				    the proposed methods.

instance_data_builder.py	    Takes a nominal bilevel knapsack
				    interdiction instance and returns a
				    robustified instance based on the
				    uncertainty parameterization given
				    by 'conservatism' and 'uncertainty'
				    or 'deviations'.

multi_follower_approach.py	    Solves a bilevel knapsack interdiction
				    problem using the multi-follower approach.

nominal_warmstart.py		    Solves the nominal knapsack interdiction
				    problem. The solution can be used to
				    warmstart the methods.

solve_instance_extended_form.py	    Builds and solves a Gamma-robust knapsack
				    interdiction problem using the extended
				    formulation.

solve_instance_multi_follower.py    Builds and solves a Gamma-robust knapsack
				    interdiction problem using the
				    multi-follower approach.

# Instance Format
Nominal instance data must be given in form of a dictionary.
For example, the nominal instance considered in Example 1 of
Caprara et al. (2016) would be given in a simple text file containing:

{
"size": 3,
"profits": [4, 3, 3],
"leader weights": [2, 1, 1],
"follower weights": [4, 3, 2],
"leader budget": 2,
"follower budget": 4
}

To account for uncertain objective function coefficients, the following
specifications may be used:

conservatism			A value between 0 and 1 is required.
				It specifies the percentage that the parameter
				Gamma takes of the instance size.
				In the case of a fractional value for Gamma,
				the closest integer is considered.

uncertainty			The percentage deviations in the objective
				function coefficients.
				The value must be between 0 and 1.

deviations			Absolute deviations for the objective
				function coefficients.

Either uncertainty or deviations must be specified.

# Usage
The scripts solve_instance_extended_form.py and solve_instance_multi_follower.py
are used to build a robustified knapsack interdiction problem and solve it
using the extended formulation or the multi-follower formulation, respectively.

Typical usage:
Run
    
python3 solve_instance_extended_form.py --instance_dict_file file.txt --conservatism conservatism_value --uncertainty uncertainty_value --output_file outfile.json

or

python3 solve_instance_multi_follower.py --instance_dict_file file.txt --conservatism conservatism_value --uncertainty uncertainty_value --output_file outfile.json

Necessary arguments:
--instance_dict_file		The file containing the nominal instance data
				as dictionary.

--conservatism			Level of conservatism (in percent)
				must be between 0 and 1.

--output_file			The file to write the output to.

and either
--uncertainty			Uncertainty (in percent)
				must be between 0 and 1.

or
--deviations			The objective function deviations,
				e.g., 1 2 1 for a problem of size 3.


Optional arguments:
--cut_type			The cut type to use: lifted (1)
				or basic interdiction cuts (0);
				default is 1.

--dominance_ineq		Add dominance inequalities (1) or not (0);
				default is 1.

--max_pack			Consider maximal packings for the follower (1)
				or not (0); default is 1.

--warmstart			Use heuristic (1) or nominal solution (2)
				to warmstart the method,
				or do not warmstart (0);
				default is 0.

Additional optional argument for the multi-follower approach only:
--cut_strategy	    	     	Use cut separation strategy All-In (0),
				Most-Violated (1), Sorting (2),
				First-In (3), or Random (4);
				default is 4.