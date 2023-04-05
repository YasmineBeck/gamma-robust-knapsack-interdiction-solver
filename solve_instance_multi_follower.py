# This file is part of the code used for the computational study in the paper
#
#     "Exact Methods for Discrete Gamma-Robust Interdiction Problems
#     with an Application to the Bilevel Knapsack Problem"
#
# by Yasmine Beck, Ivana Ljubic, and Martin Schmidt (2021)
# (https://optimization-online.org/2021/11/8678/).

# Global imports
import argparse
import json

# Local imports
from instance_data_builder import InstanceDataBuilder
from multi_follower_approach import MultiFollowerModel

argparser = argparse.ArgumentParser(description="Build and solve a "\
                                    "Gamma-robust knapsack interdiction "\
                                    "problem using the multi-follower "\
                                    "approach. Either --uncertainty or "\
                                    "--deviations must be specified.")
argparser.add_argument("--instance_dict_file", action="store", required=True,
                       help="The file containing the nominal instance data "\
                       "as dictionary (required).")
argparser.add_argument("--cut_type", action="store", type=int, default=1,
                       help="The cut type to use: lifted cuts (1) "\
                       "or basic cuts (0); default is 1.")
argparser.add_argument("--dominance_ineq", action="store", type=int, default=1,
                       help="Add dominance inequalities (1) or not (0); "\
                       "default is 1.")
argparser.add_argument("--max_pack", action="store", type=int, default=1,
                       help="Consider maximal packings for the follower (1) "\
                       "or not (0); default is 1.")
argparser.add_argument("--warmstart", action="store", type=int, default=0,
                       help="Use heuristic (1) or nominal solution (2) "\
                       "to warmstart the method, or do not warmstart (0); "\
                       "default is 0.")
argparser.add_argument("--cut_strategy", action="store", type=int, default=4,
                       help="Use cut separation strategy All-In (0), "\
                       "Most-Violated (1), Sorting (2), "\
                       "First-In (3), or Random (4); default is 4.")
argparser.add_argument('--conservatism', action="store", required=True,
                       type=float,
                       help="Level of conservatism (in percent) must be "\
                       "between 0 and 1.")
argparser.add_argument('--uncertainty', action="store", type=float, default=None,
                       help="Uncertainty (in percent) must be between 0 and 1.")
argparser.add_argument('--deviations', action="store", nargs="+",
                       type=float, default=None,
                       help="The objective function deviations, e.g., 1 2 1 "\
                       "for a problem of size 3.")
argparser.add_argument("--output_file", action="store", required=True,
                       help="The file to write the output to (required).")
arguments = argparser.parse_args()

instance_dict_file = arguments.instance_dict_file
cut_type = arguments.cut_type
dominance_ineq = arguments.dominance_ineq
max_pack = arguments.max_pack
warmstart = arguments.warmstart
cut_strategy = arguments.cut_strategy
conservatism = arguments.conservatism
uncertainty = arguments.uncertainty
deviations = arguments.deviations
output_file = arguments.output_file

# Load instance.
builder = InstanceDataBuilder(
    instance_dict_file,
    conservatism=conservatism,
    uncertainty=uncertainty,
    deviations=deviations
)

# Apply multi-follower approach.
multi_follower_data = builder.build_multi_follower_instance()
multi_follower_model = MultiFollowerModel(
    multi_follower_data,
    cut_type,
    dominance_ineq,
    max_pack,
    warmstart,
    cut_strategy
)
results_dict = multi_follower_model.run()

with open(output_file, "w") as outfile:
    json.dump(results_dict, outfile)
    
