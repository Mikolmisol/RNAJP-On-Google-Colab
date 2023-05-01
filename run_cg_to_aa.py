import sys, os
from recover_all_atom_structure import convert_cg_to_aa

cg_file = sys.argv[1]
aa_file = sys.argv[2]

# get the environment varaible RNAJP_HOME where the source codes are.
RNAJP_HOME = os.getenv("RNAJP_HOME")
if RNAJP_HOME is None:
    raise ValueError(f'Failed in finding the environment variable "RNAJP_HOME". Please follow the installation instructions in the user manual.')
if RNAJP_HOME[-1] == "/":
    RNAJP_HOME = RNAJP_HOME[0:-1]

convert_cg_to_aa(RNAJP_HOME,cg_file,aa_file)
