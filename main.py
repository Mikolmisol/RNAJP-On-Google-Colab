import sys, os
import argparse
import time
from parse2D import get_motifs_from_2D
from generate_initial_structure import fold_init_structure
from get_JAR3D_paras import get_jar3d_energy_paras
from fold_RNA import runMD
from postprocess import select_structures_from_trajectory

if __name__ == '__main__':
    # get the environment varaible RNAJP_HOME where the source codes are.
    RNAJP_HOME = os.getenv("RNAJP_HOME")
    if RNAJP_HOME is None:
        raise ValueError(f'Failed in finding the environment variable "RNAJP_HOME". Please follow the installation instructions in the user manual.')
    if RNAJP_HOME[-1] == "/":
        RNAJP_HOME = RNAJP_HOME[0:-1]

    parser = argparse.ArgumentParser(description="RNAJP model for RNA 3D structure prediction.")
    parser.add_argument('-r', '--RNA', default='test', help='The name of RNA and working directory.')
    parser.add_argument('-s', '--secondary_structure', default=None, help='The file storing the secondary structure.')
    parser.add_argument('-t', '--time', default=None, help='The MD simulation time in nanosecond, which will be set automatically if not specified by user.')
    parser.add_argument('-n', '--npred', default=10, help='The number of predicted 3D structures.')
    parser.add_argument('-c', '--constraint_file', default=None, help='The constraint file.')
    parser.add_argument('-g', '--gpu_index', default='0', help='The gpu device index to be used.')
    args = parser.parse_args()
    print(args, flush=True)

    rna_name = args.RNA
    in2D = args.secondary_structure
    simulation_time = args.time
    gpu_index = str(args.gpu_index)
    constraint_file = args.constraint_file
    npred = args.npred

    if in2D is None:
        raise ValueError(f"Please provide the 2D structure file!")

    if simulation_time is not None:
        try:
            simulation_time = float(simulation_time)
        except:
            raise ValueError(f"time should be an integer or floating number. Now it is {simulation_time}.")
        if simulation_time < 0.0:
            raise ValueError(f"time should be a positive integer or floating number. Now it is {simulation_time}.")

    if npred:
        try:
            npred = int(npred)
        except:
            raise ValueError(f"npred (the number of predicted 3D structures) should be an integer. Now it is {npred}.")
        if npred < 1:
            raise ValueError(f"npred (the number of predicted 3D structures) should be a positive integer. Now it is {npred}.")

    if not os.path.exists(in2D):
        raise ValueError(f"Secondary structure file '{in2D}' does not exist!")

    if constraint_file is not None:
        if not os.path.exists(constraint_file):
            raise ValueError(f"Constraint file '{constraint_file}' does not exist!")
   
    # get sequence and motif information from the given 2D structure.
    seq, dict_motifs, dict_resid_chain = get_motifs_from_2D(in2D)

    if simulation_time is None:
        if len(dict_motifs["4way_loops"]) > 1 or len(dict_motifs["3way_loops"]) > 1:
            simulation_time = 600.0
        elif len(dict_motifs["4way_loops"]) == 1 or len(dict_motifs["3way_loops"]) == 1:
            simulation_time = 400.0
        else:
            simulation_time = 200.0

    print(f"\n****** Input ******")
    print(f"RNA name: {rna_name}")
    print(f"Secondary structure file: {in2D}")
    print(f"Constraint file: {constraint_file}")
    print(f"Simulation time: {simulation_time} nanoseconds")
    print(f"Number of predicted structures: {npred}")
    print(f"GPU index: {gpu_index}", flush=True)

    time_start = time.time()

    outfolder = f"{rna_name}"  # the directory storing the simulation trajectory and the predicted structures
    init3D = f"{outfolder}/init.pdb" # the generated initial structure
    outtrj = f"{outfolder}/trj.pdb" # the simulation trajectory

    if not os.path.exists(outfolder):
        print(f"\nMake the working directory '{outfolder}'", flush=True)
        os.makedirs(outfolder)

    # get the JAR3D energy parameters for the hairpin and internal loops in the given 2D structure.
    dict_jar3d_energy_paras, list_2way_found_no_jar3d, list_bulge_nts_in_jar3d = get_jar3d_energy_paras(RNAJP_HOME,seq,dict_motifs,outfolder)

    # generate the initial structure if it does not exist.
    print("\n******************************************")
    print(f"Generating the initial 3D structure ...", flush=True)
    if not os.path.exists(init3D):
        fold_init_structure(RNAJP_HOME,seq,dict_motifs,dict_resid_chain,gpu_index,outfolder)
    print(f"The initial 3D structure has been generated: {init3D}", flush=True)

    # run the MD simulations 
    print("\n******************************************")
    print(f"Running the MD simulations ...")
    runMD(RNAJP_HOME,seq,dict_motifs,dict_resid_chain,constraint_file,dict_jar3d_energy_paras,list_2way_found_no_jar3d,list_bulge_nts_in_jar3d,init3D,gpu_index,simulation_time,outfolder,outtrj)
    print(f"The MD simulations have been finished and the trajectory is stored in {outtrj}", flush=True)
   
    # postprocess the simulation trajectory and output the predicted strctures after energy minimization
    print("\n******************************************")
    print(f"Postprocessing the simulation trajectory ...", flush=True)
    select_structures_from_trajectory(RNAJP_HOME,rna_name,outfolder,dict_motifs,dict_resid_chain,dict_jar3d_energy_paras,list_2way_found_no_jar3d,outtrj,npred,gpu_index)

    print("\n******************************************")
    time_end = time.time()
    time_total = time_end - time_start
    hrs = int(time_total/3600.)
    mins = int((time_total - hrs*3600.)/60.)
    secs = int(time_total - hrs*3600. - mins*60)
    if hrs == 0:
        print(f"The prediction job for RNA '{rna_name}' has been done! It took {mins} min {secs} sec.", flush=True)
    else:
        print(f"The prediction job for RNA '{rna_name}' has been done! It took {hrs} hr {mins} min {secs} sec.", flush=True)
