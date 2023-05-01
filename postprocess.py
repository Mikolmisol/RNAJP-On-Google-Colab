from __future__ import print_function
from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit
import numpy as np
import os
from add_junction_topology_force import add_interhelical_force_in_junctions, add_coaxial_force_in_2way_junctions
from add_local_force import add_torsion_and_angle_force_to_system
from add_non_local_force import add_base_pairing_stacking_force_to_system, add_force_between_hairpin_internal_loops_to_system, add_force_within_junction_loops_to_system, add_jar3d_force_to_system
import MDAnalysis
from MDAnalysis.analysis import diffusionmap, align
from recover_all_atom_structure import convert_cg_to_aa


def calc_split_energy(RNAJP_HOME,dict_motifs,dict_resid_chain,dict_jar3d_energy_paras,list_2way_found_no_jar3d,trj_file,gpu_index,outfolder,outenergyfile):
    app.topology.Topology().loadBondDefinitions(f'{RNAJP_HOME}/myresidues.xml')
    forcefield = app.ForceField(f'{RNAJP_HOME}/myRNA.xml')

    pdb = app.PDBFile(trj_file)
    #print (pdb.topology)
    #print ("Creating system...")
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=0.6*unit.nanometers, ewaldErrorTolerance=0.0005, ignoreExternalBonds=True)

    #ForceGroup 1,2,3
    system, list_torsion_atoms_index_jar3d, list_angle_atoms_index_jar3d, jar3d_dis_force_index, list_jar3d_dis_bonds, jar3d_angle_force_index, list_jar3d_angle_bonds, jar3d_torsion_force_index, list_jar3d_torsion_bonds = add_jar3d_force_to_system(pdb.topology,system,dict_motifs,dict_resid_chain,dict_jar3d_energy_paras)

    #ForceGroup 4,5
    wt_torsion = 1.0
    wt_angle = 1.0
    system = add_torsion_and_angle_force_to_system(RNAJP_HOME, pdb.topology, system, dict_motifs, dict_resid_chain, wt_torsion, wt_angle, list_torsion_atoms_index_jar3d, list_angle_atoms_index_jar3d)

    #ForceGroup 25,26,7,8
    dict_bp_stk_strength = {'bp_AA': 1.0, 'bp_AG': 1.5, 'bp_AC':1.0, 'bp_AU': 2.0, 'bp_GG': 0.5, 'bp_GC': 3.0, 'bp_GU': 2.0, 'bp_CC': 0.5, 'bp_CU':0.5, 'bp_UU':0.5, 'stk_AA': 1.0, 'stk_AG': 1.0, 'stk_AC': 0.8, 'stk_AU': 0.8, 'stk_GG': 1.0, 'stk_GC': 0.8, 'stk_GU': 0.8, 'stk_CC': 0.2, 'stk_CU': 0.2, 'stk_UU': 0.2, 'stk_aG':1.0, 'stk_aC':0.4, 'stk_aU':0.4, 'stk_gC':0.4, 'stk_gU':0.4, 'stk_cU':0.2, 'stk_aa':1.5, 'stk_ag':1.5, 'stk_ac':1.2, 'stk_au':1.2, 'stk_gg':1.5, 'stk_gc':1.2, 'stk_gu':1.2, 'stk_cc':0.3, 'stk_cu':0.3, 'stk_uu':0.3}
    for key in dict_bp_stk_strength:
        dict_bp_stk_strength[key] *= 1.0
    list_bulge_nts_in_jar3d = []
    system = add_base_pairing_stacking_force_to_system(RNAJP_HOME,pdb.topology,system,dict_motifs,dict_resid_chain,dict_bp_stk_strength,list_bulge_nts_in_jar3d,split=True)
    system = add_force_between_hairpin_internal_loops_to_system(pdb.topology,system,dict_motifs,dict_resid_chain)
    system = add_force_within_junction_loops_to_system(pdb.topology,system,dict_motifs,dict_resid_chain)

    #ForceGroup 9,10
    list_junctions_3way = dict_motifs["3way_loops"]
    list_junctions_4way = dict_motifs["4way_loops"]
    list_junctions = list_junctions_3way + list_junctions_4way
    interhelical_paras_for_junctions = None
    system, index_topo_force, list_topo_force_bonds_in_junctions, index_helical_torsion_force, list_helical_torsion_bonds_in_junctions = add_interhelical_force_in_junctions(pdb.topology,system,dict_resid_chain,list_junctions,interhelical_paras_for_junctions)

    #ForceGroup 11
    list_single_loops = dict_motifs["single_loops_non_PK"]
    if list_2way_found_no_jar3d or list_single_loops:
        for single_loop in list_single_loops:
            pseudo_2way = [single_loop[0],single_loop[0]]
            list_2way_found_no_jar3d.append(pseudo_2way)
        #print("Adding coaxial stacking interaction within 2way or single loops")
        system = add_coaxial_force_in_2way_junctions(pdb.topology,system,dict_resid_chain,list_2way_found_no_jar3d)

    Temp = 300
    integrator = mm.LangevinIntegrator(Temp*unit.kelvin, 10.0/unit.picoseconds, 1.0*unit.femtoseconds)

    platform = mm.Platform.getPlatformByName('CUDA')
    properties = {'CudaPrecision': 'mixed', 'CudaDeviceIndex': gpu_index}
    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)

    simulation.context.setParameter("kbpstk_global",1.0)
    simulation.context.setParameter("klooploop",50)
    simulation.context.setParameter("dlooploop_lower",0)
    simulation.context.setParameter("dlooploop_ave",0)
    simulation.context.setParameter("dlooploop_upper",3)
    simulation.context.setParameter("khphp",0)

    simulation.context.setParameter("kcompactjunc",50)
    simulation.context.setParameter("dcompactjunc",1.0)
    simulation.context.setParameter("kuncompactjunc",0)
    simulation.context.setParameter("koutward",0)

    simulation.context.setParameter("khelixhelix_global",100)

    list_energy = []
    for i in range(pdb.getNumFrames()):
        simulation.context.setPositions(pdb.getPositions(frame=i))
        state = simulation.context.getState(getPositions=True,getForces=True,getEnergy=True)
        dict_energy = {}
        for force in system.getForces():
            group = force.getForceGroup()
            dict_energy[group] = simulation.context.getState(getEnergy=True, groups={group}).getPotentialEnergy()
        list_energy.append(dict_energy)

    f = open(outenergyfile,"w")

    list_key = []
    for key in list_energy[0]:
        list_key.append(str(key))
    f.write(" ".join(list_key)+"\n")

    for dict_energy in list_energy:
        list_energy = []
        for key in dict_energy:
            energy = dict_energy[key].value_in_unit(unit.kilojoule/unit.mole)
            list_energy.append(str(energy))
        list_energy = " ".join(list_energy)
        f.write(list_energy+"\n")
    f.close()


def calc_raw_energy(RNAJP_HOME,wkdir,dict_motifs,dict_resid_chain,dict_jar3d_energy_paras, list_2way_found_no_jar3d,trj_file,gpu_index):
    if not os.path.exists(wkdir):
        raise ValueError(f"Working directory {wkdir} does not exist!")
    if not os.path.exists(trj_file):
        raise ValueError(f"Trajectory file {trj_file} does not exist!")
    raw_energy_file = f"{wkdir}/raw_energy.txt"
    calc_split_energy(RNAJP_HOME,dict_motifs,dict_resid_chain,dict_jar3d_energy_paras,list_2way_found_no_jar3d,trj_file,gpu_index,wkdir,raw_energy_file)
    return raw_energy_file


def calc_energy_range(pot_array):
    sorted_pot_array = np.sort(pot_array)
    num = int(len(pot_array)*0.5)
    num1 = int(num*0.1)
    ave1 = np.mean(sorted_pot_array[0:num1])
    ave2 = np.mean(sorted_pot_array[num-num1:num])
    pr = ave2 - ave1
    if pr < 0:
        print ("Wrong potential range, ",pr)
        exit()
    return pr


def calc_reweighted_energy(raw_energy_file,wt_between_terms,reweighted_energy_file):
    pot = np.loadtxt(raw_energy_file,skiprows=1,usecols=list(range(4,10)))

    list_pot_range = []
    for i in range(len(pot[0])):
        pot_range = calc_energy_range(pot[:,i])
        if pot_range == 0.0:
            pot_range = 1.0
        list_pot_range.append(pot_range)
    max_pr = max(list_pot_range)
    wt_to_level_terms = [max_pr/pr for pr in list_pot_range]
    #print ("wt to level terms")
    #print (wt_to_level_terms)
    
    for i in range(len(wt_between_terms)):
        pot[:,i] *= wt_to_level_terms[i]*wt_between_terms[i]

    sum_pot = np.sum(pot,axis=1)
    np.savetxt(reweighted_energy_file, sum_pot)
    return pot, sum_pot


def extract_low_energy_trj(trj_file,nframe_per_round,sum_pot,low_pot_trj_file,low_pot_trj_pot_file):
    trj = app.PDBFile(trj_file)
    sorted_sum_pot_idx = np.argsort(sum_pot)
    low_pot_idx = []
    for idx in sorted_sum_pot_idx:
        tag = True
        for idx0 in low_pot_idx:
            if int(idx0/nframe_per_round) == int(idx/nframe_per_round):
                tag = False
                break
        if tag:
            low_pot_idx.append(idx)
    low_pot_idx = np.sort(low_pot_idx)

    with open(low_pot_trj_file,"w") as f:
        for nmodel,idx in enumerate(low_pot_idx):
            #print (nmodel+1, idx)
            app.PDBFile.writeModel(trj.topology,trj.getPositions(frame=idx),f,modelIndex=nmodel+1)

    low_pot_trj_pot = sum_pot[low_pot_idx]
    np.savetxt(low_pot_trj_pot_file,low_pot_trj_pot)


def calc_pairwise_rmsd_in_low_pot_trj(low_pot_trj_file,pairwise_rmsd_file):
    trj = MDAnalysis.Universe(low_pot_trj_file,dt=1)
    aligner = align.AlignTraj(trj, trj, select='all',in_memory=True).run()
    matrix = diffusionmap.DistanceMatrix(trj, select='all').run()
    pairwise_rmsd = matrix.results.dist_matrix
    np.savetxt(pairwise_rmsd_file, pairwise_rmsd, fmt='%.2f')


def cluster_structures(pairwise_rmsd_file,low_pot_trj_energy_file,npred):
    pairwise_rmsd = np.loadtxt(pairwise_rmsd_file)
    low_pot = np.loadtxt(low_pot_trj_energy_file)
    sorted_low_pot_idx = np.argsort(low_pot)

    final_pred_idx = [sorted_low_pot_idx[0]]
    final_pred_idx2 = [[sorted_low_pot_idx[0],low_pot[sorted_low_pot_idx[0]]]]

    fraction = 0.3
    num = int(len(low_pot) * fraction)
    if num < npred:
        npred = num

    #print(sorted_low_pot_idx)
    #print(f"num: {num}",flush=True)
    while len(final_pred_idx) < npred:
        maxdissum = -1.
        best_pred = None
        for i in sorted_low_pot_idx[0:num]:
            if i in final_pred_idx:
                continue
            dissum = np.sum(pairwise_rmsd[final_pred_idx,i])
            if dissum > maxdissum:
                maxdissum = dissum
                best_pred = i
        if best_pred is not None:
            final_pred_idx.append(best_pred)
            similar_pred_idx = np.argwhere(pairwise_rmsd[best_pred] < 5.0).reshape(-1)
            lowest_pot = 100000. 
            for idx in similar_pred_idx:
                if low_pot[idx] < lowest_pot:
                    similar_and_lowest_pot_pred_idx = idx
                    lowest_pot = low_pot[idx]
            final_pred_idx2.append([similar_and_lowest_pot_pred_idx,low_pot[similar_and_lowest_pot_pred_idx]])
        if len(final_pred_idx) == npred:
            break
        #print(f"len(final_pred_idx): {len(final_pred_idx)}",flush=True)
    final_pred_idx2.sort(key=lambda x:x[1])
    return final_pred_idx, final_pred_idx2


def select_structures_from_trajectory(RNAJP_HOME,rna_name,wkdir,dict_motifs,dict_resid_chain,dict_jar3d_energy_paras,list_2way_found_no_jar3d,trj_file,npred,gpu_index):
    if not os.path.exists(trj_file):
        raise ValueError(f"Failed in finding the trajectory file '{trj_file}'")

    print(f"\nCalculating the energy terms for the structures in the trajectory file '{trj_file}' ...", flush=True)
    raw_energy_file = calc_raw_energy(RNAJP_HOME,wkdir,dict_motifs,dict_resid_chain,dict_jar3d_energy_paras,list_2way_found_no_jar3d,trj_file,gpu_index)
    wt_between_terms = [0.75,0.75,1,1.5,1,1]
    reweighted_energy_file = f"{wkdir}/reweighted_energy.txt"
    print(f"Reweighting the energy terms ...", flush=True)
    reweighted_split_energy, reweighted_sum_energy = calc_reweighted_energy(raw_energy_file,wt_between_terms,reweighted_energy_file)

    print(f"Extracting the low-energy structures ...", flush=True)
    nframe_per_round = 25
    low_pot_trj_file = f"{wkdir}/low_pot_trj.pdb"
    low_pot_trj_energy_file = f"{wkdir}/low_pot_trj_energy.txt"
    extract_low_energy_trj(trj_file,nframe_per_round,reweighted_sum_energy,low_pot_trj_file,low_pot_trj_energy_file)
    
    print(f"Clustering the low-energy structures ...", flush=True)
    pairwise_rmsd_file = f"{wkdir}/low_pot_trj_pairwise_rmsd.txt"
    calc_pairwise_rmsd_in_low_pot_trj(low_pot_trj_file,pairwise_rmsd_file)
    final_pred_idx, final_pred_idx2 = cluster_structures(pairwise_rmsd_file,low_pot_trj_energy_file,npred)

    trj = app.PDBFile(low_pot_trj_file)
    for nmodel,idx in enumerate(final_pred_idx):
        predicted_cg_structure_file = f"{wkdir}/predicted_{rna_name}_cg-{nmodel+1}.pdb"
        with open(predicted_cg_structure_file,"w") as f:
            app.PDBFile.writeModel(trj.topology,trj.getPositions(frame=idx),f,modelIndex=nmodel+1)
        predicted_aa_structure_file = f"{wkdir}/predicted_{rna_name}-{nmodel+1}.pdb"
        print(f"Converting CG to AA model #{nmodel+1}",flush=True)
        convert_cg_to_aa(RNAJP_HOME,predicted_cg_structure_file,predicted_aa_structure_file)
        print(f"Energy minimizing model #{nmodel+1}", flush=True)
        em_predicted_aa_structure_file = f"{wkdir}/em_predicted_{rna_name}-{nmodel+1}.pdb"
        cmd = f"{RNAJP_HOME}/QRNAS/QRNA -i {predicted_aa_structure_file} -o {em_predicted_aa_structure_file} -c {RNAJP_HOME}/QRNAS/configfile.txt > {wkdir}/em.log 2>&1"
        os.system(cmd)
        cmd = f"sed -i 's/RC3/  C/g' {em_predicted_aa_structure_file}"
        os.system(cmd)
        cmd = f"sed -i 's/RG3/  G/g' {em_predicted_aa_structure_file}"
        os.system(cmd)
        cmd = f"sed -i 's/RA3/  A/g' {em_predicted_aa_structure_file}"
        os.system(cmd)
        cmd = f"sed -i 's/RU3/  U/g' {em_predicted_aa_structure_file}"
        os.system(cmd)
        print(f"Predicted structure #{nmodel+1}: {em_predicted_aa_structure_file}",flush=True)
