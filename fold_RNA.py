from __future__ import print_function
import os
import random
import time
from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit
from add_constraint_force import fix_helices, add_base_pairing_constraint_force_to_system, add_constraint_force_to_system
from add_junction_topology_force import add_interhelical_force_in_junctions, add_coaxial_force_in_2way_junctions, add_stacking_force_between_PK_and_nonPK_helices
from add_local_force import add_torsion_and_angle_force_to_system
from add_non_local_force import add_base_pairing_stacking_force_to_system, add_force_between_hairpin_internal_loops_to_system, add_force_within_junction_loops_to_system, add_jar3d_force_to_system, add_possible_bp_attraction_force_to_system


def get_junction_topology(dict_motifs,junction):
    list_helices_in_junc = []
    list_loop_len_in_junc = []
    for i,loop in enumerate(junction):
        ib, ie = loop
        list_loop_len_in_junc.append(ie-ib-1)
        for helix in dict_motifs["helix"]:
            h1, h4 = helix[0]
            h2, h3 = helix[1]
            if ib in [h1,h2,h3,h4]:
                list_helices_in_junc.append(helix)
                break

    list_loops_connected_to_helices_in_junc = []
    for i, helix in enumerate(list_helices_in_junc):
        h1, h4 = helix[0]
        h2, h3 = helix[1]
        connected_loops = []
        for key in dict_motifs.keys():
            if key == "helix":
                continue
            if key == "PK_helix":
                continue
            if "non_PK" in key:
                continue
            motifs = dict_motifs[key]
            for motif in motifs:
                if motif == junction:
                    continue
                for loop in motif:
                    ib, ie = loop
                    if ib in [h1,h2,h3,h4] or ie in [h1,h2,h3,h4]:
                        connected_loops.append(motif)
                        break
        list_loops_connected_to_helices_in_junc.append(connected_loops)

    junc_helices_connected_by_PK_helix = []
    for i in range(len(list_loops_connected_to_helices_in_junc)-1):
        connected_loops1 = list_loops_connected_to_helices_in_junc[i]
        for j in range(i+1,len(list_loops_connected_to_helices_in_junc)):
            bool_connected_by_PK_helix = False
            connected_loops2 = list_loops_connected_to_helices_in_junc[j]
            for loops1 in connected_loops1:
                for loop1 in loops1:
                    ib1, ie1 = loop1
                    for loops2 in connected_loops2:
                        for loop2 in loops2:
                            ib2, ie2 = loop2
                            for PK_helix in dict_motifs["PK_helix"]:
                                h1, h4 = PK_helix[0]
                                h2, h3 = PK_helix[1]
                                if h1 >= ib1 and h2 <= ie1 and h3 >= ib2 and h4 <= ie2:
                                    bool_connected_by_PK_helix = True
                                    break
                                if h1 >= ib2 and h2 <= ie2 and h3 >= ib1 and h4 <= ie1:
                                    bool_connected_by_PK_helix = True
                                    break
                            if bool_connected_by_PK_helix:
                                break
                        if bool_connected_by_PK_helix:
                            break
                    if bool_connected_by_PK_helix:
                        break
                if bool_connected_by_PK_helix:
                    break
            if bool_connected_by_PK_helix:
                junc_helices_connected_by_PK_helix.append(i)
                junc_helices_connected_by_PK_helix.append(j)
                break
        if bool_connected_by_PK_helix:
            break

    selected_stacked_helices_idx = None
    selected_topology = None
    if junc_helices_connected_by_PK_helix:
        idx1, idx2 = junc_helices_connected_by_PK_helix
        if len(junction) == 3:
            if idx1 == 0 and idx2 == 1:
                weights = [1./(list_loop_len_in_junc[1]+1),1./(list_loop_len_in_junc[2]+1)]
                rand_num = random.choices([1,2],weights=weights)[0]
                if rand_num == 1:
                    selected_stacked_helices_idx = [1]
                    if random.random() < 0.8:
                        selected_topology = 0
                    else:
                        selected_topology = 2
                else:
                    selected_stacked_helices_idx = [2]
                    if random.random() < 0.8:
                        selected_topology = 1
                    else:
                        selected_topology = 2

            elif idx1 == 0 and idx2 == 2:
                weights = [1./(list_loop_len_in_junc[0]+1),1./(list_loop_len_in_junc[1]+1)]
                rand_num = random.choices([0,1],weights=weights)[0]
                if rand_num == 0:
                    selected_stacked_helices_idx = [0]
                    if random.random() < 0.8:
                        selected_topology = 0
                    else:
                        selected_topology = 2
                else:
                    selected_stacked_helices_idx = [1]
                    if random.random() < 0.8:
                        selected_topology = 1
                    else:
                        selected_topology = 2

            elif idx1 == 1 and idx2 == 2:
                weights = [1./(list_loop_len_in_junc[0]+1),1./(list_loop_len_in_junc[2]+1)]
                rand_num = random.choices([0,2],weights=weights)[0]
                if rand_num == 0:
                    selected_stacked_helices_idx = [0]
                    if random.random() < 0.8:
                        selected_topology = 1
                    else:
                        selected_topology = 2
                else:
                    selected_stacked_helices_idx = [2]
                    if random.random() < 0.8:
                        selected_topology = 0
                    else:
                        selected_topology = 2

        if len(junction) == 4:
            if (idx1 == 0 and idx2 == 1) or (idx1 == 2 and idx2 == 3):
                selected_stacked_helices_idx = [1,3]
                if random.random() < 0.8:
                    selected_topology = 1
                else:
                    selected_topology = 2
            elif (idx1 == 1 and idx2 == 2) or (idx1 == 0 and idx2 == 3):
                selected_stacked_helices_idx = [0,2]
                if random.random() < 0.8:
                    selected_topology = 1
                else:
                    selected_topology = 2
            elif (idx1 == 0 and idx2 == 2) or (idx1 == 1 and idx2 == 3):
                weights = [1./(list_loop_len_in_junc[0]+list_loop_len_in_junc[2]+1),1./(list_loop_len_in_junc[1]+list_loop_len_in_junc[3]+1)]
                rand_num = random.choices([0,1],weights=weights)[0]
                if rand_num == 0:
                    selected_stacked_helices_idx = [0,2]
                else:
                    selected_stacked_helices_idx = [1,3]
                if random.random() < 0.8:
                    selected_topology = 0
                else:
                    selected_topology = 2
    return selected_stacked_helices_idx, selected_topology


def randomly_select_stacked_helices(dict_motifs, simulation, index_topo_force, list_topo_force_bonds_in_junctions, index_helical_torsion_force, list_helical_torsion_bonds_in_junctions, list_junctions):
    topo_force = simulation.system.getForce(index_topo_force)
    helical_torsion_force = simulation.system.getForce(index_helical_torsion_force)

    for i in range(len(list_junctions)):
        junction = list_junctions[i]
        if len(junction) != len(list_topo_force_bonds_in_junctions[i]):
            print ("Junction type is not equal!")
            print (len(junction), len(list_topo_force_bonds_in_junctions))
            exit()
        junc_looplen = [v[1]-v[0]-1 for v in junction]
        if len(junction) == 3:
            stack_prob = []
            for looplen in junc_looplen:
                if looplen >= 6:
                    if min(junc_looplen) >= 4:
                        stack_prob.append(1.0)
                    else:
                        stack_prob.append(0.0)
                else:
                    stack_prob.append(1.0)
            stack_prob = [v/sum(stack_prob) for v in stack_prob]
            cdf = [sum(stack_prob[0:n]) for n in range(1,len(stack_prob)+1)]
            rn = random.random()
            selected_stacked_helices_idx = []
            for j in range(len(cdf)):
                if rn < cdf[j]:
                    selected_stacked_helices_idx.append(j)
                    break
            j = selected_stacked_helices_idx[0]
            j1 = j+1
            j2 = j+2
            if j1 >= len(junction):
                j1 -= len(junction)
            if j2 >= len(junction):
                j2 -= len(junction)
            looplen_j1 = junc_looplen[j1]
            looplen_j2 = junc_looplen[j2]
            if looplen_j1 - looplen_j2 >= 4:
                selected_topology = random.randint(1,2)
            elif looplen_j2 - looplen_j1 >= 4:
                selected_topology = random.choice([0,2])
            else:
                selected_topology = random.randint(0,2)
        elif len(junction) == 4:
            minloop1_3 = min(junc_looplen[0],junc_looplen[2])
            minloop2_4 = min(junc_looplen[1],junc_looplen[3])
            if minloop1_3 - minloop2_4 > 4:
                cdf = [0.0,0.5,0.5,1.0]
            elif minloop2_4 - minloop1_3 > 4:
                cdf = [0.5,0.5,1.0,1.0]
            else:
                cdf = [0.25,0.5,0.75,1.0]
            rn = random.random()
            selected_stacked_helices_idx = []
            for j in range(len(cdf)):
                if rn < cdf[j]:
                    selected_stacked_helices_idx.append(j)
                    break
            j2 = j + 2
            if j2 >= len(junction):
                j2 -= len(junction)
            selected_stacked_helices_idx.append(j2)
            selected_topology = random.randint(0,2)
        else:
            print ("Cannot deal with >4-way junctions")
            exit()

        new_selected_stacked_helices_idx, new_selected_topology = get_junction_topology(dict_motifs,junction)
        if new_selected_stacked_helices_idx is not None:
            selected_stacked_helices_idx = new_selected_stacked_helices_idx
            selected_topology = new_selected_topology
       
        #print ("*** selected helix-pair",selected_stacked_helices_idx,flush=True)
        #print ("*** selected junction topology",selected_topology,flush=True) # 0 for parallel, 1 for anti-parallel, 2 for vertical

        selected_paral_antiparal_vertical_helices = []
        for v in selected_stacked_helices_idx:
            if v+1 >= len(junction):
                selected_paral_antiparal_vertical_helices.append(v+1-len(junction))
            else:
                selected_paral_antiparal_vertical_helices.append(v+1)

        list_topo_force_bonds_in_one_junction = list_topo_force_bonds_in_junctions[i]
        for j in range(len(list_topo_force_bonds_in_one_junction)):
            bond_index, atoms_index = list_topo_force_bonds_in_one_junction[j]
            looplen = junction[j][1] - junction[j][0]
            if j in selected_stacked_helices_idx:
                topo_force.setBondParameters(bond_index,atoms_index,[1.0,0.0,0.0,0.0,looplen])
            elif j in selected_paral_antiparal_vertical_helices:
                if selected_topology == 0:
                    topo_force.setBondParameters(bond_index,atoms_index,[0.0,1.0,0.0,0.0,looplen])
                elif selected_topology == 1:
                    topo_force.setBondParameters(bond_index,atoms_index,[0.0,0.0,1.0,0.0,looplen])
                elif selected_topology == 2:
                    topo_force.setBondParameters(bond_index,atoms_index,[0.0,0.0,0.0,1.0,looplen])
                else:
                    print ("Wrong junction topology.")
                    exit()
            else:
                topo_force.setBondParameters(bond_index,atoms_index,[0.0,0.0,0.0,0.0,looplen])
            
        list_helical_torsion_bonds_in_one_junction = list_helical_torsion_bonds_in_junctions[i]
        for j in range(len(list_helical_torsion_bonds_in_one_junction)):
            list_bonds_info = list_helical_torsion_bonds_in_one_junction[j]
            for bond_info in list_bonds_info:
                bond_index, atoms_index, paras = bond_info
                if j in selected_stacked_helices_idx:
                    helical_torsion_force.setBondParameters(bond_index,atoms_index,[100.0,paras[1],paras[2],paras[3]])
                else:
                    helical_torsion_force.setBondParameters(bond_index,atoms_index,[5.0,paras[1],paras[2],paras[3]])

    topo_force.updateParametersInContext(simulation.context)
    helical_torsion_force.updateParametersInContext(simulation.context)
    return simulation
        
        
def randomly_select_stacked_PK_and_nonPK_helices(simulation, index_PK_nonPK_stacking_force, index_PK_nonPK_vertical_force, list_bonds_for_PK_helices):
    stacking_force = simulation.system.getForce(index_PK_nonPK_stacking_force)
    vertical_force = simulation.system.getForce(index_PK_nonPK_vertical_force)

    for i in range(len(list_bonds_for_PK_helices)):
        bonds_for_one_PK_helix = list_bonds_for_PK_helices[i]
        selected_prob = bonds_for_one_PK_helix[-2]
        loop_len = bonds_for_one_PK_helix[-1]
        assert len(selected_prob) == len(bonds_for_one_PK_helix)-2
        assert len(loop_len) == len(bonds_for_one_PK_helix)-2
        if random.random() < 0.1:
            num_selection = 0
        else:
            num_selection = random.randint(1,len(bonds_for_one_PK_helix)-2)
        if num_selection == 0:
            selected_idx = []
        elif num_selection == 1:
            selected_idx = random.choices(list(range(len(bonds_for_one_PK_helix)-2)),weights=selected_prob)
        else:
            selected_idx = random.sample(list(range(len(bonds_for_one_PK_helix)-2)),k=num_selection)
        #print(f"For {i}-th PK helix, select {selected_idx}-th stacking.",flush=True)
        for j in range(len(bonds_for_one_PK_helix)-2):
            bond_info = bonds_for_one_PK_helix[j][-1]
            if not bond_info:
                kstack = 1.0
            else:
                bond_index, atoms_index = bond_info
                if random.random() < 0.2:
                    kstack = 0.0
                    vertical_force.setBondParameters(bond_index,atoms_index,[1000.0])
                else:
                    kstack = 1.0
                    vertical_force.setBondParameters(bond_index,atoms_index,[0.0])

            for k in range(len(bonds_for_one_PK_helix[j])-1):
                bond_index, atoms_index, paras = bonds_for_one_PK_helix[j][k]
                if j in selected_idx:
                    if loop_len[j] <= 5:
                        strength = kstack*500*random.uniform(0.1,1.0)
                        stacking_force.setBondParameters(bond_index,atoms_index,[strength]+paras[1:])
                        #print(f"Setting force-{bond_index}, atoms-{atoms_index},{[strength]+paras[1:]}",flush=True)
                    else:
                        strength = kstack*50*random.uniform(0.1,1.0)
                        stacking_force.setBondParameters(bond_index,atoms_index,[strength]+paras[1:])
                        #print(f"Setting force-{bond_index}, atoms-{atoms_index},{[strength]+paras[1:]}",flush=True)
                else:
                    strength = kstack*5*random.uniform(0.0,1.0)
                    stacking_force.setBondParameters(bond_index,atoms_index,[strength]+paras[1:])
                    #print(f"Setting force-{bond_index}, atoms-{atoms_index},{[strength]+paras[1:]}",flush=True)
            
    stacking_force.updateParametersInContext(simulation.context)
    vertical_force.updateParametersInContext(simulation.context)
    return simulation
        
        
def randomly_select_GC_AU_bp(simulation, index_bp_close_force, list_bp_close_bonds_GC, list_bp_close_bonds_AU):
    bp_close_force = simulation.system.getForce(index_bp_close_force)
    num_GC = len(list_bp_close_bonds_GC)
    num_AU = len(list_bp_close_bonds_AU)
    if num_GC > 0:
        rn = random.randint(-1,num_GC-1)
        for i in range(num_GC):
            bond_index, atoms_index = list_bp_close_bonds_GC[i]
            if i == rn:
                bp_close_force.setBondParameters(bond_index,atoms_index,[1.0])
            else:
                bp_close_force.setBondParameters(bond_index,atoms_index,[0.0])
    if num_AU > 0:
        rn = random.randint(-1,num_AU-1)
        for i in range(num_AU):
            bond_index, atoms_index = list_bp_close_bonds_AU[i]
            if i == rn:
                bp_close_force.setBondParameters(bond_index,atoms_index,[1.0])
            else:
                bp_close_force.setBondParameters(bond_index,atoms_index,[0.0])
    bp_close_force.updateParametersInContext(simulation.context)
    return simulation


def runMD(RNAJP_HOME,seq,dict_motifs,dict_resid_chain,constraint_file,dict_jar3d_energy_paras,list_2way_found_no_jar3d,list_bulge_nts_in_jar3d,init3D,gpu_index,simulation_time,outfolder,outtrj):
    app.topology.Topology().loadBondDefinitions(f'{RNAJP_HOME}/myresidues.xml')
    forcefield = app.ForceField(f'{RNAJP_HOME}/myRNA.xml')

    pdb = app.PDBFile(init3D)
    #print (pdb.topology)
    #print ("Creating system ...", flush=True)
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=0.6*unit.nanometers, ewaldErrorTolerance=0.0005, ignoreExternalBonds=True)

    #ForceGroup 1,2,3
    system, list_torsion_atoms_index_jar3d, list_angle_atoms_index_jar3d, jar3d_dis_force_index, list_jar3d_dis_bonds, jar3d_angle_force_index, list_jar3d_angle_bonds, jar3d_torsion_force_index, list_jar3d_torsion_bonds = add_jar3d_force_to_system(pdb.topology,system,dict_motifs,dict_resid_chain,dict_jar3d_energy_paras)

    #ForceGroup 4,5
    wt_torsion = 1.0
    wt_angle = 1.0
    system = add_torsion_and_angle_force_to_system(RNAJP_HOME, pdb.topology, system, dict_motifs, dict_resid_chain, wt_torsion, wt_angle, list_torsion_atoms_index_jar3d, list_angle_atoms_index_jar3d)

    #ForceGroup 6,7,8
    dict_bp_stk_strength = {'bp_AA': 1.0, 'bp_AG': 1.5, 'bp_AC':1.0, 'bp_AU': 2.0, 'bp_GG': 0.5, 'bp_GC': 3.0, 'bp_GU': 2.0, 'bp_CC': 0.5, 'bp_CU':0.5, 'bp_UU':0.5, 'stk_AA': 1.0, 'stk_AG': 1.0, 'stk_AC': 0.8, 'stk_AU': 0.8, 'stk_GG': 1.0, 'stk_GC': 0.8, 'stk_GU': 0.8, 'stk_CC': 0.2, 'stk_CU': 0.2, 'stk_UU': 0.2, 'stk_aG':1.0, 'stk_aC':0.4, 'stk_aU':0.4, 'stk_gC':0.4, 'stk_gU':0.4, 'stk_cU':0.2, 'stk_aa':1.5, 'stk_ag':1.5, 'stk_ac':1.2, 'stk_au':1.2, 'stk_gg':1.5, 'stk_gc':1.2, 'stk_gu':1.2, 'stk_cc':0.3, 'stk_cu':0.3, 'stk_uu':0.3}
    for key in dict_bp_stk_strength:
        dict_bp_stk_strength[key] *= 0.5
    #print (dict_bp_stk_strength)
    system = add_base_pairing_stacking_force_to_system(RNAJP_HOME,pdb.topology,system,dict_motifs,dict_resid_chain,dict_bp_stk_strength,list_bulge_nts_in_jar3d)
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
        #print ("Adding coaxial stacking interaction within 2way or single loops")
        system = add_coaxial_force_in_2way_junctions(pdb.topology,system,dict_resid_chain,list_2way_found_no_jar3d)

    #ForceGroup 12
    system, index_bp_close_force, list_bp_close_bonds_GC, list_bp_close_bonds_AU = add_possible_bp_attraction_force_to_system(pdb.topology,system,dict_motifs,dict_resid_chain,seq)

    #ForceGroup 13
    system = fix_helices(pdb.topology, system, dict_resid_chain, pdb.positions, dict_motifs["helix"],dict_motifs["PK_helix"])

    system = add_constraint_force_to_system(RNAJP_HOME, pdb.topology, system, dict_motifs, dict_resid_chain, constraint_file)

    system, index_PK_nonPK_stacking_force, index_PK_nonPK_vertical_force, list_bonds_for_PK_helices = add_stacking_force_between_PK_and_nonPK_helices(pdb.topology,system,dict_motifs,dict_resid_chain,dict_jar3d_energy_paras)

    Temp = 300
    integrator = mm.LangevinIntegrator(Temp*unit.kelvin, 10.0/unit.picoseconds, 1.0*unit.femtoseconds)

    platform = mm.Platform.getPlatformByName('CUDA')
    properties = {'CudaPrecision': 'mixed', 'CudaDeviceIndex': gpu_index}
    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setPositions(pdb.getPositions(frame=0))
    state = simulation.context.getState(getPositions=True,getForces=True,getEnergy=True)
    print("\nPotential energy for the initial structure: ",state.getPotentialEnergy(),flush=True)

    simulation.minimizeEnergy()
    state = simulation.context.getState(getPositions=True,getForces=True,getEnergy=True)
    print("Potential energy after energy-minimization: ",state.getPotentialEnergy(),flush=True)

    simulation.context.setVelocitiesToTemperature(Temp*unit.kelvin)

    simulation.context.setParameter("kbpstk_global",0)
    simulation.context.setParameter("kangle_global",0.1)
    simulation.context.setParameter("ktorsion_global",0.1)

    simulation.integrator.setTemperature(400*unit.kelvin)
    simulation.step(500000)

    jar3d_dis_force = simulation.system.getForce(jar3d_dis_force_index)
    jar3d_angle_force = simulation.system.getForce(jar3d_angle_force_index)
    jar3d_torsion_force = simulation.system.getForce(jar3d_torsion_force_index)

    subloop_len_in_hp2way = [len(list_jar3d_dis_bonds[i]) for i in range(len(list_jar3d_dis_bonds))]
    max_subloop_len_in_hp2way = max(subloop_len_in_hp2way)

    if max_subloop_len_in_hp2way == 0:
        num_step = 10000
    else:
        num_step = int(500000/max_subloop_len_in_hp2way)
    if num_step < 10000:
        num_step = 10000

    for i in range(max_subloop_len_in_hp2way):
        for j in range(len(list_jar3d_dis_bonds)):
            dis_bonds_in_subloop = list_jar3d_dis_bonds[j]
            if i < len(dis_bonds_in_subloop):
                dis_bonds = dis_bonds_in_subloop[i]
                for bond_info in dis_bonds:
                    bond_index, atoms_index, paras = bond_info
                    jar3d_dis_force.setBondParameters(bond_index,atoms_index,[paras[0],1000])
        for j in range(len(list_jar3d_angle_bonds)):
            angle_bonds_in_subloop = list_jar3d_angle_bonds[j]
            if i < len(angle_bonds_in_subloop):
                angle_bonds = angle_bonds_in_subloop[i]
                for bond_info in angle_bonds:
                    bond_index, atoms_index, paras = bond_info
                    jar3d_angle_force.setBondParameters(bond_index,atoms_index,[paras[0],1000])
        for j in range(len(list_jar3d_torsion_bonds)):
            torsion_bonds_in_subloop = list_jar3d_torsion_bonds[j]
            if i < len(torsion_bonds_in_subloop):
                torsion_bonds = torsion_bonds_in_subloop[i]
                for bond_info in torsion_bonds:
                    bond_index, atoms_index, paras = bond_info
                    jar3d_torsion_force.setBondParameters(bond_index,atoms_index,[paras[0],1000])
        jar3d_dis_force.updateParametersInContext(simulation.context)
        jar3d_angle_force.updateParametersInContext(simulation.context)
        jar3d_torsion_force.updateParametersInContext(simulation.context)
        simulation.step(num_step)

    simulation.integrator.setTemperature(300*unit.kelvin)
    simulation.step(100000)

    state = simulation.context.getState(getPositions=True,getForces=True,getEnergy=True)
    app.PDBFile.writeFile(pdb.topology, state.getPositions(), open(f'{outfolder}/jar3d.pdb', 'w'))

    totalcycle = int(simulation_time*1000000/1250000.)
    totalstep = totalcycle * 1250000
    writefreq = 50000
    nframe = int(totalstep/writefreq)
    print(f"\nThere are {totalcycle} simulated annealing cycles in total.")
    print(f"Each cycle has 1250000 simulation steps (1.25 ns).")
    print(f"There are {totalstep} simulation steps in total.")
    print(f"A frame is saved every {writefreq} steps.")
    print(f"There are {nframe} frames to be saved in total.\n", flush=True)

    if not os.path.exists(outfolder):
        raise ValueError(f"directory {outfolder} does not exist!")

    simulation.reporters.append(app.PDBReporter(f'{outtrj}', writefreq))

    if "kconstraint_global" in simulation.context.getParameters():
        bool_kconstraint_global = True
    else:
        bool_kconstraint_global = False

    for ncycle in range(totalcycle):
        print(f"Running cycle {ncycle+1} of {totalcycle} ..." ,flush=True)
        time1 = time.time()
        simulation = randomly_select_stacked_helices(dict_motifs, simulation, index_topo_force, list_topo_force_bonds_in_junctions, index_helical_torsion_force, list_helical_torsion_bonds_in_junctions, list_junctions)

        if index_PK_nonPK_stacking_force is not None:
            simulation = randomly_select_stacked_PK_and_nonPK_helices(simulation, index_PK_nonPK_stacking_force, index_PK_nonPK_vertical_force, list_bonds_for_PK_helices)

        simulation.integrator.setTemperature(400*unit.kelvin)

        if bool_kconstraint_global:
            simulation.context.setParameter("kconstraint_global",0.0)

        simulation.context.setParameter("klooploop",50)
        simulation.context.setParameter("dlooploop_lower",0)
        simulation.context.setParameter("dlooploop_ave",3.5)
        simulation.context.setParameter("dlooploop_upper",3)
        simulation.context.setParameter("khphp",50)
        simulation.context.setParameter("dhphp_lower",0)
        simulation.context.setParameter("dhphp_ave",5.5)
        simulation.context.setParameter("dhphp_upper",5)

        simulation.context.setParameter("kbpstk_global",0)
        simulation.context.setParameter("kcompactjunc",0)
        simulation.context.setParameter("koutward",10)

        simulation.context.setParameter("kstack",0)
        simulation.context.setParameter("kparallel",0)
        simulation.context.setParameter("kantiparallel",0)
        simulation.context.setParameter("kvertical",0)
        simulation.step(25000)

        simulation.context.setParameter("kangle_global",0.1)
        simulation.context.setParameter("ktorsion_global",0.1)

        if bool_kconstraint_global:
            simulation.context.setParameter("kconstraint_global",0.01)

        simulation.context.setParameter("kuncompactjunc",50)
        simulation.context.setParameter("duncompactjunc",5)
        simulation.context.setParameter("koutward",5)
        simulation.step(25000)

        simulation.context.setParameter("kstack",500)
        simulation.context.setParameter("kparallel",500)
        simulation.context.setParameter("kantiparallel",500)
        simulation.context.setParameter("kvertical",500)
        simulation.step(200000)

        simulation.context.setParameter("klooploop",100)
        simulation.context.setParameter("dlooploop_lower",0)
        simulation.context.setParameter("dlooploop_ave",0)
        simulation.context.setParameter("dlooploop_upper",3)
        simulation.context.setParameter("khphp",100)
        simulation.context.setParameter("dhphp_lower",2.9)
        simulation.context.setParameter("dhphp_ave",2.9)
        simulation.context.setParameter("dhphp_upper",5)

        simulation.context.setParameter("koutward",0)
        simulation.context.setParameter("kuncompactjunc",0)
        simulation.context.setParameter("dcompactjunc",1)

        simulation.context.setParameter("kangle_global",1)
        simulation.context.setParameter("ktorsion_global",1)
        simulation.context.setParameter("kbpstk_global",1)

        for i in range(5):
            simulation.integrator.setTemperature((400-i*75)*unit.kelvin)
            simulation.context.setParameter("kcompactjunc",10+i*10)
            if bool_kconstraint_global:
                simulation.context.setParameter("kconstraint_global",0.2+i*0.2)
            simulation.step(200000)
        time2 = time.time()
        mins = (time2-time1)/60.
        print(f"    It took about {mins:.1f} minutes to run cycle {ncycle+1}.",flush=True)
