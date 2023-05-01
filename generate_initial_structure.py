from __future__ import print_function
from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit
from sys import stdout
import numpy as np
import sys,os,copy
from generate_circular_structure import generate_circular_initial_structure
from check_structure import check_pdb
from fold_helices_from_circular_structure import fold_helices
from add_local_force import add_torsion_and_angle_force_to_system, add_helical_torsion_force_in_junctions


def runMD_fold_helices(RNAJP_HOME,dict_motifs,dict_resid_chain,init3D,gpu_index,time,nframe,outfolder,outtrj,outpdb,fold_PK):
    app.topology.Topology().loadBondDefinitions(f'{RNAJP_HOME}/myresidues.xml')
    forcefield = app.ForceField(f'{RNAJP_HOME}/myRNA.xml')

    pdb = app.PDBFile(init3D)
    #print (pdb.topology)

    #print ("Creating system...")
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=0.65*unit.nanometers, ewaldErrorTolerance=0.0005, ignoreExternalBonds=True)

    wt_torsion = 1.5
    wt_angle = 1.0
    list_torsion_atoms_index_jar3d, list_angle_atoms_index_jar3d = [], []
    system = add_torsion_and_angle_force_to_system(RNAJP_HOME, pdb.topology, system, dict_motifs, dict_resid_chain, wt_torsion, wt_angle, list_torsion_atoms_index_jar3d, list_angle_atoms_index_jar3d)

    dict_motifs0 = copy.deepcopy(dict_motifs)
    twoway_loops = copy.deepcopy(dict_motifs0["2way_loops_non_PK"])
    for loop in twoway_loops:
        loop1 = loop[0]
        loop2 = loop[1]
        lenloop1 = loop[0][1] - loop[0][0]
        lenloop2 = loop[1][1] - loop[1][0]
        if min(lenloop1,lenloop2) == 1:
            continue
        if lenloop1 > lenloop2:
            loop1[1] = loop1[1] - (lenloop1-lenloop2)
        else:
            loop2[0] = loop2[0] + (lenloop2-lenloop1)
        helix = [[loop1[0],loop2[1]],[loop1[1]-1,loop2[0]+1]]
        dict_motifs0["helix"].append(helix)

    hairpin_loops = dict_motifs0["hairpin_loops_non_PK"]
    for loop in hairpin_loops:
        loop1 = loop[0]
        lenloop1 = loop[0][1] - loop[0][0]
        if lenloop1 - 1 >= 3:
            halflenloop = int((lenloop1-1)/2)
            if (lenloop1-1)%2 == 0:
                num_extend = int((lenloop1-1)/2) - 1
            else:
                num_extend = int((lenloop1-1)/2)
            helix = [[loop1[0],loop1[1]],[loop1[0]+num_extend,loop1[1]-num_extend]]
        dict_motifs0["helix"].append(helix)

    while True:
        bool_merge_helix = False
        for i in range(len(dict_motifs0["helix"])-1):
            helix1 = dict_motifs0["helix"][i]
            for j in range(i+1,len(dict_motifs0["helix"])):
                helix2 = dict_motifs0["helix"][j]
                h1, h4 = helix1[0]
                h2, h3 = helix1[1]
                ha, hd = helix2[0]
                hb, hc = helix2[1]
                if helix1[1] == helix2[0]:
                    dict_motifs0["helix"][i][1] = dict_motifs0["helix"][j][1]
                    dict_motifs0["helix"].pop(j)
                    bool_merge_helix = True
                    break
                elif helix2[1] == helix1[0]:
                    dict_motifs0["helix"][j][1] = dict_motifs0["helix"][i][1]
                    dict_motifs0["helix"].pop(i)
                    bool_merge_helix = True
                    break
                elif (ha-h2 == 1) and (h3-hd == 1) and (dict_resid_chain[str(h2)][0] == dict_resid_chain[str(ha)][0]) and (dict_resid_chain[str(hd)][0] == dict_resid_chain[str(h3)][0]):
                    dict_motifs0["helix"][i][1] = dict_motifs0["helix"][j][1]
                    dict_motifs0["helix"].pop(j)
                    bool_merge_helix = True
                    break
                elif (h1-hb == 1) and (hc-h4 == 1) and (dict_resid_chain[str(hb)][0] == dict_resid_chain[str(h1)][0]) and (dict_resid_chain[str(h4)][0] == dict_resid_chain[str(hc)][0]):
                    dict_motifs0["helix"][j][1] = dict_motifs0["helix"][i][1]
                    dict_motifs0["helix"].pop(i)
                    bool_merge_helix = True
                    break
            if bool_merge_helix:
                break
        if not bool_merge_helix:
            break

    system = fold_helices(pdb.positions, pdb.topology, system, dict_resid_chain, dict_motifs0, fold_PK)
    system, index_helical_torsion_force, list_helical_torsion_bonds_in_junctions = add_helical_torsion_force_in_junctions(pdb.topology,system,dict_resid_chain,twoway_loops)

    Temp = 200
    integrator = mm.LangevinIntegrator(Temp*unit.kelvin, 10.0/unit.picoseconds, 0.5*unit.femtoseconds)

    platform = mm.Platform.getPlatformByName('CUDA')
    properties = {'CudaPrecision': 'mixed', 'CudaDeviceIndex': gpu_index}
    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)

    simulation.context.setPositions(pdb.getPositions(frame=0))
    state = simulation.context.getState(getPositions=True,getForces=True,getEnergy=True)
    #print ("Initial energy",state.getPotentialEnergy(),flush=True)

    simulation.minimizeEnergy()
    state = simulation.context.getState(getPositions=True,getForces=True,getEnergy=True)
    #print ("Minimized energy",state.getPotentialEnergy(),flush=True)

    totalstep = int(time*1000000.)
    writefreq = int(totalstep/nframe)

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    simulation.reporters.append(app.PDBReporter(f'{outfolder}/{outtrj}', writefreq))

    simulation.integrator.setTemperature(Temp*unit.kelvin)

    if "kconstraint_global" in simulation.context.getParameters():
        bool_kconstraint_global = True
    else:
        bool_kconstraint_global = False

    nround = 400
    nstep = int(totalstep/float(nround))
    for i in range(nround):
        if i < 100:
            if i < 20:
                kdis = 0.5
                kdis2 = 0.5
            else:
                kdis = 1*(i-20+1)
                kdis2 = 1
            if fold_PK:
                kdis = i + 1.
                kdis2 = 1
            kagl = 20*(i+1) + 3000
            kdih = 20*(i+1) + 3000
        else:
            kdis = 10*(i+1) - 920
            if fold_PK:
                kdis = 10*(i+1) - 900
                kdis2 = 0.5*i - 49
            kagl = 10*(i+1) + 4000
            kdih = 10*(i+1) + 4000

        simulation.context.setParameter("kdis",kdis)
        simulation.context.setParameter("kdis2",kdis2)
        simulation.context.setParameter("kagl",kagl)
        simulation.context.setParameter("kdih",kdih)
        #print (f"Round {i+1}: (kdis, {kdis}), (kdis2, {kdis2}), (kagl, {kagl}), (kdih, {kdih})")

        if i < 10:
            simulation.context.setParameter("kfreeze",100.0)
        else:
            simulation.context.setParameter("kfreeze",0.0)

        if i < 20:
            simulation.context.setParameter("kbpdih",(i+1)*50.0)

        if i == 20:
            simulation.context.setParameter("kbpdih",1500)
        if i == 350:
            simulation.context.setParameter("kbpdih",2000)
            simulation.integrator.setTemperature(100*unit.kelvin)
        if i == 380:
            simulation.context.setParameter("kloop",0.1)
            simulation.context.setParameter("kPKloop",0.1)

        if i == 0:
            simulation.context.setParameter("kprledge",2000)

        if fold_PK:
            if i == 0:
                simulation.context.setParameter("kagl_PK_hp",500)
            if i == 380:
                simulation.context.setParameter("kagl_PK_hp",100)
            if i == 0:
                simulation.context.setParameter("kagl_unbent_helix",500)
            if i == 200:
                simulation.context.setParameter("kagl_unbent_helix",0)
            if i == 0:
                simulation.context.setParameter("kloop",0.1)
        else:
            if i == 0:
                simulation.context.setParameter("kagl_PK_hp",500)
            if i == 0:
                simulation.context.setParameter("kagl_unbent_helix",500)
            if i == 200:
                simulation.context.setParameter("kloop",0.1)

        if bool_kconstraint_global:
            if not fold_PK:
                simulation.context.setParameter("kconstraint_global",0.0125*(i+1)+5)
            else:
                simulation.context.setParameter("kconstraint_global",10)

        simulation.step(nstep)
        if i == 19:
            simulation.step(9*nstep)
        
    simulation.minimizeEnergy()
    state = simulation.context.getState(getPositions=True,getForces=True,getEnergy=True)
    app.PDBFile.writeFile(simulation.topology, state.getPositions(), open(f'{outfolder}/{outpdb}', 'w'))


def fold_init_structure(RNAJP_HOME,seq,dict_motifs,dict_resid_chain,gpu_index,outfolder):
    print(f"\nGenerating the circular initial 3D structure ...")
    circular_struc = f"{outfolder}/circular_struc.pdb"
    generate_circular_initial_structure(seq,circular_struc)
    print(f"The circular initial 3D structure has been generated: {circular_struc}", flush=True)

    folding_time = 1 # 1 nanosecond
    nframe = 400 # 400 frames to be recorded

    num_PK_helix = 0
    for helix in dict_motifs["PK_helix"]:
        h1 = helix[0][0] # h1 = res_index
        h2 = helix[1][0]
        h3 = helix[1][1]
        h4 = helix[0][1]
        if int(h2) - int(h1) != int(h4) - int(h3):
            print("The lengths of the two strands in the helix are not the same!")
            print (h1, h2)
            print (h3, h4)
            exit()
        if int(h1) == int(h2) or int(h3) == int(h4):
            continue
        if int(h1) > int(h2) or int(h3) > int(h4):
            print ("not in the right order!")
            print (h1, h2)
            print (h3, h4)
            exit()
        num_PK_helix += 1

    print(f"\nGenerating the folded initial 3D structure ...", flush=True)
    if num_PK_helix > 0:
        try_time_all = 0
        bool_good_init_struc_2 = False
        while try_time_all < 20:
            try_time_all += 1
            print(f"\n---------------------------------------------------------------------------")
            print(f"Time {try_time_all} to fold the nonPK helices first and then the PK helices ...")
            inpdb1 = circular_struc
            outpdb1 = "init_nonPK.pdb"
            outtrj1 = "fold_helices_trj_nonPK.pdb"
            try_time_1 = 0
            bool_good_init_struc_1 = False
            while try_time_1 < 10:
                try_time_1 += 1
                print(f"Time {try_time_1} to fold the nonPK helices ...")
                try:
                    runMD_fold_helices(RNAJP_HOME,dict_motifs,dict_resid_chain,inpdb1,gpu_index,folding_time,nframe,outfolder,outtrj1,outpdb1,fold_PK=False)
                except:
                    continue
                pass_check = check_pdb(dict_motifs,outfolder+"/"+outpdb1,check_PK_helix=False)
                if pass_check:
                    bool_good_init_struc_1 = True
                    break
            if not bool_good_init_struc_1:
                print("Failed in producing the initial 3D structure!")
                exit()
            else:
                print(f"The nonPK helices have been folded and the folded structure is stored in {outfolder}/{outpdb1}", flush=True)

            inpdb2 = f"{outfolder}/{outpdb1}"
            outpdb2 = "init.pdb"
            outtrj2 = "fold_helices_trj.pdb"
            try_time_2 = 0
            while try_time_2 < 5:
                try_time_2 += 1
                print(f"\nTime {try_time_2} to fold the PK helices ...")
                try:
                    runMD_fold_helices(RNAJP_HOME,dict_motifs,dict_resid_chain,inpdb2,gpu_index,folding_time,nframe,outfolder,outtrj2,outpdb2,fold_PK=True)
                except:
                    continue
                pass_check = check_pdb(dict_motifs,outfolder+"/"+outpdb2,check_PK_helix=True)
                if pass_check:
                    bool_good_init_struc_2 = True
                    break
            if bool_good_init_struc_2:
                break

        if not bool_good_init_struc_2:
            print("Failed in producing the initial structure!")
            exit()
    else:
        inpdb = circular_struc
        outpdb = "init.pdb"
        outtrj = "fold_helices_trj.pdb"
        try_time = 0
        bool_good_init_struc = False
        while try_time < 20:
            try_time += 1
            print(f"\n---------------------------------------------------------------------------")
            print(f"Time {try_time} to fold the nonPK helices ...")
            try:
                runMD_fold_helices(RNAJP_HOME,dict_motifs,dict_resid_chain,inpdb,gpu_index,folding_time,nframe,outfolder,outtrj,outpdb,fold_PK=False)
            except:
                continue
            pass_check = check_pdb(dict_motifs,outfolder+"/"+outpdb,check_PK_helix=False)
            if pass_check:
                bool_good_init_struc = True
                break
        if not bool_good_init_struc:
            print("Failed in producing the initial structure!")
            exit()
