import simtk.openmm as mm
import simtk.unit as unit
import numpy as np
import os
from add_local_force import add_helical_torsion_force_in_junctions

def access_to_residues_by_chain_name_and_resid(topology):
    dict_residues = {}
    for chain in topology.chains():
        chain_name = chain.id
        for res in chain.residues():
            resid = res.id
            dict_atoms = {}
            for atom in res.atoms():
                dict_atoms[atom.name] = atom
                if atom.name in ["AN9","CN1","GN9","UN1"]:
                    dict_atoms["NX"] = atom
                elif atom.name in ["AC2","CC2","GC2","UC2"]:
                    dict_atoms["C2"] = atom
                elif atom.name in ["AC6","CC4","GC6","UC4"]:
                    dict_atoms["CY"] = atom
                #print (chain_name,resid,atom.name,atom.index)
            dict_residues[(chain_name,resid)] = [res,dict_atoms]
    return dict_residues


dict_nt_atom_name = {'CGA':['BBP','BBC','AN9','AC2','AC6'], 'CGC':['BBP','BBC','CN1','CC2','CC4'], 'CGG':['BBP','BBC','GN9','GC2','GC6'], 'CGU':['BBP','BBC','UN1','UC2','UC4']}
dict_atom_to_normal_atom = {"BBP":"P","BBC":"C4'","AN9":"N9","AC2":"C2","AC6":"C6","CN1":"N1","CC2":"C2","CC4":"C4","GN9":"N9","GC2":"C2","GC6":"C6","UN1":"N1","UC2":"C2","UC4":"C4"}


def get_atoms_index_in_junction_branch(dict_resid_chain,dict_residues,junction_branch,branch_index):
    anchor1 = junction_branch[0]
    anchor2 = junction_branch[1]
    loop_len = anchor2 - anchor1 - 1
    
    chain_ntidx = dict_resid_chain[str(anchor1)]
    nt1 = dict_residues[chain_ntidx]
    chain_ntidx = dict_resid_chain[str(anchor2)]
    nt2 = dict_residues[chain_ntidx]
    
    BBP_1 = nt1[1]["BBP"].index
    BBC_1 = nt1[1]["BBC"].index
    NX_1 = nt1[1]["NX"].index
    C2_1 = nt1[1]["C2"].index
    CY_1 = nt1[1]["CY"].index

    BBP_2 = nt2[1]["BBP"].index
    BBC_2 = nt2[1]["BBC"].index
    NX_2 = nt2[1]["NX"].index
    C2_2 = nt2[1]["C2"].index
    CY_2 = nt2[1]["CY"].index

    dict_atoms_index_in_junction_branch = {}
    dict_atoms_index_in_junction_branch[(branch_index,"BBP_1")] = BBP_1
    dict_atoms_index_in_junction_branch[(branch_index,"BBC_1")] = BBC_1
    dict_atoms_index_in_junction_branch[(branch_index,"NX_1")] = NX_1
    dict_atoms_index_in_junction_branch[(branch_index,"C2_1")] = C2_1
    dict_atoms_index_in_junction_branch[(branch_index,"CY_1")] = CY_1
        
    dict_atoms_index_in_junction_branch[(branch_index,"BBP_2")] = BBP_2
    dict_atoms_index_in_junction_branch[(branch_index,"BBC_2")] = BBC_2
    dict_atoms_index_in_junction_branch[(branch_index,"NX_2")] = NX_2
    dict_atoms_index_in_junction_branch[(branch_index,"C2_2")] = C2_2
    dict_atoms_index_in_junction_branch[(branch_index,"CY_2")] = CY_2
    return dict_atoms_index_in_junction_branch, loop_len


def get_atoms_index_in_junction(dict_resid_chain,dict_residues,junction):
    dict_atoms_index_in_junction = {}
    list_loop_len = []
    for i in range(len(junction)):
        dict_atoms_index_in_junction_branch, loop_len = get_atoms_index_in_junction_branch(dict_resid_chain,dict_residues,junction[i],i)
        list_loop_len.append(loop_len)
        for key in dict_atoms_index_in_junction_branch:
            if key in dict_atoms_index_in_junction.keys():
                print ("Error: duplicate key")
                exit()
            dict_atoms_index_in_junction[key] = dict_atoms_index_in_junction_branch[key]
    return dict_atoms_index_in_junction, list_loop_len


def get_compound_bond_in_one_junction_branch(dict_atoms_index_in_junction,branch_index):
    BBP_1 = dict_atoms_index_in_junction[(branch_index,"BBP_1")]
    BBC_1 = dict_atoms_index_in_junction[(branch_index,"BBC_1")]
    NX_1 = dict_atoms_index_in_junction[(branch_index,"NX_1")]
    C2_1 = dict_atoms_index_in_junction[(branch_index,"C2_1")]
    CY_1 = dict_atoms_index_in_junction[(branch_index,"CY_1")]

    BBP_2 = dict_atoms_index_in_junction[(branch_index,"BBP_2")]
    BBC_2 = dict_atoms_index_in_junction[(branch_index,"BBC_2")]
    NX_2 = dict_atoms_index_in_junction[(branch_index,"NX_2")]
    C2_2 = dict_atoms_index_in_junction[(branch_index,"C2_2")]
    CY_2 = dict_atoms_index_in_junction[(branch_index,"CY_2")]

    d_a = [NX_1,BBP_2]
    theta1_a = [BBC_1,NX_1,BBP_2]
    theta2_a = [NX_1,BBP_2,BBC_2]
    phi1_a = [BBP_1,BBC_1,NX_1,BBP_2]
    phi2_a = [BBC_1,NX_1,BBP_2,BBC_2]
    phi3_a = [NX_1,BBP_2,BBC_2,NX_2]

    base_base = [BBP_1,NX_1,C2_1,CY_1,BBP_2,NX_2,C2_2,CY_2]
    return [d_a, theta1_a, theta2_a, phi1_a, phi2_a, phi3_a, base_base]


def get_compound_bond_in_junction(dict_resid_chain,dict_residues,junction):
    dict_atoms_index_in_junction, list_loop_len = get_atoms_index_in_junction(dict_resid_chain,dict_residues,junction)

    atoms_index_for_paras_in_junction = []
    for branch_index in range(len(junction)):
        atoms_index = get_compound_bond_in_one_junction_branch(dict_atoms_index_in_junction,branch_index)
        atoms_index_for_paras_in_junction.extend(atoms_index)
    return atoms_index_for_paras_in_junction


def get_compound_bond_in_junctions(dict_resid_chain,dict_residues,list_junctions):
    list_atoms_index_for_paras_in_junctions = []
    for junction in list_junctions:
        atoms_index_for_paras_in_junction = get_compound_bond_in_junction(dict_resid_chain,dict_residues,junction)
        list_atoms_index_for_paras_in_junctions.append(atoms_index_for_paras_in_junction)
    return list_atoms_index_for_paras_in_junctions


def add_coaxial_force_in_2way_junctions(topology,system,dict_resid_chain,list_2way_junctions):
    dict_residues = access_to_residues_by_chain_name_and_resid(topology)
    list_atoms_index_for_paras_in_junctions = get_compound_bond_in_junctions(dict_resid_chain,dict_residues,list_2way_junctions)

    num_junc = len(list_atoms_index_for_paras_in_junctions)
    if num_junc != len(list_2way_junctions):
        print (num_junc)
        print (list_2way_junctions)
        print ("The number of junctions is not right, either!")
        exit()

    coaxial_penalty = f"100*(cosbeta-1.0)^2 + 20*(dis1-dis2)^2; cosbeta=(dot_r1R1*dot_r2R2-dot_r1R2*dot_r2R1)/(distance(p1,p2)*distance(p1,p3)*distance(p4,p5)*distance(p4,p6)*sin(angle(p2,p1,p3))*sin(angle(p5,p4,p6))); dot_r1R1=a*g+b*h+c*i; dot_r2R2=d*j+e*k+f*l; dot_r1R2=a*j+b*k+c*l; dot_r2R1=d*g+e*h+f*i; a=x2-x1; b=y2-y1; c=z2-z1; d=x3-x1; e=y3-y1; f=z3-z1; g=x5-x4; h=y5-y4; i=z5-z4; j=x6-x4; k=y6-y4; l=z6-z4; dis1=distance(p1,p4); dis2=distance(p7,p10)"
    coaxial_force = mm.CustomCompoundBondForce(12, coaxial_penalty)
    
    for i in range(num_junc):
        atoms_index = list_atoms_index_for_paras_in_junctions[i][6][1:4] + list_atoms_index_for_paras_in_junctions[i][6][5:8] + list_atoms_index_for_paras_in_junctions[i][13][1:4] + list_atoms_index_for_paras_in_junctions[i][13][5:8]
        index_bond = coaxial_force.addBond(atoms_index,[])
        print (atoms_index)
    coaxial_force.setForceGroup(11)
    index_coaxial_force = system.addForce(coaxial_force)
    return system


def add_interhelical_force_in_junctions(topology,system,dict_resid_chain,list_junctions,interhelical_paras_for_junctions):
    dict_residues = access_to_residues_by_chain_name_and_resid(topology)
    list_atoms_index_for_paras_in_junctions = get_compound_bond_in_junctions(dict_resid_chain,dict_residues,list_junctions)

    if interhelical_paras_for_junctions is not None:
        if len(list_atoms_index_for_paras_in_junctions) != len(interhelical_paras_for_junctions):
            print ("The number of junctions is not right!")
            print (len(list_atoms_index_for_paras_in_junctions))
            print (len(interhelical_paras_for_junctions))
            exit()
    num_junc = len(list_atoms_index_for_paras_in_junctions)
    if num_junc != len(list_junctions):
        print (num_junc)
        print (list_junctions)
        print ("The number of junctions is not right, either!")
        exit()

    quadratic_function1 = f"k_interhelical*(step(deltaD-deltaD0)*(deltaD-deltaD0)^2); deltaD=abs(d-d0); d=distance(p1,p2)"
    distance_force = mm.CustomCompoundBondForce(2, quadratic_function1)
    distance_force.addGlobalParameter("k_interhelical",0.0*unit.kilojoule/unit.mole)
    distance_force.addPerBondParameter("d0")
    distance_force.addPerBondParameter("deltaD0")

    quadratic_function2 = f"k_interhelical*(step(deltaA-deltaA0)*(deltaA-deltaA0)^2); deltaA=abs(theta-theta0); theta=angle(p1,p2,p3)"
    angle_force = mm.CustomCompoundBondForce(3, quadratic_function2)
    angle_force.addGlobalParameter("k_interhelical",0.0*unit.kilojoule/unit.mole)
    angle_force.addPerBondParameter("theta0")
    angle_force.addPerBondParameter("deltaA0")

    quadratic_function3 = f"k_interhelical*(step(deltaT-deltaT0)*(deltaT-deltaT0)^2); deltaT=min(dt,2*pi-dt); dt=abs(theta-theta0); pi = 3.1415926535; theta=dihedral(p1,p2,p3,p4)"
    torsion_force = mm.CustomCompoundBondForce(4, quadratic_function3)
    torsion_force.addGlobalParameter("k_interhelical",0.0*unit.kilojoule/unit.mole)
    torsion_force.addPerBondParameter("theta0")
    torsion_force.addPerBondParameter("deltaT0")

    Eupper1 = 1000.0
    Eupper2 = 500.0
    stack_energy = f"stack_switch*(step(Estack-{Eupper1})*0.25 + step(Estack-{Eupper2})*step({Eupper1}-Estack)*0.5 + step({Eupper2}-Estack)*1.0)*Estack"
    parallel_energy = f"parallel_switch*(step(Eparallel-{Eupper1})*0.25 + step(Eparallel-{Eupper2})*step({Eupper1}-Eparallel)*0.5 + step({Eupper2}-Eparallel)*1.0)*Eparallel"
    antiparallel_energy = f"antiparallel_switch*(step(Eantiparallel-{Eupper1})*0.1 + step(Eantiparallel-{Eupper2})*step({Eupper1}-Eantiparallel)*0.5 + step({Eupper2}-Eantiparallel)*1.0)*Eantiparallel"
    vertical_energy = f"vertical_switch*(step(Evertical-{Eupper1})*0.25 + step(Evertical-{Eupper2})*step({Eupper1}-Evertical)*0.5 + step({Eupper2}-Evertical)*1.0)*Evertical"
    loopcoil_energy = f"5*(1-stack_switch)*Eloopcoil"

    Estack = f"Estack = kstack * ((cosbeta-1.0)^2 + 0.1*step(-cosbeta)*(cosbeta-1.0)^4 + (dis1-dis2)^2)"
    Eparallel = f"Eparallel = kparallel*((cosbeta-1.0)^2 + 0.1*step(-cosbeta)*(cosbeta-1.0)^4)"
    Eantiparallel = f"Eantiparallel = kantiparallel*((cosbeta+1.0)^2 + 0.1*step(cosbeta)*(cosbeta+1.0)^4)"
    Evertical = f"Evertical = kvertical*((cosbeta-0.0)^2 + 0.1*step(abs(cosbeta)-0.707106781)*(cosbeta-0.0)^4)"
    Eloopcoil = f"Eloopcoil = step(looplen-5)*step(dispp-dispp0)*(dispp-dispp0)^2"
    cosbeta = f"cosbeta=(dot_r1R1*dot_r2R2-dot_r1R2*dot_r2R1)/(distance(p1,p2)*distance(p1,p3)*distance(p4,p5)*distance(p4,p6)*sin(angle(p2,p1,p3))*sin(angle(p5,p4,p6))); dot_r1R1=a*g+b*h+c*i; dot_r2R2=d*j+e*k+f*l; dot_r1R2=a*j+b*k+c*l; dot_r2R1=d*g+e*h+f*i; a=x2-x1; b=y2-y1; c=z2-z1; d=x3-x1; e=y3-y1; f=z3-z1; g=x5-x4; h=y5-y4; i=z5-z4; j=x6-x4; k=y6-y4; l=z6-z4;"
    dis = f"dis1=distance(p1,p4); dis2=distance(p7,p10); dispp=distance(p13,p14); dispp0=0.15*looplen+1.0"

    topology_energy = f"khelixhelix_global*(step(abs(cosbeta)-0.866025404)*(abs(cosbeta)-1.0)^2 + step(0.5-abs(cosbeta))*(abs(cosbeta)-0.0)^2 + step(-abs(cosbeta)+0.866025404)*step(-0.5+abs(cosbeta))*(abs(cosbeta)-0.707106781)^2) + {stack_energy} + {parallel_energy} + {antiparallel_energy} + {vertical_energy} + {loopcoil_energy}; {Estack}; {Eparallel}; {Eantiparallel}; {Evertical}; {Eloopcoil}; {cosbeta}; {dis}"

    topo_force = mm.CustomCompoundBondForce(14, topology_energy)
    topo_force.addGlobalParameter("khelixhelix_global",10.0*unit.kilojoule/unit.mole)
    topo_force.addGlobalParameter("kstack",0.0*unit.kilojoule/unit.mole)
    topo_force.addGlobalParameter("kparallel",0.0*unit.kilojoule/unit.mole)
    topo_force.addGlobalParameter("kantiparallel",0.0*unit.kilojoule/unit.mole)
    topo_force.addGlobalParameter("kvertical",0.0*unit.kilojoule/unit.mole)
    topo_force.addPerBondParameter("stack_switch")
    topo_force.addPerBondParameter("parallel_switch")
    topo_force.addPerBondParameter("antiparallel_switch")
    topo_force.addPerBondParameter("vertical_switch")
    topo_force.addPerBondParameter("looplen")

    list_topo_force_bonds_in_junctions = []
    for i in range(num_junc):
        if len(list_atoms_index_for_paras_in_junctions[i])%7 != 0:
            print ("ERROR: there should be 7*nway atoms index list")
            exit()
        nway1 = int(len(list_atoms_index_for_paras_in_junctions[i])/7)
        if interhelical_paras_for_junctions is not None:
            if len(interhelical_paras_for_junctions[i])%12 != 0:
                print ("ERROR: there should be 12*nway paras")
                exit()
            nway2 = int(len(interhelical_paras_for_junctions[i])/12)
            if nway1 != nway2:
                print ("nway1 != nway2")
                exit()

        bonds_in_junction = []
        for j in range(nway1):
            if interhelical_paras_for_junctions is not None:
                m = 2.0
                distance_force.addBond(list_atoms_index_for_paras_in_junctions[i][j*7], [interhelical_paras_for_junctions[i][j*12],m*interhelical_paras_for_junctions[i][j*12+1]])
                print ("dis",list_atoms_index_for_paras_in_junctions[i][j*7],[interhelical_paras_for_junctions[i][j*12],m*interhelical_paras_for_junctions[i][j*12+1]])

                angle_force.addBond(list_atoms_index_for_paras_in_junctions[i][1+j*7], [interhelical_paras_for_junctions[i][2+j*12],m*interhelical_paras_for_junctions[i][3+j*12]])
                print ("angle",list_atoms_index_for_paras_in_junctions[i][1+j*7],[interhelical_paras_for_junctions[i][2+j*12],m*interhelical_paras_for_junctions[i][3+j*12]])

                angle_force.addBond(list_atoms_index_for_paras_in_junctions[i][2+j*7], [interhelical_paras_for_junctions[i][4+j*12],m*interhelical_paras_for_junctions[i][5+j*12]])
                print ("angle",list_atoms_index_for_paras_in_junctions[i][2+j*7], [interhelical_paras_for_junctions[i][4+j*12],m*interhelical_paras_for_junctions[i][5+j*12]])

                torsion_force.addBond(list_atoms_index_for_paras_in_junctions[i][3+j*7], [interhelical_paras_for_junctions[i][6+j*12],m*interhelical_paras_for_junctions[i][7+j*12]])
                print ("torsion",list_atoms_index_for_paras_in_junctions[i][3+j*7], [interhelical_paras_for_junctions[i][6+j*12],m*interhelical_paras_for_junctions[i][7+j*12]])

                torsion_force.addBond(list_atoms_index_for_paras_in_junctions[i][4+j*7], [interhelical_paras_for_junctions[i][8+j*12],m*interhelical_paras_for_junctions[i][9+j*12]])
                print ("torsion",list_atoms_index_for_paras_in_junctions[i][4+j*7], [interhelical_paras_for_junctions[i][8+j*12],m*interhelical_paras_for_junctions[i][9+j*12]])

                torsion_force.addBond(list_atoms_index_for_paras_in_junctions[i][5+j*7], [interhelical_paras_for_junctions[i][10+j*12],m*interhelical_paras_for_junctions[i][11+j*12]])
                print ("torsion",list_atoms_index_for_paras_in_junctions[i][5+j*7], [interhelical_paras_for_junctions[i][10+j*12],m*interhelical_paras_for_junctions[i][11+j*12]])

            if j == 0:
                atoms_index = list_atoms_index_for_paras_in_junctions[i][6+j*7][1:4] + list_atoms_index_for_paras_in_junctions[i][6+j*7][5:8] + list_atoms_index_for_paras_in_junctions[i][6+(j+1)*7][1:4] + list_atoms_index_for_paras_in_junctions[i][6+(nway1-1)*7][-3:] + [list_atoms_index_for_paras_in_junctions[i][6+j*7][0],list_atoms_index_for_paras_in_junctions[i][6+j*7][4]]
            elif j == nway1 - 1:
                atoms_index = list_atoms_index_for_paras_in_junctions[i][6+j*7][1:4] + list_atoms_index_for_paras_in_junctions[i][6+j*7][5:8] + list_atoms_index_for_paras_in_junctions[i][6+0*7][1:4] + list_atoms_index_for_paras_in_junctions[i][6+(j-1)*7][-3:] + [list_atoms_index_for_paras_in_junctions[i][6+j*7][0],list_atoms_index_for_paras_in_junctions[i][6+j*7][4]]
            else:
                atoms_index = list_atoms_index_for_paras_in_junctions[i][6+j*7][1:4] + list_atoms_index_for_paras_in_junctions[i][6+j*7][5:8] + list_atoms_index_for_paras_in_junctions[i][6+(j+1)*7][1:4] + list_atoms_index_for_paras_in_junctions[i][6+(j-1)*7][-3:] + [list_atoms_index_for_paras_in_junctions[i][6+j*7][0],list_atoms_index_for_paras_in_junctions[i][6+j*7][4]]

            index_bond = topo_force.addBond(atoms_index,[0.0,0.0,0.0,0.0,0.0])
            bonds_in_junction.append([index_bond,atoms_index])
        list_topo_force_bonds_in_junctions.append(bonds_in_junction)
       
    if interhelical_paras_for_junctions is not None:
        distance_force.setForceGroup(20)
        angle_force.setForceGroup(21)
        torsion_force.setForceGroup(22)
        index_interhelical_force_distance = system.addForce(distance_force)
        index_interhelical_force_angle = system.addForce(angle_force)
        index_interhelical_force_torsion = system.addForce(torsion_force)
    topo_force.setForceGroup(9)
    index_topo_force = system.addForce(topo_force)

    system, index_helical_torsion_force, list_helical_torsion_bonds_in_junctions = add_helical_torsion_force_in_junctions(topology,system,dict_resid_chain,list_junctions)
    return system, index_topo_force, list_topo_force_bonds_in_junctions, index_helical_torsion_force, list_helical_torsion_bonds_in_junctions


def add_stacking_force_between_helices_not_in_junctions(topology,system,dict_motifs,dict_resid_chain):
    dict_residues = access_to_residues_by_chain_name_and_resid(topology)
    list_nt_idx_in_helix = []
    list_helices_non_PK = []
    list_helices_PK = []
    for key in ["helix","PK_helix"]:
        helices = dict_motifs[key]
        for helix in helices:
            h1, h4 = helix[0]
            h2, h3 = helix[1]
            if h1 == h2:
                continue
            for i in range(h1,h2+1):
                list_nt_idx_in_helix.append(i)
            for i in range(h3,h4+1):
                list_nt_idx_in_helix.append(i)
            if key == "helix":
                list_helices_non_PK.append(helix)
            elif key == "PK_helix":
                list_helices_PK.append(helix)

    list_helices_in_junction = [] # including 2-way junction
    for loop_type in ["2way_loops","3way_loops","4way_loops"]:
        motifs = dict_motifs[loop_type]
        for motif in motifs:
            helices_in_junction = []
            for loop in motif:
                ib, ie = loop
                for key in ["helix","PK_helix"]:
                    for helix in dict_motifs[key]:
                        h1, h4 = helix[0]
                        h2, h3 = helix[1]
                        if h1 == h2:
                            continue
                        if ib in [h1,h2,h3,h4] or ie in [h1,h2,h3,h4]:
                            if helix not in helices_in_junction:
                                helices_in_junction.append(helix)
            list_helices_in_junction.append(helices_in_junction)

    list_helices_non_PK = sorted(list_helices_non_PK,key=lambda x:x[0][0])
    list_helices_PK = sorted(list_helices_PK,key=lambda x:x[0][0])
    list_helices = list_helices_non_PK + list_helices_PK
    list_two_stacked_nts_between_helices = []
    for i in range(len(list_helices)-1):
        helix1 = list_helices[i]
        h1, h4 = helix1[0]
        h2, h3 = helix1[1]
        for j in range(i+1,len(list_helices)):
            helix2 = list_helices[j]
            bool_skip = False
            for helices_in_junction in list_helices_in_junction:
                if helix1 in helices_in_junction and helix2 in helices_in_junction:
                    bool_skip = True
                    break
            if bool_skip:
                continue
            ha, hd = helix2[0]
            hb, hc = helix2[1]

            if ha > h4:
                bool_connected = True
                for k in range(h4+1,ha):
                    if k in list_nt_idx_in_helix:
                        bool_connected = False
                        break
                if bool_connected:
                    list_two_stacked_nts_between_helices.append([h4,ha])
            elif ha > h2 and hd < h3:
                bool_connected = True
                for k in range(h2+1,ha):
                    if k in list_nt_idx_in_helix:
                        bool_connected = False
                        break
                if bool_connected:
                    list_two_stacked_nts_between_helices.append([h2,ha])
            elif ha > h2 and hd > h3:
                bool_connected = True
                for k in range(hb+1,h3):
                    if k in list_nt_idx_in_helix:
                        bool_connected = False
                        break
                if bool_connected:
                    list_two_stacked_nts_between_helices.append([hb,h3])
            elif ha < h2 and hc > h2 and hd < h3:
                bool_connected = True
                for k in range(hd+1,h3):
                    if k in list_nt_idx_in_helix:
                        bool_connected = False
                        break
                if bool_connected:
                    list_two_stacked_nts_between_helices.append([hd,h3])

    list_stacked_nts_atoms_index = []
    list_stacked_nts_atoms_index_in_single_loops = []
    for nts_idx in list_two_stacked_nts_between_helices:
        idx1, idx2 = nts_idx
        chain_ntidx = dict_resid_chain[str(idx1)]
        nt1 = dict_residues[chain_ntidx]
        chain_ntidx = dict_resid_chain[str(idx2)]
        nt2 = dict_residues[chain_ntidx]
        if nt1[0].chain.id != nt2[0].chain.id:
            continue
        if abs(idx1-idx2) > 5:
            continue

        atoms_index = []
        atoms_index.append(nt1[1]["NX"].index)
        atoms_index.append(nt1[1]["C2"].index)
        atoms_index.append(nt1[1]["CY"].index)
        atoms_index.append(nt2[1]["NX"].index)
        atoms_index.append(nt2[1]["C2"].index)
        atoms_index.append(nt2[1]["CY"].index)
        list_stacked_nts_atoms_index.append(atoms_index)
        print(f"nts between stacked helix: {idx1} {idx2} {atoms_index}")

        bool_in_single_loops = False
        for motif in dict_motifs["single_loops"]:
            for loop in motif:
                ib, ie = loop
                if min(idx1,idx2) >= ib and max(idx1,idx2) <= ie:
                    bool_in_single_loops = True
                    break
            if bool_in_single_loops:
                break
        if bool_in_single_loops:
            for i in range(idx1,idx2):
                chain_ntidx = dict_resid_chain[str(i)]
                nt1 = dict_residues[chain_ntidx]
                chain_ntidx = dict_resid_chain[str(i+1)]
                nt2 = dict_residues[chain_ntidx]
                P_1 = nt1[1]["BBP"].index
                C4s_1 = nt1[1]["BBC"].index
                NX_1 = nt1[1]["NX"].index
                P_2 = nt2[1]["BBP"].index
                C4s_2 = nt2[1]["BBC"].index
                NX_2 = nt2[1]["NX"].index
                list_stacked_nts_atoms_index_in_single_loops.append([[P_1,C4s_1,P_2,C4s_2],[np.radians(-150.1),np.radians(89.0),np.radians(100.3)]])
                list_stacked_nts_atoms_index_in_single_loops.append([[NX_1,C4s_1,P_2,C4s_2],[np.radians(-57.2),np.radians(93.7),np.radians(100.3)]])
                list_stacked_nts_atoms_index_in_single_loops.append([[C4s_1,P_2,C4s_2,NX_2],[np.radians(76.3),np.radians(100.3),np.radians(92.9)]])
                list_stacked_nts_atoms_index_in_single_loops.append([[NX_1,C4s_1,C4s_2,NX_2],[np.radians(16.0),np.radians(72.6),np.radians(83.6)]])
                if i+2 <= idx2:
                    chain_ntidx = dict_resid_chain[str(i+2)]
                    nt3 = dict_residues[chain_ntidx]
                    P_3 = nt3[1]["BBP"].index
                    list_stacked_nts_atoms_index_in_single_loops.append([[C4s_1,P_2,C4s_2,P_3],[np.radians(170.0)]])
                print(f"nts in single loops between stacked helix: {idx1} {idx2} {atoms_index}")

    base_parallel_penalty = f"500*(abs(cosbeta)-1.0)^2; cosbeta=(dot_r1R1*dot_r2R2-dot_r1R2*dot_r2R1)/(distance(p1,p2)*distance(p1,p3)*distance(p4,p5)*distance(p4,p6)*sin(angle(p2,p1,p3))*sin(angle(p5,p4,p6))); dot_r1R1=a*g+b*h+c*i; dot_r2R2=d*j+e*k+f*l; dot_r1R2=a*j+b*k+c*l; dot_r2R1=d*g+e*h+f*i; a=x2-x1; b=y2-y1; c=z2-z1; d=x3-x1; e=y3-y1; f=z3-z1; g=x5-x4; h=y5-y4; i=z5-z4; j=x6-x4; k=y6-y4; l=z6-z4;"
    base_parallel_force = mm.CustomCompoundBondForce(6, base_parallel_penalty)

    for atoms_index in list_stacked_nts_atoms_index:
        base_parallel_force.addBond(atoms_index,[])
    if list_stacked_nts_atoms_index:
        base_parallel_force.setForceGroup(11)
        system.addForce(base_parallel_force)

    torsion_angle_energy = f"500*step(sin(agl1)-sin(agl0))*step(sin(agl2)-sin(agl0))*step(deltaT-deltaT0)*(deltaT-deltaT0)^2 + 500*step(abs(agl1-agl1_0)-deltaT0)*(agl1-agl1_0)^2 + 500*step(abs(agl2-agl2_0)-deltaT0)*(agl2-agl2_0)^2; deltaT=min(dt,2*pi-dt); dt=abs(theta-theta0); pi = 3.1415926535; theta=dihedral(p1,p2,p3,p4); deltaT0=0.436332313; agl1=angle(p1,p2,p3); agl2=angle(p2,p3,p4); agl0=3.054326191" #deltaT0 = 25 degree; 5 < angle1/2 < 175
    torsion_angle_force = mm.CustomCompoundBondForce(4, torsion_angle_energy)
    torsion_angle_force.addPerBondParameter("theta0")
    torsion_angle_force.addPerBondParameter("agl1_0")
    torsion_angle_force.addPerBondParameter("agl2_0")

    if list_stacked_nts_atoms_index_in_single_loops:
        for paras in list_stacked_nts_atoms_index_in_single_loops:
            torsion_angle_force.addBond(paras[0],paras[1])
            print(paras[0],paras[1])
        torsion_angle_force.setForceGroup(11)
        system.addForce(torsion_angle_force)
    return system


def add_stacking_force_between_PK_and_nonPK_helices(topology,system,dict_motifs,dict_resid_chain,dict_jar3d_energy_paras):
    dict_residues = access_to_residues_by_chain_name_and_resid(topology)

    list_nt_idx_in_jar3d = []
    for key in dict_jar3d_energy_paras:
        for i in key:
            list_nt_idx_in_jar3d.append(i)

    list_nt_idx_in_helix = []
    for helix in dict_motifs["PK_helix"]+dict_motifs["helix"]:
        h1, h4 = helix[0]
        h2, h3 = helix[1]
        if h1 == h2:
            continue
        for i in range(h1,h2+1):
            list_nt_idx_in_helix.append(i)
        for i in range(h3,h4+1):
            list_nt_idx_in_helix.append(i)

    list_stacked_nt_idx = []
    for helix1 in dict_motifs["PK_helix"]:
        h1, h4 = helix1[0]
        h2, h3 = helix1[1]
        if h1 == h2:
            continue
        list_stacked_nt_idx_for_helix1 = []
        for helix2 in dict_motifs["helix"]:
            ha, hd = helix2[0]
            hb, hc = helix2[1]
            if ha == hb:
                continue
            min_dis = 1000
            list_nearest_two_nts = []
            for i in [h1,h2,h3,h4]:
                chain_i = dict_resid_chain[str(i)][0]
                for j in [ha,hb,hc,hd]:
                    chain_j = dict_resid_chain[str(j)][0]
                    if chain_i != chain_j:
                        continue
                    bool_skip = False
                    for k in range(min(i,j)+1,max(i,j)):
                        if k in list_nt_idx_in_helix:
                            bool_skip = True
                            break
                    if bool_skip:
                        continue                        
                    dis = abs(i-j)
                    if dis == min_dis:
                        if i < j:
                            list_nearest_two_nts.append([i,j])
                        else:
                            list_nearest_two_nts.append([j,i])
                        min_dis = dis
                    elif dis < min_dis:
                        if i < j:
                            list_nearest_two_nts = [[i,j]]
                        else:
                            list_nearest_two_nts = [[j,i]]
                        min_dis = dis
            if len(list_nearest_two_nts) > 2:
                raise ValueError(f"wrong results: {list_nearest_two_nts}")
            elif len(list_nearest_two_nts) == 2:
                v1, v2 = list_nearest_two_nts
                if v1[0] > v2[0]:
                    if v1[0] not in list_nt_idx_in_jar3d or v1[1] not in list_nt_idx_in_jar3d:
                        list_stacked_nt_idx_for_helix1.append(v1)
                else:
                    if v2[0] not in list_nt_idx_in_jar3d or v2[1] not in list_nt_idx_in_jar3d:
                        list_stacked_nt_idx_for_helix1.append(v2)
            elif len(list_nearest_two_nts) == 1:
                v = list_nearest_two_nts[0]
                if v[0] not in list_nt_idx_in_jar3d or v[1] not in list_nt_idx_in_jar3d:
                    list_stacked_nt_idx_for_helix1.append(v)
        if list_stacked_nt_idx_for_helix1:
            list_stacked_nt_idx.append(list_stacked_nt_idx_for_helix1)

    for motif in dict_motifs["single_loops_non_PK"]:
        for single_loop in motif:
            h1, h2 = single_loop
            if h1 > h2:
                h1, h2 = h2, h1
            elif h1 == h2:
                raise ValueError(f"wrong nonPK single loops: {single_loop}")
            list_stacked_nt_idx_for_helix1 = []
            list_stacked_nt_idx_for_helix1.append([h1,h2])
            if list_stacked_nt_idx_for_helix1:
                list_stacked_nt_idx.append(list_stacked_nt_idx_for_helix1)

    #print(f"stacked helices candidates: {list_stacked_nt_idx}",flush=True)

    torsion_angle_energy = f"k_pk_nonpk * (step(sin(agl1)-sin(agl0))*step(sin(agl2)-sin(agl0))*step(deltaT-deltaT0)*(deltaT-deltaT0)^2 + step(abs(agl1-agl1_0)-deltaT0)*(agl1-agl1_0)^2 + step(abs(agl2-agl2_0)-deltaT0)*(agl2-agl2_0)^2); deltaT=min(dt,2*pi-dt); dt=abs(theta-theta0); pi = 3.1415926535; theta=dihedral(p1,p2,p3,p4); deltaT0=0.087266461; agl1=angle(p1,p2,p3); agl2=angle(p2,p3,p4); agl0=3.054326191" #deltaT0 = 5 degree; 5 < angle1/2 < 175
    torsion_angle_force = mm.CustomCompoundBondForce(4, torsion_angle_energy)
    torsion_angle_force.addPerBondParameter("k_pk_nonpk")
    torsion_angle_force.addPerBondParameter("theta0")
    torsion_angle_force.addPerBondParameter("agl1_0")
    torsion_angle_force.addPerBondParameter("agl2_0")

    vertical_helix_helix_energy = f"kvertical_helix_helix*(abs(cosbeta)-0.0)^2; cosbeta=(dot_r1R1*dot_r2R2-dot_r1R2*dot_r2R1)/(distance(p1,p2)*distance(p1,p3)*distance(p4,p5)*distance(p4,p6)*sin(angle(p2,p1,p3))*sin(angle(p5,p4,p6))); dot_r1R1=a*g+b*h+c*i; dot_r2R2=d*j+e*k+f*l; dot_r1R2=a*j+b*k+c*l; dot_r2R1=d*g+e*h+f*i; a=x2-x1; b=y2-y1; c=z2-z1; d=x3-x1; e=y3-y1; f=z3-z1; g=x5-x4; h=y5-y4; i=z5-z4; j=x6-x4; k=y6-y4; l=z6-z4"
    vertical_helix_helix_force = mm.CustomCompoundBondForce(6, vertical_helix_helix_energy)
    vertical_helix_helix_force.addPerBondParameter("kvertical_helix_helix")

    list_bonds_for_PK_helices = []
    for i in range(len(list_stacked_nt_idx)):
        bonds_for_one_PK_helix = []
        selected_prob = []
        loop_len = []
        list_stacked_nt_idx_for_PK_helix = list_stacked_nt_idx[i]
        for stacked_nt_idx in list_stacked_nt_idx_for_PK_helix:
            bonds_for_one_PK_helix_and_nonPK_helix = []
            ib, ie = stacked_nt_idx
            for j in range(ib,ie):
                chain_ntidx = dict_resid_chain[str(j)]
                nt1 = dict_residues[chain_ntidx]
                chain_ntidx = dict_resid_chain[str(j+1)]
                nt2 = dict_residues[chain_ntidx]
                P_1 = nt1[1]["BBP"].index
                C4s_1 = nt1[1]["BBC"].index
                NX_1 = nt1[1]["NX"].index
                P_2 = nt2[1]["BBP"].index
                C4s_2 = nt2[1]["BBC"].index
                NX_2 = nt2[1]["NX"].index

                atoms_index, paras = [[P_1,C4s_1,P_2,C4s_2],[0,np.radians(-150.1),np.radians(89.0),np.radians(100.3)]]
                index_bond = torsion_angle_force.addBond(atoms_index,paras)
                bonds_for_one_PK_helix_and_nonPK_helix.append([index_bond,atoms_index,paras])

                atoms_index, paras = [[NX_1,C4s_1,P_2,C4s_2],[0,np.radians(-57.2),np.radians(93.7),np.radians(100.3)]]
                index_bond = torsion_angle_force.addBond(atoms_index,paras)
                bonds_for_one_PK_helix_and_nonPK_helix.append([index_bond,atoms_index,paras])

                atoms_index, paras = [[C4s_1,P_2,C4s_2,NX_2],[0,np.radians(76.3),np.radians(100.3),np.radians(92.9)]]
                index_bond = torsion_angle_force.addBond(atoms_index,paras)
                bonds_for_one_PK_helix_and_nonPK_helix.append([index_bond,atoms_index,paras])

                atoms_index, paras = [[NX_1,C4s_1,C4s_2,NX_2],[0,np.radians(16.0),np.radians(72.6),np.radians(83.6)]]
                index_bond = torsion_angle_force.addBond(atoms_index,paras)
                bonds_for_one_PK_helix_and_nonPK_helix.append([index_bond,atoms_index,paras])

                if j+2 <= ie:
                    chain_ntidx = dict_resid_chain[str(j+2)]
                    nt3 = dict_residues[chain_ntidx]
                    P_3 = nt3[1]["BBP"].index
                    atoms_index, paras = [[C4s_1,P_2,C4s_2,P_3],[0,np.radians(170.0),np.radians(100.3),np.radians(89.0)]]
                    index_bond = torsion_angle_force.addBond(atoms_index,paras)
                    bonds_for_one_PK_helix_and_nonPK_helix.append([index_bond,atoms_index,paras])
            if ie-ib > 5:
                chain_ntidx = dict_resid_chain[str(ib)]
                nt1 = dict_residues[chain_ntidx]
                chain_ntidx = dict_resid_chain[str(ie)]
                nt2 = dict_residues[chain_ntidx]
                NX_1 = nt1[1]["NX"].index
                C2_1 = nt1[1]["C2"].index
                CY_1 = nt1[1]["CY"].index
                NX_2 = nt2[1]["NX"].index
                C2_2 = nt2[1]["C2"].index
                CY_2 = nt2[1]["CY"].index
                atoms_index = [NX_1,C2_1,CY_1,NX_2,C2_2,CY_2]
                index_bond = vertical_helix_helix_force.addBond(atoms_index,[0.0])
                bonds_for_one_PK_helix_and_nonPK_helix.append([index_bond,atoms_index])
            else:
                bonds_for_one_PK_helix_and_nonPK_helix.append([])
            bonds_for_one_PK_helix.append(bonds_for_one_PK_helix_and_nonPK_helix)
            selected_prob.append(1./(ie-ib))
            loop_len.append(ie-ib)
        selected_prob = [v/sum(selected_prob) for v in selected_prob]
        bonds_for_one_PK_helix.append(selected_prob)
        bonds_for_one_PK_helix.append(loop_len)
        list_bonds_for_PK_helices.append(bonds_for_one_PK_helix)

    if list_bonds_for_PK_helices:
        stacking_force_index = system.addForce(torsion_angle_force)
        vertical_force_index = system.addForce(vertical_helix_helix_force)
        return system, stacking_force_index, vertical_force_index, list_bonds_for_PK_helices
    else:
        return system, None, None, None
