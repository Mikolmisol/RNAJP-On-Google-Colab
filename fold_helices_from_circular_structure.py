import numpy as np
import simtk.openmm as mm
import simtk.unit as unit
import copy
from add_constraint_force import add_torsion_constraint_force, add_angle_constraint_force_between_four_atoms, add_atom_position_constraint_force


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


def get_all_bonds(dict_resid_chain,dict_residues):
    list_bond_constraints = []
    num_nt = len(dict_resid_chain)
    for i in range(1,num_nt+1):
        chain_ntidx1 = dict_resid_chain[str(i)]
        nt1 = dict_residues[chain_ntidx1]
        nt1_name = nt1[0].name[-1]
        if nt1_name not in ["A","U","G","C"]:
            raise ValueError(f"The nt name {nt1_name} is not in AGCU")
        P_1 = nt1[1]["BBP"].index
        C4s_1 = nt1[1]["BBC"].index
        NX_1 = nt1[1]["NX"].index
        C2_1 = nt1[1]["C2"].index
        CY_1 = nt1[1]["CY"].index
        bond1 = [[P_1,C4s_1],[0.38]]
        bond2 = [[C4s_1,NX_1],[0.34]]
        if nt1_name in ["A","G"]: 
            bond3 = [[NX_1,C2_1],[0.35]]
            bond4 = [[C2_1,CY_1],[0.24]]
            bond5 = [[CY_1,NX_1],[0.35]]
        else:
            bond3 = [[NX_1,C2_1],[0.14]]
            bond4 = [[C2_1,CY_1],[0.24]]
            bond5 = [[CY_1,NX_1],[0.28]]
        list_bond_constraints.append(bond1)
        list_bond_constraints.append(bond2)
        list_bond_constraints.append(bond3)
        list_bond_constraints.append(bond4)
        list_bond_constraints.append(bond5)
        if i < num_nt:
            chain_ntidx2 = dict_resid_chain[str(i+1)]
            if chain_ntidx1[0] != chain_ntidx2[0]:
                continue
            nt2 = dict_residues[chain_ntidx2]
            P_2 = nt2[1]["BBP"].index
            bond6 = [[C4s_1,P_2],[0.38]]
            list_bond_constraints.append(bond6)
    return list_bond_constraints


def get_constraints_in_nt(nt):
    p = nt[1]["BBP"].index
    c4s = nt[1]["BBC"].index
    nx = nt[1]["NX"].index
    c2 = nt[1]["C2"].index
    cy = nt[1]["CY"].index
    nt_name = nt[0].name[-1]

    list_dis_constraints = []
    list_dis_constraints.append([[p,c4s],[0.393]])
    list_dis_constraints.append([[c4s,nx],[0.334]])
    list_dis_constraints.append([[p,nx],[0.528]])

    list_agl_constraints = []
    list_agl_constraints.append([[p,c4s,nx],[np.radians(92.9)]])
    
    list_dih_constraints = []

    if nt_name in "A":
        list_dis_constraints.append([[nx,c2],[0.352]])
        list_dis_constraints.append([[c2,cy],[0.231]])
        list_dis_constraints.append([[cy,nx],[0.351]])
        list_dis_constraints.append([[p,c2],[0.867]])
        list_dis_constraints.append([[p,cy],[0.775]])
        list_agl_constraints.append([[c4s,nx,c2],[np.radians(149.6)]])
        list_agl_constraints.append([[nx,c2,cy],[np.radians(70.7)]])
        list_agl_constraints.append([[c2,cy,nx],[np.radians(71.0)]])
        list_agl_constraints.append([[cy,nx,c2],[np.radians(38.3)]])
        list_dih_constraints.append([[p,c4s,nx,c2],[np.radians(164.1)]])
        list_dih_constraints.append([[c4s,nx,c2,cy],[np.radians(-161.7)]])
    elif nt_name in "G":
        list_dis_constraints.append([[nx,c2],[0.355]])
        list_dis_constraints.append([[c2,cy],[0.247]])
        list_dis_constraints.append([[cy,nx],[0.355]])
        list_dis_constraints.append([[p,c2],[0.873]])
        list_dis_constraints.append([[p,cy],[0.776]])
        list_agl_constraints.append([[c4s,nx,c2],[np.radians(147.8)]])
        list_agl_constraints.append([[nx,c2,cy],[np.radians(69.6)]])
        list_agl_constraints.append([[c2,cy,nx],[np.radians(69.7)]])
        list_agl_constraints.append([[cy,nx,c2],[np.radians(40.7)]])
        list_dih_constraints.append([[p,c4s,nx,c2],[np.radians(165.1)]])
        list_dih_constraints.append([[c4s,nx,c2,cy],[np.radians(-162.7)]])
    elif nt_name in "C":
        list_dis_constraints.append([[nx,c2],[0.139]])
        list_dis_constraints.append([[c2,cy],[0.240]])
        list_dis_constraints.append([[cy,nx],[0.271]])
        list_dis_constraints.append([[p,c2],[0.659]])
        list_dis_constraints.append([[p,cy],[0.624]])
        list_agl_constraints.append([[c4s,nx,c2],[np.radians(153.0)]])
        list_agl_constraints.append([[nx,c2,cy],[np.radians(89.2)]])
        list_agl_constraints.append([[c2,cy,nx],[np.radians(30.9)]])
        list_agl_constraints.append([[cy,nx,c2],[np.radians(59.9)]])
        list_dih_constraints.append([[p,c4s,nx,c2],[np.radians(161.6)]])
        list_dih_constraints.append([[c4s,nx,c2,cy],[np.radians(-159.5)]])
    elif nt_name in "U":
        list_dis_constraints.append([[nx,c2],[0.137]])
        list_dis_constraints.append([[c2,cy],[0.246]])
        list_dis_constraints.append([[cy,nx],[0.279]])
        list_dis_constraints.append([[p,c2],[0.657]])
        list_dis_constraints.append([[p,cy],[0.623]])
        list_agl_constraints.append([[c4s,nx,c2],[np.radians(152.3)]])
        list_agl_constraints.append([[nx,c2,cy],[np.radians(88.7)]])
        list_agl_constraints.append([[c2,cy,nx],[np.radians(29.4)]])
        list_agl_constraints.append([[cy,nx,c2],[np.radians(61.9)]])
        list_dih_constraints.append([[p,c4s,nx,c2],[np.radians(162.1)]])
        list_dih_constraints.append([[c4s,nx,c2,cy],[np.radians(-160.0)]])
    else:
        print (f"Wrong nt name {nt_name}")
        exit()
    return list_dis_constraints, list_agl_constraints, list_dih_constraints


def get_constraints_in_consecutive_nts(nt1,nt2):
    p_1 = nt1[1]["BBP"].index
    c4s_1 = nt1[1]["BBC"].index
    nx_1 = nt1[1]["NX"].index
    c2_1 = nt1[1]["C2"].index
    cy_1 = nt1[1]["CY"].index

    p_2 = nt2[1]["BBP"].index
    c4s_2 = nt2[1]["BBC"].index
    nx_2 = nt2[1]["NX"].index
    c2_2 = nt2[1]["C2"].index
    cy_2 = nt2[1]["CY"].index

    list_dis_constraints = []
    list_dis_constraints.append([[p_1,c4s_1],[0.38]])
    list_dis_constraints.append([[c4s_1,p_2],[0.387]])
    list_dis_constraints.append([[p_2,c4s_2],[0.38]])
    list_dis_constraints.append([[p_1,p_2],[0.547]])
    list_dis_constraints.append([[c4s_1,c4s_2],[0.599]])
    list_dis_constraints.append([[p_1,c4s_2],[0.877]])
    list_dis_constraints.append([[c4s_2,nx_1],[0.59]])
    list_dis_constraints.append([[c4s_1,nx_2],[0.65]])
    list_dis_constraints.append([[nx_1,nx_2],[0.471]])

    list_agl_constraints = []
    list_agl_constraints.append([[p_1,c4s_1,p_2],[np.radians(89.0)]])
    list_agl_constraints.append([[c4s_1,p_2,c4s_2],[np.radians(100.3)]])
    list_agl_constraints.append([[p_1,c4s_1,nx_1],[np.radians(92.9)]])
    list_agl_constraints.append([[nx_1,c4s_1,p_2],[np.radians(93.7)]])
    list_agl_constraints.append([[nx_1,c4s_1,c4s_2],[np.radians(72.6)]])
    list_agl_constraints.append([[p_2,c4s_2,nx_2],[np.radians(92.9)]])
    list_agl_constraints.append([[c4s_1,c4s_2,nx_2],[np.radians(83.6)]])

    list_dih_constraints = []
    list_dih_constraints.append([[p_1,c4s_1,p_2,c4s_2],[np.radians(-150.1)]])
    list_dih_constraints.append([[nx_1,c4s_1,p_2,c4s_2],[np.radians(-57.2)]])
    list_dih_constraints.append([[c4s_1,p_2,c4s_2,nx_2],[np.radians(76.3)]])
    list_dih_constraints.append([[nx_1,c4s_1,c4s_2,nx_2],[np.radians(16.0)]])
    return list_dis_constraints, list_agl_constraints, list_dih_constraints
   

def get_constraints_in_PK_hairpin(dict_resid_chain, dict_residues, hairpin_loop):
    ib, ie = hairpin_loop

    list_angle_constraints = []
    
    chain_ntidx = dict_resid_chain[str(ib-1)]
    nt1 = dict_residues[chain_ntidx]
    p1 = nt1[1]["BBP"].index
    chain_ntidx = dict_resid_chain[str(ib)]
    nt2 = dict_residues[chain_ntidx]
    p2 = nt2[1]["BBP"].index
    for i in range(ib+1,ie):
        chain_ntidx = dict_resid_chain[str(i)]
        nt3 = dict_residues[chain_ntidx]
        p3 = nt3[1]["BBP"].index
        list_angle_constraints.append([p1,p2,p3])

    chain_ntidx = dict_resid_chain[str(ie+1)]
    nt1 = dict_residues[chain_ntidx]
    p1 = nt1[1]["BBP"].index
    chain_ntidx = dict_resid_chain[str(ie)]
    nt2 = dict_residues[chain_ntidx]
    p2 = nt2[1]["BBP"].index
    for i in range(ib+1,ie):
        chain_ntidx = dict_resid_chain[str(i)]
        nt3 = dict_residues[chain_ntidx]
        p3 = nt3[1]["BBP"].index
        list_angle_constraints.append([p1,p2,p3])
    return list_angle_constraints


def get_constraints_for_unbent_helix(dict_resid_chain, dict_residues, dict_motifs):
    list_nt_idx_in_helix = []
    for helix in dict_motifs["helix"]:
        h1, h4 = helix[0]
        h2, h3 = helix[1]
        if h1 == h2:
            continue
        for i in range(h1,h2+1):
            list_nt_idx_in_helix.append(i)
        for i in range(h3,h4+1):
            list_nt_idx_in_helix.append(i)

    list_nt_idx_in_hairpin = []
    for hp in dict_motifs["hairpin_loops"]:
        ib, ie = hp[0]
        for i in range(ib,ie):
            list_nt_idx_in_hairpin.append(i)

    list_angle_constraints = [] # P-P-P angle constraints

    for helix in dict_motifs["helix"] + dict_motifs["PK_helix"]:
        h1, h4 = helix[0]
        h2, h3 = helix[1]
        if h1 == h2:
            continue

        list_anchor = []
        for i in range(h1,h2):
            list_anchor.append([i+1,i])
        for anchor in list_anchor:
            chain_ntidx1 = dict_resid_chain[str(anchor[0])]
            nt1 = dict_residues[chain_ntidx1]
            p1 = nt1[1]["BBP"].index
            c1 = nt1[1]["BBC"].index
            chain_ntidx2 = dict_resid_chain[str(anchor[1])]
            nt2 = dict_residues[chain_ntidx2]
            p2 = nt2[1]["BBP"].index
            c2 = nt2[1]["BBC"].index
            num = 0
            for i in range(h1-1,0,-1):
                if helix in dict_motifs["PK_helix"] and i in list_nt_idx_in_hairpin:
                    break
                if (i>=h1 and i<=h2) or (i>=h3 and i<=h4):
                    break
                chain_ntidx3 = dict_resid_chain[str(i)]
                if chain_ntidx2[0] != chain_ntidx3[0]:
                    break
                nt3 = dict_residues[chain_ntidx3]
                p3 = nt3[1]["BBP"].index
                c3 = nt3[1]["BBC"].index
                list_angle_constraints.append([p1,p2,p3])
                list_angle_constraints.append([c1,c2,c3])
                if i in list_nt_idx_in_helix:
                    num += 1
                if num >= 2:
                    break
       
        list_anchor = []
        for i in range(h1,h2):
            list_anchor.append([i,i+1])
        for anchor in list_anchor:
            chain_ntidx1 = dict_resid_chain[str(anchor[0])]
            nt1 = dict_residues[chain_ntidx1]
            p1 = nt1[1]["BBP"].index
            c1 = nt1[1]["BBC"].index
            chain_ntidx2 = dict_resid_chain[str(anchor[1])]
            nt2 = dict_residues[chain_ntidx2]
            p2 = nt2[1]["BBP"].index
            c2 = nt2[1]["BBC"].index
            num = 0
            for i in range(h2+1,len(dict_resid_chain)+1):
                if helix in dict_motifs["PK_helix"] and i in list_nt_idx_in_hairpin:
                    break
                if (i>=h1 and i<=h2) or (i>=h3 and i<=h4):
                    break
                chain_ntidx3 = dict_resid_chain[str(i)]
                if chain_ntidx2[0] != chain_ntidx3[0]:
                    break
                nt3 = dict_residues[chain_ntidx3]
                p3 = nt3[1]["BBP"].index
                c3 = nt3[1]["BBC"].index
                list_angle_constraints.append([p1,p2,p3])
                list_angle_constraints.append([c1,c2,c3])
                if i in list_nt_idx_in_helix:
                    num += 1
                if num >= 2:
                    break
        
        list_anchor = []
        for i in range(h3,h4):
            list_anchor.append([i+1,i])
        for anchor in list_anchor:
            chain_ntidx1 = dict_resid_chain[str(anchor[0])]
            nt1 = dict_residues[chain_ntidx1]
            p1 = nt1[1]["BBP"].index
            c1 = nt1[1]["BBC"].index
            chain_ntidx2 = dict_resid_chain[str(anchor[1])]
            nt2 = dict_residues[chain_ntidx2]
            p2 = nt2[1]["BBP"].index
            c2 = nt2[1]["BBC"].index
            num = 0
            for i in range(h3-1,0,-1):
                if helix in dict_motifs["PK_helix"] and i in list_nt_idx_in_hairpin:
                    break
                if (i>=h1 and i<=h2) or (i>=h3 and i<=h4):
                    break
                chain_ntidx3 = dict_resid_chain[str(i)]
                if chain_ntidx2[0] != chain_ntidx3[0]:
                    break
                nt3 = dict_residues[chain_ntidx3]
                p3 = nt3[1]["BBP"].index
                c3 = nt3[1]["BBC"].index
                list_angle_constraints.append([p1,p2,p3])
                list_angle_constraints.append([c1,c2,c3])
                if i in list_nt_idx_in_helix:
                    num += 1
                if num >= 2:
                    break
        
        list_anchor = []
        for i in range(h3,h4):
            list_anchor.append([i,i+1])
        for anchor in list_anchor:
            chain_ntidx1 = dict_resid_chain[str(anchor[0])]
            nt1 = dict_residues[chain_ntidx1]
            p1 = nt1[1]["BBP"].index
            c1 = nt1[1]["BBC"].index
            chain_ntidx2 = dict_resid_chain[str(anchor[1])]
            nt2 = dict_residues[chain_ntidx2]
            p2 = nt2[1]["BBP"].index
            c2 = nt2[1]["BBC"].index
            num = 0
            for i in range(h4+1,len(dict_resid_chain)+1):
                if helix in dict_motifs["PK_helix"] and i in list_nt_idx_in_hairpin:
                    break
                if (i>=h1 and i<=h2) or (i>=h3 and i<=h4):
                    break
                chain_ntidx3 = dict_resid_chain[str(i)]
                if chain_ntidx2[0] != chain_ntidx3[0]:
                    break
                nt3 = dict_residues[chain_ntidx3]
                p3 = nt3[1]["BBP"].index
                c3 = nt3[1]["BBC"].index
                list_angle_constraints.append([p1,p2,p3])
                list_angle_constraints.append([c1,c2,c3])
                if i in list_nt_idx_in_helix:
                    num += 1
                if num >= 2:
                    break

    for helix in dict_motifs["helix"]+dict_motifs["PK_helix"]:
        h1, h4 = helix[0]
        h2, h3 = helix[1]
        if h1 == h2:
            continue
        for i in range(h1,h2-1):
            chain_ntidx1 = dict_resid_chain[str(i)]
            nt1 = dict_residues[chain_ntidx1]
            p1 = nt1[1]["BBP"].index
            chain_ntidx2 = dict_resid_chain[str(i+1)]
            nt2 = dict_residues[chain_ntidx2]
            p2 = nt2[1]["BBP"].index
            for j in range(i+2,h2+1):
                chain_ntidx3 = dict_resid_chain[str(j)]
                nt3 = dict_residues[chain_ntidx3]
                p3 = nt3[1]["BBP"].index
                list_angle_constraints.append([p1,p2,p3])
                #print("ppp angle constraint in helix",[p1,p2,p3])
        for i in range(h3,h4-1):
            chain_ntidx1 = dict_resid_chain[str(i)]
            nt1 = dict_residues[chain_ntidx1]
            p1 = nt1[1]["BBP"].index
            chain_ntidx2 = dict_resid_chain[str(i+1)]
            nt2 = dict_residues[chain_ntidx2]
            p2 = nt2[1]["BBP"].index
            for j in range(i+2,h4+1):
                chain_ntidx3 = dict_resid_chain[str(j)]
                nt3 = dict_residues[chain_ntidx3]
                p3 = nt3[1]["BBP"].index
                list_angle_constraints.append([p1,p2,p3])
                #print("ppp angle constraint in helix",[p1,p2,p3])

        for i in range(h1,h2-1):
            chain_ntidx1 = dict_resid_chain[str(i)]
            nt1 = dict_residues[chain_ntidx1]
            c1 = nt1[1]["BBC"].index
            chain_ntidx2 = dict_resid_chain[str(i+1)]
            nt2 = dict_residues[chain_ntidx2]
            c2 = nt2[1]["BBC"].index
            for j in range(i+2,h2+1):
                chain_ntidx3 = dict_resid_chain[str(j)]
                nt3 = dict_residues[chain_ntidx3]
                c3 = nt3[1]["BBC"].index
                list_angle_constraints.append([c1,c2,c3])
                #print("ccc angle constraint in helix",[c1,c2,c3])
        for i in range(h3,h4-1):
            chain_ntidx1 = dict_resid_chain[str(i)]
            nt1 = dict_residues[chain_ntidx1]
            c1 = nt1[1]["BBC"].index
            chain_ntidx2 = dict_resid_chain[str(i+1)]
            nt2 = dict_residues[chain_ntidx2]
            c2 = nt2[1]["BBC"].index
            for j in range(i+2,h4+1):
                chain_ntidx3 = dict_resid_chain[str(j)]
                nt3 = dict_residues[chain_ntidx3]
                c3 = nt3[1]["BBC"].index
                list_angle_constraints.append([c1,c2,c3])
                #print("ccc angle constraint in helix",[c1,c2,c3])
    return list_angle_constraints


def get_constraints_in_consecutive_three_nts(nt1,nt2,nt3):
    p_1 = nt1[1]["BBP"].index
    c4s_1 = nt1[1]["BBC"].index
    nx_1 = nt1[1]["NX"].index
    c2_1 = nt1[1]["C2"].index
    cy_1 = nt1[1]["CY"].index

    p_2 = nt2[1]["BBP"].index
    c4s_2 = nt2[1]["BBC"].index
    nx_2 = nt2[1]["NX"].index
    c2_2 = nt2[1]["C2"].index
    cy_2 = nt2[1]["CY"].index

    p_3 = nt3[1]["BBP"].index
    c4s_3 = nt3[1]["BBC"].index
    nx_3 = nt3[1]["NX"].index
    c2_3 = nt3[1]["C2"].index
    cy_3 = nt3[1]["CY"].index

    list_dih_constraints = []
    list_dih_constraints.append([[c4s_1,p_2,c4s_2,p_3],[np.radians(170.0)]])
    return list_dih_constraints
   

def get_constraints_in_paired_nts(nt1_,nt2_):
    name1 = nt1_[0].name[-1]
    name2 = nt2_[0].name[-1]
    if name1+name2 == "UG":
        nt1 = copy.deepcopy(nt2_)
        nt2 = copy.deepcopy(nt1_)
    else:
        nt1 = copy.deepcopy(nt1_)
        nt2 = copy.deepcopy(nt2_)

    name1 = nt1[0].name[-1]
    p_1 = nt1[1]["BBP"].index
    c4s_1 = nt1[1]["BBC"].index
    nx_1 = nt1[1]["NX"].index
    c2_1 = nt1[1]["C2"].index
    cy_1 = nt1[1]["CY"].index

    name2 = nt2[0].name[-1]
    p_2 = nt2[1]["BBP"].index
    c4s_2 = nt2[1]["BBC"].index
    nx_2 = nt2[1]["NX"].index
    c2_2 = nt2[1]["C2"].index
    cy_2 = nt2[1]["CY"].index

    if name1+name2 == "GU":
        list_dis_constraints = []
        list_dis_constraints.append([[p_1,p_2],[1.806]])
        list_dis_constraints.append([[p_1,c4s_2],[1.704]])
        list_dis_constraints.append([[p_1,nx_2],[1.373]])
        list_dis_constraints.append([[p_2,c4s_1],[1.704]])
        list_dis_constraints.append([[p_2,nx_1],[1.373]])

        list_dis_constraints.append([[c4s_1,c4s_2],[1.47]])
        list_dis_constraints.append([[c4s_1,nx_2],[1.19]])
        list_dis_constraints.append([[c4s_2,nx_1],[1.17]])

        list_dis_constraints.append([[nx_1,nx_2],[0.89]])

        list_dis_constraints.append([[nx_1,c2_2],[0.75]])
        list_dis_constraints.append([[nx_1,cy_2],[0.84]])

        list_dis_constraints.append([[nx_2,c2_1],[0.59]])
        list_dis_constraints.append([[nx_2,cy_1],[0.55]])

        list_dis_constraints.append([[c2_1,c2_2],[0.47]])
        list_dis_constraints.append([[cy_1,cy_2],[0.49]])

        list_dis_constraints.append([[c2_1,cy_2],[0.64]])
        list_dis_constraints.append([[cy_1,c2_2],[0.41]])

        list_agl_constraints = []
        list_agl_constraints.append([[cy_1,c2_1,c2_2],[np.radians(73.5)]])
        list_agl_constraints.append([[c2_1,c2_2,cy_2],[np.radians(106.5)]])
        list_agl_constraints.append([[c2_2,cy_2,cy_1],[np.radians(73.5)]])
        list_agl_constraints.append([[cy_2,cy_1,c2_1],[np.radians(106.5)]])

        list_dih_constraints = []
        list_dih_constraints.append([[c2_1,cy_1,cy_2,c2_2],[np.radians(-7.7)]])
        list_dih_constraints.append([[nx_1,c2_1,cy_1,nx_2],[np.radians(178.5)]])
        list_dih_constraints.append([[nx_2,c2_2,cy_2,nx_1],[np.radians(178.5)]])
        list_dih_constraints.append([[nx_1,c2_1,cy_2,nx_2],[np.radians(174.5)]])
        list_dih_constraints.append([[nx_2,c2_2,cy_1,nx_1],[np.radians(174.5)]])

    elif name1+name2 in ["AU","UA","CG","GC"]:
        list_dis_constraints = []
        list_dis_constraints.append([[p_1,p_2],[1.806]])
        list_dis_constraints.append([[p_1,c4s_2],[1.704]])
        list_dis_constraints.append([[p_1,nx_2],[1.373]])
        list_dis_constraints.append([[p_2,c4s_1],[1.704]])
        list_dis_constraints.append([[p_2,nx_1],[1.373]])

        list_dis_constraints.append([[c4s_1,c4s_2],[1.524]])
        list_dis_constraints.append([[c4s_1,nx_2],[1.213]])
        list_dis_constraints.append([[c4s_2,nx_1],[1.213]])

        list_dis_constraints.append([[nx_1,nx_2],[0.895]])

        list_dis_constraints.append([[c2_1,c2_2],[0.417]])
        list_dis_constraints.append([[cy_1,cy_2],[0.421]])

        list_dis_constraints.append([[c2_1,cy_2],[0.482]])
        list_dis_constraints.append([[cy_1,c2_2],[0.482]])

        list_agl_constraints = []
        list_agl_constraints.append([[cy_1,c2_1,c2_2],[np.radians(90.0)]])
        list_agl_constraints.append([[c2_1,c2_2,cy_2],[np.radians(90.0)]])
        list_agl_constraints.append([[c2_2,cy_2,cy_1],[np.radians(90.0)]])
        list_agl_constraints.append([[cy_2,cy_1,c2_1],[np.radians(90.0)]])

        list_dih_constraints = []
        list_dih_constraints.append([[c2_1,cy_1,cy_2,c2_2],[np.radians(-10.4)]])
        list_dih_constraints.append([[nx_1,c2_1,cy_1,nx_2],[np.radians(178.5)]])
        list_dih_constraints.append([[nx_2,c2_2,cy_2,nx_1],[np.radians(178.5)]])
        list_dih_constraints.append([[nx_1,c2_1,cy_2,nx_2],[np.radians(174.5)]])
        list_dih_constraints.append([[nx_2,c2_2,cy_1,nx_1],[np.radians(174.5)]])
    else:
        list_dis_constraints = []
        list_dis_constraints.append([[p_1,p_2],[1.806]])
        list_dis_constraints.append([[p_1,c4s_2],[1.704]])
        list_dis_constraints.append([[p_1,nx_2],[1.373]])
        list_dis_constraints.append([[p_2,c4s_1],[1.704]])
        list_dis_constraints.append([[p_2,nx_1],[1.373]])

        list_dis_constraints.append([[c4s_1,c4s_2],[1.524]])
        list_dis_constraints.append([[c4s_1,nx_2],[1.213]])
        list_dis_constraints.append([[c4s_2,nx_1],[1.213]])

        list_dis_constraints.append([[nx_1,nx_2],[0.895]])

        list_agl_constraints = []

        list_dih_constraints = []

    list_parallel_edges_constraints = [[c2_1,cy_1,c2_2,cy_2]]

    for i in range(len(list_dih_constraints)):
        list_dih_constraints[i][1].append(1.0)
    return list_dis_constraints, list_agl_constraints, list_dih_constraints, list_parallel_edges_constraints


def repulsive_dis_constraints_for_helix(nt1,nt2): # push loops away from helices and avoid loops to be trapped in helices
    list_repulsive_dis_constraints_for_helix = []
    for name1 in ["BBP","BBC","NX","C2"]:
        idx1 = nt1[1][name1].index
        for name2 in ["BBP","BBC","NX","C2"]:
            idx2 = nt2[1][name2].index
            if nt1[0].chain.id == nt2[0].chain.id:
                if int(nt2[0].id) - int(nt1[0].id) == 1:
                    if name1 == "BBC" and name2 == "BBP":
                        continue
                if int(nt1[0].id) - int(nt2[0].id) == 1:
                    if name1 == "BBP" and name2 == "BBC":
                        continue
            list_repulsive_dis_constraints_for_helix.append([idx1,idx2])

    return list_repulsive_dis_constraints_for_helix

  
def get_torsion_constraints_for_PK_strand_in_hairpin(dict_resid_chain, dict_residues, helix_strand, hairpin_motif, left_or_right_helix_strand):
    h1, h2 = helix_strand
    hp1, hp2 = hairpin_motif[0]
    if h1-hp1 < hp2-h2:
        idx1 = hp1
        idx2 = h1
    else:
        idx1 = h2
        idx2 = hp2
    chain_ntidx = dict_resid_chain[str(idx1)]
    nt1 = dict_residues[chain_ntidx]
    chain_ntidx = dict_resid_chain[str(idx2)]
    nt2 = dict_residues[chain_ntidx]
    idx_NX1 = nt1[1]["NX"].index
    idx_BBC1 = nt1[1]["BBC"].index
    idx_NX2 = nt2[1]["NX"].index
    idx_BBC2 = nt2[1]["BBC"].index
    atoms_index = [idx_NX1,idx_BBC1,idx_BBC2,idx_NX2]

    if left_or_right_helix_strand == "left":
        torsion1 = 90.0
        torsion2 = 270.0
    elif left_or_right_helix_strand == "right":
        torsion1 = -30.0
        torsion2 = 30.0
    else:
        raise ValueError(f"Wrong option {left_or_right_helix_strand} for left_or_right_helix_strand")
    paras = [torsion1,torsion2]
    return atoms_index + paras


def get_torsion_constraints_for_PK_strand_in_2way(dict_resid_chain, dict_residues, helix_strand, twoway_motif, left_or_right_helix_strand):
    h1, h2 = helix_strand
    twoway1, twoway2 = twoway_motif[0]
    twoway3, twoway4 = twoway_motif[1]
    if h1>=twoway1 and h2<=twoway2:
        idx1 = twoway1
        idx2 = h1
    else:
        idx1 = h2
        idx2 = twoway4
    chain_ntidx = dict_resid_chain[str(idx1)]
    nt1 = dict_residues[chain_ntidx]
    chain_ntidx = dict_resid_chain[str(idx2)]
    nt2 = dict_residues[chain_ntidx]
    idx_NX1 = nt1[1]["NX"].index
    idx_BBC1 = nt1[1]["BBC"].index
    idx_NX2 = nt2[1]["NX"].index
    idx_BBC2 = nt2[1]["BBC"].index
    atoms_index = [idx_NX1,idx_BBC1,idx_BBC2,idx_NX2]

    if left_or_right_helix_strand == "left":
        torsion1 = 90.0
        torsion2 = 270.0
    elif left_or_right_helix_strand == "right":
        torsion1 = -30.0
        torsion2 = 30.0
    else:
        raise ValueError(f"Wrong option {left_or_right_helix_strand} for left_or_right_helix_strand")
    paras = [torsion1,torsion2]
    return atoms_index + paras


def get_shoulder_in_hairpin(hairpin_motif,helix_strand):
    hp1, hp2 = hairpin_motif[0]
    h1, h2 = helix_strand
    if h1 < hp1 or h2 > hp2:
        raise ValueError(f"helix strand {helix_strand} is not within hairpin {hairpin_motif[0]}")
    if (h2-h1)/(hp2-hp1) >= 0.5:
        return "right"
    if h1-hp1 < hp2-h2:
        return "left"
    else:
        return "right"


def skip_angle_constraints_for_PK_strand_in_hairpin(list_PK_helix, helix_strand, hairpin_motif):
    hp1, hp2 = hairpin_motif[0]

    list_PK_helix_strand_in_hairpin = []
    for helix in list_PK_helix:
        h1, h4 = helix[0]
        h2, h3 = helix[1]
        if h1 == h2:
            continue
        if h1>=hp1 and h2<=hp2:
            list_PK_helix_strand_in_hairpin.append([h1,h2])
        if h3>=hp1 and h4<=hp2:
            list_PK_helix_strand_in_hairpin.append([h3,h4])
    if len(list_PK_helix_strand_in_hairpin) < 2:
        return False

    shoulder1 = get_shoulder_in_hairpin(hairpin_motif,helix_strand)
    list_same_shoulder_helix_strand = [helix_strand]
    for hs in list_PK_helix_strand_in_hairpin:
        if hs == helix_strand:
            continue
        shoulder2 = get_shoulder_in_hairpin(hairpin_motif,hs)
        if shoulder1 == shoulder2:
            list_same_shoulder_helix_strand.append(hs)

    if len(list_same_shoulder_helix_strand) < 2:
        return False
    
    if shoulder1 == "left":
        for hs in list_same_shoulder_helix_strand[1:]:
            if hs[0] < helix_strand[0]:
                return True
        return False
    elif shoulder1 == "right":
        for hs in list_same_shoulder_helix_strand[1:]:
            if hs[0] > helix_strand[0]:
                return True
        return False


def skip_angle_constraints_for_PK_strand_in_2way(list_PK_helix, helix_strand, twoway_motif):
    h1, h2 = helix_strand
    twoway1, twoway2 = twoway_motif[0]
    twoway3, twoway4 = twoway_motif[1]
    if h1>=twoway1 and h2<=twoway2:
        shoulder1 = "left"
    elif h1>=twoway3 and h2<=twoway4:
        shoulder1 = "right"
    else:
        raise ValueError(f"helix strand {helix_strand} is not within internal loop {twoway_motif}")

    list_same_shoulder_helix_strand = [helix_strand]
    for helix in list_PK_helix:
        h1, h4 = helix[0]
        h2, h3 = helix[1]
        if h1 == h2:
            continue
        if shoulder1=="left":
            if h1>=twoway1 and h2<=twoway2:
                if [h1,h2] != helix_strand:
                    list_same_shoulder_helix_strand.append([h1,h2])
            elif h3>=twoway1 and h4<=twoway2:
                if [h3,h4] != helix_strand:
                    list_same_shoulder_helix_strand.append([h3,h4])
        elif shoulder1=="right":
            if h1>=twoway3 and h2<=twoway4:
                if [h1,h2] != helix_strand:
                    list_same_shoulder_helix_strand.append([h1,h2])
            elif h3>=twoway3 and h4<=twoway4:
                if [h3,h4] != helix_strand:
                    list_same_shoulder_helix_strand.append([h3,h4])

    if len(list_same_shoulder_helix_strand) < 2:
        return False
    
    if shoulder1 == "left":
        for hs in list_same_shoulder_helix_strand[1:]:
            if hs[0] < helix_strand[0]:
                return True
        return False
    elif shoulder1 == "right":
        for hs in list_same_shoulder_helix_strand[1:]:
            if hs[0] > helix_strand[0]:
                return True
        return False


def get_angle_constraints_for_PK_strand_in_hairpin(dict_resid_chain, dict_residues, helix_strand, hairpin_motif, left_or_right_helix_strand):
    h1, h2 = helix_strand
    hp1, hp2 = hairpin_motif[0]

    if (h2-h1)/(hp2-hp1) >= 0.5:
        idx1 = h2 + 1
        idx2 = h2
    else:
        if h1-hp1 < hp2-h2:
            idx1 = h1 - 1
            idx2 = h1
        else:
            idx1 = h2 + 1
            idx2 = h2
    chain_ntidx = dict_resid_chain[str(idx1)]
    nt1 = dict_residues[chain_ntidx]
    chain_ntidx = dict_resid_chain[str(idx2)]
    nt2 = dict_residues[chain_ntidx]
    idx_NX1 = nt1[1]["NX"].index
    idx_BBC1 = nt1[1]["BBC"].index
    idx_BBP1 = nt1[1]["BBP"].index
    idx_NX2 = nt2[1]["NX"].index
    idx_BBC2 = nt2[1]["BBC"].index

    if left_or_right_helix_strand == "left":
        angle_constraints1 = [idx_BBC1,idx_NX1,idx_BBC2,idx_NX2,90.,90.]
        angle_constraints2 = [idx_BBC1,idx_BBP1,idx_BBC2,idx_NX2,180.,180.]
    elif left_or_right_helix_strand == "right":
        angle_constraints1 = [idx_BBC1,idx_NX1,idx_BBC2,idx_NX2,90.,90.]
        angle_constraints2 = [idx_BBC1,idx_BBP1,idx_BBC2,idx_NX2,0.,0.]
    else:
        raise ValueError(f"Wrong option {left_or_right_helix_strand} for left_or_right_helix_strand")
    angle_constraints = [angle_constraints1,angle_constraints2]
    return angle_constraints


def get_angle_constraints_for_PK_strand_in_2way(dict_resid_chain, dict_residues, helix_strand, twoway_motif, left_or_right_helix_strand):
    h1, h2 = helix_strand
    twoway1, twoway2 = twoway_motif[0]
    twoway3, twoway4 = twoway_motif[1]
    if h1>=twoway1 and h2<=twoway2:
        idx1 = h1 - 1
        idx2 = h1
    else:
        idx1 = h2 + 1
        idx2 = h2
    chain_ntidx = dict_resid_chain[str(idx1)]
    nt1 = dict_residues[chain_ntidx]
    chain_ntidx = dict_resid_chain[str(idx2)]
    nt2 = dict_residues[chain_ntidx]
    idx_NX1 = nt1[1]["NX"].index
    idx_BBC1 = nt1[1]["BBC"].index
    idx_BBP1 = nt1[1]["BBP"].index
    idx_NX2 = nt2[1]["NX"].index
    idx_BBC2 = nt2[1]["BBC"].index

    if left_or_right_helix_strand == "left":
        angle_constraints1 = [idx_BBC1,idx_NX1,idx_BBC2,idx_NX2,90.,90.]
        angle_constraints2 = [idx_BBC1,idx_BBP1,idx_BBC2,idx_NX2,180.,180.]
    elif left_or_right_helix_strand == "right":
        angle_constraints1 = [idx_BBC1,idx_NX1,idx_BBC2,idx_NX2,90.,90.]
        angle_constraints2 = [idx_BBC1,idx_BBP1,idx_BBC2,idx_NX2,0.,0.]
    else:
        raise ValueError(f"Wrong option {left_or_right_helix_strand} for left_or_right_helix_strand")
    angle_constraints = [angle_constraints1,angle_constraints2]
    return angle_constraints


def get_angle_constraints_for_PK_helices(dict_motifs, dict_resid_chain, dict_residues):
    angle_constraints = []
    
    list_PK_helix = dict_motifs["PK_helix"]
    for helix in list_PK_helix:
        h1, h4 = helix[0]
        h2, h3 = helix[1]
        if h1 == h2:
            continue
        left_strand = [h1,h2]
        right_strand = [h3,h4]
        left_strand_loop_type = None
        left_strand_loop_motif = None
        right_strand_loop_type = None
        right_strand_loop_motif = None
        for loop_type in ["hairpin_loops","2way_loops"]:
            for motif in dict_motifs[loop_type]:
                for loop in motif:
                    ib, ie = loop
                    if h1 >= ib and h2 <= ie:
                        left_strand_loop_type = loop_type
                        left_strand_loop_motif = motif
                    if h3 >= ib and h4 <= ie:
                        right_strand_loop_type = loop_type
                        right_strand_loop_motif = motif
                    if (left_strand_loop_type is not None) and (right_strand_loop_type is not None):
                        break
                if (left_strand_loop_type is not None) and (right_strand_loop_type is not None):
                    break
            if (left_strand_loop_type is not None) and (right_strand_loop_type is not None):
                break

        inverse_direction = False
        if right_strand_loop_type == "2way_loops":
            twoway_1, twoway_2 = right_strand_loop_motif[0]
            twoway_3, twoway_4 = right_strand_loop_motif[1]
            if twoway_2 <= h1:
                inverse_direction = True

        #print(helix, left_strand_loop_type, left_strand_loop_motif, right_strand_loop_type, right_strand_loop_motif)

        if left_strand_loop_type == "hairpin_loops":
            bool_skip = skip_angle_constraints_for_PK_strand_in_hairpin(list_PK_helix, left_strand, left_strand_loop_motif)
            if not bool_skip:
                angle_constraints0 = get_angle_constraints_for_PK_strand_in_hairpin(dict_resid_chain, dict_residues, left_strand, left_strand_loop_motif, left_or_right_helix_strand="left")
                angle_constraints.extend(angle_constraints0)
        elif left_strand_loop_type == "2way_loops":
            bool_skip = skip_angle_constraints_for_PK_strand_in_2way(list_PK_helix, left_strand, left_strand_loop_motif)
            if not bool_skip:
                angle_constraints0 = get_angle_constraints_for_PK_strand_in_2way(dict_resid_chain, dict_residues, left_strand, left_strand_loop_motif, left_or_right_helix_strand="left")
                angle_constraints.extend(angle_constraints0)

        if right_strand_loop_type == "hairpin_loops":
            bool_skip = skip_angle_constraints_for_PK_strand_in_hairpin(list_PK_helix, right_strand, right_strand_loop_motif)
            if not bool_skip:
                angle_constraints0 = get_angle_constraints_for_PK_strand_in_hairpin(dict_resid_chain, dict_residues, right_strand, right_strand_loop_motif, left_or_right_helix_strand="right")
                angle_constraints.extend(angle_constraints0)
        elif right_strand_loop_type == "2way_loops":
            bool_skip = skip_angle_constraints_for_PK_strand_in_2way(list_PK_helix, right_strand, right_strand_loop_motif)
            if not bool_skip:
                if inverse_direction:
                    angle_constraints0 = get_angle_constraints_for_PK_strand_in_2way(dict_resid_chain, dict_residues, right_strand, right_strand_loop_motif, left_or_right_helix_strand="left")
                else:
                    angle_constraints0 = get_angle_constraints_for_PK_strand_in_2way(dict_resid_chain, dict_residues, right_strand, right_strand_loop_motif, left_or_right_helix_strand="right")
                angle_constraints.extend(angle_constraints0)
    #print("angle constraints:",angle_constraints,flush=True)
    return angle_constraints


def fold_helices(positions, topology, system, dict_resid_chain, dict_motifs, fold_PK):
    dict_residues = access_to_residues_by_chain_name_and_resid(topology)

    list_dis_constraints = []
    list_agl_constraints = []
    list_dih_constraints = []
    list_bp_dih_constraints = []
    list_parallel_edges_constraints = []

    list_nonPK_helix = dict_motifs["helix"]
    list_PK_helix = dict_motifs["PK_helix"]
    list_helices = list_nonPK_helix + list_PK_helix
    for helix in list_helices:
        h1 = helix[0][0] # h1 = res_index
        h2 = helix[1][0]
        h3 = helix[1][1]
        h4 = helix[0][1]
        h1 = int(h1)
        h2 = int(h2)
        h3 = int(h3)
        h4 = int(h4)

        if int(h2) - int(h1) != int(h4) - int(h3):
            print("The lengths of the two strands in the helix are not the same!")
            print (h1, h2)
            print (h3, h4)
            exit()
        
        if int(h1) == int(h2) or int(h3) == int(h4):
            continue

        if h1 > h2 or h3 > h4:
           print ("not in the right order!")
           print (h1, h2)
           print (h3, h4)
           exit()

        for i in range(h1,h2+1):
            chain_ntidx = dict_resid_chain[str(i)]
            nt = dict_residues[chain_ntidx]
            list_dis_constraints0, list_agl_constraints0, list_dih_constraints0 = get_constraints_in_nt(nt)
            list_dis_constraints.extend(list_dis_constraints0)
            list_agl_constraints.extend(list_agl_constraints0)
            list_dih_constraints.extend(list_dih_constraints0)

        for i in range(h1,h2):
            chain_ntidx = dict_resid_chain[str(i)]
            nt1 = dict_residues[chain_ntidx]
            chain_ntidx = dict_resid_chain[str(i+1)]
            nt2 = dict_residues[chain_ntidx]
            list_dis_constraints0, list_agl_constraints0, list_dih_constraints0 = get_constraints_in_consecutive_nts(nt1,nt2)
            list_dis_constraints.extend(list_dis_constraints0)
            list_agl_constraints.extend(list_agl_constraints0)
            list_dih_constraints.extend(list_dih_constraints0)

        for i in range(h1,h2-1):
            chain_ntidx = dict_resid_chain[str(i)]
            nt1 = dict_residues[chain_ntidx]
            chain_ntidx = dict_resid_chain[str(i+1)]
            nt2 = dict_residues[chain_ntidx]
            chain_ntidx = dict_resid_chain[str(i+2)]
            nt3 = dict_residues[chain_ntidx]
            list_dih_constraints0 = get_constraints_in_consecutive_three_nts(nt1,nt2,nt3)
            list_dih_constraints.extend(list_dih_constraints0)

        for i in range(h3,h4+1):
            chain_ntidx = dict_resid_chain[str(i)]
            nt = dict_residues[chain_ntidx]
            list_dis_constraints0, list_agl_constraints0, list_dih_constraints0 = get_constraints_in_nt(nt)
            list_dis_constraints.extend(list_dis_constraints0)
            list_agl_constraints.extend(list_agl_constraints0)
            list_dih_constraints.extend(list_dih_constraints0)

        for i in range(h3,h4):
            chain_ntidx = dict_resid_chain[str(i)]
            nt1 = dict_residues[chain_ntidx]
            chain_ntidx = dict_resid_chain[str(i+1)]
            nt2 = dict_residues[chain_ntidx]
            list_dis_constraints0, list_agl_constraints0, list_dih_constraints0 = get_constraints_in_consecutive_nts(nt1,nt2)
            list_dis_constraints.extend(list_dis_constraints0)
            list_agl_constraints.extend(list_agl_constraints0)
            list_dih_constraints.extend(list_dih_constraints0)

        for i in range(h3,h4-1):
            chain_ntidx = dict_resid_chain[str(i)]
            nt1 = dict_residues[chain_ntidx]
            chain_ntidx = dict_resid_chain[str(i+1)]
            nt2 = dict_residues[chain_ntidx]
            chain_ntidx = dict_resid_chain[str(i+2)]
            nt3 = dict_residues[chain_ntidx]
            list_dih_constraints0 = get_constraints_in_consecutive_three_nts(nt1,nt2,nt3)
            list_dih_constraints.extend(list_dih_constraints0)
           
        helix_left = list(range(h1,h2+1)) 
        helix_right = list(range(h4,h3-1,-1))
        for i,j in zip(helix_left,helix_right):
            chain_ntidx = dict_resid_chain[str(i)]
            nt1 = dict_residues[chain_ntidx]
            chain_ntidx = dict_resid_chain[str(j)]
            nt2 = dict_residues[chain_ntidx]
            list_dis_constraints0, list_agl_constraints0, list_bp_dih_constraints0, list_parallel_edges_constraints0 = get_constraints_in_paired_nts(nt1,nt2)
            if fold_PK:
                list_dis_constraints.extend(list_dis_constraints0)
                list_agl_constraints.extend(list_agl_constraints0)
                list_bp_dih_constraints.extend(list_bp_dih_constraints0)
            else:
                if helix not in list_PK_helix:
                    list_dis_constraints.extend(list_dis_constraints0)
                    list_agl_constraints.extend(list_agl_constraints0)
                    list_bp_dih_constraints.extend(list_bp_dih_constraints0)
                else:
                    list_agl_constraints.extend(list_agl_constraints0)
                    for i in range(len(list_bp_dih_constraints0)):
                        list_bp_dih_constraints0[i][1][-1] = 0.1 #0.2
                    list_bp_dih_constraints.extend(list_bp_dih_constraints0)
            list_parallel_edges_constraints.extend(list_parallel_edges_constraints0)

    list_dis_constraints_loop = []
    list_agl_constraints_loop = []
    list_dih_constraints_loop = []
    list_dis_constraints_PK_loop = []
    list_agl_constraints_PK_loop = []
    list_dih_constraints_PK_loop = []

    list_nt_idx_in_PK_helix = []
    for helix in dict_motifs["PK_helix"]:
        h1, h4 = helix[0]
        h2, h3 = helix[1]
        for i in range(h1,h2+1):
            list_nt_idx_in_PK_helix.append(i)
        for i in range(h3,h4+1):
            list_nt_idx_in_PK_helix.append(i)

    for key in dict_motifs.keys():
        if key == "helix" or key == "PK_helix":
            continue
        if "non_PK" in key:
            continue
        motifs = dict_motifs[key]
        for motif in motifs:
            for loop in motif:
                ib, ie = loop
                for i in range(ib,ie):
                    chain_ntidx = dict_resid_chain[str(i)]
                    nt1 = dict_residues[chain_ntidx]
                    chain_ntidx = dict_resid_chain[str(i+1)]
                    nt2 = dict_residues[chain_ntidx]
                    if nt1[0].chain.id != nt2[0].chain.id:
                        raise ValueError(f"Two continuous nucleotides are not in the same chain!")
                    if int(nt2[0].id) - int(nt1[0].id) != 1:
                        raise ValueError(f"Two nucleotides are not continuous in sequence!")
                    list_dis_constraints0, list_agl_constraints0, list_dih_constraints0 = get_constraints_in_consecutive_nts(nt1,nt2)
                    bool_nt1_in_PK_helix = False
                    bool_nt2_in_PK_helix = False
                    if i in list_nt_idx_in_PK_helix:
                        bool_nt1_in_PK_helix = True
                    if (i+1) in list_nt_idx_in_PK_helix:
                        bool_nt2_in_PK_helix = True
                    if (key == "2way_loops") and (motif not in dict_motifs["2way_loops_non_PK"]):
                        if bool_nt1_in_PK_helix and (not bool_nt2_in_PK_helix):
                            list_dis_constraints_PK_loop.extend(list_dis_constraints0[0:6]+list_dis_constraints0[7:8])
                            list_agl_constraints_PK_loop.extend(list_agl_constraints0[0:2]+list_agl_constraints0[5:7])
                            list_dih_constraints_PK_loop.extend(list_dih_constraints0[0:1]+list_dih_constraints0[2:3])
                            list_agl_constraints_loop.extend(list_agl_constraints0[2:5])
                            list_dih_constraints_loop.extend(list_dih_constraints0[1:2]+list_dih_constraints0[3:4])
                        elif bool_nt2_in_PK_helix and (not bool_nt1_in_PK_helix):
                            list_dis_constraints_PK_loop.extend(list_dis_constraints0[0:6]+list_dis_constraints0[6:7])
                            list_agl_constraints_PK_loop.extend(list_agl_constraints0[0:2]+list_agl_constraints0[2:5])
                            list_dih_constraints_PK_loop.extend(list_dih_constraints0[0:1]+list_dih_constraints0[1:2])
                            list_agl_constraints_loop.extend(list_agl_constraints0[5:7])
                            list_dih_constraints_loop.extend(list_dih_constraints0[2:3]+list_dih_constraints0[3:4])
                        else:
                            list_dis_constraints_PK_loop.extend(list_dis_constraints0)
                            list_agl_constraints_PK_loop.extend(list_agl_constraints0)
                            list_dih_constraints_PK_loop.extend(list_dih_constraints0)
                    elif (key == "hairpin_loops") and (motif not in dict_motifs["hairpin_loops_non_PK"]):
                        mid_idx = int((ie-ib)/2) + ib
                        if i == mid_idx:
                            list_dis_constraints_loop.extend(list_dis_constraints0)
                            list_agl_constraints_loop.extend(list_agl_constraints0)
                            list_dih_constraints_loop.extend(list_dih_constraints0)
                        elif bool_nt1_in_PK_helix and (not bool_nt2_in_PK_helix):
                            list_dis_constraints_PK_loop.extend(list_dis_constraints0[0:6]+list_dis_constraints0[7:8])
                            list_agl_constraints_PK_loop.extend(list_agl_constraints0[0:2]+list_agl_constraints0[5:7])
                            list_dih_constraints_PK_loop.extend(list_dih_constraints0[0:1]+list_dih_constraints0[2:3])
                            list_agl_constraints_loop.extend(list_agl_constraints0[2:5])
                            list_dih_constraints_loop.extend(list_dih_constraints0[1:2]+list_dih_constraints0[3:4])
                        elif bool_nt2_in_PK_helix and (not bool_nt1_in_PK_helix):
                            list_dis_constraints_PK_loop.extend(list_dis_constraints0[0:6]+list_dis_constraints0[6:7])
                            list_agl_constraints_PK_loop.extend(list_agl_constraints0[0:2]+list_agl_constraints0[2:5])
                            list_dih_constraints_PK_loop.extend(list_dih_constraints0[0:1]+list_dih_constraints0[1:2])
                            list_agl_constraints_loop.extend(list_agl_constraints0[5:7])
                            list_dih_constraints_loop.extend(list_dih_constraints0[2:3]+list_dih_constraints0[3:4])
                        else:
                            list_dis_constraints_PK_loop.extend(list_dis_constraints0)
                            list_agl_constraints_PK_loop.extend(list_agl_constraints0)
                            list_dih_constraints_PK_loop.extend(list_dih_constraints0)
                    elif (key == "single_loops") and (motif not in dict_motifs["single_loops_non_PK"]):
                        if bool_nt1_in_PK_helix and (not bool_nt2_in_PK_helix):
                            list_dis_constraints_PK_loop.extend(list_dis_constraints0[0:6]+list_dis_constraints0[7:8])
                            list_agl_constraints_PK_loop.extend(list_agl_constraints0[0:2]+list_agl_constraints0[5:7])
                            list_dih_constraints_PK_loop.extend(list_dih_constraints0[0:1]+list_dih_constraints0[2:3])
                            list_agl_constraints_loop.extend(list_agl_constraints0[2:5])
                            list_dih_constraints_loop.extend(list_dih_constraints0[1:2]+list_dih_constraints0[3:4])
                        elif bool_nt2_in_PK_helix and (not bool_nt1_in_PK_helix):
                            list_dis_constraints_PK_loop.extend(list_dis_constraints0[0:6]+list_dis_constraints0[6:7])
                            list_agl_constraints_PK_loop.extend(list_agl_constraints0[0:2]+list_agl_constraints0[2:5])
                            list_dih_constraints_PK_loop.extend(list_dih_constraints0[0:1]+list_dih_constraints0[1:2])
                            list_agl_constraints_loop.extend(list_agl_constraints0[5:7])
                            list_dih_constraints_loop.extend(list_dih_constraints0[2:3]+list_dih_constraints0[3:4])
                        else:
                            list_dis_constraints_PK_loop.extend(list_dis_constraints0)
                            list_agl_constraints_PK_loop.extend(list_agl_constraints0)
                            list_dih_constraints_PK_loop.extend(list_dih_constraints0)
                    elif (key == "3way_loops") and (motif not in dict_motifs["3way_loops_non_PK"]):
                        if bool_nt1_in_PK_helix and (not bool_nt2_in_PK_helix):
                            list_dis_constraints_PK_loop.extend(list_dis_constraints0[0:6]+list_dis_constraints0[7:8])
                            list_agl_constraints_PK_loop.extend(list_agl_constraints0[0:2]+list_agl_constraints0[5:7])
                            list_dih_constraints_PK_loop.extend(list_dih_constraints0[0:1]+list_dih_constraints0[2:3])
                            list_agl_constraints_loop.extend(list_agl_constraints0[2:5])
                            list_dih_constraints_loop.extend(list_dih_constraints0[1:2]+list_dih_constraints0[3:4])
                        elif bool_nt2_in_PK_helix and (not bool_nt1_in_PK_helix):
                            list_dis_constraints_PK_loop.extend(list_dis_constraints0[0:6]+list_dis_constraints0[6:7])
                            list_agl_constraints_PK_loop.extend(list_agl_constraints0[0:2]+list_agl_constraints0[2:5])
                            list_dih_constraints_PK_loop.extend(list_dih_constraints0[0:1]+list_dih_constraints0[1:2])
                            list_agl_constraints_loop.extend(list_agl_constraints0[5:7])
                            list_dih_constraints_loop.extend(list_dih_constraints0[2:3]+list_dih_constraints0[3:4])
                        else:
                            list_dis_constraints_PK_loop.extend(list_dis_constraints0)
                            list_agl_constraints_PK_loop.extend(list_agl_constraints0)
                            list_dih_constraints_PK_loop.extend(list_dih_constraints0)
                    elif (key == "4way_loops") and (motif not in dict_motifs["4way_loops_non_PK"]):
                        if bool_nt1_in_PK_helix and (not bool_nt2_in_PK_helix):
                            list_dis_constraints_PK_loop.extend(list_dis_constraints0[0:6]+list_dis_constraints0[7:8])
                            list_agl_constraints_PK_loop.extend(list_agl_constraints0[0:2]+list_agl_constraints0[5:7])
                            list_dih_constraints_PK_loop.extend(list_dih_constraints0[0:1]+list_dih_constraints0[2:3])
                            list_agl_constraints_loop.extend(list_agl_constraints0[2:5])
                            list_dih_constraints_loop.extend(list_dih_constraints0[1:2]+list_dih_constraints0[3:4])
                        elif bool_nt2_in_PK_helix and (not bool_nt1_in_PK_helix):
                            list_dis_constraints_PK_loop.extend(list_dis_constraints0[0:6]+list_dis_constraints0[6:7])
                            list_agl_constraints_PK_loop.extend(list_agl_constraints0[0:2]+list_agl_constraints0[2:5])
                            list_dih_constraints_PK_loop.extend(list_dih_constraints0[0:1]+list_dih_constraints0[1:2])
                            list_agl_constraints_loop.extend(list_agl_constraints0[5:7])
                            list_dih_constraints_loop.extend(list_dih_constraints0[2:3]+list_dih_constraints0[3:4])
                        else:
                            list_dis_constraints_PK_loop.extend(list_dis_constraints0)
                            list_agl_constraints_PK_loop.extend(list_agl_constraints0)
                            list_dih_constraints_PK_loop.extend(list_dih_constraints0)
                    else:
                        list_dis_constraints_loop.extend(list_dis_constraints0)
                        list_agl_constraints_loop.extend(list_agl_constraints0)
                        list_dih_constraints_loop.extend(list_dih_constraints0)

    list_repulsive_dis_constraints_for_helix = []
    for helix in list_helices:
        h1 = helix[0][0] # h1 = res_index
        h2 = helix[1][0]
        h3 = helix[1][1]
        h4 = helix[0][1]
        h1 = int(h1)
        h2 = int(h2)
        h3 = int(h3)
        h4 = int(h4)
        list_nt_idx_in_helix = list(range(h1,h2+1)) + list(range(h3,h4+1))
        for i in list_nt_idx_in_helix:
            chain_ntidx = dict_resid_chain[str(i)]
            nt1 = dict_residues[chain_ntidx]
            for key in dict_motifs:
                if key in ["helix","PK_helix"]:
                    continue
                if "non_PK" in key:
                    continue
                motifs = dict_motifs[key]
                for motif in motifs:
                    for loop in motif:
                        ib, ie = loop
                        for j in range(ib,ie+1):
                            if j in list_nt_idx_in_helix:
                                continue
                            chain_ntidx = dict_resid_chain[str(j)]
                            nt2 = dict_residues[chain_ntidx]
                            list_repulsive_dis_constraints_for_helix0 = repulsive_dis_constraints_for_helix(nt1,nt2)
                            list_repulsive_dis_constraints_for_helix.extend(list_repulsive_dis_constraints_for_helix0)
    for i in range(len(list_helices)-1):
        helix1 = list_helices[i]
        h1 = helix1[0][0] # h1 = res_index
        h2 = helix1[1][0]
        h3 = helix1[1][1]
        h4 = helix1[0][1]
        h1 = int(h1)
        h2 = int(h2)
        h3 = int(h3)
        h4 = int(h4)
        list_nt_idx_in_helix1 = list(range(h1,h2+1)) + list(range(h3,h4+1))
        for idx1 in list_nt_idx_in_helix1:
            chain_ntidx = dict_resid_chain[str(idx1)]
            nt1 = dict_residues[chain_ntidx]
            for j in range(i+1,len(list_helices)):
                helix2 = list_helices[j]
                ha = helix2[0][0] # h1 = res_index
                hb = helix2[1][0]
                hc = helix2[1][1]
                hd = helix2[0][1]
                ha = int(ha)
                hb = int(hb)
                hc = int(hc)
                hd = int(hd)
                list_nt_idx_in_helix2 = list(range(ha,hb+1)) + list(range(hc,hd+1))
                for idx2 in list_nt_idx_in_helix2:
                    chain_ntidx = dict_resid_chain[str(idx2)]
                    nt2 = dict_residues[chain_ntidx]
                    list_repulsive_dis_constraints_for_helix0 = repulsive_dis_constraints_for_helix(nt1,nt2)
                    list_repulsive_dis_constraints_for_helix.extend(list_repulsive_dis_constraints_for_helix0)
    
    list_agl_constraints_for_PK_hairpin = []
    for hp in dict_motifs["hairpin_loops"]:
        if hp in dict_motifs["hairpin_loops_non_PK"]:
            continue
        list_agl_constraints_for_PK_hairpin0 = get_constraints_in_PK_hairpin(dict_resid_chain, dict_residues, hp[0]) # avoid PK hairpin loops to trap in helices
        list_agl_constraints_for_PK_hairpin.extend(list_agl_constraints_for_PK_hairpin0)

    for motif in dict_motifs["2way_loops"]:
        if motif in dict_motifs["2way_loops_non_PK"]:
            continue
        loop1, loop2 = motif
        list_agl_constraints_for_PK_hairpin0 = get_constraints_in_PK_hairpin(dict_resid_chain, dict_residues, loop1) # avoid PK internal loops to trap in helices
        list_agl_constraints_for_PK_hairpin.extend(list_agl_constraints_for_PK_hairpin0)
        list_agl_constraints_for_PK_hairpin0 = get_constraints_in_PK_hairpin(dict_resid_chain, dict_residues, loop2) # avoid PK internal loops to trap in helices
        list_agl_constraints_for_PK_hairpin.extend(list_agl_constraints_for_PK_hairpin0)

    list_agl_constraints_for_unbent_helix = get_constraints_for_unbent_helix(dict_resid_chain, dict_residues, dict_motifs)

    list_constrained_atoms = []
    for helix in dict_motifs["helix"]:
        h1, h4 = helix[0]
        h2, h3 = helix[1]
        if h1 == h2:
            continue

        bool_junction_helix = False
        for motif in dict_motifs["3way_loops"] + dict_motifs["4way_loops"]:
            for loop in motif:
                ib, ie = loop
                if ib in [h1,h2,h3,h4] or ie in [h1,h2,h3,h4]:
                    bool_junction_helix = True
                    break
            if bool_junction_helix:
                break
        if not bool_junction_helix:
            continue

        for i in range(h1,h2+1):
            chain_ntidx = dict_resid_chain[str(i)]
            nt = dict_residues[chain_ntidx]
            c4s_idx = nt[1]["BBC"].index
            coord_c4s = positions[c4s_idx].value_in_unit(unit.nanometer)
            constrained_atom = [c4s_idx,coord_c4s]
            list_constrained_atoms.append(constrained_atom)

        for i in range(h3,h4+1):
            chain_ntidx = dict_resid_chain[str(i)]
            nt = dict_residues[chain_ntidx]
            c4s_idx = nt[1]["BBC"].index
            coord_c4s = positions[c4s_idx].value_in_unit(unit.nanometer)
            constrained_atom = [c4s_idx,coord_c4s]
            list_constrained_atoms.append(constrained_atom)


    list_angle_constraints_for_PK_helix = get_angle_constraints_for_PK_helices(dict_motifs, dict_resid_chain, dict_residues)

    list_bond_constraints = get_all_bonds(dict_resid_chain,dict_residues)

    bond_constraint_energy = f"500000*(dis-dis0)^2; dis=distance(p1,p2)"
    bond_force = mm.CustomCompoundBondForce(2, bond_constraint_energy)
    bond_force.addPerBondParameter("dis0")

    dis_constraint_energy = f"min(kdis * (dis - dis0)^2,1000) + min(kdis2 * (dis - dis0)^2,10000); dis=distance(p1,p2)"
    dis_force = mm.CustomCompoundBondForce(2, dis_constraint_energy)
    dis_force.addGlobalParameter("kdis",0)
    dis_force.addGlobalParameter("kdis2",0)
    dis_force.addPerBondParameter("dis0")

    repulsive_dis_constraint_energy = f"step(1.0-dis) * step(dis-0.75) * 50.0 * (dis - 1.0)^2 + step(0.75-dis) * step(dis-0.5) * 500.0 * (dis - 1.0)^2 + step(0.5-dis) * 5000.0 * (dis - 1.0)^2; dis=distance(p1,p2)"
    repulsive_dis_force = mm.CustomCompoundBondForce(2, repulsive_dis_constraint_energy)

    agl_constraint_energy = f"step(20000-energy)*energy + step(energy-20000)*0.5*energy + step(0.3491-agl)*5000.0*(agl-agl0)^2; energy=kagl*(agl - agl0)^2; agl=angle(p1,p2,p3)"
    agl_force = mm.CustomCompoundBondForce(3, agl_constraint_energy)
    agl_force.addGlobalParameter("kagl",0)
    agl_force.addPerBondParameter("agl0")

    dih_constraint_energy = f"step(20000-energy)*energy + step(energy-20000)*0.5*energy; energy=step(sin(agl1)-0.087155743)*step(sin(agl2)-0.087155743) * kdih * min(deltadih,2*pi-deltadih)^2; deltadih=abs(dih-dih0); dih=dihedral(p1,p2,p3,p4); pi=3.1415926535; agl1=angle(p1,p2,p3); agl2=angle(p2,p3,p4)"
    dih_force = mm.CustomCompoundBondForce(4, dih_constraint_energy)
    dih_force.addGlobalParameter("kdih",0)
    dih_force.addPerBondParameter("dih0")

    bp_dih_constraint_energy = f"min(step(sin(agl1)-0.087155743)*step(sin(agl2)-0.087155743) * kbpdih * coeff * min(deltadih,2*pi-deltadih)^2,5000) + step(sin(agl1)-0.087155743)*step(sin(agl2)-0.087155743) * 10.0 * coeff * min(deltadih,2*pi-deltadih)^2; deltadih=abs(dih-dih0); dih=dihedral(p1,p2,p3,p4); pi=3.1415926535; agl1=angle(p1,p2,p3); agl2=angle(p2,p3,p4)"
    bp_dih_force = mm.CustomCompoundBondForce(4, bp_dih_constraint_energy)
    bp_dih_force.addGlobalParameter("kbpdih",0)
    bp_dih_force.addPerBondParameter("dih0")
    bp_dih_force.addPerBondParameter("coeff") # 1.0 for non-PK helix, 0.2 for PK helix when not folding PK helix

    loop_dis_constraint_energy = f"kloop * kdis* (dis - dis0)^2; dis=distance(p1,p2)" # constrain all nucleotides to adpot helical distances
    loop_dis_force = mm.CustomCompoundBondForce(2, loop_dis_constraint_energy)
    loop_dis_force.addGlobalParameter("kloop",1.0)
    loop_dis_force.addPerBondParameter("dis0")
    loop_dis_force.addPerBondParameter("kdis")

    loop_agl_constraint_energy = f"kloop * (step(10000-Eagl)*Eagl + step(Eagl-10000)*0.5*Eagl) + step(0.3491-agl)*2000.0*(agl-agl0)^2; Eagl=kagl*(agl-agl0)^2; agl=angle(p1,p2,p3)" # constrain all nucleotides in helix and loop to adopt helical angles
    loop_agl_force = mm.CustomCompoundBondForce(3, loop_agl_constraint_energy)
    loop_agl_force.addGlobalParameter("kloop",1.0)
    loop_agl_force.addPerBondParameter("agl0")
    loop_agl_force.addPerBondParameter("kagl")

    loop_dih_constraint_energy = f"kloop * (step(10000-Edih)*Edih + step(Edih-10000)*0.5*Edih); Edih=kdih * step(sin(agl1)-0.087155743)*step(sin(agl2)-0.087155743) * min(deltadih,2*pi-deltadih)^2; deltadih=abs(dih-dih0); dih=dihedral(p1,p2,p3,p4); pi=3.1415926535; agl1=angle(p1,p2,p3); agl2=angle(p2,p3,p4)" # constrain all nucleotides in helix and loop to adopt helical dihedral angles
    loop_dih_force = mm.CustomCompoundBondForce(4, loop_dih_constraint_energy)
    loop_dih_force.addGlobalParameter("kloop",1.0)
    loop_dih_force.addPerBondParameter("dih0")
    loop_dih_force.addPerBondParameter("kdih")

    PK_loop_dis_constraint_energy = f"kPKloop * kdis * (dis - dis0)^2; dis=distance(p1,p2)" # constrain all nucleotides to adpot helical distances
    PK_loop_dis_force = mm.CustomCompoundBondForce(2, PK_loop_dis_constraint_energy)
    PK_loop_dis_force.addGlobalParameter("kPKloop",1.0)
    PK_loop_dis_force.addPerBondParameter("dis0")
    PK_loop_dis_force.addPerBondParameter("kdis")

    PK_loop_agl_constraint_energy = f"kPKloop * (step(10000-Eagl)*Eagl + step(Eagl-10000)*0.5*Eagl) + step(0.3491-agl)*2000.0*(agl-agl0)^2; Eagl=kagl*(agl-agl0)^2; agl=angle(p1,p2,p3)" # constrain all nucleotides in helix and loop to adopt helical angles
    PK_loop_agl_force = mm.CustomCompoundBondForce(3, PK_loop_agl_constraint_energy)
    PK_loop_agl_force.addGlobalParameter("kPKloop",1.0)
    PK_loop_agl_force.addPerBondParameter("agl0")
    PK_loop_agl_force.addPerBondParameter("kagl")

    PK_loop_dih_constraint_energy = f"kPKloop * (step(10000-Edih)*Edih + step(Edih-10000)*0.5*Edih); Edih=kdih * step(sin(agl1)-0.087155743)*step(sin(agl2)-0.087155743) * min(deltadih,2*pi-deltadih)^2; deltadih=abs(dih-dih0); dih=dihedral(p1,p2,p3,p4); pi=3.1415926535; agl1=angle(p1,p2,p3); agl2=angle(p2,p3,p4)" # constrain all nucleotides in helix and loop to adopt helical dihedral angles
    PK_loop_dih_force = mm.CustomCompoundBondForce(4, PK_loop_dih_constraint_energy)
    PK_loop_dih_force.addGlobalParameter("kPKloop",1.0)
    PK_loop_dih_force.addPerBondParameter("dih0")
    PK_loop_dih_force.addPerBondParameter("kdih")

    parallel_edges_constraint_energy = f"kprledge * ( (cosbeta-1.0)^2 + (angle(p1,p2,p4)-1.5707963)^2 + (angle(p1,p3,p4)-1.5707963)^2 ); cosbeta=(a*d+b*e+c*f)/dis1/dis2; dis1=distance(p1,p2); dis2=distance(p3,p4); a=x2-x1; b=y2-y1; c=z2-z1; d=x4-x3; e=y4-y3; f=z4-z3;"
    parallel_edges_force = mm.CustomCompoundBondForce(4, parallel_edges_constraint_energy)
    parallel_edges_force.addGlobalParameter("kprledge",0)

    agl_constraint_energy_for_PK_hairpin = f"kagl_PK_hp * step(minangle-triplePangle) * (triplePangle-2.0*minangle)^2; triplePangle=angle(p1,p2,p3); minangle=1.570796327;" # minangle = 90 degree
    agl_constraint_force_for_PK_hairpin = mm.CustomCompoundBondForce(3, agl_constraint_energy_for_PK_hairpin)
    agl_constraint_force_for_PK_hairpin.addGlobalParameter("kagl_PK_hp",500.0)

    agl_constraint_energy_for_unbent_helix = f"kagl_unbent_helix * step(minangle-triplePangle) * (triplePangle-2.0*minangle)^2; triplePangle=angle(p1,p2,p3); minangle=1.570796327;" # minangle = 90 degree
    agl_constraint_force_for_unbent_helix = mm.CustomCompoundBondForce(3, agl_constraint_energy_for_unbent_helix)
    agl_constraint_force_for_unbent_helix.addGlobalParameter("kagl_unbent_helix",500.0)

    for bond_constraint in list_bond_constraints:
        atoms_index = bond_constraint[0]
        paras = bond_constraint[1]
        bond_force.addBond(atoms_index,paras)

    for dis_constraint in list_dis_constraints:
        atoms_index = dis_constraint[0]
        paras = dis_constraint[1]
        dis_force.addBond(atoms_index,paras)
    for dis_constraint in list_repulsive_dis_constraints_for_helix:
        atoms_index = dis_constraint
        paras = []
        repulsive_dis_force.addBond(atoms_index,paras)
    for agl_constraint in list_agl_constraints:
        atoms_index = agl_constraint[0]
        paras = agl_constraint[1]
        agl_force.addBond(atoms_index,paras)
    for dih_constraint in list_dih_constraints:
        atoms_index = dih_constraint[0]
        paras = dih_constraint[1]
        dih_force.addBond(atoms_index,paras)
    for bp_dih_constraint in list_bp_dih_constraints:
        atoms_index = bp_dih_constraint[0]
        paras = bp_dih_constraint[1]
        bp_dih_force.addBond(atoms_index,paras)

    for dis_constraint in list_dis_constraints_loop:
        atoms_index = dis_constraint[0]
        paras = dis_constraint[1]
        paras.append(1000.0)
        loop_dis_force.addBond(atoms_index,paras)
    for agl_constraint in list_agl_constraints_loop:
        atoms_index = agl_constraint[0]
        paras = agl_constraint[1]
        paras.append(1000.0)
        loop_agl_force.addBond(atoms_index,paras)
    for dih_constraint in list_dih_constraints_loop:
        atoms_index = dih_constraint[0]
        paras = dih_constraint[1]
        paras.append(1000.0)
        loop_dih_force.addBond(atoms_index,paras)

    for dis_constraint in list_dis_constraints_PK_loop:
        atoms_index = dis_constraint[0]
        paras = dis_constraint[1]
        if fold_PK:
            paras.append(10000.0)
        else:
            paras.append(10000.0)
        PK_loop_dis_force.addBond(atoms_index,paras)
    for agl_constraint in list_agl_constraints_PK_loop:
        atoms_index = agl_constraint[0]
        paras = agl_constraint[1]
        if fold_PK:
            paras.append(10000.0)
        else:
            paras.append(10000.0)
        PK_loop_agl_force.addBond(atoms_index,paras)
    for dih_constraint in list_dih_constraints_PK_loop:
        atoms_index = dih_constraint[0]
        paras = dih_constraint[1]
        if fold_PK:
            paras.append(10000.0)
        else:
            paras.append(10000.0)
        PK_loop_dih_force.addBond(atoms_index,paras)

    for atoms_index in list_parallel_edges_constraints:
        parallel_edges_force.addBond(atoms_index,[])

    for atoms_index in list_agl_constraints_for_PK_hairpin:
        agl_constraint_force_for_PK_hairpin.addBond(atoms_index,[])

    for atoms_index in list_agl_constraints_for_unbent_helix:
        agl_constraint_force_for_unbent_helix.addBond(atoms_index,[])

    bond_force.setForceGroup(15)
    system.addForce(bond_force)

    dis_force.setForceGroup(15)
    system.addForce(dis_force)

    repulsive_dis_force.setForceGroup(15)
    system.addForce(repulsive_dis_force)

    agl_force.setForceGroup(16)
    system.addForce(agl_force)

    dih_force.setForceGroup(17)
    system.addForce(dih_force)

    bp_dih_force.setForceGroup(18)
    system.addForce(bp_dih_force)

    parallel_edges_force.setForceGroup(18)
    system.addForce(parallel_edges_force)

    loop_dis_force.setForceGroup(15)
    system.addForce(loop_dis_force)

    loop_agl_force.setForceGroup(16)
    system.addForce(loop_agl_force)

    loop_dih_force.setForceGroup(17)
    system.addForce(loop_dih_force)

    PK_loop_dis_force.setForceGroup(15)
    system.addForce(PK_loop_dis_force)

    PK_loop_agl_force.setForceGroup(16)
    system.addForce(PK_loop_agl_force)

    PK_loop_dih_force.setForceGroup(17)
    system.addForce(PK_loop_dih_force)

    agl_constraint_force_for_PK_hairpin.setForceGroup(16)
    agl_constraint_force_for_PK_hairpin.setForceGroup(16)
    system.addForce(agl_constraint_force_for_PK_hairpin)

    agl_constraint_force_for_unbent_helix.setForceGroup(16)
    system.addForce(agl_constraint_force_for_unbent_helix)

    if list_angle_constraints_for_PK_helix:
        system = add_angle_constraint_force_between_four_atoms(system, list_angle_constraints_for_PK_helix)

    system = add_atom_position_constraint_force(system,list_constrained_atoms)
    return system
