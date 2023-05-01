import simtk.openmm as mm
import simtk.unit as unit
import numpy as np
import copy
import itertools

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


def load_bp_stk_paras(paras_file):
    with open(paras_file) as f:
        lines = f.read().splitlines()
    paras = []
    for line in lines:
        line = line.split()[1:]
        paras.extend(line)
    paras = list(map(float,paras))
    return paras


def add_base_pairing_stacking_force(RNAJP_HOME,system, dict_atoms_index, dict_bp_stk_strength):
    dict_paras = {}
    for nt in ["AA","AG","AC","AU","GG","GC","GU","CC","CU","UU","aG","aC","aU","gC","gU","cU","aa","ag","ac","au","gg","gc","gu","cc","cu","uu"]:
        bp_strength = dict_bp_stk_strength["bp_"+nt.upper()]
        paras_bp = [bp_strength]
        for i in range(9):
            paras_file = f"{RNAJP_HOME}/bp_stk_paras/bp_{nt.upper()}-{i}.txt"
            paras_bp.extend(load_bp_stk_paras(paras_file))
        
        stk_strength = dict_bp_stk_strength["stk_"+nt]
        paras_stk = [stk_strength]
        for i in range(9):
            paras_file = f"{RNAJP_HOME}/bp_stk_paras/stk_{nt}-{i}.txt"
            paras_stk.extend(load_bp_stk_paras(paras_file))
        paras = paras_bp + paras_stk
        dict_paras[nt] = paras

    bp_logprob = []
    for i in range(1,10):
        logprob = []
        for j in range(1,6):
            func = f"amplitude{j}_bp{i} / (sqrt(6.283185307)*sigma{j}_bp{i}) * exp(-(d{i}-center{j}_bp{i})^2/(2*sigma{j}_bp{i}^2))"
            logprob.append(func)
        logprob.append(f"(c_bp{i})")
        logprob = "+".join(logprob)
        bp_logprob.append(logprob)
    bp_logprob = "+".join(bp_logprob)

    stk_logprob = []
    for i in range(1,10):
        logprob = []
        for j in range(1,6):
            func = f"amplitude{j}_stk{i} / (sqrt(6.283185307)*sigma{j}_stk{i}) * exp(-(d{i}-center{j}_stk{i})^2/(2*sigma{j}_stk{i}^2))"
            logprob.append(func)
        logprob.append(f"(c_stk{i})")
        logprob = "+".join(logprob)
        stk_logprob.append(logprob)
    stk_logprob = "+".join(stk_logprob)

    wt = 500
    bp_angle_penalty = f"{wt}*(abs(cosbeta)-1)^2"
    threshold = []
    threshold.append(f"step(1.0-d1)*step(1.0-d3)*step(1.0-d7)*step(1.0-d9)*step(d1-0.3)*step(d2-0.3)*step(d3-0.3)*step(d4-0.3)*step(d5-0.3)*step(d6-0.3)*step(d7-0.3)*step(d8-0.3)*step(d9-0.3)")
    threshold.append(f"step(abs(cosbeta)-0.707106781)") # less than 45 degreee between plane norms
    threshold = "*".join(threshold)
    bp_energy = f"{threshold}*(-wt_bp*RT0*(max(({bp_logprob})-13,0)) + wt_bp*{bp_angle_penalty})"

    wt = 500
    stk_angle_penalty = f"{wt}*(abs(cosbeta)-1)^2"
    threshold = []
    threshold.append(f"step(1.0-d1)*step(1.0-d3)*step(1.0-d7)*step(1.0-d9)*step(d1-0.3)*step(d2-0.3)*step(d3-0.3)*step(d4-0.3)*step(d5-0.3)*step(d6-0.3)*step(d7-0.3)*step(d8-0.3)*step(d9-0.3)")
    threshold.append(f"step(abs(cosbeta)-0.707106781)") # less than 45 degree between plane norms
    threshold = "*".join(threshold)
    stk_energy = f"{threshold}*(-wt_stk*RT0*(max(({stk_logprob})-13,0)) + wt_stk*{stk_angle_penalty})"

    bp_stk_energy = f"kbpstk_global * min({bp_energy},{stk_energy}) ; d1=distance(p1,p4); d2=distance(p1,p5); d3=distance(p1,p6); d4=distance(p2,p4); d5=distance(p2,p5); d6=distance(p2,p6); d7=distance(p3,p4); d8=distance(p3,p5); d9=distance(p3,p6); theta1=dihedral(p1,p2,p3,p4); theta2=dihedral(p2,p3,p4,p5); theta3=dihedral(p3,p4,p5,p6); cosbeta=(dot_r1R1*dot_r2R2-dot_r1R2*dot_r2R1)/(distance(p1,p2)*distance(p1,p3)*distance(p4,p5)*distance(p4,p6)*sin(angle(p2,p1,p3))*sin(angle(p5,p4,p6))); dot_r1R1=a*g+b*h+c*i; dot_r2R2=d*j+e*k+f*l; dot_r1R2=a*j+b*k+c*l; dot_r2R1=d*g+e*h+f*i; a=x2-x1; b=y2-y1; c=z2-z1; d=x3-x1; e=y3-y1; f=z3-z1; g=x5-x4; h=y5-y4; i=z5-z4; j=x6-x4; k=y6-y4; l=z6-z4; RT0=2.494"

    bp_stk_force = mm.CustomCompoundBondForce(6, bp_stk_energy)
    bp_stk_force.addGlobalParameter("kbpstk_global",1.0)

    bp_stk_force.addPerBondParameter(f"wt_bp")
    for i in range(1,10):
        for j in range(1,6):
            bp_stk_force.addPerBondParameter(f"amplitude{j}_bp{i}")
        for j in range(1,6):
            bp_stk_force.addPerBondParameter(f"center{j}_bp{i}")
        for j in range(1,6):
            bp_stk_force.addPerBondParameter(f"sigma{j}_bp{i}")
        bp_stk_force.addPerBondParameter(f"c_bp{i}")

    bp_stk_force.addPerBondParameter(f"wt_stk")
    for i in range(1,10):
        for j in range(1,6):
            bp_stk_force.addPerBondParameter(f"amplitude{j}_stk{i}")
        for j in range(1,6):
            bp_stk_force.addPerBondParameter(f"center{j}_stk{i}")
        for j in range(1,6):
            bp_stk_force.addPerBondParameter(f"sigma{j}_stk{i}")
        bp_stk_force.addPerBondParameter(f"c_stk{i}")

    for key in dict_atoms_index:
        for atoms_index in dict_atoms_index[key]:
            paras = copy.deepcopy(dict_paras[key])
            if atoms_index[-1][0] == "0": # no base pairing
                paras[0] = 0.0
            elif atoms_index[-1][0] == "2": # canonical base pairing between different chains
                paras[0] *= 1.2

            if atoms_index[-1][1:] == "AA":
                paras[145] *= 1.2
            bp_stk_force.addBond(atoms_index[0:-1],paras)
        
    bp_stk_force.setForceGroup(6)
    system.addForce(bp_stk_force)
    return system


def add_base_pairing_stacking_force_split(RNAJP_HOME,system, dict_atoms_index, dict_bp_stk_strength):
    dict_paras = {}
    for nt in ["AA","AG","AC","AU","GG","GC","GU","CC","CU","UU","aG","aC","aU","gC","gU","cU","aa","ag","ac","au","gg","gc","gu","cc","cu","uu"]:
        bp_strength = dict_bp_stk_strength["bp_"+nt.upper()]
        paras_bp = [bp_strength]
        for i in range(9):
            paras_file = f"{RNAJP_HOME}/bp_stk_paras/bp_{nt.upper()}-{i}.txt"
            paras_bp.extend(load_bp_stk_paras(paras_file))
        
        stk_strength = dict_bp_stk_strength["stk_"+nt]
        paras_stk = [stk_strength]
        for i in range(9):
            paras_file = f"{RNAJP_HOME}/bp_stk_paras/stk_{nt}-{i}.txt"
            paras_stk.extend(load_bp_stk_paras(paras_file))
        paras = paras_bp + paras_stk
        dict_paras[nt] = paras

    bp_logprob = []
    for i in range(1,10):
        logprob = []
        for j in range(1,6):
            func = f"amplitude{j}_bp{i} / (sqrt(6.283185307)*sigma{j}_bp{i}) * exp(-(d{i}-center{j}_bp{i})^2/(2*sigma{j}_bp{i}^2))"
            logprob.append(func)
        logprob.append(f"(c_bp{i})")
        logprob = "+".join(logprob)
        bp_logprob.append(logprob)
    bp_logprob = "+".join(bp_logprob)

    stk_logprob = []
    for i in range(1,10):
        logprob = []
        for j in range(1,6):
            func = f"amplitude{j}_stk{i} / (sqrt(6.283185307)*sigma{j}_stk{i}) * exp(-(d{i}-center{j}_stk{i})^2/(2*sigma{j}_stk{i}^2))"
            logprob.append(func)
        logprob.append(f"(c_stk{i})")
        logprob = "+".join(logprob)
        stk_logprob.append(logprob)
    stk_logprob = "+".join(stk_logprob)

    wt = 500
    bp_angle_penalty = f"{wt}*(abs(cosbeta)-1)^2"
    threshold = []
    threshold.append(f"step(1.0-d1)*step(1.0-d3)*step(1.0-d7)*step(1.0-d9)*step(d1-0.3)*step(d2-0.3)*step(d3-0.3)*step(d4-0.3)*step(d5-0.3)*step(d6-0.3)*step(d7-0.3)*step(d8-0.3)*step(d9-0.3)")
    threshold.append(f"step(abs(cosbeta)-0.707106781)") # less than 45 degreee between plane norms
    threshold = "*".join(threshold)
    bp_energy = f"{threshold}*(-wt_bp*RT0*(max(({bp_logprob})-13,0)) + wt_bp*{bp_angle_penalty})"

    wt = 500
    stk_angle_penalty = f"{wt}*(abs(cosbeta)-1)^2"
    threshold = []
    threshold.append(f"step(1.0-d1)*step(1.0-d3)*step(1.0-d7)*step(1.0-d9)*step(d1-0.3)*step(d2-0.3)*step(d3-0.3)*step(d4-0.3)*step(d5-0.3)*step(d6-0.3)*step(d7-0.3)*step(d8-0.3)*step(d9-0.3)")
    threshold.append(f"step(abs(cosbeta)-0.707106781)") # less than 45 degree between plane norms
    threshold = "*".join(threshold)
    stk_energy = f"{threshold}*(-wt_stk*RT0*(max(({stk_logprob})-13,0)) + wt_stk*{stk_angle_penalty})"

    bp_energy_func = f"kbpstk_global * step(-({bp_energy})+({stk_energy})) * ({bp_energy}) ; d1=distance(p1,p4); d2=distance(p1,p5); d3=distance(p1,p6); d4=distance(p2,p4); d5=distance(p2,p5); d6=distance(p2,p6); d7=distance(p3,p4); d8=distance(p3,p5); d9=distance(p3,p6); theta1=dihedral(p1,p2,p3,p4); theta2=dihedral(p2,p3,p4,p5); theta3=dihedral(p3,p4,p5,p6); cosbeta=(dot_r1R1*dot_r2R2-dot_r1R2*dot_r2R1)/(distance(p1,p2)*distance(p1,p3)*distance(p4,p5)*distance(p4,p6)*sin(angle(p2,p1,p3))*sin(angle(p5,p4,p6))); dot_r1R1=a*g+b*h+c*i; dot_r2R2=d*j+e*k+f*l; dot_r1R2=a*j+b*k+c*l; dot_r2R1=d*g+e*h+f*i; a=x2-x1; b=y2-y1; c=z2-z1; d=x3-x1; e=y3-y1; f=z3-z1; g=x5-x4; h=y5-y4; i=z5-z4; j=x6-x4; k=y6-y4; l=z6-z4; RT0=2.494"
    stk_energy_func = f"kbpstk_global * step(({bp_energy})-({stk_energy})) * ({stk_energy}) ; d1=distance(p1,p4); d2=distance(p1,p5); d3=distance(p1,p6); d4=distance(p2,p4); d5=distance(p2,p5); d6=distance(p2,p6); d7=distance(p3,p4); d8=distance(p3,p5); d9=distance(p3,p6); theta1=dihedral(p1,p2,p3,p4); theta2=dihedral(p2,p3,p4,p5); theta3=dihedral(p3,p4,p5,p6); cosbeta=(dot_r1R1*dot_r2R2-dot_r1R2*dot_r2R1)/(distance(p1,p2)*distance(p1,p3)*distance(p4,p5)*distance(p4,p6)*sin(angle(p2,p1,p3))*sin(angle(p5,p4,p6))); dot_r1R1=a*g+b*h+c*i; dot_r2R2=d*j+e*k+f*l; dot_r1R2=a*j+b*k+c*l; dot_r2R1=d*g+e*h+f*i; a=x2-x1; b=y2-y1; c=z2-z1; d=x3-x1; e=y3-y1; f=z3-z1; g=x5-x4; h=y5-y4; i=z5-z4; j=x6-x4; k=y6-y4; l=z6-z4; RT0=2.494"

    bp_force = mm.CustomCompoundBondForce(6, bp_energy_func)
    stk_force = mm.CustomCompoundBondForce(6, stk_energy_func)
    bp_force.addGlobalParameter("kbpstk_global",1.0)
    stk_force.addGlobalParameter("kbpstk_global",1.0)

    bp_force.addPerBondParameter(f"wt_bp")
    for i in range(1,10):
        for j in range(1,6):
            bp_force.addPerBondParameter(f"amplitude{j}_bp{i}")
        for j in range(1,6):
            bp_force.addPerBondParameter(f"center{j}_bp{i}")
        for j in range(1,6):
            bp_force.addPerBondParameter(f"sigma{j}_bp{i}")
        bp_force.addPerBondParameter(f"c_bp{i}")

    bp_force.addPerBondParameter(f"wt_stk")
    for i in range(1,10):
        for j in range(1,6):
            bp_force.addPerBondParameter(f"amplitude{j}_stk{i}")
        for j in range(1,6):
            bp_force.addPerBondParameter(f"center{j}_stk{i}")
        for j in range(1,6):
            bp_force.addPerBondParameter(f"sigma{j}_stk{i}")
        bp_force.addPerBondParameter(f"c_stk{i}")

    stk_force.addPerBondParameter(f"wt_bp")
    for i in range(1,10):
        for j in range(1,6):
            stk_force.addPerBondParameter(f"amplitude{j}_bp{i}")
        for j in range(1,6):
            stk_force.addPerBondParameter(f"center{j}_bp{i}")
        for j in range(1,6):
            stk_force.addPerBondParameter(f"sigma{j}_bp{i}")
        stk_force.addPerBondParameter(f"c_bp{i}")

    stk_force.addPerBondParameter(f"wt_stk")
    for i in range(1,10):
        for j in range(1,6):
            stk_force.addPerBondParameter(f"amplitude{j}_stk{i}")
        for j in range(1,6):
            stk_force.addPerBondParameter(f"center{j}_stk{i}")
        for j in range(1,6):
            stk_force.addPerBondParameter(f"sigma{j}_stk{i}")
        stk_force.addPerBondParameter(f"c_stk{i}")

    for key in dict_atoms_index:
        for atoms_index in dict_atoms_index[key]:
            paras = copy.deepcopy(dict_paras[key])
            if atoms_index[-1][0] == "0":
                paras[0] = 0.0
            elif atoms_index[-1][0] == "2":
                paras[0] *= 1.2

            if atoms_index[-1][1:] == "AA":
                paras[145] *= 1.2
            bp_force.addBond(atoms_index[0:-1],paras)
            stk_force.addBond(atoms_index[0:-1],paras)
        
    bp_force.setForceGroup(25)
    system.addForce(bp_force)
    stk_force.setForceGroup(26)
    system.addForce(stk_force)
    return system


def get_nt_in_motifs(dict_resid_chain,dict_residues,dict_motifs):
    dict_nt_in_motifs = {}

    for key in dict_motifs:
        if key == "helix":
            continue
        if key == "PK_helix":
            continue
        if "non_PK" in key:
            continue
        motifs = dict_motifs[key]
        nt_in_motifs = []
        for motifidx,motif in enumerate(motifs):
            nt_in_motif = []
            list_nt_in_loop = []
            list_nt_in_anchor = []
            for loopidx,loop in enumerate(motif):
                loop_start_id = loop[0]
                loop_end_id = loop[1]
                for i in range(loop_start_id,loop_end_id+1):
                    chain_ntidx = dict_resid_chain[str(i)]
                    nt = dict_residues[chain_ntidx]
                    if i == loop_start_id and key != "5end_loops":
                        list_nt_in_anchor.append((nt,key+str(motifidx)+str(loopidx)))
                        continue
                    elif i == loop_end_id and key != "3end_loops":
                        list_nt_in_anchor.append((nt,key+str(motifidx)+str(loopidx)))
                        continue
                    else:
                        list_nt_in_loop.append((nt,key+str(motifidx)+str(loopidx)))
            nt_in_motif.append(list_nt_in_loop)
            nt_in_motif.append(list_nt_in_anchor)
            nt_in_motifs.append(nt_in_motif)
        dict_nt_in_motifs[key] = nt_in_motifs
    return dict_nt_in_motifs


def get_stk_type(nt1,nt2):
    nt1_id = int(nt1[0][0].id)
    nt2_id = int(nt2[0][0].id)
    nt1_chain = nt1[0][0].chain.id
    nt2_chain = nt2[0][0].chain.id
   
    if nt1_chain != nt2_chain:
        stk_type = 3
    else:
        if nt1_id == nt2_id-1:
            stk_type = 1
        elif nt1_id-1 == nt2_id:
            stk_type = 2
        elif nt1_id == nt2_id:
            print ("same nt id")
            exit()
        else:
            stk_type = 3
    return stk_type


def get_atoms_index_between_two_nts(nt1,nt2,bothanchor=False,stk_faces=None):
    # nt1[0]: the real nt; nt1[1]: the nt motif identifier 
    nt1_name = nt1[0][0].name[-1].upper()
    nt2_name = nt2[0][0].name[-1].upper()
    stk_type = get_stk_type(nt1,nt2)

    if nt1[1] != nt2[1]:
        s = nt1_name+nt2_name
        if s == "AU" or s == "UA" or s == "GU" or s == "UG" or s == "GC" or s == "CG":
            bp_candidate = "2"
            if bothanchor:
                bp_candidate += "AA"
        else:
            bp_candidate = "1"
            if bothanchor:
                bp_candidate += "AA"
    else:
        if abs(int(nt1[0][0].id)-int(nt2[0][0].id)) >= 3:
            bp_candidate = "1"
            if bothanchor:
                bp_candidate += "AA"
        else:
            bp_candidate = "0"
            if bothanchor:
                bp_candidate += "AA"

    dict_atoms_index = {}
    if nt1_name == "A":
        atoms_index = [[nt1[0][1]["NX"].index,nt1[0][1]["C2"].index,nt1[0][1]["CY"].index,nt2[0][1]["NX"].index,nt2[0][1]["C2"].index,nt2[0][1]["CY"].index,bp_candidate]]
        if stk_type == 1:
            key = nt1_name+nt2_name
        elif stk_type == 2:
            if nt1_name != nt2_name:
                key = nt1_name.lower()+nt2_name
            else:
                key = nt1_name+nt2_name
                atoms_index = [[nt2[0][1]["NX"].index,nt2[0][1]["C2"].index,nt2[0][1]["CY"].index,nt1[0][1]["NX"].index,nt1[0][1]["C2"].index,nt1[0][1]["CY"].index,bp_candidate]]
                if stk_faces == "STK35":
                    stk_faces = "STK53"
                elif stk_faces == "STK53":
                    stk_faces = "STK35"
        elif stk_type == 3:
            key = nt1_name.lower()+nt2_name.lower()
    elif nt1_name == "G":
        if nt2_name == "A":
            if stk_type == 1:
                key = nt2_name.lower()+nt1_name
            elif stk_type == 2:
                key = nt2_name+nt1_name
            elif stk_type == 3:
                key = nt2_name.lower()+nt1_name.lower()
            atoms_index = [[nt2[0][1]["NX"].index,nt2[0][1]["C2"].index,nt2[0][1]["CY"].index,nt1[0][1]["NX"].index,nt1[0][1]["C2"].index,nt1[0][1]["CY"].index,bp_candidate]]
            if stk_faces == "STK35":
                stk_faces = "STK53"
            elif stk_faces == "STK53":
                stk_faces = "STK35"
        else:
            atoms_index = [[nt1[0][1]["NX"].index,nt1[0][1]["C2"].index,nt1[0][1]["CY"].index,nt2[0][1]["NX"].index,nt2[0][1]["C2"].index,nt2[0][1]["CY"].index,bp_candidate]]
            if stk_type == 1:
                key = nt1_name+nt2_name
            elif stk_type == 2:
                if nt1_name != nt2_name: 
                    key = nt1_name.lower()+nt2_name
                else:
                    key = nt1_name+nt2_name
                    atoms_index = [[nt2[0][1]["NX"].index,nt2[0][1]["C2"].index,nt2[0][1]["CY"].index,nt1[0][1]["NX"].index,nt1[0][1]["C2"].index,nt1[0][1]["CY"].index,bp_candidate]]
                    if stk_faces == "STK35":
                        stk_faces = "STK53"
                    elif stk_faces == "STK53":
                        stk_faces = "STK35"
            elif stk_type == 3:
                key = nt1_name.lower()+nt2_name.lower()
    elif nt1_name == "C":
        if nt2_name in ["A","G"]:
            if stk_type == 1:
                key = nt2_name.lower()+nt1_name
            elif stk_type == 2:
                key = nt2_name+nt1_name
            elif stk_type == 3:
                key = nt2_name.lower()+nt1_name.lower()
            atoms_index = [[nt2[0][1]["NX"].index,nt2[0][1]["C2"].index,nt2[0][1]["CY"].index,nt1[0][1]["NX"].index,nt1[0][1]["C2"].index,nt1[0][1]["CY"].index,bp_candidate]]
            if stk_faces == "STK35":
                stk_faces = "STK53"
            elif stk_faces == "STK53":
                stk_faces = "STK35"
        else:
            atoms_index = [[nt1[0][1]["NX"].index,nt1[0][1]["C2"].index,nt1[0][1]["CY"].index,nt2[0][1]["NX"].index,nt2[0][1]["C2"].index,nt2[0][1]["CY"].index,bp_candidate]]
            if stk_type == 1:
                key = nt1_name+nt2_name
            elif stk_type == 2:
                if nt1_name != nt2_name:
                    key = nt1_name.lower()+nt2_name
                else:
                    key = nt1_name+nt2_name
                    atoms_index = [[nt2[0][1]["NX"].index,nt2[0][1]["C2"].index,nt2[0][1]["CY"].index,nt1[0][1]["NX"].index,nt1[0][1]["C2"].index,nt1[0][1]["CY"].index,bp_candidate]]
                    if stk_faces == "STK35":
                        stk_faces = "STK53"
                    elif stk_faces == "STK53":
                        stk_faces = "STK35"
            elif stk_type == 3:
                key = nt1_name.lower()+nt2_name.lower()
    elif nt1_name == "U":
        if nt2_name in ["A","G","C"]:
            if stk_type == 1:
                key = nt2_name.lower()+nt1_name
            elif stk_type == 2:
                key = nt2_name+nt1_name
            elif stk_type == 3:
                key = nt2_name.lower()+nt1_name.lower()
            atoms_index = [[nt2[0][1]["NX"].index,nt2[0][1]["C2"].index,nt2[0][1]["CY"].index,nt1[0][1]["NX"].index,nt1[0][1]["C2"].index,nt1[0][1]["CY"].index,bp_candidate]]
            if stk_faces == "STK35":
                stk_faces = "STK53"
            elif stk_faces == "STK53":
                stk_faces = "STK35"
        else:
            atoms_index = [[nt1[0][1]["NX"].index,nt1[0][1]["C2"].index,nt1[0][1]["CY"].index,nt2[0][1]["NX"].index,nt2[0][1]["C2"].index,nt2[0][1]["CY"].index,bp_candidate]]
            if stk_type == 1:
                key = nt1_name+nt2_name
            elif stk_type == 2:
                if nt1_name != nt2_name:
                    key = nt1_name.lower()+nt2_name
                else:
                    key = nt1_name+nt2_name
                    atoms_index = [[nt2[0][1]["NX"].index,nt2[0][1]["C2"].index,nt2[0][1]["CY"].index,nt1[0][1]["NX"].index,nt1[0][1]["C2"].index,nt1[0][1]["CY"].index,bp_candidate]]
                    if stk_faces == "STK35":
                        stk_faces = "STK53"
                    elif stk_faces == "STK53":
                        stk_faces = "STK35"
            elif stk_type == 3:
                key = nt1_name.lower()+nt2_name.lower()
    else:
        print ("wrong nt name",nt1_name)
        exit()

    dict_atoms_index[key] = atoms_index
    if stk_faces is not None:
        atoms_index[0].append(stk_faces)
    return dict_atoms_index


def get_nt_real_id(dict_resid_chain, nt):
    nt_id = nt[0][0].id
    nt_chain = nt[0][0].chain.id
    nt_real_id = None
    for chain_ntidx in dict_resid_chain:
        if dict_resid_chain[chain_ntidx] == (nt_chain, nt_id):
            nt_real_id = chain_ntidx
            break
    if nt_real_id is None:
        raise ValueError(f"There is no nucleotide {nt_chain}/{nt_id}.")
    return nt_real_id


def get_atoms_index_between_nts_in_loops(dict_resid_chain, list_nt_in_anchor, list_nt_in_loop, list_bulge_nts_in_jar3d, list_nt_in_helix=None):
    dict_atoms_index = {}

    for i in range(len(list_nt_in_loop)):
        nt1 = list_nt_in_loop[i]
        nt1_real_id = get_nt_real_id(dict_resid_chain,nt1)
        if int(nt1_real_id) in list_bulge_nts_in_jar3d:
            #print (f"Skipping base pairing/stacking with residue {nt1_real_id}")
            continue

        for j in range(i+1,len(list_nt_in_loop)):
            nt2 = list_nt_in_loop[j]
            nt2_real_id = get_nt_real_id(dict_resid_chain,nt2)
            if int(nt2_real_id) in list_bulge_nts_in_jar3d:
                #print (f"Skipping base pairing/stacking with residue {nt2_real_id}")
                continue

            atoms_index = get_atoms_index_between_two_nts(nt1,nt2)
            for key in atoms_index:
                if key not in dict_atoms_index.keys():
                    dict_atoms_index[key] = atoms_index[key]
                else:
                    dict_atoms_index[key].extend(atoms_index[key])

        for j in range(len(list_nt_in_anchor)):
            nt2 = list_nt_in_anchor[j]
            nt2_real_id = get_nt_real_id(dict_resid_chain,nt2)
            if int(nt2_real_id) in list_bulge_nts_in_jar3d:
                #print (f"Skipping base pairing/stacking with residue {nt2_real_id}")
                continue

            atoms_index = get_atoms_index_between_two_nts(nt1,nt2)
            for key in atoms_index:
                if key not in dict_atoms_index.keys():
                    dict_atoms_index[key] = atoms_index[key]
                else:
                    dict_atoms_index[key].extend(atoms_index[key])

        if list_nt_in_helix is None:
            continue

        for j in range(len(list_nt_in_helix)):
            nt2 = list_nt_in_helix[j]
            atoms_index = get_atoms_index_between_two_nts(nt1,nt2)
            for key in atoms_index:
                if key not in dict_atoms_index.keys():
                    dict_atoms_index[key] = atoms_index[key]
                else:
                    dict_atoms_index[key].extend(atoms_index[key])

    if len(list_nt_in_anchor) >= 4:
        for i in range(len(list_nt_in_anchor)):
            nt1 = list_nt_in_anchor[i]
            nt1_real_id = get_nt_real_id(dict_resid_chain,nt1)
            if int(nt1_real_id) in list_bulge_nts_in_jar3d:
                #print (f"Skipping base pairing/stacking with anchor residue {nt1_real_id}")
                continue
            for j in range(i+1,len(list_nt_in_anchor)):
                nt2 = list_nt_in_anchor[j]
                nt2_real_id = get_nt_real_id(dict_resid_chain,nt2)
                if int(nt2_real_id) in list_bulge_nts_in_jar3d:
                    #print (f"Skipping base pairing/stacking with anchor residue {nt2_real_id}")
                    continue
                atoms_index = get_atoms_index_between_two_nts(nt1,nt2,bothanchor=True)
                for key in atoms_index:
                    if key not in dict_atoms_index.keys():
                        dict_atoms_index[key] = atoms_index[key]
                    else:
                        dict_atoms_index[key].extend(atoms_index[key])
    return dict_atoms_index


def get_atoms_index_between_nts_in_interloops(list_nt_in_loop1, list_nt_in_loop2):
    dict_atoms_index = {}

    for i in range(len(list_nt_in_loop1)):
        nt1 = list_nt_in_loop1[i]
        for j in range(len(list_nt_in_loop2)):
            nt2 = list_nt_in_loop2[j]
            atoms_index = get_atoms_index_between_two_nts(nt1,nt2)
            for key in atoms_index:
                if key not in dict_atoms_index.keys():
                    dict_atoms_index[key] = atoms_index[key]
                else:
                    dict_atoms_index[key].extend(atoms_index[key])
    return dict_atoms_index


def combine_two_dict(dict0,dict1):
    for key in dict1:
        if key not in dict0:
            dict0[key] = dict1[key]
        else:
            dict0[key].extend(dict1[key])
    return dict0


def get_minimal_gap_between_two_index_lists(list1,list2):
    min_gap = 10000
    for i in list1:
        for j in list2:
            gap = abs(i-j)
            if gap < min_gap:
                min_gap = gap
    return min_gap        


def add_force_within_junction_loops(system,nt_in_junction_motifs,dict_residues):
    Eupper = 500.0
    compact_energy = f"(step(Ecompact-{Eupper})*0.5 + step({Eupper}-Ecompact)*1.0)*Ecompact + (step(Euncompact-{Eupper})*0.5 + step({Eupper}-Euncompact)*1.0)*Euncompact + Eoutward"
    Ecompact = f"Ecompact=kcompactjunc*(step(distance(g1,g2)-dcompactjunc)*((distance(g1,g2)-dcompactjunc)^2+0.5*(dcompactjunc-0.5)^2) + 0.5*step(-distance(g1,g2)+dcompactjunc)*step(distance(g1,g2)-0.5)*(distance(g1,g2)-0.5)^2)"
    Euncompact = f"Euncompact=kuncompactjunc*(distance(g1,g2)-duncompactjunc)^2"
    Eoutward = f"Eoutward=koutward*((distance(g1,g3)-0.0)^2+(distance(g2,g4)-0.0)^2)"
    compact_force = f"{compact_energy}; {Ecompact}; {Euncompact}; {Eoutward}"
    force = mm.CustomCentroidBondForce(4,compact_force)
    force.addGlobalParameter("kcompactjunc",0)
    force.addGlobalParameter("dcompactjunc",0)
    force.addGlobalParameter("kuncompactjunc",0)
    force.addGlobalParameter("duncompactjunc",0)
    force.addGlobalParameter("koutward",0)
    for i in range(len(nt_in_junction_motifs)):
        junction_nts = nt_in_junction_motifs[i]
        nts_in_loop, nts_in_anchor = junction_nts

        list_nts_in_junction_branches = []
        nway = int(len(nts_in_anchor)/2)
        for j in range(nway):
            list_nts_in_one_junction_branch = []
            for nt in nts_in_anchor:
                if nt[1][-1] == str(j):
                    list_nts_in_one_junction_branch.append(nt)
            for nt in nts_in_loop:
                if nt[1][-1] == str(j):
                    list_nts_in_one_junction_branch.append(nt)
            list_nts_in_junction_branches.append(list_nts_in_one_junction_branch)

        list_group_index = []
        for j in range(nway):
            group0 = []
            for nt in list_nts_in_junction_branches[j]:
                index = nt[0][1]["NX"].index
                group0.append(index)
            
            nt0 = list_nts_in_junction_branches[j][0]
            if nway == 4:
                if j == 2 or j == 3: 
                    nt0 = list_nts_in_junction_branches[j][1]
            nt0_id = nt0[0][0].id
            nt0_chain = nt0[0][0].chain.id
            for k in range(4,0,-1):
                ntx_resi = int(nt0_id) - k
                if nway == 4:
                    if j == 2 or j == 3:
                        ntx_resi = int(nt0_id) + k
                key = (nt0_chain, str(ntx_resi))
                if key in dict_residues.keys():
                    ntx = dict_residues[key]
                    break
            group1 = []
            group1.append(ntx[1]["NX"].index)
            
            group0_index = force.addGroup(group0,[1.0]*len(group0))
            group1_index = force.addGroup(group1,[1.0]*len(group1))
            list_group_index.append([group0_index,group1_index])

        for j in range(nway):
            for k in range(j+1,nway):
                group1_index = list_group_index[j][0]
                group2_index = list_group_index[k][0]
                group3_index = list_group_index[j][1]
                group4_index = list_group_index[k][1]
                force.addBond([group1_index,group2_index,group3_index,group4_index],[])

    force.setForceGroup(8)
    system.addForce(force)
    return system


def add_force_between_hairpin_internal_loops(system, loop_nt_in_2way_hairpin_motifs, dict_residues):
    face_face_constraint = f"face_face_wt=face_face + 0.5*(1-face_face); face_face=step(-cosbeta)*step(-dot_g1g4_n1); cosbeta=(dot_r1R1*dot_r2R2-dot_r1R2*dot_r2R1)/(distance(g1,g2)*distance(g1,g3)*distance(g4,g5)*distance(g4,g6)*sin(angle(g2,g1,g3))*sin(angle(g5,g4,g6))); dot_r1R1=a*g+b*h+c*i; dot_r2R2=d*j+e*k+f*l; dot_r1R2=a*j+b*k+c*l; dot_r2R1=d*g+e*h+f*i; dot_g1g4_n1=m*(b*f-e*c)+n*(c*d-a*f)+p*(a*e-b*d); a=x2-x1; b=y2-y1; c=z2-z1; d=x3-x1; e=y3-y1; f=z3-z1; g=x5-x4; h=y5-y4; i=z5-z4; j=x6-x4; k=y6-y4; l=z6-z4; m=x4-x1; n=y4-y1; p=z4-z1;"

    attraction_force1 = f"(face_face_wt)* klooploop * step(dlooploop_upper-distance(g7,g8)) * (step(distance(g7,g8)-dlooploop_lower)*(distance(g7,g8)-dlooploop_ave)^2 - (dlooploop_upper-dlooploop_ave)^2)" # short-range attraction between loops

    attraction_force2 = f"looplooptype * (face_face_wt) * khphp * step(dhphp_upper-distance(g7,g8)) * (step(distance(g7,g8)-dhphp_lower)*(distance(g7,g8)-dhphp_ave)^2 - (dhphp_upper-dhphp_ave)^2)"  # long-range attraction between hairpin-hairpin loops
    attraction_force = f"{attraction_force1} + {attraction_force2}; {face_face_constraint}"

    force = mm.CustomCentroidBondForce(8,attraction_force)
    force.addGlobalParameter("klooploop",0)
    force.addGlobalParameter("khphp",0)
    force.addGlobalParameter("dlooploop_lower",0)
    force.addGlobalParameter("dlooploop_ave",0)
    force.addGlobalParameter("dlooploop_upper",3)
    force.addGlobalParameter("dhphp_lower",3)
    force.addGlobalParameter("dhphp_ave",3)
    force.addGlobalParameter("dhphp_upper",4.5)
    force.addPerBondParameter("looplooptype")

    list_groups = []
    for i in range(len(loop_nt_in_2way_hairpin_motifs)):
        list_nt_in_loop = loop_nt_in_2way_hairpin_motifs[i][0]
        loop_type = loop_nt_in_2way_hairpin_motifs[i][1]
        list_nt_in_anchor = loop_nt_in_2way_hairpin_motifs[i][2]

        if loop_type == "single_loops":
            if len(list_nt_in_loop) == 0:
                continue

        nt1 = list_nt_in_loop[0][0]
        nt2 = list_nt_in_loop[-1][0]
        nt1_chain = nt1[0].chain.id
        nt1_resi = int(nt1[0].id)
        nt2_chain = nt2[0].chain.id
        nt2_resi = int(nt2[0].id)

        if loop_type in ["hairpin_loops","single_loops","2way_loops"]:
            nt0_chain = nt1_chain
            nt0_resi = nt1_resi - 1
            nt3_chain = nt2_chain
            nt3_resi = nt2_resi + 1
            nt0 = dict_residues[(nt0_chain,str(nt0_resi))]
            nt3 = dict_residues[(nt3_chain,str(nt3_resi))]
            middle_index = int(len(list_nt_in_loop)/2)
            nt_middle = list_nt_in_loop[middle_index][0]
        elif loop_type in ["5end_loops"]:
            if len(list_nt_in_loop) == 1:
                nt0 = nt1
                nt_middle = dict_residues[(nt1_chain,str(nt1_resi+1))]
                nt3 = dict_residues[(nt1_chain,str(nt1_resi+2))]
            elif len(list_nt_in_loop) > 1:
                nt0 = nt1
                nt3 = dict_residues[(nt2_chain,str(nt2_resi+1))]
                middle_index = int(len(list_nt_in_loop)/2)
                nt_middle = list_nt_in_loop[middle_index][0]
            else:
                print ("Loop length is 0")
                exit()
        elif loop_type in ["3end_loops"]:
            if len(list_nt_in_loop) == 1:
                nt0 = dict_residues[(nt2_chain,str(nt2_resi-2))]
                nt_middle = dict_residues[(nt2_chain,str(nt2_resi-1))]
                nt3 = nt2
            elif len(list_nt_in_loop) > 1:
                nt0 = dict_residues[(nt1_chain,str(nt1_resi-1))]
                nt3 = nt2
                middle_index = int(len(list_nt_in_loop)/2) - 1
                nt_middle = list_nt_in_loop[middle_index][0]
            else:
                print ("Loop length is 0")
        else:
            print (f"Wrong loop type: {loop_type}")
            exit()

        group = []

        group0 = []
        index = nt0[1]["BBP"].index
        group0.append(index)

        group1 = []
        index = nt_middle[1]["BBP"].index
        group1.append(index)

        group2 = []
        index = nt3[1]["BBP"].index
        group2.append(index)

        group3 = []
        for nt in list_nt_in_loop:
            index = nt[0][1]["NX"].index 
            group3.append(index)
    
        group4 = []
        for nt in list_nt_in_anchor:
            index = nt[0][1]["BBP"].index 
            group4.append(index)

        group.append(group0)
        group.append(group1)
        group.append(group2)
        group.append(group3)
        group.append(loop_type) # loop type
        group.append(group4) # P atoms index in anchors
        list_groups.append(group)

    list_group_index = []
    list_group_index_Patoms_index = []
    for group in list_groups:
        group_index0 = force.addGroup(group[0],[1.0]*len(group[0]))
        group_index1 = force.addGroup(group[1],[1.0]*len(group[1]))
        group_index2 = force.addGroup(group[2],[1.0]*len(group[2]))
        group_index3 = force.addGroup(group[3],[1.0]*len(group[3]))
        group_index = [group_index0,group_index1,group_index2,group_index3,group[4]]
        list_group_index.append(group_index)
        list_group_index_Patoms_index.append(group[5])
   
    for i in range(len(list_group_index)):
        for j in range(i+1,len(list_group_index)):
            group1_index = list_group_index[i]
            group2_index = list_group_index[j]
            loop1_type = group1_index[-1]
            loop2_type = group2_index[-1]
            all_group_index = group1_index[0:3]+group2_index[0:3]+[group1_index[3]]+[group2_index[3]]
            if (loop1_type == "2way_loops" and loop2_type == "2way_loops") or (loop1_type == "2way_loops" and loop2_type == "hairpin_loops") or (loop1_type == "hairpin_loops" and loop2_type == "2way_loops"):
                Patoms_index1 = list_group_index_Patoms_index[i]
                Patoms_index2 = list_group_index_Patoms_index[j]
                minimal_gap = get_minimal_gap_between_two_index_lists(Patoms_index1,Patoms_index2)
                if minimal_gap <= 5*12: # shorter than 12 nts
                    #print (f"Skip {loop1_type}-{loop2_type} attraction due to short sequence separation")
                    continue
            if loop1_type in ["5end_loops","3end_loops"]:
                if loop2_type not in ["5end_loops","3end_loops"]:
                    continue
            if loop1_type == "hairpin_loops" and loop2_type == "hairpin_loops":
                loop_loop_type_para = [1.0]
            else:
                loop_loop_type_para = [0.0]
            force.addBond(all_group_index,loop_loop_type_para)

    force.setForceGroup(7)
    system.addForce(force)
    return system


def add_base_pairing_stacking_force_to_system(RNAJP_HOME, topology,system,dict_motifs,dict_resid_chain,dict_bp_stk_strength,list_bulge_nts_in_jar3d,split=False):
    dict_residues = access_to_residues_by_chain_name_and_resid(topology)
    dict_atoms_index = {}
    dict_nt_in_motifs = get_nt_in_motifs(dict_resid_chain,dict_residues,dict_motifs)

    for key in dict_nt_in_motifs:
        nt_in_motifs = dict_nt_in_motifs[key]
        for nt_in_motif in nt_in_motifs:
            list_nt_in_loop, list_nt_in_anchor = nt_in_motif
            dict_atoms_index0 = get_atoms_index_between_nts_in_loops(dict_resid_chain, list_nt_in_anchor, list_nt_in_loop, list_bulge_nts_in_jar3d)
            dict_atoms_index = combine_two_dict(dict_atoms_index,dict_atoms_index0)

    loop_nt_in_2way_hairpin_motifs = []
    for key in dict_nt_in_motifs:
        if key not in ["2way_loops","hairpin_loops","5end_loops","3end_loops","single_loops"]:
            continue
        nt_in_motifs = dict_nt_in_motifs[key]
        for nt_in_motif in nt_in_motifs:
            list_nt_in_loop, list_nt_in_anchor = nt_in_motif
            loop_nt_in_2way_hairpin_motifs.append([list_nt_in_loop,key,list_nt_in_anchor])

    loop_nt_in_Nway_motifs = []
    for key in dict_nt_in_motifs:
        if key not in ["3way_loops","4way_loops"]:
            continue
        nt_in_motifs = dict_nt_in_motifs[key]
        for nt_in_motif in nt_in_motifs:
            list_nt_in_loop, list_nt_in_anchor = nt_in_motif
            loop_nt_in_Nway_motifs.append([list_nt_in_loop,key,list_nt_in_anchor])

    for i in range(len(loop_nt_in_2way_hairpin_motifs)):
        list_nt_in_loop1 = loop_nt_in_2way_hairpin_motifs[i][0]
        list_nt_in_loop1_anchor = loop_nt_in_2way_hairpin_motifs[i][2]

        list_anchor_index1 = []
        for nt in list_nt_in_loop1_anchor:
            real_nt_id = get_nt_real_id(dict_resid_chain, nt)
            list_anchor_index1.append(int(real_nt_id))

        for j in range(i+1,len(loop_nt_in_2way_hairpin_motifs)):
            list_nt_in_loop2 = loop_nt_in_2way_hairpin_motifs[j][0]
            list_nt_in_loop2_anchor = loop_nt_in_2way_hairpin_motifs[j][2]

            dict_atoms_index0 = get_atoms_index_between_nts_in_interloops(list_nt_in_loop1, list_nt_in_loop2)
            dict_atoms_index = combine_two_dict(dict_atoms_index,dict_atoms_index0)

            dict_atoms_index0 = get_atoms_index_between_nts_in_interloops(list_nt_in_loop1, list_nt_in_loop2_anchor)
            dict_atoms_index = combine_two_dict(dict_atoms_index,dict_atoms_index0)

            dict_atoms_index0 = get_atoms_index_between_nts_in_interloops(list_nt_in_loop1_anchor, list_nt_in_loop2)
            dict_atoms_index = combine_two_dict(dict_atoms_index,dict_atoms_index0)

        for j in range(len(loop_nt_in_Nway_motifs)):
            list_nt_in_loop2 = loop_nt_in_Nway_motifs[j][0]
            list_nt_in_loop2_anchor = loop_nt_in_Nway_motifs[j][2]

            list_anchor_index2 = []
            for nt in list_nt_in_loop2_anchor:
                real_nt_id = get_nt_real_id(dict_resid_chain, nt)
                list_anchor_index2.append(int(real_nt_id))
            minimal_gap = get_minimal_gap_between_two_index_lists(list_anchor_index1,list_anchor_index2)
            if minimal_gap > 10: # calculate base pairing/stacking between junction loops and hairpin/internal loops which are close to each other (no greater than 10 nucleotides)
                continue

            dict_atoms_index0 = get_atoms_index_between_nts_in_interloops(list_nt_in_loop1, list_nt_in_loop2)
            dict_atoms_index = combine_two_dict(dict_atoms_index,dict_atoms_index0)

            dict_atoms_index0 = get_atoms_index_between_nts_in_interloops(list_nt_in_loop1, list_nt_in_loop2_anchor)
            dict_atoms_index = combine_two_dict(dict_atoms_index,dict_atoms_index0)

            dict_atoms_index0 = get_atoms_index_between_nts_in_interloops(list_nt_in_loop1_anchor, list_nt_in_loop2)
            dict_atoms_index = combine_two_dict(dict_atoms_index,dict_atoms_index0)
    
    if split:
        system = add_base_pairing_stacking_force_split(RNAJP_HOME, system, dict_atoms_index, dict_bp_stk_strength)
    else:
        system = add_base_pairing_stacking_force(RNAJP_HOME, system, dict_atoms_index, dict_bp_stk_strength)
    return system


def add_force_between_hairpin_internal_loops_to_system(topology,system,dict_motifs,dict_resid_chain):
    dict_residues = access_to_residues_by_chain_name_and_resid(topology)
    dict_nt_in_motifs = get_nt_in_motifs(dict_resid_chain,dict_residues,dict_motifs)

    loop_nt_in_2way_hairpin_motifs = []
    for key in dict_nt_in_motifs:
        if key not in ["2way_loops_non_PK","hairpin_loops_non_PK","5end_loops_non_PK","3end_loops_non_PK","single_loops_non_PK"]:
            continue
        nt_in_motifs = dict_nt_in_motifs[key]
        for nt_in_motif in nt_in_motifs:
            list_nt_in_loop, list_nt_in_anchor = nt_in_motif
            loop_nt_in_2way_hairpin_motifs.append([list_nt_in_loop,key,list_nt_in_anchor])

    system = add_force_between_hairpin_internal_loops(system, loop_nt_in_2way_hairpin_motifs, dict_residues)
    return system


def add_force_within_junction_loops_to_system(topology,system,dict_motifs,dict_resid_chain):
    dict_residues = access_to_residues_by_chain_name_and_resid(topology)
    dict_nt_in_motifs = get_nt_in_motifs(dict_resid_chain,dict_residues,dict_motifs)

    nt_in_junction_motifs = []
    for key in dict_nt_in_motifs:
        if key not in ["3way_loops","4way_loops"]:
            continue
        nt_in_motifs = dict_nt_in_motifs[key]
        for nt_in_motif in nt_in_motifs:
            nt_in_junction_motifs.append(nt_in_motif)

    system = add_force_within_junction_loops(system,nt_in_junction_motifs,dict_residues)
    return system


def add_jar3d_force_to_system(topology,system,dict_motifs,dict_resid_chain,dict_jar3d_energy_paras):
    list_helices = []
    for helix in dict_motifs["helix"]:
        h1, h4 = helix[0]
        h2, h3 = helix[1]
        if h1 != h2:
            list_helices.append(helix)

    list_helices = list(itertools.chain.from_iterable(list_helices))    
    list_helix_anchor_idx = list(itertools.chain.from_iterable(list_helices))

    Etorsion = f"ktorsionjar3d*min(dtheta, 2*pi-dtheta)^2"
    torsion_energy = f"(0.1*step(Etorsion-1000)*Etorsion + step(1000-Etorsion)*Etorsion)*step(sin(agl1)-0.087155743)*step(sin(agl2)-0.087155743); Etorsion={Etorsion}; dtheta = abs(theta-theta0); pi = 3.1415926535; theta=dihedral(p1,p2,p3,p4); agl1=angle(p1,p2,p3); agl2=angle(p2,p3,p4)"
    torsion_force = mm.CustomCompoundBondForce(4, torsion_energy)
    torsion_force.addPerBondParameter("theta0")
    torsion_force.addPerBondParameter("ktorsionjar3d")

    Eangle = f"kanglejar3d*(theta-theta0)^2"
    angle_energy = f"0.1*step(Eangle-1000)*Eangle + step(1000-Eangle)*Eangle + 500*step(minangle-theta)*(theta-minangle)^2; Eangle={Eangle}; theta=angle(p1,p2,p3); minangle=min(theta0,0.523598776)" # minangle = min(theta0, 30 degree)
    angle_force = mm.CustomCompoundBondForce(3,angle_energy)
    angle_force.addPerBondParameter("theta0")
    angle_force.addPerBondParameter("kanglejar3d")

    Edis = f"kdisjar3d*(dis-dis0)^2"
    dis_energy = f"0.1*step(Edis-1000)*Edis + step(1000-Edis)*Edis; Edis={Edis}; dis=distance(p1,p2)"
    dis_force = mm.CustomCompoundBondForce(2,dis_energy)
    dis_force.addPerBondParameter("dis0")
    dis_force.addPerBondParameter("kdisjar3d")

    dict_residues = access_to_residues_by_chain_name_and_resid(topology)

    list_torsion_atoms_index_jar3d = []
    list_angle_atoms_index_jar3d = []

    list_jar3d_torsion_bonds = []
    list_jar3d_angle_bonds = []
    list_jar3d_dis_bonds = []

    list_jar3d_torsion_bonds_motif = []
    list_jar3d_angle_bonds_motif = []
    list_jar3d_dis_bonds_motif = []
    for key in dict_jar3d_energy_paras:
        nt0_idx = key[0]
        nt1_idx = key[1]
        nt2_idx = key[2]

        if nt0_idx is None:
            if list_jar3d_torsion_bonds_motif:
                list_jar3d_torsion_bonds.append(list_jar3d_torsion_bonds_motif)
                list_jar3d_torsion_bonds_motif = []
            if list_jar3d_angle_bonds_motif:
                list_jar3d_angle_bonds.append(list_jar3d_angle_bonds_motif)
                list_jar3d_angle_bonds_motif = []
            if list_jar3d_dis_bonds_motif:
                list_jar3d_dis_bonds.append(list_jar3d_dis_bonds_motif)
                list_jar3d_dis_bonds_motif = []
            
        if nt0_idx is None:
            nt0_idx = nt1_idx - 1
        paras = dict_jar3d_energy_paras[key]
        if len(paras) != 12:
            print ("JAR3D energy paras are longer or shorter than 12.")
            print (paras)
            exit()
        paras[0:9] = np.radians(paras[0:9])
        paras[9] /= 10.0
        paras[10] /= 10.0
        paras[11] /= 10.0

        chain_ntidx = dict_resid_chain[str(nt0_idx)]
        nt0 = dict_residues[chain_ntidx]

        chain_ntidx = dict_resid_chain[str(nt1_idx)]
        nt1 = dict_residues[chain_ntidx]

        chain_ntidx = dict_resid_chain[str(nt2_idx)]
        nt2 = dict_residues[chain_ntidx]
        
        C4s_0 = nt0[1]["BBC"].index
        P_1 = nt1[1]["BBP"].index
        C4s_1 = nt1[1]["BBC"].index
        NX_1 = nt1[1]["NX"].index

        P_2 = nt2[1]["BBP"].index
        C4s_2 = nt2[1]["BBC"].index
        NX_2 = nt2[1]["NX"].index
        C2_2 = nt2[1]["C2"].index
        CY_2 = nt2[1]["CY"].index

        torsion_bond_info = []
        angle_bond_info = []
        dis_bond_info = []

        k0 = 10
        idx1 = torsion_force.addBond([C4s_0,P_1,C4s_1,P_2],[paras[0],k0])
        idx2 = torsion_force.addBond([P_1,C4s_1,P_2,C4s_2],[paras[1],k0])
        idx3 = torsion_force.addBond([C4s_1,P_2,C4s_2,NX_2],[paras[2],k0])
        bond_info = [idx1,[C4s_0,P_1,C4s_1,P_2],[paras[0],k0]]
        torsion_bond_info.append(bond_info)
        bond_info = [idx2,[P_1,C4s_1,P_2,C4s_2],[paras[1],k0]]
        torsion_bond_info.append(bond_info)
        bond_info = [idx3,[C4s_1,P_2,C4s_2,NX_2],[paras[2],k0]]
        torsion_bond_info.append(bond_info)
        if nt2_idx not in list_helix_anchor_idx:
            idx4 = torsion_force.addBond([P_2,C4s_2,NX_2,C2_2],[paras[3],k0])
            idx5 = torsion_force.addBond([C4s_2,NX_2,C2_2,CY_2],[paras[4],k0])
            bond_info = [idx4,[P_2,C4s_2,NX_2,C2_2],[paras[3],k0]]
            torsion_bond_info.append(bond_info)
            bond_info = [idx5,[C4s_2,NX_2,C2_2,CY_2],[paras[4],k0]]
            torsion_bond_info.append(bond_info)

        list_torsion_atoms_index_jar3d.append([C4s_0,P_1,C4s_1,P_2])
        list_torsion_atoms_index_jar3d.append([P_1,C4s_1,P_2,C4s_2])
        list_torsion_atoms_index_jar3d.append([C4s_1,P_2,C4s_2,NX_2])
        if nt2_idx not in list_helix_anchor_idx:
            list_torsion_atoms_index_jar3d.append([P_2,C4s_2,NX_2,C2_2])
            list_torsion_atoms_index_jar3d.append([C4s_2,NX_2,C2_2,CY_2])

        idx6 = angle_force.addBond([P_1,C4s_1,P_2],[paras[5],k0])
        idx7 = angle_force.addBond([C4s_1,P_2,C4s_2],[paras[6],k0])
        bond_info = [idx6,[P_1,C4s_1,P_2],[paras[5],k0]]
        angle_bond_info.append(bond_info)
        bond_info = [idx7,[C4s_1,P_2,C4s_2],[paras[6],k0]]
        angle_bond_info.append(bond_info)
        if nt2_idx not in list_helix_anchor_idx:
            idx8 = angle_force.addBond([P_2,C4s_2,NX_2],[paras[7],k0])
            idx9 = angle_force.addBond([C4s_2,NX_2,C2_2],[paras[8],k0])
            bond_info = [idx8,[P_2,C4s_2,NX_2],[paras[7],k0]]
            angle_bond_info.append(bond_info)
            bond_info = [idx9,[C4s_2,NX_2,C2_2],[paras[8],k0]]
            angle_bond_info.append(bond_info)

        list_angle_atoms_index_jar3d.append([P_1,C4s_1,P_2])
        list_angle_atoms_index_jar3d.append([C4s_1,P_2,C4s_2])
        if nt2_idx not in list_helix_anchor_idx:
            list_angle_atoms_index_jar3d.append([C4s_1,P_2,C4s_2])
            list_angle_atoms_index_jar3d.append([C4s_1,P_2,C4s_2])

        idx10 = dis_force.addBond([P_1,P_2],[paras[9],k0])
        idx11 = dis_force.addBond([C4s_1,C4s_2],[paras[10],k0])
        idx12 = dis_force.addBond([NX_1,NX_2],[paras[11],k0])
       
        bond_info = [idx10,[P_1,P_2],[paras[9],k0]] 
        dis_bond_info.append(bond_info)
        bond_info = [idx11,[C4s_1,C4s_2],[paras[10],k0]]
        dis_bond_info.append(bond_info)
        bond_info = [idx12,[NX_1,NX_2],[paras[11],k0]]
        dis_bond_info.append(bond_info)
    
        list_jar3d_torsion_bonds_motif.append(torsion_bond_info)
        list_jar3d_angle_bonds_motif.append(angle_bond_info)
        list_jar3d_dis_bonds_motif.append(dis_bond_info)

    list_jar3d_torsion_bonds.append(list_jar3d_torsion_bonds_motif)
    list_jar3d_angle_bonds.append(list_jar3d_angle_bonds_motif)
    list_jar3d_dis_bonds.append(list_jar3d_dis_bonds_motif)

    dis_force.setForceGroup(1)
    jar3d_dis_force_index = system.addForce(dis_force)

    angle_force.setForceGroup(2)
    jar3d_angle_force_index = system.addForce(angle_force)

    torsion_force.setForceGroup(3)
    jar3d_torsion_force_index = system.addForce(torsion_force)

    return system, list_torsion_atoms_index_jar3d, list_angle_atoms_index_jar3d, jar3d_dis_force_index, list_jar3d_dis_bonds, jar3d_angle_force_index, list_jar3d_angle_bonds, jar3d_torsion_force_index, list_jar3d_torsion_bonds


def add_possible_bp_attraction_force_to_system(topology,system,dict_motifs,dict_resid_chain,seq):
    dict_residues = access_to_residues_by_chain_name_and_resid(topology)

    list_dict_nts_in_junction = []
    for key in ["3way_loops","4way_loops"]:
        if key not in dict_motifs.keys():
            continue
        junctions = dict_motifs[key]
        for junction in junctions:
            dict_nts_in_junction = {}
            for branch in junction:
                ib = branch[0]
                ie = branch[1]
                for i in range(ib+1,ie):
                    nt_name = seq[i-1]
                    if nt_name not in dict_nts_in_junction.keys():
                        dict_nts_in_junction[nt_name] = [i]
                    else:
                        dict_nts_in_junction[nt_name].append(i)
            list_dict_nts_in_junction.append(dict_nts_in_junction)

    dis_energy = f"step(dbpclose-dis1)*kbpclose_global*kbpclose_perbond*(step(0.7-dis1)*(dis1-dis10)^2 + step(dis1-dis10)*(dis1-dis10)^2 + step(dis2-dis20)*(dis2-dis20)^2 + step(dis3-dis30)*(dis3-dis30)^2); dis1=distance(p1,p2); dis2=distance(p3,p4); dis3=distance(p5,p6); dis10=1.0; dis20=0.8; dis30=0.8"
    dis_force = mm.CustomCompoundBondForce(6,dis_energy)
    dis_force.addGlobalParameter("dbpclose",0.0)
    dis_force.addGlobalParameter("kbpclose_global",0.0)
    dis_force.addPerBondParameter("kbpclose_perbond")

    list_bond_index_GC = []
    list_bond_index_AU = []
    for dict_nts_in_junction in list_dict_nts_in_junction:
        if "G" in dict_nts_in_junction.keys() and "C" in dict_nts_in_junction.keys():
            for nt1_idx in dict_nts_in_junction["G"]:
                for nt2_idx in dict_nts_in_junction["C"]:
                    if abs(nt1_idx-nt2_idx) < 4:
                        continue
                    chain_ntidx = dict_resid_chain[str(nt1_idx)]
                    nt1 = dict_residues[chain_ntidx]
                    chain_ntidx = dict_resid_chain[str(nt2_idx)]
                    nt2 = dict_residues[chain_ntidx]
                    nt1_NX = nt1[1]["NX"].index
                    nt1_C2 = nt1[1]["C2"].index
                    nt1_CY = nt1[1]["CY"].index
                    nt2_NX = nt2[1]["NX"].index
                    nt2_C2 = nt2[1]["C2"].index
                    nt2_CY = nt2[1]["CY"].index
                    atoms_index = [nt1_NX,nt2_NX,nt1_C2,nt2_C2,nt1_CY,nt2_CY]
                    bond_index = dis_force.addBond(atoms_index,[0.0])
                    list_bond_index_GC.append([bond_index,atoms_index])

        if "A" in dict_nts_in_junction.keys() and "U" in dict_nts_in_junction.keys():
            for nt1_idx in dict_nts_in_junction["A"]:
                for nt2_idx in dict_nts_in_junction["U"]:
                    if abs(nt1_idx-nt2_idx) < 4:
                        continue
                    chain_ntidx = dict_resid_chain[str(nt1_idx)]
                    nt1 = dict_residues[chain_ntidx]
                    chain_ntidx = dict_resid_chain[str(nt2_idx)]
                    nt2 = dict_residues[chain_ntidx]
                    nt1_NX = nt1[1]["NX"].index
                    nt1_C2 = nt1[1]["C2"].index
                    nt1_CY = nt1[1]["CY"].index
                    nt2_NX = nt2[1]["NX"].index
                    nt2_C2 = nt2[1]["C2"].index
                    nt2_CY = nt2[1]["CY"].index
                    atoms_index = [nt1_NX,nt2_NX,nt1_C2,nt2_C2,nt1_CY,nt2_CY]
                    bond_index = dis_force.addBond(atoms_index,[0.0])
                    list_bond_index_AU.append([bond_index,atoms_index])

    dis_force.setForceGroup(12)
    index_bp_close_force = system.addForce(dis_force)
    return system, index_bp_close_force, list_bond_index_GC, list_bond_index_AU
