import simtk.openmm as mm
import simtk.unit as unit
import numpy as np

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


def get_angle_torsion_in_one_loop(dict_resid_chain,dict_residues,loop,loop_type):
    loop_start_id = loop[0]
    loop_end_id = loop[1]

    list_nt = [] #include four nts in two helices if any
    list_anchor_nt_index = [[],[]]
    num = 0
    for i in range(loop_start_id-1,loop_end_id+2):
        if str(i) in dict_resid_chain.keys():
            chain_ntidx = dict_resid_chain[str(i)]
            if chain_ntidx not in dict_residues:
                raise ValueError(f"nucleotide {chain_ntidx[0]}/{chain_ntidx[1]} does not exist!")
            nt = dict_residues[chain_ntidx]
            list_nt.append(nt)
            if loop_type == "5end_loops":
                if i == loop_end_id or i == loop_end_id+1:
                    list_anchor_nt_index[1].append(num)
            elif loop_type == "3end_loops":
                if i == loop_start_id-1 or i == loop_start_id:
                    list_anchor_nt_index[0].append(num)
            else:
                if i in [loop_start_id-1,loop_start_id]:
                    list_anchor_nt_index[0].append(num)
                elif i in [loop_end_id,loop_end_id+1]:
                    list_anchor_nt_index[1].append(num)
            num += 1

    dict_torsion = {}
    dict_angle = {}
    for i in range(len(list_nt)-2):
        BBP_1 = list_nt[i][1]["BBP"].index
        BBC_1 = list_nt[i][1]["BBC"].index
        NX_1 = list_nt[i][1]["NX"].index
        C2_1 = list_nt[i][1]["C2"].index
        CY_1 = list_nt[i][1]["CY"].index

        BBP_2 = list_nt[i+1][1]["BBP"].index
        BBC_2 = list_nt[i+1][1]["BBC"].index
        NX_2 = list_nt[i+1][1]["NX"].index
        C2_2 = list_nt[i+1][1]["C2"].index
        CY_2 = list_nt[i+1][1]["CY"].index
        
        BBP_3 = list_nt[i+2][1]["BBP"].index
        BBC_3 = list_nt[i+2][1]["BBC"].index
        NX_3 = list_nt[i+2][1]["NX"].index
        C2_3 = list_nt[i+2][1]["C2"].index
        CY_3 = list_nt[i+2][1]["CY"].index

        key = "CPCP"
        if key not in dict_torsion.keys():
            dict_torsion[key] = [[BBC_1,BBP_2,BBC_2,BBP_3]]
        else:
            dict_torsion[key].append([BBC_1,BBP_2,BBC_2,BBP_3])

        key = "PCPC"
        if (i+1 not in list_anchor_nt_index[1]):
            if key not in dict_torsion.keys():
                dict_torsion[key] = [[BBP_2,BBC_2,BBP_3,BBC_3]]
            else:
                dict_torsion[key].append([BBP_2,BBC_2,BBP_3,BBC_3])
        if i == 0:
            if (i not in list_anchor_nt_index[0]) and (i+1 not in list_anchor_nt_index[0]):
                if key not in dict_torsion.keys():
                    dict_torsion[key] = [[BBP_1,BBC_1,BBP_2,BBC_2]]
                else:
                    dict_torsion[key].append([BBP_1,BBC_1,BBP_2,BBC_2])
       
        key = "CPCNx"
        if (i+1 not in list_anchor_nt_index[1]):
            if key not in dict_torsion.keys():
                dict_torsion[key] = [[BBC_2,BBP_3,BBC_3,NX_3]]
            else:
                dict_torsion[key].append([BBC_2,BBP_3,BBC_3,NX_3])
        if i == 0:
            if (i not in list_anchor_nt_index[0]) and (i+1 not in list_anchor_nt_index[0]):
                if key not in dict_torsion.keys():
                    dict_torsion[key] = [[BBC_1,BBP_2,BBC_2,NX_2]]
                else:
                    dict_torsion[key].append([BBC_1,BBP_2,BBC_2,NX_2])

        key = "NxCPC"
        if (i+1 not in list_anchor_nt_index[1]):
            if key not in dict_torsion.keys():
                dict_torsion[key] = [[NX_2,BBC_2,BBP_3,BBC_3]]
            else:
                dict_torsion[key].append([NX_2,BBC_2,BBP_3,BBC_3])
        if i == 0:
            if (i not in list_anchor_nt_index[0]) and (i+1 not in list_anchor_nt_index[0]):
                if key not in dict_torsion.keys():
                    dict_torsion[key] = [[NX_1,BBC_1,BBP_2,BBC_2]]
                else:
                    dict_torsion[key].append([NX_1,BBC_1,BBP_2,BBC_2])

        nt_name = list_nt[i+2][0].name[-1]

        key = nt_name+"_CNxC2Cy"
        if i+2 not in list_anchor_nt_index[1]:
            if key not in dict_torsion.keys():
                dict_torsion[key] = [[BBC_3,NX_3,C2_3,CY_3]]
            else:
                dict_torsion[key].append([BBC_3,NX_3,C2_3,CY_3])

        if i == 0:
            nt_name2 = list_nt[i][0].name[-1]
            key = nt_name2+"_CNxC2Cy"
            if (i not in list_anchor_nt_index[0]) and (i not in list_anchor_nt_index[1]):
                if key not in dict_torsion.keys():
                    dict_torsion[key] = [[BBC_1,NX_1,C2_1,CY_1]]
                else:
                    dict_torsion[key].append([BBC_1,NX_1,C2_1,CY_1])
            nt_name2 = list_nt[i+1][0].name[-1]
            key = nt_name2+"_CNxC2Cy"
            if (i+1 not in list_anchor_nt_index[0]) and (i+1 not in list_anchor_nt_index[1]):
                if key not in dict_torsion.keys():
                    dict_torsion[key] = [[BBC_2,NX_2,C2_2,CY_2]]
                else:
                    dict_torsion[key].append([BBC_2,NX_2,C2_2,CY_2])

        key = nt_name+"_PCNxC2"
        if i+2 not in list_anchor_nt_index[1]:
            if key not in dict_torsion.keys():
                dict_torsion[key] = [[BBP_3,BBC_3,NX_3,C2_3]]
            else:
                dict_torsion[key].append([BBP_3,BBC_3,NX_3,C2_3])
        if i == 0:
            nt_name2 = list_nt[i][0].name[-1]
            key = nt_name2+"_PCNxC2"
            if (i not in list_anchor_nt_index[0]) and (i not in list_anchor_nt_index[1]):
                if key not in dict_torsion.keys():
                    dict_torsion[key] = [[BBP_1,BBC_1,NX_1,C2_1]]
                else:
                    dict_torsion[key].append([BBP_1,BBC_1,NX_1,C2_1])
            nt_name2 = list_nt[i+1][0].name[-1]
            key = nt_name2+"_PCNxC2"
            if (i+1 not in list_anchor_nt_index[0]) and (i+1 not in list_anchor_nt_index[1]):
                if key not in dict_torsion.keys():
                    dict_torsion[key] = [[BBP_2,BBC_2,NX_2,C2_2]]
                else:
                    dict_torsion[key].append([BBP_2,BBC_2,NX_2,C2_2])

        key = "CPC"
        if (i+1 not in list_anchor_nt_index[1]):
            if key not in dict_angle.keys():
                dict_angle[key] = [[BBC_2,BBP_3,BBC_3]]
            else:
                dict_angle[key].append([BBC_2,BBP_3,BBC_3])
        if i == 0:
            if (i not in list_anchor_nt_index[0]) and (i+1 not in list_anchor_nt_index[0]):
                if key not in dict_angle.keys():
                    dict_angle[key] = [[BBC_1,BBP_2,BBC_2]]
                else:
                    dict_angle[key].append([BBC_1,BBP_2,BBC_2])

        key = "PCP"
        if (i+1 not in list_anchor_nt_index[1]):
            if key not in dict_angle.keys():
                dict_angle[key] = [[BBP_2,BBC_2,BBP_3]]
            else:
                dict_angle[key].append([BBP_2,BBC_2,BBP_3])
        if i == 0:
            if (i not in list_anchor_nt_index[0]) and (i+1 not in list_anchor_nt_index[0]):
                if key not in dict_angle.keys():
                    dict_angle[key] = [[BBP_1,BBC_1,BBP_2]]
                else:
                    dict_angle[key].append([BBP_1,BBC_1,BBP_2])

        key = "PCNx"
        if i+2 not in list_anchor_nt_index[1]:
            if key not in dict_angle.keys():
                dict_angle[key] = [[BBP_3,BBC_3,NX_3]]
            else:
                dict_angle[key].append([BBP_3,BBC_3,NX_3])
        if i == 0:
            if (i not in list_anchor_nt_index[0]) and (i not in list_anchor_nt_index[1]):
                if key not in dict_angle.keys():
                    dict_angle[key] = [[BBP_1,BBC_1,NX_1]]
                else:
                    dict_angle[key].append([BBP_1,BBC_1,NX_1])
            if (i+1 not in list_anchor_nt_index[0]) and (i+1 not in list_anchor_nt_index[1]):
                if key not in dict_angle.keys():
                    dict_angle[key] = [[BBP_2,BBC_2,NX_2]]
                else:
                    dict_angle[key].append([BBP_2,BBC_2,NX_2])

        key = "NxCP"
        if (i+1 not in list_anchor_nt_index[1]):
            if key not in dict_angle.keys():
                dict_angle[key] = [[NX_2,BBC_2,BBP_3]]
            else:
                dict_angle[key].append([NX_2,BBC_2,BBP_3])
        if i == 0:
            if (i not in list_anchor_nt_index[0]) and (i+1 not in list_anchor_nt_index[0]):
                if key not in dict_angle.keys():
                    dict_angle[key] = [[NX_1,BBC_1,BBP_2]]
                else:
                    dict_angle[key].append([NX_1,BBC_1,BBP_2])

        key = nt_name+"_CNxC2"
        if i+2 not in list_anchor_nt_index[1]:
            if key not in dict_angle.keys():
                dict_angle[key] = [[BBC_3,NX_3,C2_3]]
            else:
                dict_angle[key].append([BBC_3,NX_3,C2_3])
        if i == 0:
            nt_name2 = list_nt[i][0].name[-1]
            key = nt_name2+"_CNxC2"
            if (i not in list_anchor_nt_index[0]) and (i not in list_anchor_nt_index[1]):
                if key not in dict_angle.keys():
                    dict_angle[key] = [[BBC_1,NX_1,C2_1]]
                else:
                    dict_angle[key].append([BBC_1,NX_1,C2_1])
            nt_name2 = list_nt[i+1][0].name[-1]
            key = nt_name2+"_CNxC2"
            if (i+1 not in list_anchor_nt_index[0]) and (i+1 not in list_anchor_nt_index[1]):
                if key not in dict_angle.keys():
                    dict_angle[key] = [[BBC_2,NX_2,C2_2]]
                else:
                    dict_angle[key].append([BBC_2,NX_2,C2_2])

        key = nt_name+"_CNxCy"
        if i+2 not in list_anchor_nt_index[1]:
            if key not in dict_angle.keys():
                dict_angle[key] = [[BBC_3,NX_3,CY_3]]
            else:
                dict_angle[key].append([BBC_3,NX_3,CY_3])
        if i == 0:
            nt_name2 = list_nt[i][0].name[-1]
            key = nt_name2+"_CNxCy"
            if (i not in list_anchor_nt_index[0]) and (i not in list_anchor_nt_index[1]):
                if key not in dict_angle.keys():
                    dict_angle[key] = [[BBC_1,NX_1,CY_1]]
                else:
                    dict_angle[key].append([BBC_1,NX_1,CY_1])
            nt_name2 = list_nt[i+1][0].name[-1]
            key = nt_name2+"_CNxCy"
            if (i+1 not in list_anchor_nt_index[0]) and (i+1 not in list_anchor_nt_index[1]):
                if key not in dict_angle.keys():
                    dict_angle[key] = [[BBC_2,NX_2,CY_2]]
                else:
                    dict_angle[key].append([BBC_2,NX_2,CY_2])

        nt_name2 = list_nt[i+1][0].name[-1]
        key = nt_name2+"_C2NxCP"
        if (i+1 not in list_anchor_nt_index[1]):
            if key not in dict_torsion.keys():
                dict_torsion[key] = [[C2_2,NX_2,BBC_2,BBP_3]]
            else:
                dict_torsion[key].append([C2_2,NX_2,BBC_2,BBP_3])
        if i == 0:
            nt_name2 = list_nt[i][0].name[-1]
            key = nt_name2+"_C2NxCP"
            if (i not in list_anchor_nt_index[0]) and (i+1 not in list_anchor_nt_index[0]):
                if key not in dict_torsion.keys():
                    dict_torsion[key] = [[C2_1,NX_1,BBC_1,BBP_2]]
                else:
                    dict_torsion[key].append([C2_1,NX_1,BBC_1,BBP_2])
    return dict_torsion, dict_angle


def get_angle_torsion_in_loops(dict_resid_chain,dict_residues,list_loops,list_loops_type):
    dict_torsion = {}
    dict_angle = {}

    for loop,loop_type in zip(list_loops,list_loops_type):
        dict_torsion0, dict_angle0 = get_angle_torsion_in_one_loop(dict_resid_chain,dict_residues,loop,loop_type)

        for key in dict_torsion0:
            if key not in dict_torsion.keys():
                dict_torsion[key] = dict_torsion0[key]
            else:
                dict_torsion[key].extend(dict_torsion0[key])

        for key in dict_angle0:
            if key not in dict_angle.keys():
                dict_angle[key] = dict_angle0[key]
            else:
                dict_angle[key].extend(dict_angle0[key])
    return dict_torsion, dict_angle


def load_angle_torsion_paras(paras_file):
    with open(paras_file) as f:
        lines = f.read().splitlines()
    paras = []
    for line in lines:
        line = line.split()[1:]
        paras.extend(line)
    paras = list(map(float,paras))
    return paras


def add_torsion_force(RNAJP_HOME,system,dict_torsion,wt_torsion,list_torsion_atoms_index_jar3d):
    torsion_list1 = ["CPCP","PCPC","CPCNx","NxCPC"]
    torsion_list2 = ["C2NxCP","PCNxC2","CNxC2Cy"]

    dict_torsion_paras = {}
    for t in torsion_list1:
        key = f"{t}"
        paras = load_angle_torsion_paras(f"{RNAJP_HOME}/angle_torsion_paras/paras_{key}.txt")
        dict_torsion_paras[key] = paras

    for nt in ["A","C","G","U"]:
        for t in torsion_list2:
            key = f"{nt}_{t}"
            paras = load_angle_torsion_paras(f"{RNAJP_HOME}/angle_torsion_paras/paras_{key}.txt")
            dict_torsion_paras[key] = paras

    ncos = 5
    logprob_func = []
    for i in range(1,ncos+1):
        func = f"amplitude{i}*cos(frequency{i}*t+shift{i})"
        logprob_func.append(func)
    logprob_func.append("c")
    logprob_func = " + ".join(logprob_func)
    cos_func = f"-ktorsion_global * RT_torsion * ({logprob_func}) * step(sinagl1-sinagl0) * step(sinagl2-sinagl0); t=dihedral(p1,p2,p3,p4); sinagl1=sin(angle(p1,p2,p3)); sinagl2=sin(angle(p2,p3,p4)); sinagl0=0.087155743" #5 < angle0 < 175

    dih_force = mm.CustomCompoundBondForce(4, cos_func)
    dih_force.addGlobalParameter("ktorsion_global",1.0)
    dih_force.addPerBondParameter("RT_torsion")
    for i in range(1,ncos+1):
        dih_force.addPerBondParameter(f'amplitude{i}')
    for i in range(1,ncos+1):
        dih_force.addPerBondParameter(f'frequency{i}')
    for i in range(1,ncos+1):
        dih_force.addPerBondParameter(f'shift{i}')
    dih_force.addPerBondParameter(f'c')

    for key in dict_torsion:
        if key not in dict_torsion_paras.keys():
            print ("no {key} para file")
            exit()
        for atoms_index in dict_torsion[key]:
            if list_torsion_atoms_index_jar3d is not None:
                if (atoms_index in list_torsion_atoms_index_jar3d) or (atoms_index[::-1] in list_torsion_atoms_index_jar3d):
                    continue
            if key in ["CPCP","PCPC","CPCNx","NxCPC","A_C2NxCP","G_C2NxCP","C_C2NxCP","U_C2NxCP"]:
                RT = wt_torsion*2.494*unit.kilojoule/unit.mole
            else:
                RT = 2.494*unit.kilojoule/unit.mole
            dih_force.addBond(atoms_index,[RT]+dict_torsion_paras[key])

    dih_force.setForceGroup(5)
    system.addForce(dih_force)
    return system


def add_angle_force(RNAJP_HOME,system,dict_angle,wt_angle,list_angle_atoms_index_jar3d):
    angle_list1 = ["PCP","CPC","PCNx","NxCP"]
    angle_list2 = ["CNxC2","CNxCy"]

    dict_angle_paras = {}

    for t in angle_list1:
        key = f"{t}"
        paras = load_angle_torsion_paras(f"{RNAJP_HOME}/angle_torsion_paras/paras_{key}.txt")
        dict_angle_paras[key] = paras

    for nt in ["A","C","G","U"]:
        for t in angle_list2:
            key = f"{nt}_{t}"
            paras = load_angle_torsion_paras(f"{RNAJP_HOME}/angle_torsion_paras/paras_{key}.txt")
            dict_angle_paras[key] = paras

    ngaussian = 5
    logprob_func = []
    for i in range(1,ngaussian+1):
        func = f"amplitude{i} / (sqrt(2*3.1415926535)*sigma{i}) * exp(-(t-center{i})^2/(2*sigma{i}^2))"
        logprob_func.append(func)
    func = f"a*t^2+b*t+c"
    logprob_func.append(func)
    logprob_func = " + ".join(logprob_func)
    gau_qua_func = f"-kangle_global * RT_angle * ({logprob_func}) + 100*step(lowerangle-t)*(t-lowerangle)^2 + 100*step(t-upperangle)*(t-upperangle)^2" # 5 gaussian + 1 quadratic function
    gau_qua_func += "; t=angle(p1,p2,p3)"

    agl_force = mm.CustomCompoundBondForce(3, gau_qua_func)
    agl_force.addGlobalParameter("kangle_global",1.0)
    agl_force.addPerBondParameter("RT_angle")
    for i in range(1,ngaussian+1):
        agl_force.addPerBondParameter(f'amplitude{i}')
    for i in range(1,ngaussian+1):
        agl_force.addPerBondParameter(f'center{i}')
    for i in range(1,ngaussian+1):
        agl_force.addPerBondParameter(f'sigma{i}')
    agl_force.addPerBondParameter(f'a')
    agl_force.addPerBondParameter(f'b')
    agl_force.addPerBondParameter(f'c')
    agl_force.addPerBondParameter(f'lowerangle')
    agl_force.addPerBondParameter(f'upperangle')

    for key in dict_angle:
        if key not in dict_angle_paras.keys():
            print ("no {key} para file")
            exit()
        if key == "PCP":
            lowerangle = 1.396263402 # 80 degree
            upperangle = 2.617993878 # 150 degree
        elif key == "CPC":
            lowerangle = 1.396263402 # 80 degree
            upperangle = 2.443460953 # 140 degee
        elif key == "PCNx":
            lowerangle = 1.396263402 # 80 degree
            upperangle = 2.443460953 # 140 degree
        elif key == "NxCP":
            lowerangle = 1.396263402 # 80 degree
            upperangle = 2.443460953 # 140 degree
        elif key in ["A_CNxC2","G_CNxC2"] :
            lowerangle = 1.396263402 # 80 degree
            upperangle = 2.705260341 # 155 degree
        elif key in ["C_CNxC2","U_CNxC2"] :
            lowerangle = 1.483529864 # 85 degree
            upperangle = 2.705260341 # 155 degree
        elif key in ["A_CNxCy","G_CNxCy"]:
            lowerangle = 2.094395102 # 120 degree
            upperangle = 3.054326191 # 175 degree
        elif key in ["C_CNxCy","U_CNxCy"]:
            lowerangle = 2.35619449 # 135 degree
            upperangle = 2.792526803 # 160 degree
        else:
            print ("Wrong angle name: ",key)
            exit()

        for atoms_index in dict_angle[key]:
            #print (key,atoms_index)
            if list_angle_atoms_index_jar3d is not None:
                if (atoms_index in list_angle_atoms_index_jar3d) or (atoms_index[::-1] in list_angle_atoms_index_jar3d):
                    #print ("The following angle force is skipped:",atoms_index)
                    continue
            if key in ["PCP","CPC"]:
                RT = wt_angle*2.494*unit.kilojoule/unit.mole
            else:
                RT = 2.494*unit.kilojoule/unit.mole
           
            agl_force.addBond(atoms_index,[RT]+dict_angle_paras[key]+[lowerangle,upperangle])

    agl_force.setForceGroup(4)
    system.addForce(agl_force)
    return system


def add_torsion_and_angle_force_to_system(RNAJP_HOME, topology, system, dict_motifs, dict_resid_chain, wt_torsion, wt_angle, list_torsion_atoms_index_jar3d=None, list_angle_atoms_index_jar3d=None):
    dict_residues = access_to_residues_by_chain_name_and_resid(topology)
    list_loops = []
    list_loops_type = []
    for key in dict_motifs:
        if key == "helix":
            continue
        if key == "PK_helix":
            continue
        if "non_PK" in key:
            continue
        motifs = dict_motifs[key]
        for motif in motifs:
            for loop in motif:
                list_loops.append(loop)
                list_loops_type.append(key)

    dict_torsion, dict_angle = get_angle_torsion_in_loops(dict_resid_chain,dict_residues,list_loops,list_loops_type)
    system = add_angle_force(RNAJP_HOME,system,dict_angle,wt_angle,list_angle_atoms_index_jar3d)
    system = add_torsion_force(RNAJP_HOME,system,dict_torsion,wt_torsion,list_torsion_atoms_index_jar3d)

    list_loops = []
    for key in ["5end_loops_non_PK","3end_loops_non_PK"]:
        motifs = dict_motifs[key]
        for motif in motifs:
            list_loops.append(motif)
    if list_loops:
        system, index_helical_torsion_force, list_helical_torsion_bonds_in_junctions = add_helical_torsion_force_in_junctions(topology,system,dict_resid_chain,list_loops,khelical=100.0)
    return system


def get_torsion_for_one_loop_in_junction(dict_resid_chain,dict_residues,loop):
    loop_start_id = loop[0]
    loop_end_id = loop[1]

    list_nt = [] #include four nts in two helices if any
    list_anchor_nt_index = [[],[]]
    num = 0
    for i in range(loop_start_id-1,loop_end_id+2):
        if str(i) in dict_resid_chain.keys():
            chain_ntidx = dict_resid_chain[str(i)]
            nt = dict_residues[chain_ntidx]
            list_nt.append(nt)
            if i in [loop_start_id-1,loop_start_id]:
                list_anchor_nt_index[0].append(num)
            elif i in [loop_end_id,loop_end_id+1]:
                list_anchor_nt_index[1].append(num)
            num += 1

    dict_torsion = {"CPCP":[],"PCPC":[],"CPCNx":[]}
    for i in range(len(list_nt)-2):
        BBP_1 = list_nt[i][1]["BBP"].index
        BBC_1 = list_nt[i][1]["BBC"].index
        NX_1 = list_nt[i][1]["NX"].index
        C2_1 = list_nt[i][1]["C2"].index
        CY_1 = list_nt[i][1]["CY"].index

        BBP_2 = list_nt[i+1][1]["BBP"].index
        BBC_2 = list_nt[i+1][1]["BBC"].index
        NX_2 = list_nt[i+1][1]["NX"].index
        C2_2 = list_nt[i+1][1]["C2"].index
        CY_2 = list_nt[i+1][1]["CY"].index
        
        BBP_3 = list_nt[i+2][1]["BBP"].index
        BBC_3 = list_nt[i+2][1]["BBC"].index
        NX_3 = list_nt[i+2][1]["NX"].index
        C2_3 = list_nt[i+2][1]["C2"].index
        CY_3 = list_nt[i+2][1]["CY"].index

        key = "CPCP"
        if key not in dict_torsion.keys():
            dict_torsion[key] = [[BBC_1,BBP_2,BBC_2,BBP_3]]
        else:
            dict_torsion[key].append([BBC_1,BBP_2,BBC_2,BBP_3])

        key = "PCPC"
        if (i+1 not in list_anchor_nt_index[1]):
            if key not in dict_torsion.keys():
                dict_torsion[key] = [[BBP_2,BBC_2,BBP_3,BBC_3]]
            else:
                dict_torsion[key].append([BBP_2,BBC_2,BBP_3,BBC_3])
        if i == 0:
            if (i not in list_anchor_nt_index[0]) and (i+1 not in list_anchor_nt_index[0]):
                if key not in dict_torsion.keys():
                    dict_torsion[key] = [[BBP_1,BBC_1,BBP_2,BBC_2]]
                else:
                    dict_torsion[key].append([BBP_1,BBC_1,BBP_2,BBC_2])

        key = "CPCNx"
        if (i+1 not in list_anchor_nt_index[1]):
            if key not in dict_torsion.keys():
                dict_torsion[key] = [[BBC_2,BBP_3,BBC_3,NX_3]]
            else:
                dict_torsion[key].append([BBC_2,BBP_3,BBC_3,NX_3])
        if i == 0:
            if (i not in list_anchor_nt_index[0]) and (i+1 not in list_anchor_nt_index[0]):
                if key not in dict_torsion.keys():
                    dict_torsion[key] = [[BBC_1,BBP_2,BBC_2,NX_2]]
                else:
                    dict_torsion[key].append([BBC_1,BBP_2,BBC_2,NX_2])
    return dict_torsion


def get_torsion_for_loops_in_junctions(dict_resid_chain,dict_residues,list_junctions):
    list_bkbone_torsion_in_junctions = []
    for junction in list_junctions:
        list_bkbone_torsion_in_one_junction = []
        for loop in junction:
            dict_bkbone_torsion = get_torsion_for_one_loop_in_junction(dict_resid_chain,dict_residues,loop)
            pcpc = dict_bkbone_torsion["PCPC"]
            cpcp = dict_bkbone_torsion["CPCP"]
            cpcn = dict_bkbone_torsion["CPCNx"]
            loop_len = loop[1] - loop[0] - 1
            list_bkbone_torsion_in_one_junction.append([pcpc,cpcp,cpcn,loop_len])
        list_bkbone_torsion_in_junctions.append(list_bkbone_torsion_in_one_junction)

    return list_bkbone_torsion_in_junctions


def add_helical_torsion_force_in_junctions(topology,system,dict_resid_chain,list_junctions,khelical=0.0):
    dict_residues = access_to_residues_by_chain_name_and_resid(topology)
    list_bkbone_torsion_in_junctions = get_torsion_for_loops_in_junctions(dict_resid_chain,dict_residues,list_junctions)

    quadratic_function = f"step(sin(agl1)-sin(agl0))*step(sin(agl2)-sin(agl0))*step(deltaT-deltaT0)*khelical*(deltaT-deltaT0)^2 + 100*step(lowerangle-agl2)*(agl2-lowerangle)^2 + 100*step(agl2-upperangle)*(agl2-upperangle)^2; deltaT=min(dt,2*pi-dt); dt=abs(theta-theta0); pi = 3.1415926535; theta=dihedral(p1,p2,p3,p4); deltaT0=0.436332313; agl1=angle(p1,p2,p3); agl2=angle(p2,p3,p4); agl0=3.054326191" #deltaT0 = 25 degree; 5 < angle1/2 < 175
    helical_torsion_force = mm.CustomCompoundBondForce(4, quadratic_function)
    helical_torsion_force.addPerBondParameter("khelical")
    helical_torsion_force.addPerBondParameter("theta0")
    helical_torsion_force.addPerBondParameter("lowerangle")
    helical_torsion_force.addPerBondParameter("upperangle")

    list_helical_torsion_bonds_in_junctions = []
    for i in range(len(list_bkbone_torsion_in_junctions)):
        list_bonds_in_one_junction = []
        for j in range(len(list_bkbone_torsion_in_junctions[i])):
            pcpc = list_bkbone_torsion_in_junctions[i][j][0]
            cpcp = list_bkbone_torsion_in_junctions[i][j][1]
            cpcn = list_bkbone_torsion_in_junctions[i][j][2]
            loop_len = list_bkbone_torsion_in_junctions[i][j][3]
            list_bonds = []
            for atoms_index in pcpc:
                if loop_len <= 5:
                    lowerangle = np.radians(85) # 85 degree
                    upperangle = np.radians(105) # 105 degree
                else:
                    lowerangle = np.radians(80)  # 80 degree
                    upperangle = np.radians(120) # 120 degree
                index_bond = helical_torsion_force.addBond(atoms_index,[100.0,np.radians(-150.1),lowerangle,upperangle])
                bond_info = [index_bond,atoms_index,[khelical,np.radians(-150.1),lowerangle,upperangle]]
                list_bonds.append(bond_info)
            for atoms_index in cpcp:
                if loop_len <= 5:
                    lowerangle = np.radians(85) # 85 degree
                    upperangle = np.radians(105) # 105 degree
                else:
                    lowerangle = np.radians(80)  # 80 degree
                    upperangle = np.radians(120) # 120 degree
                index_bond = helical_torsion_force.addBond(atoms_index,[100.0,np.radians(170.0),lowerangle,upperangle])
                bond_info = [index_bond,atoms_index,[khelical,np.radians(170.0),lowerangle,upperangle]]
                list_bonds.append(bond_info)
            for atoms_index in cpcn:
                if loop_len <= 5:
                    lowerangle = np.radians(85) # 85 degree
                    upperangle = np.radians(105) # 105 degree
                else:
                    lowerangle = np.radians(80)  # 80 degree
                    upperangle = np.radians(120) # 120 degree
                index_bond = helical_torsion_force.addBond(atoms_index,[100.0,np.radians(76.4),lowerangle,upperangle])
                bond_info = [index_bond,atoms_index,[khelical,np.radians(76.4),lowerangle,upperangle]]
                list_bonds.append(bond_info)
            if list_bonds:
                list_bonds_in_one_junction.append(list_bonds)
        if list_bonds_in_one_junction:
            list_helical_torsion_bonds_in_junctions.append(list_bonds_in_one_junction)

    helical_torsion_force.setForceGroup(10)
    index_helical_torsion_force = system.addForce(helical_torsion_force)
    return system, index_helical_torsion_force, list_helical_torsion_bonds_in_junctions
