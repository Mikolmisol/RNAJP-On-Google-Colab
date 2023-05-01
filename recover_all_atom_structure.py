import sys
import numpy as np
from numpy import dot, transpose, sqrt
from numpy.linalg import svd, det


def get_rotran(reference_coords, coords):
    # center on centroid
    n = reference_coords.shape[0]
    av1 = sum(coords) / n
    av2 = sum(reference_coords) / n
    coords = coords - av1
    reference_coords = reference_coords - av2
    # correlation matrix
    a = dot(transpose(coords), reference_coords)
    u, d, vt = svd(a)
    rot = transpose(dot(transpose(vt), transpose(u)))
    # check if we have found a reflection
    if det(rot) < 0:
        vt[2] = -vt[2]
        rot = transpose(dot(transpose(vt), transpose(u)))
    tran = av2 - dot(av1, rot)
    return rot, tran


def calc_rmsd(reference_coords, coords):
    rot, tran = get_rotran(reference_coords, coords)
    transformed_coords = dot(coords, rot) + tran

    diff = transformed_coords - reference_coords
    rmsd = sqrt(sum(sum(diff * diff)) / reference_coords.shape[0])
    return rmsd, rot, tran


def get_transformed(rot, tran, coords):
    #Get the transformed coordinate set
    transformed_coords = dot(coords, rot) + tran
    return transformed_coords


def get_template_nucleotide(RNAJP_HOME):
    list_backbone_atoms = ["P","OP1","OP2","O5'","C5'","C4'","C3'","O3'","C2'","O2'","C1'","O4'","NX","P2"]
    with open(f"{RNAJP_HOME}/nucleotide_templates_cg_to_aa/templates_backbone.txt") as f:
        lines = f.read().splitlines()
    list_backbone_templates = []
    for line in lines:
        line = line.split()
        dict_backbone_template = {}
        for i,atom_name in enumerate(list_backbone_atoms):
            coord = np.asarray(line[i*3:(i+1)*3],dtype=float)
            dict_backbone_template[atom_name] = coord
        list_backbone_templates.append(dict_backbone_template)

    dict_base_templates = {"Abase":{},"Gbase":{},"Cbase":{},"Ubase":{}}
    with open(f"{RNAJP_HOME}/nucleotide_templates_cg_to_aa/template_Abase.pdb") as f:
        lines = f.read().splitlines()
    for line in lines:
        line = line.split()
        if line[0] != "ATOM":
            continue
        atom_name = line[2]
        coord = np.asarray(line[6:9],dtype=float)
        dict_base_templates["Abase"][atom_name] = coord

    with open(f"{RNAJP_HOME}/nucleotide_templates_cg_to_aa/template_Gbase.pdb") as f:
        lines = f.read().splitlines()
    for line in lines:
        line = line.split()
        if line[0] != "ATOM":
            continue
        atom_name = line[2]
        coord = np.asarray(line[6:9],dtype=float)
        dict_base_templates["Gbase"][atom_name] = coord

    with open(f"{RNAJP_HOME}/nucleotide_templates_cg_to_aa/template_Cbase.pdb") as f:
        lines = f.read().splitlines()
    for line in lines:
        line = line.split()
        if line[0] != "ATOM":
            continue
        atom_name = line[2]
        coord = np.asarray(line[6:9],dtype=float)
        dict_base_templates["Cbase"][atom_name] = coord

    with open(f"{RNAJP_HOME}/nucleotide_templates_cg_to_aa/template_Ubase.pdb") as f:
        lines = f.read().splitlines()
    for line in lines:
        line = line.split()
        if line[0] != "ATOM":
            continue
        atom_name = line[2]
        coord = np.asarray(line[6:9],dtype=float)
        dict_base_templates["Ubase"][atom_name] = coord

    return list_backbone_templates, dict_base_templates


def get_allatom_nucleotide(list_backbone_templates, dict_base_templates, nt_name, dict_nt, dict_nt2):
    if nt_name in ["A","G"]:
        NX = "N9"
        C2 = "C2"
        CY = "C6"
    elif nt_name in ["C","U"]:
        NX = "N1"
        C2 = "C2"
        CY = "C4"
    else:
        print ("wrong nt name",nt_name)
        exit()

    dict_atoms = dict_nt["CG_coord"]
    vec_P = dict_atoms["BBP"]
    vec_C4s = dict_atoms["BBC"]
    vec_NX = dict_atoms[nt_name+NX]
    vec_C2 = dict_atoms[nt_name+C2]
    vec_CY = dict_atoms[nt_name+CY]

    if dict_nt2 is None:
        reference_coords = np.asarray([vec_P,vec_C4s,vec_NX])
        best_rmsd = 10000.
        for tp in list_backbone_templates:
            coords = np.asarray([tp["P"], tp["C4'"], tp["NX"]])
            rmsd, rot, tran = calc_rmsd(reference_coords, coords)
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_rot = rot
                best_tran = tran
                best_backbone_template = tp
    else:
        vec_P2 = dict_nt2["CG_coord"]["BBP"]
        reference_coords = np.asarray([vec_P,vec_C4s,vec_NX,vec_P2])
        best_rmsd = 10000.
        for tp in list_backbone_templates:
            coords = np.asarray([tp["P"], tp["C4'"], tp["NX"],tp["P2"]])
            rmsd, rot, tran = calc_rmsd(reference_coords, coords)
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_rot = rot
                best_tran = tran
                best_backbone_template = tp
   
    dict_nt["AA_coord"] = {}
    for atom_name in best_backbone_template:
        if atom_name == "P2":
            continue
        vec = best_backbone_template[atom_name]
        transformed_vec = get_transformed(best_rot, best_tran, vec)
        if atom_name == "NX":
            atom_name = NX
        dict_nt["AA_coord"][atom_name] = transformed_vec

    reference_coords = np.asarray([vec_NX,vec_C2,vec_CY])
    coords = np.asarray([dict_base_templates[nt_name+"base"][NX], dict_base_templates[nt_name+"base"][C2],dict_base_templates[nt_name+"base"][CY]])
    rot, tran = get_rotran(reference_coords, coords)
    for atom_name in dict_base_templates[nt_name+"base"]:
        vec = dict_base_templates[nt_name+"base"][atom_name]
        transformed_vec = get_transformed(rot,tran,vec)
        dict_nt["AA_coord"][atom_name] = transformed_vec
    return dict_nt


def get_allatom_structure(pdbfile, list_backbone_templates, dict_base_templates):
    with open(pdbfile) as f:
        lines = f.read().splitlines()

    list_models = []
    dict_model = {}
    for line in lines:
        if line[0:6] == "ENDMDL":
            if dict_model:
                list_models.append(dict_model)
            dict_model = {}
        elif line[0:4] == "ATOM" or line[0:6] == "HETATM":
            atom_name = line[12:16].strip()
            nt_name = line[17:20].strip()[-1]
            chain = line[21]
            nt_id = int(line[22:26])
            vx = float(line[30:38])
            vy = float(line[38:46])
            vz = float(line[46:54])
            coord = np.asarray([vx,vy,vz])

            if chain not in dict_model.keys():
                dict_model[chain] = {}

            nt_key = (chain, nt_id, nt_name)
            if nt_key not in dict_model[chain].keys():
                dict_model[chain][nt_key] = {}

            if "CG_coord" not in dict_model[chain][nt_key].keys():
                dict_model[chain][nt_key]["CG_coord"] = {}
            dict_model[chain][nt_key]["CG_coord"][atom_name] = coord
        else:
            continue

    if dict_model:
        list_models.append(dict_model)

    for i, model in enumerate(list_models):
        #print(f"CG to AA: model {i+1}",flush=True)
        for chain in model:
            for nt_key in model[chain]:
                nt_name = nt_key[2]
                nt = model[chain][nt_key]
                nt2 = None
                for nt2_name in ["A","G","C","U"]:
                    nt2_key = (chain,nt_key[1]+1,nt2_name)
                    if nt2_key in model[chain].keys():
                        nt2 = model[chain][nt2_key]
                        break
                new_nt = get_allatom_nucleotide(list_backbone_templates, dict_base_templates, nt_name, nt, nt2)
                list_models[i][chain][nt_key] = new_nt
    return list_models


def write_allatom_structure(list_models, outfile):
    f = open(outfile,"w")
    atom_id = 1
    for i, model in enumerate(list_models):
        f.write(f"MODEL{i+1:>9d}\n")
        for chain in model:
            for nt_key in model[chain]:
                nt_id = nt_key[1]
                nt_name = nt_key[2]
                aa_atoms = model[chain][nt_key]["AA_coord"]
                for atom_name in aa_atoms:
                    atom_coord = aa_atoms[atom_name]
                    element = atom_name[0]                    
                    f.write(f"ATOM  {atom_id:>5d}  {atom_name:<3} {nt_name:>3} {chain}{nt_id:>4d}    {atom_coord[0]:>8.3f}{atom_coord[1]:>8.3f}{atom_coord[2]:>8.3f}{1.00:>6.2f}{1.00:>6.2f}{element:>12}  \n")
                    atom_id += 1
            f.write("TER\n")
        f.write("ENDMDL\n")
    f.write("END\n")
    f.close()


def convert_cg_to_aa(RNAJP_HOME, cg_file, aa_file):
    list_backbone_templates, dict_base_templates = get_template_nucleotide(RNAJP_HOME)
    list_models = get_allatom_structure(cg_file, list_backbone_templates, dict_base_templates)
    write_allatom_structure(list_models, aa_file)
