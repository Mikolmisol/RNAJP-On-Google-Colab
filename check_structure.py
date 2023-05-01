import sys
import numpy as np
from numpy import dot, transpose, sqrt
from numpy.linalg import svd, det


def read_pdb(pdbname):
    with open(pdbname) as f:
        lines = f.read().splitlines()

    dict_nts = {}
    last_nt_chain = None
    last_nt_id = None
    nt_idx = 0
    for line in lines:
        if line[0:6] == "ENDMDL":
            break
        if len(line) < 54:
            continue
        if (line[0:4] != "ATOM") and (line[0:6] != "HETATM"):
            continue
        atom_name = line[12:16].strip()
        nt_name = line[17:20].strip()
        chain = line[21]
        nt_id = line[22:26].strip()
        x = line[30:38].strip()
        y = line[38:46].strip()
        z = line[46:54].strip()
        x = float(x)
        y = float(y)
        z = float(z)
        coord = np.array([x,y,z])
        atom = {}
        atom["nt_name"] = nt_name
        atom["chain"] = chain
        atom["nt_id"] = nt_id
        atom["coord"] = coord

        if chain != last_nt_chain or nt_id != last_nt_id:
            last_nt_chain = chain
            last_nt_id = nt_id
            nt_idx += 1

        alias_name = None
        if "N1" in atom_name or "N9" in atom_name:
            alias_name = "NX"
        elif "C2" in atom_name:
            alias_name = "C2"
        elif "C4" in atom_name or "C6" in atom_name:
            alias_name = "CY"
        if nt_idx not in dict_nts.keys():
            dict_nts[nt_idx] = {}
            dict_nts[nt_idx][atom_name] = atom
            if alias_name is not None:
                dict_nts[nt_idx][alias_name] = atom
        else:
            if atom_name in dict_nts[nt_idx].keys():
                raise ValueError(f"Two atoms have the same name in nucleotide {nt_idx}")
            dict_nts[nt_idx][atom_name] = atom
            if alias_name is not None:
                dict_nts[nt_idx][alias_name] = atom

    return dict_nts


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
    return rmsd


def calc_rmsd_for_two_base_pairs_in_helix(nt1,nt2,nt3,nt4):
    coords = []
    for nt in [nt1,nt2,nt3,nt4]:
        for name in ["BBP","BBC","NX"]:
            coord = nt[name]["coord"]
            coords.append(coord)
    coords = np.asarray(coords)

    coord_P_1 = [3.750, 3.986, 8.742]
    coord_C4s_1 = [6.895, 6.263, 7.851]
    coord_NX_1 = [6.798, 5.461, 4.615]
    coord_P_2 = [4.485, 9.314, 7.401]
    coord_C4s_2 = [6.421, 11.385, 4.731]
    coord_NX_2 = [5.502, 9.371, 2.227]
    coord_P_3 = [6.827, 4.154, -9.040]
    coord_C4s_3 = [9.111, 6.767, -7.210]
    coord_NX_3 = [7.675, 6.169, -4.172]
    coord_P_4 = [1.483, 2.766, -9.083]
    coord_C4s_4 = [3.435, 6.161, -9.236]
    coord_NX_4 = [3.270, 6.627, -5.975]
    reference_coords = [coord_P_1,coord_C4s_1,coord_NX_1,coord_P_2,coord_C4s_2,coord_NX_2,coord_P_3,coord_C4s_3,coord_NX_3,coord_P_4,coord_C4s_4,coord_NX_4]
    reference_coords = np.asarray(reference_coords)
    rmsd = calc_rmsd(reference_coords,coords)
    return rmsd


def check_helix(dict_motifs,dict_nts,check_PK_helix):
    if check_PK_helix:
        list_helices = dict_motifs["helix"] + dict_motifs["PK_helix"]
    else:
        list_helices = dict_motifs["helix"]
    for helix in list_helices:
        h1, h4 = helix[0]
        h2, h3 = helix[1]
        left_strand = list(range(h1,h2+1))
        right_strand = list(range(h4,h3-1,-1))
        for i in range(len(left_strand)-1):
            nt1 = dict_nts[left_strand[i]]
            nt2 = dict_nts[left_strand[i+1]]
            nt3 = dict_nts[right_strand[i]]
            nt4 = dict_nts[right_strand[i+1]]
            rmsd = calc_rmsd_for_two_base_pairs_in_helix(nt1,nt2,nt3,nt4)
            if rmsd > 1.2:
                return False
    return True


def check_pdb(dict_motifs,pdbname,check_PK_helix):
    dict_nts = read_pdb(pdbname)

    pass_check = check_helix(dict_motifs,dict_nts,check_PK_helix)
    if not pass_check:
        print("Do not pass helix check!")
        return False
    else:
        print("Pass helix check!")

    return True
