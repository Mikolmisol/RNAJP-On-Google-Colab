import numpy as np


def get_decimal_num(float_num):
    if float_num > 99999999:
        print("Error: coord > 99999999")
        exit()
    if float_num < -9999999:
        print("Error: coord < -9999999")
        exit()

    if float_num >= 0:
        if 0 < float_num//10000000 < 9:
            return 0
        elif 0 < float_num//1000000 < 9:
            return 0
        elif 0 < float_num//100000 < 9:
            return 1
        elif 0 < float_num//10000 < 9:
            return 2
        else:
            return 3

    if float_num < 0:
        float_num = abs(float_num)
        if 0 < float_num//1000000 < 9:
            return 0
        elif 0 < float_num//100000 < 9:
            return 0
        elif 0 < float_num//10000 < 9:
            return 1
        elif 0 < float_num//1000 < 9:
            return 2
        else:
            return 3


def get_cg_atom_info_in_pdb(atom_name, atom_coord, atom_id, nt_id, nt_name, chain_name):
    if atom_name == "BBP":
        tag = " P"
    elif atom_name == "BBC":
        tag = " C"
    elif atom_name in ["AN9","AC2","AC6"]:
        tag = "Au"
    elif atom_name in ["CN1","CC2","CC4"]:
        tag = "Ag"
    elif atom_name in ["GN9","GC2","GC6"]:
        tag = "Cu"
    elif atom_name in ["UN1","UC2","UC4"]:
        tag = "Fe"

    coordx = atom_coord[0]
    decimal_num = get_decimal_num(coordx)
    if decimal_num == 0:
        coordx = f"{coordx:>8d}"
    elif decimal_num == 1:
        coordx = f"{coordx:>8.1f}"
    elif decimal_num == 2:
        coordx = f"{coordx:>8.2f}"
    elif decimal_num == 3:
        coordx = f"{coordx:>8.3f}"

    coordy = atom_coord[1]
    decimal_num = get_decimal_num(coordy)
    if decimal_num == 0:
        coordy = f"{coordy:>8d}"
    elif decimal_num == 1:
        coordy = f"{coordy:>8.1f}"
    elif decimal_num == 2:
        coordy = f"{coordy:>8.2f}"
    elif decimal_num == 3:
        coordy = f"{coordy:>8.3f}"

    coordz = atom_coord[2]
    decimal_num = get_decimal_num(coordz)
    if decimal_num == 0:
        coordz = f"{coordz:>8d}"
    elif decimal_num == 1:
        coordz = f"{coordz:>8.1f}"
    elif decimal_num == 2:
        coordz = f"{coordz:>8.2f}"
    elif decimal_num == 3:
        coordz = f"{coordz:>8.3f}"

    pdb_atom_info = f"ATOM  {atom_id:>5}  {atom_name:<3} {nt_name:>3} {chain_name}{nt_id:>4}    {coordx}{coordy}{coordz}  1.00  0.00          {tag}"
    return pdb_atom_info


def calc_fourth_atom_coord(r1,r2,r3,dis,agl,dih):
    r1 = np.asarray(r1)
    r2 = np.asarray(r2)
    r3 = np.asarray(r3)
    r12 = r2-r1
    r23 = r3-r2
    j = r23/np.linalg.norm(r23)
    k = np.cross(r12,r23)
    k = k/np.linalg.norm(k)
    i = np.cross(j,k)
    if abs(np.linalg.norm(i)-1.0) > 1e-5:
        print(f"The norm of unit vector {np.linalg.norm(i)} != 1.0")
        exit(0)
    agl = np.pi - agl
    a = -np.cos(dih)*np.sin(agl)
    b = np.cos(agl)
    c = np.sin(dih)*np.sin(agl)
    r4 = r3 + (i*a+j*b+k*c)*dis
    return list(r4)


def generate_circular_initial_structure(seq,outpdb):
    seq = seq.strip()
    seq = seq.upper()
    rnalen = len(seq) - seq.count(" ")
    seqlen = len(seq) # rna length + num_chains
    list_nt_coords= []

    d1 = 6.0 # C4'-C4' distance
    d2 = 3.9 # P-C4' distance
    theta = np.pi/rnalen
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    sinbeta = d1/2/d2
    cosbeta = np.sqrt(1-sinbeta**2)
    cosalpha = -costheta*cosbeta + sintheta*sinbeta
    radius = d1/2/np.sin(theta)
    op = np.sqrt(d2**2+radius**2-2*d2*radius*cosalpha)

    for n in range(rnalen):
        pn_x = op*np.cos(theta-2*n*theta)
        pn_y = op*np.sin(theta-2*n*theta)
        c4_x = radius*np.cos(-2*n*theta)
        c4_y = radius*np.sin(-2*n*theta)
        coords = [[pn_x,pn_y,0,"BBP"],[c4_x,c4_y,0,"BBC"]]
        list_nt_coords.append(coords)

    n = 0
    for i in range(seqlen):
        if seq[i] == " ":
            continue

        coords1 = list_nt_coords[n]
        if n == rnalen - 1:
            coords2 = list_nt_coords[0]
        else:
            coords2 = list_nt_coords[n+1]
        p_1 = coords1[0][0:3]
        c4_1 = coords1[1][0:3]
        p_2 = coords2[0][0:3]

        dis_c4_nx = 3.4
        agl_p_c4_nx = np.radians(92.9)
        dih_p_p_c4_nx = np.radians(180.0)

        nx_1 = calc_fourth_atom_coord(p_2,p_1,c4_1,dis_c4_nx,agl_p_c4_nx,dih_p_p_c4_nx)
        if seq[i] in "AG":
            nx_name = seq[i]+"N9"
            c2_name = seq[i]+"C2"
            cy_name = seq[i]+"C6"
            dis_nx_c2 = 3.5
            dis_c2_cy = 2.4
            agl_c4_nx_c2 = np.radians(90)
            agl_nx_c2_cy = np.radians(69.6)
            dih_p_c4_nx_c2 = np.radians(90.0)
            dih_c4_nx_c2_cy = np.radians(-90.0)
        else:
            nx_name = seq[i]+"N1"
            c2_name = seq[i]+"C2"
            cy_name = seq[i]+"C4"
            dis_nx_c2 = 1.4
            dis_c2_cy = 2.4
            agl_c4_nx_c2 = np.radians(90)
            agl_nx_c2_cy = np.radians(88.7)
            dih_p_c4_nx_c2 = np.radians(90.0)
            dih_c4_nx_c2_cy = np.radians(-90.0)

        c2_1 = calc_fourth_atom_coord(p_1,c4_1,nx_1,dis_nx_c2,agl_c4_nx_c2,dih_p_c4_nx_c2)
        cy_1 = calc_fourth_atom_coord(c4_1,nx_1,c2_1,dis_c2_cy,agl_nx_c2_cy,dih_c4_nx_c2_cy)

        nx_1.append(nx_name)
        c2_1.append(c2_name)
        cy_1.append(cy_name)

        list_nt_coords[n].append(nx_1)
        list_nt_coords[n].append(c2_1)
        list_nt_coords[n].append(cy_1)
        n += 1

    list_pdb_atom_info = []
    atom_id = 0
    nt_id = 0
    nt_real_id = 0
    chain_name = "A"
    for i in range(len(seq)):
        if seq[i] == " ":
            chain_name = chr(ord(chain_name)+1)
            nt_id = 0
            continue
        nt_name = "CG"+seq[i]
        nt_coords = list_nt_coords[nt_real_id]
        nt_id += 1
        nt_real_id += 1
        for coord in nt_coords:
            atom_name = coord[3]
            atom_coord = coord[0:3]
            atom_id += 1
            pdb_atom_info = get_cg_atom_info_in_pdb(atom_name, atom_coord, atom_id, nt_id, nt_name, chain_name)
            list_pdb_atom_info.append(pdb_atom_info)

    with open(outpdb,"w") as f:
        for pdb_atom_info in list_pdb_atom_info:
           f.write(pdb_atom_info+"\n")
