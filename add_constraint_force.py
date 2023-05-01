import os
import numpy as np
import simtk.openmm as mm
import simtk.unit as unit
import numpy.linalg as lin
import copy
from itertools import combinations
from add_non_local_force import get_atoms_index_between_two_nts, load_bp_stk_paras

def cross_product(r1, r2):
    return ((r1[1]*r2[2]-r1[2]*r2[1], r1[2]*r2[0]-r1[0]*r2[2], r1[0]*r2[1]-r1[1]*r2[0]))


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
            #print (res.id, res.name, res.chain.id, type(res.chain.id))
    return dict_residues


def fix_helices(topology, system, dict_resid_chain, positions, list_helices, list_PK_helices=[]):
    dict_residues = access_to_residues_by_chain_name_and_resid(topology)

    npower_bond_force = mm.CustomBondForce('kbond*(r-r0)^2')
    npower_bond_force.addPerBondParameter('kbond') 
    npower_bond_force.addPerBondParameter('r0')

    list_all_helices = list_helices + list_PK_helices

    for helix in list_all_helices:
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

        all_real_particles_index = []
        for res_index in [h1,h2,h3,h4]:
            chain_ntidx = dict_resid_chain[str(res_index)]
            dict_atoms = dict_residues[chain_ntidx][1]
            for name in ['BBP','BBC','NX','C2','CY']:
                if name in dict_atoms.keys():
                    all_real_particles_index.append(dict_atoms[name].index)
        if len(all_real_particles_index) != 20:
            print ("Less than 20 particles in four anchors in a helix!")
            exit()

        bbc_index = []
        backbone_real_particles_index = []
        for res_index in [h1,h2,h3,h4]:
            chain_ntidx = dict_resid_chain[str(res_index)]
            dict_atoms = dict_residues[chain_ntidx][1]
            for name in ['BBC','BBP']:
                if name in dict_atoms.keys():
                    backbone_real_particles_index.append(dict_atoms[name].index)
                    if name == 'BBC':
                        bbc_index.append(dict_atoms[name].index)
        if len(backbone_real_particles_index) != 8:
            print ("Less than 8 backbone particles in four anchors in a helix!")
            exit()

        kbond = 500.0*unit.kilojoule/unit.angstrom**2/unit.mole
        for p1, p2 in combinations(all_real_particles_index,2):
            distance = unit.norm(positions[p1]-positions[p2])
            if p1 in bbc_index and p2 in bbc_index:
                npower_bond_force.addBond(p1,p2,[kbond,distance])
            else:
                npower_bond_force.addBond(p1,p2,[kbond,distance])

        # vsiteParticles store the four particles (actually first three) used for defining virtual sites
        vsiteParticles = []
        for res_index in [h1,h2,h3,h4]:
            chain_ntidx = dict_resid_chain[str(res_index)]
            dict_atoms = dict_residues[chain_ntidx][1]
            if "BBC" not in dict_atoms.keys():
                print ("Residue {} in chain {} does not contain atom BBC.".format(res_index, chain))
                exit()
            vsiteParticles.append(dict_atoms["BBC"].index)
        
        d12 = (positions[vsiteParticles[1]]-positions[vsiteParticles[0]]).value_in_unit(unit.nanometer)
        d13 = (positions[vsiteParticles[2]]-positions[vsiteParticles[0]]).value_in_unit(unit.nanometer)
        cross = mm.Vec3(d12[1]*d13[2]-d12[2]*d13[1], d12[2]*d13[0]-d12[0]*d13[2], d12[0]*d13[1]-d12[1]*d13[0])
        matrix = np.zeros((3, 3))
        for i in range(3):
            matrix[i][0] = d12[i]
            matrix[i][1] = d13[i]
            matrix[i][2] = cross[i]
        
        virtual_site_atoms_index = []
        virtual_res_index = list(range(int(h1)+1, int(h2)))
        virtual_res_index2 = list(range(int(h3)+1, int(h4)))
        virtual_res_index.extend(virtual_res_index2)

        for res_index in virtual_res_index:
            chain_ntidx = dict_resid_chain[str(res_index)]
            dict_atoms = dict_residues[chain_ntidx][1]
            for name in ['BBP','BBC','NX','C2','CY']:
                if name in dict_atoms.keys():
                    virtual_site_atoms_index.append(dict_atoms[name].index)

        for i in virtual_site_atoms_index:
            system.setParticleMass(i, 0)
            rhs = np.array((positions[i]-positions[vsiteParticles[0]]).value_in_unit(unit.nanometer))
            weights = lin.solve(matrix, rhs)
            system.setVirtualSite(i, mm.OutOfPlaneSite(vsiteParticles[0], vsiteParticles[1], vsiteParticles[2], weights[0], weights[1], weights[2]))

    npower_bond_force.setForceGroup(13)
    system.addForce(npower_bond_force)
    return system


def add_atom_position_constraint_force(system,constrained_atoms):
    force = mm.CustomExternalForce("kfreeze*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("kfreeze", 200)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    for atom in constrained_atoms:
        atom_idx, position = atom
        force.addParticle(atom_idx, position)
    system.addForce(force)
    return system


def read_constraint_file(infile):
    if infile is None:
        dict_constraints = {"bp":[],"stk":[],"dis":[],"angle":[],"torsion":[]}
        return dict_constraints

    if not os.path.exists(infile):
        raise ValueError(f"There is no constraint file {infile}!")

    with open(infile) as f:
        lines = f.read().splitlines()

    list_bp_constraints = []
    list_stk_constraints = []
    list_dis_constraints = []
    list_angle_constraints = []
    list_torsion_constraints = []
    
    for line0 in lines:
        line = line0.split()
        if line[0].upper() == "BP":
            if len(line) != 3:
                print(f"Wrong format for the following base pairing constraint:\n{line0}")
                print("The right format is as follows:")
                print("BP chain1/nt1_index chain2/nt2_index")
                exit()
            if len(line[1].split("/")) != 2 or len(line[2].split("/")) != 2:
                print(f"Wrong format for the following base pairing constraint:\n{line0}")
                print("The right format is as follows:")
                print("BP chain1/nt1_index chain2/nt2_index")
                exit()
            chain1, nt1_idx = line[1].split("/")
            chain2, nt2_idx = line[2].split("/")
            if (not nt1_idx.isdigit()) or (not nt2_idx.isdigit()):
                print(f"Wrong format for the following base pairing constraint:\n{line0}")
                print("The right format is as follows:")
                print("BP chain1/nt1_index chain2/nt2_index")
                exit()
            list_bp_constraints.append((chain1,nt1_idx,chain2,nt2_idx))
        elif line[0].upper() in ["STK","STK35","STK53","STK33","STK55"]:
            if len(line) != 3:
                print(f"Wrong format for the following base stacking constraint:\n{line0}")
                print("The right format is as follows:")
                print("STK chain1/nt1_index chain2/nt2_index")
                exit()
            if len(line[1].split("/")) != 2 or len(line[2].split("/")) != 2:
                print(f"Wrong format for the following base stacking constraint:\n{line0}")
                print("The right format is as follows:")
                print("STK chain1/nt1_index chain2/nt2_index")
                exit()
            chain1, nt1_idx = line[1].split("/")
            chain2, nt2_idx = line[2].split("/")
            if (not nt1_idx.isdigit()) or (not nt2_idx.isdigit()):
                print(f"Wrong format for the following base stacking constraint:\n{line0}")
                print("The right format is as follows:")
                print("STK chain1/nt1_index chain2/nt2_index")
                exit()
            list_stk_constraints.append((chain1,nt1_idx,chain2,nt2_idx,line[0].upper()))
        elif line[0].upper() == "DIS":
            if len(line) != 5:
                print(f"Wrong format for the following distance constraint:\n{line0}")
                print("The right format is as follows:")
                print("DIS chain1/nt1_index/atom1_name chain2/nt2_index/atom2_name min_dis max_dis")
                exit()
            if len(line[1].split("/")) != 3 or len(line[2].split("/")) != 3:
                print(f"Wrong format for the following distance constraint:\n{line0}")
                print("The right format is as follows:")
                print("DIS chain1/nt1_index/atom1_name chain2/nt2_index/atom2_name min_dis max_dis")
                exit()
            chain1, nt1_idx, atom1_name = line[1].split("/")
            chain2, nt2_idx, atom2_name = line[2].split("/")
            if (not nt1_idx.isdigit()) or (not nt2_idx.isdigit()):
                print(f"Wrong format for the following distance constraint:\n{line0}")
                print("The right format is as follows:")
                print("DIS chain1/nt1_index/atom1_name chain2/nt2_index/atom2_name min_dis max_dis")
                exit()
            if atom1_name not in ["P","C4'","N9","C2","C4","N1","C6"] or atom2_name not in ["P","C4'","N9","C2","C4","N1","C6"]:
                print(f"Wrong atom names for the following distance constraint:\n{line0}")
                print("The valid atom names are P, C4', N9, C2, C6 (for purines), and N1, C2, C4 (for pyrimidines).")
                exit()
            try:
                min_dis = float(line[3])
                max_dis = float(line[4])
            except:
                raise ValueError(f"Wrong distance range for the following distance constraint:\n{line0}\nThe right format is as follows:\nDIS chain1/nt1_index/atom1_name chain2/nt2_index/atom2_name min_dis max_dis")
            if min_dis > max_dis:
                raise ValueError(f"Wrong distance range for the following distance constraint:\n{line0}\nThe right format is as follows:\nDIS chain1/nt1_index/atom1_name chain2/nt2_index/atom2_name min_dis max_dis\nmin_dis should be no greater than max_dis.")
            list_dis_constraints.append((chain1,nt1_idx,atom1_name,chain2,nt2_idx,atom2_name,min_dis,max_dis))
        elif line[0].upper() == "ANGLE":
            if len(line) != 6:
                print(f"Wrong format for the following angle constraint:\n{line0}")
                print("The right format is as follows:")
                print("ANGLE chain1/nt1_index/atom1_name chain2/nt2_index/atom2_name chain3/nt3_index/atom3_name min_angle max_angle")
                exit()
            if len(line[1].split("/")) != 3 or len(line[2].split("/")) != 3 or len(line[3].split("/")) != 3:
                print(f"Wrong format for the following angle constraint:\n{line0}")
                print("The right format is as follows:")
                print("ANGLE chain1/nt1_index/atom1_name chain2/nt2_index/atom2_name chain3/nt3_index/atom3_name min_angle max_angle")
                exit()
            chain1, nt1_idx, atom1_name = line[1].split("/")
            chain2, nt2_idx, atom2_name = line[2].split("/")
            chain3, nt3_idx, atom3_name = line[3].split("/")
            if (not nt1_idx.isdigit()) or (not nt2_idx.isdigit()) or (not nt3_idx.isdigit()):
                print(f"Wrong format for the following angle constraint:\n{line0}")
                print("The right format is as follows:")
                print("ANGLE chain1/nt1_index/atom1_name chain2/nt2_index/atom2_name chain3/nt3_index/atom3_name min_angle max_angle")
                exit()
            if atom1_name not in ["P","C4'","N9","C2","C4","N1","C6"] or atom2_name not in ["P","C4'","N9","C2","C4","N1","C6"] or atom3_name not in ["P","C4'","N9","C2","C4","N1","C6"]:
                print(f"Wrong atom names for the following angle constraint:\n{line0}")
                print("The valid atom names are P, C4', N9, C2, C6 (for purines), and N1, C2, C4 (for pyrimidines).")
                exit()
            try:
                min_angle = float(line[4])
                max_angle = float(line[5])
            except:
                raise ValueError(f"Wrong angle range for the following angle constraint:\n{line0}\nThe right format is as follows:\nANGLE chain1/nt1_index/atom1_name chain2/nt2_index/atom2_name chain3/nt3_index/atom3_name min_angle max_angle")
            if min_angle > max_angle:
                raise ValueError(f"Wrong angle range for the following angle constraint:\n{line0}\nThe right format is as follows:\nANGLE chain1/nt1_index/atom1_name chain2/nt2_index/atom2_name chain3/nt3_index/atom3_name min_angle max_angle\nmin_angle should be no greater than max_angle.")
            list_angle_constraints.append((chain1,nt1_idx,atom1_name,chain2,nt2_idx,atom2_name,chain3,nt3_idx,atom3_name,min_angle,max_angle))
        elif line[0].upper() == "TORSION":
            if len(line) != 7:
                print(f"Wrong format for the following torsion constraint:\n{line0}")
                print("The right format is as follows:")
                print("TORSION chain1/nt1_index/atom1_name chain2/nt2_index/atom2_name chain3/nt3_index/atom3_name chain4/nt4_index/atom4_name min_torsion max_torsion")
                exit()
            if len(line[1].split("/")) != 3 or len(line[2].split("/")) != 3 or len(line[3].split("/")) != 3 or len(line[4].split("/")) != 3:
                print(f"Wrong format for the following torsion constraint:\n{line0}")
                print("The right format is as follows:")
                print("TORSION chain1/nt1_index/atom1_name chain2/nt2_index/atom2_name chain3/nt3_index/atom3_name chain4/nt4_index/atom4_name min_torsion max_torsion")
                exit()
            chain1, nt1_idx, atom1_name = line[1].split("/")
            chain2, nt2_idx, atom2_name = line[2].split("/")
            chain3, nt3_idx, atom3_name = line[3].split("/")
            chain4, nt4_idx, atom4_name = line[4].split("/")
            if (not nt1_idx.isdigit()) or (not nt2_idx.isdigit()) or (not nt3_idx.isdigit()) or (not nt4_idx.isdigit()):
                print(f"Wrong format for the following torsion constraint:\n{line0}")
                print("The right format is as follows:")
                print("TORSION chain1/nt1_index/atom1_name chain2/nt2_index/atom2_name chain3/nt3_index/atom3_name chain4/nt4_index/atom4_name min_torsion max_torsion")
                exit()
            try:
                min_torsion = float(line[5])
                max_torsion = float(line[6])
            except:
                raise ValueError(f"Wrong torsion range for the following torsion constraint:\n{line0}\nThe right format is as follows:\nANGLE chain1/nt1_index/atom1_name chain2/nt2_index/atom2_name chain3/nt3_index/atom3_name chain4/nt4_index/atom4_name min_torsion max_torsion")
            if min_torsion > max_torsion:
                raise ValueError(f"Wrong torsion range for the following torsion constraint:\n{line0}\nThe right format is as follows:\nANGLE chain1/nt1_index/atom1_name chain2/nt2_index/atom2_name chain3/nt3_index/atom3_name chain4/nt4_index/atom4_name min_torsion max_torsion\nmin_torsion should be no greater than max_torsion.")
            list_torsion_constraints.append((chain1,nt1_idx,atom1_name,chain2,nt2_idx,atom2_name,chain3,nt3_idx,atom3_name,chain4,nt4_idx,atom4_name,min_torsion,max_torsion))
        else:
            print(f"Undefined constraint type: {line[0]} for the following constraint in file {infile}:\n{line0}")
            exit()

    dict_constraints = {}
    dict_constraints["bp"] = list_bp_constraints
    dict_constraints["stk"] = list_stk_constraints
    dict_constraints["dis"] = list_dis_constraints
    dict_constraints["angle"] = list_angle_constraints
    dict_constraints["torsion"] = list_torsion_constraints
    return dict_constraints            


def add_base_pairing_constraint_force(RNAJP_HOME, system, dict_atoms_index):
    dict_paras = {}
    for nt in ["AA","AG","AC","AU","GG","GC","GU","CC","CU","UU","aG","aC","aU","gC","gU","cU","aa","ag","ac","au","gg","gc","gu","cc","cu","uu"]:
        bp_strength = 3.
        paras_bp = [bp_strength]
        for i in range(9):
            paras_file = f"{RNAJP_HOME}/bp_stk_paras/bp_{nt.upper()}-{i}.txt"
            paras_bp.extend(load_bp_stk_paras(paras_file))
        dict_paras[nt] = paras_bp

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

    wt = 500
    bp_angle_penalty = f"{wt}*(abs(cosbeta)-1)^2"

    bp_energy = f"(-wt_bp*RT0*(max(({bp_logprob}-9),0)) + wt_bp*{bp_angle_penalty})"
    attraction_energy_for_d1 = f"step(d1-1.5)*500.0*(d1-0.99)^2 + step(1.5-d1)*step(d1-1.0)*500.0*(d1-0.99)^2 + step(0.3-d1)*500.0*(d1-0.31)^2"
    attraction_energy_for_d3 = f"step(d3-1.5)*500.0*(d3-0.99)^2 + step(1.5-d3)*step(d3-0.6)*500.0*(d3-0.60)^2 + step(0.3-d3)*500.0*(d3-0.31)^2"
    attraction_energy_for_d7 = f"step(d7-1.5)*500.0*(d7-0.99)^2 + step(1.5-d7)*step(d7-0.6)*500.0*(d7-0.60)^2 + step(0.3-d7)*500.0*(d7-0.31)^2"
    attraction_energy = f"{attraction_energy_for_d1} + {attraction_energy_for_d3} + {attraction_energy_for_d7}"

    bp_constraint_energy = f"kconstraint_global * ({attraction_energy} + {bp_energy}); d1=distance(p1,p4); d2=distance(p1,p5); d3=distance(p1,p6); d4=distance(p2,p4); d5=distance(p2,p5); d6=distance(p2,p6); d7=distance(p3,p4); d8=distance(p3,p5); d9=distance(p3,p6); theta1=dihedral(p1,p2,p3,p4); theta2=dihedral(p2,p3,p4,p5); theta3=dihedral(p3,p4,p5,p6); cosbeta=(dot_r1R1*dot_r2R2-dot_r1R2*dot_r2R1)/(distance(p1,p2)*distance(p1,p3)*distance(p4,p5)*distance(p4,p6)*sin(angle(p2,p1,p3))*sin(angle(p5,p4,p6))); dot_r1R1=a*g+b*h+c*i; dot_r2R2=d*j+e*k+f*l; dot_r1R2=a*j+b*k+c*l; dot_r2R1=d*g+e*h+f*i; a=x2-x1; b=y2-y1; c=z2-z1; d=x3-x1; e=y3-y1; f=z3-z1; g=x5-x4; h=y5-y4; i=z5-z4; j=x6-x4; k=y6-y4; l=z6-z4; RT0=2.494; costorsion1=cos(dihedral(p1,p2,p3,p4)); costorsion2=cos(dihedral(p1,p2,p3,p5)); costorsion3=cos(dihedral(p1,p2,p3,p6)); cosangle1=cos(angle(p2,p3,p4)); cosangle2=cos(angle(p2,p3,p5)); cosangle3=cos(angle(p2,p3,p6));"

    bp_constraint_force = mm.CustomCompoundBondForce(6, bp_constraint_energy)
    bp_constraint_force.addGlobalParameter("kconstraint_global", 0.)
    bp_constraint_force.addPerBondParameter(f"wt_bp")
    for i in range(1,10):
        for j in range(1,6):
            bp_constraint_force.addPerBondParameter(f"amplitude{j}_bp{i}")
        for j in range(1,6):
            bp_constraint_force.addPerBondParameter(f"center{j}_bp{i}")
        for j in range(1,6):
            bp_constraint_force.addPerBondParameter(f"sigma{j}_bp{i}")
        bp_constraint_force.addPerBondParameter(f"c_bp{i}")

    for key in dict_atoms_index:
        for atoms_index in dict_atoms_index[key]:
            paras = copy.deepcopy(dict_paras[key])
            bp_constraint_force.addBond(atoms_index[0:-1],paras)
        
    bp_constraint_force.setForceGroup(15)
    system.addForce(bp_constraint_force)
    return system


def add_base_pairing_constraint_force_to_system(RNAJP_HOME, topology, system, dict_motifs, dict_resid_chain, list_bp_constraints=[]):
    dict_residues = access_to_residues_by_chain_name_and_resid(topology)

    num_bp = 0
    list_bp_nts = []
    for bp in list_bp_constraints:
        nt1 = dict_residues[(bp[0],str(bp[1]))]
        nt2 = dict_residues[(bp[2],str(bp[3]))]
        list_bp_nts.append([(nt1,f"bp{num_bp}"),(nt2,f"bp{num_bp}")])
        num_bp += 1
        print(f"Adding base pairing constraints: {bp[0]}/{bp[1]}-{bp[2]}/{bp[3]}")

    if not list_bp_nts:
        return system

    if num_bp > 1:
        print(f"{num_bp} base pairing constraints have been added.",flush=True)
    else:
        print(f"{num_bp} base pairing constraint has been added.",flush=True)

    dict_atoms_index = {}
    for bp_nts in list_bp_nts:
        nt1 = bp_nts[0]
        nt2 = bp_nts[1]
        atoms_index = get_atoms_index_between_two_nts(nt1,nt2)
        for key in atoms_index:
            if key not in dict_atoms_index.keys():
                dict_atoms_index[key] = atoms_index[key]
            else:
                dict_atoms_index[key].extend(atoms_index[key])

    system = add_base_pairing_constraint_force(RNAJP_HOME, system, dict_atoms_index)
    return system


def add_base_stacking_constraint_force(RNAJP_HOME, system, dict_atoms_index):
    dict_paras = {}
    for nt in ["AA","AG","AC","AU","GG","GC","GU","CC","CU","UU","aG","aC","aU","gC","gU","cU","aa","ag","ac","au","gg","gc","gu","cc","cu","uu"]:
        stk_strength = 3.
        paras_stk = [stk_strength]
        for i in range(9):
            paras_file = f"{RNAJP_HOME}/bp_stk_paras/stk_{nt}-{i}.txt"
            paras_stk.extend(load_bp_stk_paras(paras_file))
        dict_paras[nt] = paras_stk

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
    stk_angle_penalty = f"{wt}*(abs(cosbeta)-1)^2"
    stk_face_penalty = f"{wt}*(sign1_5face*step(cosgamma1)*cosgamma1 + sign1_3face*step(-cosgamma1)*(-cosgamma1) + sign2_5face*step(cosgamma2)*cosgamma2 + sign2_3face*step(-cosgamma2)*(-cosgamma2))"
    threshold = f"(sign_faces + (1-sign_faces)*(sign1_5face*step(-cosgamma1)+sign1_3face*step(cosgamma1))*(sign2_5face*step(-cosgamma2)+sign2_3face*step(cosgamma2)))"
    stk_energy = f"{threshold}*(-wt_stk*RT0*(max(({stk_logprob})-9,0))) + wt_stk*{stk_angle_penalty} + {stk_face_penalty}"

    attraction_energy_for_d1 = f"step(d1-1.5)*500.0*(d1-0.99)^2 + step(1.5-d1)*step(d1-1.0)*500.0*(d1-0.99)^2 + step(0.2-d1)*500.0*(d1-0.21)^2"
    attraction_energy_for_d3 = f"step(d3-1.5)*500.0*(d3-0.99)^2 + step(1.5-d3)*step(d3-0.6)*500.0*(d3-0.60)^2 + step(0.2-d3)*500.0*(d3-0.21)^2"
    attraction_energy_for_d7 = f"step(d7-1.5)*500.0*(d7-0.99)^2 + step(1.5-d7)*step(d7-0.6)*500.0*(d7-0.60)^2 + step(0.2-d7)*500.0*(d7-0.21)^2"
    attraction_energy = f"{attraction_energy_for_d1} + {attraction_energy_for_d3} + {attraction_energy_for_d7}"

    stk_constraint_energy = f"kconstraint_global * ({stk_energy} + {attraction_energy}) ; d1=distance(p1,p4); d2=distance(p1,p5); d3=distance(p1,p6); d4=distance(p2,p4); d5=distance(p2,p5); d6=distance(p2,p6); d7=distance(p3,p4); d8=distance(p3,p5); d9=distance(p3,p6); theta1=dihedral(p1,p2,p3,p4); theta2=dihedral(p2,p3,p4,p5); theta3=dihedral(p3,p4,p5,p6); cosbeta=(dot_r1R1*dot_r2R2-dot_r1R2*dot_r2R1)/(distance(p1,p2)*distance(p1,p3)*distance(p4,p5)*distance(p4,p6)*sin(angle(p2,p1,p3))*sin(angle(p5,p4,p6))); dot_r1R1=a*g+b*h+c*i; dot_r2R2=d*j+e*k+f*l; dot_r1R2=a*j+b*k+c*l; dot_r2R1=d*g+e*h+f*i; cosgamma1=(m*(b*f-c*e)+n*(c*d-a*f)+o*(a*e-b*d))/(distance(p1,p2)*distance(p1,p3)*distance(p1,p4)*sin(angle(p2,p1,p3))); cosgamma2=(m*(i*k-h*l)+n*(g*l-i*j)+o*(h*j-g*k))/(distance(p4,p5)*distance(p4,p6)*distance(p4,p1)*sin(angle(p5,p4,p6))); a=x2-x1; b=y2-y1; c=z2-z1; d=x3-x1; e=y3-y1; f=z3-z1; g=x5-x4; h=y5-y4; i=z5-z4; j=x6-x4; k=y6-y4; l=z6-z4; m=x4-x1; n=y4-y1; o=z4-z1; RT0=2.494"

    stk_constraint_force = mm.CustomCompoundBondForce(6, stk_constraint_energy)
    stk_constraint_force.addGlobalParameter("kconstraint_global", 0.)
    stk_constraint_force.addPerBondParameter(f"wt_stk")
    for i in range(1,10):
        for j in range(1,6):
            stk_constraint_force.addPerBondParameter(f"amplitude{j}_stk{i}")
        for j in range(1,6):
            stk_constraint_force.addPerBondParameter(f"center{j}_stk{i}")
        for j in range(1,6):
            stk_constraint_force.addPerBondParameter(f"sigma{j}_stk{i}")
        stk_constraint_force.addPerBondParameter(f"c_stk{i}")
    stk_constraint_force.addPerBondParameter("sign_faces")
    stk_constraint_force.addPerBondParameter("sign1_5face")
    stk_constraint_force.addPerBondParameter("sign1_3face")
    stk_constraint_force.addPerBondParameter("sign2_5face")
    stk_constraint_force.addPerBondParameter("sign2_3face")

    for key in dict_atoms_index:
        for atoms_index in dict_atoms_index[key]:
            stk_faces = atoms_index[-1]
            paras = copy.deepcopy(dict_paras[key])
            if stk_faces == "STK":
                paras.extend([1,0,0,0,0])
            elif stk_faces == "STK53":
                paras.extend([0,1,0,0,1])
            elif stk_faces == "STK55":
                paras.extend([0,1,0,1,0])
            elif stk_faces == "STK35":
                paras.extend([0,0,1,1,0])
            elif stk_faces == "STK33":
                paras.extend([0,0,1,0,1])
            else:
                raise ValueError(f"Wrong stacking faces constraint {stk_faces}, which should be 'STK', 'STK35', 'STK33','STK53',or 'STK55'") 
            stk_constraint_force.addBond(atoms_index[0:-2],paras)
        
    stk_constraint_force.setForceGroup(16)
    system.addForce(stk_constraint_force)
    return system


def add_base_stacking_constraint_force_to_system(RNAJP_HOME, topology, system, list_stk_constraints=[]):
    if not list_stk_constraints:
        return system

    dict_residues = access_to_residues_by_chain_name_and_resid(topology)

    list_stk_nts = []
    num_stk = 1
    for stk in list_stk_constraints:
        nt1 = dict_residues[(stk[0],str(stk[1]))]
        nt2 = dict_residues[(stk[2],str(stk[3]))]
        list_stk_nts.append([(nt1,f"stk{num_stk}"),(nt2,f"stk{num_stk}"),stk[4]])
        num_stk += 1
        print(f"Adding base stacking constraints: {stk[4]} {stk[0]}/{stk[1]}-{stk[2]}/{stk[3]}")

    if num_stk - 1 > 1:
        print(f"{num_stk-1} base stacking constraints have been added.",flush=True)
    else:
        print(f"{num_stk-1} base stacking constraint has been added.",flush=True)

    dict_atoms_index = {}
    for stk_nts in list_stk_nts:
        nt1 = stk_nts[0]
        nt2 = stk_nts[1]
        stk_faces = stk_nts[2]
        atoms_index = get_atoms_index_between_two_nts(nt1,nt2,False,stk_faces)
        for key in atoms_index:
            if key not in dict_atoms_index.keys():
                dict_atoms_index[key] = atoms_index[key]
            else:
                dict_atoms_index[key].extend(atoms_index[key])

    system = add_base_stacking_constraint_force(RNAJP_HOME, system, dict_atoms_index)
    return system


def convert_atom_name(atom_name):
    if atom_name == "P":
        atom_name = "BBP"
    elif atom_name == "C4'":
        atom_name = "BBC"
    elif atom_name in ["N9","N1"]:
        atom_name = "NX"
    elif atom_name in ["C4","C6"]:
        atom_name = "CY"
    elif atom_name == "C2":
        atom_name = "C2"
    else:
        raise ValueError(f"Invalid atom name {atom_name} in the distance/angle/torsion constraints!")
    return atom_name


def add_distance_constraint_force(system, list_atoms_index_para):
    wt = 1000.0
    dis_constraint_energy = f"kconstraint_global * (step(min_dis-dis)*{wt}*(min_dis-dis)^2 + step(dis-max_dis)*{wt}*(dis-max_dis)^2); dis=distance(p1,p2)"

    dis_constraint_force = mm.CustomCompoundBondForce(2, dis_constraint_energy)
    dis_constraint_force.addGlobalParameter("kconstraint_global", 0.)
    dis_constraint_force.addPerBondParameter("min_dis")
    dis_constraint_force.addPerBondParameter("max_dis")

    for atoms_index_para in list_atoms_index_para:
        atoms_index = atoms_index_para[0:2]
        paras = atoms_index_para[2:]
        dis_constraint_force.addBond(atoms_index,paras)
        
    dis_constraint_force.setForceGroup(17)
    system.addForce(dis_constraint_force)
    return system

def add_distance_constraint_force_to_system(topology, system, list_dis_constraints=[]):
    if not list_dis_constraints:
        return system

    dict_residues = access_to_residues_by_chain_name_and_resid(topology)
    list_atoms_index_para = []
    num_dis = 1
    for dis_constraints in list_dis_constraints:
        nt1_chain, nt1_idx, atom1_name, nt2_chain, nt2_idx, atom2_name, min_dis, max_dis = dis_constraints
        print(f"Adding distance constraints: {nt1_chain}/{nt1_idx}/{atom1_name}-{nt2_chain}/{nt2_idx}/{atom2_name} {min_dis} {max_dis}")
        num_dis += 1

        min_dis /= 10.0 # angstrom to nm
        max_dis /= 10.0 
        nt1 = dict_residues[(nt1_chain,nt1_idx)]
        nt2 = dict_residues[(nt2_chain,nt2_idx)]

        atom1_name = convert_atom_name(atom1_name)
        atom2_name = convert_atom_name(atom2_name)

        atom1_idx = nt1[1][atom1_name].index
        atom2_idx = nt2[1][atom2_name].index
        list_atoms_index_para.append([atom1_idx,atom2_idx,min_dis,max_dis])

    if num_dis - 1 > 1:
        print(f"{num_dis-1} distance constraints have been added.",flush=True)
    else:
        print(f"{num_dis-1} distance constraint has been added.",flush=True)

    system = add_distance_constraint_force(system, list_atoms_index_para)
    return system


def add_angle_constraint_force(system, list_atoms_index_para):
    wt = 1000.0
    angle_constraint_energy = f"kconstraint_global * (step(cos_angle-cos_min_angle)*{wt}*(cos_angle-cos_min_angle)^2 + step(cos_max_angle-cos_angle)*{wt}*(cos_max_angle-cos_angle)^2); cos_angle=cos(angle(p1,p2,p3))"
    angle_constraint_force = mm.CustomCompoundBondForce(3, angle_constraint_energy)
    angle_constraint_force.addGlobalParameter("kconstraint_global", 0.)
    angle_constraint_force.addPerBondParameter("cos_min_angle")
    angle_constraint_force.addPerBondParameter("cos_max_angle")

    for atoms_index_para in list_atoms_index_para:
        atoms_index = atoms_index_para[0:3]
        min_angle, max_angle = atoms_index_para[3:5]
        cos_min_angle = np.cos(np.radians(min_angle))
        cos_max_angle = np.cos(np.radians(max_angle))
        angle_constraint_force.addBond(atoms_index,[cos_min_angle,cos_max_angle])
        
    angle_constraint_force.setForceGroup(18)
    system.addForce(angle_constraint_force)
    return system


def add_angle_constraint_force_between_four_atoms(system, list_atoms_index_para):
    wt = 500.0
    angle_constraint_energy = f"kconstraint_global * (step(cos_angle-cos_min_angle)*{wt}*(cos_angle-cos_min_angle)^2 + step(cos_max_angle-cos_angle)*{wt}*(cos_max_angle-cos_angle)^2); cos_angle=(a*d+b*e+c*f)/dis1/dis2; dis1=distance(p1,p2); dis2=distance(p3,p4); a=x2-x1; b=y2-y1; c=z2-z1; d=x4-x3; e=y4-y3; f=z4-z3;"
    angle_constraint_force = mm.CustomCompoundBondForce(4, angle_constraint_energy)
    angle_constraint_force.addGlobalParameter("kconstraint_global", 0.)
    angle_constraint_force.addPerBondParameter("cos_min_angle")
    angle_constraint_force.addPerBondParameter("cos_max_angle")

    for atoms_index_para in list_atoms_index_para:
        atoms_index = atoms_index_para[0:4]
        min_angle, max_angle = atoms_index_para[4:6]
        cos_min_angle = np.cos(np.radians(min_angle))
        cos_max_angle = np.cos(np.radians(max_angle))
        angle_constraint_force.addBond(atoms_index,[cos_min_angle,cos_max_angle])
        
    angle_constraint_force.setForceGroup(18)
    system.addForce(angle_constraint_force)
    return system


def add_angle_constraint_force_to_system(topology, system, list_angle_constraints=[]):
    if not list_angle_constraints:
        return system

    dict_residues = access_to_residues_by_chain_name_and_resid(topology)
    list_atoms_index_para = []
    num_angle = 1
    for angle_constraints in list_angle_constraints:
        nt1_chain, nt1_idx, atom1_name, nt2_chain, nt2_idx, atom2_name, nt3_chain, nt3_idx, atom3_name, min_angle, max_angle = angle_constraints
        print(f"Adding angle constraints: {nt1_chain}/{nt1_idx}/{atom1_name}-{nt2_chain}/{nt2_idx}/{atom2_name}-{nt3_chain}/{nt3_idx}/{atom3_name} {min_angle} {max_angle}")
        num_angle += 1

        nt1 = dict_residues[(nt1_chain,nt1_idx)]
        nt2 = dict_residues[(nt2_chain,nt2_idx)]
        nt3 = dict_residues[(nt3_chain,nt3_idx)]

        atom1_name = convert_atom_name(atom1_name)
        atom2_name = convert_atom_name(atom2_name)
        atom3_name = convert_atom_name(atom3_name)

        atom1_idx = nt1[1][atom1_name].index
        atom2_idx = nt2[1][atom2_name].index
        atom3_idx = nt3[1][atom3_name].index
        list_atoms_index_para.append([atom1_idx,atom2_idx,atom3_idx,min_angle,max_angle])

    if num_angle - 1 > 1:
        print(f"{num_angle-1} angle constraints have been added.",flush=True)
    else:
        print(f"{num_angle-1} angle constraint has been added.",flush=True)

    system = add_angle_constraint_force(system, list_atoms_index_para)
    return system


def get_min_cos_sin_torsion(min_torsion,max_torsion):
    min_torsion = np.radians(min_torsion)
    max_torsion = np.radians(max_torsion)

    if min_torsion == max_torsion:
        return [np.cos(min_torsion),np.cos(max_torsion),np.sin(min_torsion),np.sin(max_torsion)]

    binwidth = (max_torsion-min_torsion)/100.0

    mincos0 = min(np.cos(min_torsion),np.cos(max_torsion))
    maxcos0 = max(np.cos(min_torsion),np.cos(max_torsion))
    minsin0 = min(np.sin(min_torsion),np.sin(max_torsion))
    maxsin0 = max(np.sin(min_torsion),np.sin(max_torsion))
    mincos = 10000.0
    maxcos = -10000.0
    minsin = 10000.0
    maxsin = -10000.0

    torsion_list = list(np.arange(min_torsion,max_torsion+binwidth,binwidth))
    if torsion_list[-1] > max_torsion:
        torsion_list[-1] = max_torsion
    elif torsion_list[-1] < max_torsion:
        torsion_list.append(max_torsion)

    for t in torsion_list:
        cost = np.cos(t)
        sint = np.sin(t)
        if cost < mincos:
            mincos = cost
        if cost > maxcos:
            maxcos = cost
        if sint < minsin:
            minsin = sint
        if sint > maxsin:
            maxsin = sint
    if mincos < mincos0:
        mincos = -1.0
    if maxcos > maxcos0:
        maxcos = 1.0
    if minsin < minsin0:
        minsin = -1.0
    if maxsin > maxsin0:
        maxsin = 1.0
    return [mincos,maxcos,minsin,maxsin]


def add_torsion_constraint_force(system, list_atoms_index_para):
    wt = 1000.0
    torsion_constraint_energy = f"step(3000-energy)*energy + step(0.342020143-sin(angle1))*{wt}*(sin(angle1)-0.342020143)^2 + step(0.342020143-sin(angle2))*{wt}*(sin(angle2)-0.342020143)^2; energy=kconstraint_global * step(sin(angle1)-0.173648178) * step(sin(angle2)-0.173648178) * {wt} * (step(min_cos_torsion-cos_torsion)*(min_cos_torsion-cos_torsion)^2 + step(cos_torsion-max_cos_torsion)*(cos_torsion-max_cos_torsion)^2 + step(min_sin_torsion-sin_torsion)*(min_sin_torsion-sin_torsion)^2 + step(sin_torsion-max_sin_torsion)*(sin_torsion-max_sin_torsion)^2); cos_torsion=cos(dihedral(p1,p2,p3,p4)); sin_torsion=sin(dihedral(p1,p2,p3,p4)); angle1=angle(p1,p2,p3); angle2=angle(p2,p3,p4)" # 10 degree < angle(p1,p2,p3) < 170 degree, 10 degree < angle(p2,p3,p4) < 170 degree
    torsion_constraint_force = mm.CustomCompoundBondForce(4, torsion_constraint_energy)
    torsion_constraint_force.addGlobalParameter("kconstraint_global", 0.)
    torsion_constraint_force.addPerBondParameter("min_cos_torsion")
    torsion_constraint_force.addPerBondParameter("max_cos_torsion")
    torsion_constraint_force.addPerBondParameter("min_sin_torsion")
    torsion_constraint_force.addPerBondParameter("max_sin_torsion")

    for atoms_index_para in list_atoms_index_para:
        atoms_index = atoms_index_para[0:4]
        min_torsion, max_torsion = atoms_index_para[4:6]
        paras = get_min_cos_sin_torsion(min_torsion,max_torsion)
        torsion_constraint_force.addBond(atoms_index,paras)
        
    torsion_constraint_force.setForceGroup(19)
    system.addForce(torsion_constraint_force)
    return system


def add_torsion_constraint_force_to_system(topology, system, list_torsion_constraints=[]):
    if not list_torsion_constraints:
        return system

    dict_residues = access_to_residues_by_chain_name_and_resid(topology)
    list_atoms_index_para = []
    num_torsion = 1
    for torsion_constraints in list_torsion_constraints:
        nt1_chain, nt1_idx, atom1_name, nt2_chain, nt2_idx, atom2_name, nt3_chain, nt3_idx, atom3_name, nt4_chain, nt4_idx, atom4_name, min_torsion, max_torsion = torsion_constraints
        print(f"Adding torsion constraints: {nt1_chain}/{nt1_idx}/{atom1_name}-{nt2_chain}/{nt2_idx}/{atom2_name}-{nt3_chain}/{nt3_idx}/{atom3_name}-{nt4_chain}/{nt4_idx}/{atom4_name} {min_torsion} {max_torsion}")
        num_torsion += 1

        nt1 = dict_residues[(nt1_chain,nt1_idx)]
        nt2 = dict_residues[(nt2_chain,nt2_idx)]
        nt3 = dict_residues[(nt3_chain,nt3_idx)]
        nt4 = dict_residues[(nt4_chain,nt4_idx)]

        atom1_name = convert_atom_name(atom1_name)
        atom2_name = convert_atom_name(atom2_name)
        atom3_name = convert_atom_name(atom3_name)
        atom4_name = convert_atom_name(atom4_name)

        atom1_idx = nt1[1][atom1_name].index
        atom2_idx = nt2[1][atom2_name].index
        atom3_idx = nt3[1][atom3_name].index
        atom4_idx = nt4[1][atom4_name].index
        list_atoms_index_para.append([atom1_idx,atom2_idx,atom3_idx,atom4_idx,min_torsion,max_torsion])

    if num_torsion - 1 > 1:
        print(f"{num_torsion-1} torsion constraints have been added.",flush=True)
    else:
        print(f"{num_torsion-1} torsion constraint has been added.",flush=True)

    system = add_torsion_constraint_force(system, list_atoms_index_para)
    return system


def get_canonical_base_pairing_constraints(topology, dict_motifs, dict_resid_chain):
    dict_residues = access_to_residues_by_chain_name_and_resid(topology)

    list_all_helices = dict_motifs["helix"] + dict_motifs["PK_helix"]
    list_bp_nts = []
    num_bp = 1
    list_dis_constraints=[]
    list_angle_constraints=[]
    list_torsion_constraints=[]

    for helix in list_all_helices:
        h1 = helix[0][0] # h1 = res_index
        h2 = helix[1][0]
        h3 = helix[1][1]
        h4 = helix[0][1]
        if h2 - h1 != h4 - h3:
            print("The lengths of the two strands in the helix are not the same!")
            print (h1, h2)
            print (h3, h4)
            exit()
        if h1 == h2:
            chain_ntidx1 = dict_resid_chain[str(h1)]
            nt1 = dict_residues[chain_ntidx1]
            nt1_name = nt1[0].name[-1]
            if nt1_name in ["A","G"]:
                NX_1 = "N9"
                C2_1 = "C2"
                CY_1 = "C6"
            else:
                NX_1 = "N1"
                C2_1 = "C2"
                CY_1 = "C4"
            chain_ntidx2 = dict_resid_chain[str(h4)]
            nt2 = dict_residues[chain_ntidx2]
            nt2_name = nt2[0].name[-1]
            if nt2_name in ["A","G"]:
                NX_2 = "N9"
                C2_2 = "C2"
                CY_2 = "C6"
            else:
                NX_2 = "N1"
                C2_2 = "C2"
                CY_2 = "C4"
            
            dis_constraint = (chain_ntidx1[0],chain_ntidx1[1],"P",chain_ntidx1[0],chain_ntidx1[1],NX_1,4.0,6.0)
            list_dis_constraints.append(dis_constraint)
            dis_constraint = (chain_ntidx2[0],chain_ntidx2[1],"P",chain_ntidx2[0],chain_ntidx2[1],NX_2,4.0,6.0)
            list_dis_constraints.append(dis_constraint)

            dis_constraint = (chain_ntidx1[0],chain_ntidx1[1],"P",chain_ntidx2[0],chain_ntidx2[1],"P",16.0,20.0)
            list_dis_constraints.append(dis_constraint)
            dis_constraint = (chain_ntidx1[0],chain_ntidx1[1],"P",chain_ntidx2[0],chain_ntidx2[1],"C4'",15.0,19.0)
            list_dis_constraints.append(dis_constraint)
            dis_constraint = (chain_ntidx1[0],chain_ntidx1[1],"C4'",chain_ntidx2[0],chain_ntidx2[1],"P",15.0,19.0)
            list_dis_constraints.append(dis_constraint)
            dis_constraint = (chain_ntidx1[0],chain_ntidx1[1],"C4'",chain_ntidx2[0],chain_ntidx2[1],"C4'",13.0,17.0)
            list_dis_constraints.append(dis_constraint)

            dis_constraint = (chain_ntidx1[0],chain_ntidx1[1],NX_1,chain_ntidx2[0],chain_ntidx2[1],NX_2,8.5,9.5)
            list_dis_constraints.append(dis_constraint)

            if nt1_name in ["A","G"] and nt2_name in ["C","U"]:
                dis_constraint = (chain_ntidx1[0],chain_ntidx1[1],NX_1,chain_ntidx2[0],chain_ntidx2[1],C2_2,7.0,8.0)
            else:
                dis_constraint = (chain_ntidx1[0],chain_ntidx1[1],NX_1,chain_ntidx2[0],chain_ntidx2[1],C2_2,5.0,6.0)
            list_dis_constraints.append(dis_constraint)

            
            if nt1_name in ["A","G"] and nt2_name in ["C","U"]:
                dis_constraint = (chain_ntidx1[0],chain_ntidx1[1],NX_1,chain_ntidx2[0],chain_ntidx2[1],CY_2,7.2,8.2)
            else:
                dis_constraint = (chain_ntidx1[0],chain_ntidx1[1],NX_1,chain_ntidx2[0],chain_ntidx2[1],CY_2,5.5,6.5)
            list_dis_constraints.append(dis_constraint)

            if nt1_name in ["A","G"] and nt2_name in ["C","U"]:
                dis_constraint = (chain_ntidx1[0],chain_ntidx1[1],C2_1,chain_ntidx2[0],chain_ntidx2[1],NX_2,5.0,6.0)
            else:
                dis_constraint = (chain_ntidx1[0],chain_ntidx1[1],C2_1,chain_ntidx2[0],chain_ntidx2[1],NX_2,7.0,8.0)
            list_dis_constraints.append(dis_constraint)

            if nt1_name in ["A","G"] and nt2_name in ["C","U"]:
                dis_constraint = (chain_ntidx1[0],chain_ntidx1[1],C2_1,chain_ntidx2[0],chain_ntidx2[1],C2_2,4.0,5.0)
            else:
                dis_constraint = (chain_ntidx1[0],chain_ntidx1[1],C2_1,chain_ntidx2[0],chain_ntidx2[1],C2_2,4.0,5.0)
            list_dis_constraints.append(dis_constraint)

            if nt1_name in ["A","G"] and nt2_name in ["C","U"]:
                dis_constraint = (chain_ntidx1[0],chain_ntidx1[1],C2_1,chain_ntidx2[0],chain_ntidx2[1],CY_2,4.5,5.5)
            else:
                dis_constraint = (chain_ntidx1[0],chain_ntidx1[1],C2_1,chain_ntidx2[0],chain_ntidx2[1],CY_2,4.0,5.0)
            list_dis_constraints.append(dis_constraint)

            if nt1_name in ["A","G"] and nt2_name in ["C","U"]:
                dis_constraint = (chain_ntidx1[0],chain_ntidx1[1],CY_1,chain_ntidx2[0],chain_ntidx2[1],NX_2,5.5,6.5)
            else:
                dis_constraint = (chain_ntidx1[0],chain_ntidx1[1],CY_1,chain_ntidx2[0],chain_ntidx2[1],NX_2,7.5,8.5)
            list_dis_constraints.append(dis_constraint)

            if nt1_name in ["A","G"] and nt2_name in ["C","U"]:
                dis_constraint = (chain_ntidx1[0],chain_ntidx1[1],CY_1,chain_ntidx2[0],chain_ntidx2[1],C2_2,4.0,5.0)
            else:
                dis_constraint = (chain_ntidx1[0],chain_ntidx1[1],CY_1,chain_ntidx2[0],chain_ntidx2[1],C2_2,4.5,5.5)
            list_dis_constraints.append(dis_constraint)

            if nt1_name in ["A","G"] and nt2_name in ["C","U"]:
                dis_constraint = (chain_ntidx1[0],chain_ntidx1[1],CY_1,chain_ntidx2[0],chain_ntidx2[1],CY_2,4.0,5.0)
            else:
                dis_constraint = (chain_ntidx1[0],chain_ntidx1[1],CY_1,chain_ntidx2[0],chain_ntidx2[1],CY_2,4.0,5.0)
            list_dis_constraints.append(dis_constraint)

            if nt1_name in ["A","G"] and nt2_name in ["C","U"]:
                angle_constraint = (chain_ntidx1[0],chain_ntidx1[1],NX_1,chain_ntidx1[0],chain_ntidx1[1],C2_1,chain_ntidx2[0],chain_ntidx2[1],C2_2,140.,160.)
            else:
                angle_constraint = (chain_ntidx1[0],chain_ntidx1[1],NX_1,chain_ntidx1[0],chain_ntidx1[1],C2_1,chain_ntidx2[0],chain_ntidx2[1],C2_2,160.,176.)
            list_angle_constraints.append(angle_constraint)

            if nt1_name in ["A","G"] and nt2_name in ["C","U"]:
                angle_constraint = (chain_ntidx1[0],chain_ntidx1[1],C2_1,chain_ntidx1[0],chain_ntidx1[1],CY_1,chain_ntidx2[0],chain_ntidx2[1],CY_2,80.,100.)
            else:
                angle_constraint = (chain_ntidx1[0],chain_ntidx1[1],C2_1,chain_ntidx1[0],chain_ntidx1[1],CY_1,chain_ntidx2[0],chain_ntidx2[1],CY_2,80,100.0)
            list_angle_constraints.append(angle_constraint)

            torsion_constraint = (chain_ntidx1[0],chain_ntidx1[1],C2_1,chain_ntidx1[0],chain_ntidx1[1],CY_1,chain_ntidx2[0],chain_ntidx2[1],CY_2,chain_ntidx2[0],chain_ntidx2[1],C2_2,-10.,10.)
            list_torsion_constraints.append(torsion_constraint)
            torsion_constraint = (chain_ntidx1[0],chain_ntidx1[1],NX_1,chain_ntidx1[0],chain_ntidx1[1],CY_1,chain_ntidx2[0],chain_ntidx2[1],CY_2,chain_ntidx2[0],chain_ntidx2[1],NX_2,-10.,10.)
            list_torsion_constraints.append(torsion_constraint)

            num_bp += 1
            print(f"Adding canonical base pairing constraints: {chain_ntidx1[0]}/{chain_ntidx1[1]}-{chain_ntidx2[0]}/{chain_ntidx2[1]}")
    return list_dis_constraints, list_angle_constraints, list_torsion_constraints


def add_constraint_force_to_system(RNAJP_HOME, topology, system, dict_motifs, dict_resid_chain, constraint_file):
    dict_constraints = read_constraint_file(constraint_file)
    list_dis_constraints, list_angle_constraints, list_torsion_constraints = get_canonical_base_pairing_constraints(topology, dict_motifs, dict_resid_chain)
    dict_constraints["dis"].extend(list_dis_constraints)
    dict_constraints["angle"].extend(list_angle_constraints)
    dict_constraints["torsion"].extend(list_torsion_constraints)

    system = add_base_pairing_constraint_force_to_system(RNAJP_HOME, topology, system, dict_motifs, dict_resid_chain, dict_constraints["bp"])
    system = add_base_stacking_constraint_force_to_system(RNAJP_HOME, topology, system, dict_constraints["stk"])
    system = add_distance_constraint_force_to_system(topology, system, dict_constraints["dis"])
    system = add_angle_constraint_force_to_system(topology, system, dict_constraints["angle"])
    system = add_torsion_constraint_force_to_system(topology, system, dict_constraints["torsion"])
    return system
