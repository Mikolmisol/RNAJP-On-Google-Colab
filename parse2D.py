import os
import copy

def find_brackets(ss,PK=False):
    open_list = ["[","{","(","<"]
    close_list = ["]","}",")",">"]
    if PK:
        open_list = ["[","{","<"]
        close_list = ["]","}",">"]

    brackets = {}
    stack = []
    stack_char = []

    for i, c in enumerate(ss):
        if c in open_list:
            stack_char.append(c)
            stack.append(i)
        elif c in close_list:
            pos = close_list.index(c)
            if len(stack) == 0:
                raise SystemExit('Unpaired secondary structure.')
            for j,d in enumerate(reversed(stack_char)):
                if open_list[pos] == d:
                    stack_char.pop(-(j+1))
                    brackets[stack.pop(-(j+1))] = i
                    break
    if len(stack) > 0:
        raise SystemExit('Unpaired secondary structure.')
    return brackets


def find_helices(ss,PK=False):
    brackets = find_brackets(ss,PK)
    brackets = sorted(brackets.items(),key=lambda x:x[0])
    list_helices = []
    helix = []
    for i, v in enumerate(brackets):
        v = list(v)
        if not helix:
            helix.append(v)
            helix_len = 1
        else:
            if v[0] - helix[0][0] == helix_len and v[1] - helix[0][1] == -helix_len:
                helix_len += 1
                if i == len(brackets) - 1:
                    helix.append(v)
                    list_helices.append(helix)
                    helix = []
            else:
                helix.append(list(brackets[i-1]))
                list_helices.append(helix)
                helix = [v]
                helix_len = 1
    if helix:
        if len(helix) != 1:
            raise ValueError("Error in finding helices in the 2D structure!")
        helix.append(copy.deepcopy(helix[0]))
        list_helices.append(helix)
    return list_helices


def find_loops(ss,list_helices):
    helical_region = []
    for helix in list_helices:
        ib = helix[0][0]
        ie = helix[0][1]
        helical_region.extend(list(range(ib,ie+1)))
        ib = helix[1][0]
        ie = helix[1][1]
        helical_region.extend(list(range(ib,ie+1)))
    list_loops = []
    loop = []
    for i in range(len(ss)):
        if ss[i] == ".":
            if not loop:
                loop.append(i)
                if i == len(ss) - 1:
                    loop.append(i)
                    list_loops.append(loop)
            else:
                if i == len(ss)-1:
                    loop.append(i)
                    list_loops.append(loop)
        else:
            if loop:
                loop.append(i-1)
                list_loops.append(loop)
                loop = []
    for loop in list_loops:
        if loop[0] >= 1:
            if ss[loop[0]-1] != " ":
                loop[0] -= 1
        if loop[1] <= len(ss) - 2:
            if ss[loop[1]+1] != " ":
                loop[1] += 1
    return list_loops


def judge_if_helices_connected(ss,helix1,helix2):
    if helix1[0][0] > helix2[0][0]:
        helix1, helix2 = helix2, helix1

    list1 = [v for h in helix1 for v in h]
    list2 = [v for h in helix2 for v in h]

    connected1 = False
    connected2 = False
    connected3 = False

    ib = list1[2]
    ie = list2[0]
    if ib > ie:
        connected1 = False
    elif ss[ib+1:ie] == "".join(["."]*(ie-ib-1)):
        connected1 = True

    ib = list2[1]
    ie = list1[3]
    if ib > ie:
        connected2 = False
    elif ss[ib+1:ie] == "".join(["."]*(ie-ib-1)):
        connected2 = True

    ib = list1[1]
    ie = list2[0]
    if ib > ie:
        connected3 = False
    elif ss[ib+1:ie] == "".join(["."]*(ie-ib-1)):
        connected3 = True
    return connected1, connected2, connected3


def find_junction_loops(junction_helix):
    junc_loops = []
    for i in range(len(junction_helix)):
        helix1 = junction_helix[i]
        j = i + 1
        if j == len(junction_helix):
            j = 0
        helix2 = junction_helix[j]
        
        if i == 0:
            loop = [helix1[1][0],helix2[0][0]]
        elif i == len(junction_helix) - 1:
            loop = [helix1[0][1],helix2[1][1]]
        else:
            loop = [helix1[0][1],helix2[0][0]]
        junc_loops.append(loop)
    return junc_loops


def determine_if_PK_in_loop(list_PK_helices,loop,loop_type=None):
    has_PK_interaction = False
    for PK_helix in list_PK_helices:
        left_helix_ib = PK_helix[0][0]
        left_helix_ie = PK_helix[1][0]
        right_helix_ib = PK_helix[1][1]
        right_helix_ie = PK_helix[0][1]
        if loop_type == "2way_loops" and left_helix_ib == left_helix_ie:
            continue            
        for subloop in loop:
            loop_ib, loop_ie = subloop
            if left_helix_ib >= loop_ib and left_helix_ie <= loop_ie:
                has_PK_interaction = True
                break
            if right_helix_ib >= loop_ib and right_helix_ie <= loop_ie:
                has_PK_interaction = True
                break
        if has_PK_interaction:
            break
    return has_PK_interaction


def find_junctions_and_loops(ss, list_helices_all, list_PK_helices):
    list_helices = [] # helices excluding single base-paired helix
    for helix in list_helices_all:
        h1, h4 = helix[0]
        h2, h3 = helix[1]
        if h1 != h2:
            list_helices.append(copy.deepcopy(helix))

    list_2way = []
    list_3way = []
    list_4way = []
    list_5way = []
    list_6way = []
    list_7way = []
    list_8way = []

    dict_connected_helices = {}
    for i in range(len(list_helices)):
        helix1 = list_helices[i]
        for j in range(i+1,len(list_helices)):
            helix2 = list_helices[j]
            c1, c2, c3 = judge_if_helices_connected(ss,helix1,helix2)
            if c1 and c2:
                list_2way.append((list_helices[i],list_helices[j]))
            else:
                if c1 or c2 or c3:
                    if i not in dict_connected_helices.keys():
                        dict_connected_helices[i] = [j]
                    else:
                        dict_connected_helices[i].append(j)
                    if j not in dict_connected_helices.keys():
                        dict_connected_helices[j] = [i]
                    else:
                        dict_connected_helices[j].append(i)

    for i in range(len(list_helices)):
        if i not in dict_connected_helices.keys():
            continue
        for j in range(i+1,len(list_helices)):
            if j not in dict_connected_helices.keys():
                continue
            if j not in dict_connected_helices[i]:
                continue
            for k in range(j+1,len(list_helices)):
                if k not in dict_connected_helices.keys():
                    continue
                if k not in dict_connected_helices[j]:
                    continue
                if i in dict_connected_helices[k]:
                    list_3way.append((list_helices[i],list_helices[j],list_helices[k]))
                for m in range(k+1,len(list_helices)):
                    if m not in dict_connected_helices.keys():
                        continue
                    if m not in dict_connected_helices[k]:
                        continue
                    if i in dict_connected_helices[m]:
                        list_4way.append((list_helices[i],list_helices[j],list_helices[k],list_helices[m]))

                    for n in range(m+1,len(list_helices)):
                        if n not in dict_connected_helices.keys():
                            continue
                        if n not in dict_connected_helices[m]:
                            continue
                        if i in dict_connected_helices[n]:
                            list_5way.append((list_helices[i],list_helices[j],list_helices[k],list_helices[m],list_helices[n]))

                        for p in range(n+1,len(list_helices)):
                            if p not in dict_connected_helices.keys():
                                continue
                            if p not in dict_connected_helices[n]:
                                continue
                            if i in dict_connected_helices[p]:
                                list_6way.append((list_helices[i],list_helices[j],list_helices[k],list_helices[m],list_helices[n],list_helices[p]))

                            for q in range(p+1,len(list_helices)):
                                if q not in dict_connected_helices.keys():
                                    continue
                                if q not in dict_connected_helices[p]:
                                    continue
                                if i in dict_connected_helices[q]:
                                    list_7way.append((list_helices[i],list_helices[j],list_helices[k],list_helices[m],list_helices[n],list_helices[p],list_helices[q]))

                                for r in range(q+1,len(list_helices)):
                                    if r not in dict_connected_helices.keys():
                                        continue
                                    if r not in dict_connected_helices[q]:
                                        continue
                                    if i in dict_connected_helices[r]:
                                        list_8way.append((list_helices[i],list_helices[j],list_helices[k],list_helices[m],list_helices[n],list_helices[p],list_helices[q],list_helices[r]))

    list_loops =  find_loops(ss,list_helices)

    list_2way_loops = []
    list_3way_loops = []
    list_4way_loops = []
    list_5way_loops = []
    list_6way_loops = []
    list_7way_loops = []
    list_8way_loops = []
    for junc_helix in list_2way:
        junc_loops = find_junction_loops(junc_helix)
        list_2way_loops.append(junc_loops)
    for junc_helix in list_3way:
        junc_loops = find_junction_loops(junc_helix)
        list_3way_loops.append(junc_loops)
    for junc_helix in list_4way:
        junc_loops = find_junction_loops(junc_helix)
        list_4way_loops.append(junc_loops)
    for junc_helix in list_5way:
        junc_loops = find_junction_loops(junc_helix)
        list_5way_loops.append(junc_loops)
    for junc_helix in list_6way:
        junc_loops = find_junction_loops(junc_helix)
        list_6way_loops.append(junc_loops)
    for junc_helix in list_7way:
        junc_loops = find_junction_loops(junc_helix)
        list_7way_loops.append(junc_loops)
    for junc_helix in list_8way:
        junc_loops = find_junction_loops(junc_helix)
        list_8way_loops.append(junc_loops)

    list_hairpin_loops = []
    list_5end_loops = []
    list_3end_loops = []
    list_single_loops = []
    for loop in list_loops:
        if loop[0] == 0:
            list_5end_loops.append([loop])
            continue
        elif ss[loop[0]-1] == " ":
            list_5end_loops.append([loop])
            continue

        if loop[1] == len(ss) - 1:
            list_3end_loops.append([loop])
            continue
        elif ss[loop[1]+1] == " ":
            list_3end_loops.append([loop])
            continue
            
        is_hairpin = True
        for junc_loop in list_2way_loops+list_3way_loops+list_4way_loops+list_5way_loops+list_6way_loops+list_7way_loops:
            if loop in junc_loop:
                is_hairpin = False
                break
        if not is_hairpin:
            continue

        loop_ib = loop[0]
        loop_ie = loop[1]
        for helix in list_helices:
            thelix = [helix[0][0],helix[0][1],helix[1][0],helix[1][1]]
            if loop_ib in thelix:
                if loop_ie not in thelix:
                    is_hairpin = False
                    list_single_loops.append([loop])
                    break
        if not is_hairpin:
            continue

        if is_hairpin:
            list_hairpin_loops.append([loop])

    for i in dict_connected_helices:
        for j in dict_connected_helices[i]:
            helix1 = list_helices[i]
            helix2 = list_helices[j]
            found_in_junc = False
            for junctions in [list_2way,list_3way,list_4way,list_5way,list_6way,list_7way,list_8way]:
                for junc in junctions:
                    if helix1 in junc and helix2 in junc:
                        found_in_junc = True
                        break
                if found_in_junc:
                    break
            if not found_in_junc:
                if helix1[0][0] > helix2[0][0]:
                    helix1, helix2 = helix2, helix1
                    connected1, connected2, connected3 = judge_if_helices_connected(ss,helix1,helix2)
                    list1 = [v for h in helix1 for v in h]
                    list2 = [v for h in helix2 for v in h]
                    if connected1:
                        single_loop = [list1[2],list2[0]]
                    elif connected2:
                        single_loop = [list2[1],list1[3]]
                    elif connected3:
                        single_loop = [list1[1],list2[0]]
                    else:
                        raise ValueError("Two helices are not connected!")
                    if single_loop[1] - single_loop[0] > 1:
                        continue
                    list_single_loops.append([single_loop])

    list_hairpin_loops_non_PK = []
    for loop in list_hairpin_loops:
        has_PK_interaction = determine_if_PK_in_loop(list_PK_helices,loop)
        if not has_PK_interaction:
            list_hairpin_loops_non_PK.append(copy.deepcopy(loop))

    list_2way_loops_non_PK = []
    for loop in list_2way_loops:
        has_PK_interaction = determine_if_PK_in_loop(list_PK_helices,loop,loop_type="2way_loops")
        if not has_PK_interaction:
            list_2way_loops_non_PK.append(copy.deepcopy(loop))

    list_5end_loops_non_PK = []
    for loop in list_5end_loops:
        has_PK_interaction = determine_if_PK_in_loop(list_PK_helices,loop)
        if not has_PK_interaction:
            list_5end_loops_non_PK.append(copy.deepcopy(loop))

    list_3end_loops_non_PK = []
    for loop in list_3end_loops:
        has_PK_interaction = determine_if_PK_in_loop(list_PK_helices,loop)
        if not has_PK_interaction:
            list_3end_loops_non_PK.append(copy.deepcopy(loop))

    list_single_loops_non_PK = []
    for loop in list_single_loops:
        has_PK_interaction = determine_if_PK_in_loop(list_PK_helices,loop)
        if not has_PK_interaction:
            list_single_loops_non_PK.append(copy.deepcopy(loop))

    list_3way_loops_non_PK = []
    for loop in list_3way_loops:
        has_PK_interaction = determine_if_PK_in_loop(list_PK_helices,loop)
        if not has_PK_interaction:
            list_3way_loops_non_PK.append(copy.deepcopy(loop))

    list_4way_loops_non_PK = []
    for loop in list_4way_loops:
        has_PK_interaction = determine_if_PK_in_loop(list_PK_helices,loop)
        if not has_PK_interaction:
            list_4way_loops_non_PK.append(copy.deepcopy(loop))

    dict_motifs = {}
    dict_motifs["5end_loops"] = list_5end_loops
    dict_motifs["5end_loops_non_PK"] = list_5end_loops_non_PK
    dict_motifs["3end_loops"] = list_3end_loops
    dict_motifs["3end_loops_non_PK"] = list_3end_loops_non_PK
    dict_motifs["single_loops"] = list_single_loops
    dict_motifs["single_loops_non_PK"] = list_single_loops_non_PK
    dict_motifs["hairpin_loops"] = list_hairpin_loops
    dict_motifs["hairpin_loops_non_PK"] = list_hairpin_loops_non_PK
    dict_motifs["2way_loops"] = list_2way_loops
    dict_motifs["2way_loops_non_PK"] = list_2way_loops_non_PK
    dict_motifs["3way_loops"] = list_3way_loops
    dict_motifs["3way_loops_non_PK"] = list_3way_loops_non_PK
    dict_motifs["4way_loops"] = list_4way_loops
    dict_motifs["4way_loops_non_PK"] = list_4way_loops_non_PK
    dict_motifs["5way_loops"] = list_5way_loops
    dict_motifs["6way_loops"] = list_6way_loops
    dict_motifs["7way_loops"] = list_7way_loops
    dict_motifs["8way_loops"] = list_8way_loops
    dict_motifs["helix"] = list_helices_all
    dict_motifs["PK_helix"] = list_PK_helices
    if list_5way_loops:
        print ("Including 5-way junctions")
        exit()
    if list_6way_loops:
        print ("Including 6-way junctions")
        exit()
    if list_7way_loops:
        print ("Including 7-way junctions")
        exit()
    if list_8way_loops:
        print ("Including 8-way junctions")
        exit()

    for key in dict_motifs:
        motifs = dict_motifs[key]
        for motif in motifs:
            for loop in motif:
                loop[0] += 1
                loop[1] += 1
    return dict_motifs


def get_chain_resid(seq):
    seq = seq.strip()
    chain = "A"
    dict_resid_chain = {}
    nt_real_idx = 1
    nt_idx = 1
    for i in range(len(seq)):
        if seq[i] == " ":
            chain = chr(ord(chain)+1)
            nt_idx = 1
            continue
        dict_resid_chain[str(nt_real_idx)] = (chain,str(nt_idx))
        nt_idx += 1
        nt_real_idx += 1
    return dict_resid_chain


def get_motifs_from_2D(infile):
    if not os.path.exists(infile):
        raise ValueError(f"Failed in finding the 2D structure file: {infile}")

    with open(infile) as f:
        lines = f.read().splitlines()
    seq = lines[0].strip()
    seq = seq.upper()
    ss = lines[1].strip()
    if len(seq) != len(ss):
        raise ValueError(f"The lengths of sequence and 2D structure in the file {infile} are not the same!\n{seq}\n{ss}")
    for i in range(len(seq)):
        if seq[i] not in "AUGCaugc ":
            raise ValueError(f"There is an unknown nucleotide {seq[i]} in the sequence in the file {infile}!\n{seq}")
        if ss[i] not in "()[]{}<>. ":
            raise ValueError(f"There is an unknown 2D label {ss[i]} in the 2D structure in the file {infile}!\n{ss}")
        if seq[i] == " " and ss[i] != " ":
            raise ValueError(f"The chain delimiters (space symbol) in the sequence and 2D structure have different indices!\n{seq}\n{ss}")
        if seq[i] != " " and ss[i] == " ":
            raise ValueError(f"The chain delimiters (space symbol) in the sequence and 2D structure have different indices!\n{seq}\n{ss}")
        if i != len(seq) - 1:
            if seq[i] == " " and seq[i+1] == " ":
                raise ValueError(f"The chain delimiter should be one space symbol in the 2D structure!\n{ss}")

    list_PK_helices = find_helices(ss,PK=True)
    
    ss = list(ss)
    for i in range(len(ss)):
        if ss[i] not in ["(",")","."," "]:
            ss[i] = "."
    ss = "".join(ss) # remove PK

    list_helices = find_helices(ss)

    ss = list(ss)
    for helix in list_helices:
        h1, h4 = helix[0]
        h2, h3 = helix[1]
        if h1 == h2:
            if h3 != h4:
                raise ValueError(f"Wrong helix index: {helix}")
            ss[h1] = "."
            ss[h4] = "."
    ss = "".join(ss) # remove base pair not in helix
            
    dict_motifs = find_junctions_and_loops(ss, list_helices, list_PK_helices)

    if " " in ss:
        list_space_indices = []
        num_space = 1
        for i, v in enumerate(ss):
            if v == " ":
                list_space_indices.append((i+1,num_space))
                num_space += 1

        for key in dict_motifs:
            motifs = dict_motifs[key]
            for motif in motifs:
                for loop in motif:
                    for space_index in list_space_indices[::-1]:
                        if loop[0] > space_index[0]:
                            loop[0] = loop[0] - space_index[1]
                            break
                    for space_index in list_space_indices[::-1]:
                        if loop[1] > space_index[0]:
                            loop[1] = loop[1] - space_index[1]
                            break
    print(f"\nThe motifs in the given 2D structure:")
    print(dict_motifs, flush=True)
    dict_resid_chain = get_chain_resid(seq)
    return seq,dict_motifs, dict_resid_chain
