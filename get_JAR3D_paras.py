import sys
import os
import re
import string
import numpy as np
import copy


def get_model_instance_num(RNAJP_HOME):
    dict_model_instance_num = {}
    for looptype in ["HL","IL"]:
        with open(f"{RNAJP_HOME}/JAR3D/models/{looptype}/3.2/lib/all.txt") as f:
            lines = f.read().splitlines()
        for line in lines:
            model = line.split("_model")[0]
            with open(f"{RNAJP_HOME}/JAR3D/models/{looptype}/3.2/lib/{model}_data.txt") as f2:
                lines2 = f2.read().splitlines()
            for line2 in lines2:
                line2 = line2.split()
                if len(line2) < 2:
                    continue
                if "instance" in line2[1]:
                    num_instance = int(line2[0])
            dict_model_instance_num[model] = num_instance
    return dict_model_instance_num


def positionkeyforsortbynumber(tag):
    m = re.search("Position_([0-9]+)[_-]",tag)
    n = m.group(1)
    while (len(n) < 20):
        n = "0" + n

    m = re.search("Insertion_([0-9]+)_",tag)
    if m is None:
        nn = ""
    else:
        nn = m.group(1)
    while (len(nn) < 20):
        nn = "0" + nn
    return n+nn


def readcorrespondencesfromtext(lines):
    InstanceToGroup = {}          # instance of motif to conserved group position
    InstanceToPDB = {}            # instance of motif to NTs in PDBs
    InstanceToSequence = {}       # instance of motif to position in fasta file
    GroupToModel = {}             # positions in motif group to nodes in JAR3D model
    ModelToColumn = {}            # nodes in JAR3D model to display columns
    HasName = {}                  # organism name in FASTA header
    SequenceToModel = {}          # sequence position to node in JAR3D model
    HasScore = {}                 # score of sequence against JAR3D model
    HasInteriorEdit = {}          # minimum interior edit distance to 3D instances from the motif group
    HasFullEdit = {}              # minimum full edit distance to 3D instances from the motif group
    HasCutoffValue = {}           # cutoff value 'true' or 'false'
    HasCutoffScore = {}           # cutoff score, 100 is perfect, 0 is accepted, negative means reject
    HasAlignmentScoreDeficit = {} # alignment score deficit, how far below the best score among 3D instances in this group

    for line in lines:
        if re.search("corresponds_to_group",line):
            m = re.match("(.*) (.*) (.*)",line)
            InstanceToGroup[m.group(1)] = m.group(3)
        elif re.search("corresponds_to_PDB",line):
            m = re.match("(.*) (.*) (.*)",line)
            InstanceToPDB[m.group(1)] = m.group(3)
        elif re.search("corresponds_to_JAR3D",line):
            m = re.match("(.*) (.*) (.*)",line)
            GroupToModel[m.group(1)] = m.group(3)
        elif re.search("corresponds_to_sequence",line):
            m = re.match("(.*) (.*) (.*)",line)
            InstanceToSequence[m.group(1)] = m.group(3)
        elif re.search("appears_in_column",line):
            m = re.match("(.*) (.*) (.*)",line)
            ModelToColumn[m.group(1)] = m.group(3)
        elif re.search("aligns_to_JAR3D",line):
            m = re.match("(.*) (.*) (.*)",line)
            SequenceToModel[m.group(1)] = m.group(3)
        elif re.search("has_name",line):
            m = re.match("(.*) (.*) (.*)",line)
            HasName[m.group(1)] = m.group(3)
        elif re.search("has_score",line):
            m = re.match("(.*) (.*) (.*)",line)
            HasScore[m.group(1)] = m.group(3)
        elif re.search("has_minimum_interior_edit_distance",line):
            m = re.match("(.*) (.*) (.*)",line)
            HasInteriorEdit[m.group(1)] = m.group(3)
        elif re.search("has_minimum_full_edit_distance",line):
            m = re.match("(.*) (.*) (.*)",line)
            HasFullEdit[m.group(1)] = m.group(3)
        elif re.search("has_cutoff_value",line):
            m = re.match("(.*) (.*) (.*)",line)
            HasCutoffValue[m.group(1)] = m.group(3)
        elif re.search("has_cutoff_score",line):
            m = re.match("(.*) (.*) (.*)",line)
            HasCutoffScore[m.group(1)] = m.group(3)
        elif re.search("has_alignment_score_deficit",line):
            m = re.match("(.*) (.*) (.*)",line)
            HasAlignmentScoreDeficit[m.group(1)] = m.group(3)

    return InstanceToGroup, InstanceToPDB, InstanceToSequence, GroupToModel, ModelToColumn, SequenceToModel, HasName, HasScore, HasInteriorEdit, HasFullEdit, HasCutoffValue, HasCutoffScore, HasAlignmentScoreDeficit


def readcorrespondencesfromfile(filenamewithpath):
    InstanceToGroup = {}          # instance of motif to conserved group position
    InstanceToPDB = {}            # instance of motif to NTs in PDBs
    InstanceToSequence = {}       # instance of motif to position in fasta file
    GroupToModel = {}             # positions in motif group to nodes in JAR3D model
    ModelToColumn = {}            # nodes in JAR3D model to display columns
    HasName = {}                  # header lines from FASTA file
    SequenceToModel = {}          # sequence position to node in JAR3D model
    HasScore = {}                 # score of sequence against JAR3D model
    HasInteriorEdit = {}          # minimum interior edit distance to 3D instances from the motif group
    HasFullEdit = {}              # minimum full edit distance to 3D instances from the motif group
    HasCutoffValue = {}           # cutoff value 'true' or 'false'
    HasCutoffScore = {}           # cutoff score, 100 is perfect, 0 is accepted, negative means reject
    HasAlignmentScoreDeficit = {} # alignment score deficit, how far below the best score among 3D instances in this group

    with open(filenamewithpath,"r") as f:
      lines = f.readlines()

    InstanceToGroup, InstanceToPDB, InstanceToSequence, GroupToModel, ModelToColumn, SequenceToModel, HasName, HasScore, HasInteriorEdit, HasFullEdit, HasCutoffValue, HasCutoffScore, HasAlignmentScoreDeficit = readcorrespondencesfromtext(lines)

    return InstanceToGroup, InstanceToPDB, InstanceToSequence, GroupToModel, ModelToColumn, SequenceToModel, HasName, HasScore, HasInteriorEdit, HasFullEdit, HasCutoffValue, HasCutoffScore, HasAlignmentScoreDeficit

def fastatomodelalignment(libDirectory,motifID,alignmentfile):
    # read correspondences from the fasta file to the model
    InstanceToGroup, InstanceToPDB, InstanceToSequence, GroupToModel, ModelToColumn, SequenceToModel, HasName, HasScore, HasInteriorEdit, HasFullEdit, HasCutoffValue, HasCutoffScore, HasAlignmentScoreDeficit = readcorrespondencesfromfile(alignmentfile)
    #print ("Read alignment to model from " + alignmentfile)

    FN = libDirectory + motifID + "_correspondences.txt"

    # read correspondences for the given motif group; there are many such correspondences
    InstanceToGroup, InstanceToPDB, InstanceToSequence, GroupToModel, ModelToColumn, SequenceToModelDummy, ModelHasName, ModelHasScore, ModelInteriorEdit, ModelFullEdit, ModelCutoffValue, ModelCutoffScore, ModelDeficit = readcorrespondencesfromfile(FN)

    #  print HasScore
    #  print ModelHasName
    HasName.update(ModelHasName)
    HasScore.update(ModelHasScore)

    #print ("Read model correspondences from " + FN)
    dict_seq_col = {}
    for a in sorted(SequenceToModel.keys(), key=positionkeyforsortbynumber):
        m = re.search("(Sequence_[0-9]+)",a)
        t = ModelToColumn[SequenceToModel[a]]
        dict_seq_col[t] = a

    dict_aligned_conserved_seq = {}
    for a in GroupToModel.keys():
        colnum = ModelToColumn[GroupToModel[a]]
        m = re.search("Column_([0-9]+)$",a)
        if m is not None:
            motif_pos = m.group(1)
            if colnum in dict_seq_col.keys():
                dict_aligned_conserved_seq[motif_pos] = dict_seq_col[colnum]

    InteractionsFile = libDirectory + motifID + "_interactions.txt"
    with open(InteractionsFile,"r") as f:
        lines = f.read().splitlines()
    dict_interaction_in_seq = {}
    for line in lines:
        line = line.split()
        nt1 = line[0]
        nt2 = line[1]
        inter = line[2]
        nt_pair = (dict_aligned_conserved_seq[nt1],dict_aligned_conserved_seq[nt2])
        dict_interaction_in_seq[nt_pair] = inter
    #print (dict_interaction_in_seq)
    return dict_interaction_in_seq


def mapping_core_resi_to_real_resi(libDirectory,motifID,alignmentfile,motif_seq,motif_resi):
    # read correspondences from the fasta file to the model
    InstanceToGroup, InstanceToPDB, InstanceToSequence, GroupToModel, ModelToColumn, SequenceToModel, HasName, HasScore, HasInteriorEdit, HasFullEdit, HasCutoffValue, HasCutoffScore, HasAlignmentScoreDeficit = readcorrespondencesfromfile(alignmentfile)
    #print ("Read alignment to model from " + alignmentfile)

    FN = libDirectory + motifID + "_correspondences.txt"

    # read correspondences for the given motif group; there are many such correspondences
    InstanceToGroup, InstanceToPDB, InstanceToSequence, GroupToModel, ModelToColumn, SequenceToModelDummy, ModelHasName, ModelHasScore, ModelInteriorEdit, ModelFullEdit, ModelCutoffValue, ModelCutoffScore, ModelDeficit = readcorrespondencesfromfile(FN)

    #  print HasScore
    #  print ModelHasName
    HasName.update(ModelHasName)
    HasScore.update(ModelHasScore)

    #print ("Read model correspondences from " + FN)
    dict_seq_col = {}
    for a in sorted(SequenceToModel.keys(), key=positionkeyforsortbynumber):
        m = re.search("(Sequence_[0-9]+)",a)
        t = ModelToColumn[SequenceToModel[a]]
        dict_seq_col[t] = a

    dict_aligned_conserved_seq = {}
    for a in GroupToModel.keys():
        colnum = ModelToColumn[GroupToModel[a]]
        m = re.search("Column_([0-9]+)$",a)
        if m is not None:
            motif_pos = m.group(1)
            if colnum in dict_seq_col.keys():
                dict_aligned_conserved_seq[motif_pos] = dict_seq_col[colnum]

    dict_map_core_resi_to_real_resi = {}
    for key in dict_aligned_conserved_seq:
        resi = int(dict_aligned_conserved_seq[key].split("_")[3])-1
        real_resi = motif_resi[resi]
        key = int(key)
        dict_map_core_resi_to_real_resi[key] = real_resi
    return dict_map_core_resi_to_real_resi


def select_motif(seqoutput_file):
    with open(seqoutput_file) as f:
        lines = f.read().splitlines()

    list_motif_candidates = []
    for line in lines[1:]:
        line = line.split(",")
        motifID = line[2]
        passed = line[3]
        meancutoffscore = float(line[4])
        fulleditdis = float(line[-2])
        rotation = int(float(line[-1]))
        motif_info = (meancutoffscore,fulleditdis,rotation,motifID)
        if passed == "true":
            list_motif_candidates.append(motif_info)
            #print (motif_info)

    list_motif_candidates = sorted(list_motif_candidates,key=lambda x:(8*x[1]-x[0])) # the full edit distance * 8
    return list_motif_candidates


def calc_seq_mark_diff(seq_mark0,seq_mark):
    list_core_res_idx1 = []
    for i in range(len(seq_mark0)):
        if seq_mark0[i] == "N":
            list_core_res_idx1.append(i)
    list_insertion_num1 = [list_core_res_idx1[i+1]-list_core_res_idx1[i]-1 for i in range(len(list_core_res_idx1)-1)] 

    list_core_res_idx2 = []
    for i in range(len(seq_mark)):
        if seq_mark[i] == "N":
            list_core_res_idx2.append(i)
    list_insertion_num2 = [list_core_res_idx2[i+1]-list_core_res_idx2[i]-1 for i in range(len(list_core_res_idx2)-1)] 

    if len(list_insertion_num1) != len(list_insertion_num2):
        #print ("Two sequence marks have different core residues")
        #print (seq_mark0,seq_mark)
        return 10000

    list_diff = [abs(x-y) for x,y in zip(list_insertion_num1,list_insertion_num2)]
    diff = sum(list_diff)
    return diff


def get_jar3d_paras_for_known_motif(RNAJP_HOME,motif_seq,motif_resi):
    if "*" in motif_seq:
        looptype = "IL"
    else:
        looptype = "HL"

    with open(f"{RNAJP_HOME}/JAR3D/3D_paras/{looptype}/3.2/{looptype}_knownmotifseq.txt") as f:
        lines = f.read().splitlines()
    motifID = None
    for line in lines:
        line = line.split()
        if line[0] == motif_seq:
            motifID = line[1]
            break
    if motifID is None:
        return None
    
    dict_jar3d_paras = {}
    with open(f"{RNAJP_HOME}/JAR3D/3D_paras/{looptype}/3.2/paras_{motifID}.txt") as f:
        lines = f.read().splitlines()

    if looptype == "IL":
        right_core_loop_head_idx = motif_seq.index("*") + 1
    else:
        right_core_loop_head_idx = len(motif_seq) + 1

    motif_resi0 = copy.deepcopy(motif_resi)
    if looptype == "IL":
        motif_resi0.remove("*")
    
    dict_map_resi = {}
    for i in range(len(motif_resi0)):
        dict_map_resi[i+1] = motif_resi0[i]

    for i,line in enumerate(lines[1:]):
        line = line.split()
        line = list(map(float,line))
        resi1 = int(line[0])
        resi2 = int(line[1])
        real_resi1 = dict_map_resi[resi1]
        real_resi2 = dict_map_resi[resi2]
        if resi1 != 1 and resi1 != right_core_loop_head_idx:
            real_resi0 = dict_map_resi[resi1-1]
        else:
            real_resi0 = None
        key = (real_resi0,real_resi1,real_resi2)
        dict_jar3d_paras[key] = line[2:]
    return dict_jar3d_paras


def get_jar3d_paras_for_motif(RNAJP_HOME,dict_model_instance_num,motif_seq,motif_resi,outfolder):
    if "*" in motif_seq:
        looptype = "IL"
    else:
        looptype = "HL"

    with open(f"{outfolder}/tmp.fasta","w") as f:
        f.write(">tmp\n")
        f.write(f"{motif_seq}\n")     

    cmd = f"java -jar {RNAJP_HOME}/JAR3D/jar3d_2014-12-11.jar {outfolder}/tmp.fasta {RNAJP_HOME}/JAR3D/models/{looptype}/3.2/lib/all.txt {outfolder}/tmploop.txt {outfolder}/tmpseq.txt >> {outfolder}/jar3d.log 2>&1"
    #print (cmd)
    os.system(cmd)

    #print (motif_seq, "MOTIF_SEQ")
    list_motif_candidates = select_motif(f"{outfolder}/tmpseq.txt")
    if len(list_motif_candidates) == 0:
        return None, None

    list_same_motifs = []
    for m in list_motif_candidates:
        if m[1] == 0:
            list_same_motifs.append(m)
    if not list_same_motifs:  
        motif = list_motif_candidates[0]
        motif_score = motif[0] - 8*motif[1]
    else:
        max_instance_num = -1
        for i, m in enumerate(list_same_motifs):
            mid = m[-1]
            instance_num = dict_model_instance_num[mid]
            if instance_num > max_instance_num:
                best_idx = i
                max_instance_num = instance_num
        motif = list_same_motifs[best_idx]
        motif_score = 100.0
    
    motifID = motif[-1]
    rotation = motif[-2]

    #print ("selected motif",motif)

    cmd = f"java -jar {RNAJP_HOME}/JAR3D/jar3dalign_2015-04-03.jar {outfolder}/tmp.fasta {RNAJP_HOME}/JAR3D/models/{looptype}/3.2/lib/ {motifID} {rotation} {outfolder}/tmpcorr.txt >> {outfolder}/jar3d.log 2>&1"
    #print (cmd)
    os.system(cmd)
    libDirectory = f"{RNAJP_HOME}/JAR3D/models/{looptype}/3.2/lib/"
    alignmentfile = f"{outfolder}/tmpcorr.txt"

    if rotation == 1:
        idx = motif_resi.index("*")
        motif_resi = motif_resi[idx+1:] + motif_resi[idx:(idx+1)] + motif_resi[0:idx]
        motif_seq = motif_seq[idx+1:] + motif_seq[idx:(idx+1)] + motif_seq[0:idx]
    dict_map_resi = mapping_core_resi_to_real_resi(libDirectory,motifID,alignmentfile,motif_seq,motif_resi)

    cmd = f"rm {outfolder}/tmploop.txt {outfolder}/tmpseq.txt {outfolder}/tmp.fasta {outfolder}/tmpcorr.txt"
    os.system(cmd)

    val = []
    for key in dict_map_resi:
        val.append(dict_map_resi[key])
    seq_mark = ""
    for i in motif_resi:
        if i == "*":
            seq_mark += ","
        elif i in val:
            seq_mark += "N"
        else:
            seq_mark += "I"

    with open(f"{RNAJP_HOME}/JAR3D/3D_paras/{looptype}/3.2/paras_{motifID}.txt") as f:
        lines = f.read().splitlines()

    list_seq_mark_idx = []
    for i,line in enumerate(lines):
        line = line.split()
        if len(line) == 1:
            list_seq_mark_idx.append(i)

    list_seq_mark_diff = []
    for i in list_seq_mark_idx:
        seq_mark0 = lines[i].split()[0]
        diff = calc_seq_mark_diff(seq_mark0,seq_mark)
        list_seq_mark_diff.append(diff)
    if min(list_seq_mark_diff) >= 10000:
        return None, None
    min_diff_idx = list_seq_mark_diff.index(min(list_seq_mark_diff))

    ib = list_seq_mark_idx[min_diff_idx] + 1
    if min_diff_idx + 1 < len(list_seq_mark_idx):
        ie = list_seq_mark_idx[min_diff_idx+1]
    else:
        ie = len(lines)

    right_core_loop_head_idx = seq_mark.split(",")[0].count("N") + 1

    dict_jar3d_paras = {}
    for line in lines[ib:ie]:
        line = line.split()
        line = list(map(float,line))
        resi1 = int(line[0])
        resi2 = int(line[1])
        real_resi1 = dict_map_resi[resi1]
        real_resi2 = dict_map_resi[resi2]
        if resi1 != 1 and resi1 != right_core_loop_head_idx:
            real_resi0 = dict_map_resi[resi1-1]
        else:
            real_resi0 = None
        key = (real_resi0,real_resi1,real_resi2)
        dict_jar3d_paras[key] = line[2:]
    return dict_jar3d_paras, motif_score


def combine_two_dict(dict0,dict1):
    for key in dict1:
        if key not in dict0:
            dict0[key] = dict1[key]
        else:
            dict0[key].extend(dict1[key])
    return dict0


def split_hairpin_motif(motif_seq,motif_resi):
    list_hairpin_submotifs = []
    for i in range(1,len(motif_seq)-1):
        nt1 = motif_seq[i]
        for j in range(i+3,len(motif_seq)-1):
            nt2 = motif_seq[j]
            if nt1+nt2 in ["AU","UA","GC","CG","GU","UG"]:
                sub_motif_seq1 = motif_seq[0:(i+1)] + "*" + motif_seq[j:]
                sub_motif_resi1 = motif_resi[0:(i+1)] + ["*"] + motif_resi[j:]
                sub_motif_seq2 = motif_seq[i:(j+1)]
                sub_motif_resi2 = motif_resi[i:(j+1)]
                list_hairpin_submotifs.append([[sub_motif_seq1,sub_motif_resi1],[sub_motif_seq2,sub_motif_resi2]])
    return list_hairpin_submotifs


def split_2way_motif(motif_seq,motif_resi):
    list_2way_submotifs = []
    gap_index = motif_seq.index("*")
    for i in range(1,gap_index-1):
        nt1 = motif_seq[i]
        for j in range(gap_index+2,len(motif_seq)-1):
            nt2 = motif_seq[j]
            if nt1+nt2 in ["AU","UA","GC","CG","GU","UG"]:
                sub_motif_seq1 = motif_seq[0:(i+1)] + "*" + motif_seq[j:]
                sub_motif_resi1 = motif_resi[0:(i+1)] + ["*"] + motif_resi[j:]
                sub_motif_seq2 = motif_seq[i:gap_index] + "*" + motif_seq[(gap_index+1):(j+1)]
                sub_motif_resi2 = motif_resi[i:gap_index] + ["*"] + motif_resi[(gap_index+1):(j+1)]
                list_2way_submotifs.append([[sub_motif_seq1,sub_motif_resi1],[sub_motif_seq2,sub_motif_resi2]])
                for k in range(i+1,gap_index-1):
                    nt3 = motif_seq[k]
                    for m in range(gap_index+2,j):
                        nt4 = motif_seq[m]
                        if nt3+nt4 in ["AU","UA","GC","CG","GU","UG"]:
                            sub_motif_seq1 = motif_seq[0:(i+1)] + "*" + motif_seq[j:]
                            sub_motif_resi1 = motif_resi[0:(i+1)] + ["*"] + motif_resi[j:]
                            sub_motif_seq2 = motif_seq[i:(k+1)] + "*" + motif_seq[m:(j+1)]
                            sub_motif_resi2 = motif_resi[i:(k+1)] + ["*"] + motif_resi[m:(j+1)]
                            sub_motif_seq3 = motif_seq[k:gap_index] + "*" + motif_seq[(gap_index+1):(m+1)]
                            sub_motif_resi3 = motif_resi[k:gap_index] + ["*"] + motif_resi[(gap_index+1):(m+1)]
                            list_2way_submotifs.append([[sub_motif_seq1,sub_motif_resi1],[sub_motif_seq2,sub_motif_resi2],[sub_motif_seq3,sub_motif_resi3]])
    return list_2way_submotifs


def get_jar3d_energy_paras(RNAJP_HOME,seq,dict_motifs,outfolder):
    dict_model_instance_num = get_model_instance_num(RNAJP_HOME)

    dict_jar3d_paras = {}
    list_2way_found_no_motifs = []
    list_hairpin_found_no_motifs = []

    seq = seq.strip().upper()
    seq_without_space = seq.replace(" ","")
    for key in dict_motifs:
        if key == "2way_loops":
            for motif in dict_motifs[key]:
                ib = motif[0][0]
                ie = motif[0][1]
                motif_seq = []
                motif_resi = []
                for i in range(ib-1,ie):
                    motif_seq.append(seq_without_space[i])
                    motif_resi.append(i+1)
                motif_seq.append("*")
                motif_resi.append("*")
                ib = motif[1][0]
                ie = motif[1][1]
                for i in range(ib-1,ie):
                    motif_seq.append(seq_without_space[i])
                    motif_resi.append(i+1)
                motif_seq = "".join(motif_seq)

                dict_jar3d_paras0 = get_jar3d_paras_for_known_motif(RNAJP_HOME,motif_seq,motif_resi)
                if dict_jar3d_paras0 is not None:
                    dict_jar3d_paras = combine_two_dict(dict_jar3d_paras,dict_jar3d_paras0)
                    #print ("*************** From known IL motif ****************")
                    #for key0 in dict_jar3d_paras0:
                        #print (key0,dict_jar3d_paras0[key0])
                    continue

                dict_jar3d_paras0, motif_score = get_jar3d_paras_for_motif(RNAJP_HOME,dict_model_instance_num,motif_seq,motif_resi,outfolder)
                if dict_jar3d_paras0 is None:
                    list_2way_submotifs = split_2way_motif(motif_seq,motif_resi)
                    #print(f"{len(list_2way_submotifs)} submotifs for motif {motif_seq}",flush=True)
                    if len(list_2way_submotifs) == 0:
                        list_2way_found_no_motifs.append(motif)
                        continue

                    list_dict_jar3d_paras_submotifs = []
                    list_submotifs_score = []
                    list_invalid_submotifs = []
                    for num_motif, submotifs in enumerate(list_2way_submotifs):
                        #print(f"submotif {num_motif}",flush=True)
                        if submotifs[0][0] in list_invalid_submotifs:
                            list_submotifs_score.append(-10000)
                            if len(submotifs) == 2:
                                list_dict_jar3d_paras_submotifs.append([None,None])
                            else:
                                list_dict_jar3d_paras_submotifs.append([None,None,None])
                            continue
                        if submotifs[1][0] in list_invalid_submotifs:
                            list_submotifs_score.append(-10000)
                            if len(submotifs) == 2:
                                list_dict_jar3d_paras_submotifs.append([None,None])
                            else:
                                list_dict_jar3d_paras_submotifs.append([None,None,None])
                            continue
                        if len(submotifs) == 3:
                            if submotifs[2][0] in list_invalid_submotifs:
                                list_submotifs_score.append(-10000)
                                list_dict_jar3d_paras_submotifs.append([None,None,None])
                                continue

                        dict_jar3d_paras1 = get_jar3d_paras_for_known_motif(RNAJP_HOME,submotifs[0][0],submotifs[0][1])
                        if dict_jar3d_paras1 is None:
                            dict_jar3d_paras1, motif_score1 = get_jar3d_paras_for_motif(RNAJP_HOME,dict_model_instance_num,submotifs[0][0],submotifs[0][1],outfolder)
                        else:
                            motif_score1 = 100.0

                        dict_jar3d_paras2 = get_jar3d_paras_for_known_motif(RNAJP_HOME,submotifs[1][0],submotifs[1][1])
                        if dict_jar3d_paras2 is None:
                            dict_jar3d_paras2, motif_score2 = get_jar3d_paras_for_motif(RNAJP_HOME,dict_model_instance_num,submotifs[1][0],submotifs[1][1],outfolder)
                        else:
                            motif_score2 = 100.0

                        if motif_score1 is None:
                            list_invalid_submotifs.append(submotifs[0][0])
                        elif motif_score1 < 60.0:
                            list_invalid_submotifs.append(submotifs[0][0])

                        if motif_score2 is None:
                            list_invalid_submotifs.append(submotifs[1][0])
                        elif motif_score2 < 60.0:
                            list_invalid_submotifs.append(submotifs[1][0])

                        if len(submotifs) == 2:
                            if motif_score1 is None or motif_score2 is None:
                                list_submotifs_score.append(-10000)
                            elif motif_score1 < 60.0 or motif_score2 < 60.0:
                                list_submotifs_score.append(-10000)
                            else:
                                list_submotifs_score.append((motif_score1+motif_score2)/2.)
                            list_dict_jar3d_paras_submotifs.append([dict_jar3d_paras1,dict_jar3d_paras2])
                        elif len(submotifs) == 3:
                            dict_jar3d_paras3 = get_jar3d_paras_for_known_motif(RNAJP_HOME,submotifs[2][0],submotifs[2][1])
                            if dict_jar3d_paras3 is None:
                                dict_jar3d_paras3, motif_score3 = get_jar3d_paras_for_motif(RNAJP_HOME,dict_model_instance_num,submotifs[2][0],submotifs[2][1],outfolder)
                            else:
                                motif_score3 = 100.0

                            if motif_score3 is None:
                                list_invalid_submotifs.append(submotifs[2][0])
                            elif motif_score3 < 60.0:
                                list_invalid_submotifs.append(submotifs[2][0])
                            if motif_score1 is None or motif_score2 is None or motif_score3 is None:
                                list_submotifs_score.append(-10000)
                            elif motif_score1 < 60.0 or motif_score2 < 60.0 or motif_score3 < 60.0:
                                list_submotifs_score.append(-10000)
                            else:
                                list_submotifs_score.append((motif_score1+motif_score2+motif_score3)/3.)
                            list_dict_jar3d_paras_submotifs.append([dict_jar3d_paras1,dict_jar3d_paras2,dict_jar3d_paras3])
                    
                    if max(list_submotifs_score) == -10000:
                        print(f"Cannot find JAR3D motif for the internal loop: {motif_seq}, and thus skip JAR3D interactions for this loop.")
                        list_2way_found_no_motifs.append(motif)
                        continue
                    else:
                        max_motif_score_idx = np.argmax(list_submotifs_score)
                        dict_jar3d_paras1 = list_dict_jar3d_paras_submotifs[max_motif_score_idx][0]
                        dict_jar3d_paras2 = list_dict_jar3d_paras_submotifs[max_motif_score_idx][1]
                        dict_jar3d_paras = combine_two_dict(dict_jar3d_paras,dict_jar3d_paras1)
                        dict_jar3d_paras = combine_two_dict(dict_jar3d_paras,dict_jar3d_paras2)
                        if len(list_dict_jar3d_paras_submotifs[max_motif_score_idx]) == 3:
                            dict_jar3d_paras3 = list_dict_jar3d_paras_submotifs[max_motif_score_idx][2]
                            dict_jar3d_paras = combine_two_dict(dict_jar3d_paras,dict_jar3d_paras3)
                        #print ("Find submotifs:")
                        #print (list_2way_submotifs[max_motif_score_idx])
                        #print (list_submotifs_score[max_motif_score_idx])
                else:
                    if motif in dict_motifs["2way_loops_non_PK"]:
                        dict_jar3d_paras = combine_two_dict(dict_jar3d_paras,dict_jar3d_paras0)
                    else:
                        if motif_score >= 80.0:
                            dict_jar3d_paras = combine_two_dict(dict_jar3d_paras,dict_jar3d_paras0)
                        else:
                            list_2way_found_no_motifs.append(motif)
                            continue

        if key == "hairpin_loops":
            for motif in dict_motifs[key]:
                ib = motif[0][0]
                ie = motif[0][1]
                motif_seq = []
                motif_resi = []
                for i in range(ib-1,ie):
                    motif_seq.append(seq_without_space[i])
                    motif_resi.append(i+1)
                motif_seq = "".join(motif_seq)

                dict_jar3d_paras0 = get_jar3d_paras_for_known_motif(RNAJP_HOME,motif_seq,motif_resi)
                if dict_jar3d_paras0 is not None:
                    dict_jar3d_paras = combine_two_dict(dict_jar3d_paras,dict_jar3d_paras0)
                    #print ("*************** From known HL motif ****************")
                    #for key0 in dict_jar3d_paras0:
                        #print (key0,dict_jar3d_paras0[key0])
                    continue

                dict_jar3d_paras0, motif_score = get_jar3d_paras_for_motif(RNAJP_HOME,dict_model_instance_num,motif_seq,motif_resi,outfolder)
                if dict_jar3d_paras0 is None:
                    if motif not in dict_motifs["hairpin_loops_non_PK"]:
                        list_hairpin_found_no_motifs.append(motif)
                        continue

                    list_hairpin_submotifs = split_hairpin_motif(motif_seq,motif_resi)
                    if len(list_hairpin_submotifs) == 0:
                        list_hairpin_found_no_motifs.append(motif)
                        continue

                    list_dict_jar3d_paras_submotifs = []
                    list_submotifs_score = []
                    for submotifs in list_hairpin_submotifs:
                        dict_jar3d_paras1, motif_score1 = get_jar3d_paras_for_motif(RNAJP_HOME,dict_model_instance_num,submotifs[0][0],submotifs[0][1],outfolder)
                        dict_jar3d_paras2, motif_score2 = get_jar3d_paras_for_motif(RNAJP_HOME,dict_model_instance_num,submotifs[1][0],submotifs[1][1],outfolder)
                        if motif_score1 is None or motif_score2 is None:
                            list_submotifs_score.append(-10000)
                        else:
                            list_submotifs_score.append(motif_score1+motif_score2)
                        list_dict_jar3d_paras_submotifs.append([dict_jar3d_paras1,dict_jar3d_paras2])
                    
                    if max(list_submotifs_score) == -10000:
                        print(f"Cannot find JAR3D motif for the hairpin loop: {motif_seq}, and thus skip JAR3D interactions for this loop.")
                        list_hairpin_found_no_motifs.append(motif)
                        continue
                    else:
                        max_motif_score_idx = np.argmax(list_submotifs_score)
                        dict_jar3d_paras1 = list_dict_jar3d_paras_submotifs[max_motif_score_idx][0]
                        dict_jar3d_paras2 = list_dict_jar3d_paras_submotifs[max_motif_score_idx][1]
                        dict_jar3d_paras = combine_two_dict(dict_jar3d_paras,dict_jar3d_paras1)
                        dict_jar3d_paras = combine_two_dict(dict_jar3d_paras,dict_jar3d_paras2)
                        #print ("Find submotifs:")
                        #print (list_hairpin_submotifs[max_motif_score_idx])
                        #print (list_submotifs_score[max_motif_score_idx])
                else:
                    if motif in dict_motifs["hairpin_loops_non_PK"]:
                        dict_jar3d_paras = combine_two_dict(dict_jar3d_paras,dict_jar3d_paras0)
                    else:
                        if motif_score >= 80.0:
                            dict_jar3d_paras = combine_two_dict(dict_jar3d_paras,dict_jar3d_paras0)
                        else:
                            list_hairpin_found_no_motifs.append(motif)
                            continue

    list_bulge_nts_in_jar3d = []
    list_nts_in_2way_hairpin = []
    for key in dict_motifs:
        if key == "2way_loops":
            for motif in dict_motifs[key]:
                if motif in list_2way_found_no_motifs:
                    continue
                ib = motif[0][0]
                ie = motif[0][1]
                list_nts_in_2way_hairpin.extend(list(range(ib,ie+1)))
                ib = motif[1][0]
                ie = motif[1][1]
                list_nts_in_2way_hairpin.extend(list(range(ib,ie+1)))
        elif key == "hairpin_loops":
            for motif in dict_motifs[key]:
                if motif in list_hairpin_found_no_motifs:
                    continue
                ib = motif[0][0]
                ie = motif[0][1]
                list_nts_in_2way_hairpin.extend(list(range(ib,ie+1)))

    list_nts_in_jar3d = []
    for key in dict_jar3d_paras:
        list_nts_in_jar3d.extend(list(key))
    for i in list_nts_in_2way_hairpin:
        list_bulge_nts_in_jar3d.append(i)
    return dict_jar3d_paras, list_2way_found_no_motifs, list_bulge_nts_in_jar3d
