from matplotlib import pyplot as plt
import numpy as np
from itertools import groupby
import itertools
import matplotlib
import pandas as pd
import seaborn as sns
import os
import re
from scipy import stats

def fasta_iter(fasta_name):
    """
    given a fasta file. yield tuples of header, sequence
    """
    fh = open(fasta_name)
    # ditch the boolean (x[0]) and just keep the header or sequence since
    # we know they alternate.
    faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
    for header in faiter:
        # drop the ">"
        header = header.__next__()[1:].strip()
        # join all sequence lines to one.
        seq = "".join(s.strip() for s in faiter.__next__())
        print(header)
        yield header, seq

def read_gene_densities(file_path, file_name, suffixes):
    """Read gene densities from files with given suffixes."""
    with open(file_path + file_name + suffixes[0], 'r') as read_file:
        lines = read_file.readlines()

    densities1 = []
    for i in range(2, len(lines)):
        densities1.append(lines[i].replace('\n', '').replace('\r', '').split('\t'))
    densities1 = np.array(densities1).astype(np.float)

    with open(file_path + file_name + suffixes[1], 'r') as read_file:
        lines2 = read_file.readlines()

    densities2 = []
    for i in range(2, len(lines2)):
        densities2.append(lines2[i].replace('\n', '').replace('\r', '').split('\t'))
    densities2 = np.array(densities2).astype(np.float)

    max_density_index = int(max([densities1[-1, 0], densities2[-1, 0]]))
    combined_densities = np.zeros([max_density_index, 3])

    for i in range(len(densities1)):
        combined_densities[int(densities1[i, 0] - 1), 1] = densities1[i, 1]

    for i in range(len(densities2)):
        combined_densities[int(densities2[i, 0] - 1), 2] = densities2[i, 1]

    normalize_factor = np.sum(combined_densities[:, 1]) + np.sum(combined_densities[:, 2])
    combined_densities = combined_densities * 40000 / normalize_factor

    return combined_densities

# read wig files
def read_gene_densities2(filepath, filename, RiboNum = 40000):
    '''
    read wig file from filepath and filename
    read forward and reverse direction seperately
    RiboNum: total number of ribosomes
    '''
    with open(filepath + filename +"_f.wig",'r') as read_file:
            lines = read_file.readlines()

    # read forward direction
    Dwig1 = []
    for i in range(2,len(lines)):
        Dwig1.append(lines[i].replace('\n','').replace('\r','').split('\t'))
    Dwig1 = np.array(Dwig1).astype(np.float)

    with open(filepath + filename + "_r.wig",'r') as read_file:
            lines2 = read_file.readlines()

    # print wig file heads
    # print(lines2[0])
    # print(lines2[1])

    # read reverse direction
    Dwig2 = []
    for i in range(2,len(lines2)):
        Dwig2.append(lines2[i].replace('\n','').replace('\r','').split('\t'))
    Dwig2 = np.array(Dwig2).astype(np.float)

    # normalize the total reads
    normalize = np.sum(Dwig2[:,0]) + np.sum(Dwig1[:,0])
    L = len(Dwig1)
    Dwig = np.zeros((L,3))
    for i in range(len(Dwig1)):
        Dwig[i,0] = i
        Dwig[i,1] = Dwig1[i,0]*RiboNum/normalize
        Dwig[i,2] = Dwig2[i,0]*RiboNum/normalize
    return Dwig

#get the codon density
def sum_adjac(RD):
    res = []
    for i in range(int(len(RD)/3)):
        res.append(np.sum(RD[i*3:i*3+3]))
    return np.array(res)

# get pause scores
def get_pause_score(a_site, read_a_site, dwig, gene_data, sequence, y_pred, z_c, gene_index,
                    thres=0.003, normal_factor=10000, pred=0):
    codon_pause_scores = {
        ''.join(codon): [] for codon in itertools.product('ATGC', repeat=3)
    }

    for i in gene_index:
        update = 0
        strand, start, end = gene_data[i][2], int(gene_data[i][0]), int(gene_data[i][1])

        if end - start > 200:
            update = 1
            if strand == 1:
                read_density = dwig[start - 1 + a_site:end + a_site, 1]
                seq_t = sequence[start - 1:end]
            else:
                read_density = dwig[start - 1 - a_site:end - a_site, 2][::-1]
                seq_t = sequence[start - 1:end]
                seq_t = "".join(complement.get(base, base) for base in reversed(seq_t))

            if update == 1 and np.mean(read_density) > thres:
                codons = [seq_t[n:n + 3] for n in range(0, len(seq_t), 3)]
                r_pro_2 = np.copy(read_density)
                r_pro_2[0:15] = np.mean(read_density)
                r_pro_2[-15:] = np.mean(read_density)

                if pred == 1:
                    c_indices = np.where(z_c[:, 0] == i)[0]
                    for p in c_indices:
                        r_pro_2[int(z_c[p, 1]) * 3 - 30 + (read_a_site - a_site):int(z_c[p, 1]) * 3 + 3 - 30 + (
                                    read_a_site - a_site)] = 0
                        r_pro_2[int(z_c[p, 1]) * 3 + 1 - 30 + (read_a_site - a_site)] = (np.power(2, (
                        y_pred[p]) + 5) - 32) / normal_factor

                r_dc_2 = sum_adjac(r_pro_2)

                for j in range(5, len(r_dc_2) - 5):
                    if codons[j] in codon_pause_scores:
                        codon_pause_scores[codons[j]].append(r_dc_2[j] / np.mean(r_dc_2))

    return codon_pause_scores


# codon table with 3nts
pause_scores_aa = {
    'F':['TTT', 'TTC'],
    'L':['TTA', 'TTG', 'CTA', 'CTT', 'CTC', 'CTG'],
    'S':['TCA', 'TCT', 'TCC', 'TCG', 'AGT', 'AGC'],
    'Y':['TAT', 'TAC'],
    'C':['TGT', 'TGC'],
    'W':['TGG'],
    'P':['CCT', 'CCA', 'CCC', 'CCG'],
    'H':['CAT', 'CAC'],
    'Q':['CAG', 'CAA'],
    'R':['CGA', 'CGT', 'CGC', 'CGG', 'AGG', 'AGA'],
    'I':['ATC', 'ATT', 'ATA'],
    'M':['ATG'],
    'T':['ACA', 'ACT', 'ACG', 'ACC'],
    'N':['AAT', 'AAC'],
    'K':['AAA', 'AAG'],
    'V':['GTA', 'GTT', 'GTC', 'GTG'],
    'A':['GCC', 'GCG', 'GCA', 'GCT'],
    'D':['GAT', 'GAC'],
    'E':['GAA', 'GAG'],
    'G':['GGA', 'GGT', 'GGC', 'GGG'],
    }

# DNA complementary
complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

# codon list
codon_table = ['ATA', 'ATC', 'ATT', 'ATG',
        'ACA', 'ACC', 'ACG', 'ACT',
        'AAC', 'AAT', 'AAA', 'AAG',
        'AGC', 'AGT', 'AGA', 'AGG',
        'CTA', 'CTC', 'CTG', 'CTT',
        'CCA', 'CCC', 'CCG', 'CCT',
        'CAC', 'CAT', 'CAA', 'CAG',
        'CGA', 'CGC', 'CGG', 'CGT',
        'GTA', 'GTC', 'GTG', 'GTT',
        'GCA', 'GCC', 'GCG', 'GCT',
        'GAC', 'GAT', 'GAA', 'GAG',
        'GGA', 'GGC', 'GGG', 'GGT',
        'TCA', 'TCC', 'TCG', 'TCT',
        'TTC', 'TTT', 'TTA', 'TTG',
        'TAC', 'TAT', 'TAA', 'TAG',
        'TGC', 'TGT', 'TGA', 'TGG',
    ]
