import warnings
warnings.filterwarnings("ignore")
from preprocess_indel_files import preprocess_indel_files
from compute_summary_statistic import compute_summary_statistics
from simple_summary_analysis import avg_length_pred
import numpy as np
from sklearn import linear_model, metrics
from sklearn.model_selection import KFold
import pickle
from sequence_logos import plot_seq_logo
from sequence_logos import plot_QQ
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.feature_selection import f_regression
import glob
import csv
from scipy import stats
from sklearn.feature_selection import chi2
import copy
from sklearn.neural_network import MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.stats import entropy
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import ttest_ind_from_stats
import collections
from sklearn.metrics import f1_score

def load_gene_sequence_one_hot(spacer_list):

  sequence_pam_per_gene_grna = np.zeros((len(spacer_list), 6, 4), dtype = bool)

  for counter,spacer in enumerate(spacer_list):
    for i in range(6):
      sequence_pam_per_gene_grna[counter, i, one_hot_index(spacer[i])] = 1

  return np.reshape(sequence_pam_per_gene_grna, (len(sequence_pam_per_gene_grna), -1))


def oneI_oneD_fraction_over_total_finder(indel_prop_matrix,name_indel_type_unique):
    indel_num, site_num = np.shape(indel_prop_matrix)
    oneI_indicator = np.zeros(indel_num)
    oneI_fraction = np.zeros(site_num)
    oneD_fraction = np.zeros(site_num)

    for counter, cigar in enumerate(name_indel_type_unique):
        if ":1I" in cigar:
            oneI_indicator[counter] = 1

    oneD_indicator = np.zeros(indel_num)
    for counter, cigar in enumerate(name_indel_type_unique):
        if ":1D" in cigar:# and cigar.count(':')==1:
            oneD_indicator[counter] = 1

    for site_index in range(site_num):
        oneI_fraction[site_index] = np.inner(oneI_indicator, indel_prop_matrix[:, site_index])
        oneD_fraction[site_index] = np.inner(oneD_indicator, indel_prop_matrix[:, site_index])

    return oneI_fraction,oneD_fraction

def top_indel_length_matrix_finder(indel_count_matrix,indel_length_deletion,indel_length_insertion):
    [indel_num,site_num]=np.shape(indel_count_matrix)

    top_deletion_length_matrix = np.zeros((3,site_num))
    top_insertion_length_matrix = np.zeros((3,site_num))
    for siteidx in range(site_num):
        top_deletion_length_matrix[:,siteidx] = indel_length_deletion[np.argsort(indel_count_matrix[:,siteidx])[::-1][0:3]]
        top_insertion_length_matrix[:, siteidx] = indel_length_insertion[np.argsort(indel_count_matrix[:,siteidx])[::-1][0:3]]


    return top_deletion_length_matrix
    return top_insertion_length_matrix

def oneI_oneD_fraction_finder(indel_count_matrix,name_indel_type_unique):
    indel_num, site_num = np.shape(indel_count_matrix)
    oneI_indicator = np.zeros(indel_num)
    oneI_fraction = np.zeros(site_num)
    oneD_fraction = np.zeros(site_num)

    for counter, cigar in enumerate(name_indel_type_unique):
        if ":1I" in cigar:
            oneI_indicator[counter] = 1

    oneD_indicator = np.zeros(indel_num)
    for counter, cigar in enumerate(name_indel_type_unique):
        if ":1D" in cigar:# and cigar.count(':')==1:
            oneD_indicator[counter] = 1

    indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))
    for site_index in range(site_num):
        oneI_fraction[site_index] = np.inner(oneI_indicator, indel_fraction_mutant_matrix[:, site_index])
        oneD_fraction[site_index] = np.inner(oneD_indicator, indel_fraction_mutant_matrix[:, site_index])

    return oneI_fraction,oneD_fraction


def entrop_finder(indel_count_matrix):
    num_indels, num_sites = np.shape(indel_count_matrix)
    indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))
    entrop = []
    for col in range(num_sites):
        vec = np.copy(indel_fraction_mutant_matrix[:, col])
        vec = np.sort(vec)[::-1]
        entrop.append(entropy(vec))

    return np.asarray(entrop)

def flatness_finder(indel_count_matrix):
    num_indels, num_sites = np.shape(indel_count_matrix)
    indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))
    max_grad = []
    for col in range(num_sites):
        vec = np.copy(indel_fraction_mutant_matrix[:, col])
        vec = np.sort(vec)[::-1]
        max_grad.append(max(abs(np.gradient(vec))))

    return np.asarray(max_grad)



def flatness_finder(indel_count_matrix):
    num_indels, num_sites = np.shape(indel_count_matrix)
    indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))
    max_grad = []
    for col in range(num_sites):
        vec = np.copy(indel_fraction_mutant_matrix[:, col])
        vec = np.sort(vec)[::-1]
        max_grad.append(max(abs(np.gradient(vec))))

    return np.asarray(max_grad)

def length_of_repeat_finder(seq):
    maxlen = 2
    start = 0
    while start < len(seq) - 1:
        pointer = 2
        nuc1 = seq[start]
        nuc2 = seq[start + 1]
        templen = 2
        while start + pointer < len(seq) and nuc1 != nuc2:
            # print templen
            if pointer % 2 == 0:
                if seq[start + pointer] != nuc1:
                    pointer += 1
                    break
                templen += 1
                if templen > maxlen:
                    maxlen = templen

            if pointer % 2 == 1:
                if seq[start + pointer] != nuc2:
                    pointer += 1
                    if templen > maxlen:
                        maxlen = templen
                    break
                templen += 1
                if templen > maxlen:
                    maxlen = templen

            pointer += 1
        start = start + 1
    return maxlen


def coding_region_finder(name_genes_grna_unique):
    intron_exon_dict = pickle.load(open('storage/intron_exon_status.pkl', 'rb'))
    location_dict = {}
    with open('sequence_pam_gene_grna_big_file_donor_genomic_context.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        row_counter = 0
        for row in spamreader:
            location_dict[row[0].split(',')[0]]=row[0].split(',')[4]

    intron_exon_label_vec = []

    for site_name in name_genes_grna_unique:
        site_name_list = site_name.split('-')
        location = location_dict[site_name_list[1] + '-' + site_name_list[2]]

        if 2. in intron_exon_dict[location]: # if we find ANY 2 we count as exon
            intron_exon_label_vec.append(2)
        elif 1. in intron_exon_dict[location]:
            intron_exon_label_vec.append(1)
        else:
            intron_exon_label_vec.append(0)

    intron_exon_label_vec = np.asarray(intron_exon_label_vec)
    return intron_exon_label_vec

def eff_vec_finder(indel_count_matrix,name_genes_grna_unique):
    num_indel,num_site = np.shape(indel_count_matrix)
    dict_eff = {}
    for filename in glob.glob('/Users/amirali/Projects/muteff/*.txt'):
        file = open(filename)
        for line in file:
            if 'RL384' in line:
                line = line.replace('_','-')
                line = line.replace('"', '')
                if 'N' not in line.split(',')[1]:
                    eff = float(line.split(',')[1])
                    line_list = (line.split(',')[0]).split('-')
                    dict_eff[line_list[1]+'-'+line_list[2]] = eff


    eff_vec = np.zeros(num_site)
    site = 0
    for site_name in name_genes_grna_unique:
        site_name_list = site_name.split('-')
        eff_vec[site] = dict_eff[site_name_list[1] + '-' + site_name_list[2]]
        site += 1

    return eff_vec



def expected_deletion_insertion_length(indel_count_matrix,length_indel_insertion,length_indel_deletion):
  indel_num,site_num = np.shape(indel_count_matrix)

  exp_insertion_length = np.zeros(site_num,dtype=float)
  exp_deletion_length = np.zeros(site_num,dtype=float)

  insertion_indicator = np.copy(length_indel_insertion)
  deletion_indicator = np.copy(length_indel_deletion)

  insertion_indicator[insertion_indicator>0]=1.
  deletion_indicator[deletion_indicator>0]=1.

  indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))

  insertion_only_fraction_matrix =  np.multiply(indel_fraction_mutant_matrix, np.reshape(insertion_indicator,(-1,1)) )
  deletion_only_fraction_matrix = np.multiply(indel_fraction_mutant_matrix,  np.reshape(deletion_indicator,(-1,1)) )

  insertion_only_fraction_matrix = insertion_only_fraction_matrix / np.reshape(np.sum(insertion_only_fraction_matrix, axis=0), (1, -1))
  deletion_only_fraction_matrix = deletion_only_fraction_matrix / np.reshape(np.sum(deletion_only_fraction_matrix, axis=0), (1, -1))


  for site_index in range(site_num):
    exp_insertion_length[site_index] = np.inner(length_indel_insertion,insertion_only_fraction_matrix[:,site_index])
    exp_deletion_length[site_index] = np.inner(length_indel_deletion, deletion_only_fraction_matrix[:, site_index])

  # some sites do not get any insertions. this cuase nan. we make those entries zero.
  for i in range(np.size(exp_insertion_length)):
    if np.isnan(exp_insertion_length[i]):
      exp_insertion_length[i] = 0

  return exp_insertion_length,exp_deletion_length


def fraction_of_deletion_insertion(indel_count_matrix,length_indel_insertion,length_indel_deletion):
  indel_num,site_num = np.shape(indel_count_matrix)

  prop_insertions_gene_grna = np.zeros(site_num,dtype=float)
  prop_deletions_gene_grna = np.zeros(site_num,dtype=float)


  insertion_indicator = np.copy(length_indel_insertion)
  deletion_indicator = np.copy(length_indel_deletion)

  insertion_indicator[insertion_indicator>0]=1.
  deletion_indicator[deletion_indicator>0]=1.

  indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))

  for site_index in range(site_num):
    prop_insertions_gene_grna[site_index] = np.inner(insertion_indicator,indel_fraction_mutant_matrix[:,site_index])
    prop_deletions_gene_grna[site_index] = np.inner(deletion_indicator, indel_fraction_mutant_matrix[:, site_index])

  return prop_insertions_gene_grna,prop_deletions_gene_grna




def fraction_of_deletion_insertion_porportion(indel_prop_matrix,length_indel_insertion,length_indel_deletion):
  indel_num,site_num = np.shape(indel_prop_matrix)

  prop_insertions_gene_grna = np.zeros(site_num,dtype=float)
  prop_deletions_gene_grna = np.zeros(site_num,dtype=float)


  insertion_indicator = np.copy(length_indel_insertion)
  deletion_indicator = np.copy(length_indel_deletion)

  insertion_indicator[insertion_indicator>0]=1.
  deletion_indicator[deletion_indicator>0]=1.

  for site_index in range(site_num):
    prop_insertions_gene_grna[site_index] = np.inner(insertion_indicator,indel_prop_matrix[:,site_index])
    prop_deletions_gene_grna[site_index] = np.inner(deletion_indicator, indel_prop_matrix[:, site_index])

  return prop_insertions_gene_grna,prop_deletions_gene_grna




def one_hot_index(nucleotide):
  if nucleotide == 'g':
    nucleotide = 'G'
  elif nucleotide == 'a':
    nucleotide = 'A'
  elif nucleotide == 'c':
    nucleotide = 'C'
  elif nucleotide == 't':
    nucleotide = 'T'
  nucleotide_array = ['A', 'C', 'G', 'T']
  return nucleotide_array.index(nucleotide)

def load_gene_sequence(sequence_file_name, name_genes_grna_unique,homopolymer_matrix,intron_exon_label_vec):
  # Create numpy matrix of size len(name_genes_grna_unique) * 23, to store the sequence as one-hot encoded
  sequence_pam_per_gene_grna_small = np.zeros((len(name_genes_grna_unique), 6, 4), dtype = bool)
  sequence_pam_per_gene_grna = np.zeros((len(name_genes_grna_unique), 23, 4), dtype = bool)
  sequence_pam_per_gene_grna_strand = np.zeros((len(name_genes_grna_unique), 24, 4), dtype = bool)
  sequence_pam_homop_per_gene_grna = np.zeros((len(name_genes_grna_unique), 24, 4))
  sequence_pam_repeat_per_gene_grna = np.zeros((len(name_genes_grna_unique), 24, 4))
  sequence_pam_chromatin_per_gene_grna = np.zeros((len(name_genes_grna_unique), 24, 4))
  sequence_pam_coding_gccontent_per_gene_grna = np.zeros((len(name_genes_grna_unique), 24, 4))
  sequence_genom_context_gene_grna = np.zeros((len(name_genes_grna_unique), 100, 4), dtype=bool)

  # Obtain the grna and PAM sequence corresponding to name_genes_grna_unique
  with open(sequence_file_name) as f:
    for line in f:
      line = line.replace('"', '')
      line = line.replace(' ', '')
      line = line.replace('\n', '')
      l = line.split(',')
      if l[1] + '-' + l[0] in name_genes_grna_unique:
        index_in_name_genes_grna_unique = name_genes_grna_unique.index(l[1] + '-' + l[0])
        for i in range(6):
          sequence_pam_per_gene_grna_small[index_in_name_genes_grna_unique, i, one_hot_index(l[2][i+14])] = 1
        for i in range(20):
          sequence_pam_per_gene_grna[index_in_name_genes_grna_unique, i, one_hot_index(l[2][i])] = 1
          sequence_pam_per_gene_grna_strand[index_in_name_genes_grna_unique, i, one_hot_index(l[2][i])] = 1
          sequence_pam_homop_per_gene_grna[index_in_name_genes_grna_unique, i, one_hot_index(l[2][i])] = 1
          sequence_pam_coding_gccontent_per_gene_grna[index_in_name_genes_grna_unique, i, one_hot_index(l[2][i])] = 1
          sequence_pam_repeat_per_gene_grna[index_in_name_genes_grna_unique, i, one_hot_index(l[2][i])] = 1
          sequence_pam_chromatin_per_gene_grna[index_in_name_genes_grna_unique, i, one_hot_index(l[2][i])] = 1
        for i in range(3):
          sequence_pam_per_gene_grna[index_in_name_genes_grna_unique, 20 + i, one_hot_index(l[3][i])] = 1
          sequence_pam_per_gene_grna_strand[index_in_name_genes_grna_unique, 20 + i, one_hot_index(l[3][i])] = 1
          sequence_pam_homop_per_gene_grna[index_in_name_genes_grna_unique, 20 + i, one_hot_index(l[3][i])] = 1
          sequence_pam_coding_gccontent_per_gene_grna[index_in_name_genes_grna_unique, 20 + i, one_hot_index(l[3][i])] = 1
          sequence_pam_repeat_per_gene_grna[index_in_name_genes_grna_unique, 20 + i, one_hot_index(l[3][i])] = 1
          sequence_pam_chromatin_per_gene_grna[index_in_name_genes_grna_unique, 20 + i, one_hot_index(l[3][i])] = 1

        if length_of_repeat_finder(l[2])>4:
          sequence_pam_repeat_per_gene_grna[index_in_name_genes_grna_unique, 23 , 0] = 1

        if l[4][0] == l[7][0]:
          sequence_pam_per_gene_grna_strand[index_in_name_genes_grna_unique, 23 , 0] = 1


        #sequence_pam_chromatin_per_gene_grna[index_in_name_genes_grna_unique, 23 , 0] = np.nanmean(chrom_mat[l[1] + '-' + l[0]],axis=0)[chrom_col]

        homopolymer_matrix_binary = np.copy(homopolymer_matrix)
        homopolymer_matrix_binary = homopolymer_matrix_binary > 1
        sequence_pam_homop_per_gene_grna[index_in_name_genes_grna_unique, 23 , :] = homopolymer_matrix_binary[:,index_in_name_genes_grna_unique]
        if intron_exon_label_vec[index_in_name_genes_grna_unique] == 2: # if exon
          sequence_pam_coding_gccontent_per_gene_grna[index_in_name_genes_grna_unique, 23 , 0] = 1
        for i in range(100):
          sequence_genom_context_gene_grna[index_in_name_genes_grna_unique, i, one_hot_index(l[6][i])] = 1

        # sequence_pam_coding_gccontent_per_gene_grna[index_in_name_genes_grna_unique, 23, 1] = np.sum(sequence_pam_per_gene_grna[index_in_name_genes_grna_unique,:20,1:3]) / float(np.sum(sequence_pam_per_gene_grna[index_in_name_genes_grna_unique,:20,:]))
        sequence_pam_coding_gccontent_per_gene_grna[index_in_name_genes_grna_unique, 23, 1] = np.sum(sequence_genom_context_gene_grna[index_in_name_genes_grna_unique, :100, 1:3]) / float(np.sum(sequence_genom_context_gene_grna[index_in_name_genes_grna_unique, :100, :]))

  #plot_seq_logo(np.mean(sequence_pam_per_gene_grna, axis=0), "input_spacer")
  # Scikit needs only a 2-d matrix as input, so reshape and return
  return  np.reshape(sequence_pam_per_gene_grna_strand,(len(sequence_pam_per_gene_grna_strand),-1)),  np.reshape(sequence_pam_per_gene_grna_small,(len(sequence_pam_per_gene_grna_small), -1) ),    np.reshape(sequence_genom_context_gene_grna, (len(sequence_pam_repeat_per_gene_grna), -1)) ,  np.reshape(sequence_pam_chromatin_per_gene_grna, (len(sequence_pam_chromatin_per_gene_grna), -1))      ,np.reshape(sequence_pam_repeat_per_gene_grna, (len(sequence_genom_context_gene_grna), -1)) ,np.reshape(sequence_pam_coding_gccontent_per_gene_grna, (len(sequence_pam_coding_gccontent_per_gene_grna), -1))  ,np.reshape(sequence_pam_homop_per_gene_grna, (len(sequence_pam_homop_per_gene_grna), -1)),np.reshape(sequence_pam_per_gene_grna, (len(name_genes_grna_unique), -1)), np.reshape(sequence_pam_per_gene_grna[:, :20, :], (len(name_genes_grna_unique), -1)), np.reshape(sequence_pam_per_gene_grna[:, 20:, :], (len(name_genes_grna_unique), -1))

def load_gene_sequence_k_mer(sequence_file_name, name_genes_grna_unique, k):
  # Create numpy matrix of size len(name_genes_grna_unique) * 23, to store the sequence first
  sequence_pam_per_gene_grna = np.zeros((len(name_genes_grna_unique), 23), dtype = int)
  # Obtain the grna and PAM sequence corresponding to name_genes_grna_unique
  with open(sequence_file_name) as f:
    for line in f:
      line = line.replace('"', '')
      line = line.replace(' ', '')
      line = line.replace('\n', '')
      l = line.split(',')
      if l[1] + '-' + l[0] in name_genes_grna_unique:
        index_in_name_genes_grna_unique = name_genes_grna_unique.index(l[1] + '-' + l[0])
        for i in range(20):
          sequence_pam_per_gene_grna[index_in_name_genes_grna_unique, i] = one_hot_index(l[2][i])
        for i in range(3):
          sequence_pam_per_gene_grna[index_in_name_genes_grna_unique, 20 + i] = one_hot_index(l[3][i])
  # Compute k_mers
  k_mer_list = np.zeros((len(name_genes_grna_unique), 4**k), dtype = int)
  for i in range(len(name_genes_grna_unique)):
    for j in range(23 - k + 1):
      k_mer = 0
      for r in range(k):
        k_mer += sequence_pam_per_gene_grna[i][j + r]*(4**(k - r - 1))
      k_mer_list[i, k_mer] += 1
  return k_mer_list


def perform_linear_regression(sequence_pam_per_gene_grna, count_insertions_gene_grna_binary, count_deletions_gene_grna_binary, train_index, test_index, ins_coeff, del_coeff,lin_reg, to_plot = False):

  #print "----"
  #print "Number of positive testing samples in insertions is %f" % np.sum(count_insertions_gene_grna_binary[test_index])
  #print "Total number of testing samples %f" % np.size(test_index)
  #print "Number of positive training samples in insertions is %f" % np.sum(count_insertions_gene_grna_binary[train_index])
  #print "Total number of training samples %f" % np.size(train_index)
  lin_reg.fit(sequence_pam_per_gene_grna[train_index], count_insertions_gene_grna_binary[train_index])
  lin_reg_pred = lin_reg.predict(sequence_pam_per_gene_grna[test_index])
  insertions_r2_score = lin_reg.score(sequence_pam_per_gene_grna[test_index], count_insertions_gene_grna_binary[test_index])

  #insertions_r2_score = metrics.accuracy_score(count_insertions_gene_grna_binary[test_index], lin_reg_pred)
  #insertions_f1 = f1_score(count_insertions_gene_grna_binary[test_index], log_reg_pred)

  insertion_rmse = sqrt(mean_squared_error(lin_reg_pred,count_insertions_gene_grna_binary[test_index]))





  # ins_coeff.append(lin_reg.coef_)
  #if to_plot:
    #pickle.dump(lin_reg, open('models/edit_eff_tcells_chrom.p', 'wb'))
    #pvalue_vec = f_regression(sequence_pam_per_gene_grna[test_index],lin_reg_pred, center=True)[1]
    #print np.shape(sequence_pam_per_gene_grna[test_index])
    #print np.shape(lin_reg_pred)
    #scores, pvalue_vec = chi2(sequence_pam_per_gene_grna[test_index], lin_reg_pred)
    #pvalue_vec = f_regression(sequence_pam_per_gene_grna, count_deletions_gene_grna_binary)[1]
    #plot_QQ(lin_reg_pred,count_insertions_gene_grna_binary[test_index],'QQ_linear_insertion')
    #plot_seq_logo(lin_reg.coef_, "Insertion_linear")
    #plot_seq_logo(-np.log10(pvalue_vec), "Insertion_linear_pvalue")
    #print pvalue_vec
    #print lin_reg.coef_
    #pickle.dump(lin_reg, open('models/diversity_regression.p', 'wb'))
    #print 'Insertion -log10(p-value) of last 4 entries', -np.log10(pvalue_vec)[-4:]
    #print 'Insertion last four coefficients', lin_reg.coef_[-4:]

  #insertions_r2_score = lin_reg.score(sequence_pam_per_gene_grna[test_index], count_insertions_gene_grna_binary[test_index])
  #print "Test mse_score score for insertions: %f" % insertions_r2_score
  #print "Train mse_score score for insertions: %f" % lin_reg.score(sequence_pam_per_gene_grna[train_index], count_insertions_gene_grna_binary[train_index])
  #print "----"
  #print "Number of positive testing samples in deletions is %f" % np.sum(count_deletions_gene_grna_binary[test_index])
  #print "Total number of testing samples %f" % np.size(test_index)
  #print "Number of positive training samples in deletions is %f" % np.sum(count_deletions_gene_grna_binary[train_index])
  #print "Total number of training samples %f" % np.size(train_index)
  lin_reg.fit(sequence_pam_per_gene_grna[train_index], count_deletions_gene_grna_binary[train_index])
  lin_reg_pred = lin_reg.predict(sequence_pam_per_gene_grna[test_index])
  deletions_r2_score = lin_reg.score(sequence_pam_per_gene_grna[test_index], count_deletions_gene_grna_binary[test_index])
  deletion_rmse = sqrt(mean_squared_error(lin_reg_pred, count_deletions_gene_grna_binary[test_index]))
  #deletion_rmse = f1_score(count_insertions_gene_grna_binary[test_index], lin_reg_pred)



  # del_coeff.append(lin_reg.coef_)
  # if to_plot:
  #   #print np.shape(sequence_pam_per_gene_grna[test_index])
  #   #print np.shape(lin_reg_pred)
  #   pvalue_vec = f_regression(sequence_pam_per_gene_grna[test_index], lin_reg_pred, center=True)[1]
  #   #scores, pvalue_vec = chi2(sequence_pam_per_gene_grna[test_index], lin_reg_pred)
  #   #pvalue_vec = f_regression(sequence_pam_per_gene_grna[train_index], count_deletions_gene_grna_binary[train_index])[1]
  #   plot_seq_logo(-np.log10(pvalue_vec)*np.sign(lin_reg.coef_), "Deletion_linear_pvalue" )
  #   plot_QQ(lin_reg_pred, count_deletions_gene_grna_binary[test_index], 'QQ_linear_deletion')
  #   plot_seq_logo(lin_reg.coef_, "Deletion_linear")
  #   print 'Deletion -log10(p-value) of last 4 entries', -np.log10(pvalue_vec)[-4:]
  #   print 'Deletion last four coefficients', lin_reg.coef_[-4:]
  #print "Test r2_score score for deletions: %f" % deletions_r2_score
  #print "Train r2_score score for deletions: %f" % lin_reg.score(sequence_pam_per_gene_grna[train_index], count_deletions_gene_grna_binary[train_index])
  return insertions_r2_score, deletions_r2_score, insertion_rmse, deletion_rmse


def cross_validation_model(sequence_pam_per_gene_grna, count_insertions_gene_grna, count_deletions_gene_grna,lin_reg):
  total_insertion_avg_r2_score = []
  total_deletion_avg_r2_score = []
  total_insertion_avg_rmse = []
  total_deletion_avg_rmse = []

  ins_coeff = []
  del_coeff = []
  for repeat in range(10):
    print "repeat", repeat
    number_of_splits = 5
    fold_valid = KFold(n_splits = number_of_splits, shuffle = True, random_state = repeat)
    insertion_avg_r2_score = 0.0
    deletion_avg_r2_score = 0.0
    insertion_avg_rmse = 0.0
    deletion_avg_rmse = 0.0
    #count_insertions_gene_grna_copy = np.reshape(count_insertions_gene_grna, (-1, 1))
    #count_deletions_gene_grna_copy = np.reshape(count_deletions_gene_grna, (-1, 1))
    fold = 0
    for train_index, test_index in fold_valid.split(sequence_pam_per_gene_grna):
      to_plot = False
      if repeat == 2 and fold == 2:
        to_plot = True
      score_score = perform_linear_regression(sequence_pam_per_gene_grna, count_insertions_gene_grna, count_deletions_gene_grna, train_index, test_index, ins_coeff, del_coeff,lin_reg,to_plot)
      insertion_avg_r2_score += score_score[0]
      deletion_avg_r2_score += score_score[1]
      insertion_avg_rmse += score_score[2]
      deletion_avg_rmse += score_score[3]

      fold += 1
    insertion_avg_r2_score /= float(number_of_splits)
    deletion_avg_r2_score /= float(number_of_splits)
    insertion_avg_rmse /= float(number_of_splits)
    deletion_avg_rmse /= float(number_of_splits)

    total_insertion_avg_r2_score.append(float(insertion_avg_r2_score))
    total_deletion_avg_r2_score.append(float(deletion_avg_r2_score))
    total_insertion_avg_rmse.append(float(insertion_avg_rmse))
    total_deletion_avg_rmse.append(float(deletion_avg_rmse))
  # Some float overflows are happening, I will fix this sometime next week. Printing the array, it seems fine.
  print "Average r2 for insertions predictions is %f" % np.mean(np.array(total_insertion_avg_r2_score, dtype = float))
  print "Std in r2 for insertions predictions is %f" % np.std(np.array(total_insertion_avg_r2_score, dtype = float))

  print "Average rmse for insertions predictions is %f" % np.mean(np.array(total_insertion_avg_rmse, dtype = float))
  print "Std in rmse for insertions predictions is %f" % np.std(np.array(total_insertion_avg_rmse, dtype = float))

  print "Average r2 for deletions predictions is %f" % np.mean(np.array(total_deletion_avg_r2_score, dtype = float))
  print "Std in r2 for deletions predictions is %f" % np.std(np.array(total_deletion_avg_r2_score, dtype = float))

  print "Average rmse for deletions predictions is %f" % np.mean(np.array(total_deletion_avg_rmse, dtype = float))
  print "Std in rmse for deletions predictions is %f" % np.std(np.array(total_deletion_avg_rmse, dtype = float))




sequence_file_name = "sequence_pam_gene_grna_big_file_donor_genomic_context_with_donor_30nt_new.csv"
data_folder = "/Users/amirali/Projects/30nt_again/"


print "T cell files loading ..."

name_genes_grna_unique = pickle.load(open('storage_30nt_new/sample_file_name_UNIQUE.p', 'rb'))
name_indel_type_unique = pickle.load(open('storage_30nt_new/name_indel_type_ALL.p', 'rb'))
indel_count_matrix = pickle.load(open('storage_30nt_new/indel_count_matrix_UNIQUE.p', 'rb'))

no_variant_vector = pickle.load(open('storage_30nt_new/no_variant_vector_UNIQUE.p', 'rb'))
other_vector = pickle.load(open('storage_30nt_new/other_vector_UNIQUE.p', 'rb'))
snv_vector = pickle.load(open('storage_30nt_new/snv_vector_UNIQUE.p', 'rb'))

indel_prop_matrix = indel_count_matrix / np.reshape(  np.sum(indel_count_matrix, axis=0)+no_variant_vector+snv_vector , (1, -1))

length_indel_insertion = pickle.load(open('storage_30nt_new/length_indel_insertion_ALL.p', 'rb'))
length_indel_deletion = pickle.load(open('storage_30nt_new/length_indel_deletion_ALL.p', 'rb'))
homopolymer_matrix = pickle.load(open('storage_30nt_new/homology_matrix_UNIQUE.p', 'rb'))
chrom_label_matrix = pickle.load(open('storage_30nt_new/chrom_label_matrix_UNIQUE.p', 'rb'))
insertion_matrix = pickle.load(open('storage_30nt_new/insertion_matrix_UNIQUE.p', 'rb'))
insertion_matrix = insertion_matrix / np.reshape(np.sum(insertion_matrix, axis=0), (1, -1))

spacer_list = pickle.load(open('storage_30nt_new/spacer_list_UNIQUE.p', 'rb'))
spacer_list_6nt = []
for spacer in spacer_list:
    spacer_list_6nt.append(spacer[14:20])

sequence_pam_per_gene_grna_6nt = load_gene_sequence_one_hot(spacer_list_6nt)


fraction_insertions = pickle.load(open('storage_30nt_new/fraction_insertions_UNIQUE.p', 'rb'))
fraction_deletions = pickle.load(open('storage_30nt_new/fraction_deletions_UNIQUE.p', 'rb'))
fraction_insertions_all = pickle.load(open('storage_30nt_new/fraction_insertions_all_UNIQUE.p', 'rb'))
fraction_deletions_all =  pickle.load(open('storage_30nt_new/fraction_deletions_all_UNIQUE.p', 'rb'))
exp_insertion_length = pickle.load(open('storage_30nt_new/exp_insertion_length_UNIQUE.p', 'rb'))
exp_deletion_length = pickle.load(open('storage_30nt_new/exp_deletion_length_UNIQUE.p', 'rb'))
eff_vec = pickle.load(open('storage_30nt_new/eff_vec_UNIQUE.p', 'rb'))
entrop = pickle.load(open('storage_30nt_new/entrop_UNIQUE.p', 'rb'))
fraction_insertions_all_no_others = pickle.load(open('storage_30nt_new/fraction_insertions_all_no_other_UNIQUE.p', 'rb'))
oneI_frac = pickle.load(open('storage_30nt_new/oneI_frac.p', 'rb'))
oneD_frac = pickle.load(open('storage_30nt_new/oneD_frac.p', 'rb'))
oneI_frac_total = pickle.load(open('storage_30nt_new/oneI_frac_total.p', 'rb'))
oneD_frac_total = pickle.load(open('storage_30nt_new/oneD_frac_total.p', 'rb'))

sequence_pam_per_gene_grna = pickle.load(open('storage_30nt_new/spacer_pam_per_site_one_hot_UNIQUE.p', 'rb'))
sequence_pam_per_gene_grna_CHROMATIN = np.concatenate((sequence_pam_per_gene_grna, chrom_label_matrix), axis=1)



print "Other cell type files loading ..."

chrom_label_matrix_VO = pickle.load(open('storage_VO/chrom_label_matrix.p', 'rb'))
insertion_matrix_VO = pickle.load(open('storage_VO/insertion_matrix.p', 'rb'))
insertion_matrix_max_VO = np.argmax(insertion_matrix_VO,axis=0)
insertion_matrix_VO = insertion_matrix_VO / np.reshape(np.sum(insertion_matrix_VO, axis=0), (1, -1))

fraction_insertions_VO = pickle.load(open('storage_VO/fraction_insertions.p', 'rb'))
fraction_deletions_VO = pickle.load(open('storage_VO/fraction_deletions.p', 'rb'))
fraction_insertions_all_VO = pickle.load(open('storage_VO/fraction_insertions_over_total.p', 'rb'))
fraction_deletions_all_VO =  pickle.load(open('storage_VO/fraction_deletions_over_total.p', 'rb'))
exp_insertion_length_VO = pickle.load(open('storage_VO/exp_insertion_length.p', 'rb'))
exp_deletion_length_VO = pickle.load(open('storage_VO/exp_deletion_length.p', 'rb'))
eff_vec_VO = pickle.load(open('storage_VO/eff_vec.p', 'rb'))
entrop_VO = pickle.load(open('storage_VO/entrop.p', 'rb'))
cell_type_VO = pickle.load(open('storage_VO/cell_type_vector.p', 'rb'))

indel_count_matrix_VO = pickle.load(open('storage_VO/indel_count_matrix.p', 'rb'))
indel_prop_matrix_VO = pickle.load(open('storage_VO/indel_prop_matrix.p', 'rb'))
name_indel_type_unique_VO = pickle.load(open('storage_VO/all_indel_names.p', 'rb'))

oneI_frac_total_VO,oneD_frac_total_VO = oneI_oneD_fraction_over_total_finder(indel_prop_matrix_VO,name_indel_type_unique_VO)
oneI_frac_VO,oneD_frac_VO = oneI_oneD_fraction_finder(indel_count_matrix_VO,name_indel_type_unique_VO)

sequence_pam_per_gene_grna_VO = pickle.load(open('storage_VO/spacer_pam_per_site_one_hot.p', 'rb'))
sequence_pam_per_gene_grna_CHROMATIN_VO = np.concatenate((sequence_pam_per_gene_grna_VO, chrom_label_matrix_VO), axis=1)



######################################### T cell #######################################################################


###fraction of indel mutant read with insertion


#sequence_pam_per_gene_grna_new = sequence_pam_per_gene_grna[]
#lin_reg = XGBRegressor(n_estimators=241, max_depth=3)
#cross_validation_model(sequence_pam_per_gene_grna, fraction_insertions, fraction_insertions,lin_reg)


###fraction of indel mutant read with insertion

# lin_reg = XGBRegressor(n_estimators=151, max_depth=3)
# print np.shape(sequence_pam_per_gene_grna)
# print np.shape(fraction_insertions_all)
# cross_validation_model(sequence_pam_per_gene_grna, fraction_insertions_all, fraction_insertions_all,lin_reg)

###fraction of indel mutant read with insertion 6nt

# lin_reg = XGBRegressor(n_estimators=151, max_depth=3)
# cross_validation_model(sequence_pam_per_gene_grna_6nt, fraction_insertions_all, fraction_insertions_all,lin_reg)


###fraction of indel mutant read with deletion

#lin_reg = XGBRegressor(n_estimators=151, max_depth=3)
#cross_validation_model(sequence_pam_per_gene_grna, fraction_deletions_all, fraction_deletions_all,lin_reg)


###Exp insertion length

#exp_insertion_length_binary = np.zeros(len(exp_insertion_length))
#exp_insertion_length_binary[exp_insertion_length>np.median(exp_insertion_length)] = 1
#print np.median(exp_insertion_length)
#lin_reg = XGBClassifier(n_estimators=101, max_depth=3) #101,3  0.84
#cross_validation_model(sequence_pam_per_gene_grna, exp_insertion_length_binary, exp_insertion_length_binary,lin_reg)


###Exp deletion length

# exp_deletion_length_binary = np.zeros(len(exp_deletion_length))
# exp_deletion_length_binary[exp_deletion_length>np.median(exp_deletion_length)] = 1
# print np.median(exp_deletion_length)
# lin_reg = XGBClassifier(n_estimators=71, max_depth=3) #101,3  0.64
# cross_validation_model(sequence_pam_per_gene_grna, exp_deletion_length_binary, exp_deletion_length_binary,lin_reg)


###Diversity

# diversity_binary = np.zeros(len(entrop))
# diversity_binary[entrop>np.median(entrop)] = 1
# print np.median(entrop)
# lin_reg = XGBClassifier(n_estimators=71, max_depth=3)
# cross_validation_model(sequence_pam_per_gene_grna, diversity_binary, diversity_binary,lin_reg)


###edit efficiency

# lin_reg = XGBRegressor(n_estimators=161, max_depth=3) #161,3  0.22
# #cross_validation_model(sequence_pam_per_gene_grna_CHROMATIN, eff_vec, eff_vec,lin_reg)
# #cross_validation_model(sequence_pam_per_gene_grna, eff_vec, eff_vec,lin_reg)
# cross_validation_model(chrom_label_matrix, eff_vec, eff_vec,lin_reg)


######################################### VO ###########################################################################

# ## fraction of indel mutant reaad with insertion
# lin_reg = XGBRegressor(n_estimators=21, max_depth=5)
# lin_reg.fit(sequence_pam_per_gene_grna, fraction_insertions)
# insertions_r2_score_HCT = lin_reg.score(sequence_pam_per_gene_grna_VO[cell_type_VO=='HCT116'],fraction_insertions_VO[cell_type_VO=='HCT116'])
# insertions_r2_score_HEK = lin_reg.score(sequence_pam_per_gene_grna_VO[cell_type_VO=='HEK293'],fraction_insertions_VO[cell_type_VO=='HEK293'])
# insertions_r2_score_K562 = lin_reg.score(sequence_pam_per_gene_grna_VO[cell_type_VO=='K562'],fraction_insertions_VO[cell_type_VO=='K562'])
# insertions_r2_score = [insertions_r2_score_HCT,insertions_r2_score_HEK,insertions_r2_score_K562]
#
# print np.mean(insertions_r2_score)
# print np.std(insertions_r2_score)
# print insertions_r2_score
#
# pickle.dump(lin_reg, open('models_30nt/frac_indel_mutant_reads_seq.p', 'wb'))

###fraction of toatl reaads with insertion

# lin_reg = XGBRegressor(n_estimators=21, max_depth=6)
# lin_reg.fit(sequence_pam_per_gene_grna, fraction_insertions_all)
# insertions_r2_score_HCT = lin_reg.score(sequence_pam_per_gene_grna_VO[cell_type_VO=='HCT116'],fraction_insertions_all_VO[cell_type_VO=='HCT116'])
# insertions_r2_score_HEK = lin_reg.score(sequence_pam_per_gene_grna_VO[cell_type_VO=='HEK293'],fraction_insertions_all_VO[cell_type_VO=='HEK293'])
# insertions_r2_score_K562 = lin_reg.score(sequence_pam_per_gene_grna_VO[cell_type_VO=='K562'],fraction_insertions_all_VO[cell_type_VO=='K562'])
# insertions_r2_score = [insertions_r2_score_HCT,insertions_r2_score_HEK,insertions_r2_score_K562]
#
# print np.mean(insertions_r2_score)
# print np.std(insertions_r2_score)
# print insertions_r2_score

###Edit eficiency

# lin_reg = XGBRegressor(n_estimators=161, max_depth=3)
# lin_reg.fit(sequence_pam_per_gene_grna_CHROMATIN, eff_vec)
# insertions_r2_score_HCT = lin_reg.score(sequence_pam_per_gene_grna_CHROMATIN_VO[cell_type_VO=='HCT116'],eff_vec_VO[cell_type_VO=='HCT116'])
# insertions_r2_score_HEK = lin_reg.score(sequence_pam_per_gene_grna_CHROMATIN_VO[cell_type_VO=='HEK293'],eff_vec_VO[cell_type_VO=='HEK293'])
# insertions_r2_score_K562 = lin_reg.score(sequence_pam_per_gene_grna_CHROMATIN_VO[cell_type_VO=='K562'],eff_vec_VO[cell_type_VO=='K562'])
# insertions_r2_score = [insertions_r2_score_HCT,insertions_r2_score_HEK,insertions_r2_score_K562]
# print np.mean(insertions_r2_score)
# print np.std(insertions_r2_score)
# print insertions_r2_score
#
# pickle.dump(lin_reg, open('models_30nt/edit_eff_seq_chrom.p', 'wb'))


###Avg ins length

# exp_insertion_length_binary = np.zeros(len(exp_insertion_length))
# exp_insertion_length_binary[exp_insertion_length>np.median(exp_insertion_length)] = 1
#
# exp_insertion_length_binary_VO = np.zeros(len(exp_insertion_length_VO))
# exp_insertion_length_binary_VO[exp_insertion_length_VO>np.median(exp_insertion_length_VO)] = 1
#
# lin_reg = XGBClassifier(n_estimators=45, max_depth=5)
# lin_reg.fit(sequence_pam_per_gene_grna, exp_insertion_length_binary)
# insertions_r2_score_HCT = lin_reg.score(sequence_pam_per_gene_grna_VO[cell_type_VO=='HCT116'],exp_insertion_length_binary_VO[cell_type_VO=='HCT116'])
# insertions_r2_score_HEK = lin_reg.score(sequence_pam_per_gene_grna_VO[cell_type_VO=='HEK293'],exp_insertion_length_binary_VO[cell_type_VO=='HEK293'])
# insertions_r2_score_K562 = lin_reg.score(sequence_pam_per_gene_grna_VO[cell_type_VO=='K562'],exp_insertion_length_binary_VO[cell_type_VO=='K562'])
# insertions_r2_score = [insertions_r2_score_HCT,insertions_r2_score_HEK,insertions_r2_score_K562]
#
# print np.mean(insertions_r2_score)
# print np.std(insertions_r2_score)
# print insertions_r2_score



# ###Avg del length
#
# exp_deletion_length_binary = np.zeros(len(exp_deletion_length))
# exp_deletion_length_binary[exp_deletion_length>np.median(exp_deletion_length_VO)] = 1
#
# exp_deletion_length_binary_VO = np.zeros(len(exp_deletion_length_VO))
# exp_deletion_length_binary_VO[exp_deletion_length_VO>np.median(exp_deletion_length_VO)] = 1
#
# lin_reg = XGBClassifier(n_estimators=25, max_depth=5)
# lin_reg.fit(sequence_pam_per_gene_grna, exp_deletion_length_binary)
# insertions_r2_score_HCT = lin_reg.score(sequence_pam_per_gene_grna_VO[cell_type_VO=='HCT116'],exp_deletion_length_binary_VO[cell_type_VO=='HCT116'])
# insertions_r2_score_HEK = lin_reg.score(sequence_pam_per_gene_grna_VO[cell_type_VO=='HEK293'],exp_deletion_length_binary_VO[cell_type_VO=='HEK293'])
# insertions_r2_score_K562 = lin_reg.score(sequence_pam_per_gene_grna_VO[cell_type_VO=='K562'],exp_deletion_length_binary_VO[cell_type_VO=='K562'])
# insertions_r2_score = [insertions_r2_score_HCT,insertions_r2_score_HEK,insertions_r2_score_K562]
# print np.mean(insertions_r2_score)
# print np.std(insertions_r2_score)
# print insertions_r2_score



# # ###Diversity
#
# entrop_binary = np.zeros(len(entrop))
# entrop_binary[entrop>np.median(entrop)] = 1
#
# entrop_binary_VO = np.zeros(len(entrop_VO))
# entrop_binary_VO[entrop_VO>np.median(entrop_VO)] = 1
#
#
# lin_reg = XGBClassifier(n_estimators=28, max_depth=5)
# lin_reg.fit(sequence_pam_per_gene_grna, entrop_binary)
# insertions_r2_score_HCT = lin_reg.score(sequence_pam_per_gene_grna_VO[cell_type_VO=='HCT116'],entrop_binary_VO[cell_type_VO=='HCT116'])
# insertions_r2_score_HEK = lin_reg.score(sequence_pam_per_gene_grna_VO[cell_type_VO=='HEK293'],entrop_binary_VO[cell_type_VO=='HEK293'])
# insertions_r2_score_K562 = lin_reg.score(sequence_pam_per_gene_grna_VO[cell_type_VO=='K562'],entrop_binary_VO[cell_type_VO=='K562'])
# insertions_r2_score = [insertions_r2_score_HCT,insertions_r2_score_HEK,insertions_r2_score_K562]
# print np.mean(insertions_r2_score)
# print np.std(insertions_r2_score)
# print insertions_r2_score



#### AT/CG single nucleotide Tcell
# print np.shape(insertion_matrix)
# insertion_matrix_max = np.zeros((2,1521))
# insertion_matrix_max[0,:] = np.max(insertion_matrix[[0,3],:],axis=0) #A and T
# insertion_matrix_max[1,:] = np.max(insertion_matrix[[1,2],:],axis=0) #C and G
# insertion_matrix_max = np.argmax(insertion_matrix_max,axis=0)
# #insertion_matrix_max = np.argmax(insertion_matrix,axis=0)
# print "naive guess"
# print max(list(insertion_matrix_max).count(0),list(insertion_matrix_max).count(1)) / float(np.size(insertion_matrix_max))
# lin_reg = XGBClassifier(n_estimators=20, max_depth=3)
# cross_validation_model(sequence_pam_per_gene_grna, insertion_matrix_max, insertion_matrix_max,lin_reg)


#### A/T/C/G single nucleotide Tcell
# insertion_matrix_max = np.argmax(insertion_matrix,axis=0)
# print "naive guess"
# print max(list(insertion_matrix_max).count(0),list(insertion_matrix_max).count(1),list(insertion_matrix_max).count(3),list(insertion_matrix_max).count(3)) / float(np.size(insertion_matrix_max))
# lin_reg = XGBClassifier(n_estimators=31, max_depth=1)
# cross_validation_model(sequence_pam_per_gene_grna, insertion_matrix_max, insertion_matrix_max,lin_reg)


## AT/CG single nucleotide Tcell
insertion_matrix_max = np.zeros((2,1521))
insertion_matrix_max[0,:] = np.max(insertion_matrix[[0,3],:],axis=0) #A and T
insertion_matrix_max[1,:] = np.max(insertion_matrix[[1,2],:],axis=0) #C and G
insertion_matrix_max = np.argmax(insertion_matrix_max,axis=0)

insertion_matrix_max_VO = np.zeros((2,np.shape(insertion_matrix_VO)[1]))
insertion_matrix_max_VO[0,:] = np.max(insertion_matrix_VO[[0,3],:],axis=0) #A and T
insertion_matrix_max_VO[1,:] = np.max(insertion_matrix_VO[[1,2],:],axis=0) #C and G
insertion_matrix_max_VO = np.argmax(insertion_matrix_max_VO,axis=0)

print "naive guess"
print max(list(insertion_matrix_max_VO).count(0),list(insertion_matrix_max_VO).count(1)) / float(np.size(insertion_matrix_max_VO))


lin_reg = XGBClassifier(n_estimators=3, max_depth=1)
lin_reg.fit(sequence_pam_per_gene_grna, insertion_matrix_max)

insertions_score_HCT = lin_reg.score(sequence_pam_per_gene_grna_VO[cell_type_VO=='HCT116'],insertion_matrix_max_VO[cell_type_VO=='HCT116'])
insertions_score_HEK = lin_reg.score(sequence_pam_per_gene_grna_VO[cell_type_VO=='HEK293'],insertion_matrix_max_VO[cell_type_VO=='HEK293'])
insertions_score_K562 = lin_reg.score(sequence_pam_per_gene_grna_VO[cell_type_VO=='K562'],insertion_matrix_max_VO[cell_type_VO=='K562'])
insertions_score = [insertions_score_HCT,insertions_score_HEK,insertions_score_K562]

lin_reg_pred = lin_reg.predict(sequence_pam_per_gene_grna_VO[cell_type_VO=='HCT116'])
f1_HCT = f1_score(insertion_matrix_max_VO[cell_type_VO=='HCT116'], lin_reg_pred)

lin_reg_pred = lin_reg.predict(sequence_pam_per_gene_grna_VO[cell_type_VO=='HEK293'])
f1_HEK = f1_score(insertion_matrix_max_VO[cell_type_VO=='HEK293'], lin_reg_pred)

lin_reg_pred = lin_reg.predict(sequence_pam_per_gene_grna_VO[cell_type_VO=='K562'])
f1_K562 = f1_score(insertion_matrix_max_VO[cell_type_VO=='K562'], lin_reg_pred)

print np.mean(insertions_score)
print np.std(insertions_score)

print np.mean([f1_HCT,f1_HEK,f1_K562])



### A/T/C/G single nucleotide Tcell
# insertion_matrix_max = np.argmax(insertion_matrix,axis=0)
# insertion_matrix_max_VO = np.argmax(insertion_matrix_VO,axis=0)
#
# print "naive guess"
# print max(list(insertion_matrix_max_VO).count(0),list(insertion_matrix_max_VO).count(1),list(insertion_matrix_max_VO).count(2),list(insertion_matrix_max_VO).count(3)) / float(np.size(insertion_matrix_max_VO))
#
# for i in range(1,30):
#     lin_reg = XGBClassifier(n_estimators=3, max_depth=1)
#     lin_reg.fit(sequence_pam_per_gene_grna, insertion_matrix_max)
#
#     insertions_score_HCT = lin_reg.score(sequence_pam_per_gene_grna_VO[cell_type_VO=='HCT116'],insertion_matrix_max_VO[cell_type_VO=='HCT116'])
#     insertions_score_HEK = lin_reg.score(sequence_pam_per_gene_grna_VO[cell_type_VO=='HEK293'],insertion_matrix_max_VO[cell_type_VO=='HEK293'])
#     insertions_score_K562 = lin_reg.score(sequence_pam_per_gene_grna_VO[cell_type_VO=='K562'],insertion_matrix_max_VO[cell_type_VO=='K562'])
#     insertions_score = [insertions_score_HCT,insertions_score_HEK,insertions_score_K562]
#     print i
#     print np.mean(insertions_score)
#     #print np.std(insertions_score)
#     print '-'



#
# ######################################### Gene Ranking ######  T CElls ###############################################
#oneI_frac = oneD_frac_total


# gene_list = []
# for name in name_genes_grna_unique:
#     gene_list.append(name.split('-')[0])
#
# gene_list_unique = np.unique(gene_list)
#
# final_kendal_tau = []
# final_kendal_tau_random = []
# final_gene_correct_counter = []
# final_gene_correct_counter_iligable = []
# final_gene_correct_counter_rand = []
# final_top_gene_correct_counter = []
#
# for repeat in range(100):
#
#     print 'repeat ', repeat
#
#     shuffle_ind = range(len(gene_list_unique))
#     np.random.shuffle(shuffle_ind)
#
#     train_gene_ind = shuffle_ind[0:400]  ## 400 train genes
#     test_gene_ind = shuffle_ind[400:]    ## 149 test genes
#
#
#     train_gene = gene_list_unique[train_gene_ind]
#     test_gene = gene_list_unique[test_gene_ind]
#
#     train_cut_ind = []
#     test_cut_ind = []
#
#     for ind,name in enumerate(name_genes_grna_unique):
#         if name.split('-')[0] in train_gene:
#             train_cut_ind.append(ind)
#
#         if name.split('-')[0] in test_gene:
#             test_cut_ind.append(ind)
#
#
#     ## now train on training
#     lin_reg = XGBRegressor(n_estimators=91, max_depth=5)
#     lin_reg.fit(sequence_pam_per_gene_grna[train_cut_ind,:], oneI_frac[train_cut_ind])
#     vec_p = lin_reg.predict(sequence_pam_per_gene_grna[test_cut_ind,:])
#     vec_gt = oneI_frac[test_cut_ind]
#     #[u,p] = kendalltau(vec_p,vec_gt)
#     #final_kendal_tau.append(u)
#
#     local_kendal_tau = []
#     gene_correct_counter = 0
#     gene_correct_counter_elig = 0
#     gene_correct_counter_rand = 0
#     top_gene_correct_counter = 0
#     for gene in test_gene:
#         vec1 = []
#         vec2 = []
#         for aaa, name in enumerate(name_genes_grna_unique):
#             if gene == name.split('-')[0]:
#                 bbb = test_cut_ind.index(aaa)
#                 vec1.append(vec_p[bbb])
#                 vec2.append(vec_gt[bbb])
#
#         vec3 = np.copy(vec2)
#         np.random.shuffle(vec3)
#         [uu,pp] = kendalltau(vec1,vec2)
#         [uurand,pprand] = kendalltau(vec1,vec3)
#         if uurand > 0.99 and len(vec1)>1:
#             gene_correct_counter_rand += 1
#         if uu > 0.99 and len(vec1)>1:
#             gene_correct_counter += 1
#         if len(vec1) > 1 and np.argmax(vec1) ==  np.argmax(vec2):
#             top_gene_correct_counter += 1
#         if len(vec1) > 1:
#             local_kendal_tau.append(uu)
#             gene_correct_counter_elig += 1
#
#     final_top_gene_correct_counter.append(top_gene_correct_counter)
#     final_gene_correct_counter.append(gene_correct_counter)
#     final_gene_correct_counter_rand.append(gene_correct_counter_rand)
#     final_gene_correct_counter_iligable.append(gene_correct_counter_elig)
#     final_kendal_tau.append(np.nanmean(local_kendal_tau))
#
#
#     np.random.shuffle(vec_p)
#     [u, p] = kendalltau(vec_p, vec_gt)
#     final_kendal_tau_random.append(u)
#
# print "K-t"
# print np.mean(final_kendal_tau)
# print np.std(final_kendal_tau)
# print
# print "# of genes with complete ordering correct"
# print np.mean(final_gene_correct_counter)
# print np.std(final_gene_correct_counter)
# print "# of genes with top guy correct"
# print np.mean(final_top_gene_correct_counter)
# print "out of this number of genes"
# print np.mean(final_gene_correct_counter_iligable)
# print
# print "random shuffle"
# print np.mean(final_kendal_tau_random)
# print np.std(final_kendal_tau_random)
# print "# of genes"
# print np.mean(final_gene_correct_counter_rand)
# print np.std(final_gene_correct_counter_rand)
#
#
# tstat, pvalue = ttest_ind_from_stats(np.mean(final_gene_correct_counter_rand),np.std(final_gene_correct_counter_rand), np.shape(final_gene_correct_counter_rand)[0], np.mean(final_top_gene_correct_counter), np.std(final_top_gene_correct_counter), np.shape(final_top_gene_correct_counter)[0])
# print pvalue
# tstat, pvalue = ttest_ind_from_stats(np.mean(final_gene_correct_counter_rand),np.std(final_gene_correct_counter_rand), np.shape(final_gene_correct_counter_rand)[0], np.mean(final_gene_correct_counter), np.std(final_gene_correct_counter), np.shape(final_gene_correct_counter)[0])
# print pvalue

# ######################################### Gene Ranking #########other cells ##########################################
# oneI_frac = oneD_frac
# oneI_frac_VO = oneD_frac_VO
#
# lin_reg = XGBRegressor(n_estimators=91, max_depth=5)
# lin_reg.fit(sequence_pam_per_gene_grna, oneI_frac)
# vec_p = lin_reg.predict(sequence_pam_per_gene_grna_VO)
# vec_gth = oneI_frac_VO
#
# #cell_type_VO=='HCT116'
# #cell_type_VO=='HEK293'
# #cell_type_VO=='K562'
#
# name_list = pickle.load(open('storage_VO/all_file_name.p', 'rb'))
# gene_list = pickle.load(open('storage_VO/gene_names.pkl', 'rb'))
# gene_list_UNIQUE = np.unique(gene_list)
#
#
# final_kenal_tau_across_cell = []
# final_total_gene_counter = []
# final_good_gene_counter = []
#
# for cell in ['HCT116','HEK293','K562']:
#
#     final_kenal_tau = []
#     total_gene_counter = 0
#     good_gene_counter = 0
#     for gene1 in gene_list_UNIQUE:
#         vec1 = []
#         vec2 = []
#         for ind,gene2 in enumerate(gene_list):
#             if gene2 == gene1 and 'HCT116' in name_list[ind]:
#                 vec1.append(vec_p[ind])
#                 vec2.append(vec_gth[ind])
#
#         [uu, pp] = kendalltau(vec1, vec2)
#         if len(vec1)>2:
#             final_kenal_tau.append(uu)
#             total_gene_counter += 1
#         if len(vec1)>2 and uu>0.99:
#             good_gene_counter+=1
#
#     final_kenal_tau_across_cell.append(np.mean(final_kenal_tau))
#     final_total_gene_counter.append(total_gene_counter)
#     final_good_gene_counter.append(good_gene_counter)
#
#
#
#     print cell
#     print np.mean(final_kenal_tau)
#     print np.std(final_kenal_tau)
#     print good_gene_counter
#     print '--'
#
#
# print total_gene_counter




# ##########################
# vec = fraction_insertions / fraction_deletions
# print np.std(vec)
# print np.percentile(vec, 25)
# print np.percentile(vec, 75)

# insertion_matrix_max = np.zeros((2,1521))
# insertion_matrix_max[0,:] = np.max(insertion_matrix[[0,3],:],axis=0) #A and T
# insertion_matrix_max[1,:] = np.max(insertion_matrix[[1,2],:],axis=0) #C and G
# insertion_matrix_max = np.argmax(insertion_matrix_max,axis=0)
#
# print np.sum(insertion_matrix_max)
# print len(insertion_matrix_max) - np.sum(insertion_matrix_max)


# gene_list = []
# for name in name_genes_grna_unique:
#     gene_list.append(name.split('-')[0])
#
# counter=collections.Counter(gene_list)
# print counter.values().count(1)







