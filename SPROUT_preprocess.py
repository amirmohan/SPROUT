import numpy as np
import glob
import pickle
import pandas as pd
import csv
import collections, numpy
from os import listdir
from scipy.stats import entropy
import matplotlib.pyplot as plt


def preprocess_indel_files(data_folder):
  count_data_folder = data_folder + "counts/"
  #count_data_folder = data_folder + "small_counts/"
  # First process the files to glean the names of the genes and the different indels
  name_genes = []
  name_genes_grna = []
  name_indel_type = []
  for each_file in glob.glob(count_data_folder + "counts-*.txt"):
    with open(each_file) as f:
      i = 0
      process_file = False
      add_file = False
      for line in f:
        line = line.replace('\n', '')
        line = line.replace('_', '-')
        if i == 0:
          line = line.replace('"', '')
          l = line.split(',')
          process_file = True
          curr_gene_name = each_file[len(count_data_folder) + 7:-4].split('-')[0]
          curr_gene_grna_name = []
          for patient in range(np.size(l)):
            curr_gene_grna_name.append("%s-%s-%s" %(curr_gene_name,l[patient].split('-')[1],l[patient].split('-')[2] ))

        if i > 0 and process_file:
          l_indel = line.split('"')[1].split(',')
          l = line.split('"')[2].split(',')[1:]
          indel_type = ''
          # Some positions are of the form: "-23:-21D,-19:-15D", which get split by the process when we call split()
          # We try to account for such things in this space
          for j in range(np.size(l_indel)):
            indel_type += l_indel[j]
          # We only consider I or D
          if line.find('I') != -1 or line.find('D') != -1:
            name_indel_type.append(indel_type)
            if not add_file:
              name_genes.append(curr_gene_name)
              for patient in range(np.size(curr_gene_grna_name)):
                name_genes_grna.append(curr_gene_grna_name[patient])
              add_file = True
        i += 1


  # Take the unique values, in sorted order
  name_genes_unique = list(set(name_genes))
  name_genes_grna_unique = list(set(name_genes_grna))
  name_indel_type_unique = list(set(name_indel_type))
  name_genes_unique.sort()
  name_genes_grna_unique.sort()
  name_indel_type_unique.sort()

  ##
  # Then process the files again to get the actual counts from only the desired files, and from the desired rows and columns
  indel_count_matrix = np.zeros((len(name_indel_type_unique), len(name_genes_grna_unique)))
  length_indel_insertion = np.zeros(len(name_indel_type_unique), dtype = int)
  length_indel_deletion = np.zeros(len(name_indel_type_unique), dtype=int)
  no_variant_vec = np.zeros(len(name_genes_grna_unique))
  other_vec = np.zeros(len(name_genes_grna_unique))
  snv_vec = np.zeros(len(name_genes_grna_unique))

  for each_file in glob.glob(count_data_folder + "counts-*.txt"):
    print each_file
    with open(each_file) as f:
      i = 0
      process_file = False
      for line in f:
        line = line.replace('\n', '')
        line = line.replace('_', '-')
        if i == 0:
          line = line.replace('"', '')
          l = line.split(',')
          curr_gene_name = each_file[len(count_data_folder) + 7:-4].split('-')[0]
          col_index = []
          if "%s-%s-%s" %(curr_gene_name,l[0].split('-')[1],l[0].split('-')[2] )  in name_genes_grna_unique:
            process_file = True
            for patient in range(np.size(l)):
              col_index.append(name_genes_grna_unique.index("%s-%s-%s" %(curr_gene_name,l[patient].split('-')[1],l[patient].split('-')[2])))
        if i > 0 and process_file:
          l_indel = line.split('"')[1].split(',')
          l = line.split('"')[2].split(',')[1:]
          indel_type = ''
          len_indel_insertion = 0
          len_indel_deletion = 0
          # Some positions are of the form: "-23:-21D,-19:-15D", which get split by the process when we call split()
          # We try to account for such things in this space
          for j in range(0, np.size(l_indel)):
            indel_type += l_indel[j]
            if l_indel[j].find('I') != -1:
              begn_size = l_indel[j].replace("I", "")
              begn_size = begn_size.split(':')
              len_indel_insertion += int(begn_size[1])
            if l_indel[j].find('D') != -1:
              begn_size = l_indel[j].replace("D", "")
              begn_size = begn_size.split(':')
              len_indel_deletion += int(begn_size[1])
          if line.find('I') != -1 or line.find('D') != -1:
            row_index = name_indel_type_unique.index(indel_type)
            length_indel_insertion[row_index] = len_indel_insertion
            length_indel_deletion[row_index] = len_indel_deletion
            for j in range(np.size(l)):
              if l[j] != 'NA':
                indel_count_matrix[row_index,col_index[j]] = float(l[j])
          if 'variant' in line:
            for j in range(np.size(l)):
              if l[j] != 'NA':
                no_variant_vec[col_index[j]] += float(l[j])
          if 'Other' in line:
            for j in range(np.size(l)):
              if l[j] != 'NA':
                other_vec[col_index[j]] += float(l[j])
          if 'SNV' in line:
            for j in range(np.size(l)):
              if l[j] != 'NA':
                snv_vec[col_index[j]] += float(l[j])

        i += 1

  # finding the index for the indels with frequency of mutatnt reads < 0.01
  rare_indel_index = []
  indel_frac_mutant_read_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))
  for row_index in range(np.shape(indel_frac_mutant_read_matrix)[0]):
    if max(indel_frac_mutant_read_matrix[row_index]) < 0.01:
      rare_indel_index.append(row_index)



  ######
  ###### here we filter out all indels with mutant read frequency less than 0.01 (optional)
  ######
  name_indel_type_unique = np.delete(name_indel_type_unique, rare_indel_index).tolist()
  indel_count_matrix = np.delete(indel_count_matrix, rare_indel_index, 0)
  length_indel_insertion = np.delete(length_indel_insertion, rare_indel_index, 0)
  length_indel_deletion = np.delete(length_indel_deletion, rare_indel_index, 0)
  ######


  #####
  ##### here we filter out all outcomes with very small read counts
  #####
  low_read_index = []
  # you can hand pick the outcomes that dont work
  low_read_patients = ['just_in_case']
  for crispr in range(np.shape(indel_count_matrix)[1]):
    if sum(indel_count_matrix[:,crispr]) < 1000 or (name_genes_grna_unique[crispr] in low_read_patients):
      low_read_index.append(crispr)

  indel_count_matrix = np.delete(indel_count_matrix, low_read_index, 1)
  name_genes_grna_unique = list(np.delete(name_genes_grna_unique, low_read_index, 0))

  no_variant_vec = np.delete(no_variant_vec, low_read_index)
  other_vec = np.delete(other_vec, low_read_index)
  snv_vec = np.delete(snv_vec, low_read_index)

  #####


  return name_genes_unique, name_genes_grna_unique, name_indel_type_unique, indel_count_matrix, no_variant_vec, other_vec, snv_vec,  length_indel_insertion, length_indel_deletion


def insertion_matrix_finder():

  insertion_file_names = []
  insertion_matrix = np.zeros((4,96*3+95*3+93+96+96))
  folder = '/Users/amirali/Projects/ins-othercells/'
  file_counter = 0
  for cell in listdir(folder):
    if cell != ".DS_Store":
      for file in listdir(folder+cell+'/'):

        insertion_file_names.append(file[11:-4].split('-')[0]+'-'+file[11:-4].split('-')[1])

        lines = open(folder+cell+'/'+ file, 'r')
        for line_counter, line in enumerate(lines):
          if line_counter>0:
              line = line[:-1].replace('"','')
              line_splited = line.split(',')
              counter = float(line_splited[-1])
              seq = line_splited[1]
              cigar = line_splited[2:-4]
              if len(cigar)==1 and seq in ['A','T','C','G'] and isinstance(counter, numbers.Number):
                insertion_matrix[one_hot_index(seq),file_counter ] += counter

        file_counter += 1

  return insertion_file_names,insertion_matrix






def rare_insertion():
    insertion_file_names = []
    global_insertion_list = []
    folder = '/Users/amirali/Projects/ins-othercells/'
    file_counter = 0
    for cell in listdir(folder):
        if cell != ".DS_Store":
            for file in listdir(folder + cell + '/'):
                local_insertion_list = []
                insertion_file_names.append(file[11:-4].split('-')[0] + '-' + file[11:-4].split('-')[1])


                lines = open(folder + cell + '/' + file, 'r')
                for line_counter, line in enumerate(lines):
                    if line_counter > 0:
                        line = line[:-1].replace('"', '')
                        line_splited = line.split(',')
                        counter = float(line_splited[-1])
                        seq = line_splited[1]
                        cigar = line_splited[2:-4]
                        if len(cigar) == 1:
                            size = int(cigar[0].split(":")[1][0:-1])
                            if size > 10 and isinstance(counter, numbers.Number):
                                # print seq
                                # print line
                                # print counter
                                #local_insertion_list[samples.index(sample)].append(seq)
                                local_insertion_list.append(seq)


                file_counter += 1
                global_insertion_list.append(local_insertion_list)


    return global_insertion_list



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

def load_gene_sequence(spacer_list):

  sequence_pam_per_gene_grna = np.zeros((len(spacer_list), 23, 4), dtype = bool)

  for counter,spacer in enumerate(spacer_list):
    for i in range(23):
      sequence_pam_per_gene_grna[counter, i, one_hot_index(spacer[i])] = 1

  return np.reshape(sequence_pam_per_gene_grna, (len(sequence_pam_per_gene_grna), -1))


def longest_substring_passing_cutsite(strng,character):
    len_substring=0
    longest=0
    label_set = []
    midpoint = len(strng)/2
    for i in range(len(strng)):
        if i > 1:
            if strng[i] != strng[i-1] or strng[i] != character:
                len_substring = 0
                label_set = []
        if strng[i] == character:
            label_set.append(i)
            len_substring += 1
        if len_substring > longest and (midpoint-1 in label_set or 3 in label_set):
            longest = len_substring

    return longest


def homology_matrix_finder(spacer_list):
    homology_matrix = np.zeros((4, len(spacer_list)))
    for counter,spacer in enumerate(spacer_list):
        nuc_count = 0
        for nuc in ['A', 'C', 'G', 'T']:
            homology_matrix[nuc_count,counter] = int(longest_substring_passing_cutsite(spacer[16-3:16+3], nuc))
            nuc_count+=1
    return homology_matrix

def flatness_finder(indel_count_matrix):
    num_indels, num_sites = np.shape(indel_count_matrix)
    indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))
    max_grad = []
    entrop = []
    for col in range(num_sites):
        vec = np.copy(indel_fraction_mutant_matrix[:, col])
        vec = np.sort(vec)[::-1]
        max_grad.append(max(abs(np.gradient(vec))))
        entrop.append(entropy(vec))

    return np.asarray(max_grad),np.asarray(entrop)

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

  help_vec = np.sum(insertion_only_fraction_matrix, axis=0)
  help_vec[help_vec==0] = 'nan'

  insertion_only_fraction_matrix = insertion_only_fraction_matrix / np.reshape(help_vec, (1, -1))
  deletion_only_fraction_matrix = deletion_only_fraction_matrix / np.reshape(np.sum(deletion_only_fraction_matrix, axis=0), (1, -1))


  for site_index in range(site_num):
    exp_insertion_length[site_index] = np.inner(length_indel_insertion,insertion_only_fraction_matrix[:,site_index])
    exp_deletion_length[site_index] = np.inner(length_indel_deletion, deletion_only_fraction_matrix[:, site_index])

  # some sites do not get any insertions. this cuase nan. we make those entries zero.
  for i in range(np.size(exp_insertion_length)):
    if np.isnan(exp_insertion_length[i]):
      exp_insertion_length[i] = 0

  return exp_insertion_length,exp_deletion_length




###### Do this first
#name_genes_ALL, name_genes_grna_ALL, name_indel_type_ALL, indel_count_matrix_ALL, no_variant_vec_ALL, other_vec_ALL, snv_vec_ALL,  length_indel_insertion_ALL, length_indel_deletion_ALL = preprocess_indel_files('/Users/amirali/Projects/30nt_again/')

# pickle.dump(name_genes_ALL, open('storage_30nt_new/name_genes_ALL.p', 'wb'))
# pickle.dump(name_genes_grna_ALL, open('storage_30nt_new/name_genes_grna_ALL.p', 'wb'))
# pickle.dump(name_indel_type_ALL, open('storage_30nt_new/name_indel_type_ALL.p', 'wb'))
# pickle.dump(indel_count_matrix_ALL, open('storage_30nt_new/indel_count_ALL.p', 'wb'))
# pickle.dump(no_variant_vec_ALL, open('storage_30nt_new/no_variant_vec_ALL.p', 'wb'))
# pickle.dump(other_vec_ALL, open('storage_30nt_new/other_vec_ALL.p', 'wb'))
# pickle.dump(snv_vec_ALL, open('storage_30nt_new/snv_vec_ALL.p', 'wb'))
# pickle.dump(length_indel_insertion_ALL, open('storage_30nt_new/length_indel_insertion_ALL.p', 'wb'))
# pickle.dump(length_indel_deletion_ALL, open('storage_30nt_new/length_indel_deletion_ALL.p', 'wb'))



###### Do this Next
# name_genes_grna_unique = pickle.load(open('storage_30nt_new/name_genes_grna_ALL.p', 'rb'))
#
# ### get the spacer and donor lists out
# location_dict = {}
# donor_dict = {}
# all_name = []
#
# with open('sequence_pam_gene_grna_big_file_donor_genomic_context_with_donor_30nt_new.csv', 'rb') as csvfile:
#
#   spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#   row_counter = 0
#   for row in spamreader:
#     location_dict[ row[0].split(',')[1]+'-'+row[0].split(',')[0] ] = row[0].split(',')[2] + row[0].split(',')[3]
#     all_name.append(row[0].split(',')[1]+'-'+row[0].split(',')[0])
#     donor_dict[ row[0].split(',')[1]+'-'+row[0].split(',')[0] ] = row[0].split(',')[5]
#
#
# spacer_list = []
# donor_list = []
# counter = 0
# for ind,name in enumerate(name_genes_grna_unique):
#   if name in all_name:
#     spacer_list.append(location_dict[name])
#     donor_list.append(donor_dict[name])
#   else:
#     counter += 1
#
# if counter != 0:
#   print "Dude, sth is wrong! Check the list."
#
# print np.shape(spacer_list)
# print len(set(spacer_list))
#
# pickle.dump(spacer_list, open('storage_30nt_new/spacer_list_ALL.p', 'wb'))
# pickle.dump(donor_list, open('storage_30nt_new/donor_list_ALL.p', 'wb'))

# ###### Do this Next
all_file_name = pickle.load(open('storage_30nt_new/name_genes_grna_ALL.p', 'rb'))
indel_count_matrix = pickle.load(open('storage_30nt_new/indel_count_ALL.p', 'rb'))
length_indel_insertion = pickle.load(open('storage_30nt_new/length_indel_insertion_ALL.p', 'rb'))
length_indel_deletion = pickle.load(open('storage_30nt_new/length_indel_deletion_ALL.p', 'rb'))

spacer_list = pickle.load(open('storage_30nt_new/spacer_list_ALL.p', 'rb'))
donor_list = pickle.load(open('storage_30nt_new/donor_list_ALL.p', 'rb'))

no_variant_vector = pickle.load(open('storage_30nt_new/no_variant_vec_ALL.p', 'rb'))
other_vector = pickle.load(open('storage_30nt_new/other_vec_ALL.p', 'rb'))
snv_vector = pickle.load(open('storage_30nt_new/snv_vec_ALL.p', 'rb'))


# output is ready
# find eff_vec
total_vec = np.sum(indel_count_matrix,axis=0) + no_variant_vector + other_vector + snv_vector
total_vec_no_others = np.sum(indel_count_matrix,axis=0) + no_variant_vector


eff_vec = np.sum(indel_count_matrix,axis=0) / total_vec

# idel prop matrix
indel_prop_matrix = np.zeros(np.shape(indel_count_matrix))
indel_prop_matrix = indel_count_matrix / np.reshape(total_vec, (1, -1))

# idel prop matrix
indel_prop_matrix_no_others = np.zeros(np.shape(indel_count_matrix))
indel_prop_matrix_no_others = indel_count_matrix / np.reshape(total_vec_no_others, (1, -1))

# fraction of Indels
fraction_insertions, fraction_deletions = fraction_of_deletion_insertion(indel_count_matrix,length_indel_insertion,length_indel_deletion)

# fraction of indels (total)
fraction_insertions_all, fraction_deletions_all = fraction_of_deletion_insertion_porportion(indel_prop_matrix,length_indel_insertion,length_indel_deletion)

# fraction of indels (total)
fraction_insertions_all_no_others, fraction_deletions_all_no_other = fraction_of_deletion_insertion_porportion(indel_prop_matrix_no_others,length_indel_insertion,length_indel_deletion)


# flatness
max_grad,entrop = flatness_finder(indel_count_matrix)

# expected number of indels
exp_insertion_length, exp_deletion_length = expected_deletion_insertion_length(indel_count_matrix,length_indel_insertion,length_indel_deletion)

# pickle.dump(eff_vec, open('storage_30nt_new/eff_vec_ALL.p', 'wb'))
# pickle.dump(fraction_insertions, open('storage_30nt_new/fraction_insertions_ALL.p', 'wb'))
# pickle.dump(fraction_deletions, open('storage_30nt_new/fraction_deletions_ALL.p', 'wb'))
# pickle.dump(fraction_insertions_all, open('storage_30nt_new/fraction_insertions_over_total_ALL.p', 'wb'))
# pickle.dump(fraction_deletions_all, open('storage_30nt_new/fraction_deletions_over_total_ALL.p', 'wb'))
# pickle.dump(exp_insertion_length, open('storage_30nt_new/exp_insertion_length_ALL.p', 'wb'))
# pickle.dump(exp_deletion_length, open('storage_30nt_new/exp_deletion_length_ALL.p', 'wb'))
# pickle.dump(max_grad, open('storage_30nt_new/max_grad_ALL.p', 'wb'))
# pickle.dump(entrop, open('storage_30nt_new/entrop_ALL.p', 'wb'))

pickle.dump(fraction_insertions_all_no_others, open('storage_30nt_new/fraction_insertions_over_total_no_others_ALL.p', 'wb'))

# ######### here we average the outcomes to get the UNIQUE cut site statistic
#
# all_file_name = pickle.load(open('storage_30nt_new/name_genes_grna_ALL.p', 'rb'))
# indel_count_matrix = pickle.load(open('storage_30nt_new/indel_count_ALL.p', 'rb'))
# length_indel_insertion = pickle.load(open('storage_30nt_new/length_indel_insertion_ALL.p', 'rb'))
# length_indel_deletion = pickle.load(open('storage_30nt_new/length_indel_deletion_ALL.p', 'rb'))
#
# spacer_list = pickle.load(open('storage_30nt_new/spacer_list_ALL.p', 'rb'))
# donor_list = pickle.load(open('storage_30nt_new/donor_list_ALL.p', 'rb'))
#
# no_variant_vector = pickle.load(open('storage_30nt_new/no_variant_vec_ALL.p', 'rb'))
# other_vector = pickle.load(open('storage_30nt_new/other_vec_ALL.p', 'rb'))
# snv_vector = pickle.load(open('storage_30nt_new/snv_vec_ALL.p', 'rb'))
#
# eff_vec = pickle.load(open('storage_30nt_new/eff_vec_ALL.p', 'rb'))
# fraction_insertions = pickle.load(open('storage_30nt_new/fraction_insertions_ALL.p', 'rb'))
# fraction_deletions = pickle.load(open('storage_30nt_new/fraction_deletions_ALL.p', 'rb'))
# fraction_insertions_all = pickle.load(open('storage_30nt_new/fraction_insertions_over_total_ALL.p', 'rb'))
# fraction_deletions_all = pickle.load(open('storage_30nt_new/fraction_deletions_over_total_ALL.p', 'rb'))
# exp_insertion_length = pickle.load(open('storage_30nt_new/exp_insertion_length_ALL.p', 'rb'))
# exp_deletion_length = pickle.load(open('storage_30nt_new/exp_deletion_length_ALL.p', 'rb'))
# entrop = pickle.load(open('storage_30nt_new/entrop_ALL.p', 'rb'))
# fraction_insertions_all_no_other =  pickle.load(open('storage_30nt_new/fraction_insertions_over_total_no_others_ALL.p', 'rb'))
#
# spacer_list_unique = np.unique(spacer_list)
# number_of_unique_sites = len(spacer_list_unique)
# number_of_unique_indels = len(length_indel_insertion)
#
# eff_vec_UNIQUE = np.zeros(number_of_unique_sites)
# fraction_insertions_UNIQUE = np.zeros(number_of_unique_sites)
# fraction_deletions_UNIQUE= np.zeros(number_of_unique_sites)
# fraction_deletions_all_UNIQUE = np.zeros(number_of_unique_sites)
# fraction_insertions_all_UNIQUE = np.zeros(number_of_unique_sites)
# exp_insertion_length_UNIQUE = np.zeros(number_of_unique_sites)
# exp_deletion_length_UNIQUE = np.zeros(number_of_unique_sites)
# entrop_UNIQUE = np.zeros(number_of_unique_sites)
# fraction_insertions_all_no_other_UNIQUE = np.zeros(number_of_unique_sites)
#
#
# indel_count_matrix_UNIQUE = np.zeros((number_of_unique_indels,number_of_unique_sites))
# no_variant_vector_UNIQUE = np.zeros(number_of_unique_sites)
# other_vector_UNIQUE = np.zeros(number_of_unique_sites)
# snv_vector_UNIQUE = np.zeros(number_of_unique_sites)
#
# sample_file_name_UNIQUE = []
# for ind1,spacer1 in enumerate(spacer_list_unique):
#   for ind2, spacer2 in enumerate(spacer_list):
#     if spacer1 == spacer2:
#       sample_file_name_UNIQUE.append(all_file_name[ind2])
#       break
#
#
#
# for ind1,spacer1 in enumerate(spacer_list_unique):
#
#   temp_eff_vec = []
#   temp_fraction_insertions = []
#   temp_fraction_deletions = []
#   temp_fraction_insertions_all = []
#   temp_fraction_deletions_all = []
#   temp_exp_insertion_length = []
#   temp_exp_deletion_length = []
#   temp_entrop = []
#   temp_fraction_insertions_all_no_other = []
#
#   temp_no_variant_vector = []
#   temp_other_vector = []
#   temp_snv_vector = []
#
#   temp_outcome = np.zeros(number_of_unique_indels)
#   bio_repeat_counter = 0.
#
#   for ind2,spacer2 in enumerate(spacer_list):
#
#     if spacer1==spacer2:
#       temp_eff_vec.append(eff_vec[ind2])
#       temp_fraction_insertions.append(fraction_insertions[ind2])
#       temp_fraction_deletions.append(fraction_deletions[ind2])
#       temp_fraction_insertions_all.append(fraction_insertions_all[ind2])
#       temp_fraction_deletions_all.append(fraction_deletions_all[ind2])
#       temp_exp_insertion_length.append(exp_insertion_length[ind2])
#       temp_exp_deletion_length.append(exp_deletion_length[ind2])
#       temp_entrop.append(entrop[ind2])
#       temp_fraction_insertions_all_no_other.append(fraction_insertions_all_no_other[ind2])
#
#       temp_no_variant_vector.append(no_variant_vector[ind2])
#       temp_other_vector.append(other_vector[ind2])
#       temp_snv_vector.append(snv_vector[ind2])
#
#       temp_outcome += indel_count_matrix[:,ind2]
#       bio_repeat_counter += 1.
#
#   eff_vec_UNIQUE[ind1] = np.mean(temp_eff_vec)
#   fraction_insertions_UNIQUE[ind1] = np.mean(temp_fraction_insertions)
#   fraction_deletions_UNIQUE[ind1] = np.mean(temp_fraction_deletions)
#   fraction_insertions_all_UNIQUE[ind1] = np.mean(temp_fraction_insertions_all)
#   fraction_deletions_all_UNIQUE[ind1] = np.mean(temp_fraction_deletions_all)
#   exp_insertion_length_UNIQUE[ind1] = np.mean(temp_exp_insertion_length)
#   exp_deletion_length_UNIQUE[ind1] = np.mean(temp_exp_deletion_length)
#   entrop_UNIQUE[ind1] = np.mean(temp_entrop)
#   fraction_insertions_all_no_other_UNIQUE[ind1] = np.mean(temp_fraction_insertions_all_no_other)
#
#
#   no_variant_vector_UNIQUE[ind1] = np.mean(temp_no_variant_vector)
#   other_vector_UNIQUE[ind1] = np.mean(temp_other_vector)
#   snv_vector_UNIQUE[ind1] = np.mean(temp_snv_vector)
#
#   indel_count_matrix_UNIQUE[:,ind1] = temp_outcome / bio_repeat_counter
#
# # pickle.dump(indel_count_matrix_UNIQUE, open('storage_30nt_new/indel_count_matrix_UNIQUE.p', 'wb'))
# # pickle.dump(no_variant_vector_UNIQUE, open('storage_30nt_new/no_variant_vector_UNIQUE.p', 'wb'))
# # pickle.dump(other_vector_UNIQUE, open('storage_30nt_new/other_vector_UNIQUE.p', 'wb'))
# # pickle.dump(snv_vector_UNIQUE, open('storage_30nt_new/snv_vector_UNIQUE.p', 'wb'))
# #
# # pickle.dump(spacer_list_unique, open('storage_30nt_new/spacer_list_UNIQUE.p', 'wb'))
# #
# # pickle.dump(eff_vec_UNIQUE, open('storage_30nt_new/eff_vec_UNIQUE.p', 'wb'))
# # pickle.dump(fraction_insertions_UNIQUE, open('storage_30nt_new/fraction_insertions_UNIQUE.p', 'wb'))
# # pickle.dump(fraction_deletions_UNIQUE, open('storage_30nt_new/fraction_deletions_UNIQUE.p', 'wb'))
# # pickle.dump(fraction_insertions_all_UNIQUE, open('storage_30nt_new/fraction_insertions_all_UNIQUE.p', 'wb'))
# # pickle.dump(fraction_deletions_all_UNIQUE, open('storage_30nt_new/fraction_deletions_all_UNIQUE.p', 'wb'))
# # pickle.dump(exp_insertion_length_UNIQUE, open('storage_30nt_new/exp_insertion_length_UNIQUE.p', 'wb'))
# # pickle.dump(exp_deletion_length_UNIQUE, open('storage_30nt_new/exp_deletion_length_UNIQUE.p', 'wb'))
# # pickle.dump(entrop_UNIQUE, open('storage_30nt_new/entrop_UNIQUE.p', 'wb'))
# pickle.dump(entrop_UNIQUE, open('storage_30nt_new/fraction_insertions_all_no_other_UNIQUE.p', 'wb'))
#
# # pickle.dump(sample_file_name_UNIQUE, open('storage_30nt_new/sample_file_name_UNIQUE.p', 'wb'))


###########
# spacer_list_UNIQUE = pickle.load(open('storage_30nt_new/spacer_list_UNIQUE.p', 'rb'))
# sample_file_name_UNIQUE = pickle.load(open('storage_30nt_new/sample_file_name_UNIQUE.p', 'rb'))
#
# # homopolymer matrix
# homology_matrix_UNIQUE = homology_matrix_finder(spacer_list_UNIQUE)
#
# # one-hot encoded
# spacer_pam_per_site_one_hot_UNIQUE = load_gene_sequence(spacer_list_UNIQUE)
#
# location_dict = {}
# donor_dict = {}
#
# with open('sequence_pam_gene_grna_big_file_donor_genomic_context_with_donor_30nt_new.csv', 'rb') as csvfile:
#
#   spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#   row_counter = 0
#   for row in spamreader:
#     location_dict[row[0].split(',')[2] + row[0].split(',')[3] ] = row[0].split(',')[4]
#
#
# location_list_UNIQUE = []
# counter = 0
# for ind,spacer in enumerate(spacer_list_UNIQUE):
#   location_list_UNIQUE.append(location_dict[spacer])
#
#
# pickle.dump(location_list_UNIQUE, open('storage_30nt_new/location_list_UNIQUE.p', 'wb'))
# pickle.dump(homology_matrix_UNIQUE, open('storage_30nt_new/homology_matrix_UNIQUE.p', 'wb'))
# pickle.dump(spacer_pam_per_site_one_hot_UNIQUE, open('storage_30nt_new/spacer_pam_per_site_one_hot_UNIQUE.p', 'wb'))
###########




