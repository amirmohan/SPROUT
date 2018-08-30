## Amirali Aghazadeh, Aug 2018, Stanford University
import numpy as np
import pickle
import sys

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

def simple_prediction_function(spacer_pam):
    m_frac_total_ins = pickle.load(open('models/fraction_total_insertions_other_cells.p', 'rb'))
    m_frac_total_del = pickle.load(open('models/fraction_total_deletions_other_cells.p', 'rb'))
    m_frac_mutant_ins = pickle.load(open('models/fraction_insertions_other_cells.p', 'rb'))
    m_avg_ins_length = pickle.load(open('models/exp_ins_length_other_cells.p', 'rb'))
    m_avg_del_length = pickle.load(open('models/exp_deletion_length_other_cells.p', 'rb'))
    m_diversity = pickle.load(open('models/diversity_other_cells.p', 'rb'))
    single_bp_inserted = pickle.load(open('models/single_insertion_type_4class-classification_other_cells.p', 'rb'))

    sequence_pam_per_gene_grna = np.zeros((1, 23, 4), dtype=bool)
    for ind,basepair in enumerate(spacer_pam):
        sequence_pam_per_gene_grna[0,ind,one_hot_index(basepair)] = 1

    sequence_pam_per_gene_grna = np.reshape(sequence_pam_per_gene_grna , (1,-1))


    print "\nHere are the repair outcomes that SPROUT predicts for this guide:"
    frac_total_ins = 100 * float(m_frac_total_ins.predict(sequence_pam_per_gene_grna)[0])
    frac_mutant_ins = 100 * float(m_frac_mutant_ins.predict(sequence_pam_per_gene_grna)[0])

    print "Fraction of total reads with insertion \t\t %.0f %%" %frac_total_ins
    #print "Fraction of total reads with deletion \t\t %.0f %%"  % (frac_total_ins*(100/(frac_mutant_ins) -1))
    print "Insertion to deletion ratio \t\t\t\t %.0f %%"  % (100*(frac_mutant_ins / float((100 - frac_mutant_ins))))
    print "Average insertion length \t\t\t\t\t %.1f bps" %float(m_avg_ins_length.predict(sequence_pam_per_gene_grna)[0])
    print "Average deletion length \t\t\t\t\t %.1f bps" %float(m_avg_del_length.predict(sequence_pam_per_gene_grna)[0])

    diversity = m_diversity.predict(sequence_pam_per_gene_grna)[0]
    if diversity > 3.38:
        print "Diversity \t\t\t\t\t\t\t\t\t %.2f (High)" %float(diversity)
    else:
        print "Diversity \t\t\t\t\t\t\t\t\t %.2f (Low)" % float(diversity)

    nucleotide_array = ['A', 'C', 'G', 'T']
    print "Most likely inserted base pair \t\t\t\t %s" %nucleotide_array[int(single_bp_inserted.predict(sequence_pam_per_gene_grna)[0])]





def prediction_function(spacer_pam,genomic_factor):
    m_frac_total_ins = pickle.load(open('models/fraction_total_insertions_other_cells.p', 'rb'))
    m_frac_total_del = pickle.load(open('models/fraction_total_deletions_other_cells.p', 'rb'))
    m_frac_mutant_ins = pickle.load(open('models/fraction_insertions_other_cells.p', 'rb'))
    m_avg_ins_length = pickle.load(open('models/exp_ins_length_other_cells_chrom.p', 'rb'))
    m_avg_del_length = pickle.load(open('models/exp_deletion_length_other_cells_chrom.p', 'rb'))
    m_diversity = pickle.load(open('models/entropy_other_cells_chrom.p', 'rb'))
    m_single_bp_inserted = pickle.load(open('models/single_insertion_type_4class-classification_other_cells.p', 'rb'))
    m_edit_eff = pickle.load(open('models/edit_eff_other_cells_chrom.p', 'rb'))

    sequence_pam_per_gene_grna = np.zeros((1, 23, 4), dtype=bool)
    for ind,basepair in enumerate(spacer_pam):
        sequence_pam_per_gene_grna[0,ind,one_hot_index(basepair)] = 1

    sequence_pam_per_gene_grna = np.reshape(sequence_pam_per_gene_grna , (1,-1))

    sequence_pam_genomic_per_gene_grna = np.transpose(np.concatenate(( np.transpose(sequence_pam_per_gene_grna),np.transpose(genomic_factor))))


    print "\nHere are the repair outcomes that SPROUT predicts for this guide:"
    frac_total_ins = 100 * float(m_frac_total_ins.predict(sequence_pam_per_gene_grna)[0])
    frac_mutant_ins = 100 * float(m_frac_mutant_ins.predict(sequence_pam_per_gene_grna)[0])

    print "Fraction of total reads with insertion \t\t %.0f %%" %frac_total_ins
    print "Insertion to deletion ratio \t\t\t\t %.0f %%"  % (100*(frac_mutant_ins / float((100 - frac_mutant_ins))))
    print "Average insertion length \t\t\t\t\t %.1f bps" %float(m_avg_ins_length.predict(sequence_pam_genomic_per_gene_grna)[0])
    print "Average deletion length \t\t\t\t\t %.1f bps" %float(m_avg_del_length.predict(sequence_pam_genomic_per_gene_grna)[0])

    diversity = m_diversity.predict(sequence_pam_genomic_per_gene_grna)[0]
    if diversity > 3.38:
        print "Diversity \t\t\t\t\t\t\t\t\t %.2f (High)" %float(diversity)
    else:
        print "Diversity \t\t\t\t\t\t\t\t\t %.2f (Low)" % float(diversity)

    nucleotide_array = ['A', 'C', 'G', 'T']
    print "Most likely inserted base pair \t\t\t\t %s" %nucleotide_array[int(m_single_bp_inserted.predict(sequence_pam_per_gene_grna)[0])]

    print "Edit efficiency \t\t\t\t\t\t\t %.0f %%" % (100*float(m_edit_eff.predict(sequence_pam_genomic_per_gene_grna)[0]))


input_indicator = raw_input("Which input format do you prefer? \n(1) sgRNA sequence only\n(2) sgRNA sequence + genomic features (chromatin, etc.)\n(3) location on the genome and cell type \n\nSelected option:\n")
if input_indicator == '1':
    spacer_pam = raw_input("\nInput the sgRNA sequence followed by the PAM sequence:\n")
    proceed_flag = 1

    if len(spacer_pam)<23:
        print "Sequence is too short."
        proceed_flag = 0

    if spacer_pam.count('A')+spacer_pam.count('T')+spacer_pam.count('C')+spacer_pam.count('G') != len(spacer_pam):
        print "Sequence should contains four characters A, T, C, and G."
        proceed_flag = 0

    if proceed_flag == 1:
        simple_prediction_function(spacer_pam)


if input_indicator == '2':
    proceed_flag = 1
    spacer_pam = raw_input("\nInput the sgRNA sequence followed by the PAM sequence:\n")

    if len(spacer_pam)<23:
        print "Sequence is too short."
        proceed_flag = 0

    if spacer_pam.count('A')+spacer_pam.count('T')+spacer_pam.count('C')+spacer_pam.count('G') != len(spacer_pam):
        print "Sequence should contains four characters A, T, C, and G."
        proceed_flag = 0

    chrom_factor = raw_input("\nInput the genomic factors separated by ',':\n")
    chrom_factor = np.asarray(chrom_factor.split(','))
    if len(chrom_factor)!=33:
        print "Number of genomic feaures do not match. Please enter 33 features."

    else:
        chrom_factor_float = []
        for ind,item in enumerate(chrom_factor):
            chrom_factor_float.append(float(item))

        chrom_factor_float = np.expand_dims(chrom_factor_float, axis=0)
        prediction_function(spacer_pam,chrom_factor_float)



if input_indicator == '3':
    cell_type = raw_input("\nWhat is the target cell type?\n")
    chr = raw_input("\nWhich chromosome the cut site locates?\n")
    position = int(raw_input("\nWhat location the cut site is in that chromosome?\n"))


    flag = 0
    with open('/Users/amirali/Software/refgenome/hg38.fa', 'r') as inF:
        for ind,line in enumerate(inF):
            if '>chr%s\n'%chr in line:
                indstart = ind
                flag = 1
            if flag == 1 and ind == (indstart + position / 50):
                targetline1 = line
            if flag == 1 and ind == (indstart + position / 50 + 1):
                targetline2 = line

    if  position%50 < 50-23:
        guide = targetline1[position%50:position%50+23]
    else:
        guide = targetline1[position%50:] + targetline2[:23-position%50]

    print "\nThis is the selected guide sequence:"
    print guide

    if guide.count('A')+guide.count('T')+guide.count('C')+guide.count('G') != 23:
        print "The sequence is invalid. It contains characters out of the four A, T, C, and G nucleotide."
    elif guide[-2:] != 'GG':
        print "The PAM sequence is invalid. PAM should be in NGG format."
    else:
        print simple_prediction_function(guide)







