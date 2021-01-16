"""Code for parsing the all protein sequence data taken from the Gisaid website and creating files for each protein and the reference sequence"""

from Bio.SeqIO.FastaIO import FastaIterator


def f1():
    myfile = 'allprot0621.fasta'
    
    def fasta_reader(filename):
        with open(filename) as handle:
            for record in FastaIterator(handle):
                yield record
    
    proteins = []
    for entry in fasta_reader(myfile):
        sp = str(entry.id).split('|')
        if str(sp[1]) == 'hCoV-19/Wuhan-Hu-1/2019':
            refer_protein = sp[0].strip('>')
            outfile = open(f"{refer_protein}reference.fasta", 'w')
            outfile.write('>' + str(entry.description) + '\n' + str(entry.seq) + '\n')
            outfile.close()
        else:
            if len(sp) > 6 and str(sp[6]).upper() == 'HUMAN' and str(str(entry.seq).find('X')) == '-1':
                protein = sp[0].strip('>')
                if protein not in proteins:
                    fileout = open(f"{protein}.fasta", 'w')
                    proteins.append(protein)
                else:
                    fileout = open(f"{protein}.fasta", 'a')
                fileout.write('>' + str(entry.description) + '\n' + str(entry.seq) + '\n')
                fileout.close()
