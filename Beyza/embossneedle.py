"""Code for Emboss-Needle global pairwise alignment and Creating output needle files"""

from Bio.Emboss.Applications import NeedleCommandline
def f2():
    proteins=['NSP1', 'NSP2', 'NSP3','NSP4', 'NSP5', 'NSP6', 'NSP7', 'NSP8', 'NSP9', 'NSP10', 'NSP11', 'NSP12', 'NSP13', 'NSP14', 'NSP15', 'NSP16', 'Spike', 'NS3', 'E', 'M', 'NS6', 'NS7a', 'NS7b', 'NS8', 'N']
    for each in proteins:
        needle_cline = NeedleCommandline(asequence=f'{each}referance.fasta', bsequence=f'{each}.fasta', gapopen=10, gapextend=0.5, datafile='EPAM40', outfile=f'{each}needlePAM40.txt', aformat='score', nobrief=True)
        stdout, stderr = needle_cline()
        print(stdout + stderr)