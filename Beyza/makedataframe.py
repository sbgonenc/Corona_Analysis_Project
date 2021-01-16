import pandas as pd

def f3():
    proteins=['NSP1', 'NSP2', 'NSP3', 'NSP4', 'NSP5', 'NSP6', 'NSP7', 'NSP8', 'NSP9', 'NSP10', 'NSP11', 'NSP12', 'NSP13', 'NSP14', 'NSP15', 'NSP16', 'Spike', 'NS3', 'E', 'M', 'NS6', 'NS7a', 'NS7b', 'NS8', 'N']
    next=pd.DataFrame(columns= ['Accession ID','country', 'date'])
    for protein in proteins:
        dffirst=pd.DataFrame()
        list_score= []
        list_identity= []
        countries=[]
        list_id=[]
        dates=[]
        with open(f'{protein}.fasta') as hand:
            for line in hand:
                if line.startswith('>'):
                    line = line.split('|')
                    id=line[3]
                    date = line[2]
                    country = line[-1].strip()
                    list_id.append(id)
                    countries.append(country)
                    dates.append(date)
        with open(f'{protein}needlePAM40.txt') as f:
            for lines in f:
                if lines.startswith('#'):
                    continue
                if lines.isspace():
                    continue
                score=lines.split(' ')[-1].strip(' ()\n')
                identity=lines.split(' ')[-2]
                list_identity.append(identity)
                list_score.append(score)
        dffirst['country'] = countries
        dffirst['Accession ID'] = list_id
        dffirst['date'] = dates
        dffirst[f'score for {protein}'] = list_score
        dffirst[f'identity for {protein}'] = list_identity
        next = pd.merge(next, dffirst, on=['Accession ID','country', 'date'], how='outer' )
    
    """Code for adding the clade and lineage information from the metadata.tsv file obtained from Gisaid."""
    
    metadata = pd.read_csv("../metadata.tsv", sep="\t")
    df2 = next
    metadata.drop(['strain','virus', 'genbank_accession', 'date', 'region', 'country', 'division', 'location',\
             'region_exposure', 'country_exposure', 'division_exposure', 'segment', 'length', 'host', 'age',\
              'sex','originating_lab', 'submitting_lab', 'authors', 'url', 'title', 'paper_url', 'date_submitted'],\
             axis=1,  inplace=True)
    
    finaldf1 = pd.merge(df2, metadata, left_on='Accession ID', right_on='gisaid_epi_isl', how='inner')
    del finaldf1['gisaid_epi_isl']
    finaldf1.reset_index(drop=True, inplace=True)
    finaldf1.replace('2020-02-00', '01.02.2020', inplace=True)
    finaldf1.replace('2020-02', '01.02.2020', inplace=True)
    finaldf1.replace('2020-03', '01.03.2020', inplace=True)
    finaldf1.replace('2020-03-00', '01.03.2020', inplace=True)
    finaldf1.replace('2020-00-00', '01.01.2020', inplace=True)
    finaldf1.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    
    
    
    """Code for merging the other data(deaths, cases vs.)"""
    
    df1 = pd.read_table('CLEANED_d2d_merged_means(PAM40)(16.06).txt')
    
    df1.drop(['score for NSP1', 'score for NSP2', 'score for NSP3', 'score for NSP4', 'score for NSP5', 'score for NSP6', \
              'score for NSP7', 'score for NSP8', 'score for NSP9', 'score for NSP10', 'score for NSP11', 'score for NSP12',\
              'score for NSP13', 'score for NSP14', 'score for NSP15', 'score for NSP16', 'score for Spike', 'score for NS3', \
              'score for E', 'score for M', 'score for NS6', 'score for NS7a', 'score for NS7b', 'score for NS8', 'score for N', \
              'identity percentage for NSP1', 'identity percentage for NSP2', 'identity percentage for NSP3', 'identity percentage\
               for NSP4', 'identity percentage for NSP5', 'identity percentage for NSP6', 'identity percentage for NSP7', 'identity\
                percentage for NSP8', 'identity percentage for NSP9', 'identity percentage for NSP10', 'identity percentage for NSP11',\
              'identity percentage for NSP12', 'identity percentage for NSP13', 'identity percentage for NSP14', 'identity percentage \
              for NSP15', 'identity percentage for NSP16', 'identity percentage for Spike', 'identity percentage for NS3', 'identity\
               percentage for E', 'identity percentage for M', 'identity percentage for NS6', 'identity percentage for NS7a', \
              'identity percentage for NS7b', 'identity percentage for NS8', 'identity percentage for N'],1, inplace=True)
    df1.rename(columns={"Country":"country"}, inplace=True)
    
    final=pd.merge(df1, finaldf1, on=['country', 'date'], how='inner')
    pd.set_option('display.max_columns', None)
    final.to_csv('CLEANEDall.csv', sep='\t', index=False)