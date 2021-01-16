########### Aligned protein sequence scores ####

import pandas as pd

# protein_file = r'C:\Users\BERK\PycharmProjects\CORONA_PROJECT\datasets\ALL11111dataframe.csv'

pd.set_option('display.max_columns', 80)

new_protein_file = r'C:\Users\BERK\PycharmProjects\CORONA_PROJECT\PAM40_protall.csv'

name_dict = {'Democratic Republic of the Congo': 'Democratic Republic of Congo',
			 'USA': 'United States', 'Korea': 'South Korea'}

protein_dataframe = pd.read_csv(new_protein_file, header=0, sep='\t')
protein_dataframe.rename(columns={'country': 'Country'}, inplace=True)

# protein_dataframe['Country'].rename(lambda x: name_dict[x] if x in name_dict else x, inplace=True)
protein_dataframe['Country'] = protein_dataframe['Country'].apply(lambda x: name_dict[x] if x in name_dict else x)
# protein_dataframe['date'] = protein_dataframe['date'].apply(lambda x: x.replace('-','/'))
"""
country_lst = []
for country in protein_dataframe['Country']:
	if country not in country_lst:
		country_lst.append(country)
	else: continue

"""
general_data_file = r'C:\Users\BERK\PycharmProjects\CORONA_PROJECT\Smoker_14-06_added_coronadf.txt'  ####To be merged
general_df = pd.read_csv(general_data_file, sep='\t', header=0)

"""
for genel_country in general_df['Country']:
	if genel_country not in country_lst:
		index_name = general_df[general_df['Country'] == genel_country].index
		general_df.drop(index_name, inplace=True)
"""
# general_df.reset_index(inplace=True)

"""
general_df['date'] = pd.to_datetime(general_df['date'], dayfirst=True)
print(general_df['date'])


def percentage_filter(df):

	filt_lst = []
	for column in df:
		if 'percentage' in column:
			filt_lst.append(f"(df['{column}']>= {75})")

	filt_ = ' & '.join(filt_lst)
	filtered_df = df.loc[filt_]

	return filtered_df


def date_manipulator(value):
	'''converts yyyy/mm/dd to dd/mm/yyyy'''
	sp = value.split('/')
	rv_lst = sp[2], sp[1], sp[0]
	rv_str = '/'.join(rv_lst)
	return rv_str


def country_comparison(df1, df2):
	'''
	compares protein vs general dataframes on country
	:param df1: protein country list
	:param df2: general dataframe
	:return: filtered general dataframe
	'''
	genel = df2
	prot = df1
	c_lst = []
	prot_c_lst = []
	for country in genel['Country']:
		if country not in c_lst:
			c_lst.append(country)

	for prot_cntry in prot['Country']:
		if prot_cntry not in prot_c_lst:
			prot_c_lst.append(prot_cntry)

	for cntry in c_lst:
		if cntry not in prot_c_lst:
			drop_index = genel[genel['Country']==cntry].index
			genel.drop(drop_index, inplace=True)

	return genel


#protein_dataframe['date'] = protein_dataframe['date'].apply(date_manipulator)
protein_dataframe['date'] = pd.to_datetime(protein_dataframe['date'], yearfirst=True)

print(protein_dataframe['date'])


import statistics as stat
protein_dataframe.drop(columns=['Accession ID'], inplace=True)


#grouped_df = seventy5_filtered.groupby('date').agg(lambda x: list(x))
#grouped_df_c_means = seventy5_filtered.groupby('Country').agg(lambda x: stat.mean(list(x))).reset_index()
#grouped_df_c_medians = seventy5_filtered.groupby('Country').agg(lambda x: stat.median(list(x))).reset_index()
#grouped_data = seventy5_filtered.groupby(['Country', 'date']).agg(lambda x: list(x))
grouped_proteins_means = protein_dataframe.groupby(['Country', 'date']).agg(lambda x: stat.mean(list(x)))
grouped_proteins_medians = protein_dataframe.groupby(['Country', 'date']).agg(lambda x: stat.median(list(x)))
#print(grouped_data_means)



#print(general_df['Country'])

general_df = country_comparison(protein_dataframe, general_df)
"""
# merged_df = general_df.merge(grouped_df_c_means, how='outer', on='Country')
# merged_medians = general_df.merge(grouped_df_c_medians, how='outer', on='Country')
"""
general_grouped = general_df.groupby(['Country', 'date']).agg(lambda x: x)
#print(general_grouped)

date_to_date_merged_means = general_grouped.join(grouped_proteins_means, how='outer')
date_to_date_merged_means.reset_index(inplace=True)

date_to_date_merged_medians = general_grouped.join(grouped_proteins_medians, how='outer')
date_to_date_merged_medians.reset_index(inplace=True)

#d2d_means = date_to_date_merged_means.applymap(rounder_func)
#d2d_means.to_csv('d2d_merged_means(PAM40)(17-06).txt', index=False, sep='\t')


#d2d_medians = date_to_date_merged_medians.applymap(rounder_func)
#d2d_medians.to_csv('d2d_merged_medians(PAM40)(17-06).txt', index=False, sep='\t')
"""
'''merged_df = merged_df.applymap(rounder_func)

merged_df.to_csv('merged_means.txt', sep='\t', index=False)


merged_medians = merged_medians.applymap(rounder_func)

merged_medians.to_csv('merged_medians.txt', sep='\t', index=False)
'''

#### Adding clades and lineage ##### 17/06
from Scripts import manipulation_functions as mf

prot_file = r'C:\Users\BERK\PycharmProjects\CORONA_PROJECT\PAM40_protall.csv'

prot_df = pd.read_table(prot_file)
prot_clades_df = prot_df.drop(columns=['Accession ID',
									   'identity percentage for NSP1', 'score for NSP1',
									   'identity percentage for NSP2', 'score for NSP2',
									   'identity percentage for NSP3', 'score for NSP3',
									   'identity percentage for NSP4', 'score for NSP4',
									   'identity percentage for NSP5', 'score for NSP5',
									   'identity percentage for NSP6', 'score for NSP6',
									   'identity percentage for NSP7', 'score for NSP7',
									   'identity percentage for NSP8', 'score for NSP8',
									   'identity percentage for NSP9', 'score for NSP9',
									   'identity percentage for NSP10', 'score for NSP10',
									   'identity percentage for NSP11', 'score for NSP11',
									   'identity percentage for NSP12', 'score for NSP12',
									   'identity percentage for NSP13', 'score for NSP13',
									   'identity percentage for NSP14', 'score for NSP14',
									   'identity percentage for NSP15', 'score for NSP15',
									   'identity percentage for NSP16', 'score for NSP16',
									   'identity percentage for Spike', 'score for Spike',
									   'identity percentage for NS3', 'score for NS3',
									   'identity percentage for E', 'score for E',
									   'identity percentage for M', 'score for M',
									   'identity percentage for NS6', 'score for NS6',
									   'identity percentage for NS7a', 'score for NS7a',
									   'identity percentage for NS7b', 'score for NS7b',
									   'identity percentage for NS8', 'score for NS8',
									   'identity percentage for N', 'score for N'])
prot_clades_df['date'] = pd.to_datetime(prot_clades_df['date'])
prot_clades_df['Country'] = prot_clades_df['Country'].apply(lambda x: name_dict[x] if x in name_dict else x)
prot_clades_df = prot_clades_df.groupby(['Country', 'date']).agg({'Clade':mf.series_reducer, 'Lineage':mf.series_reducer}).reset_index()

#print(prot_clades_df.dtypes)

cleand_corona_file = r'C:\Users\BERK\PycharmProjects\CORONA_PROJECT\Scripts\d2d_merged_means(PAM40)(17-06).txt'
cc_df = pd.read_table(cleand_corona_file)
cc_df['date'] = pd.to_datetime(cc_df['date'])

#print(cc_df.dtypes)

clade_combined_df = cc_df.merge(prot_clades_df, how='outer', on=['Country', 'date'])

clade_combined_df = clade_combined_df.applymap(mf.rounder_func)

###clade_combined_df.to_csv('clade_combined_17_06.txt', index=False, sep='\t')

file ='clade_combined_17_06.txt'
df = pd.read_table(file)
#flt = df['iso_code'].isna()
#print(df.loc[flt, 'iso_code'])

drop_index = df[(df['total_cases'] == 0) & (df['Clade'].isna())].index
df.drop(drop_index, axis=0, inplace=True)

#print(df.describe())
df= df.applymap(mf.rounder_func)
df.to_csv('Cleaned_clade_combined_17_06.txt', index=False, sep='\t')

