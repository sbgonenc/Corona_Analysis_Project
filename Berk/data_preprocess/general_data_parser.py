import pandas as pd
from pprint import pprint

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 190)

'''
#Ön hazırlık kodları (corona statsı, HDI ve healthexpenseperGDP datalarını birleştirmek için)
hdi = "datasets/Human development index (HDI).csv"
coronastats = "datasets/latest_covid_data.csv"
HealthexpenseperGDP= "datasets/HealthexpenseperGDP.csv"

hdi_df = pd.read_csv(hdi, header=1)
hdi_df.set_index('Country', inplace=True)

corona_df = pd.read_csv(coronastats)
corona_df.rename(columns={'location': 'Country'}, inplace=True)
corona_df.set_index('Country', inplace=True)
#print(corona_df)


HpG_df = pd.read_csv(HealthexpenseperGDP)
HpG_filter = (HpG_df['TIME'] >= 2018) #& ((HpG_df['LOCATION'] == 'TUR') | (HpG_df['LOCATION'] == 'ITA') | (HpG_df['LOCATION'] == 'USA'))
HpG_df = HpG_df.loc[HpG_filter, ['LOCATION', 'TIME', 'Value']]
HpG_df.rename(columns={'Value':'HealthExpenseperGDP', 'LOCATION':'iso_code'}, inplace=True)
HpG_df.set_index('iso_code', inplace=True)
#print(HpG_df)


data= corona_df.join(hdi_df['2018'], how='outer')
data.rename(columns = {'2018': 'HDI'}, inplace= True)
data.reset_index(inplace=True)
data.set_index('iso_code', inplace=True)
data= data.join(HpG_df['HealthExpenseperGDP'], how='outer')
#print(data)

data_for_stats = data
data_for_stats.reset_index(inplace=True)
data_for_stats.to_csv("all_dataappended.txt", sep=",", index=False, header=True)
'''

#DATA preprocessing - Handling non-typical values

corona_file = r'C:\Users\BERK\PycharmProjects\CORONA_PROJECT\datasets\all_corona_data.txt'

corona_dataframe = pd.read_table(corona_file)

def str_float_convert(str_type, index, column):
	'''
	data içindeki verileri düzenler (specific olarak corona datası)
	:param str_type: dataframe column (datamalipulator'dan alır)
	:param index: dataframe index (data_manipulator'dan alır)
	:param column: ref value alabilmek icin ihtiyac duyar
	:return: float ve rounded values
	'''

	try:
		f_type = round(float(str_type), 3)

		if f_type:
			corona_dataframe.loc[index, column] = f_type
			return f_type

	except ValueError:

		s_list = str_type.split('.')

		if 'aged' in column or 'handwash' in column or 'cvd' in column or 'Expense' in column:
			s_str = s_list[0] + '.' + ''.join(s_list[1:])
			f_str = round(float(s_str), 3)

			return f_str

		if 'gdp' in column:
			s_str = ''.join(s_list[0:2]) + '.' + ''.join(s_list[2])
			f_str = round(float(s_str), 3)

			return f_str


		if index == 0: return str_type

		else:
			ref_value = corona_dataframe.loc[index - 1, column]   #bazı column'larda uygunsuz değerleri değiştirebilmek için bir referans değerine ihtiyaç duydum.
			ref_list = str(ref_value).split('.')

			if 'density' in column:
				ref_value = 237.016
				ref_list = str(ref_value).split('.')
				s_str = "".join(s_list)
				s_str = s_str[0:len(ref_list[0])] + '.' + s_str[len(ref_list[0]):]
				f_str = round(float(s_str), 3)

				return f_str

			if len(s_list) != len(ref_list):
				s_str = "".join(s_list)
				s_str= s_str[0:len(ref_list[0])] + '.' + s_str[len(ref_list[0]):]
				f_str = round(float(s_str), 3)

				corona_dataframe.loc[index, column] = f_str

				return f_str

		return '00000'


def data_manipulator(column_name, dataframe):
	'''
	corona datatsında bazı veriler new_death'i 0 olmasına rağmen 1-2 hafta öncesinden girdilere sahip.
	Bunları elimine etmek ve tüm ülkeleri ilk case- ilk date olarak sabitlemek
	:param tsv_file: birleştirilmiş corona datası (tsv) formatında
	:return: pandas dataframe
	'''

	df = dataframe
	df = df[df['total_cases'] != 0]
	df = df[df['total_deaths'] != 0]

	c_name = column_name
	df[c_name] = [i for i in map(str_float_convert, df[c_name], df.index, [c_name for _ in range(len(df[c_name]))])]

	return df


'''
manipulated_df = data_manipulator('total_cases_per_million', corona_dataframe)
manipulated_df = data_manipulator('total_deaths_per_million', manipulated_df)
manipulated_df = data_manipulator('total_cases_per_million', manipulated_df)
manipulated_df = data_manipulator('total_tests_per_thousand', manipulated_df)
manipulated_df = data_manipulator('new_deaths_per_million', manipulated_df)
manipulated_df = data_manipulator('new_cases_per_million', manipulated_df)
manipulated_df = data_manipulator('new_tests_per_thousand', manipulated_df)
manipulated_df= data_manipulator('aged_70_older', manipulated_df)
manipulated_df= data_manipulator('aged_65_older', manipulated_df)
manipulated_df= data_manipulator('gdp_per_capita', manipulated_df)
manipulated_df= data_manipulator('population_density', manipulated_df)
manipulated_df= data_manipulator('handwashing_facilities', manipulated_df)
manipulated_df= data_manipulator('cvd_death_rate', manipulated_df)
manipulated_df= data_manipulator('HealthExpenseperGDP', manipulated_df)
'''
#manipulated_df.to_csv('All_manipulated.txt', index=False, sep='\t')

# Further manipulation of the data and filling mising values
import math

'''
df_file = 'All_manipulated_cleanedUp.txt'

df = pd.read_table(df_file)



country_names = [c for c in country_prot_number.keys()]
country_filter = df['Country'].isin(country_names)


df['log_population'] = [i for i in map(math.log, df['population'])]

new_df = df[country_filter]


new_df['new_deaths'].fillna(0, inplace=True)
new_df['new_cases'].fillna(0, inplace=True)
new_df['new_tests'].fillna(0, inplace=True)


#print(new_df['new_cases_per_million'].describe())

new_df['new_cases_per_million'].fillna((new_df['new_cases']/new_df['population'])*1000000, inplace=True)
new_df['new_deaths_per_million'].fillna((new_df['new_deaths']/new_df['population'])*1000000, inplace=True)
new_df['total_cases_per_million'].fillna((new_df['total_cases']/new_df['population'])*1000000, inplace=True)
new_df['new_tests_per_thousand'].fillna((new_df['new_tests']/new_df['population'])*1000, inplace=True)
new_df['total_tests_per_thousand'].fillna((new_df['total_tests']/new_df['population'])*1000, inplace=True)

#print(new_df.isnull().sum())
'''
def rounder_func(df, column_name):

	floating_lst = [float(f) for f in df[column_name]]
	rounded = [round(x, 3) for x in floating_lst]
	df[column_name] = [r for r in rounded]

	return df

'''
rounded_new_df = rounder_func(new_df, 'new_cases_per_million')
rounded_new_df = rounder_func(rounded_new_df, 'new_deaths_per_million')
rounded_new_df = rounder_func(rounded_new_df, 'total_cases_per_million')
rounded_new_df = rounder_func(rounded_new_df, 'total_deaths_per_million')
rounded_new_df = rounder_func(rounded_new_df, 'population_density')
rounded_new_df = rounder_func(rounded_new_df, 'aged_70_older')
rounded_new_df = rounder_func(rounded_new_df, 'aged_65_older')
rounded_new_df = rounder_func(rounded_new_df, 'log_population')
#rounded_new_df = rounder_func(new_df, 'new_tests_per_thousand')
#rounded_new_df = rounder_func(new_df, 'total_tests_per_thousand')

rounded_new_df.drop(
	columns=['new_tests_per_thousand', 'total_tests_per_thousand', 'total_tests', 'new_tests_smoothed',
			 'new_tests_smoothed_per_thousand',	'tests_units', 'new_tests'], inplace=True
)
#print(new_df.describe())
#print(new_df.isnull().sum())

#rounded_new_df.to_csv('Filled_cleaned_coronadf.txt', index=False, sep='\t')
'''
#Adding smokers data


smokers_file = pd.read_csv(r'C:\Users\BERK\PycharmProjects\CORONA_PROJECT\datasets\smokers.csv')
smoker_df = smokers_file.drop(columns=['maleSmokingRate','femaleSmokingRate', 'pop2020'])
#cigar_filter = smoker_df['Country'] == ()
smoker_df.rename(columns={'name':'Country', 'totalSmokingRate':'total_smokers'}, inplace=True)


#print(smoker_df)

#rounded_new_df = rounded_new_df.merge(smoker_df, on='Country', how='outer')
#df=df.merge(smoker_df, on='Country', how='outer')

#df.to_csv('smoker_added_coronadfall.txt', index=False, sep='\t')

#print(rounded_new_df)

#rounded_new_df['iso_code'].dropna(inplace=True)


'''
def total_smoker_fill(df):
	sex_ratios = { #country: males/females
		'Algeria': 102.1/100, 'Czech Rebuplic': 96.984/100, 'Gambia':98.393/100, 'Guam': 101.79/100,
		'Kuwait': 157.866/100, 'New Zealand': 96.55/100, 'Peru': 98.707/100, 'Qatar': 302.426/100,
		'Taiwan': 98.764/100, 'Tunisia': 98.37/100, 'United Arab Emirates': 223.845/100
	}

	data = df

	for country in sex_ratios:
		filter = data['Country'] == country
		data.loc[filter, 'total_smokers'] = (data.loc[filter, 'female_smokers']*sex_ratios[country] + data.loc[filter,'male_smokers'])/(sex_ratios[country]+1)

	filled_df = data

	return filled_df

cigarated_df = total_smoker_fill(rounded_new_df)
cigarated_df = rounder_func(cigarated_df, 'total_smokers')
cigarated_df['iso_code'].dropna(inplace=True)

print(cigarated_df['total_smokers'].describe())
print(cigarated_df['total_smokers'].isna().sum())
print(cigarated_df)

cigarated_df.to_csv('smoker_added_coronadf.txt', index=False, sep='\t')
'''
#Protein datasını preprocess edebilme
protein_file = r"C:\Users\BERK\PycharmProjects\CORONA_PROJECT\datasets\ALL11111dataframe.csv"


def percent_float_converter(file):

	with open(file, 'r') as f:
		with open('all_proteins.txt', 'w') as at:
			for line in f:
				new_line = line.strip().replace('%', '')
				at.write(new_line+'\n')


#percent_float_converter(protein_file)

protein_dataframe = pd.read_csv('all_proteins.txt', header=0, sep='\t')
protein_df = protein_dataframe.drop(columns=['id'])
protein_df.rename(columns={'country':'Country'}, inplace=True)


print(protein_df.describe())
#print(protein_df['identity for NSP1'].describe())
print(protein_df.isna().sum())
#fil_tr = protein_df['Country'] == 'Turkey'
#protein_tr = protein_df[fil_tr]

#print(protein_tr)
