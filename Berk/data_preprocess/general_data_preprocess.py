import pandas as pd
from pprint import pprint

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 190)
"""
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
HpG_filter = (HpG_df['TIME'] >= 2018) #& ((HpG_df['LOCATION'] == 'TUR') | (HpG_df['LOCATION'] == 'ITA') | (HpG_df['LOCATION'] == 'USA'))  ##Deneme amaçlı yazılmış filtreler
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
data_for_stats.to_csv("all_dataappended.txt", sep=",", index=False, header=True)    # Checkpoint for combined data
"""

#DATA preprocessing - Handling non-typical values

corona_file = r'/datasets/all_corona_data.txt'
"""
corona_dataframe = pd.read_table(corona_file)

def str_float_convert(str_type, index, column):						#Her column'daki non-typical value'lar farklı bir pattern gösterdiğinden, çağırdığım fonksiyona çok fazla kural yazmam gerekti.
	'''
	data içindeki verileri düzenler (specific olarak corona datası). Value'ları tek tek döner
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
				s_str = s_str[0:len(ref_list[0])] + '.' + s_str[len(ref_list[0]):]
				f_str = round(float(s_str), 3)

				corona_dataframe.loc[index, column] = f_str

				return f_str

		return str_type


def data_manipulator(column_name, dataframe):
	'''
	Non-typical value'ları str_float_convert'e gönderir, column'
	:param tsv_file: birleştirilmiş corona datası (tsv) formatında
	:return: pandas dataframe
	'''

	df = dataframe

	c_name = column_name
	cleaned_values = map(
		str_float_convert, 						#Called function
		df[c_name], df.index, [c_name for _ in range(len(df[c_name]))]  #Inputs: values in the column, indices, column_name_list
	)
	df[c_name] = [i for i in cleaned_values]

	return df


#total_deaths_filter = corona_dataframe['total_deaths'] != 0
#manipulated_df = corona_dataframe[total_deaths_filter]			# Death growth rate'e bakacağımıza göre, total_death=0 durumları bizi ilgilendirmiyor
manipulated_df = corona_dataframe


manipulated_df = data_manipulator('total_deaths_per_million', manipulated_df)
manipulated_df = data_manipulator('total_cases_per_million', manipulated_df)
manipulated_df = data_manipulator('total_tests_per_thousand', manipulated_df)
manipulated_df = data_manipulator('new_deaths_per_million', manipulated_df)
manipulated_df = data_manipulator('new_cases_per_million', manipulated_df)
manipulated_df = data_manipulator('new_tests_per_thousand', manipulated_df)
manipulated_df = data_manipulator('aged_70_older', manipulated_df)
manipulated_df = data_manipulator('aged_65_older', manipulated_df)
manipulated_df = data_manipulator('gdp_per_capita', manipulated_df)
manipulated_df = data_manipulator('population_density', manipulated_df)
manipulated_df = data_manipulator('handwashing_facilities', manipulated_df)
manipulated_df = data_manipulator('cvd_death_rate', manipulated_df)
manipulated_df = data_manipulator('HealthExpenseperGDP', manipulated_df)


manipulated_df.to_csv('All_10-06_manipulated.txt', index=False, sep='\t')  #Checkpoint
"""
# Further manipulation of the data and filling mising values
import math


df_file = '../../All_10-06_manipulated.txt'

df = pd.read_table(df_file)

df['log_population'] = [i for i in map(math.log, df['population'])]

"""country_prot_number = {'China': 574, 'Australia': 1431, 'United States': 6390, 'United Kingdom': 15006,			#Ülkelerdeki protein data sayısı
					   'Netherlands': 598, 'Czech Republic': 30, 'Taiwan': 104, 'Luxembourg': 257,
					   'India': 363, 'South Korea': 33, 'Iran': 8, 'Nigeria': 1, 'Mexico': 17, 'Italy': 79,
					   'Iceland': 601, 'Thailand': 125, 'Turkey': 63, 'Finland': 40, 'Portugal': 100, 'France': 391,
					   'Democratic Republic of the Congo': 126, 'Japan': 130, 'Germany': 201, 'New Zealand': 8,
					   'Sweden': 163, 'Belgium': 509, 'Switzerland': 76, 'Spain': 497, 'Jordan': 28, 'Russia': 207,
					   'Brazil': 95, 'Sri Lanka': 4, 'Malaysia': 16, 'Colombia': 86, 'Qatar': 16, 'Lithuania': 3,
					   'Hungary': 32, 'Kuwait': 8, 'Georgia': 15, 'Argentina': 29, 'Panama': 1, 'Poland': 27,
					   'Croatia': 7, 'Vietnam': 28, 'Saudi Arabia': 130, 'Canada': 208, 'Philippines': 13,
					   'Slovakia': 4, 'Egypt': 2, 'Uruguay': 11, 'Peru': 2, 'Ireland': 18, 'Chile': 144, 'Latvia': 25,
					   'Singapore': 159, 'Gambia': 3, 'Slovenia': 5, 'Denmark': 627, 'Korea': 3, 'South Africa': 20,
					   'Estonia': 5, 'Algeria': 3, 'Greece': 99, 'Senegal': 23, 'Austria': 250, 'Belarus': 2,
					   'Norway': 48, 'Pakistan': 2, 'Ecuador': 4, 'Cambodia': 1, 'Nepal': 1, 'Ghana': 15,
					   'Israel': 222, 'Costa Rica': 6, 'Democratic Republic of the Congo ': 7,
					   'Kazakhstan': 4, 'United Arab Emirates': 25, 'Indonesia': 9, 'Brunei': 5,
					   'Serbia': 4, 'Myanmar': 1, 'Bangladesh': 20, 'Guam': 3, 'Romania': 3, 'Tunisia': 3,
					   'Lebanon': 10, 'Jamaica': 8, 'Uganda': 20}"""

'''country_names = [c for c in country_prot_number.keys()]
country_filter = df['Country'].isin(country_names)

new_df = df[country_filter]
'''
new_df = df
new_df['new_deaths'].fillna(0, inplace=True)
new_df['new_cases'].fillna(0, inplace=True)
new_df['new_tests'].fillna(0, inplace=True)


#print(new_df['new_cases_per_million'].describe())

new_df['new_cases_per_million'].fillna(
	(new_df['new_cases']/new_df['population'])*1000000, inplace=True
)
new_df['new_deaths_per_million'].fillna(
	(new_df['new_deaths']/new_df['population'])*1000000, inplace=True
)
new_df['total_cases_per_million'].fillna(
	(new_df['total_cases']/new_df['population'])*1000000, inplace=True
)
new_df['new_tests_per_thousand'].fillna(
	(new_df['new_tests']/new_df['population'])*1000, inplace=True
)
new_df['total_tests_per_thousand'].fillna(
	(new_df['total_tests']/new_df['population'])*1000, inplace=True
)

#print(new_df.isnull().sum())


#rounded_new_df = new_df.applymap(rounder_func)

"""
rounded_new_df.drop(
	columns=['new_tests_per_thousand', 'total_tests_per_thousand', 'total_tests', 'new_tests_smoothed',
			 'new_tests_smoothed_per_thousand',	'tests_units', 'new_tests'], inplace=True
)
"""
#print(rounded_new_df.describe())
#print(rounded_new_df.isnull().sum())

#rounded_new_df.to_csv('coronastats(14-06).txt', index=False, sep='\t')            #Checkpoint for filled_cleaned up data

####Adding smokers data###
"""
smokers_file = pd.read_csv(r'C:\#Users\BERK\PycharmProjects\CORONA_PROJECT\datasets\smokers.csv')
smoker_df = smokers_file.drop(columns=['maleSmokingRate','femaleSmokingRate', 'pop2020'])
smoker_df.rename(columns={'name':'Country', 'totalSmokingRate':'total_smokers'}, inplace=True)

#print(smoker_df)

rounded_new_df = rounded_new_df.merge(smoker_df, on='Country', how='outer')


def total_smoker_fill(df):
	'''
	Eksik 'total_smoker' datasını doldurabilmek için male/female_smoker ve alttaki sex_ratios dictionary'sinden yararlanır.
	:param df: total_smoker'la birleştirilmiş corona datası
	:return: filled total_smoker df
	'''
	sex_ratios = { #country: males/females
		'Algeria': 102.1/100, 'Czech Rebuplic': 96.984/100, 'Gambia': 98.393/100, 'Guam': 101.79/100,
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
cigarated_df = cigarated_df.applymap(rounder_func)
cigarated_df['iso_code'].dropna(inplace=True)

print(cigarated_df['total_smokers'].describe())
print(cigarated_df['total_smokers'].isna().sum())
print(cigarated_df)

cigarated_df.to_csv('Smoker_14-06_added_coronadf.txt', index=False, sep='\t')			#Checkpoint    smokers"""

### All filled data ####

from Scripts import manipulation_functions as mf


data_file = r'/CleanData_eskisürüm.txt'
dataframe = pd.read_table(data_file)

coronastats_df = dataframe.applymap(mf.rounder_func)


#coronastats_df.to_csv('CC_Data.txt', sep='\t', index=False)