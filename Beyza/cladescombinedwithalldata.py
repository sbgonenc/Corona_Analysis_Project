# -*- coding: utf-8 -*-
"""cladescombinedwithalldata.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pF6rg00B7wyfAP5YtKt6oATUYrRM-Ein

all clades data from metadata ' mode '
the most frequent clade in a week and a month 
first in the array is placed in place of the clades that have the same frequency.
empty dates are filled with the data with the most frequent clade of the country in total.
"""

import pandas as pd
import numpy as np
df = pd.read_csv("../metadata.tsv", sep="\t")
df.drop(df.columns.difference(['country','date','GISAID_clade']), 1, inplace=True)
dffillin= df.drop(df.columns.difference(['country','GISAID_clade']), 1)
cladecount=dffillin.groupby(['country'])['GISAID_clade'].agg(pd.Series.mode).to_frame()
dixt=cladecount.to_dict()
for key, value in dixt['GISAID_clade'].items():
	if isinstance(value, np.ndarray):
		dixt['GISAID_clade'][key]= value[0]
df2 = pd.read_table("weekly_clades_dgr_17_06.txt")
df2.drop(['Lineage','Clade'],1 ,inplace=True)
df2.rename(columns={'Country':'country'}, inplace=True)
df2["date"] = pd.to_datetime(df2['date'])
df.replace('2020-03-XX', '01.03.2020', inplace=True)
df.replace('2020-02-XX', '01.02.2020', inplace=True)
df.replace('2020-01-XX', '01.01.2020', inplace=True)
df.replace('2020-01-00', '01.01.2020', inplace=True)
df.replace('2020-02-00', '01.02.2020', inplace=True)
df.replace('2020-02', '01.02.2020', inplace=True)
df.replace('2020-03', '01.03.2020', inplace=True)
df.replace('2020-03-00', '01.03.2020', inplace=True)
df.replace('2020-00-00', '01.01.2020', inplace=True)
df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
df["date"] = pd.to_datetime(df['date'])
weekly_group=df.groupby(['country', pd.Grouper(key='date', freq='W')])['GISAID_clade'].agg(pd.Series.mode).to_frame()
dffinal=pd.merge(weekly_group, df2, on=['country','date'], how='outer')
dffinal['GISAID_clade']=dffinal['GISAID_clade'].str[0]
dffinal.GISAID_clade.fillna(dffinal.country.map(dixt['GISAID_clade']), inplace=True)
dffinal.dropna(inplace=True)
dffinal.to_csv('FINAL_WEEKLY_ALL1806.csv', sep='\t', index=False)
#######################################################################################
import pandas as pd
import numpy as np
df = pd.read_csv("../metadata.tsv", sep="\t")
df.drop(df.columns.difference(['country','date','GISAID_clade']), 1, inplace=True)
dffillin= df.drop(df.columns.difference(['country','GISAID_clade']), 1)
cladecount=dffillin.groupby(['country'])['GISAID_clade'].agg(pd.Series.mode).to_frame()
dixt=cladecount.to_dict()
for key, value in dixt['GISAID_clade'].items():
	if isinstance(value, np.ndarray):
		dixt['GISAID_clade'][key]= value[0]
df2 = pd.read_table("monthly_clades_dgr_17_06.txt")
df2.drop(['Lineage','Clade'],1 ,inplace=True)
df2.rename(columns={'Country':'country'}, inplace=True)
df2["date"] = pd.to_datetime(df2['date'])
df.replace('2020-03-XX', '01.03.2020', inplace=True)
df.replace('2020-02-XX', '01.02.2020', inplace=True)
df.replace('2020-01-XX', '01.01.2020', inplace=True)
df.replace('2020-01-00', '01.01.2020', inplace=True)
df.replace('2020-02-00', '01.02.2020', inplace=True)
df.replace('2020-02', '01.02.2020', inplace=True)
df.replace('2020-03', '01.03.2020', inplace=True)
df.replace('2020-03-00', '01.03.2020', inplace=True)
df.replace('2020-00-00', '01.01.2020', inplace=True)
df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
df["date"] = pd.to_datetime(df['date'])
weekly_group=df.groupby(['country', pd.Grouper(key='date', freq='M')])['GISAID_clade'].agg(pd.Series.mode).to_frame()
dffinal=pd.merge(weekly_group, df2, on=['country','date'], how='outer')
dffinal['GISAID_clade']=dffinal['GISAID_clade'].str[0]
dffinal.GISAID_clade.fillna(dffinal.country.map(dixt['GISAID_clade']), inplace=True)
dffinal.dropna(inplace=True)
dffinal.to_csv('FINAL_MONTHLY_ALL1806.csv', sep='\t', index=False)
####################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df1 = pd.read_table('All_normalized_M2206.txt', index_col=False)
df2 = pd.read_table('All_normalized_W2206.txt', index_col=False)
df3 = pd.read_table('Monthly_cladeCount_18_06.txt', index_col=False)
df4 = pd.read_table('weekly_cladeCount_dgr_18_06.txt')
df1.reset_index()
df2.reset_index()
df3.reset_index()
df4.reset_index()
isocode=pd.read_csv('../FINAL_WEEKLY_ALL1806.csv', sep='\t', index_col=False)

isocode.drop(isocode.columns.difference(['country', 'iso_code']), 1, inplace=True)
isocode.set_index('iso_code',inplace=True)
dix=isocode.to_dict()
df3['country']=df3['iso_code'].map(dix['country'])
df4['country']=df4['iso_code'].map(dix['country'])

df13month=pd.merge(df1, df3[['country','date']],left_index=True, right_index=True, how='right')
df24week=pd.merge(df2, df4[['country', 'date']], left_index=True, right_index=True, how='right')

df13month.to_csv('MonthlyNormalizedwithcountries.csv')
df24week.to_csv('WeeklyNormalizedwithcountries.csv')
####################################################

from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2


def get_code(col):
    try:
        cn_a2_code = country_name_to_country_alpha2(col)
    except:
        cn_a2_code = 'Unknown'
    return cn_a2_code


def get_cont(col):
    try:
        cn_continent = country_alpha2_to_continent_code(str(col))
    except:
        cn_continent = 'Unknown'
    return cn_continent


# function to get longitude and latitude data from country name
from geopy.geocoders import Nominatim

geolocator = Nominatim()


def latit(country):
    try:
        # Geolocate the center of the country
        loc = geolocator.geocode(country)
        # And return latitude and longitude
        return (loc.latitude)
    except:
        # Return missing value
        return np.nan


def longite(country):
    try:
        # Geolocate the center of the country
        loc = geolocator.geocode(country)
        # And return latitude and longitude
        return (loc.longitude)
    except:
        # Return missing value
        return np.nan


df = pd.read_csv('../FINAL_WEEKLY_ALL1806.csv', sep='\t', index_col=False)
data_countries = []
code_stat = df['country'].unique()
set1 = set(data_countries)
set2 = set(code_stat)
any_new = set2 - set1
data_countries += list(any_new)
count_codes = [get_code(x) for x in data_countries]
dixtofcodes = dict(zip(data_countries, count_codes))
continents = [get_cont(x) for x in count_codes]
dixtofconts = dict(zip(data_countries, continents))
df['code'] = df['country'].map(dixtofcodes)
df['continent'] = df['country'].map(dixtofconts)
latitudeslist = [latit(x) for x in data_countries]
dixtoflat = dict(zip(data_countries, latitudeslist))
longitudeslist = [longite(x) for x in data_countries]
dixtoflong = dict(zip(data_countries, longitudeslist))
df['Latitude'] = df['country'].map(dixtoflat)
df['Longitude'] = df['country'].map(dixtoflong)