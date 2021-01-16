import pandas as pd
from Scripts import manipulation_functions as mf
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 180)


file = 'Cleaned_clade_combined_17_06.txt'  ### Means
#file = 'd2d_merged_medians(PAM40)(16-06).txt'  ### Medians

combined_df = pd.read_table(file)
combined_df['date'] = pd.to_datetime(combined_df['date'], yearfirst=True)
combined_df['gdp_per_capita'] = combined_df['gdp_per_capita'].astype(float)

fltr = combined_df['total_cases'] != 0

combined_df = combined_df.loc[fltr]
combined_df.drop(['aged_70_older'], axis=1, inplace=True)
combined_df.drop(['extreme_poverty'], axis=1, inplace=True)
combined_df.drop(["handwashing_facilities"], axis = 1, inplace =True)


combined_df['new_deaths'].fillna(0, inplace=True)
combined_df['new_cases'].fillna(0, inplace=True)
combined_df["female_smokers"].loc['Jordan'] = 10.7
combined_df["male_smokers"].loc['Jordan'] = 70.2

combined_df["female_smokers"].loc['Democratic Republic of Congo'] = 0.9
combined_df["male_smokers"].loc['Democratic Republic of Congo'] = 14.1

combined_df['stringency_index'] = combined_df['stringency_index'].groupby(combined_df.index).ffill().bfill()

drop_index = combined_df[combined_df['iso_code'].isna()].index
combined_df = combined_df.drop(drop_index)


combined_df['total_deaths_per_million'].fillna((combined_df['total_deaths']/combined_df['population'])*1000000, inplace=True)
combined_df['new_deaths_per_million'].fillna((combined_df['new_deaths']/combined_df['population'])*1000000, inplace=True)


combined_df= combined_df.applymap(mf.rounder_func)


#combined_df.to_csv(to_write, index=False, sep='\t')   #### Checkpoint

###Smoker fill

'''FROM data_imputation.py -- Dila'''
gdp_data = r"C:\Users\BERK\PycharmProjects\CORONA_PROJECT\Current health expenditure (% of GDP).txt"

def parser(text):
    with open(text) as infh:
        result = {}
        for line in infh:
            if line.startswith('Country'): continue
            boluk = line.split('\t')
            country = boluk[0]
            most_recent_year = boluk[1]
            most_recent_value = str(boluk[2])
            if most_recent_year == '':
                pass
            elif int(most_recent_year) >= 2017:
                result[country] = most_recent_value
    return result

result = parser(gdp_data)
df_gdp = pd.DataFrame.from_dict(result, orient='index')
df_gdp.reset_index(inplace= True)
df_gdp.rename(columns={'index' : 'Country', 0 : 'health_expenditure(%ofGDP)'}, inplace=True)
df_gdp.set_index('Country', inplace=True)

combined_df = pd.merge(combined_df, df_gdp, how = 'left', on= 'Country')


Congo_smokers = float((3130000 /89561404)*100).__round__(3) #total number of smokers / congo population *100 to percentage  ## 2015 data: ''' https://files.tobaccoatlas.org/wp-content/uploads/pdf/dem-rep-of-congo-country-facts-en.pdf '''
Peru_smokers = 4.80 #2016 data: '''https://www.macrotrends.net/countries/PER/peru/smoking-rate-statistics'''


cntry_fltr = combined_df['Country'] == 'Peru'
combined_df.loc[cntry_fltr, 'total_smokers'] = Peru_smokers

cntry_fltr = combined_df['Country'] == 'Democratic Republic of Congo'
combined_df.loc[cntry_fltr, 'total_smokers'] = Congo_smokers
combined_df.loc[cntry_fltr, 'hospital_beds_per_100k'] = 1.6 # 2005 data: '''https://data.worldbank.org/indicator/SH.MED.BEDS.ZS?locations=CG'''
combined_df.loc[cntry_fltr, 'health_expenditure(%ofGDP)'] = 4.0 # 2017 data: '''https://knoema.com/atlas/Democratic-Republic-of-the-Congo/Health-expenditure-as-a-share-of-GDP'''

cntry_fltr = combined_df['Country'] == 'Senegal'
combined_df.loc[cntry_fltr, 'hospital_beds_per_100k'] = 0.3 #2003 data '''https://data.worldbank.org/indicator/SH.MED.BEDS.ZS?locations=CG'''


cleaned_combined = combined_df
cleaned_combined.drop(columns=['male_smokers', 'female_smokers'], inplace=True)
cleaned_combined.rename(columns={"hospital_beds_per_100k": "hospital_beds_per_1k"},  inplace=True)


cleaned_combined['health_expenditure(%ofGDP)'] = cleaned_combined['health_expenditure(%ofGDP)'].astype(float)
print(cleaned_combined['health_expenditure(%ofGDP)'].dtype)

cleaned_combined['HealthExpenseperGDP'].fillna(
    cleaned_combined['gdp_per_capita']*cleaned_combined['health_expenditure(%ofGDP)']/100, inplace= True
)
cleaned_combined['HealthExpenseperGDP'] = cleaned_combined['HealthExpenseperGDP'].apply(mf.rounder_func)

print(cleaned_combined.isna().sum())

fltr=cleaned_combined['health_expenditure(%ofGDP)'].isnull()
print(cleaned_combined.loc[fltr, ['Country', 'health_expenditure(%ofGDP)']])

cntry_fltr = (combined_df['Country'] == 'Guam') | (combined_df['Country'] == 'Taiwan')
combined_df.drop(combined_df[cntry_fltr].index, inplace=True)

to_write= 'CLEANED_Claded_d2d_means(PAM40)(17.06).txt'   ##### Mean or Median??
combined_df.to_csv(to_write, index=False, sep='\t')   #### Checkpoint