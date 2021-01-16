from typing import Dict, Any

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import missingno as msno
from sklearn import preprocessing
import seaborn as sb
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.stats.outliers_influence import variance_inflation_factor
pd.options.mode.chained_assignment = None

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 30)

data = "FINAL_WEEKLY_ALL1806.csv"
df = pd.read_table(data)
df = df.drop(["identity percentage for NSP1", "identity percentage for NSP2", "identity percentage for NSP3", "identity percentage for NSP4",
              "identity percentage for NSP5", "identity percentage for NSP6", "identity percentage for NSP7", "identity percentage for NSP8",
              "identity percentage for NSP9", "identity percentage for NSP10", "identity percentage for NSP11", "identity percentage for NSP12",
              "identity percentage for NSP13", "identity percentage for NSP14", "identity percentage for NSP15", "identity percentage for NSP16",
              "identity percentage for Spike", "identity percentage for NS3", "identity percentage for E", "identity percentage for M",
              "identity percentage for NS6","identity percentage for NS7a","identity percentage for NS7b","identity percentage for NS8","identity percentage for N"], axis=1)
#df.set_index('country', inplace=True)

'''normalization of data'''
def normalize(df):
    new_df = df.select_dtypes(include=['float64', 'int64'])
    column_names = new_df.columns
    new_df = new_df.astype(float)
    new_df = preprocessing.normalize(new_df)
    new_df = pd.DataFrame(new_df, columns=column_names)
    return new_df

n_df = normalize(df)

def outlier_changer(column_name):
    '''
    Takes a column name as str, transforms outliers with upper and lower limits and forms new data frame column without outliers
    :param column_name: required column name on df
    :return: modified data column without outliers
    '''
    # df['index'] = np.arange(0,5512)
    # df.reset_index(inplace= True)
    # df.set_index('index', inplace=True)
    new_df = df.select_dtypes(include=['float64', 'int64'])
    df_column = new_df[column_name].dropna()

    #plot = sns.boxplot(x=df_column)
    #fig = plot.get_figure()
    #fig.savefig(f'boxplot_{column_name}.png')
    #plt.show()
    Q1 = df_column.quantile(0.25)
    Q3 = df_column.quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    aykiri_lower = (df_column < lower_limit)
    aykiri_upper = (df_column > upper_limit)
    df_column[(aykiri_lower)] = lower_limit
    df_column[(aykiri_upper)] = upper_limit
    # print(df_column)
    return df_column

df_wp = n_df.drop(["score for NSP1", "score for NSP2", "score for NSP3", "score for NSP4",
              "score for NSP5", "score for NSP6", "score for NSP7", "score for NSP8",
              "score for NSP9", "score for NSP10", "score for NSP11", "score for NSP12",
              "score for NSP13", "score for NSP14", "score for NSP15", "score for NSP16",
             "score for Spike", "score for NS3", "score for E", "score for M",
              "score for NS6", "score for NS7a", "score for NS7b", "score for NS8", "score for N"], axis=1)

for column_name in list(df_wp.columns):
    df_wp[column_name] = outlier_changer(column_name)

n_df['country'] = df['country']
n_df['date'] = df['date']
n_df['GISAID_clade'] = df['GISAID_clade']
n_df['iso_code'] = df['iso_code']
df = n_df

#for k,v in df_dic.items():
#    for column_name in list(v.columns):
#        if column_name == 'GISAID_clade' or column_name == 'country' or column_name == 'iso_code' or column_name == 'date': continue
#        v[column_name] = outlier_changer(v, column_name)

#df['death_growth_rate'] = outlier_changer(df, 'death_growth_rate')

#sns.boxplot(x=n_df['death_growth_rate'])
#plt.show()

def plotter(df):
    '''
    :param column: takes column name for ex. 'score for NSP1'
    :return: returns plot with death growth rate changed by column
    '''
    for column_name in list(df.columns):
        #sns_plot = sns.pairplot(df, hue='death_growth_rate', height=2.5)
        #sns_plot.savefig(f'death_growth_{column_name}_plot.png')
        plot = sns.jointplot(x=column_name, y='death_growth_rate', data=df, kind="reg")
        #fig= plot.get_figure()

        plot.savefig(f'death_growth_{column_name}_plot.png')
        #plt.show()

    return

#plotter(df_wp)

'''creating a correlation matrix with df without protein scores'''
def protein_parser(df):
    score_column_dic = {}
    for column_name in list(df.columns):
        if 'score' in column_name:
            score_column_dic[column_name] = df[column_name]
    return score_column_dic

prot_score_dic = protein_parser(df)

def corr_matrix(df):
    corrMatrix = df.corr()
    print(corrMatrix)
    plot= sb.heatmap(corrMatrix,
                xticklabels=corrMatrix.columns,
                yticklabels=corrMatrix.columns,
                cmap='RdBu_r',
                annot=True,
                linewidth=0.5)
    fig = plot.get_figure()
    fig.savefig(f'corrMatrix.png')
    plt.show()
    return

'''after correlation analysis'''
df_wp = df_wp.drop(['total_cases', 'total_deaths', 'total_cases_per_million', 'total_deaths_per_million',
                    'new_cases', 'new_deaths', 'health_expenditure(%ofGDP)','HDI','median_age', 'log_population',
                    'new_cases_per_million', 'hospital_beds_per_1k'], axis=1)
#corr_matrix(df_wp)

'''clade'lere ayırıp fit etme'''
dfs = df.groupby('GISAID_clade')
df_G = dfs.get_group('G')
df_GH = dfs.get_group('GH')
df_GR = dfs.get_group('GR')
df_L = dfs.get_group('L')
df_O = dfs.get_group('O')
df_S = dfs.get_group('S')
df_V = dfs.get_group('V')
df_dic = {'G_clade': df_G, 'GH_clade': df_GH, 'GR_clade': df_GR, 'L_clade': df_L, 'O_clade': df_O, 'S_clade': df_S, 'V_clade': df_V}
df = df.drop(['total_cases', 'total_deaths', 'total_cases_per_million', 'total_deaths_per_million',
                    'new_cases', 'new_deaths', 'health_expenditure(%ofGDP)','HDI','median_age', 'log_population',
                    'new_cases_per_million', 'hospital_beds_per_1k'], axis=1)


def fitter(df_t, clade, protein):
    '''
    :param df: takes a clade or country df
    :return: prints model summary
    '''
    global df
    df_t = df_t.select_dtypes(include=['float64', 'int64'])
    df_t[f'score for {protein}'] = df[f'score for {protein}']
    X = df_t.drop('death_growth_rate', axis=1)
    y = df_t[['death_growth_rate']]
    # print(X.head())
    # print(y.head())
    lm = sm.OLS(y, X)
    model = lm.fit()
    print(model.summary())
    #print('Parameters: ', model.params)
    print('R2: ', model.rsquared)
    print('P-value: ', model.f_pvalue)

    '''applys linearity test and saves'''
    fig, ax = plt.subplots(1, 1)
    sns.residplot(model.predict(), y, lowess=True, scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax)
    ax.title.set_text('Residuals vs Fitted')
    ax.set(xlabel='Fitted', ylabel='Residuals')
    fig.savefig(f'linearity_test_{clade}_{protein}.png')
    plt.show()
    plt.close(fig)
    return model, clade, protein

def error_distribution (model, clade, protein):
    '''applys distribution of errors test and saves: should be normally distributed'''
    fig, ax = plt.subplots(1, 1)
    sm.ProbPlot(model.resid).qqplot(line='s', color='#1f77b4', ax=ax)
    ax.title.set_text('QQ Plot')
    fig.savefig(f'error_dist_test_{clade}_{protein}.png')
    plt.show()
    plt.close(fig)
    return

def homoscedasticity(model, clade, protein):
    '''applys homoscedastic test and saves: should be cone-shaped curve'''
    fig, ax = plt.subplots(1, 1)
    standardized_resid1 = np.sqrt(np.abs(model.get_influence().resid_studentized_internal))
    sns.regplot(model.predict(), standardized_resid1, color='#1f77b4', lowess=True, scatter_kws={'alpha': 0.5},
                line_kws={'color': 'red'}, ax=ax)
    ax.title.set_text('Scale Location')
    ax.set(xlabel='Fitted', ylabel='Standardized Residuals')
    fig.savefig(f'homoscedasticity_{clade}_{protein}.png')
    plt.show()
    plt.close(fig)
    return

#df_wp['score for M'] = df['score for M']
#fitter(df_wp, 'no', 'M')
#print(error_distribution(m, c, p))
#print(homoscedasticity(m, c, p))
#fitter(df_G,  'G', 'M')
#fitter(df_GH)
#fitter(df_GR)
#fitter(df_L)
#fitter(df_O)
#fitter(df_S)
#fitter(df_V)

'''ülkelere ayırıp fit etme: çalışmadı'''
#dfs2 = df.groupby('iso_code')
#fitter(dfs2.get_group('ARG')) ok but no valid value
#fitter(dfs2.get_group('AUS')) ok but no valid value
#fitter(dfs2.get_group('AUT')) na
#fitter(dfs2.get_group('BEL')) 0.00??
#fitter(dfs2.get_group('BRA')) ok but no valid value
#fitter(dfs2.get_group('CAN')) 0.00??
#fitter(dfs2.get_group('COL')) na
#fitter(dfs2.get_group('DNK')) na
#fitter(dfs2.get_group('FRA')) na

def calc_vif(df):
    # Calculating VIF (multicollinearity)
    '''
    :param df: takes df to calculate multicollinearity via df
    :return:
    '''
    gr_column = df['death_growth_rate']
    df = df.drop('death_growth_rate', axis=1)
    df['death_growth_rate'] = gr_column
    n_df = df.select_dtypes(include=['float64', 'int64'])
    X = n_df.iloc[:, :-1]

    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif)
    return (vif)

#calc_vif(df_L)




