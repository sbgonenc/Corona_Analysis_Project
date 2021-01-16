import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_table('All_normalized_M2206.txt', index_col=False)
df2 = pd.read_table('All_normalized_W2206.txt', index_col=False)
df3 = pd.read_table('Monthly_cladeCount_18_06.txt', index_col=False)
df4 = pd.read_table('weekly_cladeCount_dgr_18_06.txt')
df1.reset_index()
df2.reset_index()
df3.reset_index()
df4.reset_index()
isocode=pd.read_csv('FINAL_WEEKLY_ALL1806.csv', sep='\t', index_col=False)

isocode.drop(isocode.columns.difference(['country', 'iso_code']), 1, inplace=True)
isocode.set_index('iso_code',inplace=True)
dix=isocode.to_dict()
df3['country']=df3['iso_code'].map(dix['country'])
df4['country']=df4['iso_code'].map(dix['country'])

df13month=pd.merge(df1, df3[['country','date']],left_index=True, right_index=True, how='right')
df24week=pd.merge(df2, df4[['country', 'date']], left_index=True, right_index=True, how='right')

df13month.to_csv('MonthlyNormalizedwithcountries.csv')
df24week.to_csv('WeeklyNormalizedwithcountries.csv')
df13month.drop(df13month.columns.difference(['country','date','score for NSP1', 'score for NSP2', 'score for NSP3', 'score for NSP4', 'score for NSP5', 'score for NSP6', \
              'score for NSP7', 'score for NSP8', 'score for NSP9', 'score for NSP10', 'score for NSP11', 'score for NSP12',\
              'score for NSP13', 'score for NSP14', 'score for NSP15', 'score for NSP16', 'score for Spike', 'score for NS3', \
              'score for E', 'score for M', 'score for NS6', 'score for NS7a', 'score for NS7b', 'score for NS8', 'score for N']),1, inplace=True)
for f in set(df13month['country']):
    df13month[df13month['country']==f].plot(x='date', title=f)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(f'{f}.png', dpi=100)
    plt.show()