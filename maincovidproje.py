import tkinter.ttk as ttk
import tkinter as tk
from tkinter import font as tkfont
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import missingno as msno
from sklearn import preprocessing
import seaborn as sb
from statsmodels.regression.linear_model import RegressionResults

from statsmodels.stats.outliers_influence import variance_inflation_factor
pd.options.mode.chained_assignment = None
import statsmodels.api as sm
import numpy as np
matplotlib.use("TkAgg")
from datetime import datetime
from Berk import analysis_functions
from Berk import Machine_Learning
import linear_analysis_Dila
df=pd.read_table('All_normalized_M2206.txt')
df1 = pd.read_table(r'Weekly_eliminateddf_2206.txt')

def calc_vif(self):
    # Calculating VIF (multicollinearity)
    '''
    :param df: takes df to calculate multicollinearity via df
    :return:
    '''
    df=pd.read_csv(f'{self.combo.get()}.csv', sep='|t')
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
def fitter(self):
    '''
    :param df: takes a clade or country df
    :return: prints model summary
    '''
    protein = self.combo.get()
    clade = self.combo2.get()
    
    global df
    var= analysis_functions.df_summer('single_clades', 'w')
    flt = var[clade] == 1
    var[clade].loc[flt]
    df = var[clade].loc[flt]
    filtre=df['']
    df = df.drop(
        ['score for NSP1', 'score for NSP2', 'score for NSP3', 'score for NSP4', 'score for NSP5', 'score for NSP6', \
         'score for NSP7', 'score for NSP8', 'score for NSP9', 'score for NSP10', 'score for NSP11', 'score for NSP12', \
         'score for NSP13', 'score for NSP14', 'score for NSP15', 'score for NSP16', 'score for Spike', 'score for NS3', \
         'score for E', 'score for M', 'score for NS6', 'score for NS7a', 'score for NS7b', 'score for NS8',
         'score for N', \
         ], axis=1)
    df_t = analysis_functions.clade_remover(df)
    df_t = df_t.select_dtypes(include=['float64', 'int64'])
    df_t[f'score for {protein}'] = df[f'score for {protein}']
    X = df_t.drop('death_growth_rate', axis=1)
    y = df_t[['death_growth_rate']]
    # print(X.head())
    # print(y.head())
    lm = sm.OLS(y, X)
    model = lm.fit()
    print(model.summary())
    # print('Parameters: ', model.params)
    print('R2: ', model.rsquared)
    print('P-value: ', model.f_pvalue)
    
    '''applys linearity test and saves'''
    fig, ax = plt.subplots(1, 1)
    sns.residplot(model.predict(), y, lowess=True, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=ax)
    ax.title.set_text('Residuals vs Fitted')
    ax.set(xlabel='Fitted', ylabel='Residuals')
    fig.savefig(f'linearity_test_{clade}_{protein}.png')
    plt.show()
    plt.close(fig)
    window=tk.Tk()
    label=tk.Label(window, text="Current Port")
    port = tk.Text(window)
    port.tk.insert(model, clade, protein)


def remote(entry):
    hostname = entry.get()
    get_plot(hostname)
    Machine_Learning.country_lm(hostname,df1)
    
def f(x):
    return x[2] / x[3] * 100

def get_plot(country):
    country = country.upper()
    df1 = pd.read_csv("metadata.tsv", sep="\t")
    df1.drop(df1.columns.difference(['country', 'date', 'GISAID_clade']), 1, inplace=True)
    country_group = df1.groupby('country')['GISAID_clade']
    cladecount = country_group.value_counts().reset_index(name="count")
    cladecount['sum'] = cladecount.groupby(['country'])['count'].transform(sum)
    cladecount['percentage'] = cladecount.apply(f, axis=1)
    cladecount.rename(columns={'GISAID_clade': 'Clade'}, inplace=True)
    cladecount['country'] = cladecount['country'].apply(lambda x: x.upper())
    df = cladecount[cladecount['country'] == country]
    barWidth = 1
    r = [1]
    per = {}
    clades = ['GH', 'GR', 'G', 'V', 'L', 'O', 'S']
    cmap = plt.cm.get_cmap('RdYlGn', 7)
    for c in clades:
        df2 = pd.DataFrame({"country": country, "Clade": c, "count": [0], "sum": None, "percentage": [0]})
        if not df.isin([c]).any().any():
            df = df.append(df2)
            df['sum'].ffill(inplace=True)
        per[c] = df[df['Clade'] == c]['percentage'].to_numpy()[0]
    d = sorted(per, key=per.get, reverse=True)
    
    i = plt.bar(r, per[d[0]], color=cmap(1), edgecolor='white', width=barWidth, label=d[0])
    j = plt.bar(r, per[d[1]], bottom=[per[d[0]]], color='#008E42', edgecolor='white', width=barWidth,
                label=d[1])
    k = plt.bar(r, per[d[2]], bottom=[per[d[0]] + per[d[1]]], color='#FEFF18', edgecolor='white',
                width=barWidth,
                label=d[2])
    l = plt.bar(r, per[d[3]], bottom=[per[d[0]] + per[d[1]] + per[d[2]]], color='#0D7CFF', edgecolor='white',
                width=barWidth, label=d[3])
    m = plt.bar(r, per[d[4]], bottom=[per[d[0]] + per[d[1]] + per[d[2]] + per[d[3]]], color='#9DFFFD',
                edgecolor='white', width=barWidth, label=d[4])
    n = plt.bar(r, per[d[5]], bottom=[per[d[0]] + per[d[1]] + per[d[2]] + per[d[3]] + per[d[4]]],
                color='#FFA346',
                edgecolor='white', width=barWidth, label=d[5])
    p = plt.bar(r, per[d[5]], bottom=[per[d[0]] + per[d[1]] + per[d[2]] + per[d[3]] + per[d[4]] + per[d[5]]],
                color='#9E91FF', edgecolor='white', width=barWidth, label=d[6])
    plt.xticks(r, [country])
    plt.xlabel('country')
    plt.ylim(-10, 110)
    plt.suptitle('Sars-cov-2 Clades Distribution in Percentage', x=0.515, y=0.950, fontsize=12, style='italic',
                 fontname='DejaVu Sans')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    count = 0
    for rect in i + j + k + l + m + n + p:
        bl = rect.get_xy()
        height = 0.5 * rect.get_height() + bl[1]
        width = 0.5 * rect.get_width() + bl[0]
        if int(per[d[count]]) > 8:
            plt.text(width, height, str(round(per[d[count]], 1)) + '%', fontsize=12, ha='center', va='bottom')
        count = count + 1
    plt.show()

def functionofbettermap():
    df = pd.read_csv('FINAL_WEEKLY_ALL1806.csv', sep='\t', index_col=False)
    df.drop(df.columns.difference(['country', 'date', 'iso_code', 'cvd_death_rate']), 1, inplace=True)
    
    fig = px.scatter_geo(
        df,
        locations='iso_code',
        color='country',
        hover_name='country',
        size='cvd_death_rate',
        projection="natural earth",
        title=f'World COVID-19 Death Rates'
    )
    fig.show()

def functionofmap():
    df = pd.read_csv('coordinates.csv', sep='\t', index_col=False)
    fig = px.density_mapbox(df, lat=df['Latitude'], lon=df['Longitude'], z=df['death_growth_rate'], radius=10,
                            animation_frame="date")
    fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=10, mapbox_center={"lat": 38.822591, "lon": -1.117050})
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 600
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 600
    fig.layout.coloraxis.showscale = True
    fig.layout.sliders[0].pad.t = 10
    fig.layout.updatemenus[0].pad.t = 10
    
    fig.show()
    
def GraphFunc(self):
    df = pd.read_csv('WeeklyNormalizedwithcountries.csv', sep=',', index_col=False)
    entry = self.combo.get()
    df.drop(df.columns.difference([entry, 'cvd_death_rate']), 1, inplace=True)
    df.plot.scatter(x=entry, y='cvd_death_rate', title=f'{entry} versus death rate')
    plt.show()
    
HEIGHT = 600
WIDTH = 700
LARGEFONT = ("Helvetica", 30)

class tkinterApp(tk.Tk):
    
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        menubar = tk.Menu(container)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=quit)
        menubar.add_cascade(label="File", menu=filemenu)
        x = (self.winfo_screenwidth() - self.winfo_reqwidth()) / 2
        y = (self.winfo_screenheight() - self.winfo_reqheight()) / 2
        self.geometry("+%d+%d" % (x, y))
        self.bind('<Escape>', lambda e: self.destroy())
        tk.Tk.config(self, menu=menubar)

        for F in (StartPage, Page1, Page2, Page3, Page4, Page5, Page6, Page7, Page8):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            center_frame = tk.Frame(frame, relief='raised', borderwidth=2)
            center_frame.place(relx=0.5, rely=0.5)
        self.show_frame(StartPage)
    
    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        
        self.background_image = tk.PhotoImage(file='landscape.png')
        background_label = tk.Label(self, image=self.background_image)
        background_label.image = self.background_image
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        label = ttk.Label(self, text="Please Choose!", font=LARGEFONT)
        label.grid(row=0, column=1, padx=10, pady=10)
        
        button1 = ttk.Button(self, text="Graphs",
                             command=lambda: controller.show_frame(Page1))

        button1.grid(row=2, column=1, padx=10, pady=10)
        
        button2 = ttk.Button(self, text="Maps",
                             command=lambda: controller.show_frame(Page2))

        button2.grid(row=2, column=2, padx=10, pady=10)

        button3 = ttk.Button(self, text="Make your own analysis",
                             command=lambda: controller.show_frame(Page6))
        

        button3.grid(row=2, column=3, padx=10, pady=10)
        
        button4 = ttk.Button(self, text="Correlation Matrix",
                              command=lambda: controller.show_frame(Page8))
        button4.grid(row=2, column=4, padx=10, pady=10)
        
class Page1(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.background_image = tk.PhotoImage(file='landscape.png')
        background_label = tk.Label(self, image=self.background_image)
        background_label.image = self.background_image
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        label = ttk.Label(self, text="Graphs by clades, proteins or countries?", font=LARGEFONT)
        label.grid(row=0, column=1, padx=10, pady=10)
        
        button1 = ttk.Button(self, text="Go back to MainPage",
                             command=lambda: controller.show_frame(StartPage))
        button1.grid(row=1, column=1, padx=10, pady=10)
    
        button2 = ttk.Button(self, text="Go back to Maps",
                             command=lambda: controller.show_frame(Page2))
    
        button2.grid(row=2, column=1, padx=10, pady=10)
        
        button3 = ttk.Button(self, text="Graphs of Proteins and Clades",
                             command=lambda: controller.show_frame(Page3))

        button3.grid(row=1, column=2, padx=10, pady=10)

        button4 = ttk.Button(self, text="Analysis by country",
                             command=lambda: controller.show_frame(Page4))
        
        button4.grid(row=2, column=2, padx=10, pady=10)
        button5 = ttk.Button(self, text="Regression",
                             command=lambda: controller.show_frame(Page5))
        button5.grid(row=2, column=3, padx=10, pady=10)
        
        button6= ttk.Button(self, text="Collinearity",
                             command=lambda: controller.show_frame(Page6))
        button6.grid(row=1, column=3, padx=10, pady=10)


class Page2(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.background_image = tk.PhotoImage(file='landscape.png')
        background_label = tk.Label(self, image=self.background_image)
        background_label.image = self.background_image
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        label = ttk.Label(self, text="Maps by changing time or constant time?", font=LARGEFONT)
        label.grid(row=0, column=1, padx=10, pady=10)

        button1 = ttk.Button(self, text="Changing",
                             command=lambda: functionofmap())
        button1.grid(row=2, column=3, padx=10, pady=10)
        
        button3 = ttk.Button(self, text="Constant",
                             command=lambda: functionofbettermap())
        button3.grid(row=2, column=2, padx=10, pady=10)

        button2 = ttk.Button(self, text="Go back to MainPage",
                             command=lambda: controller.show_frame(StartPage))
        button2.grid(row=2, column=4, padx=10, pady=10)


class Page3(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.InitUI()
        button2 = ttk.Button(self, text="Go back to MainPage",
                                  command=lambda: controller.show_frame(StartPage))
        button2.grid(row=1, column=0, padx=10, pady=10)
        button3 = ttk.Button(self, text="Go back",
                             command=lambda: controller.show_frame(Page1))
        button3.grid(row=2, column=0, padx=10, pady=10)

    def InitUI(self):
        self.stringprot=tk.StringVar()
        self.background_image = tk.PhotoImage(file='landscape.png')
        background_label = tk.Label(self, image=self.background_image)
        background_label.image = self.background_image
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.combo = ttk.Combobox(self, width=15, textvariable=self.stringprot)
        self.combo['values'] = ('score for NSP1', 'score for NSP2', 'score for NSP3', 'score for NSP4', 'score for NSP5', 'score for NSP6', \
              'score for NSP7', 'score for NSP8', 'score for NSP9', 'score for NSP10', 'score for NSP11', 'score for NSP12',\
              'score for NSP13', 'score for NSP14', 'score for NSP15', 'score for NSP16', 'score for Spike', 'score for NS3', \
              'score for E', 'score for M', 'score for NS6', 'score for NS7a', 'score for NS7b', 'score for NS8', 'score for N', \
              'Clade_V' ,'Clade_GH','Clade_GR','Clade_G','Clade_O','Clade_L','Clade_S')
        self.combo.grid(column=1, row=0)
        self.label = ttk.Label(self, text="Which protein, or clade?")
        self.label.grid(column=0, row=0)
        self.button = ttk.Button(self, text="Go", command= lambda : GraphFunc(self))
        self.button.grid(column=1, row=1)
        

class Page4(tk.Frame):
    def __init__(self, parent, controller):  
        tk.Frame.__init__(self, parent)
        self.background_image = tk.PhotoImage(file='landscape.png')
        background_label = tk.Label(self, image=self.background_image)
        background_label.image = self.background_image
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        entry = ttk.Entry(self, font=40)
        
        entry.grid(column=1, row=2)
        label = ttk.Label(self, text="Percentage of Clades", font=LARGEFONT)
        label.grid(column=1, row=0)
       
        button1 = ttk.Button(self, text="Enter a country!",  command= lambda :remote(entry) )
        button1.grid(column=2, row=2)
        self.parent = parent
        button2 = ttk.Button(self, text="Go back to MainPage",
                             command=lambda: controller.show_frame(StartPage))
        button2.grid(row=2, column=4, padx=10, pady=10)
        button3 = ttk.Button(self, text="Go back",
                             command=lambda: controller.show_frame(Page1))
        button3.grid(row=2, column=0, padx=10, pady=10)

        self.winfo_toplevel().bind('<Return>',lambda self: remote(entry))

class Page5(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.InitUI()

        button2 = ttk.Button(self, text="Go back to MainPage",
                             command=lambda: controller.show_frame(StartPage))
        button2.grid(row=2, column=4, padx=10, pady=10)
        button3 = ttk.Button(self, text="Go back",
                             command=lambda: controller.show_frame(Page1))
        button3.grid(row=2, column=0, padx=10, pady=10)

    def InitUI(self):
        self.stringprot = tk.StringVar()
        self.stringclade= tk.StringVar()
        self.background_image = tk.PhotoImage(file='landscape.png')
        background_label = tk.Label(self, image=self.background_image)
        background_label.image = self.background_image
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.combo = ttk.Combobox(self, width=15, textvariable=self.stringprot)
        self.combo['values'] = ('NSP1', ' NSP2', ' NSP3', ' NSP4', ' NSP5', ' NSP6',
        'NSP7', 'NSP8', 'NSP9', 'score for NSP10', 'NSP11',
        'NSP12', 'NSP13', 'NSP14', 'NSP15', 'NSP16', 'Spike',
        'NS3','E', ' M', 'NS6', 'NS7a', 'NS7b', 'NS8', 'N')
        self.combo2=ttk.Combobox(self, width=15, textvariable=self.stringclade)
        self.combo2.grid(column=1, row=0)
        self.combo2['values']= ('Clade_V' ,'Clade_GH','Clade_GR','Clade_G','Clade_O','Clade_L','Clade_S')
        self.combo.grid(column=1, row=0)
        self.label = ttk.Label(self, text="Which protein, and which clade?")
        self.label.grid(column=0, row=0)
        self.button = ttk.Button(self, text="Go", command=lambda: fitter(self))
        self.button.grid(column=1, row=1)
    

class Page6(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.InitUI()
        button2 = ttk.Button(self, text="Go back to MainPage",
                             command=lambda: controller.show_frame(StartPage))
        button2.grid(row=2, column=4, padx=10, pady=10)
        button3 = ttk.Button(self, text="Go back",
                             command=lambda: controller.show_frame(Page1))
        button3.grid(row=2, column=0, padx=10, pady=10)
        
    def InitUI(self):
        self.string = tk.StringVar()
        self.background_image = tk.PhotoImage(file='landscape.png')
        background_label = tk.Label(self, image=self.background_image)
        background_label.image = self.background_image
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.combo = ttk.Combobox(self, width=15, textvariable=self.string)
        self.combo['values'] = ('Standardize','Normalize')
        self.combo.grid(column=1, row=0)
        self.label = ttk.Label(self, text="Make your own dataframe. Which one?")
        self.label.grid(column=0, row=0)
        self.button = ttk.Button(self, text="Go", command= lambda :  analysis_functions.scaler_function(df, self) )
        self.button.grid(column=1, row=1)


class Page7(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        button2 = ttk.Button(self, text="Go back to MainPage",
                             command=lambda: controller.show_frame(StartPage))
        button2.grid(row=2, column=4, padx=10, pady=10)
        button3 = ttk.Button(self, text="Go back",
                             command=lambda: controller.show_frame(Page1))
        button3.grid(row=2, column=0, padx=10, pady=10)
        

        def InitUI(self):
            self.stringprot = tk.StringVar()
            self.background_image = tk.PhotoImage(file='landscape.png')
            background_label = tk.Label(self, image=self.background_image)
            background_label.image = self.background_image
            background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
            self.combo = ttk.Combobox(self, width=15, textvariable=self.stringprot)
            self.combo['values'] = ('df', 'df1')
            button1 = ttk.Button(self, text="Pick a dataframe!", command=lambda: calc_vif(self))
            button1.grid(column=2, row=2)
            self.label = ttk.Label(self, text="Which one?")
            self.label.grid(column=0, row=0)


app = tkinterApp()
app.title('Covid-19 Project')
app.tk.call('wm', 'iconphoto', app._w, tk.PhotoImage(file='coronavirus.png'))

app.mainloop()
