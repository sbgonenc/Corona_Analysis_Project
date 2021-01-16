def clade_counter(str_value, clade_name):
    '''
    Counts clades frequency
    :param str_value: a string value of clade name in dataframe
    :param clade_name: clade name to be counted
    :return: variable filled clade_name (dummy feature)
    '''
    counter = 0
    cl_name = clade_name.upper()

    if cl_name == str_value or cl_name in str_value:
        counter += 1

    return counter

#df=pd.read_table('Monthly_cladeCount_18_06.txt')
def scaler_function(df,self):
    '''

    :param df: dataframe to be scaled (needs to be all numaeric values)
    :param standard: if true, uses scikit Standard Scaler
    :param normalize: if True, uses scikit Normalizer
    :return: processed dataframe
    '''
    import pandas as pd
    normalize=False
    standard=False
    if self.combo.get() == 'Standardize':
        standard=True
    if self.combo.get() == 'Normalize':
        normalize=True
    import pandas as pd
    from sklearn import preprocessing
    column_names = df.columns

    if standard==normalize:
        raise IOError('both standard and normalize cannot be False or True')

    if normalize:
        standard = False
        scaler = preprocessing.Normalizer()
        scaled_df = scaler.fit_transform(df)

    elif standard:
        scaler = preprocessing.StandardScaler()
        scaled_df = scaler.fit_transform(df)

    scaled_df = pd.DataFrame(scaled_df, columns=column_names)

    return scaled_df

def protein_column_list_giver(df,name_type='score'):

    import pandas as pd

    all_column_names = df.columns
    prot_column_names = []

    for name in all_column_names:
        if name_type in name:
            prot_column_names.append(name)

    return prot_column_names

def protein_column_iterator(df, col_name_type):

    import pandas as pd

    column_names_id = protein_column_list_giver(df, name_type='id')
    column_names_scores = protein_column_list_giver(df, name_type='score')

    if col_name_type == 'id':
        im_df = df.drop(columns=column_names_scores)
        column_names = column_names_id
    if col_name_type == 'score':
        im_df = df.drop(columns=column_names_id)
        column_names = column_names_scores
    else:
        raise NameError(f'inputted {col_name_type} is not valid. It should be "id" or "score"')

    for column_name in column_names:
        rv_df = im_df
        iterated_column = im_df[column_name]
        #column_names.remove(column_name)

        rv_df = rv_df.drop(columns=column_names)
        rv_df = rv_df.join(iterated_column)

        yield rv_df
        #print(rv_df[column_name])
        #column_names = protein_column_list_giver(df, name_type=col_name_type)

def clade_iterator(df):
    '''
    iterates single clades within the dataframe one by one
    :param df: pandas dataframe
    :return: dataframe with the single clade
    '''

    import pandas as pd

    #prot_col_name_id = protein_column_list_giver(df, name_type='id')
    #prot_col_name_scores = protein_column_list_giver(df, name_type='score')

    #im_df = df.drop(columns=prot_col_name_id)
    #im_df = im_df.drop(columns=prot_col_name_scores)

    clade_names = []

    for column_name in df.columns:
        if 'Clade' in column_name:
            clade_names.append(column_name)

    for clade_name in clade_names:

        iterated_column = df[clade_name]

        rv_df = df.drop(columns=clade_names)
        rv_df = rv_df.join(iterated_column)

        yield rv_df


def clade_remover(df):
    '''
    removes clades from df
    :param df: pandas df
    :return: clades removed df
    '''
    import pandas as pd
    clade_names= []
    for column_name in df.columns:
        if 'Clade' in column_name:
            clade_names.append(column_name)

    rv_df = df.drop(columns=clade_names)

    return rv_df


def dataframe_selector(w_or_m, clade_obo=False, clade_all=False,
                       score_obo=False, score_all=False,
                       clade_score_obo=False, clade_score_all=False, clade_score_none=False):

    import pandas as pd

    protein_keys ={
        0: 'NSP1',
        1: 'NSP2',
        2: 'NSP3',
        3: 'NSP4',
        4: 'NSP5',
        5: 'NSP6',
        6: 'NSP7',
        7: 'NSP8',
        8: 'NSP9',
        9: 'NSP10',
        10: 'NSP11',
        11: 'NSP12',
        12: 'NSP13',
        13: 'NSP14',
        14: 'NSP15',
        15: 'NSP16',
        16: 'Spike',
        17: 'NS3',
        18: 'E',
        19: 'M',
        20: 'NS6',
        21: 'NS7a',
        22: 'NS7b',
        23: 'NS8',
        24: 'N'
    }
    clade_keys ={
        0: 'Clade_V',
        1: 'Clade_GH',
        2: 'Clade_GR',
        3: 'Clade_G',
        4: 'Clade_O',
        5: 'Clade_L',
        6: 'Clade_S'
    }



    if w_or_m.lower() == 'w' or w_or_m.lower() == 'week':
        _df = pd.read_table(r'Weekly_eliminateddf_2206.txt')

    elif w_or_m.lower() == 'm' or w_or_m.lower() == 'month':
        _df = pd.read_table(r'Monthly_eliminateddf_2206.txt')

    else:
        raise IOError('Input should be week or month')

    _id_lst = protein_column_list_giver(_df, 'id')
    _df = _df.drop(columns=_id_lst)

    _score_lst = protein_column_list_giver(_df, 'score')

    if clade_score_all:
        return _df


    if clade_all or clade_score_none or clade_obo:  ## Sadece cladeler
        _df.drop(columns=_score_lst, inplace=True)

        if clade_all: return _df	### Tüm clade'ler
        if clade_score_none: return clade_remover(_df)  ####Sadece General Data'yı döndürür

        if clade_obo:  ### Teker teker sadece clade'ler

            for n, y in enumerate(clade_iterator(_df)):
                # saver.saver_func(y, df_name=f'{w_or_m}_clade_obo_{clade_keys[n]}')
                yield (y, clade_keys[n])

    if score_obo or score_all:    ##### Score Only
        _df = clade_remover(_df)

        if score_obo:

            for n, y in enumerate(protein_column_iterator(_df, 'score')):
                # saver.saver_func(y, df_name=f'{w_or_m}_score_obo_{protein_keys[n]}')
                yield (y, protein_keys[n])

        if score_all:

            _df = clade_remover(_df)
            return _df

    if clade_score_obo:
        for num, f in enumerate(clade_iterator(_df)):
            for n, y in enumerate(protein_column_iterator(f, 'score')):

                yield (y, protein_keys[n], clade_keys[num])

if __name__ == '__main__':
    for e in dataframe_selector('m', clade_score_obo=True):
        flt = e[0][e[2]] == 1
        print(e[1], e[2], '\n', e[0].loc[flt].head(20))
def df_summer(command_me, w_m):

	rv_dict = {}

	if command_me == 'single_clades':
		for r in dataframe_selector(w_m,clade_obo=True):
			rv_dict[r[1]] = r[0]
		return rv_dict

	if command_me == 'single_scores':
		for r in dataframe_selector(w_m,score_obo=True):
			rv_dict[r[1]] = r[0]
		return rv_dict

	if command_me == 'single_scores_clades':
		for r in dataframe_selector(w_m, clade_score_obo=True):
			rv_dict[r[2]+r[1]] = r[0]

		return rv_dict