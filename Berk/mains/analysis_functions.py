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


def scaler_function(df, command):
	'''

	:param df: dataframe to be scaled (needs to be all numaeric values)
	:param command: if command 'standardize' outputs scaled df
					if command 'normalize' outputs normalized df
					if command 'nothing' outputs unchanged df
	:return: processed dataframe
	'''


	import pandas as pd
	from sklearn import preprocessing


	inp = command.lower()
	if inp == 'standardize':
		standard=True
		normalize = False
	elif inp == 'normalize':
		normalize= True
		standard = False
	elif inp== 'nothing':
		normalize = False
		standard = False

	column_hold = df[['Country', 'iso_code', 'date']]
	im_df = df.drop(columns=['Country', 'iso_code', 'date'])

	if normalize:
		standard = False
		scaler = preprocessing.Normalizer()
		scaled_df = scaler.fit_transform(im_df)
		print('Normalized')

	elif standard:
		scaler = preprocessing.StandardScaler()
		scaled_df = scaler.fit_transform(im_df)
		print('Standardized')

	else: return df

	column_names = im_df.columns
	scaled_df = pd.DataFrame(scaled_df, columns=column_names)
	scaled_df = column_hold.join(scaled_df)

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


def dataframe_selector_all(w_or_m, clade_all=False, score_all=False, clade_score_all=False, clade_score_none=False):
	'''
	returns a dataframe based on boolean parameters
	:param w_or_m:	'week' or 'month'
	:param clade_all:
	:param score_all:
	:param clade_score_all:
	:param clade_score_none:
	:return: dataframe
	'''

	import pandas as pd

	if w_or_m.lower() == 'w' or w_or_m.lower() == 'week':
		_df = pd.read_table(r'C:\Users\BERK\PycharmProjects\CORONA_PROJECT\datasets\final_data\Weekly\Weekly_eliminateddf_2206.txt')

	elif w_or_m.lower() == 'm' or w_or_m.lower() == 'month':
		_df = pd.read_table(r'C:\Users\BERK\PycharmProjects\CORONA_PROJECT\datasets\final_data\Monthly\Monthly_eliminateddf_2206.txt')

	else:
		raise IOError('Input should be week or month')

	_id_lst = protein_column_list_giver(_df, 'id')
	_df = _df.drop(columns=_id_lst)

	_score_lst = protein_column_list_giver(_df, 'score')

	if clade_score_all:  ###Everything but ids
		return _df


	if clade_all:		### Tüm clade'ler
		_df.drop(columns=_score_lst, inplace=True)
		return _df

	if clade_score_none:  ####Sadece General Data'yı döndürür
		_df.drop(columns=_score_lst, inplace=True)
		return clade_remover(_df)

	if score_all:			###Tüm proteinler
		_df = clade_remover(_df)
		return _df


def dataframe_selector_obo(w_or_m, clade_obo=False, score_obo=False, clade_score_obo=False):
	'''
	Generator object, gives single clade/scores one by one
	:param w_or_m: 'week
	:param clade_obo: clade_one by one
	:param score_obo: score_one_by_one
	:param clade_score_obo:	clade+score one_by_one
	:return:
	'''

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
		_df = pd.read_table(r'C:\Users\BERK\PycharmProjects\CORONA_PROJECT\datasets\final_data\Weekly\Weekly_eliminateddf_2206.txt')

	elif w_or_m.lower() == 'm' or w_or_m.lower() == 'month':
		_df = pd.read_table(r'C:\Users\BERK\PycharmProjects\CORONA_PROJECT\datasets\final_data\Monthly\Monthly_eliminateddf_2206.txt')

	else:
		raise IOError('Input should be week or month')

	_id_lst = protein_column_list_giver(_df, 'id')
	_df = _df.drop(columns=_id_lst)

	_score_lst = protein_column_list_giver(_df, 'score')


	if clade_obo:  ## Sadece cladeler
		_df.drop(columns=_score_lst, inplace=True)

		for n, y in enumerate(clade_iterator(_df)):
			# saver.saver_func(y, df_name=f'{w_or_m}_clade_obo_{clade_keys[n]}')
			yield (y, clade_keys[n])

	if score_obo:   ##### Score Only
		_df = clade_remover(_df)

		for n, y in enumerate(protein_column_iterator(_df, 'score')):
			# saver.saver_func(y, df_name=f'{w_or_m}_score_obo_{protein_keys[n]}')
			yield (y, protein_keys[n])


	if clade_score_obo:
		for num, f in enumerate(clade_iterator(_df)):
			for n, y in enumerate(protein_column_iterator(f, 'score')):

				yield (y, protein_keys[n], clade_keys[num])


def df_summer(command_me, w_m):
	'''
	uses dataframe_selector_obo or dataframe_selector_all
	:param command_me: 'single_clades', 'single_score', 'single_clades_scores', 'all_clades', 'all_clades_scores',
	:param w_m: 'week' or 'month'
	:return: True value for respective command
	'''

	rv_dict = {}

	if command_me == 'single_clades':
		for r in dataframe_selector_obo(w_m,clade_obo=True):
			rv_dict[r[1]] = r[0]
		return rv_dict

	if command_me == 'single_scores':
		for r in dataframe_selector_obo(w_m,score_obo=True):
			rv_dict[r[1]] = r[0]
		return rv_dict

	if command_me == 'single_clades_scores':
		for r in dataframe_selector_obo(w_m, clade_score_obo=True):
			rv_dict[r[2]+r[1]] = r[0]

		return rv_dict

	if command_me == 'all_scores':

		rv_dict[command_me] = dataframe_selector_all(w_or_m=w_m, score_all=True)
		return rv_dict

	if command_me == 'all_clades':

		rv_dict[command_me] = dataframe_selector_all(w_or_m=w_m, clade_all=True)
		return rv_dict

	if command_me == 'all_clades_scores':

		rv_dict[command_me] = dataframe_selector_all(w_or_m=w_m, clade_score_all=True)
		return rv_dict

	if command_me == 'clade_score_none':	###Only corona stats

		rv_dict[command_me] = dataframe_selector_all(w_or_m=w_m, clade_score_none=True)
		return rv_dict


if __name__ == '__main__':

	print(df_summer('all_clades', w_m='m'))