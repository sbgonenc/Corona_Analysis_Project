import pandas as pd

def rounder_func(str):
	'''
	if it's convertible, converts str to float and rounds to 4 decimal places
	:param str: string values
	:return: float and rounded values
	'''
	try:
		flt = float(str).__round__(4)
		return flt

	except (TypeError, ValueError):
		return str


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
	filtr_country = df1
	for country in genel['Country']:
		if country not in filtr_country['Country']:
			drop_index = genel[genel==country].index
			genel.drop(drop_index, inplace=True)

	return genel

if __name__ == '__main__':
	referans_file = r'C:\Users\BERK\PycharmProjects\CORONA_PROJECT\Smoker_14-06_added_coronadf.txt'
	referans_df = pd.read_table(referans_file)

	def str_float_convert(str_type, index, column, ref_df=referans_df):						#Her column'daki non-typical value'lar farklı bir pattern gösterdiğinden, çağırdığım fonksiyona çok fazla kural yazmam gerekti.
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
				ref_df.loc[index, column] = f_type
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
				ref_value = ref_df.loc[index - 1, column]   #bazı column'larda uygunsuz değerleri değiştirebilmek için bir referans değerine ihtiyaç duydum.
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

					ref_df.loc[index, column] = f_str

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


def series_reducer(lst):
	'''
	takes a list, reduces to unique values
	:param lst: a series list
	:return: reduced list
	'''
	rv = []
	for element in lst:

		if element not in rv:
			rv.append(element)
		else:
			continue

	clean_lst = [x for x in rv if str(x) != 'nan']

	if not clean_lst: return None

	return clean_lst


def str_strip(value):

	try:
		for e in value:
			word = e.replace('"','')
			word = word.replace('[', '')
			word = word.replace(']', '')
			word = word.replace("'",'')
			word_lst = word.strip().split(',')

		return word

	except (ValueError, TypeError):
		return value

def inf_remover(value):

	if value == 'inf':
		return 0.0