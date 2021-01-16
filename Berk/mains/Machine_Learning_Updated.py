#### Linear Model calculation, machine learning #####
class regression():

	'''
	Class of regression models for each dataframe to be analyzed
	'''
	dataframe_num = 0
	def __init__(self, df, global_df, predict='death_growth_rate', feature_name='new_cases', data_name=''):
		self.df = df			#### ülkece df
		self.global_df = global_df		#### world df
		self.predict = predict
		self.feature_name = feature_name
		self.data_name = data_name

		dataframe_num = +1

	def linear_regression(self, test_size=0.25, accuracy=0.00, drop_non_ordinal=True, ordinal_date=False):
		'''
		Simple linear regression
		:param test_size: ratio of the dataframe that will be used for testing
		:param accuracy: min accuracy of the model
		:return: linear model
		'''
		import sklearn
		from sklearn import linear_model
		import pandas as pd

		acc = 0.00
		_predict = self.predict

		if drop_non_ordinal:
			c_dataframe = self.df.drop(columns=['Country', 'iso_code', 'date'])				###c_dataframe: country filtered df
			_dataframe = self.global_df.drop(columns=['Country', 'iso_code', 'date'])		####_dataframe: global df

			if ordinal_date:
				import datetime as dt
				the_df = self.global_df
				the_df['ordinal_date'] = pd.to_datetime(the_df['date']).apply(lambda date: date.toordinal())
				_dataframe = the_df.drop(columns=['Country', 'iso_code', 'date'])

				cthe_df=self.df
				cthe_df['ordinal_date'] = pd.to_datetime(cthe_df['date']).apply(lambda date: date.toordinal())
				c_dataframe = cthe_df.drop(columns=['Country', 'iso_code', 'date'])

		else:
			_dataframe = self.global_df
			c_dataframe = self.df

		features = _dataframe.drop(columns=_predict)
		label = _dataframe[_predict]

		cfeatures = c_dataframe.drop(columns=_predict)
		clabel = c_dataframe[_predict]

		counter = 0
		while acc <= accuracy:

			self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(features, label, test_size=test_size)
			linear = sklearn.linear_model.LinearRegression()
			linear.fit(self.x_train, self.y_train)
			acc = linear.score(self.x_test, self.y_test)
			counter += 1

			if counter >10000:
				print(f'Could not obtain accuracy limit. \n Best values is {acc.__round__(2)}')
				break


		self.resulting_accuracy = acc
		self.lin_model = linear
		#self.y_predict = linear.predict(self.x_test)

		#if self.resulting_accuracy > 0.55:
		#	from Scripts import saver
		#	saver.model_saver(l_model=linear, df_name=self.data_name, acc=acc, predict=self.predict)

		self.cnt_x_train, self.cnt_x_test, self.cnt_y_train, self.cnt_y_test = sklearn.model_selection.train_test_split(cfeatures, clabel,  ##### Filtered Country train/test sets
																										test_size=0.25)

		#clinear = sklearn.linear_model.LinearRegression()
		#self.clin_model = clinear
		#clinear.fit(self.cnt_x_train, self.cnt_y_train)
		glob_on_country_acc = linear.score(self.cnt_x_test, self.cnt_y_test)

		y_predict = linear.predict(self.cnt_x_test)

		self.MSE = sklearn.metrics.mean_squared_error(y_true=self.cnt_y_test, y_pred=y_predict)
		self.MAE = sklearn.metrics.mean_absolute_error(y_true=self.cnt_y_test, y_pred=y_predict)
		self.R_squared = sklearn.metrics.r2_score(y_true=self.cnt_y_test, y_pred=y_predict)

		print(f'Global model on country accuracy is : {glob_on_country_acc}')

		return linear


	def predictions(self):
		'''
		when called, prints predicted vs true values of the model
		:return:
		'''
		import pandas as pd
		expected = self.lin_model.predict(self.x_test)

		for x in range(len(expected)):
			print('predicted:', expected[x], "\n",
				  # x_test.iloc[x], "\n",
				  'true values: ', self.y_test.iloc[x], "\n")

	def prediction_plotter(self):
		'''
		Plots 2 D graph, prediction vs dependent variable
		:return:
		'''
		import matplotlib.pyplot as pyplot
		from matplotlib import style

		data = self.df
		p = self.predict
		f = self.feature_name

		import seaborn
		from matplotlib import pyplot
		import numpy as np

		p = self.predict
		pyplot.title(f"{self.data_name} {p} vs {f}")
		pyplot.xlabel(f'{f}')
		pyplot.ylabel(f'{p}')

		seaborn.set(style="white", color_codes=True)

		ax = seaborn.regplot(
			x=self.cnt_x_train['ordinal_date'],
			y=self.lin_model.predict(self.cnt_x_train),
			x_estimator=np.mean,
			line_kws={'color': 'red'},
			scatter_kws={'color': 'cyan'},
			truncate=False)

		print(f'Resulting accuracy is: {self.resulting_accuracy} '
			  f'\nMean Squared Error: {self.MSE} '
			  f'\nMean Absolute Error: {self.MAE} '
			  f'\n R_Squared is : {self.R_squared}')
		#pyplot.legend()
		pyplot.show()


	def date_plotter(self):
		'''
		if regression model contains date, need to use this one. (ordinal date to normal date conversion)
		:return:
		'''
		import seaborn
		from matplotlib import pyplot
		import numpy as np

		p = self.predict
		pyplot.title(f"{self.data_name} {p} vs ordinal_date")
		pyplot.xlabel('dates')
		pyplot.ylabel(f'{p}')

		seaborn.set(style="white", color_codes=True)

		ax = seaborn.regplot(
			x= self.cnt_x_train['ordinal_date'],
			y= self.lin_model.predict(self.cnt_x_train),
			x_estimator=np.mean,
			line_kws = {'color':'red'},
			scatter_kws= {'color':'cyan'},
			truncate=False)

		## Tighten up the axes for prettiness
		##ax.set_xlim(self.x_train['ordinal_date'].min() - 1, self.x_train['ordinal_date'].max() + 1)
		##ax.set_ylim(0, self.x_train.max() + 1)

		import datetime as dt
		new_labels = [dt.date.fromordinal(int(item)) for item in ax.get_xticks()]
		ax.set_xticklabels(new_labels, rotation=45)

		print(f'Global accuracy is: {self.resulting_accuracy} '
			  f'\nMean Squared Error: {self.MSE} '
			  f'\nMean Absolute Error: {self.MAE} ')
			 # f'\n R_Squared is : {self.R_squared}'

		pyplot.show()



############ Date Time + Country'ye göre linear model prediction #######

def weeklyextrapolating_func(df_to_exp):
	'''
	will extrapolate the dates (10 weeks)
	:param df_to_exp:  dataframe to be extrapolated weekly df
	:return: appended and filled datafram
	:return:
	'''
	import pandas as pd
	pd.set_option('display.max_columns', 30)
	import sklearn
	from sklearn import impute

	column_names = df_to_exp.columns
	to_be_appended =pd.DataFrame(columns=column_names)

	df_to_exp['date'] = pd.to_datetime(df_to_exp['date'])

	last_day = df_to_exp['date'].max()
	to_be_appended['date'] = pd.Series(pd.date_range(last_day+pd.DateOffset(7), freq='W', periods=10))

	df_to_exp = df_to_exp.append(to_be_appended)


	for column in column_names:  ##extrapolation
		if column == 'Country' or column == 'iso_code':
			df_to_exp[column] = df_to_exp[column].ffill()

		elif column == 'death_growth_rate':
			df_to_exp[column] = df_to_exp[column].fillna(df_to_exp[column].median())

		else:
			df_to_exp[column] = df_to_exp[column].fillna(df_to_exp[column].mean())

	return df_to_exp

def montlyextrapolating_func(df_to_exp):
	'''
	will extrapolate the dates (2 months)
	:param df_to_exp: dataframe to be extrapolated_monthly df
	:return: appended and filled datafram
	'''
	import pandas as pd
	pd.set_option('display.max_columns', 30)
	import sklearn
	from sklearn import impute

	column_names = df_to_exp.columns
	to_be_appended =pd.DataFrame(columns=column_names)

	df_to_exp['date'] = pd.to_datetime(df_to_exp['date'])

	last_day = df_to_exp['date'].max()
	
	to_be_appended['date'] = pd.Series(pd.date_range(last_day+pd.DateOffset(30), freq='M', periods=3))

	
	df_to_exp = df_to_exp.append(to_be_appended)


	for column in column_names:		#extrapolation
		if column == 'Country' or column == 'iso_code':
			df_to_exp[column] = df_to_exp[column].ffill()

		elif column == 'death_growth_rate':
			df_to_exp[column] = df_to_exp[column].fillna(df_to_exp[column].median())

		else:
			df_to_exp[column] = df_to_exp[column].fillna(df_to_exp[column].mean())

	return df_to_exp



def country_lm(country_name, df, w_m, predict='death_growth_rate', feature_name='ordinal_date',extrapolation_date=True, test_size=0.5, accuracy=0.50, drop_non_ordinal=True, ordinal_date=True, plot_date=True):
	'''
	The main function that calls linear model and plot functions of the regression class
	:param country_name: Country name to be analyzed : will be used for filtering the data
	:param df: data to be analyzed : later it will be used as global dataframe
	:param w_m: 'week' or 'month' data
	:param predict:	default parameter: 'death_growth_rate' can be changed
	:param feature_name: default parameter: 'ordinal_date', date will converted as ordinal_date to be analyzed
	:param extrapolation_date:	boolean: if True, calls extrapolating_func:: will extrapolate the dates (10 weeks or 2 months)
	:param test_size:  input for linear_regression:: default parameter :0.50 Can be decreased to improve accuracy in weekly data
	:param accuracy:	input for linear_regression :: minimum accuracy that the model needs to please
	:param drop_non_ordinal: input for linear_regression
	:param ordinal_date: input for linear_regression
	:param plot_date: boolean: if True calls date_plotter, if False calls predict_plotter
	:return:
	'''
	import pandas as pd
	import datetime as dt

	gl_df = df
	cntry_fltr = df['Country'] == country_name
	filtrd_df = df.loc[cntry_fltr]


	if extrapolation_date:
		if w_m=='week' or w_m=='w':
			gl_df = weeklyextrapolating_func(gl_df)
			filtrd_df = weeklyextrapolating_func(filtrd_df)

		elif w_m=='month' or w_m=='m':
			gl_df = montlyextrapolating_func(gl_df)
			filtrd_df = montlyextrapolating_func(filtrd_df)


	filtrd_df['ordinal_date'] = pd.to_datetime(filtrd_df['date']).apply(lambda date: date.toordinal())
	gl_df['ordinal_date'] = pd.to_datetime(gl_df['date']).apply(lambda date: date.toordinal())

	regres_model = regression(global_df=gl_df, df=filtrd_df, feature_name=feature_name, predict=predict, data_name=country_name)
	lin_model = regres_model.linear_regression(test_size=test_size, accuracy=accuracy, drop_non_ordinal=drop_non_ordinal, ordinal_date=ordinal_date)

	if plot_date:

		aa = regres_model.date_plotter()
	else:

		aa = regres_model.prediction_plotter()


if __name__=='__main__':  ####### TESTING #######
	from main_scripts import analysis_functions as af

	im_df = af.df_summer('all_scores', 'm')['all_scores']
	im_df = af.scaler_function(im_df, 'standardize')


	country_lm('Turkey', w_m='month', df=im_df, plot_date=True, feature_name='ordinal_date', extrapolation_date=False)


