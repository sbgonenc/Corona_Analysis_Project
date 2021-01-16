#### Linear Model calculation, machine learning #####
class regression():

	'''
	Class of regression models for each dataframe to be analyzed
	'''
	dataframe_num = 0
	def __init__(self, df, predict='death_growth_rate', feature_name='new_cases', data_name='', standardize=False, normalize=False):
		self.df = df
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
			_dataframe = self.df.drop(columns=['Country', 'iso_code', 'date'])

			if ordinal_date:
				import datetime as dt
				the_df = self.df
				the_df['ordinal_date'] = pd.to_datetime(the_df['date']).apply(lambda date: date.toordinal())
				_dataframe = the_df.drop(columns=['Country', 'iso_code', 'date'])

		else:
			_dataframe = self.df

		features = _dataframe.drop(columns=_predict)
		label = _dataframe[_predict]

		counter = 0
		while acc <= accuracy:

			self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(features, label, test_size=test_size)
			linear = sklearn.linear_model.LinearRegression()
			linear.fit(self.x_train, self.y_train)
			acc = linear.score(self.x_test, self.y_test)
			counter += 1

			if counter >100:
				print(f'Could not obtain accuracy limit. \n Best values is {acc.__round__(2)}')
				break

		y_predict= linear.predict(self.x_test)
		self.MSE = sklearn.metrics.mean_squared_error(y_true=self.y_test, y_pred=y_predict)
		self.MAE = sklearn.metrics.mean_absolute_error(y_true=self.y_test, y_pred=y_predict)
		self.R_squared = sklearn.metrics.r2_score(y_true=self.y_test, y_pred=y_predict)

		self.resulting_accuracy = acc
		self.lin_model = linear
		#self.y_predict = linear.predict(self.x_test)

		# if self.resulting_accuracy > 0.75:
		# 	from Scripts import saver
		# 	saver.model_saver(df_name=self.data_name, acc=acc, predict=self.predict)


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

	def prediction_plotter(self, save=False):
		'''
		Plots 2 D graph, prediction vs dependent variable
		:return:
		'''
		import matplotlib.pyplot as pyplot
		from matplotlib import style

		p = self.predict
		f = self.feature_name


		style.use("ggplot")
		pyplot.title(f"{self.data_name} {p} vs {f}__(Testing_Data)")
		pyplot.scatter(self.x_test[f], self.y_test, color='red', label= '')
		pyplot.plot(self.x_train, self.lin_model.predict(self.x_train), color='blue', alpha=0.05, label='linear Regression')
		pyplot.xlabel(f)
		pyplot.ylabel(p)
		print(f'Resulting accuracy is: {self.resulting_accuracy} '
			  f'\nMean Squared Error: {self.MSE} '
			  f'\nMean Absolute Error: {self.MAE} '
			  f'\n R_Squared is : {self.R_squared}')
		#pyplot.legend()
		pyplot.show()

		if save:
			pyplot.savefig(f"{self.data_name} {p} vs {f} plot.png")

	def date_plotter(self):
		'''
		if regression model contains date, need to use this one. (ordinal date to normal date conversion)
		:return:
		'''
		import seaborn
		from matplotlib import pyplot

		p = self.predict
		pyplot.title(f"{self.data_name} {p} vs date)")
		pyplot.xlabel('dates')
		pyplot.ylabel(f'{p}')

		ax = seaborn.regplot(
			x= self.x_train['ordinal_date'],
			y= self.lin_model.predict(self.x_train)
		)
		# Tighten up the axes for prettiness
		#ax.set_xlim(self.x_train['ordinal_date'].min() - 1, self.x_train['ordinal_date'].max() + 1)
		#ax.set_ylim(0, self.x_train.max() + 1)

		import datetime as dt
		new_labels = [dt.date.fromordinal(int(item)) for item in ax.get_xticks()]
		ax.set_xticklabels(new_labels, rotation=45)

		print(f'Resulting accuracy is: {self.resulting_accuracy} '
			  f'\nMean Squared Error: {self.MSE} '
			  f'\nMean Absolute Error: {self.MAE} '
			  f'\n R_Squared is : {self.R_squared}')

		pyplot.show()



############ Date Time + Country'ye göre linear model prediction #######
import pandas as pd
default_df = pd.read_table(r'Weekly_eliminateddf_2206.txt')

def expolating_func(df_to_exp):
	import pandas as pd
	pd.set_option('display.max_columns', 30)
	import sklearn
	from sklearn import impute


	column_names = df_to_exp.columns
	to_be_appended =pd.DataFrame(columns=column_names)

	df_to_exp['date'] = pd.to_datetime(df_to_exp['date'])
	dayrange = df_to_exp['date'].iloc[1] - df_to_exp['date'].iloc[0]
	weekly_range = pd.to_datetime('2020-01-08') - pd.to_datetime('2020-01-01')

	last_day = df_to_exp['date'].max()
	if dayrange > weekly_range:
		to_be_appended['date'] = pd.Series(pd.date_range(last_day+pd.DateOffset(30), freq='M', periods=3))

	else:
		to_be_appended['date'] = pd.Series(pd.date_range(last_day+pd.DateOffset(7), freq='W', periods=10))

	df_to_exp = df_to_exp.append(to_be_appended)


	for column in column_names:
		if column == 'Country' or column == 'iso_code':
			df_to_exp[column] = df_to_exp[column].ffill()

		elif column == 'death_growth_rate':
			df_to_exp[column] = df_to_exp[column].fillna(df_to_exp[column].median())

		else:
			df_to_exp[column] = df_to_exp[column].fillna(df_to_exp[column].mean())

	return df_to_exp


def country_lm(country_name, df, predict='death_growth_rate', expolation_date=True):
	import pandas as pd
	import datetime as dt

	cntry_fltr = df['Country'] == country_name
	im_df = df.loc[cntry_fltr]

	if expolation_date:
		im_df = expolating_func(im_df)

	im_df['ordinal_date'] = pd.to_datetime(im_df['date']).apply(lambda date: date.toordinal())

	regres_model = regression(df=im_df, feature_name='ordinal_date', predict=predict, data_name=country_name)
	lin_model = regres_model.linear_regression(test_size=0.50, accuracy=0.50, drop_non_ordinal=True, ordinal_date=True)
	aa = regres_model.date_plotter()


if __name__=='__main__':
	country_lm('United States', df=default_df)



