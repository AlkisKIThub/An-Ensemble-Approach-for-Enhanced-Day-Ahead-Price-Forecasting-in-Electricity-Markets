
# function that makes the evaluation of the models for specific project.

def Evaluation_project(zone, TablePred, ColumPred, Training_version, server_credentials,startdate, daysBack=10, create_plots=False):
	
	## ---- Libraries and connections ----
	import pandas as pd
	# set option to see all columns
	pd.set_option('display.max_columns', 100)
	import numpy as np
	from numpy import arange, mean, std
	from datetime import datetime, date, timedelta
	from functools import reduce
	from sklearn.metrics import mean_absolute_error as mae
	from sklearn.metrics import mean_squared_error as rmse
	from sklearn.metrics import mean_absolute_percentage_error as mape
	# excel
	import openpyxl
	# System
	import os
	import pickle

	# graphics
	import seaborn as sns
	import matplotlib.pyplot as plt
	from jupyterthemes import jtplot
	import plotly.express as px
	import six


	ModelPathFolder = r"/day_ahead_EPF/Models"
 
	# create folders if they do not exist
	EvaluationPathFolder = r'/day_ahead_EPF/Evaluation/DE/Prices/v1'
 
	try:
		os.makedirs(EvaluationPathFolder)
	except FileExistsError:
		pass

	# Define time period to evaluate or let it start from the beggining or end
#	 Today = datetime.now()
	Today = pd.to_datetime(startdate, format = ('%Y-%m-%d %H:%M:%S') )
	start = (Today - timedelta(days=daysBack)).strftime('%Y-%m-%d %H:%M:%S')
	end = (Today).strftime('%Y-%m-%d %H:%M:%S')

	# define project name
	project = zone + '_' + TablePred + '_' + Training_version + '_'
	print('Current project name to evaluate:', project)

	# define table name for forecasted data
	TableForecasts = TablePred + '_forecast'
	# define table name for actual data
	TableActual = TablePred

	## ---- Import data ---- 

 	# retrieve Historical forecasts data
	Historical_forecasts = pd.read_csv(r"/day_ahead_EPF/data/historical_predictions.csv")
	Historical_forecasts['Datetime']= pd.to_datetime(Historical_forecasts['Datetime'], format = ('%Y-%m-%d %H:%M:%S') )
	Historical_forecasts = Historical_forecasts[(Historical_forecasts['Datetime']>start) & (Historical_forecasts['Datetime']<end)]
	if Forecasts.empty:
		print('No Forecast data for the selected dates')
	else:
		# Keep forecasts of the same project
		Forecasts = Forecasts[Forecasts['project'] == project]
		# retrieve actual data
		Actual = pd.read_csv(r"/day_ahead_EPF/data/actual_data.csv")
		Actual['Datetime']= pd.to_datetime(Actual['Datetime'], format = ('%Y-%m-%d %H:%M:%S') )
		Actual = Actual[(Actual['Datetime']>start) & (Actual['Datetime']<end)]
		# retrieve Naive model data as actual prices of day before
		Naive = pd.read_csv(r"/day_ahead_EPF/data/DE_LU_DAM_prices.csv")
		start_Naive = pd.to_datetime(start, format='%Y-%m-%d') - timedelta(days=1)
		end_Naive = pd.to_datetime(end, format='%Y-%m-%d') - timedelta(days=1)
		Naive['Datetime']= pd.to_datetime(Naive['Datetime'], format = ('%Y-%m-%d %H:%M:%S') )
		Naive = Naive[(Naive['Datetime']>start_Naive) & (Naive['Datetime']<end_Naive)]

		## ---- Preprocess data ---- 

		# Time column appears twice in the Forecasts so drop one
		Forecasts = Forecasts.T.drop_duplicates().T
		# drop column project
		try:
			Forecasts.drop(['project'], axis=1, inplace=True)
		except:
			pass
		# for the Naive model set datetime one day ahead
		Naive['Datetime'] = ( Naive['Datetime'] + timedelta(days=1) )
		# change model name to Naive
		Naive.rename(columns={ColumPred:'Naive'}, inplace=True)
		# Merge Forecasts with Actual data
		DF = pd.merge(Actual, Forecasts, on='Datetime', how='inner').merge(Naive, on='Datetime', how='inner')
		# Check for missing values and drop
		print('forecast missing values:\n', DF.isnull().sum(),'\nforecast table size: ',DF.shape)
		# # remove columns having at least one missing (NaN) value
		DF = DF.loc[:,DF.notna().all(axis=0)]
		#remove rows having at least one missing (NaN) value
		#DF = DF.dropna()
		# show hours and dates in the dataset
		print('Evaluation days:', DF.shape[0]/24)
		# set Datetime column as index
		DF.set_index('Datetime', inplace=True)
		# replace None with np.nan
		DF.replace(('None',np.nan), inplace=True)
		# transform all numeric data to float64
		DF = DF.astype(float)


		## ---- Calculate Metrics ----

		# define list with metrics
		metrics_list = ['Model','MAE','RMSE','MAPE','sMAPE','NegPos']
		
		# initiate dataframe with metrics
		Metrics = pd.DataFrame(columns=metrics_list)
		# initiate dataframe to store forecasts and respective deviations from actual
		ResultsSub = pd.DataFrame(DF.iloc[:, 0])
		
		# Define the function to return the MAPE values
		def calculate_mape(actual, forecast):  
			# APE value for each of the records in dataset
			MAPE = []
			length = 0
			# Iterate over the list values
			for day in range(len(actual)):
				if abs(actual[day]) > 10:
					# Calculate percentage error
					per_err = (abs(actual[day] - forecast[day])) / abs(actual[day]) 
					# Append it to the APE list
					MAPE.append(per_err)
					length += 1
				else:
					pass
			# Calculate the MAPE
			MAPE = sum(MAPE)/length*100
			return(MAPE)
		
		# Define the function to return the symmetric MAPE sMAPE values
		def calculate_smape(actual, forecast):  
			# APE value for each of the records in dataset
			sMAPE = []
			length = 0
			# Iterate over the list values
			for day in range(len(actual)):
				# Calculate percentage error
				per_err = (abs(actual[day]) - abs(forecast[day] - actual[day]) +  abs(forecast[day]))  
				# Append it to the APE list
				sMAPE.append(per_err)
				length += 1
			# Calculate the MAPE
			sMAPE = sum(sMAPE)/length*100
			return(sMAPE)

		# loop through models
		for column in DF.columns.difference([ColumPred]):
			# Metrics cannot be calculated with NaNs, so exclude them
			df = DF[DF[column].notna()]
			# calculate metrics
			MAE = mae(df[ColumPred], df[column]).round(2)
			RMSE = rmse(df[ColumPred], df[column],squared=False).round(2)
			MAPE = calculate_mape(df[ColumPred], df[column]).round(2)
			sMAPE = calculate_smape(df[ColumPred], df[column]).round(2)
			NegPos = round(len(df.loc[df[ColumPred].mul(df[column]).ge(0)])/len(df),2)
			# consolidate metric results
			metric_series = pd.Series([column,MAE,RMSE,MAPE, sMAPE, NegPos], index = metrics_list )
			Metrics = pd.concat([Metrics, metric_series], axis=1, ignore_index=True)
			
			# Collect the initial predictions plus the deviations from the actual
			ResultsSub = pd.concat([ResultsSub, df[column]], axis=1)
			Diviation_Column_Header = r"Deviation-" + column
			Diviation = pd.DataFrame(df[column].subtract(df[ColumPred]),columns=[Diviation_Column_Header])
			ResultsSub = pd.concat([ResultsSub, Diviation], axis=1)
		
		# Results table to excel
		ResultsEval = Metrics.dropna(axis=1,how='all').T.sort_values(by=['MAE','RMSE','MAPE','sMAPE'])
		ResultsEval.dropna(axis=0, how='any', inplace=True)
		#Save Results to excel file
		ExcelFilePath = EvaluationPathFolder + 'Evaluation_Results.xlsx'
		try:
			# Append mode will fail if file does not exist
			with pd.ExcelWriter(ExcelFilePath, engine = 'openpyxl', mode="a", if_sheet_exists='replace')  as writer: 
				ResultsEval.to_excel(writer,sheet_name="Evaluation Results", index = False)
		except FileNotFoundError:
			with pd.ExcelWriter(ExcelFilePath) as writer: 
				ResultsEval.to_excel(writer,sheet_name="Evaluation Results", index = False)


		## ---- Create table of 'Best Hourly' models ---- 

		# Make the Timestamp chart
		Timestamp = pd.to_datetime(ResultsSub.index).hour
		Timestamp = pd.DataFrame(Timestamp)
		# Get the deviations columns
		Deviations_columns = [col for col in ResultsSub.columns if 'Deviation' in col]
		Deviations = ResultsSub[Deviations_columns]
		# Get the deviations columns without the best columns
		Best_columns = [col for col in Deviations.columns if 'Best' in col]
		Deviations_WithoutBestModel = Deviations[Deviations.columns.difference(Best_columns)]
		# Get the absolute number of deviations
		Deviations_Absolute = Deviations_WithoutBestModel.abs()
		# Reset index in order to concatenate with the Timestamp column
		Deviations_Absolute.reset_index(drop=True, inplace=True)
		AbsDeviations = pd.concat([Timestamp, Deviations_Absolute], axis=1)
		# Group Deviations by hour
		GroupBy = AbsDeviations.groupby(by="Datetime").mean()
		# Table with the best model per hour of the day
		BestModelPerHour = GroupBy.apply(lambda x: pd.Series(np.concatenate([x.nsmallest(len(GroupBy.columns)).index.values])), axis=1)
		# Make index as column
		BestModelPerHour.reset_index(level=0, inplace=True)
		# Remove the word Deviation from the model name
		for m in range(1, len(BestModelPerHour.columns)):
			BestModelPerHour.iloc[:,m] = BestModelPerHour.iloc[:,m].str.split("Deviation-").str[1]

		# Load Best hourly table dictionary from disc
		TableFilePath = ModelPathFolder + 'Best_Hourly_Table.pkl'
		Best_Hourly_Dictionary = pickle.load(open(TableFilePath, 'rb'))
		# dictionary format: { zone : {TablePred : {Training_version : BestModelPerHour}}}
		# store results into dictionary
		try:
			Best_Hourly_Dictionary[zone][TablePred][Training_version] = BestModelPerHour
			# catch the exception when a new country is added on the dictionary
		except KeyError:
			Best_Hourly_Dictionary[zone] = {TablePred : {Training_version : BestModelPerHour}}
		# Save dictionary to disc
		with open(TableFilePath, 'wb') as writer:
			pickle.dump(Best_Hourly_Dictionary, writer)


		# -------- Plot evaluation graphs -------- 
		if create_plots:
			# ------------------- Hourly evaluation graph -------------------
			# Change the picture theme
			jtplot.style(theme='chesterish',context='paper',fscale=1.6,ticks=True, grid=False, figsize=(10, 10))
			# Create a figure and axes
			fig, ax = plt.subplots(figsize=(12,5))
			# Setting title to graph
			ax.set_title(str('Deviations from actual price '))
			# Function to plot and show graph
			GroupBy.plot(ax=ax)
			# Format the legend
			ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 12})
			# label x-axis and y-axis
			ax.set_ylabel('Mean absolute error')
			ax.set_xlabel('hour')
			# Save graph to folder
			ImagePathFile = EvaluationPathFolder + r'Hourly Deviations.png'
			plt.savefig(ImagePathFile, bbox_inches='tight',transparent=True)
			plt.close()
			plt.clf()

			# ------------------- Hourly evaluation table -------------------
			# Make table as graph and export it
			def render_mpl_table(data, col_width=3.0, row_height=0.6, font_size=14, header_color='#40466e', row_colors=['#f1f1f2', 'w'],
								edge_color='w', bbox=[0, 0, 1, 1], header_columns=0, ax=None, **kwargs):
				if ax is None:
					size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
					fig, ax = plt.subplots(figsize=size)
					ax.axis('off')

				mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

				mpl_table.auto_set_font_size(False)
				mpl_table.set_fontsize(font_size)

				for k, cell in  six.iteritems(mpl_table._cells):
					cell.set_edgecolor(edge_color)
					if k[0] == 0 or k[1] < header_columns:
						cell.set_text_props(weight='bold', color='w')
						cell.set_facecolor(header_color)
					else:
						cell.set_facecolor(row_colors[k[0]%len(row_colors) ])	
				return ax

			# Change the picture theme
			jtplot.style(theme='gruvboxl',context='paper',fscale=1.6,ticks=True, grid=False, figsize=(10, 10))
			render_mpl_table(BestModelPerHour.iloc[:,:6], header_columns=0, col_width=3.5)
			# Save graph to folder
			ImagePathFile = EvaluationPathFolder + 'Best model per hour.png'
			plt.savefig(ImagePathFile,transparent=True)
			plt.close()
			plt.clf()

			# ------------------- Total evaluation plot -------------------
			# change the picture theme
			jtplot.style(theme='chesterish',context='paper',fscale=1.6,ticks=True, grid=False, figsize=(25, 25))
			# Change the picture theme
			fig = plt.figure(figsize=(25,25)) 
			x = np.arange(len(df))
			ax1 = plt.subplot(1,1,1)
			w = 0.4
			df = ResultsEval.reset_index()
			# plt.xticks(), will label the bars on x axis with the respective metrics.
			plt.xticks(x + w /2, df['Model'], rotation='vertical')
			MAE = ax1.bar(x=x, height=df['MAE'], width=w, color='b', align='center')
			# The trick is to use two different axes that share the same x axis, we have used ax1.twinx() method.
			ax2 = ax1.twinx()
			RMSE =ax2.bar(x=x + w, height=df['RMSE'], width=w,color='g',align='center')
			# Set the Y axis label.
			ax1.set_ylabel('MAE')
			ax2.set_ylabel('RMSE')
			plt.xlabel('Model')
			plt.title(str('Model evaluation'))
			# To set the legend on the plot we have used plt.legend()
			plt.legend([MAE, RMSE],['MAE', 'RMSE'], loc="upper left", prop=dict(size=30))
			# Save graph to folder
			ImagePathFile = EvaluationPathFolder + r'Model evaluation.png'
			plt.savefig(ImagePathFile, bbox_inches='tight',dpi=100, transparent=True)
			plt.close()
			plt.clf()

			# ------------------- Total normalized evaluation plot -------------------
			df = ResultsEval.reset_index()
			# change the picture theme
			jtplot.style(theme='chesterish',context='paper',fscale=1.6,ticks=True, grid=False, figsize=(25, 25))
			# Initiate figure, axes and set the size
			fig = plt.figure(figsize=(25,25)) 
			ax1 = plt.subplot(1,1,1)
			# plt.xticks(), will label the bars on x axis with the respective metrics.
			x = np.arange(len(df))
			w = 0.4
			plt.xticks(x + w /2, df['Model'], rotation='vertical')
			# create figure
			MAPE = ax1.bar(x=x, height=df['MAPE'], width=w, color='b', align='center')
			# Use two different axes that share the same x axis
			ax2 = ax1.twinx()
			sMAPE = ax2.bar(x=x+ w, height=df['sMAPE'], width=w, color='g', align='center')
			ax1.set_ylabel('MAPE')
			ax2.set_ylabel('sMAPE')
			plt.xlabel('Model')
			plt.title(str('Model evaluation'))
			# To set the legend on the plot we have used plt.legend()
			plt.legend([MAE, RMSE],['MAPE', 'sMAPE'], loc="upper left", prop=dict(size=30))
			# Save graph to folder
			ImagePathFile = EvaluationPathFolder + r'Model percentage evaluation.png'
			plt.savefig(ImagePathFile, bbox_inches='tight',dpi=100, transparent=True)
			plt.close()
			plt.clf()

			# ------------------- Residuals boxplot -------------------
			# Get right Data
			Residuals = Deviations
			# Change the picture theme
			jtplot.style(theme='grade3',context='paper',fscale=1.6,ticks=True, grid=False, figsize=(20, 20))
			# Design the Boxplot
			fig = plt.figure(figsize =(20, 15))
			ax = fig.add_subplot(111)
			Residuals.boxplot( vert=False,grid = False,fontsize=16)
			plt.title(str('Residuals distribution - boxplot '),fontsize = 20)
			ax.set_xlabel('Residuals',fontsize = 18)
			# Save graph to folder
			ImagePathFile = EvaluationPathFolder + r'Residuals evaluation.png'
			plt.savefig(ImagePathFile, bbox_inches='tight',dpi=100, facecolor = '#e6f2ff')
			plt.close()
			plt.clf()
			# ------------------- Residual Histogram -------------------
			# Histogram plot
			fig = plt.figure(figsize = (26,26))
			ax = fig.gca()
			Residuals.hist(ax = ax, bins=20)
			# Save graph to folder
			ImagePathFile = EvaluationPathFolder + r'Residuals histogram.png'
			plt.savefig(ImagePathFile, bbox_inches='tight',dpi=100, facecolor = '#e6f2ff')
			plt.figure().clear()
			plt.close()
			plt.clf()

			# ----------- Plot Best models vs Actual prices --------------
			DF.reset_index(inplace=True)
			# set Datetime as string and not datetime object that causes its extrapolation in x axis plots
			DF["Datetime"] = DF["Datetime"].map(lambda x: x.strftime('%d-%m-%Y %H:%M'))
			
			# output the three top performed models
			for i in range(0,3):
				best_model = str(ResultsEval.iloc[i,0])
				if best_model in DF.columns:		
					# Initiate figure, axes and set the size
					fig, ax = plt.subplots(figsize=(20,15))
					# Set title to graph
					#ax.set_title(str(best_model+' model vs Actual prices'), fontsize = 24)
					# Function to plot and show graph
					plt.plot(DF["Datetime"], DF[ColumPred], label ='Actual prices', color='#2A3D54')
					plt.plot(DF["Datetime"], DF[best_model], label =str(best_model+' model'),color="#8A1108")
					# label x-axis and y-axis
					ax.set_ylabel('Price [Euro/MWh]', fontsize = 18)
					ax.set_xlabel('Datetime', fontsize = 16)
					# format axis
					ax.tick_params(labelsize=16, size=4)
					ax.set_xticks(ax.get_xticks()[::24])
					plt.xticks(rotation=65)
					# Format the legend
					plt.legend(prop = { "size": 18 }, loc ="upper left")
					# Save graph to folder
					ImagePathFile = EvaluationPathFolder + 'Top_'+ str(i+1) + ' model vs Actual prices.png'
					plt.savefig(ImagePathFile, bbox_inches='tight',dpi=100, facecolor = '#FFFFFF')
					plt.figure().clear()
					plt.close()
					plt.clf()
				else:
					pass
		
		
		# -------- Summary notepad --------	   
		
		# reset index before returning
		ResultsEval.reset_index(drop=True, inplace=True)
		# find accuracy of top model vs average market price
		Accuracy_metric = ((mean(DF[ColumPred])-ResultsEval.iloc[0,1])*100/mean(DF[ColumPred])).round(2)
		# make Datetime index as column
		if 'Datetime' not in DF.columns:
			DF.reset_index(inplace=True)
		else:
			pass
		# write summary results on notebook
		notepad = [
		'Evaluation period: ' + str(DF.iloc[0,0])+ str(' until ')+ str(DF.iloc[-1,0]),
		'Days in the dataset: ' + str(DF.shape[0]/24),
		'Top model: ' + str(ResultsEval.iloc[0,0]),
		'with MAE in euro/MWh: ' + str(ResultsEval.iloc[0,1]),
		'Average Market price in euro/MWh: ' + str(round(mean(DF[ColumPred]),2)),
		'Top model calculated accuracy %: ' + str(Accuracy_metric)
		 ]
		# write on notepad
		NotepadPathFile = EvaluationPathFolder + 'Summary results.txt'
		with open(NotepadPathFile, 'w') as f:
			for line in notepad:
				f.write(line)
				f.write('\n')

		print('Evaluation completed')
		
	return (ResultsEval)


