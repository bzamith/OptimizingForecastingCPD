import os
import numpy as np
import pandas as pd

inputdir = '/Users/zamith/Downloads/joined/'

missing_df = pd.DataFrame(
	{
		'filename': [],
		'dataset_missing_%': [],
		'dataset_size': [],
		'dataset_first_date': []
	}
)

for filename in os.listdir(inputdir):
	if filename.endswith('.csv'):
		df = pd.read_csv(f"{inputdir}{filename}", sep=';')
		missing_perc = (sum([True for idx,row in df.iterrows() if any(row.isnull())])/df.shape[0])*100
		missing_df_ = pd.DataFrame(
			{
				'filename': [filename],
				'dataset_missing_%': [round(missing_perc, 2)],
				'dataset_size': [df.shape[0]],
				'dataset_first_date': [df['ds'].iloc[0]]
			}
		)
		missing_df = pd.concat([missing_df, missing_df_], axis=0)

missing_df.to_excel(f"{inputdir}missing_df.xlsx")
