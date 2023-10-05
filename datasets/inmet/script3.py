import os
import numpy as np
import pandas as pd

inputdir = '/Users/brusnto/Downloads/joined/'

for filename in os.listdir(inputdir):
	if filename.endswith('.csv'):
		df = pd.read_csv(inputdir + filename, sep=';')
		missing_per = (sum([True for idx,row in df.iterrows() if any(row.isnull())])/df.shape[0])*100
		print(filename + " - " + str(missing_per) + " - " + df['Date'].iloc[0] + " - " + str(df.shape[0]))
