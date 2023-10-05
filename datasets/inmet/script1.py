import os
import numpy as np
import pandas as pd

inputdir = '/Users/brusnto/Downloads/2000'

for filename in os.listdir(inputdir):
	if filename.endswith('.CSV'):
		x = open(filename, encoding = "ISO-8859-1")
		s = x.read().replace(",", ".") 
		x.close()
		x = open(filename,"w", encoding = "ISO-8859-1")
		x.write(s)
		x.close()
		date_col = ['Date']
		var_cols = ['Precipitacao', 'Pressao Atmosferica', 'Temperatura', 'Umidade Relativa', 'Velocidade Vento']
		df = pd.read_csv(filename, sep=';', skiprows=8, encoding = "ISO-8859-1")
		df = df[['DATA (YYYY-MM-DD)', 'PRECIPITAÇÃO TOTAL. HORÁRIO (mm)', 'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO. HORARIA (mB)', 'TEMPERATURA DO AR - BULBO SECO. HORARIA (°C)', 'UMIDADE RELATIVA DO AR. HORARIA (%)', 'VENTO. VELOCIDADE HORARIA (m/s)']]
		df.columns = [date_col + var_cols]
		df[var_cols] = df[var_cols].apply(pd.to_numeric)
		df.to_csv(filename, sep=';')
		df = pd.read_csv(filename, sep=';')
		df = df.replace(-9999.0, np.nan)
		df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
		df = df.groupby(df['Date']).agg({'Precipitacao': 'sum', 'Pressao Atmosferica': 'mean', 'Temperatura': 'mean', 'Umidade Relativa': 'mean', 'Velocidade Vento': 'mean'})
		df.to_csv(filename, sep=';')
		print("Processed: "+filename)