import os
import numpy as np
import pandas as pd

inputdir = '/Users/zamith/Downloads/2007'

for filename in os.listdir(inputdir):
	if filename.endswith('.CSV'):
		print(f"Processing: {filename}")
		x = open(f"{inputdir}/{filename}", encoding = "ISO-8859-1")
		s = x.read().replace(",", ".").replace("/", "-")  
		x.close()
		x = open(f"{inputdir}/{filename}", "w", encoding = "ISO-8859-1")
		x.write(s)
		x.close()
		df = pd.read_csv(f"{inputdir}/{filename}", sep=';', skiprows=8, encoding = "ISO-8859-1")
		orig_var_cols = ['PRECIPITAÇÃO TOTAL. HORÁRIO (mm)', 'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO. HORARIA (mB)', 'TEMPERATURA DO AR - BULBO SECO. HORARIA (°C)', 'UMIDADE RELATIVA DO AR. HORARIA (%)', 'VENTO. VELOCIDADE HORARIA (m-s)']
		if 'Data' in df.columns:
			df = df[['Data'] + orig_var_cols]
		else:
			df = df[['DATA (YYYY-MM-DD)'] + orig_var_cols]
		date_col = ['Date']
		var_cols = ['Precipitacao', 'Pressao Atmosferica', 'Temperatura', 'Umidade Relativa', 'Velocidade Vento']
		df.columns = [date_col + var_cols]
		df[var_cols] = df[var_cols].apply(pd.to_numeric)
		df.to_csv(f"{inputdir}/{filename}", sep=';')
		df = pd.read_csv(f"{inputdir}/{filename}", sep=';')
		df = df.replace(-9999.0, np.nan)
		df['ds'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
		df = df.groupby(df['ds']).agg({'Precipitacao': 'sum', 'Pressao Atmosferica': 'mean', 'Temperatura': 'mean', 'Umidade Relativa': 'mean', 'Velocidade Vento': 'mean'})
		df.to_csv(f"{inputdir}/{filename}", sep=';')
		print(f"Processed: {filename}")