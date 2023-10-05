import os
import numpy as np
import pandas as pd

years = list(range(2000,2023))
stations_years_dict = {}
stations_filenames_dict = {}

for year in years:
	inputdir = '/Users/brusnto/Downloads/' + str(year) + '/'
	for filename in os.listdir(inputdir):
		if filename.endswith('.CSV'):
			match = re.search(r"_A(\d{3})_", filename)
			if match:
				station = match.group(1)
				print(station)
				try:
					curr = stations_years_dict[station] 
					curr.append(year)
					stations_years_dict[station] = curr
					curr = stations_filenames_dict[station]
					curr.append(inputdir + filename)
					stations_filenames_dict[station] = curr
				except KeyError:
					stations_years_dict[station] = [year]
					stations_filenames_dict[station] = [inputdir + filename]
			else:
				raise Exception("Not found for " + filename)

stations_years_size_dict = {}
for station in stations_years_dict:
	stations_years_size_dict[station] = len(stations_years_dict[station])
	joined_df = None
	for filename in stations_filenames_dict[station]:
		outputdir = '/Users/brusnto/Downloads/joined/A' + station + '.csv'
		curr_df = pd.read_csv(filename, sep=';')
		if joined_df is None:
			joined_df = curr_df
		else:
			joined_df = pd.concat([joined_df, curr_df], ignore_index=True)
	joined_df.to_csv(outputdir, sep=';')
