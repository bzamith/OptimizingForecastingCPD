import os
import numpy as np
import pandas as pd
import re

years = list(range(2007,2025))
stations_years_dict = {}
stations_filenames_dict = {}

dir = '/Users/zamith/Downloads'
errored = dict()

for year in years:
	inputdir = f"{dir}/{str(year)}/"
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
					stations_filenames_dict[station] = [f"{inputdir}{filename}"]
			else:
				print(f"Not found for {inputdir}{filename}")
				if not station in list(errored.keys()):
					errored[station] = True

stations_years_size_dict = {}
for station in stations_years_dict:
	if not station in list(errored.keys()):
		stations_years_size_dict[station] = len(stations_years_dict[station])
		joined_df = None
		for filename in stations_filenames_dict[station]:
			outputdir = f"{dir}/joined/A{station}.csv"
			curr_df = pd.read_csv(filename, sep=';')
			if joined_df is None:
				joined_df = curr_df
			else:
				joined_df = pd.concat([joined_df, curr_df], ignore_index=True)
		joined_df.to_csv(outputdir, sep=';')

print(f"Errored: {list(errored.keys())}")