# # !/bin/bash

# # Removed: Clinear, Cosine, Mahalanobis

# UCI APPLIANCES_ENERGY
for method in Window Bin_Seg Bottom_Up; do
  for metric in L1 L2 Normal RBF Linear Rank AR; do
    nice -n -10 ./run.sh execute UCI APPLIANCES_ENERGY "$method" "$metric"
  done
done

nice -n -10 ./run.sh execute UCI APPLIANCES_ENERGY Fixed_Perc Fixed_Cut_0.0

# UCI METRO_TRAFFIC
for method in Window Bin_Seg Bottom_Up; do
  for metric in L1 L2 Normal RBF Linear Rank AR; do
    nice -n -10 ./run.sh execute UCI METRO_TRAFFIC "$method" "$metric"
  done
done

nice -n -10 ./run.sh execute UCI METRO_TRAFFIC Fixed_Perc Fixed_Cut_0.0

# UCI PRSA_BEIJING
for method in Window Bin_Seg Bottom_Up; do
  for metric in L1 L2 Normal RBF Linear Rank AR; do
    nice -n -10 ./run.sh execute UCI PRSA_BEIJING "$method" "$metric"
  done
done

nice -n -10 ./run.sh execute UCI PRSA_BEIJING Fixed_Perc Fixed_Cut_0.0

# UCI AIR_QUALITY
for method in Window Bin_Seg Bottom_Up; do
  for metric in L1 L2 Normal RBF Linear Rank AR; do
    nice -n -10 ./run.sh execute UCI AIR_QUALITY "$method" "$metric"
  done
done

nice -n -10 ./run.sh execute UCI AIR_QUALITY Fixed_Perc Fixed_Cut_0.0

./run.sh execute UCI APPLIANCES_ENERGY Window L1