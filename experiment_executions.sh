# !/bin/bash

# UCI APPLIANCES_ENERGY
for method in Window Bin_Seg Bottom_Up; do
  for metric in L1 L2 Normal RBF Cosine Linear Clinear Rank Mahalanobis AR; do
    nice -n -10 ./run.sh execute UCI APPLIANCES_ENERGY "$method" "$metric"
  done
done

for cut in Fixed_Cut_0.0; do
  nice -n -10 ./run.sh execute UCI APPLIANCES_ENERGY Fixed_Perc "$cut"
done

# UCI METRO_TRAFFIC
for method in Window Bin_Seg Bottom_Up; do
  for metric in L1 L2 Normal RBF Cosine Linear Clinear Rank Mahalanobis AR; do
    nice -n -10 ./run.sh execute UCI METRO_TRAFFIC "$method" "$metric"
  done
done

for cut in Fixed_Cut_0.0; do
  nice -n -10 ./run.sh execute UCI METRO_TRAFFIC Fixed_Perc "$cut"
done

# UCI PRSA_BEIJING
for method in Window Bin_Seg Bottom_Up; do
  for metric in L1 L2 Normal RBF Cosine Linear Clinear Rank Mahalanobis AR; do
    nice -n -10 ./run.sh execute UCI PRSA_BEIJING "$method" "$metric"
  done
done

for cut in Fixed_Cut_0.0; do
  nice -n -10 ./run.sh execute UCI PRSA_BEIJING Fixed_Perc "$cut"
done

# UCI AIR_QUALITY
for method in Window Bin_Seg Bottom_Up; do
  for metric in L1 L2 Normal RBF Cosine Linear Clinear Rank Mahalanobis AR; do
    nice -n -10 ./run.sh execute UCI AIR_QUALITY "$method" "$metric"
  done
done

for cut in Fixed_Cut_0.0; do
  nice -n -10 ./run.sh execute UCI AIR_QUALITY Fixed_Perc "$cut"
done

# INMET SAOPAULO_SP
for method in Window Bin_Seg Bottom_Up; do
  for metric in L1 L2 Normal RBF Cosine Linear Clinear Rank Mahalanobis AR; do
    nice -n -10 ./run.sh execute INMET SAOPAULO_SP "$method" "$metric"
  done
done

for cut in Fixed_Cut_0.0; do
  nice -n -10 ./run.sh execute INMET SAOPAULO_SP Fixed_Perc "$cut"
done
