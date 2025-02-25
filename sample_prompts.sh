
parallel --jobs 4 './run.sh execute INMET SAOPAULO_SP {1} {2}' ::: Window Bin_Seg Bottom_Up ::: L1 L2 Normal RBF Cosine Linear Clinear Rank Mahalanobis AR
parallel --jobs 4 './run.sh execute INMET SAOPAULO_SP Fixed_Perc {1}' ::: Fixed_Cut_0.0 Fixed_Cut_0.2 Fixed_Cut_0.5 Fixed_Cut_0.7
parallel --jobs 4 './run.sh execute UCI AIR_QUALITY {1} {2}' ::: Window Bin_Seg Bottom_Up ::: L1 L2 Normal RBF Cosine Linear Clinear Rank Mahalanobis AR
parallel --jobs 4 './run.sh execute UCI AIR_QUALITY Fixed_Perc {1}' ::: Fixed_Cut_0.0 Fixed_Cut_0.2 Fixed_Cut_0.5 Fixed_Cut_0.7
parallel --jobs 4 './run.sh execute UCI PRSA_BEIJING {1} {2}' ::: Window Bin_Seg Bottom_Up ::: L1 L2 Normal RBF Cosine Linear Clinear Rank Mahalanobis AR
parallel --jobs 4 './run.sh execute UCI PRSA_BEIJING Fixed_Perc {1}' ::: Fixed_Cut_0.0 Fixed_Cut_0.2 Fixed_Cut_0.5 Fixed_Cut_0.7
parallel --jobs 4 './run.sh execute UCI APPLIANCES_ENERGY {1} {2}' ::: Window Bin_Seg Bottom_Up ::: L1 L2 Normal RBF Cosine Linear Clinear Rank Mahalanobis AR
parallel --jobs 4 './run.sh execute UCI APPLIANCES_ENERGY Fixed_Perc {1}' ::: Fixed_Cut_0.0 Fixed_Cut_0.2 Fixed_Cut_0.5 Fixed_Cut_0.7
parallel --jobs 4 './run.sh execute UCI METRO_TRAFFIC {1} {2}' ::: Window Bin_Seg Bottom_Up ::: L1 L2 Normal RBF Cosine Linear Clinear Rank Mahalanobis AR
parallel --jobs 4 './run.sh execute UCI METRO_TRAFFIC Fixed_Perc {1}' ::: Fixed_Cut_0.0 Fixed_Cut_0.2 Fixed_Cut_0.5 Fixed_Cut_0.7
