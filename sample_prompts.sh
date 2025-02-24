
parallel --jobs 4 './run.sh execute INMET BRASILIA_DF {1} {2}' ::: Window Bin_Seg Bottom_Up ::: L1 L2 Normal RBF Cosine Linear Clinear Rank Mahalanobis AR
parallel --jobs 4 './run.sh execute INMET BRASILIA_DF Fixed_Perc {1}' ::: Fixed_Cut_0.0
parallel --jobs 4 './run.sh execute INMET VITORIA_ES {1} {2}' ::: Window Bin_Seg Bottom_Up ::: L1 L2 Normal RBF Cosine Linear Clinear Rank Mahalanobis AR
parallel --jobs 4 './run.sh execute INMET VITORIA_ES Fixed_Perc {1}' ::: Fixed_Cut_0.0
parallel --jobs 4 './run.sh execute INMET PORTOALEGRE_RS {1} {2}' ::: Window Bin_Seg Bottom_Up ::: L1 L2 Normal RBF Cosine Linear Clinear Rank Mahalanobis AR
parallel --jobs 4 './run.sh execute INMET PORTOALEGRE_RS Fixed_Perc {1}' ::: Fixed_Cut_0.0
parallel --jobs 4 './run.sh execute INMET SAOPAULO_SP {1} {2}' ::: Window Bin_Seg Bottom_Up ::: L1 L2 Normal RBF Cosine Linear Clinear Rank Mahalanobis AR
parallel --jobs 4 './run.sh execute INMET SAOPAULO_SP Fixed_Perc {1}' ::: Fixed_Cut_0.0
parallel --jobs 4 './run.sh execute UCI AIR_QUALITY {1} {2}' ::: Window Bin_Seg Bottom_Up ::: L1 L2 Normal RBF Cosine Linear Clinear Rank Mahalanobis AR
parallel --jobs 4 './run.sh execute UCI AIR_QUALITY Fixed_Perc {1}' ::: Fixed_Cut_0.0
parallel --jobs 4 './run.sh execute UCI PRSA_BEIJING {1} {2}' ::: Window Bin_Seg Bottom_Up ::: L1 L2 Normal RBF Cosine Linear Clinear Rank Mahalanobis AR
parallel --jobs 4 './run.sh execute UCI PRSA_BEIJING Fixed_Perc {1}' ::: Fixed_Cut_0.0
