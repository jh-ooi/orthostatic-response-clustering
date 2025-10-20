# orthostatic-response-clustering
This repository contains the final 14-feature matrix and analysis scripts used in the study:
"Cluster-Based Insights into Cardiovascular and Autonomic Responses to Head-Up Tilt in Hypertension."

Data:/
├── 14_Features_Matrix.mat      # Final de-identified dataset (14 features) with group label (0: CON; 1: HTN)

clustering_scripts:/
├── PCA_ClusteringsAlgorithm.mlx    # PCA-reduced data with K-means, Fuzzy C-means, and Hierarchical clustering
├── UMAP_ClusteringsAlgorithm.mlx   # UMAP-reduced data with K-means, Fuzzy C-means, and Hierarchical clustering
├── tSNE_visualization.mlx          # t-SNE visualization of the original 14D feature space
├── validate_kmeans_on_reduced.m    # K-means validation across k = 2–4 (Silhouette, DBI)
├── fuzzy_validate_on_reduced.m     # Fuzzy C-means validation across k = 2–4 (Silhouette, DBI)
└── hierarchical_validate_ward.m    # Hierarchical clustering with Ward’s linkage validation

analysis scripts:/
├── Analysis_14selectedFeatures    # Post-clustering analysis on 14 features and radar plot generation


Dependencies
1. MATLAB R2024a
2. Statistics and Machine Learning Toolbox
3. UMAP Toolbox for MATLAB (Connor Meehan, Jonathan Ebrahimian, Wayne Moore, and Stephen Meehan (2025). Uniform Manifold Approximation and Projection (UMAP) (https://www.mathworks.com/matlabcentral/fileexchange/71902), MATLAB Central File Exchange.)
4. Fuzzy Logic Toolbox (for FCM)
