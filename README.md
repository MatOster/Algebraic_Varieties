# Algebraic_Varieties
Stochastical methods in algebraic geometry and material science. First using PCA to analyze real algebraic varieties. Later using artificial neural networks to find some kind of representation of algebraic variety and analyze material properties form this representation.

Idea for PCA:
Use something like persistent PCA (as in Persistent Homology), i.e. make a stochastical analysis of the variety by changing the diameter of sample cloud and check how certain ratios (for example of the eigenvalues) change with size of samples.


Stochastical_Analysis executes PCA on the inverted BMN Structure 

plot_BMN plots a sample cloud of the algebraic variety induced by the inverted BMN structure after intersecting with 30 random planes and then using results from PCA to project to the 3 most important coordinate directions
