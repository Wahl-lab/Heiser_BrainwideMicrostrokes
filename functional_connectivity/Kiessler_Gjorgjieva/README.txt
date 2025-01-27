Author: Filippo Kiessler, Technische Universität München, filippo.kiessler@tum.de
both the folders figure4/ and figure/5 need an outputs/ subfolder, respectively

figure-4:
Generates some of the plots of figure 4, i.e. figures 4e, 4f and 4g.
figure-4e.py: generates the matrices of pearson correlation of spatially binned DF/F for 4 different mice from healthy, sham, recovery and no-recovery mice.
figure-4-extract-similarity-matrices.py: calculates the data that is plotted in 4f and 4g


figure-5:
Generates some of the plots of figure 5, namely figures 5a, 5c and 5d.
Start by executing precompute-decon-dff-calc-corrmat-distancemat-all-cells.py to precompute the data used in figure-5a.py and figure-5cd.py.
figure-5a.py: generates the graph schematic of figure 5a.
figure-5cd.py: calculates the data that is used in figures 5c and d. the output is a .csv file that is further to be analysed with a statistics software

utilities.py: utility functions for the above scripts
plotstyle.mplstyle: matplotlib stylesheet