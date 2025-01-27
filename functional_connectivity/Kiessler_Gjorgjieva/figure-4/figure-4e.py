#script to produce matrices in figure 4 E
#the mice used in figure 4 E are mouse 116 (healthy), mouse 91 (sham), mouse 113 (recovery), 69 (no recovery)
#Author: Filippo Kiessler, Technische Universität München, filippo.kiessler@tum.de
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import scipy as sp
from sklearn.cluster import KMeans
from matplotlib.colors import TwoSlopeNorm

import seaborn as sns
import sys
sys.path.append('../')
from utilities import get_correlation_vector

control =  [33, 83, 91, 93, 95, 108, 111, 112, 114, 115, 116, 122]

stroke = [41, 63, 69, 85, 86, 89, 90, 110, 113]

no_recovery = [41, 63, 69, 110]
recovery = [85, 86, 89, 90, 113]
sham = [91, 111, 115, 122, 33, 83, 93, 95, 108, 112, 114, 116]

#remove unwanted mice
control.remove(112)
sham.remove(112)

categories_coarse= ['control', 'stroke']
categories_fine = ['no recovery', 'recovery', 'sham']

mouse_coarse_mapping = {}
mouse_fine_mapping = {}

for mouse in control + stroke:
    #coarse mapping
    if mouse in control:
        mouse_coarse_mapping[mouse] = 'control'
    else:
        mouse_coarse_mapping[mouse] = 'stroke'

    #fine mapping
    if mouse in sham:
        mouse_fine_mapping[mouse] = 'sham'
    elif mouse in recovery:
        mouse_fine_mapping[mouse] = 'recovery'
    elif mouse in no_recovery:
        mouse_fine_mapping[mouse] = 'no recovery'
        
def apply_function_to_cells(df1, df2, func, ignore_nan=False, ignore_inside = False):
    """
    Apply a function to corresponding cells of two dataframes.
    Assumes df1 and df2 have equal shapes!

    Parameters:
    - df1, df2: Input dataframes.
    - func: Function to apply to corresponding cells.
    - ignore_nan: If True, ignores np.NaN values in cells.

    Returns:
    - Resulting dataframe.
    """
    result_data = []
    rows, cols = df1.shape

    for i in range(rows):
        row_data = []
        for j in range(cols):
            val1, val2 = df1.iat[i, j], df2.iat[i, j]

            # Check if any value is NaN of type float
            if not ignore_inside:
                nan_condition = ignore_nan and (isinstance(val1, float) and np.isnan(val1) or
                                            isinstance(val2, float) and np.isnan(val2))
            else:
                nan_condition = ignore_nan and (np.any(np.isnan(val1)) or
                                            np.any(np.isnan(val2)))
            if nan_condition:
                cell_result = np.NaN
            else:
                cell_result = func(val1, val2)

            row_data.append(cell_result)

        result_data.append(row_data)

    result_df = pd.DataFrame(result_data, index=df1.index, columns=df1.columns)
    return result_df

#####################################
def remove_unwanted_mice(dict, mice):
    dict_copy = dict
    for mouse in mice:
        if mouse in list(dict_copy.keys()):
            del dict_copy[mouse]
    return dict_copy

def elbow_method(corrmat, max_clusters, random_state = 42):
    x = range(1,min(len(corrmat), max_clusters) + 1)
    corrmeans = []
    for k in x:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit(corrmat)
        corrmeans.append(kmeans.inertia_)
    corrmean_fit = np.polyfit(x,corrmeans, deg = 5)
    straight = np.polyfit([min(x), max(x)], [corrmeans[0], corrmeans[-1]], deg=1)
    elbow_fun = np.polysub(straight, corrmean_fit)
    res = np.polyval(elbow_fun, x)
    return np.argmax(res) + 1 #this is the elbow point, +1 because the counting starts at 0 but x starts at 1
    
def labels_cluster_elbow(corrmat, max_clusters = 25, random_state = 42):
    nclust = elbow_method(corrmat, max_clusters, random_state)
    kmeans = KMeans(n_clusters=nclust, random_state=12)
    labels = kmeans.fit_predict(corrmat)
    return labels

def cluster_pc_count(cluster_labels, pc_vec):
    nclusters = max(cluster_labels)
    if sum(pc_vec) == 0:
        return np.zeros(nclusters)
    
    x = range(1, nclusters + 1)
    cluster_pc_count = np.zeros(nclusters)
    for i in x:
        cluster_pc_count[i-1] = np.sum(pc_vec[cluster_labels == i])
    return cluster_pc_count

def get_correlation_matrices_maxneurons(traces):
    #retrieve the correlation matrix of all pairwise neuron correlations across days
    correlation_matrices_mice = {}
    for mouse, t in traces.items():
        mouse_result = {}
        traces_clean = t.dropna(axis=1, how='all')
        day_pairs = itertools.product(traces_clean.columns, traces_clean.columns)
        for p in day_pairs:
            traces_pair = traces_clean.loc[:,p].dropna()
            cm1 = np.corrcoef(np.vstack(traces_pair.iloc[:,0]))#get the correlation matrices of the two sessions
            cm2 = np.corrcoef(np.vstack(traces_pair.iloc[:,1]))#get the correlation matrices of the two sessions
            mouse_result[p] = (cm1, cm2)
        correlation_matrices_mice[mouse] = mouse_result
    return correlation_matrices_mice

def plot_matrices_with_masked_diagonals(mat1, mat2, mouse, day1, day2):
    # Mask the diagonals of both matrices
    mat1_masked = np.ma.masked_array(mat1, mask=np.eye(mat1.shape[0], dtype=bool))
    mat2_masked = np.ma.masked_array(mat2, mask=np.eye(mat2.shape[0], dtype=bool))
    res = sp.stats.pearsonr(get_correlation_vector(mat1_masked), get_correlation_vector(mat2_masked))
    
    vmin = min(np.min(mat1_masked), np.min(mat2_masked))
    vmax = max(np.max(mat1_masked), np.max(mat2_masked))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    
    # Plot first matrix
    cax1 = axs[0].imshow(mat1_masked, aspect='auto', cmap='coolwarm', norm = norm)
    axs[0].set_title(f'Mouse {mouse}, day {day1}', fontsize = 10)
    axs[0].set_axis_off()
    # Plot second matrix
    cax2 = axs[1].imshow(mat2_masked, aspect='auto', cmap='coolwarm', norm = norm)
    axs[1].set_title(f'Mouse {mouse}, day {day2}, corr = {res.statistic:.4f}, pval = {res.pvalue:.4f}', fontsize = 10)
    axs[1   ].set_axis_off()

    # Remove ticks (optional)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    # Create a colorbar
    fig.subplots_adjust(0,0.05, 0.96, 0.95)
    cbar_ax = fig.add_axes([0.98, 0.01, 0.018, 0.9])
    fig.colorbar(cax1, cax=cbar_ax, orientation='vertical')
    
    fig.savefig(f'{savestr}/mouse-{mouse}-days-{day1}-{day2}-{args.dataset}.png', dpi=300)
    fig.savefig(f'{savestr}/mouse-{mouse}-days-{day1}-{day2}-{args.dataset}.svg')
    return fig

class Arguments:
    #for debugging
    def __init__(self, dset, corr):
        self.dataset = dset
        self.correlation = corr

if __name__ == '__main__':
    np.random.seed(0)
    os.chdir('/home/ga48kuh/NASgroup/labmembers/filippokiessler/wahl-colab')
    scriptname = os.path.basename(__file__)[:-3]
    
    args = Arguments('sa', 'pearson') #for debugging
       
    savestr_base = f'code/paper-review/figure-4/outputs/{scriptname}'
    if not os.path.isdir(savestr_base):
        os.mkdir(savestr_base)
    
    savestr = f'{savestr_base}/{args.dataset}'
    if not os.path.isdir(savestr):
        os.mkdir(savestr)

    plt.style.use('code/paper-review/plotstyle.mplstyle')
    
    if args.dataset == 'dff':
        traces_path = 'data/neural-data/neural-data-clean/dff_tracked_normal.pkl'
    elif args.dataset == 'decon':
        traces_path = 'data/neural-data/neural-data-clean/decon_tracked_normal.pkl'
    elif args.dataset == 'sa':
        traces_path = 'data/neural-data/spatial_activity_maps_dff.pkl'
    else:
        raise ValueError('Dataset for correlation matrices of neural traces does not exist!')
    
    unwanted_mice = [121, 112]
    with open(traces_path, 'rb') as file:
        traces = pickle.load(file)
    traces = remove_unwanted_mice(traces, unwanted_mice)
    
    corrmats_day_pais_maxneurons = get_correlation_matrices_maxneurons(traces)

    def get_common_matrices(mouse, d1, d2):
        return corrmats_day_pais_maxneurons[mouse][(d1, d2)]
        

    def sorted_cluster_matrices(mouse, d1, d2):
        cm1, cm2 = get_common_matrices(mouse, d1, d2)
        labels = labels_cluster_elbow(cm1)
        order = np.argsort(labels)
        d1_clust = cm1[order][:,order]
        d2_clust = cm2[order][:,order]
        return d1_clust, d2_clust


    def plot_grid_correlation_matrices(matrix_mouse_dict):
        np.random.seed(0)#set random seed for plotting
        #make one consistent figure using days comparable across different mice (i.e. all in late post-stroke)
        matrices = []
        for period, d in matrix_mouse_dict.items():
            matrices.append(sorted_cluster_matrices(d['mouse'], d['d1'], d['d2']))

        matrices_masked = []
        #mask the diagonals of the matrices
        for m1, m2 in matrices:
                mat1_masked = np.ma.masked_array(m1, mask=np.eye(m1.shape[0], dtype=bool))
                mat2_masked = np.ma.masked_array(m2, mask=np.eye(m2.shape[0], dtype=bool))
                matrices_masked.append((mat1_masked, mat2_masked))

        #get a common minimum and maximum for shared colorbar of the matrices:
        minimas = list(map(lambda p: np.min(p), [item for pair in matrices_masked for item in pair]))
        maximas = list(map(lambda p: np.max(p), [item for pair in matrices_masked for item in pair]))
        mi, ma = np.min(minimas), np.max(maximas)

        fig, axs = plt.subplots(2, 4, figsize = (10.5, 5.2), layout = 'compressed')
        for (i, pair), (k,v) in zip(enumerate(matrices_masked),matrix_mouse_dict.items()):
            mouse = v['mouse']
            d1 = v['d1']
            d2 = v['d2']
            for j, m in enumerate(pair):
                im = axs[j,i].imshow(m, vmin = mi, vmax = ma, cmap = 'coolwarm')
                axs[j,i].set_axis_off()
            axs[0,i].set_title(f'M {mouse}, d {d1} ({k})', fontsize = 10)
            axs[1,i].set_title(f'Day {d2}', fontsize = 10)

        fig.subplots_adjust(0, 0.01, 0.97, 0.96)
        cbar_ax = fig.add_axes([0.98, 0.01, 0.01, 0.943])
        fig.colorbar(im, cax=cbar_ax, orientation='vertical')

        savename = 'mice-healthy-m{}-{}-{}-sham-m{}-{}-{}-rec-m{}-{}-{}-norec-m{}-{}-{}'.format(*[v for item in matrix_mouse_dict.values() for v in item.values()])
        
        fig.savefig(f'{savestr}/{savename}-{args.dataset}.png', dpi=300)
        fig.savefig(f'{savestr}/{savename}-{args.dataset}.svg')
        fig.savefig(f'{savestr}/{savename}-{args.dataset}.pdf')

    matrix_mouse_dict_2 = {
        'healthy': {'mouse': 116, 'd1': -1, 'd2': 0},
        'sham': {'mouse': 91, 'd1': 25, 'd2': 27},
        'recovery': {'mouse': 113, 'd1': 24, 'd2': 27},
        'no recovery': {'mouse': 69, 'd1': 12, 'd2': 15}
    }
    plot_grid_correlation_matrices(matrix_mouse_dict_2) #EITHER THIS

    matrix_mouse_dict_3 = {
        'healthy': {'mouse': 116, 'd1': -1, 'd2': 0},
        'sham': {'mouse': 91, 'd1': 9, 'd2': 12},
        'recovery': {'mouse': 113, 'd1': 9, 'd2': 12},
        'no recovery': {'mouse': 69, 'd1': 9, 'd2': 12}
    }
    plot_grid_correlation_matrices(matrix_mouse_dict_3) #OR THIS
