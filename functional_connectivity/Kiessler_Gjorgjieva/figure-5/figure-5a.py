#figure 5A (but percentile does not match the text: 0.8 instead of 0.95, as reported. however, all the data is correct)
#NOTE: the output graph from this script was modified in a vector graphics software and the colors changed to match the rest of the colorscheme in the manuscript
#Author: Filippo Kiessler, Technische Universität München, filippo.kiessler@tum.de
import pickle
import numpy as np

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.colors import ListedColormap
import networkx as nx

import pandas as pd

import sys
sys.path.append('../')#to include 'utilities.txt'
from utilities import df_corr_result, get_correlation_vector

import argparse
np.random.seed(42)

control =  [33, 83, 91, 93, 95, 108, 111, 112, 114, 115, 116, 122]

stroke = [41, 63, 69, 85, 86, 89, 90, 110, 113]

no_recovery = [41, 63, 69, 110]
recovery = [85, 86, 89, 90, 113]
sham = [91, 111, 115, 122, 33, 83, 93, 95, 108, 112, 114, 116]

#remove unwanted mice
control.remove(112)
sham.remove(112)

unwanted_mice = [121, 112]

def remove_mice_from_df(df, mice, axis = 0):
    filtered_df = df
    for m in mice:
        if m in df.index:
            filtered_df.drop(m, axis = axis, inplace = True)
    return filtered_df

def get_unique_cell_pair_categories(pc_df):
    unique_values_in_cells = pc_df.applymap(lambda x: np.unique(x), na_action = 'ignore')
    flattened_uniques_withnans = unique_values_in_cells.apply(lambda col: np.hstack(col))
    unique_colvals = flattened_uniques_withnans.apply(np.unique)
    valarray_colvals = np.hstack(unique_colvals)
    unique_general = np.unique(valarray_colvals)
    nonan_unique_final = unique_general[~np.isnan(unique_general)].astype(int)
    return nonan_unique_final
    
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

def plot_graph(graph, positions, scale = 100,
               node_cmap = ListedColormap(['#92a5ad', '#d93b29']),#gray for non coding, red for pc
               edge_cmap = ListedColormap(['#92a5ad', '#26bf5c', '#d93b29'])):
    
    weightlist = np.array(list(nx.get_edge_attributes(graph, 'weight').values()))
    alpha_from_weights = (weightlist-np.min(weightlist))/(np.max(weightlist)-np.min(weightlist))
    
    node_type_array = np.array(list(nx.get_node_attributes(graph, 'cell-type').values()))
    edge_type_array = np.array(list(nx.get_edge_attributes(graph, 'conn-type').values()))
    
    fig, ax = plt.subplots()
    nodes = nx.draw_networkx_nodes(graph, pos = positions,
                                   ax=ax,
                                   node_size = 12,
                                   node_color=node_type_array,
                                   cmap = node_cmap,
                                   edgecolors='k',
                                   linewidths= 0.1)
    
    edges = nx.draw_networkx_edges(graph, pos = positions,
        ax=ax,
        width = weightlist,
        edge_cmap = edge_cmap,
        edge_color = edge_type_array,
        alpha = alpha_from_weights)

    ax.axis('off')
    
    scalebar = AnchoredSizeBar(ax.transData,
                           scale, f'{scale} um', 'lower left', 
                           pad=0.1,
                           color='black',
                           frameon=False,
                           size_vertical=1)

    ax.add_artist(scalebar)
    
    return fig, ax

def construct_graph(cmat, pc_div, edges_types, quantile = 0.8):
    
    #cmat: correlation matrixto serve as adjacency matrix. ignore self-loops
    #quantile: weights (cmat entries) below this quantile will not be included as edges
    cmat_qt = np.quantile(get_correlation_vector(cmat), quantile)
    cmat_greater_qt = np.copy(cmat)
    cmat_greater_qt[cmat <= cmat_qt] = np.NaN
    idx = np.indices(edges_types.shape)

    g = nx.Graph(cmat_greater_qt)
    nx.set_node_attributes(g, {u:pc_div[u] for u in range(len(pc_div))}, name = 'cell-type')#set cell type
    nx.set_edge_attributes(g, {(u,v):edges_types[u,v] for u,v in zip(idx[0].flatten(), idx[1].flatten())}, name='conn-type')#set type of connection
    g.remove_edges_from(nx.selfloop_edges(g)) #remove self loops
    g.remove_edges_from([(u, v) for u, v, data in g.edges(data=True) if np.isnan(data['weight'])])
    return g


def plot_graph_mouse_session(cmat, pc_div, edgetype, quantile = 0.8,
                             node_cmap = ListedColormap(['#8c8c8c', '#d93b29']),
                             edge_cmap = ListedColormap(['#8c8c8c', '#26bf5c', '#d93b29']),
                             legend = True):

    graph = construct_graph(cmat, pc_div, edgetype, quantile = quantile)
    fig, ax = plot_graph(graph, positions,
                    node_cmap = node_cmap, 
                    edge_cmap = edge_cmap)
    ax.set_title(f'Mouse {sample_mouse}, day {sample_session}, corr. quantile = {quantile}')
    
    #generate legend
    if legend:
        legend_elements = [
            Line2D([0], [0], marker = 'o', linestyle = 'none', color='#8c8c8c', lw=1, markeredgewidth = 0.2, markeredgecolor='k', label='Nc'),
            Line2D([0], [0], marker = 'o', linestyle = 'none', color='#d93b29', lw=1, markeredgewidth = 0.2, markeredgecolor='k', label='Pc'),
            Line2D([0], [0], color='#8c8c8c', lw=1, label='Nc-Nc'),
            Line2D([0], [0], color='#26bf5c', lw=1, label='Nc-Pc'),
            Line2D([0], [0], color='#d93b29', lw=1, label='Pc-Pc')]
        ax.legend(handles = legend_elements, loc = 'upper right', fontsize = 8, frameon = False)
    
    fig.savefig(f'{savestr}/visualisation-graph-neurons-mouse-{sample_mouse}-day-{sample_session}-quantile-{quantile}.png')
    fig.savefig(f'{savestr}/visualisation-graph-neurons-mouse-{sample_mouse}-day-{sample_session}-quantile-{quantile}.svg')
    
    return fig

def remove_unwanted_mice(dict, mice):
    dict_copy = dict
    for mouse in mice:
        if mouse in list(dict_copy.keys()):
            del dict_copy[mouse]
    return dict_copy


class Arguments:
    def __init__(self, dset):
        self.dataset = dset

if __name__ == '__main__':
    
    os.chdir('/home/ga48kuh/NASgroup/labmembers/filippokiessler/wahl-colab')
    scriptname = os.path.basename(__file__)[:-3]
    
    args = Arguments('dff')
    
    savestr_base = f'code/paper-review/figure-5/outputs/{scriptname}'
    if not os.path.isdir(savestr_base):
        os.mkdir(savestr_base)
    
    savestr = f'{savestr_base}/{args.dataset}'
    if not os.path.isdir(savestr):
        os.mkdir(savestr)
    
    plt.style.use('code/paper-review/plotstyle.mplstyle')
    
    #dataset selection:
    if args.dataset == 'dff':
        traces_corrmat_path = 'code/paper-review/figure-5/outputs/precompute-decon-dff-calc-corrmat-distancemat-all-cells/correlation-mat-unsorted-dff.pkl' #df/f traces
    else:
        raise ValueError('Dataset for correlation matrices of neural traces does not exist!')
    
    with open(traces_corrmat_path, 'rb') as file:
        traces_corrmat_dict = pickle.load(file)
    
    pc_division_path = 'code/paper-review/figure-5/outputs/precompute-decon-dff-calc-corrmat-distancemat-all-cells/mouse-cell-pair-identifiers.pkl' #cell pair type identifiers (precomputed)
    with open(pc_division_path, 'rb') as file:
        pc_classes_pairs_matrix =  pickle.load(file)
    
    pc_classes_pairs_matrix = remove_mice_from_df(pc_classes_pairs_matrix, unwanted_mice)
    
    pc_path = 'data/neural-data/neural-data-clean/is_pc_all_cells.pkl' #cell type identifiers
    with open(pc_path, 'rb') as file:
        pc_classes =  pickle.load(file)
    pc_classes = {k: v.reset_index(drop=True) for k, v in pc_classes.items()}
    pc_classes = remove_unwanted_mice(pc_classes, unwanted_mice)
    

    #load coordinates
    coords_path = 'data/neural-data/neural-data-clean/cell_coords_all_cells.pkl'
    with open(coords_path, 'rb') as file:
        coords = pickle.load(file)
    coords ={k: v.reset_index(drop=True) for k, v in coords.items()}
    coords = remove_unwanted_mice(coords, unwanted_mice)
    
    #get offdiagonals of correlation matrices
    traces_corrmat_dict_filtered = remove_unwanted_mice(traces_corrmat_dict, unwanted_mice)#remove mouse 121 and 112.
    filtered_corrmat_traces_df = df_corr_result(traces_corrmat_dict_filtered)
    
    #remap the cell pair categories to place cell and non-place cell
    remapped_final_pc_vec = pc_classes_pairs_matrix.applymap(get_correlation_vector, na_action = 'ignore')
    #compute dataframe of vectors for correlation statistic for every pair category:
    unique_categories = get_unique_cell_pair_categories(remapped_final_pc_vec)#the ordering of these unique values applies to all contents of the
    #derivative of the apply_function_to_cells function
        
    #data preparation THIS!! 
    sample_mouse = 115
    sample_session = 3
    cmat = filtered_corrmat_traces_df.loc[sample_mouse, sample_session]
    
    
    positions = coords[sample_mouse][sample_session].dropna().reset_index(drop = True)
    pc_div = pc_classes[sample_mouse][sample_session].dropna().reset_index(drop = True)
    edgetype = pc_classes_pairs_matrix.loc[sample_mouse,sample_session]#could colr edges by cell pair types
    
    fig = plot_graph_mouse_session(cmat, pc_div, edgetype, quantile = 0.8)
    fig = plot_graph_mouse_session(cmat, pc_div, edgetype, quantile = 0.95)
    
    #this should be done with tracked cells across multiple days to visualise the change in neuronal rewiring!