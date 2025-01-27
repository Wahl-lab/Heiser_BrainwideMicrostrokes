#script for figures 4 F,G (only the data, the plotting is done in MATLAB)
#Author: Filippo Kiessler, Technische Universität München, filippo.kiessler@tum.de

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import itertools
import pandas as pd
import scipy as sp


#mouse groups
#coarse
control =  [33, 83, 91, 93, 95, 108, 111, 112, 114, 115, 116, 122]
stroke = [41, 63, 69, 85, 86, 89, 90, 110, 113]

#fine
no_recovery = [41, 63, 69, 110]
recovery = [85, 86, 89, 90, 113]
sham = [91, 111, 115, 122, 33, 83, 93, 95, 108, 112, 114, 116]

#remove unwanted mice (exclude from the analysis)
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
        
def mapmouse(mouse):
    if mouse in no_recovery:
        return 'no recovery'
    elif mouse in recovery:
        return 'recovery'
    else:
        return 'sham'

#take the lower (or upper, does not matter, just be consistent!) of the unclustered correlation matrices, excluding the diagonal elements, and flatten them
def get_correlation_vector(cell):
    return cell[np.tril_indices_from(cell, k = -1)] #distance vector, i.e. flattened lower triangle of symmetric correlation matrix, excluding diagonal elements


#####################################
def remove_unwanted_mice(dict, mice):
    dict_copy = dict
    for mouse in mice:
        if mouse in list(dict_copy.keys()):
            del dict_copy[mouse]
    return dict_copy

def get_correlation_of_correlation_traces_maxneurons(traces, correlation = 'pearson'):
    #retrieve the correlation matrix of all pairwise neuron correlations across days
    correlation_of_correlations_matrices = {}
    for mouse, t in traces.items():
        mouse_result = {}
        traces_clean = t.dropna(axis=1, how='all')
        day_pairs = itertools.product(traces_clean.columns, traces_clean.columns)
        index_dict = {d:c for c,d in enumerate(traces_clean.columns)} #mapping of columns of correlation matrix to days
        corrmat = np.zeros((len(traces_clean.columns),len(traces_clean.columns)))
        pval_mat = np.zeros((len(traces_clean.columns),len(traces_clean.columns)))
        num_neurons_mat = np.zeros((len(traces_clean.columns),len(traces_clean.columns)))
        for p in day_pairs:
            traces_pair = traces_clean.loc[:,p].dropna()
            cm1 = np.corrcoef(np.vstack(traces_pair.iloc[:,0]))#get the correlation matrices of the two sessions
            cm2 = np.corrcoef(np.vstack(traces_pair.iloc[:,1]))#get the correlation matrices of the two sessions
            diag1 = get_correlation_vector(cm1)
            diag2 = get_correlation_vector(cm2)
            if correlation == 'pearson':
                res = sp.stats.pearsonr(diag1,diag2)
            elif correlation == 'spearman':
                res = sp.stats.spearmanr(diag1,diag2)
            elif correlation == 'cosine':
                res = 1 - sp.spatial.distance.cosine(diag1, diag2) #cosine-similarity, = 1- cosine distance
            else:
                raise NotImplementedError(f"Not implemented correlation type {correlation}")
            sess_corrcoef = res.statistic if correlation in ['pearson', 'spearman'] else res
            pval = res.pvalue if correlation in ['pearson', 'spearman'] else np.NaN
            pval_mat[index_dict[p[0]],index_dict[p[1]]] = pval
            corrmat[index_dict[p[0]],index_dict[p[1]]] = sess_corrcoef
            num_pairs = len(diag1)
            num_neurons = int(0.5*(1+np.sqrt(1+8*num_pairs))) #solve num_pairs = 0.5 *(n-1)*n where n is num neurons
            num_neurons_mat[index_dict[p[0]],index_dict[p[1]]] = num_neurons
        
        mouse_result['days'] = traces_clean.columns
        mouse_result['cmat'] = corrmat
        mouse_result['num-neurons'] = num_neurons_mat
        mouse_result['days-idx-mapping'] = index_dict
        correlation_of_correlations_matrices[mouse] = mouse_result
    
    return correlation_of_correlations_matrices

def export_to_df(corrs_of_corrs):
    savestr_mat = f'{savestr}/corrs-of-corrs'
    
    if not os.path.isdir(savestr_mat):
        os.mkdir(savestr_mat)
    for mouse, obj in corrs_of_corrs.items():
        days = obj['days']
        mat = obj['cmat']
        df = pd.DataFrame(data = mat, index = days, columns = days)
        df.to_csv(f'{savestr_mat}/mouse-{mouse}-similarity-{args.correlation}-{args.dataset}.csv')

class Arguments:
    def __init__(self, dset, corr):
        self.dataset = dset
        self.correlation = corr

if __name__ == '__main__':
    np.random.seed(0)
    os.chdir('/home/ga48kuh/NASgroup/labmembers/filippokiessler/wahl-colab')
    scriptname = os.path.basename(__file__)[:-3]
    
    args = Arguments('sa', 'cosine') #instead of argparse
       
    savestr_base = f'code/paper-review/figure-4/outputs/{scriptname}'
    if not os.path.isdir(savestr_base):
        os.mkdir(savestr_base)
    
    savestr = f'{savestr_base}/{args.dataset}-{args.correlation}'
    if not os.path.isdir(savestr):
        os.mkdir(savestr)

    plt.style.use('code/paper-review/plotstyle.mplstyle')
    
    traces_path = 'data/neural-data/spatial_activity_maps_dff.pkl'
    pc_classes_binary_path = 'data/neural-data/is_pc.pkl'
    
    unwanted_mice = [121, 112]
    
    with open(traces_path, 'rb') as file:
        traces = pickle.load(file)
    with open(pc_classes_binary_path, 'rb') as file:
        pc_classes = pickle.load(file)
    traces = remove_unwanted_mice(traces, unwanted_mice)
    
    corrs_of_corrs = get_correlation_of_correlation_traces_maxneurons(traces, args.correlation)
    with open(f'{savestr}/corrs-of-corrs-similarity-matrices-{args.dataset}-{args.correlation}.pkl', 'wb') as f:
        pickle.dump(corrs_of_corrs, f, protocol = pickle.HIGHEST_PROTOCOL)
        
    # Example list of N correlation matrices of different sizes
    N = len(corrs_of_corrs)  # number of matrices

    # Determine the number of rows and columns for the subplots
    n_cols = int(np.ceil(np.sqrt(N)))
    n_rows = int(np.ceil(N / n_cols))

    ###########################################
    #Make same figure with equal colorbar for all mice

    # Create the figure and subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols + 0.5, 6 * n_rows))
    # Flatten the axes array for easy iterating
    minimas, maximas = [],[]
    for mouse, d in corrs_of_corrs.items():
        mask = np.identity(d['cmat'].shape[0])
        mask_mat = np.ma.masked_array(d['cmat'], mask = mask)
        minimas.append(np.nanmin(mask_mat))
        maximas.append(np.nanmax(mask_mat))

    # Set common colorbar limits for consistency
    vmin, vmax = np.min(minimas), np.max(maximas)

    # Plot each matrix in its subplot
    for (mouse,cmat_dict), ax in zip(corrs_of_corrs.items(), axes.flat):
        # Display the matrix
        #im = ax.imshow(matrix.transpose(), cmap='coolwarm', vmin=vmin, vmax=vmax)
        matrix = cmat_dict['cmat']
        mask = np.identity(matrix.shape[0])
        matrix = np.ma.masked_array(matrix, mask = mask)
        #vmax = np.max(matrix)
        index = cmat_dict['days']
        im = ax.imshow(matrix.transpose(), cmap='coolwarm', norm = TwoSlopeNorm(0., vmin = vmin, vmax= vmax))#symmetric colorbar
        zero_ind = np.argwhere(index > 0)[0]
        ax.vlines(zero_ind - 0.5, ymin = 0-0.5, ymax = len(index) -0.5, color = 'k')
        ax.vlines(zero_ind - 0.5 + 2, ymin = 0-0.5, ymax = len(index) -0.5, color = 'black', linestyle = '--')
        ax.hlines(zero_ind - 0.5, xmin = 0-0.5, xmax = len(index) -0.5, color = 'k')
        ax.hlines(zero_ind - 0.5 + 2, xmin = 0-0.5, xmax = len(index) -0.5, color = 'black', linestyle = '--')
        if mouse in recovery:
            mouse_group_fine = 'recovery'
        elif mouse in no_recovery:
            mouse_group_fine = 'no recovery'
        else:
            mouse_group_fine = 'sham'
        ax.set_title("M: {} ({})".format(mouse, mouse_group_fine))
        ax.set_xticks(range(len(matrix)), index)
        ax.set_yticks(range(len(matrix)), index)  # Turn off axis labels and ticks

    # Hide any unused axes if N is not a perfect square
    for i in range(len(corrs_of_corrs), len(axes)):
        axes[i].axis('off')

    # Create an axis for the colorbar
    fig.subplots_adjust(right = 0.97, top = 0.95)
    fig.suptitle(f'{args.correlation} correlation of correlations over measurement days ({args.dataset})')
    cbar_ax = fig.add_axes([0.98, 0.1, 0.02, 0.85])  # Adjust these values to fit the layout
    fig.colorbar(im, cax=cbar_ax, label = f'{args.correlation} r')

    fig.savefig(f'{savestr}/correlation-of-correlation-vectors-shared-cbar-more-data-{args.dataset}-{args.correlation}.png', dpi = 300)
    fig.savefig(f'{savestr}/correlation-of-correlation-vectors-shared-cbar-more-data-{args.dataset}-{args.correlation}.svg')

    ######## Export neural data to excel
    mouse_dfs_neuron_numbers = {}
    for (mouse,cmat_dict), ax in zip(corrs_of_corrs.items(), axes):
        matrix = cmat_dict['num-neurons']
        mask = np.identity(matrix.shape[0])
        matrix = np.ma.masked_array(matrix, mask = mask)
        index = cmat_dict['days']
    
        df = pd.DataFrame(data = matrix, index = index, columns = index)
        mouse_dfs_neuron_numbers[mouse] = df
    
    #write number of neurons into excel spreadsheet
    writer = pd.ExcelWriter(f'{savestr}/neuron-number.xlsx')
    for mouse, df in mouse_dfs_neuron_numbers.items():
        df.to_excel(writer, f'Mouse {mouse}')
    writer.save()

    #export data mouse by mouse for processing in matlab
    export_to_df(corrs_of_corrs) #this exports a series of dataframes that are then used by plot_similarity.m to plot the matrices seen in figure 4f

