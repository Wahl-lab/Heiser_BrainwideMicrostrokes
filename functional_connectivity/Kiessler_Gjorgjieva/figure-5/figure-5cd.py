#figure 5D,C
#Author: Filippo Kiessler, Technische Universität München, filippo.kiessler@tum.de
import pickle
import numpy as np

import os
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import sys
sys.path.append('../')#to include 'utilities.txt'
from utilities import plot_quantiles_with_data, remove_unwanted_mice, df_corr_result, get_correlation_vector, avg_over_columns, divide_pre_early_late, avg_over_columns_nanmean

import argparse


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
    
def apply_function_to_cells(df1, df2, func, ignore_nan=False):
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
            nan_condition = ignore_nan and (isinstance(val1, float) and np.isnan(val1) or
                                            isinstance(val2, float) and np.isnan(val2))

            if nan_condition:
                cell_result = np.NaN
            else:
                cell_result = func(val1, val2)

            row_data.append(cell_result)

        result_data.append(row_data)

    result_df = pd.DataFrame(result_data, index=df1.index, columns=df1.columns)
    return result_df


def make_boxplots_pre_early_late(fig, axs, statistic_df_list, mouse_groups_coarse,
                                 categories_str_list = ['control', 'stroke'],
                                 titlestr_list = [None, None, None]):
    #cb: callable
    #cstring: string of the callable
    
    
    for i, titlestr, statistic in zip(range(len(axs)), titlestr_list, statistic_df_list):
        data_arrays = [statistic.loc[group].values.flatten() for group in mouse_groups_coarse]
    
        axs[i] = plot_quantiles_with_data(axs[i], categories_str_list, 
                                data_arrays, titlestr = titlestr)
    
    return fig

def counts_from_uniques(cell, values):
    return np.array([np.sum(cell == v) for v in values])

def map_days_pre_early_late(day):
    if day <= 0:
        return 'pre'
    elif 0 < day < 7:
        return 'early'
    else:
        return 'late'

def map_mouse_behaviour_group_coarse(mouse):
    if mouse in control:
        return 'sham'
    elif mouse in stroke:
        return 'stroke'
    
def map_mouse_behaviour_group_fine(mouse):
    if mouse in sham:
        return 'sham'
    elif mouse in recovery:
        return 'recovery'
    elif mouse in no_recovery:
        return 'no recovery'

def plot_grouped_boxplots(df, suptitle, baseline):
    # Set the figure and axes
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey='row')
    fig.suptitle(suptitle)

    # Columns to plot
    data_columns = [
        'fraction_nc_nc_above_div_totalcount',
        'fraction_nc_pc_above_div_totalcount',
        'fraction_pc_pc_above_div_totalcount'
    ]
    y_labels = ['nc-nc', 'nc-pc', 'pc-pc']  # Short labels for the y-axis

    # Periods for the columns in the grid
    periods = ['pre', 'early', 'late']

    # Iterate over the DataFrame rows (data_columns) and columns (periods) in the plot grid
    for i, column in enumerate(data_columns):
        for j, period in enumerate(periods):
            ax = axes[i, j]
            # Filter data for the current period
            period_data = df[df['period'] == period]
            ax.hlines([baseline], 0, 3, linestyle = 'dashed', lw = 1, color = 'gray')
            # Plot the boxplot with seaborn
            sns.boxplot(x='mouse_group_fine', y=column, data=period_data, ax=ax, color='#D3D3D3', width=0.5)  # Lighter grey and narrower boxes
            sns.stripplot(x='mouse_group_fine', y=column, data=period_data, ax=ax, color='red', jitter=True, size=4, alpha=0.7)

            # Set titles and labels
            if i == 0:
                ax.set_title(period)
            if j == 0:
                ax.set_ylabel(y_labels[i])
            else:
                ax.set_ylabel('')

            # Remove x-axis labels
            ax.set_xlabel('')

    plt.tight_layout()
    return fig, axes

class Arguments:
    #for debugging
    def __init__(self, dset, quantile):
        self.dataset = dset
        self.quantile = quantile

if __name__ == '__main__':
    
    os.chdir('/home/ga48kuh/NASgroup/labmembers/filippokiessler/wahl-colab')
    scriptname = os.path.basename(__file__)[:-3]

    args = Arguments('dff', 0.95) #for debugging
       
    savestr_base = f'code/paper-review/figure-5/outputs/{scriptname}'
    if not os.path.isdir(savestr_base):
        os.mkdir(savestr_base)
    
    savestr = f'{savestr_base}/{args.dataset}'
    if not os.path.isdir(savestr):
        os.mkdir(savestr)
        
    plt.style.use('code/paper-review/plotstyle.mplstyle')
    
    
    #dataset selection:
    if args.dataset == 'dff':
        traces_path = 'code/paper-review/figure-5/outputs/precompute-decon-dff-calc-corrmat-distancemat-all-cells/correlation-mat-unsorted-dff.pkl'
    elif args.dataset == 'decon':
        traces_path = 'code/paper-review/figure-5/outputs/precompute-decon-dff-calc-corrmat-distancemat-all-cells/correlation-mat-unsorted-decon.pkl'
    elif args.dataset == 'sa':
        traces_path = 'code/paper-review/figure-5/outputs/precompute-decon-dff-calc-corrmat-distancemat-all-cells/correlation-mat-unsorted-sa.pkl'
    else:
        raise ValueError('Dataset for correlation matrices of neural traces does not exist!')
    
    with open(traces_path, 'rb') as file:
        traces_corrmat_dict = pickle.load(file)
    
    pc_division_path = 'code/paper-review/figure-5/outputs/precompute-decon-dff-calc-corrmat-distancemat-all-cells/mouse-cell-pair-identifiers.pkl'
    with open(pc_division_path, 'rb') as file:
        pc_classes_matrix =  pickle.load(file)
    
    pc_classes_matrix = remove_mice_from_df(pc_classes_matrix, unwanted_mice)

    traces_corrmat_dict_filtered = remove_unwanted_mice(traces_corrmat_dict, unwanted_mice)#remove mouse 121 and 63. 63 is removed here because it has too few cells that are tracked on all days
    filtered_corrmat_traces_df = df_corr_result(traces_corrmat_dict_filtered)
    traces_correlation_vectors = filtered_corrmat_traces_df.applymap(get_correlation_vector, na_action = 'ignore')
    
    #remap the cell pair categories to place cell and non-place cell
    remapped_final_pc_vec = pc_classes_matrix.applymap(get_correlation_vector, na_action = 'ignore')
    #compute dataframe of vectors for correlation statistic for every pair category:
    unique_categories = get_unique_cell_pair_categories(remapped_final_pc_vec)#the ordering of these unique values applies to all contents of the
    cellpair_type_counts = remapped_final_pc_vec.applymap(lambda cell: np.array([sum(cell==i) for i in unique_categories]), na_action = 'ignore')
    
    
    string_pair_mapping = {0: 'non-coding-non-coding', 1:'non-coding-place-cell', 2:'place-cell-place-cell'}
    
    #derivative of the apply_function_to_cells function
    
    #calculate quantiles of correlation vectors (equivalently of correlation matrices)
    def get_quantile_means_of_distribution_pc_counts(quantile, traces_correlation_vectors, remapped_final_pc_vec, qfunction = lambda x,y: x>y):
        correlation_vec_quantiles = traces_correlation_vectors.applymap(lambda cell: np.quantile(cell, quantile), na_action = 'ignore')
        corr_greater_than_quantile = apply_function_to_cells(traces_correlation_vectors, correlation_vec_quantiles, qfunction, ignore_nan = True)
        
        #calculate fraction of cells that place cells and greater than the 0.8 quantile
        cellpairs_greater_than_quantile = apply_function_to_cells(corr_greater_than_quantile, remapped_final_pc_vec, lambda x,y: y[x], ignore_nan=True)
        
        cellcats_greater_than_quantile_counts = cellpairs_greater_than_quantile.applymap(lambda x: counts_from_uniques(x, unique_categories), na_action = 'ignore')
        cellcats_greater_than_quantile_fractions = cellcats_greater_than_quantile_counts.applymap(lambda x: x/x.sum(), na_action = 'ignore')
        
        #split fractions according to pre, early late
        cellcats_quantile_frac_pre_early_late_division = divide_pre_early_late(cellcats_greater_than_quantile_fractions)
        
        #calculate mean fractions by pair category over pre, early and late poststroke
        cellcats_quantile_frac_pre_early_late_meanlist =  [avg_over_columns(cellcats_quantile_frac_pre_early_late_division[0], last3=True),
                                                          avg_over_columns(cellcats_quantile_frac_pre_early_late_division[1]),
                                                           avg_over_columns(cellcats_quantile_frac_pre_early_late_division[2])] #results should sum to 1
        
        return cellcats_quantile_frac_pre_early_late_meanlist, cellcats_greater_than_quantile_fractions, cellcats_greater_than_quantile_counts
    
    quantile = args.quantile
    cellcats_quantile_frac_pre_early_late_meanlist_greater, cellcats_greater_than_quantile_fractions, cellcats_greater_than_quantile_counts = get_quantile_means_of_distribution_pc_counts(quantile, traces_correlation_vectors, remapped_final_pc_vec)
    
    
    #######################################################################################################################################################
    #make plots for coarse and for fine division
    length = len(cellcats_quantile_frac_pre_early_late_meanlist_greater[0].iloc[0])

    
    cellcats_quantile_frac_pre_early_late_meanlist_smaller, cellcats_smaller_than_quantile_fractions,  cellcats_smaller_than_quantile_counts = get_quantile_means_of_distribution_pc_counts(quantile, traces_correlation_vectors,
                                                                                                          remapped_final_pc_vec, lambda x,y: x<y)

    #compute fraction of means above and below the 0.8th quantile
    frac = apply_function_to_cells(cellcats_greater_than_quantile_fractions, cellcats_smaller_than_quantile_fractions, lambda x,y: x/y)
    div_frac = divide_pre_early_late(frac)
    pre_early_late_frac_means = [avg_over_columns_nanmean(div_frac[0], last3=True), avg_over_columns_nanmean(div_frac[1]), avg_over_columns_nanmean(div_frac[2])]


    #fractions of counts above quantile divided by total count by cell pair type        
    above_fractions_div_totalcount = apply_function_to_cells(cellcats_greater_than_quantile_counts, cellpair_type_counts, lambda x,y: x/y, ignore_nan = True)
    div_above_fractions_div_totalcount = divide_pre_early_late(above_fractions_div_totalcount)
    #old, wrong cell affected by nans of arrays in cells: pre_early_late_div_above_fractions_div_totalcount = [avg_over_columns(div_above_fractions_div_totalcount[0], last3=True), avg_over_columns(div_above_fractions_div_totalcount[1]), avg_over_columns(div_above_fractions_div_totalcount[2])]
    pre_early_late_div_above_fractions_div_totalcount = [avg_over_columns_nanmean(div_above_fractions_div_totalcount[0], last3=True), avg_over_columns_nanmean(div_above_fractions_div_totalcount[1]), avg_over_columns_nanmean(div_above_fractions_div_totalcount[2])]

    #export data to csv
    datalist = [cellcats_quantile_frac_pre_early_late_meanlist_greater,cellcats_quantile_frac_pre_early_late_meanlist_smaller, pre_early_late_frac_means, pre_early_late_div_above_fractions_div_totalcount]
    datalist_renamed = []
    for l, name in zip(datalist, ['greater', 'smaller', 'greater divided by smaller', 'above count divided by total count by cell type']):
        for stat, period in zip(l, ['pre', 'early', 'late']):
            pairtype_stats = [stat.apply(lambda cell: cell[i]).rename((period, f'{name} quantile {quantile}',pairtype)) for i, pairtype in string_pair_mapping.items()]
            datalist_renamed.append(pairtype_stats)
            
    datalist_flat = [x for quantile_stat in datalist_renamed for x in quantile_stat]
    data_df = pd.concat(datalist_flat, axis = 1)
    
    data_df.to_csv(f'{savestr}/cell-pair-fractions-in-correlation-distribution-quantile-{quantile}-{args.dataset}.csv')
