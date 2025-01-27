#utilities
#Author: Filippo Kiessler, Technische Universität München, filippo.kiessler@tum.de
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy as sp
import os
import itertools

#correlation_matrices
def df_corr_result(res, key = 'corr'): #since the default key of df_corr_result is 'corr', what is written are the unclustered correlation matrices!

    to_df_dict = {k: pd.Series([res[k][sess][key] for sess in res[k].keys()], index = res[k].keys()) for k in res.keys()}
    df = pd.DataFrame(data = to_df_dict).T
    return df


def mice_category_safe(df_ind, group):
    return list(set(df_ind).intersection(group))

def find_closest_columns_to_zero(row, n=3, dropna = False):
    # Find the 3 columns closest to 0
    if not dropna:
        return sorted(row[row.index <= 0].index)[-n:]
    else:
        return sorted(row[row.index <= 0].dropna().index)[-n:]
    

def get_poststroke_mapping():
    
    aligned_perfect = np.arange(0,30,3)
    
    remapping_df = {}
    for i in range(30):
        for j in aligned_perfect:
            if (np.abs(i-j) <= 1) and i>0:
                remapping_df[i] = j
    
    remapping_df[1] = 3
    return remapping_df


def align_days_poststroke(indices):
    remapping_df = get_poststroke_mapping()
    new_idx = []
    for i in indices:
        if i in list(remapping_df.keys()):
                new_idx.append(remapping_df[i]) 
        else:
            new_idx.append(i)
    return new_idx

def align_row(row):
    idx = align_days_poststroke(row.index)
    idx, clean_row = clean_arrays(idx, row.values)
    return idx, clean_row


def align_df_poststroke(df):
    #expect row-major df, i.e. where there is a column with relative day values to be aligned
    melted = df.melt(ignore_index = False, var_name = 'rel_days')
    melted['rel_days'] = align_days_poststroke(melted['rel_days'])
    melted.dropna(inplace = True)
    melted.index.rename('mouse', inplace = True)
    melted_series = melted.groupby('mouse').apply(lambda group: group.groupby('rel_days')['value'].mean())
    melted_df = pd.DataFrame(melted_series).reset_index('rel_days')
    
    return melted_df.pivot(columns = 'rel_days').droplevel(0, axis = 1)
    
def clean_arrays(a, b, callable = np.mean):
    #by chat gpt. if a contains duplicates, perform 'callable' on the corresponding
    #values of b. returns two arrays, i.e. unique values of a and corresponding
    #unique values of b, where when necessary the 'callable' was performed to obtain a unique corresponding value for b 
    cleaned_a, cleaned_b = [], []
    unique_a_values = np.unique(a)

    for val in unique_a_values:
        mask = (a == val)
        averaged_b = callable(b[mask])
        cleaned_a.append(val)
        cleaned_b.append(averaged_b)

    return np.array(cleaned_a), np.array(cleaned_b)


#plot median and mean correlation for every mouse on every day.
#drop mouse 63 because it has very few simultaneously tracaked neurons
def plot_qty_correlation_vectors(df, savename, savestr = None, callable = np.mean, labelstr = '{}', xlabel = 'Relative days', ylabel = 'Correlation',
                                 titlestr = None, normalized = False,
                                 colors = None, legend = False,
                                 align = False):
    toplot_qtys = df.applymap(callable, na_action = 'ignore')
    
    if savestr is None:
        savestr = os.getcwd()
    
    if colors is None:
        cm = plt.cm.get_cmap('hsv', len(toplot_qtys.index))
        colors = {mouse: cm(i) for i, mouse in enumerate(toplot_qtys.index)}
    
    if normalized:
        for i in toplot_qtys.index:
            l3_poststroke_sessions = find_closest_columns_to_zero(toplot_qtys.loc[i])
            toplot_qtys.loc[i] *= 1/toplot_qtys.loc[i, l3_poststroke_sessions].mean() #normalize to last 3/4 prestroke sessions
    
    f = plt.figure(figsize=(10,8), dpi = 300)
    for i in toplot_qtys.index:
        row = toplot_qtys.loc[i].dropna()
        idx, clean_row = align_row(row) if align else (row.index, row.values)
        plt.plot(idx, clean_row, label = labelstr.format(i),
                 color = colors[i])
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend()
    if titlestr is not None:
        plt.title(titlestr)
    if savename is not None:
        plt.savefig(f'{savestr}/{savename}')
    
    return f

def plot_qty_correlation_mean_std(df, savestr = None, savename = None, groups = None, group_colors = 'k', callable = np.mean, labelstr = '{}', xlabel = 'Relative days', ylabel = 'Correlation',
                                titlestr = None, normalized = False,
                                legend = False,
                                align = False):
    
    if savestr is None:
        savestr = os.getcwd()
    
    toplot_qtys = df.applymap(callable, na_action = 'ignore')
    
    #filter out mice that don't match
    
    if groups is not None:
        safe_groups = [mice_category_safe(toplot_qtys.index, group) for group in groups]
        division = [toplot_qtys.loc[group] for group in safe_groups]
    else:
        division = [toplot_qtys]
        safe_groups = [[toplot_qtys.index]]
    
    
    if type(group_colors) is str:
        colors_mice = [[group_colors for _ in group.index] for group in division]
    elif ((type(group_colors) == list) and (len(group_colors) == len(groups))) and len(group_colors) > 1:
        colors_mice = [[c for _ in group.index] for c, group in zip(group_colors, division)]
    else:
        raise ValueError('Number of colors does not match number of groups')
    
    means, stds = [g.mean(axis = 0) for g in division], [g.std(axis = 0) for g in division]

    if normalized:
        for raw_group in division:
            for i in raw_group.index:
                l3_poststroke_sessions = find_closest_columns_to_zero(raw_group.loc[i])
                raw_group.loc[i] *= 1/raw_group.loc[i, l3_poststroke_sessions].mean() 
            
        for gm, gstd in zip(means, stds): 
            for im, istd in zip(gm.index, gstd.index):

                l3_poststroke_sessions = find_closest_columns_to_zero(gm)
                gm.loc[im] *= 1/gm.loc[l3_poststroke_sessions].mean()
                gstd.loc[istd] *= 1/gstd.loc[l3_poststroke_sessions].mean()
    
    
    
    f = plt.figure(figsize=(10,8), dpi = 300)
    for groupcount, raw_group in enumerate(division):    
        for mousecount, mouse in enumerate(raw_group.index):
            row = raw_group.loc[mouse].dropna()
            idx, clean_row = align_row(row) if align else (row.index, row.values)
            plt.plot(idx, clean_row, label = labelstr.format(mouse),
                    color = colors_mice[groupcount][mousecount], alpha = 0.3)
        
        clean_mean_idx, clean_mean = align_row(means[groupcount]) if align else (means[groupcount].index, means[groupcount].values)
        clean_std_idx, clean_std = align_row(stds[groupcount]) if align else (stds[groupcount].index, stds[groupcount].values)
        plt.plot(clean_mean_idx, clean_mean, color = colors_mice[groupcount][0])
        plt.fill_between(clean_mean_idx, clean_mean - clean_std,
                        clean_mean + clean_std, color = colors_mice[groupcount][0], alpha = 0.2)
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend()
    if titlestr is not None:
        plt.title(titlestr)
    if savename is not None:
        plt.savefig(f'{savestr}/{savename}')
    
    return f

def ax_plot_coor_fit(axes, mousecount, mouse, row, xframe, yframe, xrange = np.array([-0.2, 0.8]), significance_level = 0.05):
    #function to plot scatters and make both linear fit and correlation value of the point cloud
    x = xrange 
    axes[mousecount][row].scatter(xframe.loc[mouse], yframe.loc[mouse], s = 1, marker = 'x', color = 'r')
    axes[mousecount][row].set_title(f'Mouse {mouse}')
    axes[mousecount][row].plot(x, x, color = 'k')
    axes[mousecount][row].plot(x, x * coef.slope + coef.intercept , color = 'green')
    try:
        coef = sp.stats.linregress(xframe.loc[mouse], yframe.loc[mouse])
        axes[mousecount][row].annotate(f'slope = {coef.slope:.2f}\nr = {coef.rvalue:.2f}\np = {coef.pvalue:.4f}', xy = (-0.1, 0.6), fontsize = 10)
        if coef.pvalue > significance_level: #significance level of the pvalue
            print(f"mouse {mouse} gave too high p-value {coef.pvalue}")
            return np.nan, np.nan, coef.pvalue #technically I should put these to 0, but the only reason why p-values are high in this case is that there are too few datapoints
        else:
            return coef.slope, coef.rvalue, coef.pvalue 
    except:
        axes[mousecount][row].annotate(f'slope = {coef.slope:.2f}\ntoo few data points', xy = (-0.1, 0.6), fontsize = 10)
        return np.nan, np.nan, np.nan
    
    
def get_group_results_slope_corr(group, pre_early_late_fit_corrcoef):
    #outputs 3 numpy arrays, where the row corresponds to the mouse id as set in 'group' and the columns correspond to different pairings,
    #i.e. pre-early, pre-late and early-late
    
    res_slope = np.array([(pre_early_late_fit_corrcoef[m]['fit_pre_early'], pre_early_late_fit_corrcoef[m]['fit_pre_late'],
                           pre_early_late_fit_corrcoef[m]['fit_early_late']) for m in group])
    
    res_corr = []
    for m in group:
        try:
            res_corr.append((pre_early_late_fit_corrcoef[m]['corr_pre_early'], pre_early_late_fit_corrcoef[m]['corr_pre_late'],
                           pre_early_late_fit_corrcoef[m]['corr_early_late']))
        except:
            res_corr.append((np.nan, np.nan, np.nan))
    res_corr = np.array(res_corr)
    
    res_pval = []
    for m in group:
        try:
            res_pval.append((pre_early_late_fit_corrcoef[m]['pval_pre_early'], pre_early_late_fit_corrcoef[m]['pval_pre_late'],
                           pre_early_late_fit_corrcoef[m]['pval_early_late']))
        except:
            res_pval.append((np.nan, np.nan, np.nan))
    res_pval = np.array(res_pval)
    
    return res_slope, res_corr, res_pval



def plot_quantiles_with_data(ax, categories, data_arrays, titlestr='', ylabel='', xlabel=''):
    #make boxplots of different data in 'data arrays' sorted by 'categories', and make t-test for significance

    # Define the quantiles you want to plot
    quantiles = [0.25, 0.5, 0.75]

    # Create numerical values for categories
    category_indices = np.arange(len(categories))

    # Store combinations of categories for t-tests
    combinations = [(i, j) for i in category_indices for j in category_indices if i < j]

    # Set box width
    box_width = 0.2

    for category, data in zip(category_indices, data_arrays):
        # Calculate quantiles for the data
        data = np.array(data)
        data_clean = data[np.isfinite(data)]
        quantile_values = np.nanquantile(data_clean, quantiles)
        # Plot the quantiles as box plots
        ax.boxplot(data_clean, positions=[category], widths=box_width, labels=[categories[category]],
                   patch_artist=True, boxprops=dict(facecolor='lightgray'),
                   medianprops={'color': 'black'})

        # Plot individual data points as scattered dots in the background
        jittered_x = np.random.normal(category, 0.04, len(data_clean))
        ax.plot(jittered_x, data_clean, 'ro', alpha=0.3)

    # Get the maximum range among all data arrays
    max_range = max([np.max(np.array(data)[~np.isnan(data)]) - np.min(np.array(data)[~np.isnan(data)]) for data in data_arrays])
    max_value = max([np.max(np.array(data)[~np.isnan(data)]) for data in data_arrays])
    
    # Calculate dynamic line_offset based on the data range
    line_offset = max_range * 0.25  # You can adjust the factor as needed
    # Perform t-tests for all unique combinations of categories
    significance_count = 0
    for i, comb in enumerate(combinations):
        category1, category2 = comb
        data1 = data_arrays[category1]
        data2 = data_arrays[category2]
        _, p_value = sp.stats.ttest_ind(np.array(data1)[~np.isnan(data1)], np.array(data2)[~np.isnan(data2)])

        # Add significance stars for different combinations
        if p_value < 0.001:
            star = '***'
        elif p_value < 0.01:
            star = '**'
        elif p_value < 0.05:
            star = '*'
        else:
            star = ''
            continue

        # Determine the x-coordinates for the lines
        x1, x2 = category1, category2

        # Calculate the vertical position for the line (above the highest datapoint)
        y = max_value + line_offset * significance_count + 0.1 * line_offset

        # Add lines connecting significantly different combinations
        ax.plot([x1, x2], [y, y], 'k-', lw=0.5)

        # Add significance stars for each combination
        ax.text((x1 + x2) / 2, y * 1.02 , star, fontsize=12, horizontalalignment='center')
        significance_count += 1

    # Set x-axis labels to category names
    ax.set_xticks(category_indices)
    ax.set_xticklabels(categories)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titlestr)

    return ax

#average over the pre-stroke, early and late post-stroke mice
def avg_over_columns(df, last3 = False, skipna = True):
    meanlist = []
    for mouse in df.index:
        if not last3:
            m = df.loc[mouse].dropna().mean(skipna = skipna)
            meanlist.append(m)
        else:
            clean_df = df.loc[mouse].dropna()
            last3 = find_closest_columns_to_zero(clean_df)
            meanlist.append(clean_df[last3].mean(skipna = skipna))
    return pd.Series(meanlist, index = df.index)

def avg_over_columns_series(series, last3 = False, skipna = False):
    if not last3:
        m = series.dropna().mean(skipna = skipna)
        return m
    else:
        clean_df = series.dropna()
        last3 = find_closest_columns_to_zero(clean_df)
        return clean_df[last3].mean(skipna = skipna)
    
def avg_over_columns_nanmean(df, last3 = False, mean_axis = 0, skipna = None):
        meanlist = []
        for mouse in df.index:
            if not last3:
                m = np.nanmean(np.vstack(df.loc[mouse].dropna().values), axis = mean_axis)
                meanlist.append(m)
            else:
                clean_df = df.loc[mouse].dropna()
                last3 = find_closest_columns_to_zero(clean_df)
                meanlist.append(np.nanmean(np.vstack(clean_df[last3].values), axis = mean_axis))
        return pd.Series(meanlist, index = df.index)

def remove_unwanted_mice(dict, mice):
    #dict: dictionary with mouse keys
    dict_copy = dict
    for mouse in mice:
        if mouse in list(dict_copy.keys()):
            del dict_copy[mouse]
    return dict_copy

def remove_unwanted_mice_df(df, mice):
    #dict: dictionary with mouse keys
    df_copy = df
    for mouse in mice:
        if mouse in df_copy.index:
            df_copy = df_copy.drop(mouse)
    return df_copy

#take the lower (or upper, does not matter, just be consistent!) of the input matrix (cell), excluding the diagonal elements, and flatten them
def get_correlation_vector(cell):
    return cell[np.tril_indices_from(cell, k = -1)] #distance vector, i.e. flattened lower triangle of symmetric correlation matrix, excluding diagonal elements

def divide_pre_early_late(df):
    #separate the dataframe (according to columns) into pre-stroke, early and late poststroke
    pre = df.loc[:,df.columns <= 0]
    early = df.loc[:,np.logical_and(df.columns > 0, df.columns < 7)]
    late = df.loc[:,df.columns >= 7]
    return (pre, early, late)

def correlation_cluster(session, metric = 'euclidean', method = 'average'):
    #calculate correlation of neural activity for one day (session). return the sorted labels and distance matrix
    correlation_matrix = np.corrcoef(np.vstack(session))
    distances = sp.spatial.distance.pdist(correlation_matrix, metric = metric) #compute the distance matrix of correlation vectors
    linkage_matrix = sp.cluster.hierarchy.linkage(distances, method=method)
    dendrogram = sp.cluster.hierarchy.dendrogram(linkage_matrix, no_plot = True)
    leaves = dendrogram['leaves'] #clustered indices of mouse id
    
    distance_matrix = sp.spatial.distance.squareform(distances)
    clustered_correlation_matrix =  correlation_matrix[leaves, :][:, leaves]
    
    return correlation_matrix, clustered_correlation_matrix, distance_matrix, leaves

def get_mapping_dict(unique_categories):
    return {pair: count for count, pair in enumerate(itertools.combinations_with_replacement(unique_categories, 2))}

def map_cat_pairlist_make_matrix(cat_pairvec, N, unique_categories):
    
    '''
    the mapping is calculated in the get_mapping_dict() function
    cat_pairvec: number of cell pairs 
    N: number of tracked cells. should be len(cat_pairvec) ** 2

    The mapping is:
    (1,1) -> 0
    (1,2) = (2,1) -> 1
    (1,3) = (3,1) -> 2
    (2,2) -> 3
    (2,3) = (3,2) -> 4
    (3,3) -> 5
    '''
    mapping_dict = get_mapping_dict(unique_categories)
    mapped_cellcat_pairs = np.array(list(map(lambda x : mapping_dict[tuple(sorted(x))], cat_pairvec)))
    return mapped_cellcat_pairs.reshape(N,N)

def create_pair_vector(class_vector):
    pair_vector = []
    for i in class_vector:
        for j in class_vector:
            pair_vector.append((i,j))
    return pair_vector

def mouse_cell_pair_matrices_series(mouse, matched_place_cells, unique_cellcats = np.array([1,2,3])):
    #create a pd.Series of cell pair matrices for every session in a mouse 

    mouse_pair_matrices = {}

    for session, mouse_place_cells in matched_place_cells[mouse].items():
        print(f"mouse: {mouse}, session: {session}")
        #should implement this with series.apply!
        session_cellcats_nonzero = mouse_place_cells.dropna().astype(int)

        if not np.all(np.isin(session_cellcats_nonzero.unique(), unique_cellcats)):
            raise ValueError(f"mouse {mouse}, session {session} has a 0 as cell category\nunique categories: {session_cellcats_nonzero.unique()},\nonly {unique_cellcats} are allowed")
        else:
            session_pair_vector = create_pair_vector(session_cellcats_nonzero.values)
        
        mapped_pair_matrix = map_cat_pairlist_make_matrix(session_pair_vector, len(session_cellcats_nonzero), unique_cellcats)
        mouse_pair_matrices[session] = mapped_pair_matrix

    return pd.Series(mouse_pair_matrices)

def plot_df_columns_by_groups(ax, df, column_name, index_groups, index_group_names, ylabel = None, titlestr = None):
        data_arrays = [df.loc[group, column_name].values.astype(float).flatten()  for group in index_groups]
        ax = plot_quantiles_with_data(ax, index_group_names, data_arrays, ylabel = ylabel, titlestr=titlestr)
        return ax

def plot_df_allcolumns(ax, df, index_groups, index_group_names, ylabel = None, titlestr = None):
        data_arrays = [df.loc[group].values.astype(float).flatten()  for group in index_groups]
        ax = plot_quantiles_with_data(ax, index_group_names, data_arrays, ylabel = ylabel, titlestr=titlestr)
        return ax

def subplot_layout(n):
    """
    Generate a subplot layout based on the number of columns in a DataFrame.

    Parameters:
    - n: Number of columns in the DataFrame.

    Returns:
    - Tuple (rows, cols) representing the subplot layout.
    """
    # Determine the closest square layout for the given number of columns
    side_length = int(np.ceil(np.sqrt(n)))
    rows = np.ceil(n / side_length)
    cols = side_length
    return (int(rows), int(cols))

control =  [33, 83, 91, 93, 95, 108, 111, 112, 114, 115, 116, 122]

stroke = [41, 63, 69, 85, 86, 89, 90, 110, 113]

no_recovery = [41, 63, 69, 110]
recovery = [85, 86, 89, 90, 113]
sham = [91, 111, 115, 122, 33, 83, 93, 95, 108, 112, 114, 116]

#remove unwanted mice
control.remove(112)
sham.remove(112)

unwanted_mice = [121, 112]


def melt_join_dataframes(df_list, names_list, suffix = ''):
    dfs_melted = [df.reset_index().rename(columns = {'index':'mouse'}).melt(id_vars = ['mouse'], var_name = 'day', value_name = f'{name}_{suffix}') for df, name in zip(df_list, names_list)]
    big_df = pd.merge(dfs_melted[0],dfs_melted[1],on=['mouse', 'day'])
    for df in dfs_melted[2:]:
        big_df = pd.merge(big_df, df, on = ['mouse', 'day'])
    return big_df

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
