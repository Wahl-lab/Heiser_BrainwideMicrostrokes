#precompute correlation matrices, place cell pairs and distance matrices for Figures 5A, C, D 
#Author: Filippo Kiessler, Technische Universität München, filippo.kiessler@tum.de
import pickle
import numpy as np
import os
#import matplotlib.pyplot as plt

import scipy as sp
import itertools
import pandas as pd

import sys
sys.path.append('../') #to include 'utilities.txt'
from utilities import remove_unwanted_mice

def filter_always_present(dict):
    #return a dict, such as decon or dff, that contains for every mouse
    #only neurons that were imaged on all sessions across time.
    
    filtered = {k: v.dropna() for k,v in dict.items()}
    return filtered
    
def correlation_cluster(session, metric = 'euclidean', method = 'average'):
    #calculate correlation of neural activity for one day (session). return the sorted labels and distance matrix
    correlation_matrix = np.corrcoef(np.vstack(session))
    try:
        distances = sp.spatial.distance.pdist(correlation_matrix, metric = metric) #compute the distance matrix of correlation vectors
        linkage_matrix = sp.cluster.hierarchy.linkage(distances, method=method)
        dendrogram = sp.cluster.hierarchy.dendrogram(linkage_matrix, no_plot = True)
        leaves = dendrogram['leaves'] #clustered indices of mouse id
    
        distance_matrix = sp.spatial.distance.squareform(distances)
        clustered_correlation_matrix =  correlation_matrix[leaves, :][:, leaves]
    
        return correlation_matrix, clustered_correlation_matrix, distance_matrix, leaves
    except:#if distance computation fails
        return correlation_matrix, np.NaN, np.NaN, np.NaN
    
def neuraldata_cluster_corr(dset):
    #function that iterates over all mice in a dataset (either deconvolved or dff or other dicts with int:dataframe)
    #for every dataframe, compute the correlation of all arrays stored in every row of a single column. this returns
    #a correlation matrix for a single column, a clustered version of the correlation matrix, a distance matrix, as
    #well as the ordering
    
    #the output is a dict, where every key is a mouse, and every value is another dict, with the session as key.
    #this inner dict has as values another dict, with following keys:    
    #corr: correlation matrix
    #clustcorr: clustered correlation matrix
    #dist: distance matrix, where column ordering corresponds to ordering of corr
    #order: ordering according to which corr is reordered into clustcorr
        
    mice = {}

    for k in dset.keys():
        print('mouse', k)
        sessions = {}
        for sess in dset[k].columns:
            results_dict = {}
            res = correlation_cluster(dset[k].loc[:, sess].dropna())
            
            results_dict['sesssion'] = sess
            results_dict['corr'] = res[0]
            results_dict['clustcorr'] = res[1]
            results_dict['dist'] = res[2]
            results_dict['order'] = res[3]
        
            sessions[sess] = results_dict
            
        mice[k] = sessions
    
    return mice

#want a function that takes results from neuraldata_cluster_corr and includes a 

#correlation_matrices
def df_corr_result(res, key = 'corr'): #since the default key of df_corr_result is 'corr', what is written are the unclustered correlation matrices!

    to_df_dict = {k: pd.Series([res[k][sess][key] for sess in res[k].keys()], index = res[k].keys()) for k in res.keys()}
    df = pd.DataFrame(data = to_df_dict).T
    return df

#create vector of cell category pairs
def create_pair_vector(class_vector):
    pair_vector = []
    for i in class_vector:
        for j in class_vector:
            pair_vector.append((i,j))
    return pair_vector

#take a list of cell category pairs and map it to a series of integers.
#then reshape the integers into a matrix

def get_mapping_dict():
    return {pair: count for count, pair in enumerate(itertools.combinations_with_replacement([0,1], 2))}

def map_cat_pairlist_make_matrix(cat_pairvec, N):
    
    '''
    the mapping is calculated in the get_mapping_dict() function
    cat_pairvec: number of cell pairs 
    N: number of tracked cells. should be len(cat_pairvec) ** 2

    The mapping is:
    (0,0)-> 0
    (0,1) -> 1
    (1,1) -> 2
    '''
    mapping_dict = get_mapping_dict()
    mapped_cellcat_pairs = np.array(list(map(lambda x : mapping_dict[tuple(sorted(x))], cat_pairvec)))
    return mapped_cellcat_pairs.reshape(N,N)

def mouse_cell_pair_matrices_series(mouse, matched_place_cells):
    #create a pd.Series of cell pair matrices for every session in a mouse 

    mouse_pair_matrices = {}

    for session, mouse_place_cells in matched_place_cells[mouse].items():
        print(f"mouse: {mouse}, session: {session}")
        #should implement this with series.apply!
        session_cellcats_nonzero = mouse_place_cells.dropna().astype(int)

        if not np.all(np.isin(session_cellcats_nonzero.unique(), np.array([0,1]))):
            raise ValueError(f"mouse {mouse}, session {session} has a 0 as cell category\nunique categories: {session_cellcats_nonzero.unique()}")
        else:
            session_pair_vector = create_pair_vector(session_cellcats_nonzero.values)
        
        mapped_pair_matrix = map_cat_pairlist_make_matrix(session_pair_vector, len(session_cellcats_nonzero))
        mouse_pair_matrices[session] = mapped_pair_matrix
    
    return pd.Series(mouse_pair_matrices)

def pair_matrix_allmice_df(matched_place_cells):
    allmice_cellpairmats = {}

    for mouse, cell_pairs in matched_place_cells.items():
        print(mouse)
        allmice_cellpairmats[mouse] = mouse_cell_pair_matrices_series(mouse, matched_place_cells)
    return pd.DataFrame(allmice_cellpairmats).transpose()

def filter_coords_by_dff(coords_dict, dff_dict):
    filtered_coords = {}
    
    if list(coords_dict.keys()) != list(dff_dict.keys()):
        raise ValueError(f"Mice of two dataframes do not match!")
    for (mouse_dff, dff_df), (mouse_c, coords_series) in zip(dff_dict.items(), coords_dict.items()):
        filtered_coords[mouse_c] = coords_series.loc[dff_df.index]
    return filtered_coords

def get_distance_matrix_from_coord_series(coord_series):
    return sp.spatial.distance_matrix(np.vstack(coord_series.values), np.vstack(coord_series.values))

def get_distance_matrix_series(coord_dict_allmice):
    distance_matrix_dict = {}
    for mouse, coord_series in coord_dict_allmice.items():
        distance_matrix_dict[mouse] = get_distance_matrix_from_coord_series(coord_series)
    filtered_dm_series = pd.Series(distance_matrix_dict)
    return filtered_dm_series

def get_distance_matrix_from_df(df):
    session_distance_matrices = {}
    for sess, col in df.items():
        session_distance_matrices[sess] = get_distance_matrix_from_coord_series(col.dropna())
    return pd.Series(session_distance_matrices)

def get_distance_matrix_df_allmice(coord_df_dict):
    coord_series_dict = {}
    for mouse, df in coord_df_dict.items():
        coord_series_dict[mouse] = get_distance_matrix_from_df(df)
    return pd.DataFrame(coord_series_dict).transpose()

if __name__ == '__main__':
    os.chdir('/home/ga48kuh/NASgroup/labmembers/filippokiessler/wahl-colab')    
    scriptname = os.path.basename(__file__)[:-3]
    
    savestr = f'code/paper-review/figure-5/outputs/{scriptname}'
    if not os.path.isdir(savestr):
        os.mkdir(savestr)

    
    dff_path = 'data/neural-data/neural-data-clean/dff_all_cells_normal.pkl'
    decon_path = 'data/neural-data/neural-data-clean/decon_all_cells_normal.pkl'
    sa_path = 'data/neural-data/spatial_activity_maps_dff_all_cells.pkl'
    pc_path = 'data/neural-data/neural-data-clean/is_pc_all_cells.pkl'
    coords_path = 'data/neural-data/neural-data-clean/cell_coords_all_cells.pkl'

    with open(dff_path, 'rb') as file:
        dff = pickle.load(file)
    with open(decon_path, 'rb') as file:
        decon  = pickle.load(file)    
    with open(sa_path, 'rb') as file:
        sa  = pickle.load(file)  
    with open(coords_path, 'rb') as file:
        coords = pickle.load(file)
    with open(pc_path, 'rb') as handle:
        place_cells_binary = pickle.load(handle)
        
    #remove mice 112 and 121
    unwanted_mice = [112,121]
    dff = remove_unwanted_mice(dff, unwanted_mice)
    decon = remove_unwanted_mice(decon, unwanted_mice)
    sa = remove_unwanted_mice(sa, unwanted_mice)
    coords = remove_unwanted_mice(coords, unwanted_mice)
    place_cells_binary = remove_unwanted_mice(place_cells_binary, unwanted_mice)
    
    
    #calculate correlation matrices
    dff_corr_clust = neuraldata_cluster_corr(dff)
    decon_corr_clust = neuraldata_cluster_corr(decon)
    sa_corr_clust = neuraldata_cluster_corr(sa)
    #calculate distance matrices
    distance_matrix = get_distance_matrix_df_allmice(coords)

    
    with open(f'{savestr}/correlation-mat-unsorted-dff.pkl', 'wb') as handle:
        pickle.dump(dff_corr_clust, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(f'{savestr}/correlation-mat-unsorted-decon.pkl', 'wb') as handle:
        pickle.dump(decon_corr_clust, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{savestr}/correlation-mat-unsorted-sa.pkl', 'wb') as handle:
        pickle.dump(sa_corr_clust, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{savestr}/distance-mat-unsorted.pkl', 'wb') as handle:
        pickle.dump(distance_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    #now calculate the matrices of cell pairs for the correlation matrices

    #for every mouse, for every session, calculate the matrix of pair type identifiers
    cell_pair_matrices_allmice = pair_matrix_allmice_df(place_cells_binary)    
    
    #save the matrices of pair identifiers for every mouse for every session
    with open(f'{savestr}/mouse-cell-pair-identifiers.pkl', 'wb') as handle:
        pickle.dump(cell_pair_matrices_allmice, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    