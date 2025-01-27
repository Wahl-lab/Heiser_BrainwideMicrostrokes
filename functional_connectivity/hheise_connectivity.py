#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 31/10/2023 12:55
@author: hheise

"""

import numpy as np
from scipy import stats

import datajoint as dj
import login

login.connect()

from util import helper
from schema import common_img, hheise_placecell, hheise_behav

schema = dj.schema('hheise_connectivity', locals(), create_tables=True)


@schema
class NeuronNeuronCorrelation(dj.Computed):
    definition = """ # Correlation of unbinned traces and spatial activity maps between neurons of the same session
    -> hheise_placecell.BinnedActivity
    trace_type              : varchar(32)   # Data type that was correlated. Can be raw trace or spatial activity maps, and dF/F or decon. 
    ----
    corr_matrix             : longblob      # Lower triangle (half matrix) of neuron-neuron correlation matrix with shape (n_cells, n_cells).
    perc99                  : float         # 99th percentile of the correlation matrix
    perc95                  : float         # 95th percentile of the correlation matrix
    perc90                  : float         # 90th percentile of the correlation matrix
    perc80                  : float         # 80th percentile of the correlation matrix
    skewness                : float         # Skewness of the (Fisher z-corrected) correlation coefficient distribution
    p_skew                  : float         # P-value of the null hypothesis that skewness is not significantly different from 0
    avg_corr                : float         # Mean correlation across all cells
    sd_corr                 : float         # Standard deviation of correlation across all cells
    median_corr             : float         # Median correlation across all cells
    avg_corr_pc = NULL      : float         # Mean correlation across all place cells (Bartos). NULL if session did not have place cells.
    median_corr_pc = NULL   : float         # Median correlation across all place cells (Bartos). NULL if session did not have place cells.
    """

    _key_source = hheise_placecell.BinnedActivity & 'place_cell_id=2'

    def make(self, key):

        # key = {'username': 'hheise', 'mouse_id': 121, 'day': datetime(2022, 8, 24).date(), 'session_num': 1, 'place_cell_id': 2}

        def compute_correlation_matrix(arr, mask) -> dict:

            # Compute correlation matrix
            corr_mat = np.corrcoef(arr)

            """
            # Clustering the correlation matrix
            
            sns.heatmap(corr_mat)

            # Compute distances
            distances = distance.pdist(corr_mat)
            dist_mat = hierarchy.linkage(distances, method='ward')

            # Plot dendrogram
            dn = hierarchy.dendrogram(dist_mat)

            # Set number of clusters and assign labels
            threshold = distances.max()/2
            labels = hierarchy.fcluster(dist_mat, threshold, criterion='distance')
            idx_sort = np.argsort(labels)

            corr_mat_clust = corr_mat[idx_sort, :][:, idx_sort]

            plt.figure()
            sns.heatmap(corr_mat_clust, cmap="icefire", vmin=-1, vmax=1)
            """

            # Take lower triangle
            corr_mat[np.triu_indices_from(corr_mat, k=0)] = np.nan

            # Compute stats from Fisher z-transformed correlation coefficients
            corr_mat_z = np.arctanh(corr_mat)

            # Compute stats
            return dict(corr_matrix=corr_mat, perc99=np.nanpercentile(corr_mat, 99),
                        perc95=np.nanpercentile(corr_mat, 95),
                        perc90=np.nanpercentile(corr_mat, 90),
                        perc80=np.nanpercentile(corr_mat, 80),
                        skewness=stats.skew(corr_mat_z, nan_policy='omit', axis=None),
                        p_skew=stats.skewtest(corr_mat_z, nan_policy='omit', axis=None)[1],
                        avg_corr=np.tanh(np.nanmean(corr_mat_z)),
                        sd_corr=np.tanh(np.nanstd(corr_mat_z)),
                        median_corr=np.tanh(np.nanmedian(corr_mat_z)),
                        avg_corr_pc=np.tanh(np.nanmean(corr_mat_z[mask, :][:, mask])),
                        median_corr_pc=np.tanh(np.nanmedian(corr_mat_z[mask, :][:, mask])))

        # print(key)

        # Fetch data
        dff = (common_img.Segmentation & key).get_traces('dff')
        decon = (common_img.Segmentation & key).get_traces('decon', decon_id=1)
        spat_dff = (hheise_placecell.BinnedActivity & key).get_trial_avg('dff')
        spat_decon = (hheise_placecell.BinnedActivity & key).get_trial_avg('decon')

        # Make trial and running masks
        trial_mask = (hheise_placecell.PCAnalysis & key).fetch1('trial_mask')
        norm_trials = (hheise_behav.VRSession & key).get_normal_trials()
        running_mask, aligned_frames = (hheise_placecell.Synchronization.VRTrial & key &
                                        f'trial_id in {helper.in_query(norm_trials)}').fetch('running_mask', 'aligned_frames')
        running_mask = np.concatenate(running_mask)

        # First filter out non-normal trials
        dff_filt = dff[:, np.isin(trial_mask, norm_trials)]
        decon_filt = decon[:, np.isin(trial_mask, norm_trials)]
        # Then filter out non-running frames
        dff_filt = dff_filt[:, running_mask]
        decon_filt = decon_filt[:, running_mask]

        # Make place cell mask
        mask_ids = (common_img.Segmentation.ROI & key & 'accepted=1').fetch('mask_id')
        pc_mask = np.zeros(mask_ids.shape, dtype=bool)
        pc_ids = (hheise_placecell.PlaceCell.ROI & key & 'corridor_type=0' & 'is_place_cell=1').fetch('mask_id')
        pc_mask[np.isin(mask_ids, pc_ids)] = True

        entries = [dict(**key, trace_type='dff', **compute_correlation_matrix(dff_filt, pc_mask)),
                   dict(**key, trace_type='decon', **compute_correlation_matrix(decon_filt, pc_mask)),
                   dict(**key, trace_type='spat_dff', **compute_correlation_matrix(spat_dff, pc_mask)),
                   dict(**key, trace_type='spat_decon', **compute_correlation_matrix(spat_decon, pc_mask))]

        self.insert(entries)


@schema
class NeuronNeuronCorrelationKDE(dj.Computed):
    definition = """ # Kernel density estimate from correlation matrix in NeuronNeuronCorrelation
    -> NeuronNeuronCorrelation
    ----
    y_kde    : longblob      # Y-values of KDE at 200 bin resolution
    """

    def make(self, key):

        corrmat = (NeuronNeuronCorrelation & key).fetch1('corr_matrix')

        coefs = corrmat.flatten()[~np.isnan(corrmat.flatten())]  # Extract correlation matrix, skip NaNs
        kde = stats.gaussian_kde(coefs)       # Compute KDE
        # Compute y-values of KDE at x-values from -1 to 1 (correlation coefficient range), with 200 bins
        x = np.linspace(-1, 1, 200)
        self.insert1(dict(**key, y_kde=kde.pdf(x)))

