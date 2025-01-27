#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 22/08/2023 17:44
@author: hheise

Tables to store different ways of grouping mice (primarily based on VR task performance)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from sklearn.decomposition import PCA
from typing import Union
import datajoint as dj
import login

login.connect()

from schema import common_mice, hheise_behav, hheise_hist
from util import helper

schema = dj.schema('hheise_grouping', locals(), create_tables=True)


@schema
class GroupingParameter(dj.Manual):
    definition = """ # Parameters that determine behavior grouping
    grouping_id     : tinyint
    ---
    perf_metric     : varchar(1048)     # Metric to base performance drop on. Has to be attribute in hheise_behav.VRPerformance
    normalized      : tinyint           # Boolean flag whether the raw or normalized (divided by mean prestroke performance) is used
    early_thresh    : float             # Threshold for early poststroke performance deficit
    late_thresh     : float             # Threshold for late poststroke performance deficit
    early_day       : tinyint           # Day after microsphere injection until when to compute early poststroke performance
    n_last_sessions : tinyint           # Number of sessions that should be averaged for late poststroke period
    late_day        : tinyint           # Day after microsphere injection from when to compute late poststroke performance. Overrides n_last_sessions if mouse has less than >n_last_sessions< sessions after late_day.
    description=NULL: varchar(128)      # Description of parameter set
    include_d1='0'  : tinyint # Bool flag whether to include poststroke day 1 in performance clustering
    """


@schema
class BehaviorGroupCluster(dj.Lookup):
    definition = """ # Different ways of grouping mice based on behavior
    cluster         : varchar(64)     # Name of group cluster
    ---
    cluster_info    : varchar(1048)   # Additional details
    """

    contents = [
        ['coarse', 'Broad clustering, dividing into "Control" and "Stroke".'],
        ['fine',
         'Finer clustering, dividing into "Sham" (No spheres in brain), "No Deficit" (above the early-poststroke'
         'threshold), "Recovery" (below threshold in early, but not late poststroke) and "No Recovery"'
         '(below threshold in early and late poststroke).'],
    ]


@schema
class BehaviorGroup(dj.Lookup):
    definition = """ # Different behavior groups
    group                       : varchar(64)       # Name of the behavioral group
    ---
    description                 : varchar(256)      # Description/criteria of the group
    """

    contents = [
        ['Control', 'Above a certain threshold of relative VR performance in early poststroke.'],
        ['Stroke', 'Below the threshold in early poststroke.'],
        ['Sham', 'No spheres in brain, irrespective of performance.'],
        ['No Deficit', 'Above the threshold in early poststroke.'],
        ['Recovery', 'Below the threshold in early, but not late poststroke.'],
        ['No Recovery', 'Below the threshold in early and late poststroke.'],
    ]


@schema
class BehaviorGrouping(dj.Computed):
    definition = """ # Group membership and performance effects of single mice.
    -> GroupingParameter
    -> BehaviorGroupCluster
    -> common_mice.Mouse        # Only mice that have entries in hheise_behav.VRSession are included
    ---
    -> BehaviorGroup
    early       : float         # Performance value in early poststroke phase (x coordinate in matrix)
    late        : float         # Performance value in late poststroke phase (y coordinate in matrix)
    """

    # Only include mice that are completely available
    include_mice = [33, 41, 63, 69, 83, 85, 86, 89, 90, 91, 93, 95, 108, 110, 111, 112, 113, 114, 115, 116, 122]
    _key_source = (common_mice.Mouse() * GroupingParameter * BehaviorGroupCluster) & \
                  "username='hheise'" & f"mouse_id in {helper.in_query(include_mice)}"

    def make(self, key: dict):

        # print(key)

        # Fetch parameters
        params = (GroupingParameter & key).fetch1()

        perf = (hheise_behav.VRPerformance &
                f'mouse_id={key["mouse_id"]}').get_normalized_performance(attr=params['perf_metric'], pre_days=3,
                                                                          baseline_days=3, plotting=False,
                                                                          normalize=params['normalized'])

        # Drop a few outlier sessions (usually last session of a mouse that should not be used)
        perf = perf.drop(perf[(perf['mouse_id'] == 83) & (perf['day'] == 27)].index)
        perf = perf.drop(perf[(perf['mouse_id'] == 69) & (perf['day'] == 23)].index)

        # Make sure that data is sorted chronologically for n_last_sessions to work
        perf = perf.sort_values('day')

        # Early timepoint (invert include_d1 to get first poststroke day)
        early = perf[(perf['day'] > (1 - params['include_d1'])) & (perf['day'] <= params['early_day'])]['performance'].mean()

        # Late timepoint
        if (perf['day'] >= params['late_day']).sum() < params['n_last_sessions']:
            # If mouse has less than >n_last_sessions< sessions after late_day,
            # take mean of all available sessions >= late_date
            late = perf[perf['day'] >= params['late_day']]['performance'].mean()
        else:
            # Otherwise, compute late performance from the last "n_last_sessions" sessions
            late = perf['performance'].iloc[-params['n_last_sessions']:].mean()

        # Sort mouse into groups
        if key['cluster'] == 'coarse':
            if early < params['early_thresh']:
                group = 'Stroke'
            else:
                group = 'Control'
        elif key['cluster'] == 'fine':
            if key["mouse_id"] in [91, 111, 115, 122]:  # Sham mice are hardcoded
                group = 'Sham'
            elif early < params['early_thresh']:
                if late < params['late_thresh']:
                    group = 'No Recovery'
                else:
                    group = 'Recovery'
            else:
                group = 'No Deficit'
        else:
            raise ValueError(f'Cluster {params["cluster"]} not defined.')

        self.insert1(dict(**key, group=group, early=early, late=late))

    def get_groups(self, as_frame: bool = True):

        data = pd.DataFrame(self.fetch(as_dict=True))

        if len(data.cluster.unique()) != 1:
            raise IndexError('Can only query a single cluster at a time.')

        if len(data.grouping_id.unique()) == 1:
            if as_frame:
                output = data[['mouse_id', 'group']]
            else:
                output = {group: list(data[data['group'] == group]['mouse_id']) for group in data['group'].unique()}
        else:
            if as_frame:
                output = data[['grouping_id', 'mouse_id', 'group']]
            else:
                output = {grouping_id: {group: list(data[(data['group'] == group) &
                                                         (data['grouping_id'] == grouping_id)]['mouse_id'])
                                        for group in data['group'].unique()} for grouping_id in data['grouping_id'].unique()}
        return output


@schema
class ClusterSummary(dj.Computed):
    definition = """ # Summary of behavioral clusters across all mice. Has to be populated after BehaviorGrouping.
    -> GroupingParameter
    -> BehaviorGroupCluster
    ---
    sphere_early        : float     # Correlation of sphere count with position on X-axis (early performance) 
    sphere_late         : float     # Correlation of sphere count with position on Y-axis (late performance)
    sphere_pca          : float     # Correlation of sphere count with 1st principal component dimension
    var_explained       : float     # Fraction variance explained by 1st principal component of early-late performance
    """

    def make(self, key: dict):

        curr_grouping = pd.DataFrame((BehaviorGrouping & key).fetch(as_dict=True))

        if len(curr_grouping) == 0:
            raise ImportError(f'No data available in hheise_grouping.BehaviorGrouping for key {key}. Populate first!')

        # Fetch total (extrapolated) sphere counts and add to DataFrame
        sphere_count = pd.DataFrame((hheise_hist.MicrosphereSummary.Metric &
                                     f'mouse_id in {helper.in_query(curr_grouping["mouse_id"].unique())}' &
                                     'metric_name="spheres"').fetch('count_extrap', 'mouse_id', as_dict=True))
        sphere_count = sphere_count.rename(columns={'count_extrap': 'spheres'})
        joint_grouping = curr_grouping.set_index('mouse_id').join(sphere_count.set_index('mouse_id'))

        # Perform PCA on early + late performance to find common dimension of both metrics
        pca = PCA(n_components=1)
        pca_data = joint_grouping[['early', 'late']]        # Extract the 2D data used for PCA
        data_mean = pca_data.mean(axis=0)                   # Center data before PCA (subtract mean to get mean of 0)
        data_centered = pca_data - data_mean
        reduced_metric = pca.fit_transform(data_centered)

        # If primary component is negative (arrow points down-left), invert to make consistent with other metrics
        if np.all(pca.components_ < 0):
            reduced_metric = -reduced_metric
            pca.components_ = -pca.components_

        # Correlate sphere count to metrics
        sphere_early = np.corrcoef(joint_grouping['early'], joint_grouping['spheres'])[0, 1]
        sphere_late = np.corrcoef(joint_grouping['late'], joint_grouping['spheres'])[0, 1]
        sphere_pca = np.corrcoef(reduced_metric.squeeze(), joint_grouping['spheres'])[0, 1]

        self.insert1(dict(**key, sphere_early=sphere_early, sphere_late=sphere_late, sphere_pca=sphere_pca,
                          var_explained=pca.explained_variance_ratio_[0]))

    def plot_matrix(self, pca_results: bool = True, sphere_hue: bool = True, plot_pca: bool = False,
                    equalize_axis: bool = True):

        def plot_scatter(data, ax, early_thresh, late_thresh, title=None, legend=True, ):

            sns.scatterplot(data=data, x='early', y='late', hue='spheres', palette='flare', hue_norm=LogNorm(), s=100,
                            ax=ax, legend=legend)
            if legend:
                ax.legend(title='Spheres', fontsize='10', title_fontsize='12')
            # Label each point with mouse_id
            for i, point in enumerate(ax.collections):
                # Extract the x and y coordinates of the data point
                x = point.get_offsets()[:, 1]
                y = point.get_offsets()[:, 0]

                # Add labels to the data point
                for j, y_ in enumerate(y):
                    ax.text(y[j], x[j] - 0.05, data[data['early'] == y_].index[0],
                            ha='center', va='bottom', fontsize=12)

            ax.axvline(early_thresh, linestyle='--', color='grey')
            ax.axhline(late_thresh, linestyle='--', color='grey')

            if title is not None:
                ax.set_title(title)

        def draw_pca(model, ax, reduced_data=None, mean_offset: Union[np.ndarray, int] = 0):

            def reset_ax_lim(curr_lim, points):
                lower_lim = np.min(points[points < curr_lim[1]]) if np.sum(points < curr_lim[0]) > 0 else curr_lim[0]
                upper_lim = np.max(points[points > curr_lim[1]]) if np.sum(points > curr_lim[1]) > 0 else curr_lim[1]
                return lower_lim, upper_lim

            # Draw arrow in direction of 1st principal component
            # The arrow's direction comes from the PCA vector "components_", and the length is determined by the explained variance
            lims = []
            for length, vector in zip(model.explained_variance_, model.components_):
                v = vector * 3 * np.sqrt(length)  # End coordinates of arrow vector
                origin = model.mean_ + mean_offset  # Origin coordinates of vector (shifted by mean of dataset)
                head = model.mean_ + v + mean_offset  # End coordinates of arrow (shifted by mean of dataset)

                arrowprops = dict(arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0)
                ax.annotate('', head, origin, arrowprops=arrowprops)

                lims.append(origin)
                lims.append(head)
            lims = np.vstack(lims)

            if reduced_data is not None:
                reduced_data = reduced_data + mean_offset
                ax.scatter(x=reduced_data[:, 0], y=reduced_data[:, 1], color='g', alpha=0.6)

            ax.set_xlim(reset_ax_lim(curr_lim=ax.get_xlim(), points=lims[:, 0]))
            ax.set_ylim(reset_ax_lim(curr_lim=ax.get_ylim(), points=lims[:, 1]))

        summaries = self.fetch(as_dict=True)
        grouping = [pd.DataFrame((BehaviorGrouping & row).fetch(as_dict=True)) for row in self.fetch()]
        params = [(GroupingParameter & row).fetch1() for row in self.fetch()]

        # Fetch total (extrapolated) sphere counts and add to DataFrame
        if sphere_hue:
            sphere_count = pd.DataFrame((hheise_hist.MicrosphereSummary.Metric &
                                         f'mouse_id in {helper.in_query(grouping[0]["mouse_id"].unique())}' &
                                         'metric_name="spheres"').fetch('count_extrap', 'mouse_id', as_dict=True))
            sphere_count = sphere_count.rename(columns={'count_extrap': 'spheres'})

        if len(grouping) > 1:
            fig, axes = plt.subplots(nrows=2, ncols=int(np.ceil(len(grouping)/2)), layout='constrained')
        else:
            fig = plt.figure(layout='constrained')
            ax = plt.gca()
            axes = np.array([ax])

        for i, (curr_summary, curr_grouping, curr_params, curr_ax) in enumerate(zip(summaries, grouping, params, axes.flatten())):
            if sphere_hue:
                joint_grouping = curr_grouping.set_index('mouse_id').join(sphere_count.set_index('mouse_id'))
            else:
                joint_grouping = curr_grouping

            if pca_results or plot_pca:
                # Perform PCA on early + late performance to find common dimension of both metrics
                pca = PCA(n_components=1)
                pca_data = joint_grouping[['early', 'late']]    # Extract the 2D data used for PCA
                data_mean = pca_data.mean(axis=0)               # Center data before PCA (subtract mean to get mean of 0)
                data_centered = pca_data - data_mean
                reduced_metric = pca.fit_transform(data_centered)

                # If primary component is negative (arrow points down-left), invert to make consistent with other metrics
                if np.all(pca.components_ < 0):
                    reduced_metric = -reduced_metric
                    pca.components_ = -pca.components_

            # Display stats of PCA in textbox
            props = dict(boxstyle='round', alpha=0)
            text = r'$r_{early}$' + f' = {curr_summary["sphere_early"]:.2f}\n' +\
                   r'$r_{late}$' + f' = {curr_summary["sphere_late"]:.2f}'

            if pca_results:
                text = f'%var: {pca.explained_variance_ratio_[0]:.2f}\n' +\
                       r'$r_{pca}$' + f' = {curr_summary["sphere_pca"]:.2f}\n' + text

            curr_ax.text(0.95, 0.05, text, transform=curr_ax.transAxes, fontsize=14, verticalalignment='bottom',
                         horizontalalignment='right', bbox=props, fontfamily='monospace')

            title = f'Grouping_ID {curr_params["grouping_id"]} - {curr_params["perf_metric"]}'

            plot_scatter(data=joint_grouping, ax=curr_ax, legend=(i < len(summaries) - 1),
                         title=title, early_thresh=curr_params['early_thresh'],
                         late_thresh=curr_params['late_thresh'])

            if equalize_axis:
                helper.equalize_axis(curr_ax, plot_diagonal=True)

            if plot_pca:
                draw_pca(model=pca, ax=curr_ax, reduced_data=pca.inverse_transform(reduced_metric),
                         mean_offset=np.array(data_mean))



