#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 11/04/2023 14:00
@author: hheise

"""
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from schema import hheise_placecell, common_match, hheise_behav, hheise_pvc, hheise_grouping
import preprint.data_cleaning as dc


def plot_pvc_curve(y_vals, session_stdev, bin_size=5, show=False):
    """Plots the pvc curve

        Parameters
        ----------
        y_vals : array-like
            data points of the pvc curve (idx = bin distance)
        bin_size : bool, optional
        show : bool, optional

       Returns
       -------
       fig: figure object
           a figure object of the pvc curve
    """
    fig = plt.figure()
    x_axis = np.arange(0., len(y_vals) * bin_size, bin_size)  # bin size
    plt.errorbar(x_axis, y_vals, session_stdev, figure=fig)
    plt.ylim(bottom=0); plt.ylabel('Mean PVC')
    plt.xlim(left=0); plt.xlabel('Offset Distances (cm)')
    if show:
        plt.show(block=True)
    return fig


def pvc_curve(session1, session2, plot=True, max_delta_bins=79, circular=False):
    """Calculate the mean pvc curve between two sessions.

        Parameters
        ----------
        activity_matrix : 2D array containing (float, dim1 = bins, dim2 = neurons)
        plot: bool, optional
        max_delta_bins: int, optional
            max difference in bin distance. Default is entire corridor.

       Returns
       -------
       curve_yvals:
           array of mean pvc curve (idx = delta_bin)
    """

    # Filter out neurons that are not active in both sessions
    neuron_mask = np.sum(np.array([~np.isnan(session1).any(axis=1),
                                   ~np.isnan(session2).any(axis=1)]).astype(int), axis=0) == 2
    session1 = session1[neuron_mask].T
    session2 = session2[neuron_mask].T
    logging.info(f'{np.sum(neuron_mask)} neurons available')

    num_bins = np.size(session1, 0)
    num_neurons = np.size(session1, 1)
    curve_yvals = np.empty(max_delta_bins + 1)
    curve_stdev = np.empty(max_delta_bins + 1)

    pvc_mat = np.zeros((max_delta_bins + 1, num_bins)) * np.nan

    for delta_bin in range(max_delta_bins + 1):
        pvc_vals = []

        if circular:
            max_offset = num_bins
        else:
            max_offset = num_bins - delta_bin

        for offset in range(max_offset):
            idx_x = offset

            if circular:
                idx_y = offset + (delta_bin - num_bins)     # This wraps around the end of the corridor
            else:
                idx_y = offset + delta_bin  # This only yields the next bin in the same corridor (no wrapping)

            pvc_xy_num = pvc_xy_den_term1 = pvc_xy_den_term2 = 0
            for neuron in range(num_neurons):
                pvc_xy_num += session1[idx_x][neuron] * session2[idx_y][neuron]
                pvc_xy_den_term1 += session1[idx_x][neuron] * session1[idx_x][neuron]
                pvc_xy_den_term2 += session2[idx_y][neuron] * session2[idx_y][neuron]
            pvc_xy = pvc_xy_num / (np.sqrt(pvc_xy_den_term1 * pvc_xy_den_term2))
            pvc_vals.append(pvc_xy)

        if circular:
            pvc_mat[delta_bin] = pvc_vals
        else:
            # If not wrapping around, the matrix for all delta_bin > 0 do not fill the array completely
            pvc_mat[delta_bin, :len(pvc_vals)] = pvc_vals

        mean_pvc_delta_bin = np.mean(pvc_vals)
        stdev_delta_bin = np.std(pvc_vals)
        curve_yvals[delta_bin] = mean_pvc_delta_bin
        curve_stdev[delta_bin] = stdev_delta_bin

    if plot:
        plot_pvc_curve(curve_yvals, curve_stdev, show=True)

    return curve_yvals, curve_stdev, pvc_mat


def pvc_across_sessions(session1, session2, plot_heatmap=False, plot_in_ax=None, plot_zones=False,):
    """
    Compute PVC across sessions (see Shuman2020 Fig 3c).
    For each position bin X, compute population vector correlation of all neurons between two sessions.
    The resulting matrix has position bins of session1 on x-axis, and position bins of session2 on y-axis.
    Mean PVC curves are computed by comparing not the same position bin X between both sessions, but offsetting the
    position bin in one session by N centimeters, and average across all position-bin-pairs with the same offset.
    """

    # Filter out neurons that are not active in both cells
    neuron_mask = np.sum(np.array([~np.isnan(session1).any(axis=1),
                                   ~np.isnan(session2).any(axis=1)]).astype(int), axis=0) == 2
    session1 = session1[neuron_mask]
    session2 = session2[neuron_mask]
    logging.info(f'{np.sum(neuron_mask)} neurons available')

    ### Compute PVC across all positions between the two sessions
    pvc_matrix = np.zeros((session1.shape[1], session2.shape[1])) * np.nan
    for x in range(session1.shape[1]):
        for y in range(session2.shape[1]):
            # Multiply spatial maps of the same neuron of both sessions and sum across neurons (numerator)
            numerator = np.sum(session1[:, x] * session2[:, y])
            # Multiply spatial maps of the same neuron of both sessions with each other, sum across neurons, and multiply (denominator)
            denominator = np.sum(session1[:, x] * session1[:, x]) * np.sum(session2[:, y] * session2[:, y])

            # Compute PVC for this position combination
            pvc_matrix[x, y] = numerator / np.sqrt(denominator)

    if plot_heatmap:
        if plot_in_ax is not None:
            curr_ax = plot_in_ax
        else:
            plt.figure()
            curr_ax = plt.gca()
        sns.heatmap(pvc_matrix, ax=curr_ax, vmin=0, vmax=1, cbar=False, cmap='turbo')
        if plot_zones:
            zone_borders = (hheise_behav.CorridorPattern & 'pattern="training"').rescale_borders(n_bins=80)
            for zone in zone_borders:
                curr_ax.axvspan(zone[0], zone[1], facecolor='green', alpha=0.4)

    return pvc_matrix


def draw_single_mouse_heatmaps(mouse_dict, day_diff=3, v_min=None, v_max=None, cmap='turbo', plot_last_cbar=True,
                               draw_zone_borders=True, verbose=False, only_return_matrix=False,
                               directory=None):
    """
    Draw PVC heatmaps across time (pre, pre-post, early, late). One figure per mouse.

    Args:
        mouse_dict: Dict with one entry (key is mouse_id). Value is a list with 2 elements -
            3D array with shape (n_neurons, n_sessions, n_bins) of spatial activity maps for a single mouse
            List of days relative to microsphere injection per session.
        day_diff: day difference at which to compare sessions.
        v_min: minimum scale of the heatmap. If both are set, dont draw color bar.
        v_max: maximum scale of the heatmap. If both are set, dont draw color bar.

    Returns:

    """

    if verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO, force=True)

    if v_min is not None and v_max is not None:
        plot_cbar = False
    else:
        plot_cbar = True

    if len(mouse_dict) != 1:
        raise KeyError('Only accepting single-key dicts.')

    mouse_id = list(mouse_dict.keys())[0]
    days = np.array(mouse_dict[mouse_id][1])
    last_pre_day = days[np.searchsorted(days, 0, side='right') - 1]  # Find last prestroke day
    spat_map = mouse_dict[mouse_id][0]

    prestroke = []
    pre_poststroke = []
    early_stroke = []
    late_stroke = []
    for day_idx, day in enumerate(days):
        next_day_idx = np.where(days == day + day_diff)[0]

        # After stroke, ignore small differences between sessions (do not have to be 3 days apart, sometimes 2 days)
        # In that case, use the next session irrespective of distance
        if (len(next_day_idx) == 0) and (1 < day < np.max(days)):
            next_day_idx = [day_idx + 1]

        # If a session 3 days later exists, compute the correlation of all cells between these sessions
        # Do not analyze session 1 day after stroke (unreliable data)
        if len(next_day_idx) == 1:
            curr_mat = pvc_across_sessions(session1=spat_map[:, day_idx],
                                           session2=spat_map[:, next_day_idx[0]],
                                           plot_heatmap=False)
            logging.info(f'Day {day} - next_day {days[next_day_idx[0]]}')
            if day < last_pre_day:
                prestroke.append(curr_mat)
                logging.info('\t-> Pre')
            elif day == last_pre_day:
                pre_poststroke.append(curr_mat)
                logging.info('\t-> Pre-Post')
            # elif last_pre_day < days[next_day_idx[0]] <= 6:
            elif last_pre_day < day < 6:
                early_stroke.append(curr_mat)
                logging.info('\t-> Early Post')
            elif 6 <= day:
                late_stroke.append(curr_mat)
                logging.info('\t-> Late Post')

    avg_prestroke = np.mean(np.stack(prestroke), axis=0)
    pre_poststroke = pre_poststroke[0]
    avg_early_stroke = np.mean(np.stack(early_stroke), axis=0)
    avg_late_stroke = np.mean(np.stack(late_stroke), axis=0)

    if only_return_matrix:
        return {'pre': avg_prestroke, 'pre_post': pre_poststroke, 'early': avg_early_stroke, 'late': avg_late_stroke}

    fig, axes = plt.subplots(1, 4, sharex='all', sharey='all', figsize=(23, 6), layout='constrained')
    sns.heatmap(avg_prestroke, ax=axes[0], vmin=v_min, vmax=v_max, square=True, cbar=plot_cbar, cmap=cmap)
    sns.heatmap(pre_poststroke, ax=axes[1], vmin=v_min, vmax=v_max, square=True, cbar=plot_cbar, cmap=cmap)
    sns.heatmap(avg_early_stroke, ax=axes[2], vmin=v_min, vmax=v_max, square=True, cbar=plot_cbar, cmap=cmap)
    sns.heatmap(avg_late_stroke, ax=axes[3], vmin=v_min, vmax=v_max, square=True, cbar=plot_last_cbar, cmap=cmap)

    # Make plot pretty
    titles = ['Prestroke', 'Pre-Post', 'Early Post', 'Late Post']
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('Session 2')
    axes[0].set_ylabel('Session 1')

    if draw_zone_borders:
        zone_borders = (hheise_behav.CorridorPattern & 'pattern="training"').rescale_borders(n_bins=80)
        for curr_ax in axes:
            for zone in zone_borders:
                # curr_ax.axvspan(zone[0], zone[1], facecolor='gray', alpha=0.4)
                curr_ax.axvline(zone[0], color='black', linestyle='--')
                curr_ax.axvline(zone[1], color='black', linestyle='--')

    fig.canvas.manager.set_window_title(mouse_id)

    if directory is not None:
        plt.savefig(os.path.join(directory, f'{mouse_id}.png'))
        plt.close()

    # Re-set logging level
    if verbose:
        logging.basicConfig(level=logging.WARNING, force=True)


def figure_plots(matrices, vmin=0, vmax=1, cmap='turbo', draw_zone_borders=True, directory=None):

    curves = []
    for row, (mouse_id, mouse_mats) in enumerate(matrices.items()):
        for col, (phase, mat) in enumerate(mouse_mats.items()):

            if directory is not None:

                # Plot PVC matrix
                plt.figure(layout='constrained', figsize=(12, 12), dpi=300)
                ax = sns.heatmap(mat, vmin=vmin, vmax=vmax, square=True, cbar=False, cmap=cmap)
                ax.set(xticks=[], yticks=[])

                plt.savefig(os.path.join(directory, f"{mouse_id}_{phase}.png"))

                if draw_zone_borders:
                    zone_borders = (hheise_behav.CorridorPattern & 'pattern="training"').rescale_borders(n_bins=80)
                    # zone_borders = np.round(zone_borders)
                    for zone in zone_borders:
                        # curr_ax.axvspan(zone[0], zone[1], facecolor='gray', alpha=0.4)
                        ax.axvline(zone[0], color='black', linestyle='--')
                        ax.axvline(zone[1], color='black', linestyle='--')
                    plt.savefig(os.path.join(directory, f"{mouse_id}_{phase}_zones.png"))
                plt.close()

            # Query, compute and plot average PVC curve
            curve = np.nanmean(np.stack((hheise_pvc.PvcCrossSessionEval * hheise_pvc.PvcCrossSession &
                                         'locations="all"' & 'circular=0' & f'mouse_id={mouse_id}' &
                                        f'phase="{phase}"').fetch('pvc_curve')), axis=0)

            curves.append(pd.DataFrame(dict(mouse_id=mouse_id, phase=phase, pvc=curve,
                                            pos=np.linspace(5, 400, 80).astype(int))))
    return pd.concat(curves, ignore_index=True)

#%% PVC Heatmaps
spatial_maps = dc.load_data('dff_maps')

####################################################################################################################

### Take average of all pre, early and post pvc matrices
DAY_DIFF = 3    # The day difference between sessions to be compared (usually 3)
vmin = 0
vmax = 1

for mouse in spatial_maps:
    draw_single_mouse_heatmaps(mouse_dict=mouse, day_diff=DAY_DIFF, v_min=vmin, v_max=vmax, verbose=False,
                               directory=r"W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\PVC\avg_matrices")


# Make final overview for figure 3: One mouse per group + avg PVC curves
sham_mouse = 91
nodef_mouse = 114
rec_mouse = 90
norec_mouse = 41

plot_matrices = {91: draw_single_mouse_heatmaps(spatial_maps[9], v_min=vmin, v_max=vmax, verbose=False,
                                           plot_last_cbar=False, only_return_matrix=True),
            114: draw_single_mouse_heatmaps(spatial_maps[16], v_min=vmin, v_max=vmax, verbose=False,
                                           plot_last_cbar=False, only_return_matrix=True),
            90: draw_single_mouse_heatmaps(spatial_maps[8], v_min=vmin, v_max=vmax, verbose=False,
                                           plot_last_cbar=False, only_return_matrix=True),
            41: draw_single_mouse_heatmaps(spatial_maps[1], v_min=vmin, v_max=vmax, verbose=False,
                                           plot_last_cbar=False, only_return_matrix=True)
            }
curves = figure_plots(plot_matrices, directory=r"W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\PVC\avg_matrices\figure3")
