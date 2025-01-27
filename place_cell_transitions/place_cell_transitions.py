#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 03/11/2023 09:26
@author: hheise

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

from preprint import data_cleaning as dc
from preprint import placecell_heatmap_transition_functions as func


def quantify_place_cell_transitions(pc_list, align_days=False, day_diff=3, shuffle=None, avg_mat=False):

    def get_adjusted_cell_counts(class_arr):
        counts = np.bincount(class_arr)[1:]
        if len(counts) == 1:
            counts = np.append(counts, values=0)
        elif len(counts) == 2:
            pass
        else:
            return
        return counts

    pc_transitions = []

    for mouse_idx, mouse in enumerate(pc_list):

        mouse_id = list(mouse.keys())[0]
        # if mouse_id == '63_1':
        #     break
        # Extract data and align days if necessary
        rel_days = np.array(mouse[mouse_id][1])

        if align_days:
            if 3 not in rel_days:
                rel_days[(rel_days == 2) | (rel_days == 4)] = 3
            rel_days[(rel_days == 5) | (rel_days == 6) | (rel_days == 7)] = 6
            rel_days[(rel_days == 8) | (rel_days == 9) | (rel_days == 10)] = 9
            rel_days[(rel_days == 11) | (rel_days == 12) | (rel_days == 13)] = 12
            rel_days[(rel_days == 14) | (rel_days == 15) | (rel_days == 16)] = 15
            rel_days[(rel_days == 17) | (rel_days == 18) | (rel_days == 19)] = 18
            rel_days[(rel_days == 20) | (rel_days == 21) | (rel_days == 22)] = 21
            rel_days[(rel_days == 23) | (rel_days == 24) | (rel_days == 25)] = 24
            if 28 not in rel_days:
                rel_days[(rel_days == 26) | (rel_days == 27) | (rel_days == 28)] = 27
            rel_sess = np.arange(len(rel_days)) - np.argmax(np.where(rel_days <= 0, rel_days, -np.inf))
            rel_days[(-5 < rel_sess) & (rel_sess < 1)] = np.arange(-np.sum((-5 < rel_sess) & (rel_sess < 1)) + 1, 1)

        pc_data = mouse[mouse_id][0]
        pc_data = pd.DataFrame(pc_data, columns=rel_days)

        # Ignore day 1
        rel_days = rel_days[rel_days != 1]
        pc_data = pc_data.loc[:, pc_data.columns != 1]

        if shuffle is not None:
            iterations = shuffle
        else:
            iterations = 1

        pc_trans = {'pre': np.zeros((iterations, 3, 3)), 'early': np.zeros((iterations, 3, 3)), 'late': np.zeros((iterations, 3, 3))}

        for i in range(iterations):

            rng = np.random.default_rng()   # Initialize the random generator

            # Loop through days and get place cell transitions between sessions that are 3 days apart
            for day_idx, day in enumerate(rel_days):

                next_day_idx = np.where(rel_days == day + day_diff)[0]

                # If a session 3 days later exists, get place cell transitions
                if len(next_day_idx) == 1:

                    # General PC - non-PC transitions
                    # Add 1 to the pc data to include "Lost" cells
                    day1_pc = pc_data.iloc[:, day_idx].to_numpy() + 1
                    day1_pc = np.nan_to_num(day1_pc).astype(int)
                    day2_pc = pc_data.iloc[:, next_day_idx].to_numpy().squeeze() + 1
                    day2_pc = np.nan_to_num(day2_pc).astype(int)

                    if shuffle is not None:

                        # mat_true = func.transition_matrix(mask1=day1_pc, mask2=day2_pc, num_classes=3, percent=False)
                        # mat_shuff = []
                        # for i in range(shuffle):
                        #     day2_pc_shuff = np.random.default_rng().permutation(day2_pc)
                        #     mat_shuff.append(func.transition_matrix(mask1=day1_pc, mask2=day2_pc_shuff, num_classes=3, percent=False))
                        # mat = np.stack(mat_shuff)

                        mat = func.transition_matrix(mask1=rng.permutation(day1_pc), mask2=day2_pc,
                                                     num_classes=3, percent=False)

                        # mat_shuff_mean = np.mean(mat_shuff, axis=0)
                        #
                        # # Plot shuffled distribution and true value
                        # nrows = mat_true.shape[0]
                        # ncols = mat_true.shape[1]
                        # fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
                        # for i in range(nrows):
                        #     for j in range(ncols):
                        #         # sns.violinplot(y=mat_shuff[:, i, j], ax=ax[i, j], cut=0)
                        #         sns.stripplot(y=mat_shuff[:, i, j], ax=ax[i, j], alpha=0.5)
                        #         ax[i, j].axhline(mat_true[i, j], color='red')
                        #         perc = (np.sum(mat_shuff[:, i, j] < mat_true[i, j]) / len(mat_shuff[:, i, j])) * 100
                        #         ax[i, j].text(0.05, 0.95, perc, transform=ax[i, j].transAxes, verticalalignment='top',
                        #                       horizontalalignment='left')
                    else:
                        mat = func.transition_matrix(mask1=day1_pc, mask2=day2_pc, num_classes=3, percent=False)

                    # Compute Kullback-Leibler divergence for each day separately (from cell class frequencies)
                    day2_counts = get_adjusted_cell_counts(day2_pc)
                    if day2_counts is None:
                        raise IndexError(f'Wrong number of classes in M{mouse_id}, day {day}.')

                    if rel_days[next_day_idx] <= 0:
                        pc_trans['pre'][i] = pc_trans['pre'][i] + mat
                        # print(f'Day {rel_days[next_day_idx]} sorted under "Pre"')

                    elif rel_days[next_day_idx] <= 7:
                        pc_trans['early'][i] = pc_trans['early'][i] + mat
                        # print(f'Day {rel_days[next_day_idx]} sorted under "Early"')

                    else:
                        pc_trans['late'][i] = pc_trans['late'][i] + mat
                        # print(f'Day {rel_days[next_day_idx]} sorted under "Late"')

        if avg_mat:
            pc_trans = {k: np.nanmean(v, axis=0) for k, v in pc_trans.items()}

        pc_transitions.append(pd.DataFrame([dict(mouse_id=int(mouse_id.split('_')[0]),
                                                 pc_pre=pc_trans['pre'].squeeze(), pc_early=pc_trans['early'].squeeze(), pc_late=pc_trans['late'].squeeze(),
                                                 )]))

    return pd.concat(pc_transitions, ignore_index=True)

#%% Function calls
is_pc = dc.load_data('is_pc')

pc_transition = quantify_place_cell_transitions(pc_list=is_pc, align_days=True)
# Many iterations (>500) is needed to get at least one of the more unlikely transitions)
pc_transition_rng = quantify_place_cell_transitions(pc_list=is_pc, shuffle=1000, avg_mat=True, align_days=True)

