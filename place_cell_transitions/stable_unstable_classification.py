#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 22/08/2023 14:08
@author: hheise

"""
import os
import numpy as np
import pandas as pd
from typing import Optional, Iterable, List, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns

from schema import hheise_grouping, hheise_placecell, common_match, hheise_behav
from preprint import data_cleaning as dc
from preprint import placecell_heatmap_transition_functions as func


def classify_stability(is_pc_list, spatial_map_list, for_prism=True, ignore_lost=False, align_days=False, aligned_column_names=False):

    mice = [next(iter(dic.keys())) for dic in spatial_map_list]

    dfs = []
    for i, mouse in enumerate(mice):
        rel_dates = np.array(is_pc_list[i][mouse][1])

        if align_days:
            if 3 not in rel_dates:
                rel_dates[(rel_dates == 2) | (rel_dates == 4)] = 3
            rel_dates[(rel_dates == 5) | (rel_dates == 6) | (rel_dates == 7)] = 6
            rel_dates[(rel_dates == 8) | (rel_dates == 9) | (rel_dates == 10)] = 9
            rel_dates[(rel_dates == 11) | (rel_dates == 12) | (rel_dates == 13)] = 12
            rel_dates[(rel_dates == 14) | (rel_dates == 15) | (rel_dates == 16)] = 15
            rel_dates[(rel_dates == 17) | (rel_dates == 18) | (rel_dates == 19)] = 18
            rel_dates[(rel_dates == 20) | (rel_dates == 21) | (rel_dates == 22)] = 21
            rel_dates[(rel_dates == 23) | (rel_dates == 24) | (rel_dates == 25)] = 24
            if 28 not in rel_dates:
                rel_dates[(rel_dates == 26) | (rel_dates == 27) | (rel_dates == 28)] = 27

            # Uncomment this code if prestroke days should also be aligned (probably not good nor necessary)
            rel_sess = np.arange(len(rel_dates)) - np.argmax(np.where(rel_dates <= 0, rel_dates, -np.inf))
            rel_dates[(-5 < rel_sess) & (rel_sess < 1)] = np.arange(-np.sum((-5 < rel_sess) & (rel_sess < 1)) + 1, 1)

        mask_pre = rel_dates <= 0
        mask_early = (0 < rel_dates) & (rel_dates <= 7)
        mask_late = rel_dates > 7

        classes_pre, stability_thresh = func.quantify_stability_split(is_pc_arr=is_pc_list[i][mouse][0][:, mask_pre],
                                                                      spat_dff_arr=spatial_map_list[i][mouse][0][:, mask_pre],
                                                                      rel_days=rel_dates[mask_pre], mouse_id=mouse)
        classes_early = func.quantify_stability_split(is_pc_arr=is_pc_list[i][mouse][0][:, mask_early],
                                                      spat_dff_arr=spatial_map_list[i][mouse][0][:, mask_early],
                                                      rel_days=rel_dates[mask_early], stab_thresh=stability_thresh)
        classes_late = func.quantify_stability_split(is_pc_arr=is_pc_list[i][mouse][0][:, mask_late],
                                                     spat_dff_arr=spatial_map_list[i][mouse][0][:, mask_late],
                                                     rel_days=rel_dates[mask_late], stab_thresh=stability_thresh)

        df = pd.concat([func.summarize_quantification(classes_pre, 'pre'),
                        func.summarize_quantification(classes_early, 'early'),
                        func.summarize_quantification(classes_late, 'late')])

        df['mouse_id'] = mouse
        df['classes'] = [classes_pre, classes_early, classes_late]
        dfs.append(df)

    class_df = pd.concat(dfs, ignore_index=True)

    if ignore_lost:

        def remove_lost_cells(row):
            n_cells = row[['n1', 'n2', 'n3']].sum()
            row['n0_r'] = np.nan
            row['n1_r'] = (row['n1'] / n_cells) * 100
            row['n2_r'] = (row['n2'] / n_cells) * 100
            row['n3_r'] = (row['n3'] / n_cells) * 100

            return row

        class_df = class_df.apply(remove_lost_cells, axis=1)

    if for_prism:
        return func.pivot_classdf_prism(class_df, percent=True, col_order=['pre', 'early', 'late'])
    else:
        return class_df


def transition_matrix_for_prism(matrix_df: pd.DataFrame, phase, include_lost=False, norm='rows'):

    # Forward (row-normalization)
    dicts = []
    for _, row in matrix_df.iterrows():
        if include_lost:
            if norm in ['rows', 'forward']:
                mat = row[phase] / np.sum(row[phase], axis=1)[..., np.newaxis] * 100
            elif norm in ['cols', 'backward']:
                mat = row[phase] / np.sum(row[phase], axis=0) * 100
            elif norm in ['all']:
                mat = row[phase] / np.sum(row[phase]) * 100
            elif norm in ['none']:
                mat = row[phase]
            else:
                raise NotImplementedError

            ### Include "lost"
            dicts.append(dict(trans='non-coding > non-coding', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[1, 1]))
            dicts.append(dict(trans='non-coding > unstable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[1, 2]))
            dicts.append(dict(trans='non-coding > stable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[1, 3]))
            dicts.append(dict(trans='unstable > non-coding', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[2, 1]))
            dicts.append(dict(trans='unstable > unstable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[2, 2]))
            dicts.append(dict(trans='unstable > stable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[2, 3]))
            dicts.append(dict(trans='stable > non-coding', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[3, 1]))
            dicts.append(dict(trans='stable > unstable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[3, 2]))
            dicts.append(dict(trans='stable > stable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[3, 3]))

        else:
            if norm in ['rows', 'forward']:
                mat = row[phase][1:, 1:] / np.sum(row[phase][1:, 1:], axis=1)[..., np.newaxis] * 100
            elif norm in ['cols', 'backward']:
                mat = row[phase][1:, 1:] / np.sum(row[phase][1:, 1:], axis=0) * 100
            elif norm in ['all']:
                mat = row[phase][1:, 1:] / np.sum(row[phase][1:, 1:]) * 100
            elif norm in ['none']:
                mat = row[phase]
            else:
                raise NotImplementedError

            ### Exclude "lost"
            dicts.append(dict(trans='non-coding > non-coding', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[0, 0]))
            dicts.append(dict(trans='non-coding > unstable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[0, 1]))
            dicts.append(dict(trans='non-coding > stable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[0, 2]))
            dicts.append(dict(trans='unstable > non-coding', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[1, 0]))
            dicts.append(dict(trans='unstable > unstable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[1, 1]))
            dicts.append(dict(trans='unstable > stable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[1, 2]))
            dicts.append(dict(trans='stable > non-coding', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[2, 0]))
            dicts.append(dict(trans='stable > unstable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[2, 1]))
            dicts.append(dict(trans='stable > stable', mouse_id=int(row['mouse_id'].split('_')[0]), perc=mat[2, 2]))

    df = pd.DataFrame(dicts).pivot(index='trans', columns='mouse_id', values='perc')
    if norm in ['rows', 'forward']:
        df = df.reindex(['non-coding > non-coding', 'non-coding > unstable', 'non-coding > stable', 'unstable > non-coding',
                         'unstable > unstable', 'unstable > stable', 'stable > non-coding', 'stable > unstable', 'stable > stable'])
    elif norm in ['cols', 'backward']:
        df = df.reindex(['non-coding > non-coding', 'unstable > non-coding', 'stable > non-coding',
                         'non-coding > unstable', 'unstable > unstable', 'stable > unstable',
                         'non-coding > stable', 'unstable > stable', 'stable > stable'])
    elif norm in ['all', 'none']:
        df = df.reindex(['non-coding > non-coding', 'non-coding > unstable', 'non-coding > stable', 'unstable > non-coding',
                         'unstable > unstable', 'unstable > stable', 'stable > non-coding', 'stable > unstable', 'stable > stable'])
    # df = df.fillna(0)

    return df

#%%
if __name__ == '__main__':
    spatial_maps = dc.load_data('spat_dff_maps')
    is_pc = dc.load_data('is_pc')

    # To produce CSVs for sankey plots (plotting happens in Jupyter Notebook via Plotly)
    stability_classes = classify_stability(is_pc_list=is_pc, spatial_map_list=spatial_maps, for_prism=False, ignore_lost=True, align_days=True)

    # Quantify transitions
    trans_matrices = stability_sankey(df=stability_classes, directory=None)
    trans_matrices_rng = stability_sankey(df=stability_classes, directory=None, shuffle=500, return_shuffle_avg=True)

    # Export transition matrix for prism
    transition_matrix_for_prism(trans_matrices, phase='pre_early', include_lost=False, norm='forward').to_clipboard(header=True, index=True)
    transition_matrix_for_prism(trans_matrices_rng, phase='early_late', include_lost=False, norm='forward').to_clipboard(header=True, index=True)
    transition_matrix_for_prism(trans_matrices_rng, phase='pre_early', include_lost=False, norm='forward').to_clipboard(header=True, index=True)
