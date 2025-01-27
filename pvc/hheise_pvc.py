#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 19/09/2023 17:47
@author: hheise

"""
import warnings
import pandas as pd
import numpy as np
from scipy import signal
import datajoint as dj
import login

login.connect()

from schema import common_mice, common_img, common_match, hheise_placecell, hheise_behav
from util import helper

schema = dj.schema('hheise_pvc', locals(), create_tables=True)


@schema
class PvcCrossSession(dj.Computed):
    definition = """ # Compute PVC matrix across sessions from matched neurons
    -> common_img.Segmentation
    circular    : tinyint       # Bool flag whether the matrix was computed with the corridor wrapping around (full matrix)
    ---
    rel_day     : tinyint       # Day relative to microsphere injection for the reference (primary) session
    target_date : date          # Date of the target session
    rel_tar_day : tinyint       # Day relative to microsphere injection for the target session
    phase       : varchar(64)   # Phase of the comparison (pre, pre_post, early or late)
    n_neurons   : int           # Number of neurons that were matched in both sessions and used in the matrix
    pvc_mat     : longblob      # 2D array of shape (n_shifts, n_bins) containing the PVC across delta-X and corridor positions
    pvc_time = CURRENT_TIMESTAMP    : timestamp
    """

    # Only include mice that are completely available (ignore 112, no matched cells)
    include_mice = [33, 41, 63, 69, 83, 85, 86, 89, 90, 91, 93, 95, 108, 110, 111, 113, 114, 115, 116, 121, 122]
    _key_source = common_img.Segmentation & "username='hheise'" & f"mouse_id in {helper.in_query(include_mice)}"

    def make(self, key):

        # key = (common_img.Segmentation & 'mouse_id=121' & 'day="2022-08-27"').fetch1('KEY')
        # print(key)

        # Find key for a next session that is in 3 days (for prestroke) or the next session (poststroke)
        surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' &
                       key).fetch('surgery_date')[0].date()

        # Get matched spikerate data for the current mouse
        if key['mouse_id'] in [63, 69]:
            query = (common_match.MatchedIndex & 'username="hheise"' &
                     f'mouse_id={key["mouse_id"]}' & 'day<="2021-03-23"')
        else:
            query = (common_match.MatchedIndex & 'username="hheise"' &
                     f'mouse_id={key["mouse_id"]}' & 'day<"2022-09-09"')

        matched_data = query.get_matched_data(table=hheise_placecell.BinnedActivity.ROI, attribute='bin_spikerate',
                                              extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                              return_array=True, relative_dates=True,
                                              surgery='Microsphere injection')
        spat_maps = matched_data[list(matched_data.keys())[0]][0]
        matched_days = np.array(matched_data[list(matched_data.keys())[0]][1])

        # Filter out dates where no matched data exists
        dates = np.unique((common_img.Segmentation() & f'mouse_id = {key["mouse_id"]}' &
                          f'username = "{key["username"]}"').fetch('day'))
        rel_segmentation_days = np.array(pd.Series(dates - surgery_day).dt.days)
        dates = dates[np.isin(rel_segmentation_days, matched_days)]

        last_pre_day = matched_days[np.searchsorted(matched_days, 0, side='right') - 1]  # Find last prestroke day

        rel_day = (key['day'] - surgery_day).days

        # If the current session is the last matched session, or no matched data exists for it, exit function
        if rel_day == matched_days[-1] or rel_day not in matched_days:
            return

        if rel_day < last_pre_day:
            phase = 'pre'
        elif rel_day == last_pre_day:
            phase = 'pre_post'
        elif last_pre_day < rel_day < 6:
            phase = 'early'
        elif 6 <= rel_day:
            phase = 'late'
        else:
            raise ValueError(f'Could not determine phase of session {key}')

        if phase == 'pre':
            # For prestroke sessions, only compare sessions that are 3 days apart
            if rel_day + 3 in matched_days:
                next_day = dates[np.where(matched_days == rel_day + 3)[0][0]]
            else:
                # print(f'No session found 3 days after this one: {key}')
                return
        else:
            # For other sessions, assume that they are 3 days apart and just take the next one
            next_day = dates[np.where(matched_days == rel_day)[0][0] + 1]
        next_rel_day = (next_day - surgery_day).days

        # Only keep two sessions, the current one and the one in 3 days
        matched_spat_map = spat_maps[:, [np.where(matched_days == rel_day)[0][0], np.where(matched_days == next_rel_day)[0][0]]]
        matched_spat_map = np.moveaxis(matched_spat_map, 1, 0)

        # Filter out neurons that are not matched in both sessions
        neuron_mask = np.sum(np.array([~np.isnan(matched_spat_map[0]).any(axis=1),
                                       ~np.isnan(matched_spat_map[1]).any(axis=1)]).astype(int), axis=0) == 2

        # Compute PVC matrix
        mat = self.pvc_matrix(matched_spat_maps=matched_spat_map[:, neuron_mask])
        mat_circ = self.pvc_matrix(matched_spat_maps=matched_spat_map[:, neuron_mask], circular=True)

        # Insert entries into database
        entry = dict(**key, circular=False, rel_day=rel_day, target_date=next_day, rel_tar_day=next_rel_day,
                     phase=phase, n_neurons=np.sum(neuron_mask), pvc_mat=mat)
        self.insert1(entry)

        entry['circular'] = True
        entry['pvc_mat'] = mat_circ
        self.insert1(entry)

    @staticmethod
    def pvc_matrix(matched_spat_maps: np.ndarray, max_delta_bins: int = None, circular: bool = False) -> np.ndarray:
        """
        Calculate the PVC matrix between two sessions.

        Args:
            matched_spat_maps: 3D array with shape (2, n_neurons, n_bins) containing spatial activity maps of matched neurons in session 1 and 2 (axis 0)
            max_delta_bins: Max difference in bin distance, including delta_x = 0 (no shift). Default is entire corridor.
            circular: Bool flag whether to wrap corridor around to produce full PVC matrix.

        Returns:
            2D numpy array with shape (n_shifts, n_bins) containing PVC matrix
        """

        session1 = matched_spat_maps[0]
        session2 = matched_spat_maps[1]

        # Filter out neurons that are not active in both sessions
        neuron_mask = np.sum(np.array([~np.isnan(session1).any(axis=1),
                                       ~np.isnan(session2).any(axis=1)]).astype(int), axis=0) == 2
        session1 = session1[neuron_mask].T
        session2 = session2[neuron_mask].T

        num_bins, num_neurons = session1.shape
        if max_delta_bins is None:
            max_delta_bins = num_bins
        pvc_mat = np.zeros((max_delta_bins, num_bins)) * np.nan

        for delta_bin in range(max_delta_bins):
            pvc_vals = []
            max_offset = num_bins if circular else num_bins - delta_bin

            for offset in range(max_offset):
                idx_x = offset
                idx_y = offset + (delta_bin - num_bins) if circular else offset + delta_bin
                session1_bin = session1[idx_x]
                session2_bin = session2[idx_y]
                pvc_xy_num = np.dot(session1_bin, session2_bin)
                pvc_xy_den_term1 = np.dot(session1_bin, session1_bin)
                pvc_xy_den_term2 = np.dot(session2_bin, session2_bin)
                pvc_xy = pvc_xy_num / (np.sqrt(pvc_xy_den_term1 * pvc_xy_den_term2))
                pvc_vals.append(pvc_xy)
            if circular:
                pvc_mat[delta_bin] = pvc_vals
            else:
                pvc_mat[delta_bin, :len(pvc_vals)] = pvc_vals

        return pvc_mat


@schema
class PvcCrossSessionEval(dj.Computed):
    definition = """ # Evaluation of PVC matrices
    -> PvcCrossSession
    locations                   : varchar(6)    # Set of location bins over which the PVC matrix was averaged. Can be 'all', 'rz', or 'non_rz'
    ---
    pvc_curve                   : longblob      # PVC curve, 1D array of shape (n_shifts,) (averaged PVC matrix across all corridor positions)
    slope                       : longblob      # First derivative of PVC curve, 1D array of shape (n_shifts-1,), measured in delta-PVC per cm
    min_slope                   : float         # Minimum slope in the first quadrant
    max_pvc                     : float         # PVC value at the same position (0 shift)
    min_pvc                     : float         # Minimum PVC value in the first quadrant
    pvc_dif                     : float         # Difference between max_pvc and min_pvc
    pvc_rel_dif                 : float         # Relative difference between max_pvc and min_pvc (min_pvc is X percent lower than max_pvc)
    avg_prominence              : float         # Average of q1 and q2 prominence
    avg_rel_prominence          : float         # Average of q1 and q2 relative prominence
    avg_slope                   : float         # Average of q1 and q2 slopes
    q1_diff = NULL              : float         # Difference between PVC value at position 0 and peak PVC at quadrant 1
    q1_rel_diff = NULL          : float         # Relative difference between PVC value at position 0 and peak PVC at quadrant 1 (Q1 peak is X percent higher than pos-0 peak)
    q1_prominence = NULL        : float         # Difference between min and max PVC around first quadrant repetition peak (bin 21)
    q1_rel_prominence = NULL    : float         # Relative difference between min and max PVC around first quadrant repetition peak (bin 21) (valley is at X percent of total height of peak)
    q1_slope = NULL             : float         # Steepest slope of the first quadrant repetition peak
    q2_diff = NULL              : float         # Difference between PVC value at position 0 and peak PVC at quadrant 2
    q2_rel_diff = NULL          : float         # Relative difference between PVC value at position 0 and peak PVC at quadrant 2
    q2_prominence = NULL        : float         # Difference between min and max PVC around first quadrant repetition peak (bin 42)
    q2_rel_prominence = NULL    : float         # Relative difference between min and max PVC around first quadrant repetition peak (bin 42)
    q2_slope = NULL             : float         # Steepest slope of the second quadrant repetition peak
    eval_time = CURRENT_TIMESTAMP    : timestamp
    """

    def make(self, key):

        # key = PvcCrossSession().fetch('KEY')[0]
        #
        # warnings.filterwarnings("error")
        # print(key)

        # Compute actual bin size from corridor length and number of bins
        bin_size = (hheise_behav.VRSessionInfo & key).fetch1('length') / (hheise_placecell.PCAnalysis & key &
                                                                          'place_cell_id=2').fetch1('n_bins')

        # Fetch PVC matrix
        mat = (PvcCrossSession & key).fetch1('pvc_mat')

        # Get RZ mask
        rz_borders = (hheise_behav.CorridorPattern & 'pattern="training"').rescale_borders(mat.shape[1])
        rz_borders[:, 0] = np.floor(rz_borders[:, 0])
        rz_borders[:, 1] = np.ceil(rz_borders[:, 1])
        rz_mask = np.zeros(80, dtype=bool)
        for b in rz_borders.astype(int):
            rz_mask[b[0]:b[1]] = True

        # Quantify curves with all, only RZ or only non-RZ locations
        for location in ('all', 'rz', 'non_rz'):

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if location == 'all':
                    pvc_curve = np.nanmean(mat, axis=1)
                elif location == 'rz':
                    pvc_curve = np.nanmean(mat[:, rz_mask], axis=1)
                else:
                    pvc_curve = np.nanmean(mat[:, ~rz_mask], axis=1)

            quant = self.quantify_curve(curve=pvc_curve, bin_s=bin_size)

            self.insert1(dict(**key, locations=location, pvc_curve=pvc_curve, **quant))

    @staticmethod
    def quantify_curve(curve: np.ndarray, bin_s: float, q_bin: int = 21) -> dict:
        """
        Quantify a PVC curve with various metrics.

        Args:
            curve: 1D array with shape (n_shifts,), averaged PVC matrix across all corridor positions)
            bin_s: Bin size in cm. Used to compute slope in delta-PVC per cm.
            q_bin: Bins per quadrant. Hard-coded for 80-bin training corridor.

        Returns:
            Dict with metrics.
        """

        def find_highest_peak(curve_data: np.ndarray, lower_idx: int) -> int:
            """ Return global index of the highest peak in the data trace. Maxima at the borders are not condsidered peaks. """
            # Give dummy minimum height to get height of peaks returned
            peaks = signal.find_peaks(curve_data, height=0)
            if len(peaks[0]) > 0:
                return peaks[0][np.argmax(peaks[1]['peak_heights'])] + lower_idx
            else:  # Data did not include any peak
                return np.nan

        def quantify_peak(curve_data: np.ndarray, slope_data: np.ndarray, peak_idx: int, max_peak: float):
            """ Quantify a single peak. If peak_idx is nan, all metrics are nan. """
            if np.isnan(peak_idx):
                peak = np.nan
                peak_diff = np.nan
                peak_rel_diff = np.nan
                prominence = np.nan
                rel_prominence = np.nan
                peak_slope = np.nan
            else:
                peak = curve_data[peak_idx]
                peak_diff = peak - max_peak
                peak_rel_diff = (peak - max_peak) / max_peak
                peak_prom = signal.peak_prominences(curve_data, [peak_idx], wlen=q_bin * 2)
                prominence = peak_prom[0][0]
                rel_prominence = peak_prom[0][0] / peak
                left_slope = np.median(slope_data[peak_prom[1][0]:peak_idx])
                right_slope = np.median(slope_data[peak_idx:peak_prom[2][0]])
                peak_slope = left_slope if np.abs(left_slope) > np.abs(right_slope) else right_slope

            return peak, peak_diff, peak_rel_diff, prominence, rel_prominence, peak_slope

        upper_bin = int(np.floor(q_bin / 2))
        lower_bin = int(np.ceil(q_bin / 2))

        # Compute main peak differences
        max_pvc = curve[0]
        min_pvc = np.min(curve[:q_bin])
        pvc_dif = min_pvc - max_pvc
        pvc_rel_dif = pvc_dif / max_pvc

        # Compute slopes
        slope = np.diff(curve) / bin_s
        min_slope = np.min(slope[:q_bin])  # The initial slope is taken as the raw minimum

        # Find peaks of quadrants
        # curve_smooth = gaussian_filter1d(curve, 1)        # Don't use smoothed curve, it can shift peak locations
        q1_peak_idx = find_highest_peak(curve[q_bin - lower_bin: q_bin + upper_bin], q_bin - lower_bin)
        q2_peak_idx = find_highest_peak(curve[q_bin * 2 - lower_bin: q_bin * 2 + upper_bin], q_bin * 2 - lower_bin)

        q1_peak, q1_diff, q1_rel_diff, q1_prom, q1_rel_prom, q1_slope = quantify_peak(curve, slope, q1_peak_idx,
                                                                                      max_pvc)
        q2_peak, q2_diff, q2_rel_diff, q2_prom, q2_rel_prom, q2_slope = quantify_peak(curve, slope, q2_peak_idx,
                                                                                      max_pvc)

        return dict(slope=slope, min_slope=min_slope, max_pvc=max_pvc, min_pvc=min_pvc, pvc_dif=pvc_dif,
                    pvc_rel_dif=pvc_rel_dif,
                    avg_prominence=np.nanmean([q1_prom, q2_prom]),
                    avg_rel_prominence=np.nanmean([q1_rel_prom, q2_rel_prom]),
                    avg_slope=np.nanmean([np.abs(q1_slope), np.abs(q2_slope)]),
                    q1_diff=q1_diff, q1_rel_diff=q1_rel_diff, q1_prominence=q1_prom, q1_rel_prominence=q1_rel_prom,
                    q1_slope=q1_slope,
                    q2_diff=q2_diff, q2_rel_diff=q2_rel_diff, q2_prominence=q2_prom, q2_rel_prominence=q2_rel_prom,
                    q2_slope=q2_slope)


@schema
class PvcPrestroke(dj.Computed):
    definition = """ # Compute PVC matrix across prestroke sessions from matched neurons
    -> common_img.Segmentation
    circular    : tinyint       # Bool flag whether the matrix was computed with the corridor wrapping around (full matrix)
    ---
    rel_day     : tinyint       # Day relative to microsphere injection for the reference (primary) session
    target_date : date          # Date of the target session
    rel_tar_day : tinyint       # Day relative to microsphere injection for the target session
    phase       : varchar(64)   # Phase of the comparison (pre, pre_post, early or late)
    n_neurons   : int           # Number of neurons that were matched in both sessions and used in the matrix
    pvc_mat     : longblob      # 2D array of shape (n_shifts, n_bins) containing the PVC across delta-X and corridor positions
    pvc_time = CURRENT_TIMESTAMP    : timestamp
    """

    # Only include mice that are completely available (ignore 112, no matched cells)
    include_mice = [33, 41, 63, 69, 83, 85, 86, 89, 90, 91, 93, 95, 108, 110, 111, 113, 114, 115, 116, 121, 122]
    _key_source = common_img.Segmentation & "username='hheise'" & f"mouse_id in {helper.in_query(include_mice)}"

    def make(self, key):

        # key = (common_img.Segmentation & 'mouse_id=121' & 'day="2022-08-12"').fetch1('KEY')
        # print(key)

        # Find key for a next session that is in 3 days (for prestroke) or the next session (poststroke)
        surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' &
                       key).fetch('surgery_date')[0].date()

        # Exit function immediately if the session is poststroke to save time
        if key['day'] > surgery_day:
            return

        # Get matched spikerate data for the current mouse
        if key['mouse_id'] in [63, 69]:
            query = (common_match.MatchedIndex & 'username="hheise"' &
                     f'mouse_id={key["mouse_id"]}' & 'day<="2021-03-23"')
        else:
            query = (common_match.MatchedIndex & 'username="hheise"' &
                     f'mouse_id={key["mouse_id"]}' & 'day<"2022-09-09"')

        matched_data = query.get_matched_data(table=hheise_placecell.BinnedActivity.ROI, attribute='bin_spikerate',
                                              extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                              return_array=True, relative_dates=True,
                                              surgery='Microsphere injection')
        spat_maps = matched_data[list(matched_data.keys())[0]][0]
        matched_days = np.array(matched_data[list(matched_data.keys())[0]][1])

        # Filter out dates where no matched data exists
        dates = np.unique((common_img.Segmentation() & f'mouse_id = {key["mouse_id"]}' &
                          f'username = "{key["username"]}"').fetch('day'))
        rel_segmentation_days = np.array(pd.Series(dates - surgery_day).dt.days)
        dates = dates[np.isin(rel_segmentation_days, matched_days)]

        last_pre_day = matched_days[np.searchsorted(matched_days, 0, side='right') - 1]  # Find last prestroke day

        rel_day = (key['day'] - surgery_day).days

        # If the current session the last prestroke or a poststroke session, skip it
        if (rel_day not in matched_days) or (rel_day >= last_pre_day):
            return

        if rel_day < last_pre_day:
            phase = 'pre'
        elif rel_day == last_pre_day:
            phase = 'pre_post'
        elif last_pre_day < rel_day < 6:
            phase = 'early'
        elif 6 <= rel_day:
            phase = 'late'
        else:
            raise ValueError(f'Could not determine phase of session {key}')

        # Use the next session as target (for this table we dont care about the 3-day period to get more comparisons)
        next_day = dates[np.where(matched_days == rel_day)[0][0] + 1]
        next_rel_day = (next_day - surgery_day).days

        # Only keep two sessions, the current one and the one in 3 days
        matched_spat_map = spat_maps[:, [np.where(matched_days == rel_day)[0][0], np.where(matched_days == next_rel_day)[0][0]]]
        matched_spat_map = np.moveaxis(matched_spat_map, 1, 0)

        # Filter out neurons that are not matched in both sessions
        neuron_mask = np.sum(np.array([~np.isnan(matched_spat_map[0]).any(axis=1),
                                       ~np.isnan(matched_spat_map[1]).any(axis=1)]).astype(int), axis=0) == 2

        # Compute PVC matrix
        mat = PvcCrossSession.pvc_matrix(matched_spat_maps=matched_spat_map[:, neuron_mask])
        mat_circ = PvcCrossSession.pvc_matrix(matched_spat_maps=matched_spat_map[:, neuron_mask], circular=True)

        # Insert entries into database
        entry = dict(**key, circular=False, rel_day=rel_day, target_date=next_day, rel_tar_day=next_rel_day,
                     phase=phase, n_neurons=np.sum(neuron_mask), pvc_mat=mat)
        self.insert1(entry)

        entry['circular'] = True
        entry['pvc_mat'] = mat_circ
        self.insert1(entry)


@schema
class PvcPrestrokeEval(dj.Computed):
    definition = """ # Evaluation of PVC matrices
    -> PvcPrestroke
    locations                   : varchar(6)    # Set of location bins over which the PVC matrix was averaged. Can be 'all', 'rz', or 'non_rz'
    ---
    pvc_curve                   : longblob      # PVC curve, 1D array of shape (n_shifts,) (averaged PVC matrix across all corridor positions)
    slope                       : longblob      # First derivative of PVC curve, 1D array of shape (n_shifts-1,), measured in delta-PVC per cm
    min_slope                   : float         # Minimum slope in the first quadrant
    max_pvc                     : float         # PVC value at the same position (0 shift)
    min_pvc                     : float         # Minimum PVC value in the first quadrant
    pvc_dif                     : float         # Difference between max_pvc and min_pvc
    pvc_rel_dif                 : float         # Relative difference between max_pvc and min_pvc (min_pvc is X percent lower than max_pvc)
    avg_prominence              : float         # Average of q1 and q2 prominence
    avg_rel_prominence          : float         # Average of q1 and q2 relative prominence
    avg_slope                   : float         # Average of q1 and q2 slopes
    q1_diff = NULL              : float         # Difference between PVC value at position 0 and peak PVC at quadrant 1
    q1_rel_diff = NULL          : float         # Relative difference between PVC value at position 0 and peak PVC at quadrant 1 (Q1 peak is X percent higher than pos-0 peak)
    q1_prominence = NULL        : float         # Difference between min and max PVC around first quadrant repetition peak (bin 21)
    q1_rel_prominence = NULL    : float         # Relative difference between min and max PVC around first quadrant repetition peak (bin 21) (valley is at X percent of total height of peak)
    q1_slope = NULL             : float         # Steepest slope of the first quadrant repetition peak
    q2_diff = NULL              : float         # Difference between PVC value at position 0 and peak PVC at quadrant 2
    q2_rel_diff = NULL          : float         # Relative difference between PVC value at position 0 and peak PVC at quadrant 2
    q2_prominence = NULL        : float         # Difference between min and max PVC around first quadrant repetition peak (bin 42)
    q2_rel_prominence = NULL    : float         # Relative difference between min and max PVC around first quadrant repetition peak (bin 42)
    q2_slope = NULL             : float         # Steepest slope of the second quadrant repetition peak
    eval_time = CURRENT_TIMESTAMP    : timestamp
    """

    def make(self, key):

        # key = PvcCrossSession().fetch('KEY')[0]
        #
        # warnings.filterwarnings("error")
        # print(key)

        # Compute actual bin size from corridor length and number of bins
        bin_size = (hheise_behav.VRSessionInfo & key).fetch1('length') / (hheise_placecell.PCAnalysis & key &
                                                                          'place_cell_id=2').fetch1('n_bins')

        # Fetch PVC matrix
        mat = (PvcPrestroke & key).fetch1('pvc_mat')

        # Get RZ mask
        rz_borders = (hheise_behav.CorridorPattern & 'pattern="training"').rescale_borders(mat.shape[1])
        rz_borders[:, 0] = np.floor(rz_borders[:, 0])
        rz_borders[:, 1] = np.ceil(rz_borders[:, 1])
        rz_mask = np.zeros(80, dtype=bool)
        for b in rz_borders.astype(int):
            rz_mask[b[0]:b[1]] = True

        # Quantify curves with all, only RZ or only non-RZ locations
        for location in ('all', 'rz', 'non_rz'):

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if location == 'all':
                    pvc_curve = np.nanmean(mat, axis=1)
                elif location == 'rz':
                    pvc_curve = np.nanmean(mat[:, rz_mask], axis=1)
                else:
                    pvc_curve = np.nanmean(mat[:, ~rz_mask], axis=1)

            quant = PvcCrossSessionEval.quantify_curve(curve=pvc_curve, bin_s=bin_size)

            self.insert1(dict(**key, locations=location, pvc_curve=pvc_curve, **quant))
