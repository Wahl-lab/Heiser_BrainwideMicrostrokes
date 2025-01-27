#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 28/08/2023 11:14
@author: hheise

"""
import warnings
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import confusion_matrix
from typing import Tuple
import datajoint as dj
import login

login.connect()

from schema import common_mice, common_img, common_match, hheise_behav, hheise_placecell
from util import helper

schema = dj.schema('hheise_decoder', locals(), create_tables=True)


@schema
class BayesianParameter(dj.Manual):
    definition = """ # Parameters for the Bayesian decoder.
    bayesian_id     : tinyint
    ---
    window_halfsize : float                 # Half-size of decoding window (for raw activity trace) in seconds
    n_cells         : int                   # How many cells (sorted by 'neuron_subset') to include in the model
    data_type       : enum('dff', 'decon')  # Whether to use deconvolved or dF/F fluorescence traces.
    neuron_subset   : varchar(256)          # Name of a specific subset of neurons that are used for the decoder. Handling of subset string has to be implemented in the decoder function.
    include_resting : tinyint               # Bool flag whether to include resting frames with speed < 5 cm/s in the testing dataset
    ignore_zeros    : tinyint               # Bool flag whether to ignore predictions of bin 0 (NaN prediction) when computing prediction errors
    description     : varchar(256)          # Description of the parameter set
    """


@schema
class BayesianDecoderWithinSession(dj.Computed):
    definition = """ # Predict position bin from neural activity with training data from the same session. Error metrics are averages of single-trial leave-one-out cross-validation.
    -> BayesianParameter
    -> common_img.Segmentation
    ---
    failed_prediction       : float     # Fraction of frames where prediction failed completely (bin 0 falsely predicted)
    accuracy                : float     # Fraction of frames where the bin was correctly predicted 
    abs_error               : float     # Absolute (cumulative) error in cm (per trial)
    mae                     : float     # Mean absolute error in cm (per frame)
    mae_std                 : float     # Standard deviation of the absolute error (SD-AE) (per frame)
    mse                     : float     # Mean squared error in cm2 (variance of the estimator)
    mse_std                 : float     # Standard deviation of the mean squared error
    rmse                    : float     # Root mean squared error in cm (standard deviation of the estimator)
    mae_shift               : float     # Time shift of the predicted position that minimized the MAE (in seconds). Negative shift means that prediction was delayed relative to true position.
    mae_min                 : float     # MAE with predicted positions shifted by 'mae_shift' seconds
    acc_shift               : float     # Time shift of the predicted position that maximized the accuracy (in seconds). Negative shift means that prediction was delayed relative to true position.
    acc_max                 : float     # Accuracy with predicted positions shifted by 'acc_shift' seconds
    mae_in_rz               : float     # MAE when considering only frames where the mouse was inside a reward zone
    mae_in_rz_std           : float     # SD-AE when considering only frames where the mouse was inside a reward zone
    mae_out_rz              : float     # MAE when considering only frames where the mouse was outside a reward zone
    mae_out_rz_std          : float     # SD-AE when considering only frames where the mouse was outside a reward zone
    
    # Same errors, but computed when splitting the corridor into quadrants, taking into account the periodicity of reward zones
    accuracy_quad           : float     # Quadrant-wise accuracy
    abs_error_quad          : float     # Absolute (cumulative) quadrant-wise error in cm (per trial)
    mae_quad                : float     # Quadrant-wise MAE
    mae_quad_std            : float     # Standard deviation of the quadrant-wise MAE
    mse_quad                : float     # Quadrant-wise MSE
    mse_quad_std            : float     # Standard deviation of the quadrant-wise MSE
    rmse                    : float     # Root mean squared error in cm (standard deviation of the estimator)
    rmse_quad               : float     # Quadrant-wise RMSE
    mae_shift_quad          : float     # Time shift of the predicted position that minimized the quadrant-wise MAE (in seconds)
    mae_min_quad            : float     # Quadrant-wise MAE with predicted positions shifted by 'mae_shift' seconds
    acc_shift_quad          : float     # Time shift of the predicted position that maximized the quadrant-wise accuracy (in seconds)
    acc_max_quad            : float     # Quadrant-wise accuracy with predicted positions shifted by 'acc_shift' seconds
    mae_in_rz_quad          : float     # Quadrant-wise MAE when using only frames where the mouse was inside a RZ
    mae_in_rz_quad_std      : float     # Quadrant-wise SD-AE when using only frames where the mouse was inside a RZ
    mae_out_rz_quad         : float     # Quadrant-wise MAE when using only frames where the mouse was outside a RZ
    mae_out_rz_quad_std     : float     # Quadrant-wise SD-AE when using only frames where the mouse was outside a RZ
    
    # Binary confusion matrix metrics for classifying inside/outside RZ frames
    confusion_matrix = NULL : longblob  # 2x2 confusion matrix (rows: true classes, columns: predicted classes)
    accuracy_rz = NULL      : float     # Probability to correctly identify a frame as inside or outside RZ
    precision_rz = NULL     : float     # When the model predicts a RZ, how likely is it actually a RZ? NaN if the model never predicted a RZ.
    sensitivity_rz = NULL   : float     # How often does the model correctly detect a RZ?
    specificity_rz = NULL   : float     # How often does the model correctly detect a non-RZ?
    mcc = NULL              : float     # Matthews correlation coefficient. Most reliable single metric for a confusion matrix, accounting for imbalances in class sizes.

    run_time = CURRENT_TIMESTAMP    : timestamp
    """

    class Trial(dj.Part):
        definition = """ # Predicted position for each trial. Training set are all other trials of that session (leave-one-out).
        -> BayesianDecoderWithinSession
        trial_id                    : tinyint   # Trial count per session, same as hheise_behav.VRSession's 'trial_id', base 0
        -----
        pos_predict                 : longblob  # Predicted position bin per frame 
        pos_true                    : longblob  # True VR position bin per frame
        confidence                  : longblob  # Confidence distribution across all bins per frame (log-likelihood of bin) (n_frames, n_bins)
        unique_pred_bins            : tinyint   # Number of unique bins that were predicted. A number < 80 might hint at problems with the prediction.
        failed_prediction           : float     # Fraction of frames where prediction failed completely (bin 0 falsely predicted)
        accuracy = NULL             : float     # Fraction of frames where the bin was correctly predicted 
        abs_error = NULL            : float     # Absolute (cumulative) error in cm (per trial)
        mae = NULL                  : float     # Mean absolute error in cm (per frame)
        mae_std = NULL              : float     # Standard deviation of the absolute error (SD-AE) (per frame)
        mse = NULL                  : float     # Mean squared error in cm2 (variance of the estimator)
        mse_std = NULL              : float     # Standard deviation of the mean squared error
        rmse = NULL                 : float     # Root mean squared error in cm (standard deviation of the estimator)
        mae_shift = NULL            : float     # Time shift of the predicted position that minimized the MAE (in seconds). Negative shift means that prediction was delayed relative to true position.
        mae_min = NULL              : float     # MAE with predicted positions shifted by 'mae_shift' seconds
        acc_shift = NULL            : float     # Time shift of the predicted position that maximized the accuracy (in seconds). Negative shift means that prediction was delayed relative to true position.
        acc_max = NULL              : float     # Accuracy with predicted positions shifted by 'acc_shift' seconds
        mae_in_rz = NULL            : float     # MAE when considering only frames where the mouse was inside a reward zone
        mae_in_rz_std = NULL        : float     # SD-AE when considering only frames where the mouse was inside a reward zone
        mae_out_rz = NULL           : float     # MAE when considering only frames where the mouse was outside a reward zone
        mae_out_rz_std = NULL       : float     # SD-AE when considering only frames where the mouse was outside a reward zone
        
        # Same errors, but computed when splitting the corridor into quadrants, taking into account the periodicity of reward zones
        accuracy_quad = NULL        : float     # Quadrant-wise accuracy
        abs_error_quad = NULL       : float     # Absolute (cumulative) quadrant-wise error in cm (per trial)
        mae_quad = NULL             : float     # Quadrant-wise MAE
        mae_quad_std = NULL         : float     # Standard deviation of the quadrant-wise MAE
        mse_quad = NULL             : float     # Quadrant-wise MSE
        mse_quad_std = NULL         : float     # Standard deviation of the quadrant-wise MSE
        rmse = NULL                 : float     # Root mean squared error in cm (standard deviation of the estimator)
        rmse_quad = NULL            : float     # Quadrant-wise RMSE
        mae_shift_quad = NULL       : float     # Time shift of the predicted position that minimized the quadrant-wise MAE (in seconds)
        mae_min_quad = NULL         : float     # Quadrant-wise MAE with predicted positions shifted by 'mae_shift' seconds
        acc_shift_quad = NULL       : float     # Time shift of the predicted position that maximized the quadrant-wise accuracy (in seconds)
        acc_max_quad = NULL         : float     # Quadrant-wise accuracy with predicted positions shifted by 'acc_shift' seconds
        mae_in_rz_quad = NULL       : float     # Quadrant-wise MAE when using only frames where the mouse was inside a RZ
        mae_in_rz_quad_std = NULL   : float     # Quadrant-wise SD-AE when using only frames where the mouse was inside a RZ
        mae_out_rz_quad = NULL      : float     # Quadrant-wise MAE when using only frames where the mouse was outside a RZ
        mae_out_rz_quad_std = NULL  : float     # Quadrant-wise SD-AE when using only frames where the mouse was outside a RZ
        
        # Binary confusion matrix metrics for classifying inside/outside RZ frames
        confusion_matrix = NULL     : longblob  # 2x2 confusion matrix (rows: true classes, columns: predicted classes)
        accuracy_rz = NULL          : float     # Probability to correctly identify a frame as inside or outside RZ
        precision_rz = NULL         : float     # When the model predicts a RZ, how likely is it actually a RZ? NaN if the model never predicted a RZ.
        sensitivity_rz = NULL       : float     # How often does the model correctly detect a RZ?
        specificity_rz = NULL       : float     # How often does the model correctly detect a non-RZ?
        mcc = NULL                  : float     # Matthews correlation coefficient. Most reliable single metric for a confusion matrix, accounting for imbalances in class sizes.
        """
    # Only include mice that are completely available
    include_mice = [33, 41, 63, 69, 83, 85, 86, 89, 90, 91, 93, 95, 108, 110, 111, 112, 113, 114, 115, 116, 121, 122]
    _key_source = (BayesianParameter() * common_img.Segmentation) & "username='hheise'" & f"mouse_id in {helper.in_query(include_mice)}"

    def make(self, key):

        # print(key)

        # Fetch data
        params = (BayesianParameter & key).fetch1()
        fr = (common_img.ScanInfo & key).fetch1('fr')
        half_win_frames = int(np.round(fr * params['window_halfsize']))

        # Hard-code place_cell_id=2, most commonly used
        key['place_cell_id'] = 2

        bin_size = (hheise_placecell.PlaceCellParameter & key).fetch1('bin_length')

        if params['data_type'] == 'dff':
            act = (common_img.Segmentation & key).get_traces('dff')
            bin_act = np.stack((hheise_placecell.BinnedActivity.ROI & key).fetch('bin_activity'))
        elif params['data_type'] == 'decon':
            act = (common_img.Segmentation & key).get_traces('decon', decon_id=1)
            bin_act = np.stack((hheise_placecell.BinnedActivity.ROI & key).fetch('bin_spikerate'))
        else:
            raise NotImplementedError(f'Data type "{params["data_type"]}" not implemented.')
        trial_mask = (hheise_placecell.PCAnalysis & key).fetch1('trial_mask')
        pos = (hheise_behav.VRSession & key).get_array(attr='pos', get_frame_avg=True,
                                                       as_dataframe=True)['pos'].to_numpy()

        # Transform position from VR units to bin indices
        bin_borders = np.linspace(-10, 110, bin_act.shape[1])
        pos = np.digitize(pos, bin_borders) - 1

        # Only include normal trials
        norm_trials = (hheise_behav.VRSession & key).get_normal_trials()
        norm_trial_mask = np.isin(trial_mask, norm_trials)
        trial_mask = trial_mask[norm_trial_mask]
        act = act[:, norm_trial_mask]
        bin_act = bin_act[:, :, norm_trials]
        running_mask, aligned_frames = (hheise_placecell.Synchronization.VRTrial &
                                        key & f'trial_id in {helper.in_query(norm_trials)}').fetch('running_mask',
                                                                                                   'aligned_frames')
        aligned_frames = np.stack(aligned_frames)

        # Only keep useful neurons (e.g. place cells)
        mask_ids = (common_img.Segmentation.ROI & key & 'accepted=1').fetch('mask_id')
        if params['neuron_subset'] == 'all_place_cells':
            place_cell_ids = (hheise_placecell.PlaceCell.ROI & key & 'is_place_cell=1').fetch('mask_id')
            cell_mask = np.isin(mask_ids, place_cell_ids)
            cell_sort = []
        elif params['neuron_subset'] == 'place_cells':
            df = pd.DataFrame((hheise_placecell.PlaceCell.ROI & key & 'is_place_cell=1').fetch('mask_id', 'p', as_dict=True))
            if len(df) > 0:
                cell_sort = np.array(df.sort_values(by='p')['mask_id'])
            else:
                cell_sort = []
        elif params['neuron_subset'] == 'si':
            df = pd.DataFrame((hheise_placecell.SpatialInformation.ROI & key).fetch('mask_id', 'si', as_dict=True))
            cell_sort = np.array(df.sort_values(by='si', ascending=False)['mask_id'])
        elif params['neuron_subset'] == 'stability':
            df = pd.DataFrame((hheise_placecell.SpatialInformation.ROI & key).fetch('mask_id', 'stability', as_dict=True))
            cell_sort = np.array(df.sort_values(by='stability', ascending=False)['mask_id'])
        elif params['neuron_subset'] == 'firing_rate':
            df = pd.DataFrame((common_img.ActivityStatistics.ROI & key).fetch('mask_id', 'rate_spikes', as_dict=True))
            cell_sort = np.array(df.sort_values(by='rate_spikes', ascending=False)['mask_id'])
        else:
            raise NotImplementedError(f'Neuronal subset of "{params["neuron_subset"]}" not implemented yet.')

        # After all fetching is complete, delete the place_cell_id key again
        del key['place_cell_id']

        if params['neuron_subset'] == 'all_place_cells':
            pass
        elif params['n_cells'] > len(cell_sort):
            cell_mask = np.ones(len(mask_ids)).astype(bool)
            print(f'Only {len(cell_sort)} (with n_cells = {params["n_cells"]}) found in {key}.')
        else:
            cell_mask = np.isin(mask_ids, cell_sort[:params['n_cells']])

        if np.sum(cell_mask) < 10:
            print(f'Less than 10 cells included in subset "{params["neuron_subset"]}" on session {key}. Skipped.')
            return

        act_cells = act[cell_mask]
        bin_act_cells = bin_act[cell_mask]

        # Only keep running frames (speed > 5 cm/s)
        running_mask = np.concatenate(running_mask)
        if params['include_resting']:
            act_cells_run = act_cells
            pos_run = pos
            trial_mask_run = trial_mask
        else:
            act_cells_run = act_cells[:, running_mask]
            pos_run = pos[running_mask]
            trial_mask_run = trial_mask[running_mask]

        # Perform leave-one-out cross-correlation: Run the model n_trials time, each time with one trial being the test
        # and the rest being the training set.
        trial_entries = []
        for test_trial_id in norm_trials:
            # print(test_trial_id)
            # Split data into training and testing sets
            training_trial_mask = norm_trials != test_trial_id
            test_trial_frame_mask = trial_mask_run == test_trial_id
            act_test = act_cells_run[:, test_trial_frame_mask]
            pos_test = pos_run[test_trial_frame_mask]

            # For each neuron, smooth binned trace and compute mean and std for each bin across training trials
            bin_cells_mean = np.mean(gaussian_filter1d(bin_act_cells[:, :, training_trial_mask], 1, axis=1), axis=2)
            bin_cells_sd = np.std(gaussian_filter1d(bin_act_cells[:, :, training_trial_mask], 1, axis=1), axis=2)

            # Adjust SD of low-SD bins. This is somehow necessary for the decoder to work (without it all probabilities are 0)
            bin_cells_sd_adjust = bin_cells_sd.copy()
            bin_cells_mean_adjust = bin_cells_mean.copy()
            for i in range(bin_cells_sd.shape[0]):
                sd_ratio = np.mean(bin_cells_sd[i]) / bin_cells_sd[i]
                low_sd = sd_ratio > 1
                bin_cells_sd_adjust[i, low_sd] = bin_cells_sd[i, low_sd] * sd_ratio[low_sd]
                # bin_cells_mean_adjust[i, low_sd] = bin_cells_mean[i, low_sd] * sd_ratio[low_sd]

            # Calculate occupancy probability of spatial bins in training trials
            occ = np.sum(aligned_frames[training_trial_mask], axis=0) / np.sum(aligned_frames[training_trial_mask])

            # Run the decoder with the mean/sd activity of training trials on the activity/position of the testing trial
            pos_predict, pos_confidence = self.estimate_position(y=act_test, x_mean=bin_cells_mean_adjust,
                                                                 x_sd=bin_cells_sd_adjust, occ=occ, win=half_win_frames)

            if np.all(pos_predict[~np.isnan(pos_predict)] == 0):
                print(f'Prediction failed: Session {key}, Trial {test_trial_id}')
                errors = self.quantify_error(y_pred=pos_predict, y_true=pos_test, bin_size=bin_size,
                                             ignore_zeros=bool(params['ignore_zeros']), failed=True)
            else:
                # Compute error metrics for the current test trial
                with np.errstate(divide='ignore'):
                    errors = self.quantify_error(y_pred=pos_predict, y_true=pos_test, bin_size=bin_size,
                                                 ignore_zeros=bool(params['ignore_zeros']), framerate=fr)

            trial_entries.append(dict(**key, trial_id=test_trial_id, pos_predict=pos_predict, confidence=pos_confidence,
                                      pos_true=pos_test, unique_pred_bins=len(np.unique(pos_predict[~np.isnan(pos_predict)])),
                                      **errors))

        # Compute average errors across trials
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_errors = {key: np.nanmean([d[key] for d in trial_entries]) for key in errors.keys() if key != 'confusion_matrix'}
        if avg_errors['failed_prediction'] == 1:
            print(f'Prediction failed for all trials. Session skipped.')
            return
        avg_errors['confusion_matrix'] = np.nansum(np.stack([d['confusion_matrix'] for d in trial_entries if not np.all(np.isnan(d['confusion_matrix']))]), axis=0)

        # Insert entries
        self.insert1(dict(**key, **avg_errors))
        self.Trial().insert(trial_entries)

    @staticmethod
    def estimate_position(y: np.ndarray, x_mean: np.ndarray, x_sd: np.ndarray, occ: np.ndarray,
                          win: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate position of the mouse from testing dataset y using bin means, SD and occupancy from the training dataset.
        Args:
            y: 2D array with shape (n_cells, n_frames) of raw neural activity traces from testing set.
            x_mean: Mean activity of spatial bins from training set (shape (80,)).
            x_sd: Standard deviation of activity of spatial bins from training set (shape (80,)).
            occ: Occupancy probability of spatial bins from training set (shape (80,)).
            win: Sliding window half-size in frames.

        Returns:
            1D array of shape (n_frames,) with estimated bin, and
            1D array of same shape with prediction confidence.
        """

        """
        For each time point t in the test data set
        - For each neuron (i)
            - Calculate mean activity S(i) in T = [t-0.125, t+0.125]

            - For each position bin (x)
                - Compute Pi(x, t) see formula in Shuman

            --> we get an array Pi of x bins
            - Normalize Pi by diving it by its maximum value

        --> we obtain matrix (neurons, bins) of all the Pi
        - multiply matrix across neuron dimension
        - multiply vector of probabilities by occupancy
        - decoded position is the most likely position -> take argmax over bins
        """

        pos_estimate = np.zeros(y.shape[1]) * np.nan
        pos_confidence = np.zeros((y.shape[1], x_mean.shape[1])) * np.nan

        # neuron_zero_prob = np.zeros(y.shape) * np.nan

        for t in range(win, y.shape[1] - win):

            # print(t)

            curr_win = np.arange(t - win, t + win + 1)  # Get indices of time t +/- "win" frame window

            # Get mean activity in the current window
            win_mean = np.mean(y[:, curr_win], axis=1)

            # Get probability to be in each bin for each neuron
            # -> Formula adjusted from Shuman2020 to work in log space to handle very small probability numbers
            offset = -0.5 * np.log(2 * np.pi * x_sd ** 2)
            exponent = -((win_mean[:, None] - x_mean) ** 2) / (2 * x_sd ** 2)
            bin_prob = offset + exponent

            # Normalize probability (bin with highest probability is always == 1)
            with np.errstate(divide='ignore', invalid='ignore'):
                bin_prob_norm = bin_prob - np.max(bin_prob, axis=1)[:, None]

            # Multiply across neurons
            bin_prob_neur = np.nansum(bin_prob_norm, axis=0)

            # Multiply by occupancy probability
            bin_prob_occ = bin_prob_neur + np.log(occ)

            # Predicted position bin is location of highest probability (most likely bin)
            pos_estimate[t] = np.argmax(bin_prob_occ)

            # Store likelihood distribution across all bins for the current time window
            pos_confidence[t] = bin_prob_occ

        return pos_estimate, pos_confidence

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # fig, ax = plt.subplots(2, 3)
    # fig.suptitle('M41_2020-08-30_bayesian_14')
    #
    # sns.heatmap(x_mean, ax=ax[0, 0])
    # ax[0, 0].set_title('Average spikerate per bin')
    # sns.heatmap(x_sd, ax=ax[0, 1])
    # ax[0, 1].set_title('Standard deviation of binned spikerate')
    # sns.heatmap(y, ax=ax[0, 2])
    # ax[0, 2].set_title('Deconvolved activity across frames')
    #
    # ax[1, 0].plot(pos_test, label='True position')
    # ax[1, 0].plot(pos_estimate, label='Predicted position')
    # ax[1, 0].legend()
    #
    # out = ax[1, 1].plot(pos_confidence.T)
    # ax[1, 1].set_title('Probability distributions across bins')


    @staticmethod
    def quantify_error(y_pred: np.ndarray, y_true: np.ndarray, bin_size: float = 5.0, ignore_zeros: bool = False,
                       framerate: float = 30.0, corridor_context: str = 'training', failed: bool = False) -> dict:
        """
        Quantify prediction error of a given binned position estimation with many different error metrics.

        Args:
            y_pred: 1D numpy array with shape (n_frames,) with predicted positions per frame
            y_true: 1D array with same shape with true positions per frame
            bin_size: Size of each position bin in cm
            ignore_zeros: Bool flag whether to ignore predictions of bin 0 (NaN prediction)
            framerate: Frame rate of the recording in Hz, to translate between frames and time
            corridor_context: Context of the session to get reward zone borders.
            failed: If the prediction for a trial failed completely, all errors can be nan

        Returns:
            Dict with error metrics
        """

        def shift_array(xs, n):
            """ Shift a 1D array 'xs' by 'n' elements. End elements are shifted out and buffered with np.nan. """
            e = np.empty_like(xs)
            if n == 0:
                e = xs
            elif n > 0:
                e[:n] = np.nan
                e[n:] = xs[:-n]
            else:
                e[n:] = np.nan
                e[:n] = xs[-n:]
            return e

        def circularize_quadrants(positions, quadrant_size=64/3):
            """
            Transform an array of position bins into a version where each quadrant is circularized, reflecting
            the periodicity of the corridor.

            Args:
                positions: 1D numpy array with shape (n_frames) in standard corridor coordinates. Default for training corridor.
                quadrant_size:  Size of each quadrant in standard corridor coordinates

            Returns:
                1D array with same shape as 'positions', transformed into circular quadrant coordinates
            """

            # Rescale positions to a single quadrant and take cosine to map it to a circle (one circle/period per quadrant)
            pos_cos = np.cos(positions / quadrant_size * 2 * np.pi)

            # Rescale the circular positions to corridor coordinates (peak distance is 10 cm (half quadrant size)
            pos_quad = np.arccos(pos_cos) * quadrant_size / np.pi / 2

            return pos_quad

        def compute_errors(true, pred, rz_mask_true):
            """ Compute different errors. Called for absolute position and quadrant-corrected separately. """

            # Accuracy (fraction of bins that were predicted correctly)
            accuracy = np.sum(pred == true) / len(pred)

            # Absolute error per trial [cm]
            abs_error = np.sum(np.abs((pred - true))) * bin_size

            # Average error per frame [cm] (MAE)
            avg_error = np.mean(np.abs((pred - true))) * bin_size
            std_error = np.std(np.abs((pred - true))) * bin_size

            # Average squared error per frame [cm2] (MSE)
            mse = np.mean((pred - true)**2) * bin_size
            std_mse = np.std((pred - true)**2) * bin_size

            # Error-minimizing/accuracy-maximizing time shift (+- 1 sec) [s]
            framerate_int = int(np.round(framerate))
            shifts = np.arange(-np.round(framerate_int), framerate_int + 1).astype(int)
            avg_error_shift = [np.nanmean(np.abs((shift_array(pred, shift) - true))) * bin_size for shift in shifts]
            min_error = np.min(avg_error_shift)
            min_error_shift = shifts[np.argmin(avg_error_shift)] / framerate

            accuracy_shift = [np.sum(shift_array(pred, shift) == true) / len(pred) for shift in shifts]
            max_accuracy = np.max(accuracy_shift)
            max_accuracy_shift = shifts[np.argmax(accuracy_shift)] / framerate

            # MAE inside vs outside reward zones
            avg_error_in_rz = np.mean(np.abs((pred[rz_mask_true] - true[rz_mask_true]))) * bin_size
            avg_error_out_rz = np.mean(np.abs((pred[~rz_mask_true] - true[~rz_mask_true]))) * bin_size
            std_error_in_rz = np.std(np.abs((pred[rz_mask_true] - true[rz_mask_true]))) * bin_size
            std_error_out_rz = np.std(np.abs((pred[~rz_mask_true] - true[~rz_mask_true]))) * bin_size

            return dict(accuracy=accuracy, abs_error=abs_error, mae=avg_error, mae_std=std_error, mse=mse,
                        mse_std=std_mse, rmse=np.sqrt(mse), mae_min=min_error, mae_shift=min_error_shift,
                        acc_max=max_accuracy, acc_shift=max_accuracy_shift, mae_in_rz=avg_error_in_rz,
                        mae_out_rz=avg_error_out_rz, mae_in_rz_std=std_error_in_rz, mae_out_rz_std=std_error_out_rz)

        def rename_error_dict(dic):
            new_dic = {}
            for key, value in dic.items():
                if '_std' in key:
                    new_key = key[:key.index('_std')] + '_quad' + key[key.index('_std'):]
                else:
                    new_key = key + '_quad'
                new_dic[new_key] = value
            return new_dic

        # Immediately return NaNs if the whole trial failed (only Zero-predictions)
        if failed:
            error_raw = dict(accuracy=np.nan, abs_error=np.nan, mae=np.nan, mae_std=np.nan, mse=np.nan, mse_std=np.nan,
                             rmse=np.nan, mae_min=np.nan, mae_shift=np.nan, acc_max=np.nan, acc_shift=np.nan,
                             mae_in_rz=np.nan, mae_out_rz=np.nan, mae_in_rz_std=np.nan, mae_out_rz_std=np.nan)
            error_quad = rename_error_dict(error_raw)

            return dict(**error_raw, **error_quad, confusion_matrix=np.nan, accuracy_rz=np.nan, precision_rz=np.nan,
                        sensitivity_rz=np.nan, specificity_rz=np.nan, mcc=np.nan, failed_prediction=1)

        # Remove times without prediction
        y_true_ = y_true[~ np.isnan(y_pred)]
        y_pred_ = y_pred[~ np.isnan(y_pred)]

        # Fraction of false zero-predictions (used as metric for failed prediction)
        bad_pred = np.sum(y_pred_[y_true_ != 0] == 0) / len(y_true_)

        if ignore_zeros:
            y_true_ = y_true_[y_pred_ != 0]
            y_pred_ = y_pred_[y_pred_ != 0]

        # Get reward zone borders and masks for when the true or predicted position was inside a RZ
        borders = (hheise_behav.CorridorPattern() & f'pattern="{corridor_context}"').rescale_borders(80)
        borders[:, 0] -= 1
        borders[:, 1] += 1
        rz_true = np.stack([(b[0] <= y_true_) & (y_true_ < b[1]) for b in borders]).sum(axis=0) >= 1
        rz_pred = np.stack([(b[0] <= y_pred_) & (y_pred_ < b[1]) for b in borders]).sum(axis=0) >= 1

        # Circularize corridor quadrants to reflect periodic nature of the corridor
        y_true_quad = circularize_quadrants(y_true_)
        y_pred_quad = circularize_quadrants(y_pred_)

        # Compute errors for the raw and circular predictions
        errors_raw = compute_errors(true=y_true_, pred=y_pred_, rz_mask_true=rz_true)
        errors_quad = rename_error_dict(compute_errors(true=y_true_quad, pred=y_pred_quad, rz_mask_true=rz_true))
        errors = {**errors_raw, **errors_quad}

        # Confusion matrix of inside vs outside RZ prediction (binary, independent of quadrant)
        conf_matrix = confusion_matrix(~rz_true, ~rz_pred)     # Invert masks so that "in RZ" is the class to predict
        if conf_matrix.shape == (2, 2):
            tp = conf_matrix[0, 0]  # True positive (number of frames in RZs classified as RZs)
            fn = conf_matrix[0, 1]  # False negative (number of frames in RZs classified as non-RZs)
            fp = conf_matrix[1, 0]  # False positive (number of frames out of RZs classified as RZs)
            tn = conf_matrix[1, 1]  # True positive (number of frames out of RZs classified as non-RZs)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                acc_rz = (tp + tn) / np.sum(conf_matrix)  # Accuracy in classifying RZs (correctly identifying a frame as inside or outside RZ)
                precision_rz = tp / np.sum(conf_matrix[:, 0])    # When the model predicts a RZ, how likely is it actually a RZ?
                sensitivity_rz = tp / np.sum(conf_matrix[0])     # How often does the model correctly detect a RZ?
                specificity_rz = tn / np.sum(conf_matrix[1])     # How often does the model correctly detect a non-RZ?

            # Matthews correlation coefficient (correlation between predicted and true RZ status). Most reliable single metric, taking into account imbalances in the class sizes.
            denom_products = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            if denom_products == 0:
                mcc = 0
            else:
                mcc = (tp * tn - fp * fn) / np.sqrt(denom_products)
        else:
            # Only one class, happens sometimes for very badly predicted trials
            conf_matrix = acc_rz = precision_rz = sensitivity_rz = specificity_rz = mcc = np.nan

        return dict(**errors, confusion_matrix=conf_matrix, accuracy_rz=acc_rz, precision_rz=precision_rz,
                    sensitivity_rz=sensitivity_rz, specificity_rz=specificity_rz, mcc=mcc, failed_prediction=bad_pred)


@schema
class BayesianDecoderErrorChanceLevels(dj.Manual):
    definition = """ # Chance levels of error metrics for Bayesian Decoder.
    chance_level_id         : tinyint   # ID of the set of chance levels
    ---
    calculation_procedure   : varchar(256)  # How the chance levels were computed.
    failed_prediction       : float     # Fraction of frames where prediction failed completely (bin 0 falsely predicted)
    accuracy                : float     # Fraction of frames where the bin was correctly predicted
    abs_error               : float     # Absolute (cumulative) error in cm (per trial)
    mae                     : float     # Mean absolute error in cm (per frame)
    mae_std                 : float     # Standard deviation of the absolute error (SD-AE) (per frame)
    mse                     : float     # Mean squared error in cm2 (variance of the estimator)
    mse_std                 : float     # Standard deviation of the mean squared error
    rmse                    : float     # Root mean squared error in cm (standard deviation of the estimator)
    mae_min                 : float     # MAE with predicted positions shifted by 'mae_shift' seconds
    acc_max                 : float     # Accuracy with predicted positions shifted by 'acc_shift' seconds
    mae_in_rz               : float     # MAE when considering only frames where the mouse was inside a reward zone
    mae_in_rz_std           : float     # SD-AE when considering only frames where the mouse was inside a reward zone
    mae_out_rz              : float     # MAE when considering only frames where the mouse was outside a reward zone
    mae_out_rz_std          : float     # SD-AE when considering only frames where the mouse was outside a reward zone

    # Same errors, but computed when splitting the corridor into quadrants, taking into account the periodicity of reward zones
    accuracy_quad           : float     # Quadrant-wise accuracy
    abs_error_quad          : float     # Absolute (cumulative) quadrant-wise error in cm (per trial)
    mae_quad                : float     # Quadrant-wise MAE
    mae_quad_std            : float     # Standard deviation of the quadrant-wise MAE
    mse_quad                : float     # Quadrant-wise MSE
    mse_quad_std            : float     # Standard deviation of the quadrant-wise MSE
    rmse                    : float     # Root mean squared error in cm (standard deviation of the estimator)
    rmse_quad               : float     # Quadrant-wise RMSE
    mae_min_quad            : float     # Quadrant-wise MAE with predicted positions shifted by 'mae_shift' seconds
    acc_max_quad            : float     # Quadrant-wise accuracy with predicted positions shifted by 'acc_shift' seconds
    mae_in_rz_quad          : float     # Quadrant-wise MAE when using only frames where the mouse was inside a RZ
    mae_in_rz_quad_std      : float     # Quadrant-wise SD-AE when using only frames where the mouse was inside a RZ
    mae_out_rz_quad         : float     # Quadrant-wise MAE when using only frames where the mouse was outside a RZ
    mae_out_rz_quad_std     : float     # Quadrant-wise SD-AE when using only frames where the mouse was outside a RZ

    # Binary confusion matrix metrics for classifying inside/outside RZ frames
    accuracy_rz             : float     # Probability to correctly identify a frame as inside or outside RZ
    precision_rz            : float     # When the model predicts a RZ, how likely is it actually a RZ?
    sensitivity_rz          : float     # How often does the model correctly detect a RZ?
    specificity_rz          : float     # How often does the model correctly detect a non-RZ?
    mcc                     : float     # Matthews correlation coefficient. Most reliable single metric for a confusion matrix, accounting for imbalances in class sizes.
    """

    def import_chance_levels(self, file_path: str, chance_level_id: int, calculation_procedure: str,
                             include_high_sd: bool = False):
        """
        Import a set of chance levels from a CSV file.

        Args:
            file_path               : Absolute file path to the CSV file.
            chance_level_id         : ID of the chance level set.
            calculation_procedure   : Calculation procedure of the chance level set.
            include_high_sd         : Insert chance levels with a SD between mice of more than 5% of their mean.
        """

        # file_path = r'W:\Helmchen Group\Neurophysiology-Storage-01\Wahl\Hendrik\PhD\Papers\preprint\bayesian_decoder\chance_levels.csv'

        data = pd.read_csv(file_path, index_col=0)

        # Ignore rows without index (probably added manually after exporting the file)
        data = data[data.index.notnull()]

        # Take relevant error columns
        error_columns = ['accuracy', 'abs_error', 'mae', 'mae_std', 'mse', 'mse_std', 'rmse', 'mae_min', 'acc_max',
                         'mae_in_rz', 'mae_out_rz', 'mae_in_rz_std', 'mae_out_rz_std', 'accuracy_quad', 'abs_error_quad',
                         'mae_quad', 'mae_quad_std', 'mse_quad', 'mse_quad_std', 'rmse_quad', 'mae_min_quad',
                         'acc_max_quad', 'mae_in_rz_quad', 'mae_out_rz_quad', 'mae_in_rz_quad_std',
                         'mae_out_rz_quad_std', 'accuracy_rz', 'precision_rz', 'sensitivity_rz', 'specificity_rz',
                         'mcc', 'failed_prediction']
        data = data[error_columns]

        # Compute mean chance level and relative standard deviation across rows (mice)
        mean_cl = data.mean(axis=0).round(decimals=4)
        rel_sd_cl = data.std(axis=0) / mean_cl * 100

        # Raise warning if a chance level has a relative SD > 5%
        high_sd = rel_sd_cl.index[rel_sd_cl > 5]
        if len(high_sd) > 0:
            if include_high_sd:
                print('The chance levels of some errors have a high SD:', list(high_sd))
            else:
                raise ImportWarning('The chance levels of some errors have a high SD:', list(high_sd),
                                    '\nSet "include_high_sd=True" to enter the chance levels anyway.')

        self.insert1(dict(chance_level_id=chance_level_id, calculation_procedure=calculation_procedure, **mean_cl))


@schema
class BayesianDecoderLastPrestroke(dj.Computed):
    definition = """ # Predict position bin from neural activity with training data from last prestroke session of that mouse. Whole session is predicted at once, trials are concatenated.
    -> BayesianParameter
    -> common_img.Segmentation
    ---
    pos_predict             : longblob  # Predicted position bin per frame 
    pos_true                : longblob  # True VR position bin per frame
    confidence              : longblob  # Confidence distribution across all bins per frame (log-likelihood of bin) (n_frames, n_bins)
    unique_pred_bins        : tinyint   # Number of unique bins that were predicted. A number < 80 might hint at problems with the prediction.
    failed_prediction       : float     # Fraction of frames where prediction failed completely (bin 0 falsely predicted)
    accuracy                : float     # Fraction of frames where the bin was correctly predicted 
    abs_error               : float     # Absolute (cumulative) error in cm
    mae                     : float     # Mean absolute error in cm (per frame)
    mae_std                 : float     # Standard deviation of the absolute error (SD-AE) (per frame)
    mse                     : float     # Mean squared error in cm2 (variance of the estimator)
    mse_std                 : float     # Standard deviation of the mean squared error
    rmse                    : float     # Root mean squared error in cm (standard deviation of the estimator)
    mae_shift               : float     # Time shift of the predicted position that minimized the MAE (in seconds). Negative shift means that prediction was delayed relative to true position.
    mae_min                 : float     # MAE with predicted positions shifted by 'mae_shift' seconds
    acc_shift               : float     # Time shift of the predicted position that maximized the accuracy (in seconds). Negative shift means that prediction was delayed relative to true position.
    acc_max                 : float     # Accuracy with predicted positions shifted by 'acc_shift' seconds
    mae_in_rz               : float     # MAE when considering only frames where the mouse was inside a reward zone
    mae_in_rz_std           : float     # SD-AE when considering only frames where the mouse was inside a reward zone
    mae_out_rz              : float     # MAE when considering only frames where the mouse was outside a reward zone
    mae_out_rz_std          : float     # SD-AE when considering only frames where the mouse was outside a reward zone

    # Same errors, but computed when splitting the corridor into quadrants, taking into account the periodicity of reward zones
    accuracy_quad           : float     # Quadrant-wise accuracy
    abs_error_quad          : float     # Absolute (cumulative) quadrant-wise error in cm (per trial)
    mae_quad                : float     # Quadrant-wise MAE
    mae_quad_std            : float     # Standard deviation of the quadrant-wise MAE
    mse_quad                : float     # Quadrant-wise MSE
    mse_quad_std            : float     # Standard deviation of the quadrant-wise MSE
    rmse                    : float     # Root mean squared error in cm (standard deviation of the estimator)
    rmse_quad               : float     # Quadrant-wise RMSE
    mae_shift_quad          : float     # Time shift of the predicted position that minimized the quadrant-wise MAE (in seconds)
    mae_min_quad            : float     # Quadrant-wise MAE with predicted positions shifted by 'mae_shift' seconds
    acc_shift_quad          : float     # Time shift of the predicted position that maximized the quadrant-wise accuracy (in seconds)
    acc_max_quad            : float     # Quadrant-wise accuracy with predicted positions shifted by 'acc_shift' seconds
    mae_in_rz_quad          : float     # Quadrant-wise MAE when using only frames where the mouse was inside a RZ
    mae_in_rz_quad_std      : float     # Quadrant-wise SD-AE when using only frames where the mouse was inside a RZ
    mae_out_rz_quad         : float     # Quadrant-wise MAE when using only frames where the mouse was outside a RZ
    mae_out_rz_quad_std     : float     # Quadrant-wise SD-AE when using only frames where the mouse was outside a RZ

    # Binary confusion matrix metrics for classifying inside/outside RZ frames
    confusion_matrix = NULL : longblob  # 2x2 confusion matrix (rows: true classes, columns: predicted classes)
    accuracy_rz = NULL      : float     # Probability to correctly identify a frame as inside or outside RZ
    precision_rz = NULL     : float     # When the model predicts a RZ, how likely is it actually a RZ? NaN if the model never predicted a RZ.
    sensitivity_rz = NULL   : float     # How often does the model correctly detect a RZ?
    specificity_rz = NULL   : float     # How often does the model correctly detect a non-RZ?
    mcc = NULL              : float     # Matthews correlation coefficient. Most reliable single metric for a confusion matrix, accounting for imbalances in class sizes.

    run_time = CURRENT_TIMESTAMP    : timestamp
    """

    # Only include mice that are completely available (ignore 112, no matched cells)
    include_mice = [33, 41, 63, 69, 83, 85, 86, 89, 90, 91, 93, 95, 108, 110, 111, 113, 114, 115, 116, 121, 122]
    _key_source = (BayesianParameter() * common_img.Segmentation) & \
                  "username='hheise'" & f"mouse_id in {helper.in_query(include_mice)}"

    def make(self, key):

        # print(key)
        def fetch_data(session_key, parameters):

            if parameters['data_type'] == 'dff':
                act = (common_img.Segmentation & session_key).get_traces('dff')
                bin_act = np.stack((hheise_placecell.BinnedActivity.ROI & session_key).fetch('bin_activity'))
            elif parameters['data_type'] == 'decon':
                act = (common_img.Segmentation & session_key).get_traces('decon', decon_id=1)
                bin_act = np.stack((hheise_placecell.BinnedActivity.ROI & session_key).fetch('bin_spikerate'))
            else:
                raise NotImplementedError(f'Data type "{parameters["data_type"]}" not implemented.')
            trial_mask = (hheise_placecell.PCAnalysis & session_key).fetch1('trial_mask')
            pos = (hheise_behav.VRSession & session_key).get_array(attr='pos', get_frame_avg=True,
                                                                   as_dataframe=True)['pos'].to_numpy()

            # Transform position from VR units to bin indices
            bin_borders = np.linspace(-10, 110, bin_act.shape[1])
            pos = np.digitize(pos, bin_borders) - 1

            # Only include normal trials
            norm_trials = (hheise_behav.VRSession & session_key).get_normal_trials()
            norm_trial_mask = np.isin(trial_mask, norm_trials)
            trial_mask = trial_mask[norm_trial_mask]
            act = act[:, norm_trial_mask]
            bin_act = bin_act[:, :, norm_trials]
            running_mask, aligned_frames = (hheise_placecell.Synchronization.VRTrial & session_key &
                                            f'trial_id in {helper.in_query(norm_trials)}').fetch('running_mask', 'aligned_frames')
            aligned_frames = np.stack(aligned_frames)

            # Only keep running frames (speed > 5 cm/s)
            running_mask = np.concatenate(running_mask)
            if parameters['include_resting']:
                act_run = act
                pos_run = pos
                trial_mask_run = trial_mask
            else:
                act_run = act[:, running_mask]
                pos_run = pos[running_mask]
                trial_mask_run = trial_mask[running_mask]

            return dict(norm_trials=norm_trials, bin_act=bin_act, act=act_run, pos=pos_run, trial_masks=trial_mask_run,
                        aligned_frames=aligned_frames)

        # Get primary keys of the training session (last prestroke day)
        surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' & key).fetch('surgery_date')[0].date()
        train_key = {'username': key['username'], 'mouse_id': key['mouse_id'], 'session_num': key['session_num']}
        train_key = (common_img.Segmentation & train_key & f'day<="{surgery_day}"').fetch('KEY')[-1]
        train_key['place_cell_id'] = 2

        # Check if the current session is the last prestroke session (exit the function immediately if it is)
        if key['day'] == train_key['day']:
            # print(f'Skipping last prestroke session for mouse {key["mouse_id"]}')
            return

        # Fetch parameters
        params = (BayesianParameter & key).fetch1()
        fr = (common_img.ScanInfo & key).fetch1('fr')
        half_win_frames = int(np.round(fr * params['window_halfsize']))

        # Hard-code place_cell_id=2, most commonly used
        key['place_cell_id'] = 2

        bin_size = (hheise_placecell.PlaceCellParameter & key).fetch1('bin_length')

        # Fetch data from the current testing and training sessions
        test_data = fetch_data(key, parameters=params)
        train_data = fetch_data(train_key, parameters=params)

        # Make mask for cells that were matched in both sessions
        if key['mouse_id'] == 63:
            start_with_ref = True
        else:
            start_with_ref = False
        match_matrix = (common_match.MatchedIndex & f'mouse_id = {key["mouse_id"]}' & f'day in {helper.in_query([key["day"], train_key["day"]])}').construct_matrix(start_with_ref=start_with_ref)[f'{key["mouse_id"]}_1']
        match_matrix = match_matrix[(match_matrix == -1).sum(axis=1) == 0]
        train_col = common_match.MatchedIndex().key2title(train_key)
        test_col = common_match.MatchedIndex().key2title(key)
        test_mask_ids = (common_img.Segmentation.ROI & key & 'accepted=1').fetch('mask_id')
        train_mask_ids = (common_img.Segmentation.ROI & train_key & 'accepted=1').fetch('mask_id')

        if train_col not in match_matrix:
            print(f'Session "{train_col}" not found in match_matrix for mouse {key["mouse_id"]}. Skipping session.')
            return
        if test_col not in match_matrix:
            print(f'Session "{test_col}" not found in match_matrix for mouse {key["mouse_id"]}. Skipping session.')
            return

        # Make mask for useful neurons that will be fed into the decoder (e.g. place cells)
        train_matched_mask_ids = match_matrix[train_col].astype(int)
        if params['neuron_subset'] == 'all_place_cells':
            place_cell_ids = (hheise_placecell.PlaceCell.ROI & train_key & 'is_place_cell=1' &
                              f'mask_id in {helper.in_query(train_matched_mask_ids)}').fetch('mask_id')
            cell_mask = np.isin(train_mask_ids, place_cell_ids)
            cell_sort = []
        elif params['neuron_subset'] == 'place_cells':
            df = pd.DataFrame((hheise_placecell.PlaceCell.ROI & train_key & 'is_place_cell=1' &
                               f'mask_id in {helper.in_query(train_matched_mask_ids)}').fetch('mask_id', 'p', as_dict=True))
            if len(df) > 0:
                cell_sort = np.array(df.sort_values(by='p')['mask_id'])
            else:
                cell_sort = []
        elif params['neuron_subset'] == 'si':
            df = pd.DataFrame((hheise_placecell.SpatialInformation.ROI & train_key &
                               f'mask_id in {helper.in_query(train_matched_mask_ids)}').fetch('mask_id', 'si', as_dict=True))
            cell_sort = np.array(df.sort_values(by='si', ascending=False)['mask_id'])
        elif params['neuron_subset'] == 'stability':
            df = pd.DataFrame((hheise_placecell.SpatialInformation.ROI & train_key &
                               f'mask_id in {helper.in_query(train_matched_mask_ids)}').fetch('mask_id', 'stability', as_dict=True))
            cell_sort = np.array(df.sort_values(by='stability', ascending=False)['mask_id'])
        elif params['neuron_subset'] == 'firing_rate':
            df = pd.DataFrame((common_img.ActivityStatistics.ROI & train_key &
                               f'mask_id in {helper.in_query(train_matched_mask_ids)}').fetch('mask_id', 'rate_spikes', as_dict=True))
            cell_sort = np.array(df.sort_values(by='rate_spikes', ascending=False)['mask_id'])
        else:
            raise NotImplementedError(f'Neuronal subset of "{params["neuron_subset"]}" not implemented yet.')

        # After all fetching is complete, delete the place_cell_id key again
        del key['place_cell_id']

        if params['neuron_subset'] == 'all_place_cells':
            pass
        elif params['n_cells'] > len(cell_sort):
            cell_mask = np.ones(len(train_mask_ids)).astype(bool)
            print(f'Only {len(cell_sort)} (with n_cells = {params["n_cells"]}) found in training session {train_key}.')
        else:
            cell_mask = np.isin(train_mask_ids, cell_sort[:params['n_cells']])

        if np.sum(cell_mask) < 10:
            print(f'Less than 10 cells included in subset "{params["neuron_subset"]}" on session {train_key}. Skipped.')
            return

        # Filter match_matrix to only keep useful neurons
        train_mask_ids_filt = train_mask_ids[cell_mask]
        match_matrix_filt = match_matrix[match_matrix[train_col].isin(train_mask_ids_filt)]

        # Get indices of filtered cells (to filter whole array) in the order of match_matrix
        sorter = np.argsort(train_mask_ids)
        train_mask_id_sort = sorter[np.searchsorted(train_mask_ids, match_matrix_filt[train_col], sorter=sorter)]
        sorter = np.argsort(test_mask_ids)
        test_mask_id_sort = sorter[np.searchsorted(test_mask_ids, match_matrix_filt[test_col], sorter=sorter)]

        # Use sorted indices to query data arrays, which maintains the order of neurons in match_matrix, giving
        # matched neurons the same row in train and test dataset
        train_data['act_cells'] = train_data['act'][train_mask_id_sort]
        train_data['bin_act_cells'] = train_data['bin_act'][train_mask_id_sort]
        test_data['act_cells'] = test_data['act'][test_mask_id_sort]
        test_data['bin_act_cells'] = test_data['bin_act'][test_mask_id_sort]

        # For each neuron, smooth binned trace and compute mean and std for each bin across training trials
        train_data['bin_cells_mean'] = np.mean(gaussian_filter1d(train_data['bin_act_cells'][:, :], 1, axis=1), axis=2)
        train_data['bin_cells_sd'] = np.std(gaussian_filter1d(train_data['bin_act_cells'][:, :], 1, axis=1), axis=2)

        # Adjust SD of low-SD bins. This is somehow necessary for the decoder to work (without it all probabilities are 0)
        train_data['bin_cells_sd_adjust'] = train_data['bin_cells_sd'].copy()
        train_data['bin_cells_mean_adjust'] = train_data['bin_cells_mean'].copy()
        for i in range(train_data['bin_cells_sd'].shape[0]):
            sd_ratio = np.mean(train_data['bin_cells_sd'][i]) / train_data['bin_cells_sd'][i]
            low_sd = sd_ratio > 1
            train_data['bin_cells_sd_adjust'][i, low_sd] = train_data['bin_cells_sd'][i, low_sd] * sd_ratio[low_sd]
            # train_data['bin_cells_mean_adjust'][i, low_sd] = train_data['bin_cells_mean'][i, low_sd] * sd_ratio[low_sd]

        # Calculate occupancy probability of spatial bins in training trials
        train_data['occ'] = np.sum(train_data['aligned_frames'], axis=0) / np.sum(train_data['aligned_frames'])

        # Run the decoder with the mean/sd activity of training trials on the activity/position of the testing trial
        pos_predict, pos_confidence = BayesianDecoderWithinSession().estimate_position(y=test_data['act_cells'],
                                                                                       x_mean=train_data['bin_cells_mean_adjust'],
                                                                                       x_sd=train_data['bin_cells_sd_adjust'],
                                                                                       occ=train_data['occ'],
                                                                                       win=half_win_frames)

        if np.all(pos_predict[~np.isnan(pos_predict)] == 0):
            print(f'Prediction failed: Session {key}')
            errors = BayesianDecoderWithinSession().quantify_error(y_pred=pos_predict, y_true=test_data['pos'],
                                                                   bin_size=bin_size,
                                                                   ignore_zeros=bool(params['ignore_zeros']),
                                                                   failed=True)
            # Session-wide errors are not nullable, so we set NaNs to -1 as a dummy value
            errors = {k: -1 if np.isnan(v) else v for k, v in errors.items()}

        else:
            # Compute error metrics for the current test trial
            with np.errstate(divide='ignore'):
                errors = BayesianDecoderWithinSession().quantify_error(y_pred=pos_predict, y_true=test_data['pos'],
                                                                       bin_size=bin_size,
                                                                       ignore_zeros=bool(params['ignore_zeros']),
                                                                       framerate=fr)

        # Insert entries
        self.insert1(dict(**key, pos_predict=pos_predict, confidence=pos_confidence, pos_true=test_data['pos'],
                          unique_pred_bins=len(np.unique(pos_predict[~np.isnan(pos_predict)])), **errors))

