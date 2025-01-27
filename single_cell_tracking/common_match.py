#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 13/01/2022 18:03
@author: hheise

Schema for tracking and matching single neurons across sessions
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
from skimage.draw import polygon
from skimage import measure
from scipy import spatial, ndimage
from typing import Iterable, Tuple, List, Optional, Union, Dict
import math
import bisect
import ctypes
import itertools

import datajoint as dj

from schema import common_img, common_mice, hheise_placecell
from util import helper

schema = dj.schema('common_match', locals(), create_tables=True)


@schema
class CellMatchingParameter(dj.Manual):
    definition = """ # Parameters used in the cross-session cell matching pipeline. Defaults are params of Annas model.
    match_param_id              : int           # Index of parameter set
    ------
    contour_thresh = 0.05       : float         # Intensity threshold of ROI contour finding
    true_binary = 0             : tinyint       # Bool flag whether the footprint should truly be binarized or areas outside the contour but with weight > 0 should retain their non-zero weights.
    neighbourhood_radius = 80   : int           # Radius in um of the area around the ROI where matches are considered. Default of 80 translates to 50 pixels at 1x zoom
    nearby_neuron_cap = 15      : int           # Maximum nearest neurons found by KDTree nearest-neighbor-analysis
    fov_shift_patches = 8       : tinyint       # Number of patches squared into which the FOV will be split for shift estimation. E.g. with 8, the FOV will be split into 8x8=64 patches.
    -> CellMatchingClassifier
    landmark_aoe = 200          : int           # Radius of area-of-effect of manual confirmed landmarks between FOVs
    match_param_description     : varchar(512)  # Short description of the background and effect of the parameter set
    """


@schema
class CellMatchingClassifier(dj.Lookup):
    definition = """ # Table to store decision tree classifier models for cell matching GUI.
    classifier_id   : int           # Index of classifier version
    ------
    model_path                      : varchar(256)  # Path to classifier model, relative to Wahl folder server
    description                     : varchar(256)  # Description of the model
    n_cells                         : smallint      # Number of inputs for the classifier
    n_features                      : tinyint       # Number of features per cell
    """
    contents = [
        [0, 'CellMatching\\models\\model0.pkl', 'Original model trained by Anna Schmidt-Rohr', 3, 12]
    ]


@schema
class FieldOfViewShift(dj.Manual):
    definition = """ # Piecewise FOV shift between two sessions. Used to correct CoM coordinates. Manual table instead 
    # of Computed, because not every Session needs a shift, it is only computed once it is queried via the GUI. 
    -> common_img.QualityControl
    -> CellMatchingParameter
    matched_session : varchar(64)   # Identifier for the matched session: YYYY-MM-DD_sessionnum_motionid_caimanid
    ------
    shifts         : longblob       # 3D array with shape (n_dims, x, y), holding pixel-wise shifts for x (shifts[0]) and y (shifts[1]) coordinates.
    """

    def make(self, key: dict) -> None:
        """
        Compute FOV-shift map between the queried reference and a target image.
        Mean intensity images (avg_image) are used instead of local correlation images because they are a better
        representation of the FOV anatomy, more stable over time, and are less influenced by  activity patterns. Images
        are split into patches, and the shift is calculated for each patch separately with phase correlation. The
        resulting shift map is scaled up and missing values interpolated to the original FOV size to get an estimated
        shift value for each pixel.

        Args:
            key: Primary keys of the reference session in common_img.QualityControl(),
                    ID of the CellMatchingParameter() entry,
                    and identifier string of the matched session: YYYY-MM-DD_sessionnum_motionid_caimanid.
                    It is assumed that the reference and matched sessions come from the same individual mouse.
        """

        from skimage import registration
        from scipy import ndimage

        """
        ToDo: check if the reverse shift matrix already exist (matched_session -> reference_session), and use existing
        matrix (with potentially better landmarks) instead of computing new FOV shift. Potentially this ChatGPT code
        works, but has not been tested:
    
        # Get the shape of the shift matrix
        rows, cols = old_shifts.shape
        
        # Create a new matrix for aligning image B to image A
        reverse_shift = np.zeros((rows, cols), dtype=np.int32)
        
        # Assign the reversed shift values to the new matrix
        for i in range(rows):
            for j in range(cols):
                shift_x, shift_y = old_shifts[i, j]
                new_i = i - shift_y
                new_j = j - shift_x
                if new_i >= 0 and new_i < rows and new_j >= 0 and new_j < cols:
                    reverse_shift[new_i, new_j] = (-shift_x, -shift_y)        
        """

        # Fetch reference FOV, parameter set, and extract the primary keys of the matched session from the ID string
        match_keys = key['matched_session'].split('_')
        match_key = dict(username=key['username'], mouse_id=key['mouse_id'], day=match_keys[0],
                         session_num=int(match_keys[1]), motion_id=int(match_keys[2]))

        print_dict = dict(username=key['username'], mouse_id=key['mouse_id'], day=key['day'],
                          session_num=key['session_num'], motion_id=key['motion_id'])
        print(f"Computing FOV shift between sessions\n\t{print_dict} and \n\t{match_key}")

        fov_ref = (common_img.QualityControl & key).fetch1('avg_image')
        fov_match = (common_img.QualityControl & match_key).fetch1('avg_image')
        params = (CellMatchingParameter & key).fetch1()

        # Calculate pixel size of each patch
        img_dim = fov_ref.shape
        patch_size = int(img_dim[0] / params['fov_shift_patches'])

        # Shift maps are a 2D matrix of shape (n_patch, n_patch), with the phase correlation of each patch
        shift_map = np.zeros((2, params['fov_shift_patches'], params['fov_shift_patches']))
        for row in range(params['fov_shift_patches']):
            for col in range(params['fov_shift_patches']):
                # Get a view of the current patch by slicing rows and columns of the FOVs
                curr_ref_patch = fov_ref[row * patch_size:row * patch_size + patch_size,
                                 col * patch_size:col * patch_size + patch_size]
                curr_tar_patch = fov_match[row * patch_size:row * patch_size + patch_size,
                                 col * patch_size:col * patch_size + patch_size]
                # Perform phase cross correlation to estimate image translation shift for each patch
                patch_shift, _, _ = registration.phase_cross_correlation(curr_ref_patch, curr_tar_patch,
                                                                         upsample_factor=100)
                shift_map[:, row, col] = patch_shift

        # Use scipy's zoom to upscale single-patch shifts to FOV size and get pixel-wise shifts via spline interpolation
        x_shift_map_big = ndimage.zoom(shift_map[0], patch_size, order=3)  # Zoom X and Y shifts separately
        y_shift_map_big = ndimage.zoom(shift_map[1], patch_size, order=3)
        # Further smoothing (e.g. Gaussian) is not necessary, the interpolation during zooming smoothes harsh borders
        shift_map_big = np.stack((x_shift_map_big, y_shift_map_big))

        # If caiman_id is in the primary keys, remove it, because it is not in QualityControl, this table's parent
        if 'caiman_id' in key:
            del key['caiman_id']

        # Insert shift map into the table
        self.insert1(dict(**key, shifts=shift_map_big))

    def update_with_landmark(self, coord_ref: Iterable[float], coord_tar: Iterable[float], landmark_radius: int) -> None:
        """
        Update a FOV shift matrix with a new, manually accepted landmark shift (e.g. center of mass of matched cells).
        The function replaces the old FOV shift at the landmark location with the new landmark shift, and gradually
        decreases the influence of the landmark shift with distance from the landmark location using a Gaussian filter
        with a radius of 'landmark_radius' and standard deviation of `landmark_radius/3`.

        Args:
            coord_ref:  X, Y coordinates of the landmark in the reference image.
            coord_tar: X, Y coordinates of the same landmark in the target image.
            landmark_radius: Radius of the smoothing Gaussian kernel, as well as 3x its standard deviation.
                Decreasing this number reduces the "impact size" of the landmark shift to the nearby area.
        """

        key = self.fetch1('KEY')
        fov_shift = self.fetch1('shifts').copy()

        com_shift = np.array(coord_ref) - np.array(coord_tar)  # The shift is computed from target to reference session

        # Compute Gaussian mask to smooth the new shift with the existing FOV shift array
        sigma = landmark_radius / 3         # Smaller sigma means less smoothing and smaller impact area
        mask = np.zeros_like(fov_shift[1])
        mask[int(np.round(coord_tar[0])), int(np.round(coord_tar[1]))] = 1
        # Mode "constant" cuts off the filter at the edges, to avoid edge artifacts
        gaussian_mask = ndimage.gaussian_filter(mask, sigma=sigma, mode='constant', radius=landmark_radius)
        # Normalize kernel to have the shift at the center be 100% the new shift
        gaussian_mask_norm = (gaussian_mask - np.min(gaussian_mask)) / (np.max(gaussian_mask) - np.min(gaussian_mask))

        # Apply X and Y shifts
        fov_shift[0] = fov_shift[0] * (1 - gaussian_mask_norm) + com_shift[0] * gaussian_mask_norm
        fov_shift[1] = fov_shift[1] * (1 - gaussian_mask_norm) + com_shift[1] * gaussian_mask_norm

        FieldOfViewShift().update1(dict(key, shifts=fov_shift))

    def quiver_plot(self, downsampling: int = 20, arrow_scale: int = 10, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Draw a quiver plot (arrows) representing the FOV shift vectors.

        Args:
            downsampling: Downsampling factor of the original FOV shift matrix. Higher downsampling means fewer arrows.
            arrow_scale: Scale factor of the arrow lengths (shift size in px).
            ax: If provided, draw the quiver plot in that axes. Otherwise, create a new figure.

        Returns:
            Axes object containing the quiver plot.
        """

        # Fetch shift matrix
        fov_shift = self.fetch1('shifts')

        # Create coordinate grid for the arrows
        x, y = np.meshgrid(np.arange(fov_shift[0].shape[0]), np.arange(fov_shift[0].shape[1]))

        # Draw the figure from a downsampled matrix to reduce the number of arrows in the grid
        if ax is None:
            plt.figure()
            ax = plt.gca()

        ax.quiver(x[::downsampling, ::downsampling], y[::downsampling, ::downsampling],
                  fov_shift[0, ::downsampling, ::downsampling], fov_shift[1, ::downsampling, ::downsampling],
                  pivot='middle', scale=1/arrow_scale, angles='xy', scale_units='xy')
        ax.invert_yaxis()

        return ax

    def stream_plot(self, density: int = 2, color: Optional[str] = 'magnitude', cmap: str = 'turbo', cbar: bool = False,
                    ax: Optional[plt.Axes] = None, title: bool=True) -> plt.Axes:
        """
        Draw a stream plot (flowing field lines) representing the FOV shift vectors.

        Args:
            density: Line density. Higher number means more lines.
            color: If 'magnitude' (default), color the lines by the shift vector magnitude at that location. If a
                matplotlib color string, all lines are drawn in that color.
            cmap: Colormap used to encode magnitude. Only used when color = 'magnitude'.
            cbar: Bool flag whether to draw a color bar. Only used when color = 'magnitude'. Does not work if 'ax' is provided.
            ax: If provided, draw the stream plot in that axes. Otherwise, create a new figure.
            plot_title: Whether the reference and target sessions should be written in the title.

        Returns:
            Axes object containing the stream plot.
        """
        # Fetch shift matrix
        fov_shift = self.fetch1('shifts')
        mouse, day, matched_day = self.fetch1('mouse_id', 'day', 'matched_session')

        # Create coordinate grid for the field lines
        x, y = np.meshgrid(np.arange(fov_shift[0].shape[0]), np.arange(fov_shift[0].shape[1]))

        # Set the color of the lines to either magnitude of shift, or just the color string
        if color == 'magnitude':
            line_color = np.hypot(fov_shift[0], fov_shift[1])
        else:
            line_color = color

        # Create new figure, if no ax is given. Otherwise, prevent drawing a colo bar
        if ax is None:
            fig = plt.figure()
            ax = plt.gca()
        else:
            cbar = False

        # Draw stream plot and invert y-axis to match the automatically inverted y-axis of FOV images
        strm = ax.streamplot(x, y, fov_shift[0], fov_shift[1], density=density, color=line_color, cmap=cmap,
                             arrowsize=1.5)
        ax.invert_yaxis()

        # Draw colorbar
        if cbar:
            colorbar = fig.colorbar(strm.lines)
            colorbar.set_label('Shift magnitude [px]', rotation=270, labelpad=15, fontsize=12)

        if title:
            ax.set_title(f'M{mouse}: {day} --> {matched_day[:10]}')

        return ax

@schema
class MatchingFeatures(dj.Computed):
    definition = """ # Features of the ROIs that are used for the cell matching algorithm
    -> common_img.Segmentation
    -> CellMatchingParameter
    ------
    match_time = CURRENT_TIMESTAMP  : timestamp
    """

    class ROI(dj.Part):
        definition = """ # Features of neurons that are used for the cell matching algorithm
        -> master
        mask_id                     : int           # ID of the ROI (same as common_img.Segmentation)
        ------
        contour                     : longblob      # (Semi)-binarized spatial contour of the ROI, maximally cropped
        neighbourhood               : longblob      # Local area of the mean intensity template around the neuron
        rois_nearby                 : int           # Number of neurons in the neighbourhood. Capped by parameter.
        closest_roi                 : int           # Index of the nearest ROI (smallest physical distance)
        closest_roi_angle           : float         # Radial angular distance of the closest ROI
        neighbours_quadrant         : longblob      # Number of nearby ROIs split into quadrants (clockwise, from top-left)
        """

    def make(self, key: dict, ignore_rejected: Optional[bool] = True,
             adaptive_footprint: Optional[bool] = False) -> None:
        """
        Compute ROI features that are used as criteria for matching cells across sessions (through Anna's GUI).

        Args:
            key: Primary keys of the current MatchingFeatures() entry.
            ignore_rejected: Optional make keyword argument to also match rejected components.
            adaptive_footprint: Optional make keyword argument to adaptively set footprint threshold if given threshold
                fails to produce valid footprint. WARNING: This will lead to the threshold parameter being ignored.
        """

        print(f"Computing matching features for ROIs in entry {key}.")

        # Fetch relevant data
        if ignore_rejected:
            footprints = (common_img.Segmentation.ROI & key & 'accepted=1').get_rois()
            mask_ids, coms = np.vstack((common_img.Segmentation.ROI & key & 'accepted=1').fetch('mask_id', 'com'))
        else:
            footprints = (common_img.Segmentation.ROI & key).get_rois()
            mask_ids, coms = np.vstack((common_img.Segmentation.ROI & key).fetch('mask_id', 'com'))
        template = (common_img.QualityControl & key).fetch1("avg_image")
        params = (CellMatchingParameter & key).fetch1()

        # Convert neighbourhood radius (in microns) to zoom-dependent radius in pixels
        mean_res = np.mean((common_img.CaimanParameter & key).get_parameter_obj(key)['dxy'])
        margin_px = int(np.round(params['neighbourhood_radius'] / mean_res))

        coms_list = [list(com) for com in coms]  # KDTree expects a list of lists as input
        neighbor_tree = spatial.KDTree(coms_list)  # Build kd-tree to query nearest neighbours of any ROI

        new_entries = []
        for rel_idx, glob_idx in enumerate(mask_ids):
            com = coms[rel_idx]

            # Crop and binarize current footprint
            footprint = self.binarize_footprint(footprints[rel_idx], params['contour_thresh'],
                                                true_binary=params['true_binary'])

            # Catch bad ROI without a contour
            if np.all(footprint == 0):
                if adaptive_footprint:
                    contour_thresh = params['contour_thresh']
                    while np.all(footprint == 0) and contour_thresh > 0:
                        contour_thresh = contour_thresh / 10
                        footprint = self.binarize_footprint(footprints[rel_idx], contour_thresh,
                                                            true_binary=params['true_binary'])
                    if np.all(footprint == 0):
                        raise Warning(f'No valid footprint for neuron {glob_idx} in session {key},\nwith any positive '
                                      f'threshold. Consider rejecting the ROI!')
                    else:
                        print(f'Neuron {glob_idx} in session {key} needed a contour threshold of {contour_thresh} to '
                              f'yield a valid footprint.\nThreshold value in CellMatchingParameter was discarded for '
                              f'this neuron.')
                else:
                    raise Warning(f'No valid footprint for neuron {glob_idx} in session {key}.\nConsider rejecting the '
                                  f'ROI or set "adaptive_footprint=True"!')

            # Crop the template around the current ROI
            area_image = self.crop_template(template, com, margin_px)[0]

            # Use KDTree's nearest-neighbour analysis to get all ROIs sorted by distance to the current ROI
            distance, index = neighbor_tree.query(coms_list[rel_idx], k=len(coms_list))

            # Get the relative index and angle (direction) of the nearest ROI
            closest_idx = index[1]
            closest_roi_angle = math.atan2(com[1] - coms[closest_idx][1], com[0] - coms[closest_idx][0])

            # get the number of ROIs in the neighbourhood and their relative indices
            num_neurons_in_radius = bisect.bisect(distance, margin_px) - 1  # -1 to not count the ROI itself
            index_in_radius = index[1: max(0, num_neurons_in_radius) + 1]   # start at 1 to not count the ROI itself

            # Get the number of neighbours in each neighbourhood quadrant (top left to bottom right)
            neighbours_quadrants = self.neighbours_in_quadrants(coms, rel_idx, index_in_radius)

            # Create part-entry (translating relative IDs to global IDs
            neuron_features = {'contour': footprint, 'neighbourhood': area_image,
                               'closest_roi': mask_ids[closest_idx], 'closest_roi_angle': closest_roi_angle,
                               'rois_nearby': num_neurons_in_radius,
                               'neighbours_quadrant': neighbours_quadrants}
            new_entries.append(dict(**key, **neuron_features, mask_id=glob_idx))

        # After all ROIs have been processed, insert master and part entries
        self.insert1(key)
        self.ROI().insert(new_entries)

    @staticmethod
    def binarize_footprint(footprint: np.ndarray, contour_thresh: float, true_binary: bool) -> np.ndarray:
        """
        Crops a spatial footpring to its minimal rectangle and binarizes the footprint with a given threshold.
        As cropping happens before thresholding, the final output array may be larger than the thresholded footprint.

        Args:
            footprint: 2D array with shape (x_fov, y_fov) of the FOV with the spatial footprint weights
            contour_thresh: Threshold of the contour finding algorithm which is applied to the spatial weights
            true_binary: Bool flag whether the image should truly be binarized or areas that are outside the contour
                            but still have a weight > 0 should retain their non-zero weights.

        Returns:
            2D array with shape (x_crop, y_crop) with the cropped and binarized footprint.
        """
        # crop FOV to the minimal rectangle of the footprint
        coords_non0 = np.argwhere(footprint)
        x_min, y_min = coords_non0.min(axis=0)
        x_max, y_max = coords_non0.max(axis=0)
        cropped_footprint = footprint[x_min:x_max + 1, y_min:y_max + 1]

        # measure contour area
        contours = measure.find_contours(cropped_footprint, contour_thresh, fully_connected='high')

        # Todo: validate that the actual binary image works as well as the semi-binarized version
        if true_binary:
            new_footprint = np.zeros(cropped_footprint.shape, dtype=np.int8)
        else:
            new_footprint = np.copy(cropped_footprint)

        if contours:
            # Compute polygon area of the contour and fill it with 1
            rr, cc = polygon(contours[0][:, 0], contours[0][:, 1], cropped_footprint.shape)
            if true_binary:
                new_footprint[rr, cc] = 1
            else:
                new_footprint[rr, cc] = 255

        else:
            # if no contour could be found, binarize the entire footprint manually
            # print("no contour")
            new_footprint = np.where(cropped_footprint > contour_thresh, 1, 0)

        return new_footprint

    @staticmethod
    def crop_template(template: np.ndarray, roi_com: Iterable, margin_px: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Crops the template FOV around the center of mass of an ROI.

        Args:
            template    : 2D array, FOV in which the ROI is located
            roi_com     : X and Y coordinates of the center of mass of the ROI
            margin_px   : Radius of the cropped image, in pixel

        Returns:
            Cropped FOV, and new CoM coordinates relative to the cropped FOV
        """
        # Translate the margin in um into pixels through the mean resolution (x and y resolution differ slightly)

        ymin = max(0, int(roi_com[1] - margin_px))
        ymax = min(template.shape[1], int(roi_com[1] + margin_px))
        xmin = max(0, int(roi_com[0] - margin_px))
        xmax = min(template.shape[0], int(roi_com[0] + margin_px))
        cropped = template[xmin:xmax, ymin:ymax]
        return cropped, (roi_com[1] - ymin, roi_com[0] - xmin)

    @staticmethod
    def neighbours_in_quadrants(coms: np.ndarray, neuron_ref: int, neuron_idxs: np.ndarray) -> List[int]:
        """ Calculates the number of ROIs that are in each quadrant of the reference ROI (excluding itself).

        Args:
            coms: 2D array of shape (n_neurons, 2) with center of mass of all ROIs in the FOV.
            neuron_ref: Index of the reference ROI.
            neuron_idxs: Indices of the ROI that should be sorted into quadrants (with respect to the ref neuron).

        Returns:
            List with four integers, the number of ROIs in each quadrant of the reference ROI, in order top-left, top-
                right, bottom-left, bottom-right.
        """
        quadrant_split = [0, 0, 0, 0]
        x_ref = coms[neuron_ref][0]
        y_ref = coms[neuron_ref][1]
        for idx in neuron_idxs:
            x = coms[idx][0]
            y = coms[idx][1]
            if x_ref <= x and y_ref <= y:
                quadrant_split[0] += 1  # top-left quadrant
            elif x_ref <= x and y_ref >= y:
                quadrant_split[1] += 1  # top-right quadrant
            elif x_ref >= x and y_ref >= y:
                quadrant_split[2] += 1  # bottom-left quadrant
            else:
                quadrant_split[3] += 1  # bottom-right quadrant
        return quadrant_split


@schema
class IgnoreMatchedSession(dj.Manual):
    definition = """ # Sessions that should be ignored when constructing MatchedIndex matrices.
    -> common_img.Scan
    ------
    insert_time = CURRENT_TIMESTAMP : timestamp
    """


@schema
class MatchedIndex(dj.Manual):
    definition = """ # Matched indices of ROIs in other sessions, created through Cell Matching GUI
    -> MatchingFeatures.ROI
    matched_session : varchar(64)   # Identifier for the matched session: YYYY-MM-DD_sessionnum_motionid_caimanid
    ------
    matched_id      : int           # Mask ID of the same neuron in the matched session
    reverse='1'     : tinyint       # Boolean flag whether this entry is a reverse of a manual match
    matched_time = CURRENT_TIMESTAMP  : timestamp
    """

    def helper_insert1(self, key: dict, auto_reject_duplicates: bool = False,
                       insert_reverse_entries: bool = True, force_reverse_val: int = 0) -> Optional[bool]:
        """
        Helper function that inserts a confirmed neuron match for both sessions and warns if a cell has been tracked
        twice with different reference cells.

        Args:
            key: Primary keys of the current matched cell entry
            auto_reject_duplicates: Bool flag that automatically rejects duplicate entries and does not show the popup.
            insert_reverse_entries: Bool flag if reverse entries should automatically be computed and inserted.

        Returns:
            If nothing was inserted, return False for the GUI to keep track of failed inserts. Otherwise return None.
        """

        def make_popup(message: str) -> bool:
            """
            Draws a popup window to get user feedback in case of database inconsistencies.

            Args:
                message: Error-specific message to tell user where the inconsistencies are.

            Returns:
                User response; 6 (YES, transaction should continue), 7 (NO, transaction should be aborted),
                or 2 (CANCEL, which deletes both matches).
            """

            popup_msg = '\nIf you think your current match is correct, press "YES" to overwrite the old match.\n' \
                        '\nIf you think that the old match is correct, press "NO" to discard the current match.\n' \
                        '\nIf you are not sure, press "Cancel" to delete both matches. These cells will show up as ' \
                        '"unmatched" when re-loading the GUI and will have to be matched again.'
            answer = ctypes.windll.user32.MessageBoxW(0, message + popup_msg, 'Conflicting entry!', 0x00000011 | 0x0002)

            return answer

        def revert_key(orig_key: dict) -> dict:
            """
            Transforms a query dictionary of a single MatchedIndex entry (Cell X in Session A is matched to cell Y in
            Session B) into its reverse entry (Cell Y in Session B is matched to cell X in Session A).

            Args:
                orig_key: Keys of the entry. Has to uniquely identify one MatchedIndex entry.

            Returns:
                The transformed reverse entry.
            """
            return dict(username=orig_key['username'],
                        mouse_id=orig_key['mouse_id'],
                        day=orig_key['matched_session'].split('_')[0],
                        session_num=orig_key['matched_session'].split('_')[1],
                        motion_id=orig_key['matched_session'].split('_')[2],
                        caiman_id=orig_key['matched_session'].split('_')[3],
                        match_param_id=orig_key['match_param_id'],
                        mask_id=orig_key['matched_id'],
                        matched_session=f"{orig_key['day']}_{orig_key['session_num']}_{orig_key['motion_id']}_"
                                        f"{orig_key['caiman_id']}",
                        matched_id=orig_key['mask_id'],
                        reverse=int(not orig_key['reverse']))

        # Dict that remembers which entries should be inserted, updated or deleted based on database integrity checks
        change_dict = {'inserts': [],
                       'updates': [],
                       'deletes': []}

        # Dicts with different primary keys left out to query duplicate/wrong entries
        # Main entry
        dup_keys = key.copy()
        del dup_keys['matched_id']
        if 'reverse' in dup_keys:
            del dup_keys['reverse']

        # Reverse entry
        if 'reverse' not in key:
            key['reverse'] = force_reverse_val
        reverse_key = revert_key(key)
        diff_ref = reverse_key.copy()
        diff_tar = reverse_key.copy()
        del diff_ref['mask_id']     # This queries sessions where a different reference is matched to the same target
        del diff_ref['reverse']
        del diff_tar['matched_id']  # This queries sessions where a different target is matched to the same reference
        del diff_tar['reverse']

        ### Decision tree of the whole MatchedIndex process. Visualized on the Wahl Server under "CellMatching" ###
        ### Branch 1
        if key['matched_id'] == -1:
            # Check if current neuron X in session A (x_a) is already matched (Branch 1.1)
            if len(self & dup_keys) > 0:                                # Branch 1.1
                existing_entry = (self & dup_keys).fetch1()                 # X(a) --> M(b)
                if existing_entry['matched_id'] != -1:                  # Branch 1.1.1
                    msg = f'Cell {key["mask_id"]} in session {key["day"]} is already matched with cell ' \
                          f'{existing_entry["matched_id"]} in session {existing_entry["matched_session"][:11]}.\n' \
                          f'You selected it as "No Match" now.\n'

                    if not auto_reject_duplicates:
                        answer = make_popup(msg)
                    else:
                        answer = -1

                    if answer == 6:          # Branch 1.1.1.2
                        change_dict['updates'].append(key)
                        change_dict['deletes'].append(revert_key(existing_entry))
                    elif answer == 2:        # Branch 1.1.1.1
                        change_dict['deletes'].append(revert_key(existing_entry))
                        change_dict['deletes'].append(existing_entry)
                    else:
                        print('New match rejected, no change to database.')
                        return False
                else:                                                   # Branch 1.1.2
                    print('Match already in database, insert skipped.')
                    return False
            else:                                                       # Branch 1.2

                # Check if another reference neuron targets the current neuron
                if len(self & diff_ref) > 0:                            # Branch 1.2.1
                    existing_rev_entry = (self & diff_ref).fetch1()         # M(b) --> X(a)
                    msg = f'Cell {key["mask_id"]} in session {key["day"]} is already a target cell from' \
                          f'{existing_rev_entry["mask"]} in session {existing_rev_entry["day"][:11]}.\n' \
                          f'You selected cell {key["mask_id"]} as "No Match" now.\n'

                    if not auto_reject_duplicates:
                        answer = make_popup(msg)
                    else:
                        answer = -1

                    if answer == 6:             # Branch 1.2.1.2
                        change_dict['inserts'].append(key)
                        change_dict['deletes'].append(existing_rev_entry)
                    elif answer == 2:
                        change_dict['deletes'].append(existing_rev_entry)
                        change_dict['deletes'].append(revert_key(existing_rev_entry))
                    else:                                                   # Branch 1.2.1.1
                        print('New match rejected, no change to database.')
                        return False
                else:                                                   # Branch 1.2.2
                    change_dict['inserts'].append(key)

        ### Branch 2
        else:
            if len(self & dup_keys) > 0:                                # Branch 2.1
                existing_entry = (self & dup_keys).fetch1()                 # X(a) --> ?(b)
                if existing_entry['matched_id'] == key['matched_id']:   # Branch 2.1.1, X(a) --> Y(b) exists
                    if len(self & diff_tar) > 0:                        # Branch 2.1.1.1
                        existing_rev_entry = (self & diff_tar).fetch1()     # Y(b) --> ?(a)
                        if existing_rev_entry['matched_id'] == key['mask_id']:  # Branch 2.1.1.1.1, Y(b) --> X(a)
                            print('Match already in database, insert skipped.')
                            return False
                        else:                                           # Branch 2.1.1.1.2, Y(b) --> Z(a)
                            msg = f'You matched cell {key["mask_id"]} in session {key["day"]} to cell ' \
                                  f'{key["matched_id"]} in session {key["matched_session"]}.\nHowever, that cell is ' \
                                  f'already matched to cell {existing_rev_entry["matched_id"]} in session ' \
                                  f'{existing_rev_entry["matched_session"]}.\n'

                            if not auto_reject_duplicates:
                                answer = make_popup(msg)
                            else:
                                answer = -1

                            if answer == 6:   # Branch 2.1.1.1.2.2
                                change_dict['updates'].append(reverse_key)
                                if existing_rev_entry["matched_id"] != -1:      # Branch 2.1.1.1.2.2.2
                                    change_dict['deletes'].append(revert_key(existing_rev_entry))
                            elif answer == 2:
                                if existing_rev_entry["matched_id"] != -1:      # Branch 2.1.1.1.2.2.2
                                    change_dict['deletes'].append(revert_key(existing_rev_entry))
                                    change_dict['deletes'].append(existing_rev_entry)
                            else:                                       # Branch 2.1.1.1.2.1
                                print('New match rejected, no change to database.')
                                return False
                    else:                                               # Branch 2.1.1.2
                        if insert_reverse_entries:
                            change_dict['inserts'].append(reverse_key)
                        else:
                            print('Reverse entry does not exist, but insert_reverse_entries == False, insert skipped.')
                            return False
                else:                                                   # Branch 2.1.2, X(a) --> M(b) exists
                    msg = f'You matched cell {key["mask_id"]} in session {key["day"]} to cell ' \
                          f'{key["matched_id"]} in session {key["matched_session"]}.\nHowever, cell {key["mask_id"]} is ' \
                          f'already matched to cell {existing_entry["matched_id"]} in session ' \
                          f'{existing_entry["matched_session"]}.\n'

                    if not auto_reject_duplicates:
                        answer = make_popup(msg)
                    else:
                        answer = -1

                    if answer == 6:    # Branch 2.1.2.2
                        # Check if the new target neuron Y(b) has a match in session A
                        if len(self & diff_tar) > 0:                 # Branch 2.1.2.2.2, Y(b) --> ?(a) exists
                            change_dict['updates'].append(key)           # Branch 2.1.2.2.2.1
                            change_dict['updates'].append(reverse_key)
                            if existing_entry['matched_id'] != -1:      # Branch 2.1.2.2.2.2
                                change_dict['deletes'].append(revert_key(existing_entry))
                        else:                                           # Branch 2.1.2.2.1, Y(b) --> ?(a) does not exist
                            change_dict['updates'].append(key)           # Branch 2.1.2.2.1.1
                            change_dict['inserts'].append(reverse_key)
                            if existing_entry['matched_id'] != -1:      # Branch 2.1.2.2.1.2
                                change_dict['deletes'].append(revert_key(existing_entry))
                    elif answer == 2:
                        if existing_entry['matched_id'] != -1:     # If existing match is -1, no reverse entry to delete
                            change_dict['deletes'].append(revert_key(existing_entry))
                        change_dict['deletes'].append(existing_entry)
                    else:                                               # Branch 2.1.2.1
                        print('New match rejected, no change to database.')
                        return False
            else:                                                       # Branch 2.2
                if len(self & diff_tar) > 0:                            # Branch 2.2.1
                    existing_rev_entry = (self & diff_tar).fetch1()         # Y(b) --> ?(a)
                    if existing_rev_entry['matched_id'] == key['mask_id']:      # Branch 2.2.1.1
                        change_dict['inserts'].append(key)
                    else:                                               # Branch 2.2.1.2
                        msg = f'You matched cell {key["mask_id"]} in session {key["day"]} to cell ' \
                              f'{key["matched_id"]} in session {key["matched_session"]}.\nHowever, that cell is ' \
                                  f'already matched to cell {existing_rev_entry["matched_id"]} in session ' \
                                  f'{existing_rev_entry["matched_session"]}.\n'

                        if not auto_reject_duplicates:
                            answer = make_popup(msg)
                        else:
                            answer = -1

                        if answer == 6:      # Branch 2.2.1.2.2
                            change_dict['updates'].append(reverse_key)          # Branch 2.2.1.2.2.1
                            change_dict['inserts'].append(key)
                            if existing_rev_entry['matched_id'] != -1:          # Branch 2.2.1.2.2.2
                                change_dict['deletes'].append(revert_key(existing_rev_entry))
                        elif answer == 2 and existing_rev_entry['matched_id'] != -1:
                            change_dict['deletes'].append(revert_key(existing_rev_entry))
                            change_dict['deletes'].append(existing_rev_entry)
                        else:                                           # Branch 2.2.1.2.1
                            print('New match rejected, no change to database.')
                            return False
                else:                                                   # Branch 2.2.2
                    change_dict['inserts'].append(key)
                    if insert_reverse_entries:
                        change_dict['inserts'].append(reverse_key)

        # After traversing the decision tree, execute the changes within a transaction to avoid incomplete changes
        connection = self.connection
        with connection.transaction:
            # Timestamp that is used to replace matched_time when updating a match
            curr_time = datetime.utcnow().replace(microsecond=0)
            for update in change_dict['updates']:
                self.update1(dict(update, matched_time=curr_time))
            for delete in change_dict['deletes']:
                # Another safety check that we are only deleting one row
                curr_delete_query = (self & delete)
                if len(curr_delete_query) == 1:
                    (self & delete).delete_quick()
                elif len(curr_delete_query) == 0:
                    print('Entry already deleted.')
                else:
                    print(f'Aborted delete of {delete} because more than one row was affected.')
            for insert in change_dict['inserts']:
                self.insert1(insert)

    def remove_match_from_dict(self, key: dict) -> None:
        """
        Remove matches of one cell and its reverse matches in all sessions by providing a primary key dict.

        Args:
            key: Primary keys, need to identify the reference cell
        """

        raise NotImplementedError('Implementation is faulty, dont use.')

        day = (self & key).fetch('day')
        mask_id = (self & key).fetch('mask_id')

        if len(np.unique(day)) > 1:
            raise KeyError('Provided key has to specify a single session.')
        elif len(np.unique(mask_id)) > 1:
            raise KeyError('Provided key has to specify a single ROI.')
        else:
            sessions = self & f'day="{day[0]}"' & f'mask_id={mask_id[0]}'
            rev_sessions = self & f'matched_id={mask_id[0]}'

            print('These matches will be deleted:\n', sessions)
            print('These reverse matches will be deleted:\n', rev_sessions)
            response = input('Confirm? (y/n)')
            if response == 'y':
                sessions.delete()
                rev_sessions.delete()
            else:
                print('Aborted.')
                return

    def remove_match(self) -> None:
        """
        Remove matches of one cell and its reverse matches in all sessions with a restricting query.

        Args:
            key: Primary keys, need to identify the reference cell
        """

        entries = self.fetch()

        response = input(f'Found {len(entries)} entries to remove in queried session. Delete? [y/n]')

        if response in ['yes', 'y']:
            delete_entries = []

            for entry in entries:

                if len(self & entry) == 1:
                    delete_entries.append((self & entry))
                else:
                    print(f'Entry {entry} yielded not 1, but {len(self & entry)} entries. '
                          f'Cannot delete.')

                # Construct PKs of reverse match, if the match as not a "no-match"
                if entry['matched_id'] != -1:
                    rev_entry = self.string2key(entry['matched_session'])
                    rev_entry['username'] = entry['username']
                    rev_entry['mouse_id'] = entry['mouse_id']
                    rev_entry['match_param_id'] = entry['match_param_id']
                    rev_entry['mask_id'] = entry['matched_id']
                    rev_entry['matched_session'] = self.key2title(entry)

                    if len(self & rev_entry) == 1:
                        delete_entries.append((self & rev_entry))
                    else:
                        print(f'Reverse entry {rev_entry} yielded not 1, but {len(self & rev_entry)} entries. '
                              f'Cannot delete.')
                else:
                    print('ROI noted as "no-match", skipping reverse entry.')

            # Delete found entries in one transaction to avoid partly removing entries
            connection = self.connection
            with connection.transaction:
                for del_entry in delete_entries:
                    del_entry.delete_quick()

            print('Delete successful!')

        else:
            print('Delete aborted.')

    def string2key(self, title: Optional[Union[str, List[str]]] = None) -> Union[dict, List[dict]]:
        """
        Create a DataJoint-queryable dict from a matched_session ID string produced by the cell matching GUI.
        If multiple entries are queried or multiple titles are provided, return a list of dicts.

        Args:
            title: Matched session ID string(s) as in MatchedIndex(). Structure: YYYY-MM-DD_sessionnum_motionid_caimanid.

        Returns:
            Primary key dict for DataJoint queries.
        """

        # If no title string is provided, it is queried from the (restricted) MatchedIndex() table, and dicts with ALL
        # primary keys are constructed
        if title is None:
            title, usernames, mouse_ids = self.fetch('matched_session', 'username', 'mouse_id')

            new_keys = []
            for tit, mouse_id, username in zip(title, mouse_ids, usernames):
                keys = tit.split("_")
                new_keys.append(dict(username=username, mouse_id=mouse_id, day=keys[0], session_num=int(keys[1]),
                                     motion_id=int(keys[2]), caiman_id=int(keys[3])))

        # Otherwise, test if the title is a string, and iterate over all provided titles
        else:
            if type(title) == str:
                title = [title]

            new_keys = []
            for tit in title:
                keys = tit.split("_")
                new_keys.append(dict(day=keys[0], session_num=int(keys[1]), motion_id=int(keys[2]),
                                     caiman_id=int(keys[3])))

        # If only one dict was created, return a dict, otherwise the list
        if len(new_keys) == 1:
            new_keys = new_keys[0]
        return new_keys

    def key2title(self, key: Optional[dict]) -> Optional[str]:
        """
        Create a title string for a session entry in common_img.Segmentation() with all its primary keys, separated by "_".

        Args:
            key: Primary keys of a session, can be None (if a session is not loaded)

        Returns:
            Primary keys (day, session_num, motion_id, caiman_id) separated by "_". Returns None if key is None.
        """
        if key:
            return "{}_{}_{}_{}".format(key['day'], key['session_num'], key['motion_id'], key['caiman_id'])
        else:
            return None

    def construct_matrix(self, start_with_ref: Optional[bool] = False) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Construct matchID matrices for the queried matches. Each network gets one matrix, which has the imaged days as
        columns and the unique neurons as rows. Matrices are returned as a dict, with 'mouseID_sessionNum' as keys.

        Returns:
            Dict of matrices (pandas DataFrames) containing the matched IDs of unique neurons.
        """
        # Test
        matrices = {}  # Each individual network is stored as one entry in the dict

        # Find primary keys of individual networks (unique entries in common_img.Segmentation, except the date)
        unique_sessions = pd.DataFrame((common_img.Segmentation & self).fetch('KEY'))
        unique_network = unique_sessions.drop(columns=['day', 'motion_id', 'caiman_id']).drop_duplicates()

        if len(unique_network) == 0:
            print('Current query yielded no networks with matched ROIs.')
            return

        for _, net in unique_network.iterrows():

            # Query the current network and all sessions where matches of this network are on record
            # Exclude sessions that should be ignored for matching
            ignore_entrys = IgnoreMatchedSession & dict(net)  # reference sessions
            ignore_pks = (common_img.Segmentation & ignore_entrys).fetch('KEY', as_dict=True)   # Get all combinations of parameter IDs for these sessions
            ignore_keys = [self.key2title(entry) for entry in ignore_pks]  # target sessions
            # Ignored sessions should not appear as reference or target sessions
            curr_net = MatchedIndex - ignore_entrys & dict(net)
            if len(ignore_keys) > 0:
                curr_net = curr_net & f'matched_session not in {helper.in_query(ignore_keys)}'

            # Fetch all matches from this mouse/network
            matches = curr_net.fetch(as_dict=True)
            error_count = 0

            if start_with_ref:
                # Figure out the main reference session (day of reverse=0)
                ref_days, counts = np.unique((curr_net & 'reverse=0').fetch('day'), return_counts=True)
                ref_day = ref_days[np.argmax(counts)]

                # Define a custom sorting key function to sort by the ref_day
                def custom_sort_key(item):
                    if item["day"] == ref_day:
                        return 0, item["day"]
                    else:
                        return 1, item["day"]

                # Sort the list using the custom sorting key
                matches = sorted(matches, key=custom_sort_key)

            # Construct a match_id 2D DataFrame:
            # Columns: sessions (dates); Rows: distinct neurons; Elements: mask_id of that neuron on that day

            # Construct matrix DataFrame with first entry
            matched_sessions = np.unique(curr_net.fetch('matched_session'))
            first_entry = {k: [np.nan] for k in matched_sessions}
            first_entry[self.key2title(matches[0])] = [matches[0]['mask_id']]
            first_entry[matches[0]['matched_session']] = [matches[0]['matched_id']]
            match_matrix = pd.DataFrame(data=first_entry)

            for match in matches[1:]:

                # Check if the current matched neuron already has an entry in the matrix (mask or matched_id exists)
                mask_exists = np.where(match_matrix[self.key2title(match)] == match['mask_id'])[0]
                if match['matched_id'] != -1:
                    match_exists = np.where(match_matrix[match['matched_session']] == match['matched_id'])[0]
                else:
                    # If the matched ID is -1 (no match), then we set it to "not exist" because its not a real ID
                    match_exists = []

                if len(mask_exists) > 1 or len(match_exists) > 1:
                    # If an ID occurs more than once in one session, this is an error (mask_ids should be unique)
                    raise ValueError(f'Duplicate entry found for {match}.')

                elif len(mask_exists) != len(match_exists):
                    # If a neuron already exists in only one of the sessions, then this is a new match of a neuron that
                    # already has a match in the matrix. The new match has to be inserted in the same row.
                    if len(mask_exists) == 1:
                        row_idx = mask_exists[0]
                        new_match_sess = match['matched_session']
                        new_match_id = match['matched_id']
                    else:
                        row_idx = match_exists[0]
                        new_match_sess = self.key2title(match)
                        new_match_id = match['mask_id']

                    # Safety check that we are overwriting NaN or the same ID, not an existing different entry
                    curr_id = match_matrix[new_match_sess].iloc[row_idx]
                    if not np.isnan(curr_id) and curr_id != new_match_id:
                        print(f'Error for {match}:\n\tEntry in col {new_match_sess}, row '
                              f'{row_idx} should be empty, but is {curr_id}.')
                        error_count += 1
                        print('Error count', error_count)
                    # If the safety check has been passed, we can enter the new ID into the row
                    else:
                        with pd.option_context('mode.chained_assignment', None):
                            match_matrix[new_match_sess].iloc[row_idx] = new_match_id

                elif len(mask_exists) == 1:
                    # Both mask_id and matched_id already exist in their sessions (if mask_exists has one entry, then
                    # match_exists also has to have one entry, otherwise the first elif statement would have been true)
                    if mask_exists[0] == match_exists[0]:
                        # If they both exist in the same row, then the current match is the reverse of an existing
                        # match, and we can skip it.
                        continue
                    else:
                        # Otherwise, we have a mismatch (the ROI is matched to two different neurons)
                        print(f'Error for {match}:\n\tMask_ID exists in col {match["matched_session"]}, row '
                              f'{mask_exists[0]}, but Matched_ID exists in col {match["matched_session"]}, '
                              f'row {match_exists[0]}!')
                        error_count += 1
                        print('Error count', error_count)

                else:
                    # The only other option is that this is a completely new cell and needs a new row in the DataFrame
                    # new_row = {k: [np.nan] for k in queried_sessions}
                    new_row = {k: [np.nan] for k in matched_sessions}
                    new_row[self.key2title(match)] = [match['mask_id']]
                    new_row[match['matched_session']] = [match['matched_id']]
                    match_matrix = pd.concat([match_matrix, pd.DataFrame(data=new_row)], ignore_index=True)

            # After gathering all matches, remove sessions that were not queried (has to be done after constructing the
            # matrix to allow filtering out reference sessions (which are necessary to establish neuron identity)
            unique_sessions_filt = pd.DataFrame((common_img.Segmentation - ignore_entrys & self).fetch('KEY')).sort_values(['day', 'session_num'])
            unique_session_ids = [self.key2title(dict(x)) for row_id, x in unique_sessions_filt.iterrows()]
            match_matrix_restrict = match_matrix[unique_session_ids]

            # Remove cells with incomplete matches (rows containing NaN)
            match_matrix_restrict = match_matrix_restrict.dropna(axis='index')

            # Make unique string as a key for the current network (assume that each session only is matched with
            # one motion_id, caiman_id and match_param_id)
            net_id = f'{net["mouse_id"]}_{net["session_num"]}'
            matrices[net_id] = match_matrix_restrict

            # ## Manually resolve mismatch
            # id1 = 1390
            # id2 = 89
            # sess1 = "2022-08-27_1_1_0"
            # sess2 = "2020-09-20_1_1_0"
            # match_id = 537
            # common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=110' & f'day="2022-08-09"' & f'mask_id={id1}'
            # common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=41' & f'day="2020-09-08"' & f'mask_id={id2}'
            #
            # common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=110' & f'day="2022-08-07"' \
            #     & f'mask_id={match_id}' & f'matched_session="{sess2}"'
            # common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=41' & f'matched_session="{sess1}"' \
            #     & f'matched_id={match_id}'
            #
            # key = dict(username='hheise', mouse_id=41, day="2020-08-21", mask_id=348, matched_session="2020-08-18_1_1_0",
            #            session_num=1, motion_id=1, caiman_id=0, match_param_id=1, matched_id=1239)
            # # Update the reverse entry with the new ID
            # common_match.MatchedIndex().update1(key)
            # # Delete the wrong entry from the main reference session, then use GUI to find last missing match (that neuron should be matchable again)
            # (common_match.MatchedIndex & 'day="2020-09-08"' & f'mask_id=134' & f'matched_session="2020-08-27_1_1_0"').delete()

        return matrices

    def insert_matches_from_matrix(self, mat: pd.DataFrame, username: str, mouse_id: int,
                                   match_param_id: int = 1) -> None:
        """
        Insert all possible combinations of matches from a matrix returned by common_match.MatchedIndex.construct_matrix().
        Should only be used when construct_matrix() does not return any errors or warnings, meaning that the match logic
        integrity is intact. Useful to enable loading matches in the GUI from a different reference session.
        All entries created through this function are marked as reverse=1 to show that these are
        automatically, technically not user-confirmed matches.

        Args:
            mat: Match matrix, output from common_match.MatchedIndex().construct_matrix().
            username: Username of the query used for the construct_matrix() call.
            mouse_id: Mouse_ID of the query used for the construct_matrix() call.
            match_param_id: Match_param_ID used for the matches. Defaults to 1, commonly used parameter set.
        """

        for idx, row in mat.iterrows():

            # Get all permutations of the Series elements (every combination of all elements)
            perm = list(itertools.permutations(row.items(),
                                               2))  # row.items() returns a tuple with index + value of each element

            for ref, tar in perm:
                if ref[1] != -1:
                    ref_key = self.string2key(ref[0])
                    ref_key['username'] = username
                    ref_key['mouse_id'] = mouse_id
                    ref_key['mask_id'] = ref[1]
                    ref_key['matched_session'] = tar[0]
                    ref_key['matched_id'] = tar[1]
                    ref_key['match_param_id'] = match_param_id

                    # Set reverse to 1 for all these matches to show that these are auto-generated
                    self.helper_insert1(ref_key, insert_reverse_entries=False, force_reverse_val=1)

    def get_matched_data(self, table: dj.Table, attribute: str, trial_avg: bool = True, verbose: bool = False,
                         extra_restriction: Optional[dict] = None, return_array: bool = True, relative_dates: bool = True,
                         surgery: str = 'Microsphere injection'):
        """
        Get analysis data of matched cells in one ordered matrix (sessions in columns, cells in rows). Chosen table and
        attribute have to have exactly one entry for each matched cell.
        If each element of the matrix has the same number of data points (sessions have same length, binned data, etc.),
        The matrix can be returned as an n-dimensional numpy array, which might be more handy for further analysis.

        Args:
            table: DataJoint table where the data should be queried from. Must have exactly one entry per matched cell.
            attribute: Attribute of the DataJoint table that should be fetched.
            trial_avg: For tables that store trial-wise data (e.g. hheise_placecell), data can be averaged across trials
                before returning.
            verbose: Bool flag whether to shop print outputs.
            extra_restriction: Dict of additional restriction parameters when selecting cells.
            return_array: Bool flag whether to return a n-D numpy array instead of a DataFrame. Only works if every cell
                element has the same number of datapoints in every session.
            relative_dates: Bool flag whether relative dates of sessions should be returned.
            surgery: Name of the surgery the dates should be relative to.

        Returns:
            pd.DataFrames (one per network) with the queried matched data, or n-D numpy array if return_array=True.
        """

        def dataframe2array(df) -> np.ndarray:
            # Get length of non-NaN entries to determine array shape
            mask = np.where(~pd.isna(df))
            try:
                lengths = np.array([len(df.iloc[x, y]) for x, y in zip(mask[0], mask[1])])
                third_axis = True
            except TypeError:
                # If the item is not a list (no len()), check if all are a single-value variable
                lengths = np.array([isinstance(df.iloc[x, y], (int, float, np.integer, np.floating, bool))
                                    for x, y in zip(mask[0], mask[1])])
                third_axis = False

            if np.all(lengths == lengths[0]):
                # Initialize NaN array (non-matched sessions remain NaN)
                if third_axis:
                    arr = np.zeros((network.shape[0], network.shape[1], lengths[0])) * np.nan
                else:
                    arr = np.zeros((network.shape[0], network.shape[1])) * np.nan

                # Iterate through the DataFrame and set each row in the array
                for x, y in zip(mask[0], mask[1]):
                    arr[x, y] = df.iloc[x, y]
                return arr
            else:
                print(f'IndexError: Error in network {net_id}: Not all elements have the same length, cannot return '
                      f'array. Returning DataFrame instead.')
                return df

        # import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt

        # from schema import hheise_placecell, common_match, common_img
        #
        # table = hheise_placecell.BinnedActivity.ROI
        # attribute = 'bin_spikerate'
        #
        # self = common_match.MatchedIndex & 'username="hheise"' & 'mouse_id=110' & 'day<"2022-08-15"'

        # Get the matching ID matrices for each animal/network
        if self.fetch('mouse_id')[0] == 63:
            start_with_ref = True
        else:
            start_with_ref = False
        matrix = self.construct_matrix(start_with_ref=start_with_ref)

        queried_usernames = np.unique(self.fetch('username'))
        if len(queried_usernames) > 1:
            raise IndexError('More than 1 username queried. Only query mice from one user.')
        else:
            username = queried_usernames[0]

        # For each network, go through all matched IDs and get the requested data, in another matrix of the same shape
        data = {}
        for net_id, network in matrix.items():

            if verbose:
                print(f'Getting data for matched network {net_id}...')
            # Initialize array that will store the data. It has to be typecast to "object" to accept arrays as elements
            curr_data = network.copy()
            curr_data = curr_data.astype('object')

            # Find the primary keys of all matched ROIs
            coord_list = []
            roi_key_list = []
            for row_idx, cell in network.iterrows():
                # print(f'\t...fetching cell {row_idx+1}/{len(network)} ({((row_idx)/network.shape[0])*100:.2f}%)', end='\r')
                for date_id, match_id in cell.items():
                    if match_id == -1:
                        # Enter NaN for no-matched sessions
                        curr_data.at[row_idx, date_id] = np.nan
                    else:
                        # Build restriction dict for each matched cell
                        roi_keys = dict(username=username, mouse_id=int(net_id.split("_")[0]),
                                        day=pd.to_datetime(date_id.split('_')[0]).date(),
                                        session_num=int(net_id.split("_")[1]), motion_id=int(date_id.split('_')[2]),
                                        caiman_id=int(date_id.split('_')[3]), mask_id=int(match_id))
                        if extra_restriction is not None:
                            roi_keys = dict(**roi_keys, **extra_restriction)
                        coord_list.append((row_idx, date_id))
                        roi_key_list.append(roi_keys)

            ## Query the data from these ROIs, some tables have to be handled differently
            if table == hheise_placecell.PlaceCell.ROI:
                # Only place cells are in PlaceCell.ROI, so a cell might not exist in the table, then its
                # for sure not a place cell
                out = [(table & roi).fetch1('KEY', attribute) if len(table & roi) > 0
                       else (roi, 0) for roi in roi_key_list]
                roi_key_order, cell_data = zip(*out)

            elif table == hheise_placecell.PlaceCell.PlaceField:

                # Set all entries of cells that were found to an empty list by default, to catch cells that were found
                # but are not PCs. These cells would be missing from the following query, and their ID would not be
                # replaced in curr_data. However, since they are not accepted place cells, they should have no place
                # fields, so set it to an empty list by default.
                curr_data = curr_data.applymap(lambda x: [] if not pd.isna(x) else x)

                # Fetch only fully accepted place fields from accepted place cells, np.nan if a cell is not a place cell
                res = dict(large_enough=1, strong_enough=1, transients=1, is_place_cell=1)
                out = [(table * hheise_placecell.PlaceCell.ROI & roi & res).fetch('KEY', attribute)
                       if len(table * hheise_placecell.PlaceCell.ROI & roi & res) > 0 else (roi, []) for roi in roi_key_list]
                roi_key_order, cell_data = zip(*out)

                # Transform to list to allow combining entries
                roi_key_order = list(roi_key_order)
                cell_data = list(cell_data)

                # Merge cells with multiple place fields, get list of place fields per cell
                for cell_idx in range(len(cell_data)):
                    if type(cell_data[cell_idx]) == np.ndarray:
                        cell_data[cell_idx] = list(cell_data[cell_idx])
                        roi_key_order[cell_idx] = roi_key_order[cell_idx][0]

                # Delete place_field_id key from dict to enable later comparison with roi_key_list (one dict per cell,
                # not place field, so place_field_id is useless here anyway)
                roi_key_order = [{k: v for k, v in key.items() if k != 'place_field_id'} for key in roi_key_order]

            elif table == hheise_placecell.BinnedActivity.ROI:
                # Data can be returned as single trials or trial-averaged
                if trial_avg:
                    roi_key_order, cell_data = (table & roi_key_list).get_normal_act(trace=attribute, return_pks=True)
                else:
                    roi_key_order, cell_data = (table & roi_key_list).fetch('KEY', attribute)

            else:
                roi_key_order, cell_data = (table & roi_key_list).fetch('KEY', attribute)

            # Check if there was data for each day in the tables
            data_dates = np.unique([x['day'] for x in roi_key_order])
            if len(data_dates) != network.shape[1]:
                # If there is a mismatch, find which day is missing and return an error
                match_dates = np.array([pd.to_datetime(d.split('_')[0]).date() for d in network.columns])

                if len(data_dates) > len(match_dates):
                    longer_set = set(data_dates)
                    shorter_set = set(match_dates)
                    shorter_arr = 'the MatchingIndex table'
                else:
                    longer_set = set(match_dates)
                    shorter_set = set(data_dates)
                    shorter_arr = 'the provided data table'

                missing_dates = longer_set - shorter_set
                if len(missing_dates) > 0:
                    raise IndexError(f'The following dates are missing from {shorter_arr}: {missing_dates}')

            # Sort the data into the correct location of the array
            for curr_roi, curr_cell_data in zip(roi_key_order, cell_data):

                # Find index of the current ROI in the ROI list with this generator expression
                roi_key_idx = next((i for i, item in enumerate(roi_key_list) if curr_roi.items() <= item.items()), None)

                if roi_key_idx is None:
                    raise ValueError(f'ROI dict {curr_roi} could not be found in roi_key_list. Something went wrong.')
                else:
                    curr_data.at[coord_list[roi_key_idx][0], coord_list[roi_key_idx][1]] = curr_cell_data

            # print(f'\t...fetching cell {row_idx+1}/{len(network)} ({((row_idx+1)/network.shape[0])*100:.2f}%)')
            # print(f'Done!')
            # Transform dates into relative dates to a surgery of that mouse
            if relative_dates:
                mouse = net_id.split("_")[0]
                microsphere_date = (common_mice.Surgery() & f'surgery_type="{surgery}"' &
                                    f'mouse_id={mouse}').fetch('surgery_date')
                if len(microsphere_date) > 1:
                    raise IndexError(f'More than one microsphere injection on record for mouse {mouse}. Not defined '
                                     f'how to handle this.')
                elif len(microsphere_date) == 0:
                    print(f'No surgery of type {surgery} on record for mouse {mouse}, but relative_dates==True. Setting'
                          f'relative_dates to False now, returning absolute dates.')
                    relative_dates = False
                microsphere_date = microsphere_date[0].date()

                # For each column, transform session identifier into key dict, transform the day into a datetime
                # object, subtract the microsphere date from it, and take the difference in days
                rel_dates = [(datetime.strptime(self.string2key(date)['day'], '%Y-%m-%d').date() -
                              microsphere_date).days for date in curr_data.columns]

            # If it was not requested, use session ID strings instead
            else:
                rel_dates = curr_data.columns

            # Transform DataFrame into array. If DataFrame cant be coerced into array, an error is printed and the same
            # DataFrame is returned.
            if return_array:
                curr_data = dataframe2array(curr_data)
            else:
                curr_data.columns = rel_dates

            data[net_id] = [curr_data, rel_dates]

        return data

    @staticmethod
    def load_matched_data(match_queries: Iterable, data_type: str, as_array: bool = False, relative_dates: bool = True) \
            -> Dict[int, Union[pd.DataFrame, list]]:
        """
        Wrapper function to load cell-matched data for multiple mice (one query per mouse).

        Args:
            match_queries: List of MatchedIndex queries. One query per mouse.
            data_type: Name of data type/table that should be loaded
            as_array: Bool flag whether to return a n-D numpy array instead of a DataFrame. Only works if every cell
                element has the same number of datapoints in every session. Otherwise, DataFrame will be returned.

        Returns:
            Dict of data, one key per mouse/network.
                If return_arrays=False and relative_dates=True, the DataFrames have relative days as column names and
                list with relative days is not explicitly returned.
        """

        data = []
        if data_type == "is_place_cell":
            print(f'Fetching place cell data...')
            for query in match_queries:
                data.append(query.get_matched_data(table=hheise_placecell.PlaceCell.ROI, attribute='is_place_cell',
                                                   extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                                   return_array=as_array, relative_dates=relative_dates,
                                                   surgery='Microsphere injection'))

        elif data_type == "place_field_idx":
            print(f'Fetching place field indices...')
            for query in match_queries:
                data.append(query.get_matched_data(table=hheise_placecell.PlaceCell.PlaceField, attribute='bin_idx',
                                                   extra_restriction=dict(corridor_type=0, place_cell_id=2,
                                                                          large_enough=1, strong_enough=1, transients=1),
                                                   return_array=as_array, relative_dates=relative_dates,
                                                   surgery='Microsphere injection'))
        elif data_type == "place_field_com":
            print(f'Fetching place field center-of-mass...')
            for query in match_queries:
                data.append(query.get_matched_data(table=hheise_placecell.PlaceCell.PlaceField, attribute='com',
                                                   extra_restriction=dict(corridor_type=0, place_cell_id=2,
                                                                          large_enough=1, strong_enough=1, transients=1),
                                                   return_array=as_array, relative_dates=relative_dates,
                                                   surgery='Microsphere injection'))
        elif data_type == "place_field_sd":
            print(f'Fetching place field center-of-mass standard deviations...')
            for query in match_queries:
                data.append(query.get_matched_data(table=hheise_placecell.PlaceCell.PlaceField, attribute='com_sd',
                                                   extra_restriction=dict(corridor_type=0, place_cell_id=2,
                                                                          large_enough=1, strong_enough=1, transients=1),
                                                   return_array=as_array, relative_dates=relative_dates,
                                                   surgery='Microsphere injection'))
        elif data_type == "bin_spikerate":
            print(f'Fetching spatial activity maps based on spike rate...')
            for query in match_queries:
                data.append(
                    query.get_matched_data(table=hheise_placecell.BinnedActivity.ROI, attribute='bin_spikerate',
                                           extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                           return_array=as_array, relative_dates=relative_dates,
                                           surgery='Microsphere injection'))
        elif data_type == "bin_activity":
            print(f'Fetching spatial activity maps based on dF/F...')
            for query in match_queries:
                data.append(
                    query.get_matched_data(table=hheise_placecell.BinnedActivity.ROI, attribute='bin_activity',
                                           extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                           return_array=as_array, relative_dates=relative_dates,
                                           surgery='Microsphere injection'))
        elif data_type == "dff":
            print(f'Fetching dF/F traces...')
            for query in match_queries:
                data.append(query.get_matched_data(table=common_img.Segmentation.ROI, attribute='dff',
                                                   extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                                   return_array=as_array, relative_dates=relative_dates,
                                                   surgery='Microsphere injection'))
        elif data_type == "decon":
            print(f'Fetching deconvolved traces...')
            for query in match_queries:
                data.append(query.get_matched_data(table=common_img.Deconvolution.ROI, attribute='decon',
                                                   extra_restriction=dict(corridor_type=0, place_cell_id=2,
                                                                          decon_id=1),
                                                   return_array=as_array, relative_dates=relative_dates,
                                                   surgery='Microsphere injection'))
        elif data_type == "com":
            print(f'Fetching cell coordinates...')
            for query in match_queries:
                data.append(query.get_matched_data(table=common_img.Segmentation.ROI, attribute='com',
                                                   extra_restriction=dict(corridor_type=0, place_cell_id=2),
                                                   return_array=as_array, relative_dates=relative_dates,
                                                   surgery='Microsphere injection'))

        data_dict = {
            int(k.split('_')[0]): (v[0] if (type(v[0]) == pd.DataFrame and v[0].columns.dtype == 'int64') else v)
            for item in data for k, v in item.items()}
        return data_dict
