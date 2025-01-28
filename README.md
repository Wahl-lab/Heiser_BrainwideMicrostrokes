# Brain-wide microstrokes affect the stability of memory circuits in the hippocampus - Heiser et al., 2025
## Introduction
This document shall serve as a manual for the code used to process and analyze data for the manuscript titled “Brain-wide microstrokes affect the stability of memory circuits in the hippocampus”. It outlines the functionality and order of execution of the enclosed Python scripts, as well as the figure panels which this data contributed to.

The research data management and analysis was mostly performed with DataJoint (Yatsenko et al., 2015), a MySQL-based database environment for research data management and analysis pipelines. Here, analyzed data is stored in tables, which include the code that was used to produce this data in the `make()` function of that table. Each table corresponds to one processing step within the analysis pipeline. For more information about DataJoint, please see the official documentation: https://datajoint.com/docs/core/datajoint-python/0.14/.

Thus, most scripts do not rely on files for accessing the experimental and analysis data, but are based on DataJoint’s syntax and rely on the locally hosted MySQL database to load and store the data. The provided scripts contain the tables and associated, documented code. The tables which are part of the core analysis pipeline are described in this document in more detail. To further facilitate the code review and increase comprehensibility, exemplary files containing relevant output data of intermediate analysis steps are enclosed to functions when applicable. Some exemplary files are too large for Github and are hosted on [this Google Drive](https://drive.google.com/drive/folders/1ab3ikKsryVG2jC4eWNAiDETfdJxLhd-J?usp=sharing), with an identical folder structure. The respective external files are labeled in the following document.

## System requirements
- Main software dependencies and versions:
  - see [`requirements.txt`](/requirements.txt) file for main codebase
  - see [`requirements_caiman.txt`](/requirements_caiman.txt) file for CaImAn-specific scripts
 
## Installation and demo
As mentioned in the introduction, scripts relying on DataJoint are not executable in the current form without direct access to the local database. Instead, great effort has been made to document and annotate the code as much as possible, and provide example input and output data wherever possible. Scripts written by Filippo Kiessler and Prof. Julijana Gjorgjieva should be executable, with all necessary data being provided within the [`functional_connectivity/Kiessler_Gjorgjieva/data`](/functional_connectivity/Kiessler_Gjorgjieva/data) directory. Small adjustments to local paths referencing these data files may be necessary when executing the scripts. Please refer to the [Readme file](/functional_connectivity/Kiessler_Gjorgjieva/README.txt) for a more detailed description of each script.

## Structure of provided data
- If not stated otherwise, exemplary data files are from the same single session: mouse 41, recorded on 2020-08-24, one day before microsphere injection (“healthy” period)
- Calcium imaging files from two-photon microscope are provided as TIF files (readable with ImageJ)
- Dictionaries are provided as txt files in JSON format
- 1D and 2D numpy arrays are provided as CSV, >2D arrays as NPY files
- Images are provided as TIF file
- Files, folders and functions are marked in `this style`

## 1. Processing of VR-based spatial navigation task behavior data
Directory: [`vr_behavior`](/vr_behavior)

Associated script: [`hheise_behav.py`](/vr_behavior/hheise_behav.py)

### Core analysis functions
- `VRSession().make()`
  - Input
    - Behavioral files (3 per trial; one trial = one corridor traversal) produced by LabView program enclosed
      - `Encoder[…].txt` (rotary encoder)
      - `TCP[…].txt` (VR position in arbitrary units)
      - `TDT[…].txt` (Lick valve (col 1), frame acquisition (col 2))
  - Analysis
    - Align behavioral data streams to a common timeline
    - Convert rotary encoder ticks into running speed
    - Resample data to 125 Hz
  - Output
    - Combined, aligned and resampled behavior data stream
      - `behavior.csv` enclosed
- `VRPerformance().make()`
  - Input
    - Aligned behavioral data stream
  - Analysis
    - Spatially bin licking
    - Compute histogram of trial-wise spatial lick distributions
    - Compute spatial information of session-wise histogram
  - Output
    - Lick histogram
      - `lick_histogram.csv` enclosed
    - Spatial Information value
      - `licking_si.txt` enclosed
## 2. Experimental stages & groups
Directory: [`experimental_groups`](/experimental_groups)

Associated script: [`hheise_grouping.py`](/experimental_groups/hheise_grouping.py)

### Core analysis functions
- BehaviorGrouping().make()
  - Input
    - VR performance normalized to the average performance of the last three sessions before microsphere injection
  - Analysis
    - Group animals based on the average relative performance during early and late post-stroke phases
  - Output
    - Each animal assigned its coarse and fine group
      - `animal_groups.csv` enclosed
## 3. Preprocessing of two-photon calcium imaging data
Directory: [`2p_imaging`](/2p_imaging)

Associated scripts: [`common_img.py`](/2p_imaging/common_img.py), [`motion_correction.py`](/2p_imaging/motion_correction.py)

### External software used
- CaImAn (Giovannucci et al., 2019) used for motion correction, segmentation and ROI evaluation
- CASCADE (Rupprecht et al., 2021) used to predict deconvolved spike probabilities from ΔF/F traces
### Core analysis functions (`common_img.py`)
- `MotionCorrection().make()`
  - Input
    - TIF files from two-photon microscope, one per trial
      - TIF files of 5 trials enclosed (`file_00001.tif`, `file_00002.tif`, `file_00003.tif`, `file_00004.tif`, `file_00005.tif`; in Google Drive)
    - Dict of CaImAn motion correction parameters
      - `motion_params.txt` enclosed
  - Analysis
    - Preprocessing of movies
      - Crop edges of movies by 12 px to remove scanning artifacts
      - Apply offset to pixel values to avoid negative pixel values
    - Motion correction (CaImAn)
    - Quality check of motion correction (correlation of motion-corrected frames with template)
  - Output
    - Motion correction template
      - `mc_template.tif` enclosed
    - X- and Y-shifts to align each frame to the template
      - `mc_shifts.csv` enclosed
- `QualityControl().make()`
  - Input
    - Motion-corrected two-photon calcium imaging movie
  - Analysis
    - Compute pixel statistics of motion-corrected movie
  - Output
    - Image of average pixel intensity
      - `avg_image.tif` enclosed
    - Image of pixel-wise correlation with 8 neighboring pixels
      - `cor_image.tif` enclosed
- `Segmentation().make()`
  - Input
    - Motion-corrected two-photon calcium imaging movie
    - Dict of CaImAn segmentation parameters
      - `segmentation_params.txt` enclosed
  - Analysis
    - CaImAn segmentation (using constrained non-negative matrix factorization)
    - Evaluation of extracted ROIs (accept/reject as neuronal)
    - Detrend calcium trace to ΔF/F
  - Output
    - Spatial and temporal background components
      - `spat_bg.npy` and `temp_bg.npy` enclosed
    - ROI map
      - `roi_map.npy` enclosed
    - ROI features (center of mass, SNR, R, CNN, ΔF/F percentile)
      - `roi_features.csv` enclosed
    - ΔF/F traces per ROI
      - `dff.npy` enclosed (Google Drive)
- `Deconvolution().make()`
  - Input
    - ΔF/F traces from `Segmentation()` table
    - Name of CASCADE model to use for spike prediction
  - Analysis
    - Use CASCADE to predict spike probabilities from ΔF/F traces
  - Output
    - Deconvolved spike probability traces per ROI
      - `decon.npy` enclosed (Google Drive)
    - Noise level per ROI, as determined by CASCADE
      - `noise_lvl.csv` enclosed

## Single cell tracking
Directory: [`single_cell_tracking`](/single_cell_tracking)

Associated scripts: [`common_match.py`](/single_cell_tracking/common_match.py)

### Core analysis functions
- `FieldOfViewShift().make()`
  - Input
    - Average intensity projection of the FOV of two sessions that should be matched
  - Analysis
    - Split FOV into 4 quadrants
    - Estimate the translation shift with phase cross correlation (scikit-image package) of each quadrant to allow for non-rigid shifts
    - Upscale quadrant shifts to FOV size and get pixel-wise shifts via spline interpolation (SciPy package)
  - Output
    - X and Y shifts between the two FOVs
      - `fov_shifts.npy` enclosed (shift between session recorded on 2020-08-18 and 2020-08-24)
- `MatchedIndex().make()`
  - Input
    - Indices of ROIs from two sessions
  - Analysis
    - Match ROIs based on distance
    - Manually confirm match through Dash web app GUI
  - Output
    - Matrix of ROI indices that have been confirmed to be the same cell
      - `cell_matched_matrix.csv` enclosed

## Linear Corridor Analysis
Directory: [`linear_corridor_analysis`](/linear_corridor_analysis)

Associated scripts: [`hheise_placecell.py`](/linear_corridor_analysis/hheise_placecell.py), [`pc_classifier.py`](/linear_corridor_analysis/pc_classifier.py)

### Core analysis functions
- `BinnedActivity().make()`
  - Input
    - ΔF/F traces
    - Frame mask that indicates frames where animal was running
      - running_mask.csv enclosed
    - frames synchronized to 80 VR position bins (5 cm per bin)
      - aligned_frames.csv (frames per position bin for each trial) enclosed
  - Analysis
    - Exclude frames where animal was not running
    - Spatially bin ΔF/F traces by averaging running frames that were acquired in the same VR corridor bin for each trial
  - Output
    - ΔF/F traces split into trials and spatially binned
      - Binned_activity.csv enclosed (example cell 685)
- `PlaceCell().make()`
  - Place cell classification adapted from Hainmüller & Bartos (2018)
    - Input
      - Spatially binned activity (trial-wise)
        - `binned_activity.csv` enclosed
      - Isolated significant transients of ΔF/F traces
        - `transient_only.npy` enclosed (Google Drive)
      - Parameters for place cell classification
        - `place_cell_params.txt` enclosed
    - Analysis
      - See Hainmüller & Bartos (2018)
    - Output
      - Cells that passed place cell criteria, and associated place fields
        - `place_field_result.txt` enclosed (place field of cell 685)
- `SpatialInformation().make()`
  - Place cell classification adapted from Shuman et al. (2020)
    - Not used in favor of algorithm mentioned above
    - Used for algorithm to compute within-session stability
  - Input
    - Spatially binned activity (trial-wise)#
      - `binned_activity.csv` enclosed
  - Analysis
    - Compute correlation of trial-wise spatially binned activity across even vs. odd trials, first half vs. second half trials, and averaging the Fisher z-scored correlation coefficients
  - Output
    - Within-session stability score per neuron (1.9673 for cell 685)

## 6. Place Cell Stability & Transitions
Directory: [`place_cell_transitions`](/place_cell_transitions)

Associated scripts: [`place_cell_transitions.py`](/place_cell_transitions/place_cell_transitions.py), [`stable_unstable_classification.py`](/place_cell_transitions/stable_unstable_classification.py)

### Core analysis functions
- `place_cell_transitions.quantify_place_cell_transitions()`
  - Input
    - Place cell classification labels of tracked cells
      - `is_pc.pkl` enclosed
  - Analysis
    - Compute transitions between classes (place cell – noncoding cell) across days within experimental phases
    - Same procedure, but with permuted cell class assignments to produce chance level distribution
  - Output
    - Transition matrices of place cell - noncoding classes
      - `trans_matrix.csv` enclosed
- `stable_unstable_classification.classify_stability()`
  - Input
    - Spatially binned activity of tracked cells
      - `spatial_activity_maps_dff.pkl` enclosed (Google Drive)
    - Place cell classification labels of tracked cells
      - `is_pc.pkl` enclosed
  - Analysis
    - Yield baseline stability score for each network
    - Classify place cells as stable or unstable depending on their cross-session stability compared to the baseline stability
  - Output
    - Baseline stability score for each network
    - Class labels for each cell (stable, unstable, noncoding)
      - `Stable_unstable_classification.csv` enclosed
- `stable_unstable_classification.stability_sankey()`
  - Input
    - Stability classification from `classify_stability()`
  - Analysis
    - Compute transitions between classes (stable, unstable, noncoding) across experimental phases
    - Same procedure, but with permuted cell class assignments to produce chance level distribution
  - Output
    - Transition matrices for stable/unstable/noncoding classes
      - `trans_matrix_stable.csv` enclosed

## 7. Population vector correlation
PVC algorithm adapted from Shuman et al. (2020)

Directory: [`pvc`](/pvc)

Associated scripts: [`plot_pvc.py`](/pvc/plot_pvc.py), [`hheise_pvc.py`](/pvc/hheise_pvc.py)

### Core analysis functions
- `plot_pvc.py`
  - Input
    - Spatially binned activity of tracked cells
      - `spatial_activity_maps_dff.pkl` enclosed (see place_cell_transitions)
  - Analysis
    - Plot PVC matrices (see Fig. 4A)
  - Output
    - PVC matrices across experimental phases per animal
      - Matrices of animals 41 (No Recovery), 90 (Recovery) and 91 (Sham) enclosed
- `hheise_pvc.PvcCrossSessionEval().make()`
  - Input
    - PVC half-matrices from `PvcCrossSession()` per animal and session
  - Analysis
    - Compute curve from matrix by averaging across positions
    - Evaluate curve based on multiple features
  - Output
    - Features shown in manuscript: max_pvc (y-intercept, Fig. 4B) & min_slope (Fig. 4C)
    - Values for example session (M41, 2020-08-21 correlated with 2020-08-24): max_pvc = 0.598; min_slope = 1.45
## 8. Bayesian decoder
Decoder algorithm adapted from Shuman et al. (2020)

Directory: [`bayesian_decoder`](/bayesian_decoder)

Associated scripts: [`hheise_decoder.py`](/bayesian_decoder/hheise_decoder.py), [`bayesian_decoder.py`](/bayesian_decoder/bayesian_decoder.py)

### Core analysis functions
- `BayesianDecoderWithinSession.make()`
  - Input
    - ΔF/F traces of all neurons
    - Decoder parameters
      - `decoder_params.txt` enclosed
  - Algorithm
    - See Shuman et al. (2020)
    - Decoder trained and tested on a single session (short-term decoder)
  - Output
    - Frame-wise predicted animal position for each trial
      - `within_position_predict.csv` and `within_position_true.csv` enclosed (array shape: n_frames x n_trials)
      - Within-session decoder performance quantification, averaged over repetitions of leave-one-out validation
      - Features shown in manuscript: accuracy (Fig. 3D) & reward-zone sensitivity (Fig. 3E)
      - Values for example session: accuracy=21.35%, sensitivity=87.23%
- `BayesianDecoderLastPrestroke.make()`
  - Input
    - ΔF/F traces of all neurons
    - Decoder parameters
      - decoder_params.txt enclosed
  - Analysis
    - See Shuman et al. (2020)
    - Decoder trained on the last prestroke session and tested on all other sessions (long-term decoder)
  - Output
    - Frame-wise predicted animal position across tested session
      - `cross_session_decoder.csv` (first column: true position; second column: predicted position)
      - Cross-session decoder performance quantification, averaged over sessions within each experimental phase
      - Features shown in manuscript: accuracy (Fig. 3D) & reward-zone sensitivity (Fig. 3E)
      - Values for example session (2020-08-24) correlated with another session (2020-08-27): accuracy=2.36%, sensitivity=51.80%

## 9. Functional connectivity
Directory: [`functional_connectivity`](/functional_connectivity)

Associated scripts: [`hheise_connectivity.py`](/functional_connectivity/hheise_connectivity.py)

### Core analysis functions
- `NeuronNeuronCorrelation().make()`
  - Input
    - Unbinned ΔF/F traces of all neurons
  - Analysis
    - Compute correlation of ΔF/F traces between all neurons
  - Output
    - Cross-correlation matrix of neuron-neuron correlation coefficients
      - Matrix of example session (`cross_corr_matrix.csv`) enclosed
    - For the analysis concerning functional correlation matrices (Fig. 4E), cosine similarities (Fig. 4F, G) and network connectivity distributions (Fig. 5A, C, D) please refer to the scripts and associated README in the [`Kiessler_Gjorgjieva`](/functional_connectivity/Kiessler_Gjorgjieva) subdirectory.

## 10. Microsphere histology
Directory: [`microsphere_histology`](/microsphere_histology)

Associated scripts: [`hheise_hist.py`](/microsphere_histology/hheise_hist.py)

### Core analysis functions
- `Microsphere().make()`
  - Input
    - Microspheres and lesions manually detected and located in brain slices
      - `spheres_annotation.csv` enclosed
  - Analysis
    - Combine single-sphere datapoints into brain regions (`get_structure_groups()`)
  - Output
    - Lesion and sphere data for different brain regions
      - `microsphere_summary.csv` enclosed
