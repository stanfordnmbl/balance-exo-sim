# Allow using the osimpipeline git submodule.
import sys
sys.path.insert(1, 'code')
sys.path.insert(1, 'osimpipeline')
sys.path.insert(1, 'osimpipeline/osimpipeline')

import os
import inspect

import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
if not 'opensim_home' in config:
    raise Exception('You must define the field `opensim_home` in config.yaml '
                    'to point to the root of your OpenSim 4.0 (or later) '
                    'installation.')
sys.path.insert(1, os.path.join(config['opensim_home'], 'sdk', 'python'))

DOIT_CONFIG = {
        'verbosity': 2,
        'default_tasks': None,
        }

# Settings for plots.
import matplotlib
import matplotlib.pyplot as plt
plt.rc('font', family='Helvetica, Arial, sans-serif', size=8)
plt.rc('errorbar', capsize=1.5)
plt.rc('lines', markeredgewidth=1)
plt.rc('legend', fontsize=8)

import osimpipeline as osp

# This line is necessary for registering the tasks with python-doit.
from vital_tasks import *

# Custom tasks for this project.
from tasks import *

# Custom helper functions for this project
from helpers import *

model_fname = 'Rajagopal2015_passiveCal_hipAbdMoved_noArms_subtalar_36musc_optMaxIsoForces.osim'
generic_model_fpath = os.path.join('model', model_fname)
study = osp.Study('ankle_perturb_sim', generic_model_fpath=generic_model_fpath)

# Bump'em module locations
# ------------------------
study.module0 = osim.Vec3(2.0870, 0.9810, -0.9907)
study.module1 = osim.Vec3(0.5568, 0.9748, -2.6964)
study.module2 = osim.Vec3(-0.8451, 1.0016, -0.9612)
study.module3 = osim.Vec3(0.6447, 0.9897, 0.4308)

# Rotate the module location to match the data transformation
# performed by TaskTransformExperimentalData below
import opensim as osim
def rotateVec(vec):
    R = osim.Rotation(np.deg2rad(-90), osim.CoordinateAxis(1))
    vec_rotated = R.multiply(vec)
    return vec_rotated

study.module0 = rotateVec(study.module0)
study.module1 = rotateVec(study.module1)
study.module2 = rotateVec(study.module2)
study.module3 = rotateVec(study.module3)

# Set the treadmill walking speed for the study
study.walking_speed = 1.25

# Copy data files
# ---------------
study.add_task(TaskCopyGenericModelFilesToResults)
subjects = [1]
# The experimental data files for the first trial in this list gets duplicated
# to create the 'unperturbed' condition.
# 05: side-step (pert at heel strike, right-back)
# 08: large forward step (pert at heel strike, forward)
# 17: crossover, torso roll (pert at heel strike, back left)
# 25: trunk pitch, wide step (pert at mid-swing, back-right)
# 26: crossover step (pert at mid-swing, left)
# 28: very slight kinematic change from normal walking (Pert at toeoff, left, half previous magnitudes)
# 31: cross-over, torso roll (pert at heel strike, right)
# 39: backward cross step (Pert mid-swing, right)
# 40: cross over with slight hop in recovery (Pert at heelstrike, left-front)
# 43: almost places swing leg, then changes position (Pert at mid swing, front left)
# trials = [5, 8, 17, 25, 26, 28, 31, 39, 40, 43]
unperturbed_trial = 8
study.add_task(TaskCopyMotionCaptureData, subjects, unperturbed_trial)
study.add_task(TaskTransformExperimentalData, subjects)
study.add_task(TaskFilterAndShiftGroundReactions, subjects)
lowpass_freq = 7.0 / 1.25 # Based on Bianco et al. 2018
study.add_task(TaskExtractAndFilterEMG, subjects, 
    frequency_band=[10.0, 400.0], filter_order=4.0, lowpass_freq=lowpass_freq)
study.add_task(TaskExtractAndFilterPerturbationForces, subjects, 
    lowpass_freq=10.0)

# Model markers to compute errors for
marker_suffix = ['ASI','PSI','TH1','TH2','TH3',
                 'CAL','TOE','MT5','LKN','SH1','SH2','SH3',
                 'LAK']
error_markers = ['*' + marker for marker in marker_suffix] 
error_markers.append('CLAV')
error_markers.append('C7')
study.error_markers = error_markers

study.weights = {
    'state_tracking_weight'  : 1e3,
    'control_weight'         : 1e2,
    'grf_tracking_weight'    : 5e-1,
    'com_tracking_weight'    : 1e2,
    'base_of_support_weight' : 0,
    'head_accel_weight'      : 0,
    'upright_torso_weight'   : 0,
    'torso_tracking_weight'  : 1e1,
    'foot_tracking_weight'   : 1e1,
    'pelvis_tracking_weight' : 0, 
    'aux_deriv_weight'       : 1e-1,
    'metabolics_weight'      : 0,
    'accel_weight'           : 1e-2,
    'regularization_weight'  : 0
    }

# Results
# -------
# Add tasks for each subject
import subject01
subject01.add_to_study(study)

# Analysis
# -------=
subjects = ['subject01']
times = [30, 40, 50, 60]
cmap_indices = [0.1, 0.3, 0.5, 0.9]
delay = 0.400
colormap = 'nipy_spectral'

# study.add_task(TaskPlotCOMTrackingErrorsAnklePerturb, subjects, [40, 50, 60], 
#     colormap, cmap_indices[1:], delay)
# study.add_task(TaskPlotNormalizedImpulseAnklePerturb, subjects, times, colormap,
#     cmap_indices, delay)
# study.add_task(TaskPlotGroundReactionsAnklePerturb, subjects[0], times, colormap,
#     cmap_indices, delay)
# study.add_task(TaskPlotCOMVersusAnklePerturbTime, subjects, 100, times, colormap,
#     cmap_indices, delay)
# study.add_task(TaskPlotCOMVersusAnklePerturbTorque, subjects, 40, delay=delay)
# study.add_task(TaskPlotCOMVersusAnklePerturbTorque, subjects, 50, delay=delay)
# study.add_task(TaskPlotCOMVersusAnklePerturbTorque, subjects, 60, delay=delay)
# delays40 = np.arange(0.0, 1.1, 0.1)
# delays50 = np.arange(0.0, 1.1, 0.1)
# delays60 = np.arange(0.0, 0.8, 0.1)
# study.add_task(TaskPlotCOMVersusAnklePerturbDelay, subjects, delays40, 100, 40)
# study.add_task(TaskPlotCOMVersusAnklePerturbDelay, subjects, delays50, 100, 50)
# study.add_task(TaskPlotCOMVersusAnklePerturbDelay, subjects, delays60, 100, 60)
# study.add_task(TaskPlotCOMTrackingErrorsAnklePerturbDelay, subjects, delays40, 100, 40)
# study.add_task(TaskPlotCOMTrackingErrorsAnklePerturbDelay, subjects, delays50, 100, 50)
# study.add_task(TaskPlotCOMTrackingErrorsAnklePerturbDelay, subjects, delays60, 100, 60)

# conditions = ['walk2']
# # Experiment results
# study.add_task(TaskAggregateMuscleDataExperiment, cond_names=conditions)  
# study.add_task(TaskPlotMuscleData, study.tasks[-1])
# study.add_task(TaskAggregateMomentsExperiment, cond_names=conditions)
# study.add_task(TaskPlotMoments, study.tasks[-1])
# study.add_task(TaskAggregateReservesExperiment, cond_names=conditions)
# study.add_task(TaskComputeReserves, study.tasks[-1])

# # Validation
# # ----------
# study.add_task(TaskValidateMarkerErrors, cond_names=conditions)
# study.add_task(TaskValidateMuscleActivity, cond_names=conditions)
# study.add_task(TaskValidateKinematics, cond_name=conditions[0])
# study.add_task(TaskValidateKinetics, cond_name=conditions[0])


