# Allow using the osimpipeline git submodule.
import sys
sys.path.insert(1, 'code')
sys.path.insert(1, 'osimpipeline')
sys.path.insert(1, 'osimpipeline/osimpipeline')

import os
import yaml
import numpy as np
with open('config.yaml') as f:
    config = yaml.safe_load(f)
if 'opensim_home' not in config:
    raise Exception('You must define the field `opensim_home` in config.yaml '
                    'to point to the root of your OpenSim 4.0 (or later) '
                    'installation.')
sys.path.insert(1, os.path.join(config['opensim_home'], 'sdk', 'python'))

DOIT_CONFIG = {
        'verbosity': 2,
        'default_tasks': None,
        }

# Settings for plots.
import matplotlib.pyplot as plt
plt.rc('font', family='Helvetica, Arial, sans-serif', size=8)
plt.rc('errorbar', capsize=1.5)
plt.rc('lines', markeredgewidth=1)
plt.rc('legend', fontsize=8)

import osimpipeline as osp
from osimpipeline import postprocessing as pp

# This line is necessary for registering the tasks with python-doit.
from vital_tasks import *

# Custom tasks for this project.
from tasks import *

# Custom helper functions for this project
from helpers import *

model_fname = 'Rajagopal2015_passiveCal_hipAbdMoved_EBCForces_ankleBushings_toesAligned.osim'
generic_model_fpath = os.path.join('model', model_fname)
study = osp.Study('ankle_perturb_sim', 
    generic_model_fpath=generic_model_fpath)

# Set the treadmill walking speed for the study
study.walking_speed = 1.25

# Generic model file
# ------------------
study.add_task(TaskCopyGenericModelFilesToResults)
study.add_task(TaskApplyMarkerSetToGenericModel)

# Model markers to compute errors for
marker_suffix = ['ASI', 'PSI', 'TH1', 'TH2', 'TH3', 'CAL', 'TOE', 'MT5']
error_markers = ['*' + marker for marker in marker_suffix] 
error_markers.append('CLAV')
error_markers.append('C7')
study.error_markers = error_markers

scale = 1.0
study.weights = {
    'state_tracking_weight':    50 * scale,
    'control_weight':           25 * scale,
    'grf_tracking_weight':      7500 * scale,
    'torso_orientation_weight': 10 * scale,
    'feet_orientation_weight':  10 * scale,
    'control_tracking_weight':  0 * scale, 
    'aux_deriv_weight':         1000 * scale,
    'acceleration_weight':      1 * scale,
    }
study.constraint_tolerance = 1e-4
study.convergence_tolerance = 1e-2

# Maximum perturbation torque
study.torques = [0, 10]
study.times = [20, 25, 30, 35, 40, 45, 50, 55, 60] 
study.rise = 10
study.fall = 5
study.subtalar_peak_torques = [-10, 0, 10]
study.subtalar_suffixes = list()
for peak_torque in study.subtalar_peak_torques:
    if peak_torque:
        study.subtalar_suffixes.append(f'_subtalar{peak_torque}')
    else:
        study.subtalar_suffixes.append('')

study.lumbar_stiffnesses = [0.1, 1.0, 10.0]

colormap = 'plasma'
cmap = plt.get_cmap(colormap)
indices = np.linspace(0, 1.0, len(study.subtalar_suffixes)) 
study.subtalar_colors = [cmap(idx) for idx in indices]

# Add subject tasks
# -----------------
import subject01
subject01.add_to_study(study)

import subject02
subject02.add_to_study(study)

import subject04
subject04.add_to_study(study)

import subject18
subject18.add_to_study(study)

import subject19
subject19.add_to_study(study)

# Copy mocap data
# ---------------
study.add_task(TaskCopyMotionCaptureData, walk125=(2, ''))

# Plot settings
# -------------
subjects = [
            'subject01', 
            'subject02', 
            'subject04', 
            'subject18', 
            'subject19'
            ]
masses = [
          study.get_subject(1).mass,
          study.get_subject(2).mass,
          study.get_subject(4).mass,
          study.get_subject(18).mass,
          study.get_subject(19).mass
          ]

study.plot_torques = [0, 10, 10, 10, 0]

plot_subtalars = list()
plot_subtalars.append(study.subtalar_suffixes[0]) 
plot_subtalars.extend(study.subtalar_suffixes)
plot_subtalars.append(study.subtalar_suffixes[-1])
study.plot_subtalars = plot_subtalars

lightorange = [c / 255.0 for c in [253,141,60]]
orange =      [c / 255.0 for c in [217,71,1]]
blue =        [c / 255.0 for c in [33,113,181]]
lightblue =   [c / 255.0 for c in [107,174,214]]
study.plot_colors = [pp.adjust_lightness(lightorange, amount=1.0),
                     pp.adjust_lightness(orange, amount=1.0),
                     'black',
                     pp.adjust_lightness(blue, amount=1.0),
                     pp.adjust_lightness(lightblue, amount=1.0)
                    ]

# Methods figure
# --------------
study.add_task(TaskPlotMethodsFigure, subjects, study.times)

# Validate
# --------
study.add_task(TaskPlotUnperturbedResults, subjects, masses,
    study.times)
study.add_task(TaskValidateTrackingErrors, subjects, masses,
    study.times)
study.add_task(TaskValidateMarkerErrors)
study.add_task(TaskComputeCenterOfMassTimesteppingError,
    subjects, study.times, study.rise, study.fall)
study.add_task(TaskValidateMuscleActivity, subjects)

# Center-of-mass analysis
# -----------------------
study.add_task(TaskPlotCenterOfMassVector, subjects,
    study.times, study.rise, study.fall)
study.add_task(TaskCreateCenterOfMassStatisticsTables, subjects,
    study.times, study.rise, study.fall)
study.add_task(TaskAggregateCenterOfMassStatistics,
    study.times, study.rise, study.fall)
study.add_task(TaskPlotInstantaneousCenterOfMass, subjects, 
    study.times, study.rise, study.fall)
study.add_task(TaskPlotCOMVersusCOP, subjects, 
    study.times, study.rise, study.fall)

# Center-of-pressure analysis
# ---------------------------
study.add_task(TaskPlotCenterOfPressureVector, subjects,
    study.times, study.rise, study.fall)
study.add_task(TaskCreateCenterOfPressureStatisticsTables, subjects,
    study.times, study.rise, study.fall)
study.add_task(TaskAggregateCenterOfPressureStatistics,
    study.times, study.rise, study.fall)
study.add_task(TaskPlotInstantaneousCenterOfPressure, subjects, 
    study.times, study.rise, study.fall)



# Device powers
# -------------
study.add_task(TaskCreatePerturbationPowersTable, subjects)
study.add_task(TaskCreatePerturbationPowersTable, subjects,
    torque_actuators=True)
study.add_task(TaskPlotPerturbationPowers, subjects)