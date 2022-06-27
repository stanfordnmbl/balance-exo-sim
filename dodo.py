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

# This line is necessary for registering the tasks with python-doit.
from vital_tasks import *

# Custom tasks for this project.
from tasks import *

# Custom helper functions for this project
from helpers import *

model_fname = 'Rajagopal2015_passiveCal_hipAbdMoved_noArms_EBCForces_toesAligned.osim'
generic_model_fpath = os.path.join('model', model_fname)
study = osp.Study('ankle_perturb_sim', 
    generic_model_fpath=generic_model_fpath)

# Set the treadmill walking speed for the study
study.walking_speed = 1.25

# Generic model file
# ------------------
study.add_task(TaskCopyGenericModelFilesToResults)

# Model markers to compute errors for
marker_suffix = ['ASI', 'PSI', 'TH1', 'TH2', 'TH3', 'CAL', 'TOE', 'MT5']
error_markers = ['*' + marker for marker in marker_suffix] 
error_markers.append('CLAV')
error_markers.append('C7')
study.error_markers = error_markers

scale = 0.01
study.weights = {
    'state_tracking_weight':   1e1 * scale,
    'control_weight':          1e3 * scale,
    'grf_tracking_weight':     1e5 * scale,
    'upright_torso_weight':    1e2 * scale,
    'control_tracking_weight': 0 * scale, 
    'aux_deriv_weight':        1e5 * scale,
    'metabolics_weight':       0 * scale,
    'regularization_weight':   0 * scale,
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
study.add_task(TaskCopyMotionCaptureData, walk125=(2, '_newCOP3'))

# Validate
# --------
subjects = [
            'subject01', 
            'subject02', 
            'subject04', 
            # 'subject18', 
            # 'subject19'
            ]
masses = [
          study.get_subject(1).mass,
          study.get_subject(2).mass,
          study.get_subject(4).mass,
          # study.get_subject(18).mass,
          # study.get_subject(19).mass
          ]
# study.add_task(TaskPlotSensitivityResults, subjects[:1])
study.add_task(TaskPlotUnperturbedResults, subjects, masses,
    study.times, colors)

# Analysis
# --------
study.add_task(TaskPlotCenterOfMass, subjects, [20, 30, 40, 50, 60], 
    study.torques, study.rise, study.fall)
study.add_task(TaskPlotCenterOfMass, subjects, [20], 
    study.torques, study.rise, study.fall)
study.add_task(TaskPlotCenterOfMass, subjects, [30], 
    study.torques, study.rise, study.fall)
study.add_task(TaskPlotCenterOfMass, subjects, [40], 
    study.torques, study.rise, study.fall)
study.add_task(TaskPlotCenterOfMass, subjects, [50], 
    study.torques, study.rise, study.fall)
study.add_task(TaskPlotCenterOfMass, subjects, [60], 
    study.torques, study.rise, study.fall)

study.add_task(TaskPlotInstantaneousCenterOfMass, subjects, 
    study.times, study.rise, study.fall)

study.add_task(TaskPlotInstantaneousGroundReactions, subjects, 
    study.times, study.rise, study.fall)

study.add_task(TaskPlotCenterOfMassVector, subjects, 
    study.times, study.rise, study.fall)

study.add_task(TaskPlotGroundReactionBreakdown, subjects, 30, 0, -10, study.rise, study.fall)
study.add_task(TaskPlotGroundReactionBreakdown, subjects, 30, 10, 0, study.rise, study.fall)
study.add_task(TaskPlotGroundReactionBreakdown, subjects, 30, 10, -10, study.rise, study.fall)
study.add_task(TaskPlotGroundReactionBreakdown, subjects, 40, 0, -10, study.rise, study.fall)
study.add_task(TaskPlotGroundReactionBreakdown, subjects, 40, 10, -10, study.rise, study.fall)
study.add_task(TaskPlotGroundReactionBreakdown, subjects, 50, 0, -10, study.rise, study.fall)
study.add_task(TaskPlotGroundReactionBreakdown, subjects, 50, 10, -10, study.rise, study.fall)

for time in study.times:
    study.add_task(TaskPlotGroundReactions, subjects, time, study.rise, study.fall)
    study.add_task(TaskPlotBodyAccelerations, subjects, time, study.rise, study.fall)

# for itorque, torque in enumerate(study.torques):

#     study.add_task(TaskPlotInstantaneousCenterOfMassLumbarStiffness, subjects[:1], 
#         study.times, study.torques[itorque], study.rise, study.fall)

    # for subtalar in study.subtalar_peak_torques:
    #     for lumbar_stiffness in study.lumbar_stiffnesses:
    #         study.add_task(TaskPlotGroundReactionsVersusEffectiveForces, subjects, 
    #             study.times, study.torques[itorque], study.rise, study.fall, subtalar,
    #             lumbar_stiffness=lumbar_stiffness)