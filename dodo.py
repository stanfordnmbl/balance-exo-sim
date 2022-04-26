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
    'state_tracking_weight':   1e2 * scale,
    'control_weight':          1e3 * scale,
    'grf_tracking_weight':     1e0 * scale,
    'upright_torso_weight':    1e4 * scale,
    'control_tracking_weight': 0 * scale, 
    'aux_deriv_weight':        1e5 * scale,
    'metabolics_weight':       0 * scale,
    'accel_weight':            0 * scale,
    'regularization_weight':   0 * scale,
    }

# Maximum perturbation torque
study.max_torque_percent = 50.0

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
colormap = 'plasma'
cmap = plt.get_cmap(colormap)
torques = [25]
times = [50, 52, 54, 56, 58, 60]
N = len(times)
indices = np.linspace(0.2, 0.8, N) 
colors = [cmap(idx) for idx in indices]

subjects = ['subject01', 
            # 'subject02', 
            # 'subject04', 
            # 'subject18', 
            # 'subject19'
            ]
masses = [72.85, 76.48, 80.30, 64.09, 68.5]
# study.add_task(TaskPlotSensitivityResults, subjects)
study.add_task(TaskPlotUnperturbedResults, subjects, masses,
    times, colors)

# Analysis
# -------=
masses = [study.get_subject(1).mass,
          # study.get_subject(2).mass,
          # study.get_subject(4).mass,
          # study.get_subject(18).mass,
          # study.get_subject(19).mass
          ]

rise = 10
fall = 5

# study.add_task(TaskPlotCenterOfMass, subjects, 40, 
#     torques, rise, fall, colors[0], 
#     posAPbox=[60, 70, 0.9, 1.1],
#     velAPbox=[30, 50, 0.38, 0.42],
#     accAPbox=[30, 50, -0.75, 0.75],
#     posSIbox=[50, 65, 0.0, 0.04],
#     velSIbox=[40, 60, -0.2, 0.2],
#     accSIbox=[40, 55, -1.0, 5.0],
#     posMLbox=[60, 70, -0.01, 0.01],
#     velMLbox=[40, 60, -0.04, -0.01],
#     accMLbox=[25, 50, -0.75, -0.25])
# study.add_task(TaskPlotGroundReactions, subjects, 40, 
#     torques, rise, fall, colors[0], 
#     APbox=[45, 60, 0.05, 0.2],
#     SIbox=[35, 55, 0.7, 1.3],
#     MLbox=[25, 50, -0.06, -0.02])
# study.add_task(TaskPlotAnkleTorques, subjects[0], 40, 
#     torques, rise, fall, colors[0])
# study.add_task(TaskCreatePerturbedVisualization, subjects, 40,
#     torques, rise, fall)

# study.add_task(TaskPlotGroundReactionBreakdown, [subjects[0]], 40, 
#     100, rise, fall, 
#     APbox=[45, 60, 0.05, 0.2],
#     SIbox=[35, 55, 0.7, 1.3],
#     MLbox=[25, 50, -0.06, -0.02])

# study.add_task(TaskPlotCenterOfMass, subjects, 50, 
#     torques, rise, fall, colors[0], 
#     posAPbox=[60, 70, 0.9, 1.1],
#     velAPbox=[30, 50, 0.38, 0.42],
#     accAPbox=[30, 50, -0.75, 0.75],
#     posSIbox=[50, 65, 0.0, 0.04],
#     velSIbox=[40, 60, -0.2, 0.2],
#     accSIbox=[40, 55, -1.0, 5.0],
#     posMLbox=[60, 70, -0.01, 0.01],
#     velMLbox=[40, 60, -0.04, -0.01],
#     accMLbox=[25, 50, -0.75, -0.25])
# study.add_task(TaskPlotGroundReactions, subjects, 50, 
#     torques, rise, fall, colors[0], 
#     APbox=[45, 60, 0.05, 0.2],
#     SIbox=[35, 55, 0.7, 1.3],
#     MLbox=[25, 50, -0.06, -0.02])
# study.add_task(TaskPlotAnkleTorques, subjects[0], 50, 
#     torques, rise, fall, colors[0])
# study.add_task(TaskCreatePerturbedVisualization, subjects, 50,
#     torques, rise, fall)

# study.add_task(TaskPlotGroundReactionBreakdown, [subjects[0]], 50, 
#     10, rise, fall, 
#     APbox=[45, 60, 0.05, 0.2],
#     SIbox=[35, 55, 0.7, 1.3],
#     MLbox=[25, 50, -0.06, -0.02])

# study.add_task(TaskPlotCenterOfMass, subjects, 60, 
#     torques, rise, fall, colors[-1],
#     posAPbox=[70, 80, 0.95, 1.05],
#     velAPbox=[55, 70, 0.4, 0.48],
#     accAPbox=[40, 65, 0.25, 2.25],
#     posSIbox=[60, 70, -0.01, 0.05],
#     velSIbox=[55, 70, 0.0, 0.4],
#     accSIbox=[45, 65, 0.0, 5.0],
#     posMLbox=[80, 100, -0.04, 0.0],
#     velMLbox=[50, 70, -0.05, -0.01],
#     accMLbox=[40, 65, -1.20, -0.4])
# study.add_task(TaskPlotGroundReactions, subjects, 60, 
#     torques, rise, fall, colors[-1],
#     APbox=[50, 70, 0.0, 0.22],
#     SIbox=[45, 60, 0.9, 1.2],
#     MLbox=[40, 60, -0.05, 0.01])
# study.add_task(TaskPlotAnkleTorques, subjects[0], 60, 
#     torques, rise, fall, colors[-1])
# study.add_task(TaskCreatePerturbedVisualization, subjects, 60,
#     torques, rise, fall)

# study.add_task(TaskPlotInstantaneousCenterOfMass, subjects, 
#     times, torques, rise, fall, colors)
# study.add_task(TaskPlotInstantaneousGroundReactions, subjects, 
#     times, torques, rise, fall, colors)


rise = 25
fall = 10

# study.add_task(TaskPlotCenterOfMass, subjects, 40, 
#     torques, rise, fall, colors[0], 
#     posAPbox=[60, 70, 0.9, 1.1],
#     velAPbox=[30, 50, 0.38, 0.42],
#     accAPbox=[30, 50, -0.75, 0.75],
#     posSIbox=[50, 65, 0.0, 0.04],
#     velSIbox=[40, 60, -0.2, 0.2],
#     accSIbox=[40, 55, -1.0, 5.0],
#     posMLbox=[60, 70, -0.01, 0.01],
#     velMLbox=[40, 60, -0.04, -0.01],
#     accMLbox=[25, 50, -0.75, -0.25])
# study.add_task(TaskPlotGroundReactions, subjects, 40, 
#     torques, rise, fall, colors[0], 
#     APbox=[45, 60, 0.05, 0.2],
#     SIbox=[35, 55, 0.7, 1.3],
#     MLbox=[25, 50, -0.06, -0.02])
# study.add_task(TaskPlotAnkleTorques, subjects[0], 40, 
#     torques, rise, fall, colors[0])
# study.add_task(TaskCreatePerturbedVisualization, subjects, 40,
#     torques, rise, fall)

# study.add_task(TaskPlotCenterOfMass, subjects, 50, 
#     torques, rise, fall, colors[0], 
#     posAPbox=[60, 70, 0.9, 1.1],
#     velAPbox=[30, 50, 0.38, 0.42],
#     accAPbox=[30, 50, -0.75, 0.75],
#     posSIbox=[50, 65, 0.0, 0.04],
#     velSIbox=[40, 60, -0.2, 0.2],
#     accSIbox=[40, 55, -1.0, 5.0],
#     posMLbox=[60, 70, -0.01, 0.01],
#     velMLbox=[40, 60, -0.04, -0.01],
#     accMLbox=[25, 50, -0.75, -0.25])
# study.add_task(TaskPlotGroundReactions, subjects, 50, 
#     torques, rise, fall, colors[0], 
#     APbox=[45, 60, 0.05, 0.2],
#     SIbox=[35, 55, 0.7, 1.3],
#     MLbox=[25, 50, -0.06, -0.02])
# study.add_task(TaskPlotAnkleTorques, subjects[0], 50, 
#     torques, rise, fall, colors[0])
# study.add_task(TaskCreatePerturbedVisualization, subjects, 50,
#     torques, rise, fall)

# study.add_task(TaskPlotGroundReactionBreakdown, [subjects[0]], 50, 
#     50, rise, fall, 
#     APbox=[45, 60, 0.05, 0.2],
#     SIbox=[35, 55, 0.7, 1.3],
#     MLbox=[25, 50, -0.06, -0.02])

# study.add_task(TaskPlotCenterOfMass, subjects, 60, 
#     torques, rise, fall, colors[-1],
#     posAPbox=[70, 80, 0.95, 1.05],
#     velAPbox=[55, 70, 0.4, 0.48],
#     accAPbox=[40, 65, 0.25, 2.25],
#     posSIbox=[60, 70, -0.01, 0.05],
#     velSIbox=[55, 70, 0.0, 0.4],
#     accSIbox=[45, 65, 0.0, 5.0],
#     posMLbox=[80, 100, -0.04, 0.0],
#     velMLbox=[50, 70, -0.05, -0.01],
#     accMLbox=[40, 65, -1.20, -0.4])
# study.add_task(TaskPlotGroundReactions, subjects, 60, 
#     torques, rise, fall, colors[-1],
#     APbox=[50, 70, 0.0, 0.22],
#     SIbox=[45, 60, 0.9, 1.2],
#     MLbox=[40, 60, -0.05, 0.01])
# study.add_task(TaskPlotAnkleTorques, subjects[0], 60, 
#     torques, rise, fall, colors[-1])
# study.add_task(TaskCreatePerturbedVisualization, subjects, 60,
#     torques, rise, fall)

study.add_task(TaskPlotInstantaneousCenterOfMass, subjects, 
    times, torques, rise, fall, colors)
study.add_task(TaskPlotInstantaneousGroundReactions, subjects, 
    times, torques, rise, fall, colors)

