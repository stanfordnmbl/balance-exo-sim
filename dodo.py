# Allow using the osimpipeline git submodule.
import sys
sys.path.insert(1, 'code')
sys.path.insert(1, 'osimpipeline')
sys.path.insert(1, 'osimpipeline/osimpipeline')

import os
import yaml
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

model_fname = ('Rajagopal2015_passiveCal_hipAbdMoved_noArms'
               '_36musc_optMaxIsoForces.osim')
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

scale = 0.1
study.weights = {
    'state_tracking_weight':   5e3 * scale,
    'control_weight':          1e3 * scale,
    'grf_tracking_weight':     1e1 * scale,
    'upright_torso_weight':    1e0 * scale,
    'control_tracking_weight': 0 * scale, 
    'aux_deriv_weight':        1e4 * scale,
    'metabolics_weight':       0 * scale,
    'accel_weight':            1e4 * scale,
    'regularization_weight':   0 * scale,
    }

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
subjects = ['subject01', 'subject02', 'subject04', 
            'subject18', 'subject19']
masses = [72.85, 76.48, 80.30, 64.09, 68.5]
study.add_task(TaskPlotSensitivityResults, subjects)
study.add_task(TaskPlotUnperturbedResults, subjects, masses)

# Analysis
# -------=
masses = [study.get_subject(1).mass,
          study.get_subject(2).mass,
          study.get_subject(4).mass,
          study.get_subject(18).mass,
          study.get_subject(19).mass]

colormap = 'nipy_spectral'
cmap = plt.get_cmap(colormap)

# study.add_task(TaskPlotNormalizedImpulse, subjects, [50, 60], 
#     colormap, [0.5, 0.9], delay)

study.add_task(TaskPlotCenterOfMass, subjects, 50, 
    [25, 50, 75, 100], cmap(0.5), 
    posAPbox=[60, 70, 0.8, 0.9],
    velAPbox=[40, 60, 1.05, 1.30],
    accAPbox=[30, 50, -0.75, 0.75],
    posSIbox=[50, 65, 0.0, 0.04],
    velSIbox=[40, 60, -0.2, 0.2],
    accSIbox=[40, 55, -1.0, 5.0],
    posMLbox=[90, 100, -0.03, 0.0],
    velMLbox=[60, 65, -0.22, -0.14],
    accMLbox=[40, 60, -1.0, -0.5])
study.add_task(TaskPlotGroundReactions, subjects, 50, 
    [25, 50, 75, 100], cmap(0.5), 
    APbox=[30, 55, -0.04, 0.1],
    SIbox=[40, 60, 1.0, 1.5],
    MLbox=[40, 60, -0.10, -0.04])
study.add_task(TaskPlotAnkleTorques, subjects[0], 50, 
    [25, 50, 75, 100], cmap(0.5))


study.add_task(TaskPlotCenterOfMass, subjects, 60, 
    [25, 50, 75, 100], cmap(0.9),
    posAPbox=[70, 80, 0.95, 1.05],
    velAPbox=[60, 70, 1.3, 1.5],
    accAPbox=[40, 65, 0.25, 2.25],
    posSIbox=[60, 70, -0.01, 0.05],
    velSIbox=[55, 70, 0.0, 0.4],
    accSIbox=[45, 65, 0.0, 5.0],
    posMLbox=[80, 100, -0.04, 0.0],
    velMLbox=[60, 70, -0.24, -0.1],
    accMLbox=[40, 65, -1.20, -0.4])
study.add_task(TaskPlotGroundReactions, subjects, 60, 
    [25, 50, 75, 100], cmap(0.9),
    APbox=[55, 65, 0.1, 0.28],
    SIbox=[45, 60, 1.0, 1.40],
    MLbox=[40, 65, -0.10, -0.04])
study.add_task(TaskPlotAnkleTorques, subjects[0], 60, 
    [25, 50, 75, 100], cmap(0.9))
