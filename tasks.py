import os

import numpy as np
import pylab as pl
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import copy
import shutil
import opensim as osim

import osimpipeline as osp
from osimpipeline import utilities as util
from osimpipeline import postprocessing as pp
from matplotlib import colors as mcolors
import matplotlib.patches as patches
from matplotlib.ticker import AutoMinorLocator

from tracking_problem import TrackingProblem, TrackingConfig
from timestepping_problem import TimeSteppingProblem, TimeSteppingConfig

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['hatch.linewidth'] = 1.0
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# Helper functions
# ----------------
# Update plot axis limits 'lim' based on input data and desired tick
# interval 'interval'. Optional argument 'mirror' enforces that the upper
# and lower plot limits have the same magnitude (i.e., mirrored about zero). 
def update_lims(data, interval, lims, mirror=False):
    if np.min(data) < lims[0]:
        lims[0] = interval * np.floor(np.min(data) / interval)
        if mirror: lims[1] = -lims[0]
    if np.max(data) > lims[1]:
        lims[1] = interval * np.ceil(np.max(data) / interval)
        if mirror: lims[0] = -lims[1]

# Get the ticks for an axis based on the limits 'lims' and desired
# tick interval 'interval'.
def get_ticks_from_lims(lims, interval):
    N = int(np.around((lims[1] - lims[0]) / interval, decimals=3)) + 1
    ticks = np.linspace(lims[0], lims[1], N)
    return ticks

# Add errorbars to a bar chart. For positive values, the errorbars will
# be above the bars, and for negative values, the errorbars will be
# below the bars.  
def plot_errorbar(ax, x, y, yerr):
    lolims = y > 0
    uplims = y < 0
    ple, cle, ble = ax.errorbar(x, y, yerr=yerr, 
        fmt='none', ecolor='black', 
        capsize=0, solid_capstyle='projecting', lw=0.25, 
        zorder=2.5, clip_on=False, lolims=lolims, uplims=uplims,
        elinewidth=0.4, markeredgewidth=0.4)
    for cl in cle:
        cl.set_marker('_')
        cl.set_markersize(4)

# Add errorbars to a horizontal bar chart. For positive values, the 
# errorbars will be to the right of the bars, and for negative values, 
# the errorbars will be to the left of the bars.  
def plot_errorbarh(ax, y, x, xerr):
    xlolims = x > 0
    xuplims = x < 0
    ple, cle, ble = ax.errorbar(x, y, xerr=xerr, 
        fmt='none', ecolor='black', capsize=0, 
        solid_capstyle='projecting', lw=0.25, 
        zorder=2.5, clip_on=False, xlolims=xlolims, xuplims=xuplims,
        elinewidth=0.4, markeredgewidth=0.4)
    for cl in cle:
        cl.set_marker('|')
        cl.set_markersize(4)

# Compute the Euclidean distance between two points in the same frame.
# Each vector is type SimTK::Vec3.
def compute_distance(vec1, vec2):
    x = vec1[0] - vec2[0]
    y = vec1[1] - vec2[1]
    z = vec1[2] - vec2[2]

    return np.sqrt(x*x + y*y + z*z)


# Preprocessing
# -------------

class working_directory():
    """Use this to temporarily run code with some directory as a working
    directory and to then return to the original working directory::

        with working_directory('<dir>'):
            pass
    """
    def __init__(self, path):
        self.path = path
        self.original_working_dir = os.getcwd()
    def __enter__(self):
        os.chdir(self.path)
    def __exit__(self, *exc_info):
        os.chdir(self.original_working_dir)


class TaskApplyMarkerSetToGenericModel(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study):
        super(TaskApplyMarkerSetToGenericModel, self).__init__(study)
        self.name = f'{study.name}_apply_markerset_to_generic_model'
        self.model_fpath = os.path.join(study.config['results_path'],
            'generic_model.osim')
        self.markerset_fpath = os.path.join(study.config['doit_path'],
            'templates', 'scale', 'prescale_markerset.xml')
        self.fixed_markers = list()
        bilateral_markers = ['ASI', 'PSI', 'HJC', 'ASH', 'PSH', 'ACR', 'LEL', 
                             'MEL', 'FAradius', 'FAulna', 'LFC', 'MFC', 'KJC', 
                             'LMAL', 'MMAL', 'AJC', 'CAL', 'MT5', 'TOE']

        for marker in bilateral_markers:
            for side in ['L', 'R']:
                self.fixed_markers.append(f'{side}{marker}')

        self.fixed_markers.append('C7')
        self.fixed_markers.append('CLAV')

        self.output_model_fpath = os.path.join(study.config['results_path'],
            'generic_model_prescale_markers.osim')

        self.add_action([self.model_fpath,
                         self.markerset_fpath], 
                        [self.output_model_fpath],
                        self.apply_markerset_to_model)

    def apply_markerset_to_model(self, file_dep, target):
        model = osim.Model(file_dep[0])
        markerSet = osim.MarkerSet(file_dep[1])

        for fixed_marker in self.fixed_markers:
            marker = markerSet.get(fixed_marker)
            marker.set_fixed(True)

        model.initSystem()
        currMarkerSet = model.updMarkerSet()
        currMarkerSet.clearAndDestroy()

        model.updateMarkerSet(markerSet)

        model.finalizeConnections()
        model.printToXML(target[0])


class TaskCopyMotionCaptureData(osp.TaskCopyMotionCaptureData):
    REGISTRY = []
    def __init__(self, study, walk125=None):
        regex_replacements = list()

        default_args = dict()
        default_args['walk125'] = walk125

        for subject in study.subjects:
            cond_args = subject.cond_args            
            if 'walk125' in cond_args: walk125 = cond_args['walk125']
            else: walk125 = default_args['walk125']

            for datastr, condname, arg in [('Walk_125', 'walk2', walk125)]:
                # Marker trajectories.
                regex_replacements.append(
                    (
                        os.path.join(subject.name, 'Data',
                            '%s %02i.trc' % (datastr, arg[0])).replace('\\',
                            '\\\\'),
                        os.path.join('experiments',
                            subject.name, condname, 'expdata', 
                            'marker_trajectories.trc').replace('\\','\\\\')
                        ))
                # Ground reaction.
                regex_replacements.append(
                    (
                        os.path.join(subject.name, 'Data',
                            '%s %02i%s.mot' % (datastr, arg[0],arg[1])).replace(
                                '\\','\\\\'),
                        os.path.join('experiments', subject.name, condname,
                            'expdata','ground_reaction_orig.mot').replace(
                                '\\','\\\\') 
                        )) 
                # EMG
                regex_replacements.append(
                    (
                        os.path.join(subject.name, 'Results', datastr,
                            '%s%02i_gait_controls.sto' % (datastr, arg[0])
                            ).replace('\\','\\\\'),
                        os.path.join('experiments', subject.name,
                            condname, 'expdata', 'emg.sto'
                            ).replace('\\','\\\\')
                        ))
            regex_replacements.append((
                        os.path.join(subject.name, 'Data',
                            'Static_FJC.trc').replace('\\','\\\\'),
                        os.path.join('experiments', subject.name, 'static',
                            'expdata',
                            'marker_trajectories.trc').replace('\\','\\\\') 
                        ))

        super(TaskCopyMotionCaptureData, self).__init__(study,
                regex_replacements)


class TaskUpdateGroundReactionLabels(osp.TrialTask):
    REGISTRY = []
    def __init__(self, trial):
        super(TaskUpdateGroundReactionLabels, self).__init__(trial)
        self.name = trial.id + '_update_ground_reaction_labels'
        self.add_action(
                [os.path.join(trial.expdata_path, 'ground_reaction_orig.mot')],
                [os.path.join(trial.expdata_path, 'ground_reaction_unfiltered.mot')],
                self.dispatch)

    def dispatch(self, file_dep, target):
        import re
        data = util.storage2numpy(file_dep[0])
        new_names = list()
        for name in data.dtype.names:
            if name == 'time':
                new_name = name
            elif name.endswith('_1'):
                new_name = re.sub('ground_(.*)_(.*)_1', 'ground_\\1_l_\\2',
                        name)
            else:
                new_name = re.sub('ground_(.*)_(.*)', 'ground_\\1_r_\\2',
                        name)
            new_names.append(new_name)
        data.dtype.names = new_names
        util.ndarray2storage(data, target[0])


class TaskFilterGroundReactions(osp.TrialTask):
    REGISTRY = []
    def __init__(self, trial, sample_rate=2000,
                 critically_damped_order=4, 
                 critically_damped_cutoff_frequency=20,
                 gaussian_smoothing_sigma=0):
        super(TaskFilterGroundReactions, self).__init__(trial)
        self.name = trial.id + '_filter_ground_reactions'

        # Recorded force sample rate (Hz)
        self.sample_rate = sample_rate

        # Critically damped filter order and cutoff frequency
        self.critically_damped_order = critically_damped_order
        self.critically_damped_cutoff_frequency = \
            critically_damped_cutoff_frequency

        # Smoothing factor for Gaussian smoothing process
        # trial.ground_reaction_fpath
        self.gaussian_smoothing_sigma = gaussian_smoothing_sigma

        self.add_action(
            [os.path.join(trial.expdata_path, 
                'ground_reaction_unfiltered.mot').replace('\\', '\\\\')],
            [os.path.join(trial.expdata_path, 
                'ground_reaction.mot').replace('\\', '\\\\'),
             os.path.join(trial.expdata_path, 
                'ground_reactions_filtered.png').replace('\\', '\\\\')],
            self.filter_ground_reactions)

    def filter_ground_reactions(self, file_dep, target):

        grfs = osim.TimeSeriesTable(file_dep[0])
        nrow = grfs.getNumRows()
        time = grfs.getIndependentColumn()
        sides = ['l', 'r']

        forces = {side: np.zeros((nrow, 3)) for side in sides}
        moments = {side: np.zeros((nrow, 3)) for side in sides}
        cops = {side: np.zeros((nrow, 3)) for side in sides}

        for side in sides:
            forces[side][:, 0] = util.simtk2numpy(grfs.getDependentColumn(f'ground_force_{side}_vx'))
            forces[side][:, 1] = util.simtk2numpy(grfs.getDependentColumn(f'ground_force_{side}_vy'))
            forces[side][:, 2] = util.simtk2numpy(grfs.getDependentColumn(f'ground_force_{side}_vz'))
            moments[side][:, 0] = util.simtk2numpy(grfs.getDependentColumn(f'ground_torque_{side}_x'))
            moments[side][:, 1] = util.simtk2numpy(grfs.getDependentColumn(f'ground_torque_{side}_y'))
            moments[side][:, 2] = util.simtk2numpy(grfs.getDependentColumn(f'ground_torque_{side}_z'))
            cops[side][:, 0] = util.simtk2numpy(grfs.getDependentColumn(f'ground_force_{side}_px'))
            cops[side][:, 1] = util.simtk2numpy(grfs.getDependentColumn(f'ground_force_{side}_py'))
            cops[side][:, 2] = util.simtk2numpy(grfs.getDependentColumn(f'ground_force_{side}_pz'))

        # Plot raw GRF (before cutting off or filtering)
        fig = pl.figure(figsize=(12, 10))
        ax_FX = fig.add_subplot(6, 1, 1)
        ax_FY = fig.add_subplot(6, 1, 2)
        ax_FZ = fig.add_subplot(6, 1, 3)
        ax_MX = fig.add_subplot(6, 1, 4)
        ax_MY = fig.add_subplot(6, 1, 5)
        ax_MZ = fig.add_subplot(6, 1, 6)
        for iside, side in enumerate(sides):
            ax_FX.plot(time, forces[side][:, 0], lw=2, color='black')
            ax_FY.plot(time, forces[side][:, 1], lw=2, color='black')
            ax_FZ.plot(time, forces[side][:, 2], lw=2, color='black')
            ax_MX.plot(time, moments[side][:, 0], lw=2, color='black')
            ax_MY.plot(time, moments[side][:, 1], lw=2, color='black')
            ax_MZ.plot(time, moments[side][:, 2], lw=2, color='black')

        # Gaussian filter
        # ---------------
        if self.gaussian_smoothing_sigma:
            for side in sides:
                for item in [forces, moments]:
                    for i in np.arange(item[side].shape[1]):
                        item[side][:, i] = gaussian_filter1d(item[side][:, i], 
                            self.gaussian_smoothing_sigma)

        # Critically damped filter (prevents overshoot).
        for item in [forces, moments]:
            for side in sides:
                for direc in range(3):
                    item[side][:, direc] = util.filter_critically_damped(
                            item[side][:, direc], self.sample_rate,
                            self.critically_damped_cutoff_frequency,
                            order=self.critically_damped_order)

        # Create structured array for MOT file.
        # -------------------------------------
        dtype_names = ['time']
        data_dict = dict()
        for side in sides:
            # Forces
            for idirec, direc in enumerate(['x', 'y', 'z']):
                colname = f'ground_force_{side}_v{direc}'
                dtype_names.append(colname)
                data_dict[colname] = forces[side][:, idirec].reshape(-1)

            # Centers of pressure
            for idirec, direc in enumerate(['x', 'y', 'z']):
                colname = f'ground_force_{side}_p{direc}'
                dtype_names.append(colname)
                data_dict[colname] = cops[side][:, idirec].reshape(-1)

            # Moments
            for idirec, direc in enumerate(['x', 'y', 'z']):
                colname = f'ground_torque_{side}_{direc}'
                dtype_names.append(colname)
                data_dict[colname] = moments[side][:, idirec].reshape(-1)

        mot_data = np.empty(nrow, dtype={'names': dtype_names,
            'formats': len(dtype_names) * ['f8']})
        mot_data['time'] = time #[[0] + range(nrow-1)]
        for k, v in data_dict.items():
            mot_data[k] = v

        util.ndarray2storage(mot_data, target[0], name='ground reactions')

        # Plot filtered GRFs
        # ------------------
        for side in sides:
            ax_FX.plot(time, forces[side][:, 0], lw=1.5, color='red')
            ax_FY.plot(time, forces[side][:, 1], lw=1.5, color='red')
            ax_FZ.plot(time, forces[side][:, 2], lw=1.5, color='red')
            ax_MX.plot(time, moments[side][:, 0], lw=1.5, color='red')
            ax_MY.plot(time, moments[side][:, 1], lw=1.5, color='red')
            ax_MZ.plot(time, moments[side][:, 2], lw=1.5, color='red')
        ax_FX.set_ylabel('FX')
        ax_FY.set_ylabel('FY')
        ax_FZ.set_ylabel('FZ')
        ax_MX.set_ylabel('MX')
        ax_MY.set_ylabel('MY')
        ax_MZ.set_ylabel('MZ')
        fig.savefig(target[1])
        pl.close()


# Scaling
# -------

class TaskCopyModelSegmentMasses(osp.SubjectTask):
    REGISTRY = []
    def __init__(self, subject):
        super(TaskCopyModelSegmentMasses, self).__init__(subject)
        self.subject = subject
        self.study = subject.study
        self.name = '%s_copy_model_segment_masses' % self.subject.name
        self.doc = 'Copy model segment masses to addbio model'
        self.scale_model_fpath = os.path.join(self.subject.results_exp_path, 
            '%s.osim' % self.subject.name)
        self.addbio_model_fpath = os.path.join(self.subject.results_exp_path, 
            '%s_addbio.osim' % self.subject.name)
        self.updated_model_fpath = os.path.join(
            self.subject.results_exp_path, 
            '%s_addbio_updated_masses.osim' % self.subject.name)

        self.add_action([self.scale_model_fpath,
                         self.addbio_model_fpath],
                        [self.updated_model_fpath],
                        self.copy_segment_masses)

    def copy_segment_masses(self, file_dep, target):

        scale_model = osim.Model(file_dep[0])
        scale_model.initSystem()

        addbio_model = osim.Model(file_dep[1])
        addbio_model.initSystem()
        addbio_model.setName(f'{self.study.name}_{self.subject.name}')

        bodyNames = osim.ArrayStr()
        bodySet_scale = scale_model.getBodySet()
        bodySet_addbio = addbio_model.updBodySet()

        bodySet_scale.getNames(bodyNames)

        for ibody in range(bodyNames.getSize()):
            body_scale = bodySet_scale.get(bodyNames.get(ibody))
            body_addbio = bodySet_addbio.get(bodyNames.get(ibody))
            body_addbio.set_mass(body_scale.get_mass())

        state_scale = scale_model.initSystem()
        state_addbio = addbio_model.initSystem()

        assert(scale_model.getTotalMass(state_scale) == 
               addbio_model.getTotalMass(state_addbio))

        addbio_model.initSystem()
        addbio_model.finalizeConnections()
        addbio_model.printToXML(target[0])


class TaskScaleMuscleMaxIsometricForce(osp.SubjectTask):
    REGISTRY = []
    """The generic model mass and heights are based on the generic Rajagopal
       et al. 2015 model.
    """
    def __init__(self, subject, generic_mass=75.337, generic_height=1.6557):
        super(TaskScaleMuscleMaxIsometricForce, self).__init__(subject)
        self.subject = subject
        self.name = '%s_scale_max_force' % self.subject.name
        self.doc = 'Scale subject muscle Fmax parameters from Handsfield2014'
        self.generic_model_fpath = self.study.source_generic_model_fpath
        self.subject_model_fpath = os.path.join(self.subject.results_exp_path, 
            '%s_addbio_updated_masses.osim' % self.subject.name)
        self.scaled_param_model_fpath = os.path.join(
            self.subject.results_exp_path, 
            '%s_final.osim' % self.subject.name)
        self.generic_mass = generic_mass
        self.generic_height = generic_height

        self.add_action([self.generic_model_fpath, self.subject_model_fpath],
                        [self.scaled_param_model_fpath],
                        self.scale_model_parameters)

    def scale_model_parameters(self, file_dep, target):
        """From Handsfields 2014 figure 5a and from Apoorva's muscle properties
       spreadsheet.
       
       v: volume fraction
       V: total volume
       F: max isometric force
       l: optimal fiber length

       F = v * sigma * V / l

       *_g: generic model.
       *_s: subject-specific model.

       F_g = v * sigma * V_g / l_g
       F_s = v * sigma * V_s / l_s

       F_s = (F_g * l_g / V_g) * V_s / l_s
           = F_g * (V_s / V_g) * (l_g / l_s)

        Author: Chris Dembia 
        Borrowed from mrsdeviceopt GitHub repo:
        https://github.com/chrisdembia/mrsdeviceopt          
       """

        print("Muscle force scaling: "
              "total muscle volume and optimal fiber length.")

        # def total_muscle_volume_regression(mass):
        #     return 91.0*mass + 588.0

        def total_muscle_volume_regression(mass, height):
            return 47.0*mass*height + 1285.0

        generic_TMV = total_muscle_volume_regression(self.generic_mass, 
            self.generic_height)
        subj_TMV = total_muscle_volume_regression(self.subject.mass, 
            self.subject.height)

        import opensim as osm
        generic_model = osm.Model(file_dep[0])
        subj_model = osm.Model(file_dep[1])

        generic_mset = generic_model.getMuscles()
        subj_mset = subj_model.getMuscles()

        for im in range(subj_mset.getSize()):
            muscle_name = subj_mset.get(im).getName()

            generic_muscle = generic_mset.get(muscle_name)
            subj_muscle = subj_mset.get(muscle_name)

            generic_OFL = generic_muscle.get_optimal_fiber_length()
            subj_OFL = subj_muscle.get_optimal_fiber_length()

            scale_factor = (subj_TMV / generic_TMV) * (generic_OFL / subj_OFL)
            print("Scaling '%s' muscle force by %f." % (muscle_name,
                scale_factor))

            generic_force = generic_muscle.getMaxIsometricForce()
            scaled_force = generic_force * scale_factor
            subj_muscle.setMaxIsometricForce(scaled_force)

        subj_model.printToXML(target[0])


# Tracking data
# -------------

class TaskComputeJointAngleStandardDeviations(osp.TrialTask):
    REGISTRY = []
    def __init__(self, trial, ik_setup_task):
        super(TaskComputeJointAngleStandardDeviations, self).__init__(trial)
        self.name = trial.id + '_joint_angle_standard_deviations'
        self.trial = trial
        self.ik_solution_fpath = ik_setup_task.solution_fpath

        self.add_action([self.ik_solution_fpath],
                        [os.path.join(trial.results_exp_path, 
                         f'{trial.id}_joint_angle_standard_deviations.csv')],
                        self.compute_joint_angle_standard_deviations)

    def compute_joint_angle_standard_deviations(self, file_dep, target):
        
        kinematics = osim.TimeSeriesTable(file_dep[0])
        labels = kinematics.getColumnLabels()
        angles = np.ndarray(shape=(100, len(labels), len(self.trial.cycles)))     
        for icycle, cycle in enumerate(self.trial.cycles):
            istart = kinematics.getNearestRowIndexForTime(cycle.start)
            iend = kinematics.getNearestRowIndexForTime(cycle.end)+1
            time = kinematics.getIndependentColumn()[istart:iend]
            timeInterp = np.linspace(cycle.start, cycle.end, 100)
            for ilabel, label in enumerate(labels):
                col = kinematics.getDependentColumn(label).to_numpy()[istart:iend]
                colInterp = np.interp(timeInterp, time, col)
                angles[:, ilabel, icycle] = colInterp

        # Normalize by magnitude
        for ilabel, label in enumerate(labels):
            minAngle = np.min(angles[:, ilabel, :])
            maxAngle = np.max(angles[:, ilabel, :])
            magnitude = maxAngle - minAngle

            if magnitude > np.finfo(float).eps:
                angles[:, ilabel, :] /= magnitude

        angles_std = np.std(angles, axis=2)
        angles_std_mean = np.mean(angles_std, axis=0)
        df = pd.DataFrame(data=angles_std_mean, index=labels)
        df.to_csv(target[0])


class TaskTrimTrackingData(osp.TrialTask):
    REGISTRY = []
    def __init__(self, trial, ik_setup_task, id_setup_task, initial_time, final_time):
        super(TaskTrimTrackingData, self).__init__(trial)
        self.name = trial.id + '_trim_tracking_data'
        self.trial = trial
        self.ik_solution_fpath = ik_setup_task.solution_fpath
        self.extloads_fpath = id_setup_task.results_extloads_fpath
        self.grf_fpath = os.path.join(trial.results_exp_path, 'expdata', 
            'ground_reaction.mot')
        self.initial_time = initial_time
        self.final_time = final_time

        expdata_dir = os.path.join(trial.results_exp_path, 'tracking_data', 'expdata')
        extloads_dir = os.path.join(trial.results_exp_path, 'tracking_data', 'extloads')
        self.tracking_coordinates_fpath = os.path.join(expdata_dir, 'coordinates.sto')
        self.tracking_extloads_fpath = os.path.join(extloads_dir, 'external_loads.xml')
        self.tracking_grfs_fpath = os.path.join(expdata_dir, 'ground_reaction.mot')

        if not os.path.exists(expdata_dir): os.makedirs(expdata_dir)
        if not os.path.exists(extloads_dir): os.makedirs(extloads_dir)

        self.add_action([self.ik_solution_fpath,
                         self.extloads_fpath,
                         self.grf_fpath],
                        [self.tracking_coordinates_fpath,
                         self.tracking_extloads_fpath,
                         self.tracking_grfs_fpath],
                        self.trim_tracking_data)

    def trim_tracking_data(self, file_dep, target):
        
        self.copy_file([file_dep[1]], [target[1]])

        sto = osim.STOFileAdapter()
        kinematics = osim.TimeSeriesTable(file_dep[0])
        kinematics.trim(self.initial_time, self.final_time)
        sto.write(kinematics, target[0])

        grfs = osim.TimeSeriesTable(file_dep[2])
        grfs.trim(self.initial_time, self.final_time)
        sto.write(grfs, target[2])


# Unperturbed walking
# -------------------

class TaskMocoUnperturbedWalkingGuess(osp.TrialTask):
    REGISTRY = []
    def __init__(self, trial, initial_time, final_time, mesh_interval=0.02,
                walking_speed=1.25, constrain_average_speed=True, guess_fpath=None, 
                constrain_initial_state=False, periodic=True, cost_scale=1.0,
                costs_enabled=True, pelvis_boundary_conditions=True,
                reserve_strength=0, implicit_multibody_dynamics=False,
                implicit_tendon_dynamics=False, 
                create_and_insert_guess=False,
                convergence_tolerance=1e-2,
                constraint_tolerance=1e-2,
                **kwargs):
        super(TaskMocoUnperturbedWalkingGuess, self).__init__(trial)
        config_name = (f'unperturbed_guess_mesh{mesh_interval}'
                       f'_scale{cost_scale}_reserve{reserve_strength}')
        if not costs_enabled: config_name += '_costsDisabled'
        if periodic: config_name += '_periodic'
        self.config_name = config_name
        self.name = trial.subject.name + '_moco_' + config_name
        self.initial_time = initial_time
        self.final_time = final_time
        self.mesh_interval = mesh_interval
        self.convergence_tolerance = convergence_tolerance
        self.constraint_tolerance = constraint_tolerance
        self.num_max_iterations = 10000
        self.walking_speed = walking_speed
        self.weights = trial.study.weights
        self.cost_scale = cost_scale
        self.root_dir = trial.study.config['doit_path']
        self.constrain_initial_state = constrain_initial_state
        self.constrain_average_speed = constrain_average_speed
        self.periodic = periodic
        self.costs_enabled = costs_enabled
        self.pelvis_boundary_conditions = pelvis_boundary_conditions
        self.guess_fpath = guess_fpath
        self.reserve_strength = reserve_strength
        self.implicit_multibody_dynamics = implicit_multibody_dynamics
        self.implicit_tendon_dynamics = implicit_tendon_dynamics
        self.create_and_insert_guess = create_and_insert_guess

        expdata_dir = os.path.join(trial.results_exp_path, 'tracking_data', 'expdata')
        extloads_dir = os.path.join(trial.results_exp_path, 'tracking_data', 'extloads')
        self.tracking_coordinates_fpath = os.path.join(expdata_dir, 'coordinates.sto')
        self.coordinates_std_fpath = os.path.join(trial.results_exp_path, 
            f'{trial.id}_joint_angle_standard_deviations.csv')
        self.tracking_extloads_fpath = os.path.join(extloads_dir, 'external_loads.xml')
        self.tracking_grfs_fpath = os.path.join(expdata_dir, 'ground_reaction.mot')

        self.result_fpath = os.path.join(self.study.config['results_path'],
            'guess', trial.subject.name)
        if not os.path.exists(self.result_fpath): os.makedirs(self.result_fpath)

        self.archive_fpath = os.path.join(self.study.config['results_path'],
            'guess', trial.subject.name, 'archive')
        if not os.path.exists(self.archive_fpath): os.makedirs(self.archive_fpath)

        self.grf_fpath = os.path.join(trial.results_exp_path, 'expdata', 
            'ground_reaction.mot')
        self.emg_fpath = os.path.join(trial.results_exp_path, 'expdata', 
            'emg.sto')

        self.add_action([trial.subject.sim_model_fpath,
                         self.tracking_coordinates_fpath,
                         self.coordinates_std_fpath,
                         self.tracking_extloads_fpath,
                         self.tracking_grfs_fpath,
                         self.emg_fpath],
                        [os.path.join(self.result_fpath, 
                            self.config_name + '.sto')],
                        self.run_tracking_problem)

    def run_tracking_problem(self, file_dep, target):

        weights = copy.deepcopy(self.weights)
        for weight_name in weights:
            weights[weight_name] *= self.cost_scale

        weights['state_tracking_weight'] *= 5.0
        weights['grf_tracking_weight'] /= 0.5
        # weights['subtalar_weight'] = 1.0

        config = TrackingConfig( 
            self.config_name, self.config_name, 'black', weights,
            periodic=self.periodic,
            guess=self.guess_fpath,
            effort_enabled=self.costs_enabled,
            tracking_enabled=self.costs_enabled,
            create_and_insert_guess=self.create_and_insert_guess
            )

        cycles = list()
        for cycle in self.trial.cycles:
            cycles.append([cycle.start, cycle.end])

        result = TrackingProblem(
            self.root_dir, # root directory
            self.result_fpath, # result directory
            file_dep[0], # model file path
            file_dep[1], # IK coordinates path
            file_dep[2], # Coordinates standard deviations
            file_dep[3], # external loads file 
            file_dep[4], # GRF MOT file
            file_dep[5], # EMG data
            self.initial_time,
            self.final_time, 
            cycles,
            self.trial.right_strikes,
            self.trial.left_strikes,
            self.mesh_interval,
            self.convergence_tolerance,
            self.constraint_tolerance, 
            self.num_max_iterations,
            self.walking_speed,
            [config],
            reserve_strength=self.reserve_strength,
            implicit_multibody_dynamics=self.implicit_multibody_dynamics,
            implicit_tendon_dynamics=self.implicit_tendon_dynamics,
        )

        result.generate_results()
        result.report_results()


class TaskMocoUnperturbedWalking(osp.TrialTask):
    REGISTRY = []
    def __init__(self, trial, initial_time, final_time, mesh_interval=0.02,
                 walking_speed=1.25, guess_fpath=None, periodic=True,
                 lumbar_stiffness=1.0, create_and_insert_guess=False, 
                 **kwargs):
        super(TaskMocoUnperturbedWalking, self).__init__(trial)
        self.lumbar_subpath = ''
        suffix = ''
        if not lumbar_stiffness == 1.0:
            self.lumbar_subpath = f'lumbar{lumbar_stiffness}'
            suffix = f'_{self.lumbar_subpath}'
        self.config_name = f'unperturbed{suffix}'
        self.name = f'{trial.subject.name}_moco_{self.config_name}'
        self.initial_time = initial_time
        self.final_time = final_time
        self.mesh_interval = mesh_interval
        self.convergence_tolerance = trial.study.convergence_tolerance
        self.constraint_tolerance = trial.study.constraint_tolerance
        self.num_max_iterations = 10000
        self.walking_speed = walking_speed
        self.guess_fpath = guess_fpath
        self.root_dir = trial.study.config['doit_path']
        self.periodic = periodic
        self.lumbar_stiffness = lumbar_stiffness
        self.create_and_insert_guess = create_and_insert_guess
        self.weights = trial.study.weights

        expdata_dir = os.path.join(
            trial.results_exp_path, 'tracking_data', 'expdata')
        extloads_dir = os.path.join(
            trial.results_exp_path, 'tracking_data', 'extloads')
        self.tracking_coordinates_fpath = os.path.join(
            expdata_dir, 'coordinates.sto')
        self.coordinates_std_fpath = os.path.join(
            trial.results_exp_path, 
            f'{trial.id}_joint_angle_standard_deviations.csv')
        self.tracking_extloads_fpath = os.path.join(
            extloads_dir, 'external_loads.xml')
        self.tracking_grfs_fpath = os.path.join(
            expdata_dir, 'ground_reaction.mot')
        self.emg_fpath = os.path.join(
            trial.results_exp_path, 'expdata', 'emg.sto')

        self.result_fpath = os.path.join(
            self.study.config['results_path'], self.lumbar_subpath, 
            self.config_name,  trial.subject.name)
        if not os.path.exists(self.result_fpath): 
            os.makedirs(self.result_fpath)

        self.archive_fpath = os.path.join(
            self.study.config['results_path'], self.lumbar_subpath,
            self.config_name,  trial.subject.name, 'archive')
        if not os.path.exists(self.archive_fpath): 
            os.makedirs(self.archive_fpath)

        self.add_action([trial.subject.sim_model_fpath,
                         self.tracking_coordinates_fpath,
                         self.coordinates_std_fpath,
                         self.tracking_extloads_fpath,
                         self.tracking_grfs_fpath,
                         self.emg_fpath],
                        [os.path.join(
                            self.result_fpath, 
                            self.config_name + '.sto')],
                        self.run_tracking_problem)

    def run_tracking_problem(self, file_dep, target):

        weights = copy.deepcopy(self.weights)
        config = TrackingConfig(
            self.config_name, self.config_name, 'black', weights,
            periodic=self.periodic,
            periodic_values=True,
            periodic_speeds=True,
            periodic_actuators=True,
            lumbar_stiffness=self.lumbar_stiffness,
            guess=self.guess_fpath,
            create_and_insert_guess=self.create_and_insert_guess,
            )

        cycles = list()
        for cycle in self.trial.cycles:
            cycles.append([cycle.start, cycle.end])

        result = TrackingProblem(
            self.root_dir,      # root directory
            self.result_fpath,  # result directory
            file_dep[0],        # model file path
            file_dep[1],        # IK coordinates path
            file_dep[2],        # Coordinates standard deviations
            file_dep[3],        # external loads file 
            file_dep[4],        # GRF MOT file
            file_dep[5],        # EMG data
            self.initial_time,
            self.final_time, 
            cycles,
            self.trial.right_strikes,
            self.trial.left_strikes,
            self.mesh_interval, 
            self.convergence_tolerance,
            self.constraint_tolerance,
            self.num_max_iterations,
            self.walking_speed,
            [config])

        result.generate_results()
        result.report_results()


class TaskPlotUnperturbedResults(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, masses, times):
        super(TaskPlotUnperturbedResults, self).__init__(study)
        self.config_name = 'unperturbed'
        self.name = f'plot_{self.config_name}_results'
        self.figures_path = os.path.join(study.config['figures_path']) 
        self.results_path = os.path.join(study.config['results_path'], 
            'unperturbed')
        self.validate_path = os.path.join(study.config['validate_path'],
            'unperturbed')
        if not os.path.exists(self.validate_path): 
            os.makedirs(self.validate_path)
        self.subjects = subjects
        self.masses = masses
        self.exp_color = 'black'
        self.unp_color = 'deepskyblue'
        self.emg_map = {
           'tibant_r' : 'tibant_r',
           'soleus_r' : 'soleus_r',
           'gasmed_r' : 'gasmed_r',
           'semimem_r': 'semimem_r',
           'vasint_r' : 'vaslat_r',
           'recfem_r' : 'recfem_r',
           'glmax2_r' : 'glmax2_r',
           'glmed1_r' : 'glmed1_r',
        }
        self.forces = ['contactHeel_r',
                       'contactLateralRearfoot_r',
                       'contactLateralMidfoot_r',
                       'contactMedialMidfoot_r',
                       'contactLateralToe_r',
                       'contactMedialToe_r']
        self.models = list()
        self.times = times

        unperturbed_fpaths = list()
        unperturbed_grf_fpaths = list()
        coordinates_fpaths = list()
        grf_fpaths = list()
        emg_fpaths = list()
        experiment_com_fpaths = list()
        unperturbed_com_fpaths = list()

        for subject in subjects:
            self.models.append(os.path.join(
                self.study.config['results_path'], self.config_name, 
                subject, f'model_{self.config_name}.osim'))
            expdata_dir = os.path.join(
                study.config['results_path'], 'experiments', subject, 'walk2')
            coordinates_fpaths.append(os.path.join(
                self.study.config['results_path'], self.config_name, 
                subject, f'{self.config_name}_experiment_states.sto'))
            grf_fpaths.append(os.path.join(
                expdata_dir, 'tracking_data', 'expdata', 'ground_reaction.mot'))
            emg_fpaths.append(os.path.join(
                expdata_dir, 'expdata', 'emg.sto'))
            unperturbed_fpaths.append(os.path.join(
                self.study.config['results_path'], self.config_name, 
                subject, f'{self.config_name}.sto'))
            unperturbed_grf_fpaths.append(os.path.join(
                self.study.config['results_path'], self.config_name, 
                subject, f'{self.config_name}_grfs.sto'))
            experiment_com_fpaths.append(os.path.join(
                self.results_path, subject,
                'center_of_mass_experiment.sto'))
            unperturbed_com_fpaths.append(os.path.join(
                self.results_path, subject,
                f'center_of_mass_{self.config_name}.sto'))

        self.add_action(unperturbed_fpaths + coordinates_fpaths, 
                        [os.path.join(self.validate_path, 
                            f'{self.config_name}_coordinates.png'),
                         os.path.join(self.figures_path, 
                            'figureS10', 'figureS10.png')], 
                        self.plot_unperturbed_coordinates)

        self.add_action(unperturbed_grf_fpaths + grf_fpaths, 
                        [os.path.join(self.validate_path, 
                            f'{self.config_name}_grfs.png'),
                         os.path.join(self.figures_path, 
                            'figureS11', 'figureS11.png')], 
                        self.plot_unperturbed_grfs)

        self.add_action(unperturbed_fpaths + emg_fpaths, 
                        [os.path.join(self.validate_path, 
                            f'{self.config_name}_muscle_activity.png'),
                         os.path.join(self.figures_path, 
                            'figureS12', 'figureS12.png')], 
                        self.plot_unperturbed_muscle_activity)

        self.add_action(unperturbed_com_fpaths + experiment_com_fpaths, 
                        [os.path.join(self.validate_path, 
                            f'{self.config_name}_center_of_mass.png')], 
                        self.plot_unperturbed_center_of_mass)

        self.add_action(unperturbed_fpaths, 
                        [os.path.join(self.validate_path, 
                            f'{self.config_name}_step_widths.png'),
                         os.path.join(self.validate_path, 
                            f'{self.config_name}_step_widths.txt')], 
                        self.plot_unperturbed_step_widths)

        self.add_action(unperturbed_fpaths, 
                        [os.path.join(self.validate_path, 
                            f'{self.config_name}_gait_landmarks.txt')], 
                        self.compute_unperturbed_gait_landmarks)

    def plot_unperturbed_coordinates(self, file_dep, target): 

        numSubjects = len(self.subjects)
        N = 100
        pgc = np.linspace(0, 100, N)
        labels = ['hip flexion', 'hip adduction', 'hip rotation', 'knee flexion', 
                  'ankle dorsiflexion', 'subtalar inversion', 'mtp extension']
        coordinates = ['hip_flexion', 'hip_adduction', 'hip_rotation', 'knee_angle', 
                       'ankle_angle', 'subtalar_angle', 'mtp_angle']
        joints = ['hip', 'hip', 'hip', 'walker_knee', 'ankle', 'subtalar', 'mtp']
        bounds = [[-40, 40],
                  [-15, 10],
                  [-15, 10],
                  [0, 80],
                  [-20, 20],
                  [-10, 10],
                  [0, 40]]
        yticks = [[-40, -30, -20, -10, 0, 10, 20, 30, 40],
                  [-15, -10, -5, 0, 5, 10],
                  [-15, -10, -5, 0, 5, 10],
                  [0, 20, 40, 60, 80],
                  [-20, -10, 0, 10, 20],
                  [-10, -5, 0, 5, 10],
                  [0, 10, 20, 30, 40]]
        unperturbed_dict = dict()
        experiment_dict = dict()
        for ic, coord in enumerate(coordinates):
            for side in ['l', 'r']:
                key = f'/jointset/{joints[ic]}_{side}/{coord}_{side}/value'
                unperturbed_dict[key] = np.zeros((N, numSubjects))
                experiment_dict[key] = np.zeros((N, numSubjects))

        for i in np.arange(numSubjects):
            unperturbed = osim.TimeSeriesTable(file_dep[i])
            utime = np.array(unperturbed.getIndependentColumn())
            utime_interp = np.linspace(utime[0], utime[-1], N)
            experiment = osim.TimeSeriesTable(file_dep[i+numSubjects])
            etime = np.array(experiment.getIndependentColumn())
            istart = np.argmin(abs(etime-utime[0]))
            iend = np.argmin(abs(etime-utime[-1]))
            etime_interp = np.linspace(etime[istart], etime[iend], N)

            for ic, coord in enumerate(coordinates):
                for side in ['l', 'r']:
                    key = f'/jointset/{joints[ic]}_{side}/{coord}_{side}/value'
                    unperturbed_col = (180.0 / np.pi) * util.simtk2numpy(
                        unperturbed.getDependentColumn(key))
                    experiment_col = (180.0 / np.pi) * util.simtk2numpy(
                        experiment.getDependentColumn(key))[istart:iend+1]
                    unperturbed_dict[key][:, i] = np.interp(
                        utime_interp, utime, unperturbed_col)
                    experiment_dict[key][:, i] = np.interp(
                        etime_interp, etime[istart:iend+1], experiment_col)

        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(6, 10))
        gs = gridspec.GridSpec(len(coordinates), 2)
        for ic, coord in enumerate(coordinates):
            for iside, side in enumerate(['l','r']):
                key = f'/jointset/{joints[ic]}_{side}/{coord}_{side}/value'
                ax = fig.add_subplot(gs[ic, iside])
                exp_mean = np.mean(experiment_dict[key], axis=1)
                exp_std = np.std(experiment_dict[key], axis=1)
                unp_mean = np.mean(unperturbed_dict[key], axis=1)
                unp_std = np.std(unperturbed_dict[key], axis=1)

                if (not 'subtalar' in coord) and (not 'mtp' in coord):
                    h_exp, = ax.plot(pgc, exp_mean, color=self.exp_color, lw=2.5)
                    ax.fill_between(pgc, exp_mean + exp_std, exp_mean - exp_std, color=self.exp_color,
                        alpha=0.3, linewidth=0.0, edgecolor='none')
                h_unp, = ax.plot(pgc, unp_mean, color=self.unp_color, lw=2)
                ax.fill_between(pgc, unp_mean + unp_std, unp_mean - unp_std, color=self.unp_color,
                    alpha=0.3, linewidth=0.0, edgecolor='none')
                if (not 'knee' in coord) and (not 'mtp' in coord):
                    ax.axhline(y=0, color='black', linestyle='-', zorder=0, lw=0.5)
                ax.set_ylim(bounds[ic][0], bounds[ic][1])
                ax.set_yticks(yticks[ic])
                ax.set_xlim(0, 100)
                ax.grid(color='black', zorder=0, alpha=0.2, lw=0.5, ls='--', clip_on=False)
                util.publication_spines(ax)

                if not ic and not iside:
                    ax.legend([h_exp, h_unp], ['experiment', 'simulation'],
                        fancybox=False, frameon=True)

                if ic == len(coordinates)-1:
                    ax.set_xlabel('time [% gait cycle]')
                    ax.spines['bottom'].set_position(('outward', 10))
                else:
                    ax.spines['bottom'].set_visible(False)
                    ax.set_xticklabels([])
                    ax.tick_params(axis='x', which='both', bottom=False, 
                                top=False, labelbottom=False)

                if not iside:
                    ax.spines['left'].set_position(('outward', 10))
                    ax.set_ylabel(f'{labels[ic]} ' + r'$[^\circ]$')
                else:
                    ax.spines['left'].set_visible(False)
                    ax.set_yticklabels([])
                    ax.tick_params(axis='y', which='both', left=False, 
                                right=False, labelbottom=False)
                if not ic:
                    if 'l' in side:
                        ax.set_title('left leg')
                    elif 'r' in side:
                        ax.set_title('right leg')

        fig.tight_layout()
        fig.savefig(target[0], dpi=600)
        fig.savefig(target[1], dpi=600)
        plt.close()

    def plot_unperturbed_grfs(self, file_dep, target):

        numSubjects = len(self.subjects)
        N = 101
        pgc = np.linspace(0, 100, N)
        labels = ['anterior-posterior', 'vertical', 'medio-lateral']
        forces = ['vx', 'vy', 'vz']
        bounds = [[-0.25, 0.25],
                  [0, 1.25],
                  [-0.1, 0.1]]
        tick_intervals = [0.1, 0.25, 0.05]
        unperturbed_dict = dict()
        experiment_dict = dict()
        for force in forces:
            for side in ['l', 'r']:
                unperturbed_dict[f'ground_force_{side}_{force}'] = np.zeros((N, numSubjects))
                experiment_dict[f'ground_force_{side}_{force}'] = np.zeros((N, numSubjects))

        for isubj in np.arange(numSubjects):
            unperturbed = osim.TimeSeriesTable(file_dep[isubj])
            utime = np.array(unperturbed.getIndependentColumn())
            utime_interp = np.linspace(utime[0], utime[-1], N)
            experiment = osim.TimeSeriesTable(file_dep[isubj+numSubjects])
            etime = np.array(experiment.getIndependentColumn())
            istart = np.argmin(abs(etime-utime[0]))
            iend = np.argmin(abs(etime-utime[-1]))
            etime_interp = np.linspace(etime[istart], etime[iend], N)
            BW = 9.81 * self.masses[isubj]

            for iforce, force in enumerate(forces):
                for side in ['l', 'r']:
                    label = f'ground_force_{side}_{force}'
                    unperturbed_col = util.simtk2numpy(unperturbed.getDependentColumn(label)) / BW
                    experiment_col = util.simtk2numpy(
                        experiment.getDependentColumn(label))[istart:iend+1] / BW

                    unperturbed_dict[label][:, isubj] = np.interp(
                        utime_interp, utime, unperturbed_col)
                    experiment_dict[label][:, isubj] = np.interp(
                        etime_interp, etime[istart:iend+1], experiment_col)

        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(len(forces), 2)
        for iforce, force in enumerate(forces):
            for iside, side in enumerate(['l', 'r']):
                label = f'ground_force_{side}_{force}'
                ax = fig.add_subplot(gs[iforce, iside])

                exp_mean = np.mean(experiment_dict[label], axis=1)
                exp_std = np.std(experiment_dict[label], axis=1)
                unp_mean = np.mean(unperturbed_dict[label], axis=1)
                unp_std = np.std(unperturbed_dict[label], axis=1)

                h_exp, = ax.plot(pgc, exp_mean, color=self.exp_color, lw=2.5)
                ax.fill_between(pgc, exp_mean + exp_std, exp_mean - exp_std, color=self.exp_color,
                    alpha=0.3, linewidth=0.0, edgecolor='none')
                h_unp, = ax.plot(pgc, unp_mean, color=self.unp_color, lw=2)
                ax.fill_between(pgc, unp_mean + unp_std, unp_mean - unp_std, color=self.unp_color,
                    alpha=0.3, linewidth=0.0, edgecolor='none')
                if not 'vy' in force:
                    ax.axhline(y=0, color='black', linestyle='-', zorder=0, lw=0.5)
                ax.set_ylim(bounds[iforce][0], bounds[iforce][1])
                ax.set_yticks(np.linspace(bounds[iforce][0], bounds[iforce][1], 
                    int((bounds[iforce][1] - bounds[iforce][0]) / tick_intervals[iforce]) + 1))
                ax.set_xlim(0, 100)
                ax.grid(color='black', zorder=0, alpha=0.2, lw=0.5, ls='--', clip_on=False)
                util.publication_spines(ax)

                if not iforce and not iside:
                    ax.legend([h_exp, h_unp], ['experiment', 'simulation'],
                        fancybox=False, frameon=True)

                if iforce == len(forces)-1:
                    ax.set_xlabel('time (% gait cycle)')
                    ax.spines['bottom'].set_position(('outward', 10))
                else:
                    ax.spines['bottom'].set_visible(False)
                    ax.set_xticklabels([])
                    ax.tick_params(axis='x', which='both', bottom=False, 
                                top=False, labelbottom=False)

                if not iside:
                    ax.spines['left'].set_position(('outward', 10))
                    ax.set_ylabel(f'{labels[iforce]} ' + r'$[BW]$')
                else:
                    ax.spines['left'].set_visible(False)
                    ax.set_yticklabels([])
                    ax.tick_params(axis='y', which='both', left=False, 
                                right=False, labelbottom=False)

                if not iforce:
                    if 'l' in side:
                        ax.set_title('left leg')
                    elif 'r' in side:
                        ax.set_title('right leg')

        fig.tight_layout()
        fig.savefig(target[0], dpi=600)
        fig.savefig(target[1], dpi=600)
        plt.close()

    def plot_unperturbed_muscle_activity(self, file_dep, target):

        numSubjects = len(self.subjects)
        N = 200
        pgc = np.linspace(0, 100, N)
        labels = ['glut. max.', 'glut. med.', 'rec. fem.', 'vas. int.', 'semimem.',
                  'gas. med.', 'soleus', 'tib. ant.']
        activations = ['glmax2', 'glmed1', 'recfem', 'vasint', 'semimem', 'gasmed', 
                       'soleus', 'tibant']

        unperturbed_dict = dict()
        experiment_dict = dict()
        for activation in activations:
            for side in ['l', 'r']:
                unperturbed_dict[f'{activation}_{side}'] = np.zeros((N, numSubjects))
                experiment_dict[f'{activation}_{side}'] = np.zeros((N, numSubjects))

        for isubj in np.arange(numSubjects):
            unperturbed = osim.TimeSeriesTable(file_dep[isubj])
            utime = np.array(unperturbed.getIndependentColumn())
            utime_interp = np.linspace(utime[0], utime[-1], N)
            experiment = osim.TimeSeriesTable(file_dep[isubj+numSubjects])
            etime = np.array(experiment.getIndependentColumn())
            istart = np.argmin(abs(etime-utime[0]))
            iend = np.argmin(abs(etime-utime[-1]))
            etime_interp = np.linspace(etime[istart], etime[iend], N)

            for iact, activation in enumerate(activations):
                for side in ['l', 'r']:
                    label = f'{activation}_{side}'
                    unperturbed_col = util.simtk2numpy(unperturbed.getDependentColumn(
                        f'/forceset/{label}/activation'))
                    unperturbed_dict[label][:, isubj] = np.interp(
                        utime_interp, utime, unperturbed_col)

                    if 'r' in side:
                        experiment_col = util.simtk2numpy(
                            experiment.getDependentColumn(self.emg_map[label]))[istart:iend+1]
                        max_emg = np.max(experiment_col)
                        experiment_col_temp = experiment_col - 0.05
                        experiment_col_rescale = experiment_col_temp * (max_emg / np.max(experiment_col_temp))
                        experiment_dict[label][:, isubj] = np.interp(
                            etime_interp, etime[istart:iend+1], experiment_col_rescale)

        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(6, 8))
        gs = gridspec.GridSpec(len(activations), 2)
        for iact, activation in enumerate(activations):
            for iside, side in enumerate(['l', 'r']):
                label = f'{activation}_{side}'
                ax = fig.add_subplot(gs[iact, iside])
                
                unp_mean = np.mean(unperturbed_dict[label], axis=1)
                unp_std = np.std(unperturbed_dict[label], axis=1)
                h_unp, = ax.plot(pgc, unp_mean, color=self.unp_color, lw=2)
                ax.fill_between(pgc, unp_mean + unp_std, unp_mean - unp_std, color=self.unp_color,
                    alpha=0.3, linewidth=0.0, edgecolor='none')

                if 'r' in side:
                    exp_mean = np.mean(experiment_dict[label], axis=1)
                    exp_std = np.std(experiment_dict[label], axis=1)
                    h_exp, = ax.plot(pgc, exp_mean, color=self.exp_color, lw=2.5)
                    ax.fill_between(pgc, exp_mean + exp_std, exp_mean - exp_std, color=self.exp_color,
                        alpha=0.3, linewidth=0.0, edgecolor='none')
                
                # ax.axhline(y=0, color='black', alpha=0.4, linestyle='--', zorder=0, lw=0.75)
                ax.set_ylim(0, 1)
                ax.set_yticks([0, 0.5, 1])
                ax.set_xlim(0, 100)
                ax.grid(color='black', zorder=0, alpha=0.2, lw=0.5, ls='--', clip_on=False)
                util.publication_spines(ax)

                if not iact and ('r' in side):
                    ax.legend([h_exp, h_unp], ['experiment', 'simulation'],
                        fancybox=False, frameon=True)

                if iact == len(activations)-1:
                    ax.set_xlabel('time (% gait cycle)')
                    ax.spines['bottom'].set_position(('outward', 10))
                else:
                    ax.spines['bottom'].set_visible(False)
                    ax.set_xticklabels([])
                    ax.tick_params(axis='x', which='both', bottom=False, 
                                top=False, labelbottom=False)

                if not iside:
                    ax.spines['left'].set_position(('outward', 10))
                    ax.set_ylabel(f'{labels[iact]}')
                else:
                    ax.spines['left'].set_visible(False)
                    ax.set_yticklabels([])
                    ax.tick_params(axis='y', which='both', left=False, 
                                right=False, labelbottom=False)

                if not iact:
                    if 'l' in side:
                        ax.set_title('left leg')
                    elif 'r' in side:
                        ax.set_title('right leg')

        fig.tight_layout()
        fig.savefig(target[0], dpi=600)
        fig.savefig(target[1], dpi=600)
        plt.close()

    def plot_unperturbed_center_of_mass(self, file_dep, target):

        numSubjects = len(self.subjects)
        N = 200
        dims = ['x', 'y', 'z']
        unperturbed_dict = dict()
        experiment_dict = dict()
        for dim in dims:
            unperturbed_dict[f'/|com_position_{dim}'] = np.zeros((N, numSubjects))
            experiment_dict[f'/|com_position_{dim}'] = np.zeros((N, numSubjects))

        for isubj in np.arange(numSubjects):
            unperturbed = osim.TimeSeriesTable(file_dep[isubj])
            utime = np.array(unperturbed.getIndependentColumn())
            utime_interp = np.linspace(utime[0], utime[-1], N)
            experiment = osim.TimeSeriesTable(file_dep[isubj+numSubjects])
            etime = np.array(experiment.getIndependentColumn())
            istart = np.argmin(abs(etime-utime[0]))
            iend = np.argmin(abs(etime-utime[-1]))
            etime_interp = np.linspace(etime[istart], etime[iend], N)

            for idim, dim in enumerate(dims):
                label = f'/|com_position_{dim}'
                unperturbed_col = util.simtk2numpy(unperturbed.getDependentColumn(label))
                experiment_col = util.simtk2numpy(
                    experiment.getDependentColumn(label))[istart:iend+1]

                unperturbed_dict[label][:, isubj] = np.interp(
                    utime_interp, utime, unperturbed_col - experiment_col[0])
                experiment_dict[label][:, isubj] = np.interp(
                    etime_interp, etime[istart:iend+1], experiment_col - experiment_col[0])

        fig = plt.figure(figsize=(4, 6))
        ax = fig.add_subplot(1,1,1)
        exp_x_mean = np.mean(experiment_dict['/|com_position_x'], axis=1)
        unp_x_mean = np.mean(unperturbed_dict['/|com_position_x'], axis=1)
        # exp_x_std = np.std(experiment_dict['/|com_position_x'], axis=1)
        # unp_x_std = np.std(unperturbed_dict['/|com_position_x'], axis=1)
        exp_z_mean = np.mean(experiment_dict['/|com_position_z'], axis=1)
        unp_z_mean = np.mean(unperturbed_dict['/|com_position_z'], axis=1)
        exp_z_std = np.std(experiment_dict['/|com_position_z'], axis=1)
        unp_z_std = np.std(unperturbed_dict['/|com_position_z'], axis=1)

        h_exp, = ax.plot(exp_z_mean, exp_x_mean, color=self.exp_color, lw=2.5)
        h_unp, = ax.plot(unp_z_mean, unp_x_mean, color=self.unp_color, lw=2)
        ax.grid(color='black', zorder=0, alpha=0.2, lw=0.5, ls='--')
        # ax.fill_between(exp_z_mean, exp_x_mean + exp_x_std, exp_x_mean - exp_x_std, 
        #     color=self.exp_color, alpha=0.3, linewidth=0.0, edgecolor='none')
        # ax.fill_between(unp_z_mean, unp_x_mean + unp_x_std, unp_x_mean - unp_x_std, 
        #     color=self.unp_color, alpha=0.3, linewidth=0.0, edgecolor='none')
        ax.fill_betweenx(exp_x_mean, exp_z_mean + exp_z_std, exp_z_mean - exp_z_std, 
            color=self.exp_color, alpha=0.3, linewidth=0.0, edgecolor='none')
        ax.fill_betweenx(unp_x_mean, unp_z_mean + unp_z_std, unp_z_mean - unp_z_std, 
            color=self.unp_color, alpha=0.3, linewidth=0.0, edgecolor='none')
        ax.set_ylim(-0.1, 1.5)
        ax.set_yticks([-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
        ax.set_xlim(-0.03, 0.04)
        ax.set_xticks([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04])

        s = 35
        ax.scatter(exp_z_mean[0], exp_x_mean[0], 
            s=s, color=self.exp_color, marker='o', clip_on=False)
        ax.scatter(exp_z_mean[-1], exp_x_mean[-1], 
            s=s, color=self.exp_color, marker='X', clip_on=False)
        ax.scatter(unp_z_mean[0], unp_x_mean[0], 
            s=s, color=self.unp_color, marker='o', clip_on=False)
        ax.scatter(unp_z_mean[-1], unp_x_mean[-1], 
            s=s, color=self.unp_color, marker='X', clip_on=False)

        util.publication_spines(ax)
        ax.legend([h_exp, h_unp], ['experiment', 'unperturbed'],
            fancybox=False, frameon=True)
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))
        ax.set_xlabel(r'medio-lateral position $[m]$')
        ax.set_ylabel(r'anterior-posterior position $[m]$')

        fig.tight_layout()
        fig.savefig(target[0], dpi=600)
        plt.close()
   
    def plot_unperturbed_step_widths(self, file_dep, target): 

        numSubjects = len(self.subjects)
        N = 100
        unperturbed_dict = dict()
        positions = ['x', 'y', 'z']
        for ip, pos in enumerate(positions):
            for side in ['l', 'r']:
                unperturbed_dict[f'pos_{pos}_{side}'] = np.zeros((N, numSubjects))

        for i in np.arange(numSubjects):
            unperturbed = osim.MocoTrajectory(file_dep[i])
            model = osim.Model(self.models[i])
            model.initSystem()

            for side in ['l', 'r']:
                posTable = osim.analyzeVec3(
                    model, unperturbed.exportToStatesTable(),
                    unperturbed.exportToControlsTable(),
                    [f'.*calcn_{side}\|position']).flatten(['_x','_y','_z'])
                utime = np.array(posTable.getIndependentColumn())
                utime_interp = np.linspace(utime[0], utime[-1], N)

                for ip, pos in enumerate(positions):
                    pos_elt = util.simtk2numpy(
                        posTable.getDependentColumn(
                            f'/bodyset/calcn_{side}|position_{pos}'))
                    unperturbed_dict[f'pos_{pos}_{side}'][:, i] = np.interp(
                        utime_interp, utime, pos_elt)

        fig = plt.figure(figsize=(3, 5))
        ax = fig.add_subplot(1,1,1)
        step_widths = list()
        for i in np.arange(numSubjects):

            time_r = 22
            time_l = 72
            pos_z_r = unperturbed_dict['pos_z_r'][time_r, i]
            pos_z_l = unperturbed_dict['pos_z_l'][time_l, i]
            width = pos_z_r - pos_z_l
            step_widths.append(width)

            h_unp, = ax.bar(i+1, width, color=self.unp_color, zorder=2)
            util.publication_spines(ax)
            ax.set_xlabel('subject')
            ax.set_xlim(0.5, 5.5)
            ax.margins(x=0)
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_ylim(0, 0.35)
            y1 = 0.124*np.ones(7)
            y2 = 0.212*np.ones(7) 
            ax.fill_between([0, 1, 2, 3, 4, 5, 6], y1, y2, 
                color='lightgray', 
                alpha=0.5, zorder=0)
            ax.axhline(y=0.168, xmin=0, xmax=6, color='black', 
                zorder=1, alpha=0.8, lw=0.8)
            ax.axhline(y=0.08, xmin=0, xmax=6, color='black', 
                zorder=1, alpha=0.2, ls='--', lw=0.8)
            ax.axhline(y=0.32, xmin=0, xmax=6, color='black', 
                zorder=1, alpha=0.2, ls='--', lw=0.8)
            ax.spines['bottom'].set_position(('outward', 10))
            ax.spines['left'].set_position(('outward', 10))
            ax.set_ylabel('step width (m)')

        fig.tight_layout()
        fig.savefig(target[0], dpi=600)
        plt.close()

        with open(target[1], 'w') as f:
            f.write('Step widths (meters), mean +/- std across subjects\n')
            f.write('\n')
            f.write(f'Step widths: {np.mean(step_widths):.2f} +/- {np.std(step_widths):.2f} m\n')

    def compute_unperturbed_gait_landmarks(self, file_dep, target):

        # Aggregate data
        # --------------
        import collections
        unp_dict = collections.defaultdict(dict)
        N = 202
        pgc = np.linspace(0, 100, N)
        for force in ['total'] + self.forces:
            unp_dict[force]['x'] = np.zeros((N, len(self.subjects)))
            unp_dict[force]['y'] = np.zeros((N, len(self.subjects)))
            unp_dict[force]['z'] = np.zeros((N, len(self.subjects)))

        for isubj, subject in enumerate(self.subjects):
            model = osim.Model(self.models[isubj])
            state = model.initSystem()

            unperturbed = osim.MocoTrajectory(file_dep[isubj])
            timeVec = unperturbed.getTimeMat()
            timeInterp = np.linspace(timeVec[0], timeVec[-1], N)

            for force in self.forces:
                sshsForce = osim.SmoothSphereHalfSpaceForce.safeDownCast(
                    model.getComponent(f'/forceset/{force}'))

                # Unperturbed
                grfx = np.zeros_like(timeVec)
                grfy = np.zeros_like(timeVec)
                grfz = np.zeros_like(timeVec)
                statesTraj = unperturbed.exportToStatesTrajectory(model)
                for istate in np.arange(statesTraj.getSize()):
                    state = statesTraj.get(int(istate))
                    model.realizeVelocity(state)
                    forceValues = sshsForce.getRecordValues(state)

                    grfx[istate] = forceValues.get(0)
                    grfy[istate] = forceValues.get(1)
                    grfz[istate] = forceValues.get(2)

                unp_dict[force]['x'][:, isubj] = np.interp(timeInterp, timeVec, grfx)
                unp_dict[force]['y'][:, isubj] = np.interp(timeInterp, timeVec, grfy)
                unp_dict[force]['z'][:, isubj] = np.interp(timeInterp, timeVec, grfz)

        for force in self.forces:
            for direc in ['x', 'y', 'z']:
                unp_dict['total'][direc] += unp_dict[force][direc]

        # Detect foot-flat
        # ----------------
        threshold = 5.0 # Newtons
        footFlats = np.zeros(len(self.subjects))
        for isubj, subject in enumerate(self.subjects):
            for i in np.arange(N):
                allSpheresInContact = True
                for force in self.forces:
                    if unp_dict[force]['y'][i, isubj] < threshold:
                        allSpheresInContact = False

                if (i > 10) and allSpheresInContact:
                    footFlats[isubj] = pgc[i]
                    break

        # Detect heel-off
        # ---------------
        threshold = 5.0 # Newtons
        heelOffs = np.zeros(len(self.subjects))
        for isubj, subject in enumerate(self.subjects):
            vertHeelForce = unp_dict['contactHeel_r']['y'][:, isubj]
            vertHeelDetect = vertHeelForce > threshold
            for i in np.arange(N):
                if (i > 10) and (not vertHeelDetect[i]):
                    heelOffs[isubj] = pgc[i]
                    break

        # Detect toe-off
        # --------------
        threshold = 5.0 # Newtons
        toeOffs = np.zeros(len(self.subjects))
        for isubj, subject in enumerate(self.subjects):
            vertForce = unp_dict['total']['y'][:, isubj]
            vertDetect = vertForce > threshold
            for i in np.arange(N):
                if (i > 10) and (not vertDetect[i]):
                    toeOffs[isubj] = pgc[i]
                    break

        with open(target[0], 'w') as f:
            f.write('Times in percent gait cycle, mean +/- std across subjects\n')
            f.write('\n')
            f.write(f'Foot-flat time: {np.mean(footFlats):.1f} +/- {np.std(footFlats):.1f} %\n')
            f.write(f'Heel-off time: {np.mean(heelOffs):.1f} +/- {np.std(heelOffs):.1f} %\n')
            f.write(f'Toe-off time: {np.mean(toeOffs):.1f} +/- {np.std(toeOffs):.1f} %\n')


class TaskComputeObjectiveContributions(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects):
        super(TaskComputeObjectiveContributions, self).__init__(study)
        self.name = 'compute_objective_contributions'
        self.results_path = os.path.join(study.config['results_path'], 
            'unperturbed')
        self.validate_path = os.path.join(study.config['validate_path'],
            'objective_contributions')
        if not os.path.exists(self.validate_path): 
            os.makedirs(self.validate_path)
        self.subjects = subjects
        self.objective_names = ['control_effort',
                                'contact',
                                'state_tracking',
                                'torso_orientation_goal',
                                'torso_angular_velocity_goal',
                                'feet_orientation_goal',
                                'feet_angular_velocity_goal']

        unperturbed_fpaths = list()
        for subject in subjects:
            unperturbed_fpaths.append(os.path.join(
                self.study.config['results_path'], 'unperturbed', 
                subject, 'unperturbed.sto'))

        self.add_action(unperturbed_fpaths, 
                        [os.path.join(self.validate_path, 
                            'objective_contributions.txt')], 
                        self.compute_objective_contributions)

    def compute_objective_contributions(self, file_dep, target):

        import collections
        terms = collections.defaultdict(list)
        for i in np.arange(len(self.subjects)):
            unperturbed = osim.TimeSeriesTable(file_dep[i])

            objective = float(unperturbed.getTableMetaDataString('objective'))

            control_effort = float(unperturbed.getTableMetaDataString(
                'objective_control_effort'))
            terms['control_effort'].append(100.0 * (control_effort / objective))

            contact = float(unperturbed.getTableMetaDataString(
                'objective_contact'))
            terms['contact'].append(100.0 * (contact / objective))

            state_tracking = float(unperturbed.getTableMetaDataString(
                'objective_state_tracking'))
            terms['state_tracking'].append(100.0 * (state_tracking / objective))

            torso_orientation_goal = float(unperturbed.getTableMetaDataString(
                'objective_torso_orientation_goal'))
            torso_angular_velocity_goal = float(unperturbed.getTableMetaDataString(
                'objective_torso_angular_velocity_goal'))
            torso_tracking = torso_orientation_goal + torso_angular_velocity_goal
            terms['torso_tracking'].append(100.0 * (torso_tracking / objective))

            feet_orientation_goal = float(unperturbed.getTableMetaDataString(
                'objective_feet_orientation_goal'))
            feet_angular_velocity_goal = float(unperturbed.getTableMetaDataString(
                'objective_feet_angular_velocity_goal'))
            feet_tracking = feet_orientation_goal + feet_angular_velocity_goal
            terms['feet_tracking'].append(100.0 * (feet_tracking / objective))

        with open(target[0], 'w') as f:
            f.write('Objective function contributions\n')
            f.write('--------------------------------\n')
            for key, value in terms.items():
                f.write(f' -- {key}: {np.mean(value):.1f}'
                        f' +/- {np.std(value):.1f} [%] \n')

# Validation
# ----------

class TaskValidateMarkerErrors(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, cond_names=['walk2']):
        super(TaskValidateMarkerErrors, self).__init__(study)
        self.name = 'validate_marker_errors'
        self.doc = 'Compute marker errors across subjects and conditions.'
        self.results_path = study.config['results_path']
        self.validate_path = os.path.join(study.config['validate_path'],
            'marker_errors')
        self.cond_names = cond_names
        self.subjects = study.subjects

        errors_fpaths = list()
        for cond_name in cond_names:
            for subject in study.subjects:
                errors_fpaths.append(os.path.join(self.results_path, 
                    'experiments', subject.name, cond_name, 'ik', 
                    f'{study.name}_{subject.name}_{cond_name}_ik_ik_marker_errors.sto'))
                
        val_fname = os.path.join(self.validate_path, 
            'marker_errors.txt')
        self.add_action(errors_fpaths, [val_fname],
                        self.validate_marker_errors)

    def validate_marker_errors(self, file_dep, target):
        if not os.path.isdir(self.validate_path): 
            os.mkdir(self.validate_path)

        mean_rms_errors = list()
        max_marker_errors = list()
        for dep in file_dep:
            marker_errors = osim.TimeSeriesTable(dep)
            rms_errors = marker_errors.getDependentColumn(
                'marker_error_RMS').to_numpy()
            mean_rms_errors.append(np.mean(rms_errors))

            max_errors = marker_errors.getDependentColumn(
                'marker_error_max').to_numpy()
            max_marker_errors.append(np.max(max_errors))

        with open(target[0], 'w') as f:
            f.write('subjects: ')
            for isubj, subject in enumerate(self.subjects):
                if isubj:
                    f.write(', %s' % subject.name)
                else:
                    f.write('%s' % subject.name)
            f.write('\n')
            f.write('conditions: ')
            for icond, cond_name in enumerate(self.cond_names):
                if icond:
                    f.write(', %s' % subject)
                else:
                    f.write('%s' % cond_name)
            f.write('\n')
            f.write('Max overall marker error (avg. across subj): %1.3f cm \n' 
                    % (100*np.mean(max_marker_errors)))
            f.write('Mean RMS marker error (avg. across subj): %1.3f cm \n'
                    % (100*np.mean(mean_rms_errors)))


class TaskValidateTrackingErrors(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, masses, times):
        super(TaskValidateTrackingErrors, self).__init__(study)
        self.config_name = 'unperturbed'
        self.name = 'validate_tracking_errors'
        self.results_path = os.path.join(study.config['results_path'], 
            'unperturbed')
        self.validate_path = os.path.join(study.config['validate_path'],
            'tracking_errors')
        if not os.path.exists(self.validate_path): 
            os.makedirs(self.validate_path)
        self.subjects = subjects
        self.masses = masses
        self.models = list()
        self.times = times

        unperturbed_fpaths = list()
        unperturbed_grf_fpaths = list()
        coordinates_fpaths = list()
        grf_fpaths = list()

        for subject in subjects:
            self.models.append(os.path.join(
                self.study.config['results_path'], self.config_name, 
                subject, f'model_{self.config_name}.osim'))
            expdata_dir = os.path.join(
                study.config['results_path'], 'experiments', subject, 'walk2')
            coordinates_fpaths.append(os.path.join(
                self.study.config['results_path'], self.config_name, 
                subject, f'{self.config_name}_experiment_states.sto'))
            grf_fpaths.append(os.path.join(
                expdata_dir, 'tracking_data', 'expdata', 'ground_reaction.mot'))
            unperturbed_fpaths.append(os.path.join(
                self.study.config['results_path'], self.config_name, 
                subject, f'{self.config_name}.sto'))
            unperturbed_grf_fpaths.append(os.path.join(
                self.study.config['results_path'], self.config_name, 
                subject, f'{self.config_name}_grfs.sto'))

        self.add_action(unperturbed_fpaths + coordinates_fpaths, 
                        [os.path.join(self.validate_path, 
                            'coordinate_tracking_errors.txt')], 
                        self.compute_coordinate_errors)

        self.add_action(unperturbed_grf_fpaths + grf_fpaths, 
                        [os.path.join(self.validate_path, 
                            'grf_tracking_errors.txt')], 
                        self.compute_grf_errors)

    def compute_coordinate_errors(self, file_dep, target): 

        numSubjects = len(self.subjects)
        N = 101
        coordinates = ['pelvis_tilt',
                       'pelvis_list',
                       'pelvis_rotation',
                       'pelvis_tx',
                       'pelvis_ty',
                       'pelvis_tz',
                       'lumbar_extension',
                       'lumbar_bending',
                       'lumbar_rotation']
        joints = ['ground_pelvis',
                  'ground_pelvis',        
                  'ground_pelvis',        
                  'ground_pelvis',        
                  'ground_pelvis',        
                  'ground_pelvis',
                  'back',
                  'back',
                  'back']

        coordinates_lr = ['arm_flex',
                          'arm_add',
                          'arm_rot',
                          'elbow_flex',
                          'pro_sup',
                          'hip_flexion', 
                          'hip_adduction', 
                          'hip_rotation', 
                          'knee_angle', 
                          'ankle_angle']
        joints_lr = ['acromial',        
                     'acromial',        
                     'acromial',
                     'elbow',
                     'radioulnar',        
                     'hip', 
                     'hip', 
                     'hip', 
                     'walker_knee', 
                     'ankle']
        unperturbed_dict = dict()
        experiment_dict = dict()
        for ic, coord in enumerate(coordinates):
            key = f'/jointset/{joints[ic]}/{coord}/value'
            unperturbed_dict[key] = np.zeros((N, numSubjects))
            experiment_dict[key] = np.zeros((N, numSubjects))

        for ic, coord in enumerate(coordinates_lr):
            for side in ['l', 'r']:
                key = f'/jointset/{joints_lr[ic]}_{side}/{coord}_{side}/value'
                unperturbed_dict[key] = np.zeros((N, numSubjects))
                experiment_dict[key] = np.zeros((N, numSubjects))

        for i in np.arange(numSubjects):
            unperturbed = osim.TimeSeriesTable(file_dep[i])
            utime = np.array(unperturbed.getIndependentColumn())
            utime_interp = np.linspace(utime[0], utime[-1], N)
            experiment = osim.TimeSeriesTable(file_dep[i+numSubjects])
            etime = np.array(experiment.getIndependentColumn())
            istart = np.argmin(abs(etime-utime[0]))
            iend = np.argmin(abs(etime-utime[-1]))
            etime_interp = np.linspace(etime[istart], etime[iend], N)

            for ic, coord in enumerate(coordinates):
                key = f'/jointset/{joints[ic]}/{coord}/value'
                unperturbed_col = (180.0 / np.pi) * unperturbed.getDependentColumn(key).to_numpy()
                experiment_col = (180.0 / np.pi) * experiment.getDependentColumn(key).to_numpy()[istart:iend+1]
                unperturbed_dict[key][:, i] = np.interp(
                    utime_interp, utime, unperturbed_col)
                experiment_dict[key][:, i] = np.interp(
                    etime_interp, etime[istart:iend+1], experiment_col)

            for ic, coord in enumerate(coordinates_lr):
                for side in ['l', 'r']:
                    key = f'/jointset/{joints_lr[ic]}_{side}/{coord}_{side}/value'
                    unperturbed_col = (180.0 / np.pi) * unperturbed.getDependentColumn(key).to_numpy()
                    experiment_col = (180.0 / np.pi) * experiment.getDependentColumn(key).to_numpy()[istart:iend+1]
                    unperturbed_dict[key][:, i] = np.interp(
                        utime_interp, utime, unperturbed_col)
                    experiment_dict[key][:, i] = np.interp(
                        etime_interp, etime[istart:iend+1], experiment_col)

        import collections
        rmse_dict = collections.defaultdict(list)
        for ic, coord in enumerate(coordinates):
            key = f'/jointset/{joints[ic]}/{coord}/value'
            for isubj in np.arange(numSubjects):
                errors = experiment_dict[key][:, isubj] - unperturbed_dict[key][:, isubj]
                rmse = np.sqrt(np.sum(np.square(errors)) / N)
                rmse_dict[f'{coord}'].append(rmse)

        for ic, coord in enumerate(coordinates_lr):
            for iside, side in enumerate(['l','r']):
                key = f'/jointset/{joints_lr[ic]}_{side}/{coord}_{side}/value'
                for isubj in np.arange(numSubjects):
                    errors = experiment_dict[key][:, isubj] - unperturbed_dict[key][:, isubj]
                    rmse = np.sqrt(np.sum(np.square(errors)) / N)
                    rmse_dict[f'{coord}_{side}'].append(rmse)

        with open(target[0], 'w') as f:
            f.write('Coordinate tracking RMS errors (mean +/- std across subjects)\n')
            f.write('-------------------------------------------------------------\n')
            for key, value in rmse_dict.items():
                f.write(f' -- {key}: {np.mean(value):.2f} +/- {np.std(value):.2f} [deg]\n')

    def compute_grf_errors(self, file_dep, target):

        numSubjects = len(self.subjects)
        N = 101
        forces = ['vx', 'vy', 'vz']
        unperturbed_dict = dict()
        experiment_dict = dict()
        for force in forces:
            for side in ['l', 'r']:
                unperturbed_dict[f'ground_force_{side}_{force}'] = np.zeros((N, numSubjects))
                experiment_dict[f'ground_force_{side}_{force}'] = np.zeros((N, numSubjects))

        for isubj in np.arange(numSubjects):
            unperturbed = osim.TimeSeriesTable(file_dep[isubj])
            utime = np.array(unperturbed.getIndependentColumn())
            utime_interp = np.linspace(utime[0], utime[-1], N)
            experiment = osim.TimeSeriesTable(file_dep[isubj+numSubjects])
            etime = np.array(experiment.getIndependentColumn())
            istart = np.argmin(abs(etime-utime[0]))
            iend = np.argmin(abs(etime-utime[-1]))
            etime_interp = np.linspace(etime[istart], etime[iend], N)

            model = osim.Model(self.models[isubj])
            state = model.initSystem()
            mass = model.getTotalMass(state)
            gravity = model.getGravity()[1]
            BW = mass * np.abs(gravity)

            for iforce, force in enumerate(forces):
                for side in ['l', 'r']:
                    label = f'ground_force_{side}_{force}'
                    unperturbed_col = util.simtk2numpy(
                        unperturbed.getDependentColumn(label)) / BW
                    experiment_col = util.simtk2numpy(
                        experiment.getDependentColumn(label))[istart:iend+1] / BW

                    unperturbed_dict[label][:, isubj] = np.interp(
                        utime_interp, utime, unperturbed_col)
                    experiment_dict[label][:, isubj] = np.interp(
                        etime_interp, etime[istart:iend+1], experiment_col)

        import collections
        rmse_dict = collections.defaultdict(list)
        for iforce, force in enumerate(forces):
            for iside, side in enumerate(['l', 'r']):
                label = f'ground_force_{side}_{force}'
                for isubj in np.arange(numSubjects):
                    errors = experiment_dict[label][:, isubj] - unperturbed_dict[label][:, isubj]
                    rmse = np.sqrt(np.sum(np.square(errors)) / N)
                    rmse_dict[label].append(rmse)

        with open(target[0], 'w') as f:
            f.write('GRF tracking RMS errors (mean +/- std across subjects)\n')
            f.write('------------------------------------------------------\n')
            for key, value in rmse_dict.items():
                f.write(f' -- {key}: {100.0*np.mean(value):.2f}'
                        f' +/- {100.0*np.std(value):.2f} [%BW] \n')


class TaskValidateMuscleActivity(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects):
        super(TaskValidateMuscleActivity, self).__init__(study)
        self.name = 'validate_muscle_activity'
        self.doc = 'Plot muscle activity from simulation against EMG data.'
        self.validate_path = os.path.join(study.config['validate_path'], 
            'muscle_activity')
        if not os.path.isdir(self.validate_path): os.mkdir(self.validate_path)
        self.subjects = subjects
        self.emg_names = ['bflh_r', 'gaslat_r', 'gasmed_r', 'glmax1_r', 
                        'glmax2_r', 'glmax3_r', 'glmed1_r', 'glmed2_r', 
                        'glmed3_r', 'recfem_r', 'semimem_r', 'semiten_r', 
                        'soleus_r', 'tibant_r', 'vaslat_r','vasmed_r']

        emg_fpaths = list()
        solution_fpaths = list()
        for subject in self.subjects:
            solution_fpaths.append(os.path.join(
                study.config['results_path'], 'unperturbed', 
                subject,'unperturbed.sto'))
            emg_fpaths.append(os.path.join(
                study.config['results_path'], 'experiments',
                subject, 'walk2', 'expdata', 'emg.sto'))
   
        self.add_action(emg_fpaths + solution_fpaths,
                        [os.path.join(self.validate_path, 
                            'muscle_activity.txt')],
                        self.validate_muscle_activity)

    def validate_muscle_activity(self, file_dep, target):

        def calc_rms_error(vec1, vec2):
            N = len(vec1)
            errors = vec1 - vec2
            sq_errors = np.square(errors)
            sumsq_errors = np.sum(sq_errors)
            return np.sqrt(sumsq_errors / N)

        numSubjects = len(self.subjects)
        numEMG = len(self.emg_names)

        emg_fpaths = file_dep[0:numSubjects]
        solution_fpaths = file_dep[numSubjects:]
        fpaths = zip(emg_fpaths, solution_fpaths)

        shape = (numEMG, numSubjects)
        act_peak_times = np.zeros(shape)
        act_peak_values = np.zeros(shape)
        emg_peak_times = np.zeros(shape)
        emg_peak_values = np.zeros(shape)
        onset_percent_errors = np.zeros(shape)
        # meansq_errors = np.zeros(shape)
        rms_errors = np.zeros(shape)
        for isubj, (emg_fpath, solution_fpath) in enumerate(fpaths):

            from collections import deque

            emg = util.storage2numpy(emg_fpath)
            time = emg['time']
            pgc = np.linspace(0, 100, 201)

            def min_index(vals):
                idx, val = min(enumerate(vals), key=lambda p: p[1])
                return idx

            muscle_array = list()
            emg_data = list()
            act_data = list()

            solution = osim.TimeSeriesTable(solution_fpath)
            solTime = np.array(solution.getIndependentColumn())
            start_idx = min_index(abs(time-solTime[0]))
            end_idx = min_index(abs(time-solTime[-1]))
            duration = solTime[-1] - solTime[0]

            for emg_name in self.emg_names:
                emg_interp = np.interp(pgc, 
                    np.linspace(0, 100, len(time[start_idx:end_idx])), 
                    emg[emg_name][start_idx:end_idx])
                emg_peak = max(emg_interp)
                scale = emg_peak / (emg_peak - 0.05);
                emg_interp_rescaled = scale * (emg_interp - 0.05);
                emg_data.append(emg_interp_rescaled)

                act = solution.getDependentColumn(
                    f'/forceset/{emg_name}/activation').to_numpy()
                act_interp = np.interp(pgc,
                    np.linspace(0, 100, len(solTime)), 
                    act)
                act_data.append(act_interp)

                muscle_array.append(emg_name)

            # Convert from n_muscles x n_times
            #         to   n_times x n_muscles
            emg_data_array = np.array(emg_data).transpose()
            act_data_array = np.array(act_data).transpose()
            columns = pd.MultiIndex.from_arrays([muscle_array], names=['muscle'])
            df_emg = pd.DataFrame(emg_data_array, columns=columns, index=pgc)
            df_act = pd.DataFrame(act_data_array, columns=columns, index=pgc)

            for iemg, emg_name in enumerate(self.emg_names):

                emg_values = df_emg[emg_name]
                act_values = df_act[emg_name]

                emg_peak_idx = np.argmax(emg_values.values)
                emg_peak_times[iemg, isubj] = emg_values.index[emg_peak_idx]
                emg_peak_values[iemg, isubj] = emg_values.values[emg_peak_idx]

                act_peak_idx = np.argmax(act_values.values)
                act_peak_times[iemg, isubj] = act_values.index[act_peak_idx]
                act_peak_values[iemg, isubj] = act_values.values[act_peak_idx]

                threshold = 0.05
                this_emg = np.array(emg_values.values.transpose()[0])
                this_act = np.array(act_values.values.transpose()[0])
                emg_bool = this_emg > threshold
                act_bool = this_act > threshold
                
                onset_diff = emg_bool.astype(int) - act_bool.astype(int)
                percent_error = np.abs(onset_diff).sum() / float(len(onset_diff))
                onset_percent_errors[iemg, isubj] = percent_error

                rms_errors[iemg, isubj] = calc_rms_error(this_emg, this_act)
                
        act_peak_times_mean = np.mean(act_peak_times, axis=1)
        act_peak_values_mean = np.mean(act_peak_values, axis=1)
        emg_peak_times_mean = np.mean(emg_peak_times, axis=1)
        emg_peak_values_mean = np.mean(emg_peak_values, axis=1)
        diff_peak_times = act_peak_times - emg_peak_times
        diff_peak_values = act_peak_values - emg_peak_values
        diff_peak_times_mean = np.mean(diff_peak_times, axis=1)
        diff_peak_values_mean = np.mean(diff_peak_values, axis=1)
        onset_percent_errors_mean = np.mean(onset_percent_errors, axis=1)
        rms_errors_mean = np.mean(rms_errors, axis=1)

        with open(target[0], 'w') as f:
            f.write('EMG versus simulation peak values: \n')
            for iemg, emg_name in enumerate(self.emg_names):
                f.write(f'{emg_name}: {emg_peak_values_mean[iemg]:.2f}, '
                        f'{emg_name}: {act_peak_values_mean[iemg]:.2f}, '
                        f'diff: {diff_peak_values_mean[iemg]:.2f}\n')
            f.write('\n')
            f.write('EMG versus simulation peak times: \n')
            for iemg, emg_name in enumerate(self.emg_names):
                f.write(f'{emg_name}: {emg_peak_times_mean[iemg]:.2f}, '
                        f'{emg_name}: {act_peak_times_mean[iemg]:.2f}, '
                        f'diff: {diff_peak_times_mean[iemg]:.2f}\n')
            f.write('\n')
            f.write('Mean onset-offset percent errors: \n')
            for iemg, emg_name in enumerate(self.emg_names):
                f.write(f'{emg_name}: {onset_percent_errors_mean[iemg]:.2f}\n')
            f.write('\n')
            f.write('RMS errors: \n')
            for iemg, emg_name in enumerate(self.emg_names):
                f.write(f'{emg_name}: {rms_errors_mean[iemg]:.2f}\n')


# Perturbed walking
# -----------------

class TaskMocoPerturbedWalking(osp.TrialTask):
    REGISTRY = []
    def __init__(self, trial, initial_time, final_time, right_strikes, 
                 left_strikes, walking_speed=1.25, side='right', 
                 torque_parameters=[0.5, 0.5, 0.25, 0.1],
                 subtalar_torque_perturbation=False, subtalar_peak_torque=0,
                 lumbar_stiffness=1.0, use_coordinate_actuators=False):
        super(TaskMocoPerturbedWalking, self).__init__(trial)
        torque = int(round(100*torque_parameters[0]))
        time = int(round(100*torque_parameters[1]))
        rise = int(round(100*torque_parameters[2]))
        fall = int(round(100*torque_parameters[3]))

        self.config_name = (f'perturbed_torque{torque}'
                            f'_time{time}_rise{rise}_fall{fall}')

        if subtalar_torque_perturbation:
            subtalar = int(round(100*subtalar_peak_torque))
            self.config_name += f'_subtalar{subtalar}'
        
        self.subpath = ''
        self.unperturbed_name = 'unperturbed'
        self.lumbar_stiffness = lumbar_stiffness
        self.use_coordinate_actuators = use_coordinate_actuators
        if ((not self.lumbar_stiffness == 1.0) and 
            self.use_coordinate_actuators):
            raise Exception('Cannot change lumbar stiffness and '
                            'use path actuators in same perturbed '
                            'configuration.')

        if not self.lumbar_stiffness == 1.0:
            self.subpath = f'lumbar{self.lumbar_stiffness}'
            self.config_name += f'_{self.subpath}'
            self.unperturbed_name += f'_{self.subpath}'

        elif self.use_coordinate_actuators:
            self.subpath = 'torque_actuators'
            self.config_name += f'_{self.subpath}'

        else:
            self.subpath = 'perturbed'

        self.name = f'{trial.subject.name}_moco_{self.config_name}'
        self.walking_speed = walking_speed
        self.mesh_interval = 0.01
        self.root_dir = trial.study.config['doit_path']
        self.weights = trial.study.weights
        self.initial_time = initial_time
        self.final_time = final_time
        self.right_strikes = right_strikes
        self.left_strikes = left_strikes
        self.model_fpath = trial.subject.sim_model_fpath
        self.ankle_torque_parameters = torque_parameters
        self.subtalar_torque_perturbation = subtalar_torque_perturbation
        self.subtalar_peak_torque = subtalar_peak_torque

        expdata_dir = os.path.join(
            trial.results_exp_path, 'tracking_data', 'expdata')
        extloads_dir = os.path.join(
            trial.results_exp_path, 'tracking_data', 'extloads')
        self.tracking_coordinates_fpath = os.path.join(
            expdata_dir, 'coordinates.sto')
        self.coordinates_std_fpath = os.path.join(
            trial.results_exp_path, 
            f'{trial.id}_joint_angle_standard_deviations.csv')
        self.tracking_extloads_fpath = os.path.join(
            extloads_dir, 'external_loads.xml')
        self.tracking_grfs_fpath = os.path.join(
            expdata_dir, 'ground_reaction.mot')
        self.emg_fpath = os.path.join(
            trial.results_exp_path, 'expdata', 'emg.sto')

        self.result_fpath = os.path.join(
            self.study.config['results_path'], self.subpath,
            self.config_name, trial.subject.name)
        if not os.path.exists(self.result_fpath): 
            os.makedirs(self.result_fpath)

        self.archive_fpath = os.path.join(
            self.study.config['results_path'], self.subpath, 
            self.config_name, trial.subject.name, 'archive')
        if not os.path.exists(self.archive_fpath): 
            os.makedirs(self.archive_fpath)

        self.solution_fpath = os.path.join(self.result_fpath, 
            f'{self.config_name}.sto')

        self.unperturbed_result_fpath = os.path.join(
            self.study.config['results_path'], self.unperturbed_name, 
            trial.subject.name)
        self.unperturbed_fpath = os.path.join(self.unperturbed_result_fpath, 
            f'{self.unperturbed_name}.sto')
        self.muscle_mechanics_fpath = os.path.join(self.unperturbed_result_fpath, 
            f'muscle_mechanics_{self.unperturbed_name}.sto')

        self.add_action([], [], self.copy_experiment_states)

        self.add_action([self.model_fpath,
                         self.tracking_coordinates_fpath,
                         self.coordinates_std_fpath,
                         self.tracking_extloads_fpath,
                         self.tracking_grfs_fpath,
                         self.emg_fpath,
                         self.muscle_mechanics_fpath,
                         self.unperturbed_fpath], 
                         [self.solution_fpath],
                        self.run_timestepping_problem)

    def copy_experiment_states(self, file_dep, target):
        shutil.copyfile(
            os.path.join(self.unperturbed_result_fpath, 
                f'{self.unperturbed_name}_experiment_states.sto'),
            os.path.join(self.result_fpath, 
                f'{self.unperturbed_name}_experiment_states.sto'))
        shutil.copyfile(
            os.path.join(self.unperturbed_result_fpath, 
                f'{self.unperturbed_name}_experiment_states.sto'),
            os.path.join(self.result_fpath, 
                f'{self.config_name}_experiment_states.sto'))


    def run_timestepping_problem(self, file_dep, target):

        config = TimeSteppingConfig(
            self.config_name, self.config_name, 'black', self.weights,
            unperturbed_fpath=file_dep[7],
            ankle_torque_perturbation=True,
            ankle_torque_parameters=self.ankle_torque_parameters,
            subtalar_torque_perturbation=self.subtalar_torque_perturbation,
            subtalar_peak_torque=self.subtalar_peak_torque,
            lumbar_stiffness=self.lumbar_stiffness,
            use_coordinate_actuators=self.use_coordinate_actuators)

        cycles = list()
        for cycle in self.trial.cycles:
            cycles.append([cycle.start, cycle.end])

        result = TimeSteppingProblem(
            self.root_dir,      # root directory
            self.result_fpath,  # result directory
            file_dep[0],        # model file path
            file_dep[1],        # IK coordinates path
            file_dep[2],        # Coordinate STD path
            file_dep[3],        # external loads file 
            file_dep[4],        # GRF MOT file
            file_dep[5],        # EMG data
            file_dep[6],        # muscle mechanics
            self.initial_time,
            self.final_time, 
            cycles,
            self.right_strikes,
            self.left_strikes,
            self.mesh_interval, 
            self.walking_speed,
            [config]
        )

        result.generate_results()
        result.report_results()


class TaskMocoPerturbedWalkingPost(osp.TrialTask):
    REGISTRY = []
    def __init__(self, trial, generate_task, **kwargs):
        super(TaskMocoPerturbedWalkingPost, self).__init__(trial)
        self.name = f'{generate_task.name}_post'
        self.weights = trial.study.weights
        self.root_dir = trial.study.config['doit_path']
        self.walking_speed = generate_task.walking_speed
        self.mesh_interval = generate_task.mesh_interval
        self.unperturbed_fpath = generate_task.unperturbed_fpath
        self.result_fpath = generate_task.result_fpath
        self.archive_fpath = generate_task.archive_fpath
        self.model_fpath = generate_task.model_fpath
        self.tracking_coordinates_fpath = \
            generate_task.tracking_coordinates_fpath
        self.coordinates_std_fpath = generate_task.coordinates_std_fpath
        self.tracking_extloads_fpath = generate_task.tracking_extloads_fpath
        self.tracking_grfs_fpath = generate_task.tracking_grfs_fpath
        self.emg_fpath = generate_task.emg_fpath
        self.muscle_mechanics_fpath = generate_task.muscle_mechanics_fpath
        self.ankle_torque_parameters = \
            generate_task.ankle_torque_parameters
        self.initial_time = generate_task.initial_time
        self.final_time = generate_task.final_time
        self.right_strikes = generate_task.right_strikes
        self.left_strikes = generate_task.left_strikes
        self.config_name = generate_task.config_name
        self.subtalar_torque_perturbation = generate_task.subtalar_torque_perturbation
        self.subtalar_peak_torque = generate_task.subtalar_peak_torque
        self.lumbar_stiffness = generate_task.lumbar_stiffness
        self.use_coordinate_actuators = generate_task.use_coordinate_actuators
        self.unperturbed_name = generate_task.unperturbed_name

        self.unperturbed_result_fpath = os.path.join(
            self.study.config['results_path'], self.unperturbed_name, 
            trial.subject.name)
        self.output_fpath = os.path.join(self.result_fpath, 
            f'tracking_walking_{self.unperturbed_name}_{self.config_name}_report.pdf')

        self.add_action([], [], self.copy_experiment_states)
        
        self.add_action([self.model_fpath,
                         self.tracking_coordinates_fpath,
                         self.coordinates_std_fpath,
                         self.tracking_extloads_fpath,
                         self.tracking_grfs_fpath,
                         self.emg_fpath,
                         self.muscle_mechanics_fpath,
                         self.unperturbed_fpath],
                        [self.output_fpath],
                        self.plot_timestepping_results)

    def copy_experiment_states(self, file_dep, target):

        # Copy over unperturbed solution so we can plot against the
        # perturbed solution
        shutil.copyfile(
            os.path.join(self.unperturbed_result_fpath, 
                f'{self.unperturbed_name}.sto'),
            os.path.join(self.result_fpath, 
                f'{self.unperturbed_name}.sto'))
        shutil.copyfile(
            os.path.join(self.unperturbed_result_fpath, 
                         f'{self.unperturbed_name}_grfs.sto'),
            os.path.join(self.result_fpath, 
                         f'{self.unperturbed_name}_grfs.sto'))


    def plot_timestepping_results(self, file_dep, target):

        configs = list()
        config = TimeSteppingConfig(
            self.unperturbed_name, self.unperturbed_name, 'black', self.weights,
            lumbar_stiffness=self.lumbar_stiffness)
        configs.append(config)

        config = TimeSteppingConfig(
            self.config_name, self.config_name, 'red', self.weights,
            ankle_torque_perturbation=True,
            ankle_torque_parameters=self.ankle_torque_parameters,
            subtalar_torque_perturbation=self.subtalar_torque_perturbation,
            subtalar_peak_torque=self.subtalar_peak_torque,
            lumbar_stiffness=self.lumbar_stiffness,
            use_coordinate_actuators=self.use_coordinate_actuators
            )
        configs.append(config)

        cycles = list()
        for cycle in self.trial.cycles:
            cycles.append([cycle.start, cycle.end])

        result = TimeSteppingProblem(
            self.root_dir,      # root directory
            self.result_fpath,  # result directory
            file_dep[0],        # model file path
            file_dep[1],        # IK coordinates path
            file_dep[2],        # Coordinates STD path
            file_dep[3],        # external loads file 
            file_dep[4],        # GRF MOT file
            file_dep[5],        # EMG data
            file_dep[6],        # muscle mechanics
            self.initial_time,
            self.final_time,
            cycles,
            self.right_strikes,
            self.left_strikes,
            self.mesh_interval, 
            self.walking_speed,
            configs,
        )

        result.report_results()


class TaskCreatePerturbedVisualization(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, time, torques, rise, fall):
        super(TaskCreatePerturbedVisualization, self).__init__(study)
        self.name = f'create_perturbed_visualization_time{time}_rise{rise}_fall{fall}'
        self.results_path = os.path.join(study.config['results_path'], 
            'unperturbed')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'perturbed_visualization', f'time{time}_rise{rise}_fall{fall}')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)
        self.subjects = subjects
        self.time = time
        self.torques = torques
        self.rise = rise
        self.fall = fall
        self.column_labels = ['ground_force_r_vx', 
                              'ground_force_r_vy',
                              'ground_force_r_vz', 
                              'ground_force_r_px',
                              'ground_force_r_py',
                              'ground_force_r_pz',
                              'ground_torque_r_x',
                              'ground_torque_r_y',
                              'ground_torque_r_z']

        model_fpaths = list()
        perturbed_fpaths = list()
        torque_fpaths = list()
        for subject in subjects:
            for torque in self.torques:
                label = (f'torque{torque}_time{self.time}'
                        f'_rise{self.rise}_fall{self.fall}')
                model_fpaths.append(os.path.join(
                    self.study.config['results_path'], 
                    'perturbed',
                    f'perturbed_{label}', subject, 
                    f'model_perturbed_{label}.osim'))
                perturbed_fpaths.append(os.path.join(
                    self.study.config['results_path'], 
                    'perturbed',
                    f'perturbed_{label}', subject, 
                    f'perturbed_{label}.sto'))
                torque_fpaths.append(
                    os.path.join(self.study.config['results_path'],
                    'perturbed', 
                    f'perturbed_{label}', subject, 
                    'ankle_perturbation_curve_right.sto'))

        self.add_action(model_fpaths + perturbed_fpaths + torque_fpaths, 
                        [], 
                        self.create_perturbed_visualization)

    def create_perturbed_visualization(self, file_dep, target):
        numSubjects = len(self.subjects)
        numTorques = len(self.torques)
        numCond = numSubjects*numTorques

        i = 0
        for isubj, subject in enumerate(self.subjects):
            for itorque, torque in enumerate(self.torques):
                model = osim.Model(file_dep[i])
                perturbed = osim.TimeSeriesTable(
                    file_dep[i + numCond])
                torqueTable = osim.TimeSeriesTable(
                    file_dep[i + 2*numCond])
                model.initSystem()
                
                time = perturbed.getIndependentColumn()
                statesTraj = osim.StatesTrajectory().createFromStatesTable(
                    model, perturbed, True, True)

                extloads = osim.TimeSeriesTable()
                torqueVec = torqueTable.getDependentColumn(
                    '/forceset/perturbation_ankle_angle_r').to_numpy()
                scale = -2000
                for istate in np.arange(statesTraj.getSize()-2):
                    row = osim.RowVector(9, 0.0)
                    state = statesTraj.get(int(istate))
                    model.realizePosition(state)
                    joint = model.getJointSet().get('ankle_r')
                    frame = joint.getParentFrame()
                    position = frame.getPositionInGround(state)
                    rotation = frame.getTransformInGround(state).R().toMat33()

                    torqueIdx = torqueTable.getNearestRowIndexForTime(
                        time[int(istate)])
                    row[0] = scale * torqueVec[int(torqueIdx)] * rotation.get(0,2)
                    row[1] = scale * torqueVec[int(torqueIdx)] * rotation.get(1,2)
                    row[2] = scale * torqueVec[int(torqueIdx)] * rotation.get(2,2)
                    row[3] = position[0]
                    row[4] = position[1]
                    row[5] = position[2]

                    extloads.appendRow(time[int(istate)], row)
                    extloads.setColumnLabels(self.column_labels)

                    osim.STOFileAdapter().write(extloads, 
                        os.path.join(self.analysis_path, 
                            (f'{subject}_torque{torque}_time{self.time}'
                             f'_rise{self.rise}_fall{self.fall}.sto')))
                
                i = i + 1


# Methods figure
# --------------

class TaskPlotMethodsFigure(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, times):
        super(TaskPlotMethodsFigure, self).__init__(study)
        self.name = 'plot_methods_figure'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'methods_figure')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)

        self.subjects = subjects
        self.times = times
        self.time = 35
        self.rise = study.rise
        self.fall = study.fall
        self.labels = ['unperturbed', 'perturbed']
        self.colors = ['dimgrey', [c / 255.0 for c in [200,0,0]]]
        deps = list()
        for isubj, subject in enumerate(subjects):

            # Unperturbed COM
            deps.append(
                os.path.join(
                    self.study.config['results_path'], 
                    'unperturbed', subject,
                    'center_of_mass_unperturbed.sto'))

            # Perturbed COM
            label = (f'perturbed_torque10_time{self.time}'
                    f'_rise{self.rise}_fall{self.fall}_subtalar-10')
            deps.append(
                os.path.join(
                    self.study.config['results_path'], 
                    'perturbed', label, subject,
                    f'center_of_mass_{label}.sto')
                )

        # Perturbation curve
        deps.append(
                os.path.join(
                    self.study.config['results_path'], 
                    'perturbed', label, subjects[0],
                    'ankle_perturbation_curve.sto'))

        self.add_action(deps, 
                        [os.path.join(self.analysis_path, 
                            'methods_figure.png')], 
                        self.plot_methods_figure)

        self.add_action([],
                        [os.path.join(self.analysis_path, 
                            'perturbation_time_axis.png')],
                        self.plot_perturbation_times_axis)

    def plot_methods_figure(self, file_dep, target):

        # Initialize figures
        # ------------------
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(8, 1)
        fig0 = plt.figure(figsize=(3, 5))
        ax_tau = fig0.add_subplot(gs[0:2, :])
        ax_acc = fig0.add_subplot(gs[2:5, :])
        ax_vel = fig0.add_subplot(gs[5:8, :])

        # Arrowy helper functions
        # -----------------------
        def set_arrow_patch_vertical(ax, x, y1, y2):
            arrowstyle = patches.ArrowStyle.CurveAB(head_length=0.4, 
                head_width=0.15)
            lw = 1.0

            arrow = patches.FancyArrowPatch((x, y1), (x, y2),
                    arrowstyle=arrowstyle, mutation_scale=7.5, shrinkA=1.5, shrinkB=1.5,
                    joinstyle='miter', color='black', clip_on=False, zorder=2.5, lw=lw)
            ax.add_patch(arrow)

        def set_arrow_patch_horizontal(ax, y, x1, x2):
            arrowstyle = patches.ArrowStyle.CurveAB(head_length=0.4, 
                head_width=0.15)
            lw = 0.5

            arrow = patches.FancyArrowPatch((x1, y), (x2, y),
                    arrowstyle=arrowstyle, mutation_scale=7.5, shrinkA=1.5, shrinkB=1.5,
                    joinstyle='miter', color='black', clip_on=False, zorder=2.5, lw=lw)
            ax.add_patch(arrow)

        # Plot formatting
        # ---------------
        xfs = 6
        yfs = 7
        for ax in [ax_tau, ax_acc, ax_vel]:
            ax.axvline(x=self.time-self.rise, color='gray', linestyle='--',
                linewidth=0.4, alpha=0.5, zorder=0, clip_on=False)
            ax.axvline(x=self.time, color='gray', linestyle='--',
                linewidth=0.4, alpha=0.5, zorder=0, clip_on=False)
            ax.axvline(x=self.time+self.fall, color='gray', linestyle='--',
                linewidth=0.4, alpha=0.5, zorder=0, clip_on=False)

            util.publication_spines(ax)
            xlim = [self.time-self.rise, self.time+self.fall]
            ax.set_xlim(xlim)
            ax.set_xticks(get_ticks_from_lims(xlim, 5))
            ax.spines['left'].set_position(('outward', 10))

        ax_vel.spines['bottom'].set_position(('outward', 10))
        ax_vel.set_xticklabels(['torque\nonset', '', 'peak\ntorque', 'torque\nend'],
            fontsize=xfs)
        xticks = ax_vel.xaxis.get_major_ticks()
        xticks[1].set_visible(False)

        for ax in [ax_tau, ax_acc]:
            ax.set_xticklabels([])
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both', bottom=False, 
                           top=False, labelbottom=False)

        for ax in [ax_acc, ax_vel]:
            ax.set_yticklabels([])
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', which='both', left=False, 
                           right=False, labelbottom=False)


        # Plot perturbation torque
        # ------------------------
        N = 1001
        perturbation = osim.TimeSeriesTable(file_dep[-1])
        perTimeVec = np.array(perturbation.getIndependentColumn())
        unperturbed = osim.TimeSeriesTable(file_dep[0])
        unpTimeVec = np.array(unperturbed.getIndependentColumn())

        duration = unpTimeVec[-1] - unpTimeVec[0]
        time_at_rise = unpTimeVec[0] + (duration * ((self.time - self.rise) / 100.0))
        time_at_fall = unpTimeVec[0] + (duration * ((self.time + self.fall) / 100.0))
        irise = np.argmin(np.abs(perTimeVec - time_at_rise))
        ifall = np.argmin(np.abs(perTimeVec - time_at_fall))

        tau = -perturbation.getDependentColumn(
            '/forceset/perturbation_ankle_angle_r').to_numpy()[irise:ifall]
        timeInterp = np.linspace(perTimeVec[irise], perTimeVec[ifall-1], N)
        tauInterp = np.interp(timeInterp, perTimeVec[irise:ifall], tau)
        pgc = np.linspace(self.time - self.rise, self.time + self.fall, N)
        ax_tau.plot(pgc, tauInterp, color=self.colors[1], linewidth=2, 
            clip_on=False, solid_capstyle='round')
        ax_tau.set_ylabel(r'exoskeleton torque $[\frac{N\cdot m}{kg}]$', fontsize=yfs)
        ax_tau.set_ylim([0, 0.1])
        ax_tau.set_yticks([0, 0.1])
        ax_tau.tick_params(axis='both', which='major', labelsize=6)

        # Aggregate data
        # --------------
        numLabels = len(self.labels)
        import collections
        com_dict = collections.defaultdict(dict)
        for label in self.labels:
            com_dict[label]['vel'] = np.zeros((N, len(self.subjects)))
            com_dict[label]['acc'] = np.zeros((N, len(self.subjects)))

        for isubj, subject in enumerate(self.subjects):
            for ilabel, label in enumerate(self.labels):
                table = osim.TimeSeriesTable(file_dep[ilabel + isubj*numLabels])
                timeVec = np.array(table.getIndependentColumn())
                
                if 'unperturbed' in label:
                    duration = timeVec[-1] - timeVec[0]
                    time_at_rise = timeVec[0] + (duration * ((self.time - self.rise) / 100.0))
                    time_at_fall = timeVec[0] + (duration * ((self.time + self.fall) / 100.0))
                    irise = np.argmin(np.abs(timeVec - time_at_rise))
                    ifall = np.argmin(np.abs(timeVec - time_at_fall))
                else:
                    irise = np.argmin(np.abs(timeVec - time_at_rise))
                    ifall = len(timeVec)

                velx = table.getDependentColumn('/|com_velocity_x').to_numpy()[irise:ifall]
                accx = table.getDependentColumn('/|com_acceleration_x').to_numpy()[irise:ifall]

                timeInterp = np.linspace(timeVec[irise], timeVec[ifall-1], N)
                
                com_dict[label]['vel'][:, isubj] = np.interp(timeInterp, timeVec[irise:ifall], velx)
                com_dict[label]['acc'][:, isubj] = np.interp(timeInterp, timeVec[irise:ifall], accx)

        # Plotting
        # --------
        vel_mean = np.zeros((N, len(self.labels)))
        acc_mean = np.zeros((N, len(self.labels)))
        for ilabel, label in enumerate(self.labels):
            vel_mean[:, ilabel] = np.mean(com_dict[label]['vel'], axis=1)
            acc_mean[:, ilabel] = np.mean(com_dict[label]['acc'], axis=1)

        vel_step = 0.01
        acc_step = 0.1
        vel_lim = [np.mean(vel_mean), np.mean(vel_mean)]
        acc_lim = [np.mean(acc_mean), np.mean(acc_mean)]
        for ilabel, label in enumerate(self.labels): 
            update_lims(vel_mean[:, ilabel], vel_step, vel_lim)
            update_lims(acc_mean[:, ilabel], acc_step, acc_lim)

        pgc = np.linspace(self.time - self.rise, self.time + self.fall, N)
        handles = list()
        for ilabel, (label, color) in enumerate(zip(self.labels, self.colors)):
            lw = 3 if 'unperturbed' in label else 2

            h, = ax_vel.plot(pgc, vel_mean[:, ilabel], label=label, color=color, 
                linewidth=lw, clip_on=False, solid_capstyle='round')
            handles.append(h)
            ax_vel.set_ylabel('center of mass\nvelocity, position', fontsize=yfs)
            ax_vel.set_ylim(vel_lim)
            ax_vel.set_yticks(get_ticks_from_lims(vel_lim, vel_step))
                
            ax_acc.plot(pgc, acc_mean[:, ilabel], label=label, color=color, 
                linewidth=lw, clip_on=False, solid_capstyle='round')
            ax_acc.set_ylabel('center of mass acceleration, \ncenter of pressure', fontsize=yfs)
            ax_acc.set_ylim(acc_lim)
            ax_acc.set_yticks(get_ticks_from_lims(acc_lim, acc_step))


        # Plot decorations
        ax_acc.text(self.time-6.5, -0.275, 'normal', fontsize=6,
            color=self.colors[0], fontweight='bold')
        ax_acc.text(self.time-4.0, -0.495, 'exoskeleton', fontsize=6,
            color=self.colors[1], fontweight='bold')

        index = np.argmin(np.abs(pgc-self.time))
        set_arrow_patch_vertical(ax_acc, self.time, 
            acc_mean[index, 0], acc_mean[index, 1])

        index = np.argmin(np.abs(pgc-(self.time+self.fall)))
        set_arrow_patch_vertical(ax_vel, self.time+self.fall, 
            vel_mean[index, 0], vel_mean[index, 1])

        # ax_acc.text(self.time-2.75, -0.35, r'$\Delta a_{COM}$', fontsize=8,
        #     color='black', fontweight='bold')

        vloc = (vel_mean[index, 0] + vel_mean[index, 1]) / 2.0
        # ax_vel.text(self.time+self.fall-2.75, vloc, r'$\Delta v_{COM}$', fontsize=8,
        #     color='black', fontweight='bold')

        set_arrow_patch_horizontal(ax_vel, vloc-0.022, 
            self.time-self.rise, self.time)
        set_arrow_patch_horizontal(ax_vel, vloc-0.022, 
            self.time, self.time+self.fall)

        ax_vel.text(self.time-5, vloc-0.02, '10% gait cycle', fontsize=5,
            color='black', ha='center')
        ax_vel.text(self.time+2.5, vloc-0.02, '5% gait cycle', fontsize=5,
            color='black', ha='center')

        fig0.tight_layout()
        fig0.savefig(target[0], dpi=600)
        plt.close()

    def plot_perturbation_times_axis(self, file_dep, target):

        fig = plt.figure(figsize=(5, 1))
        ax = fig.add_subplot(111)
        ax.set_xticks(np.arange(len(self.times)))
        ax.set_xlim(0, len(self.times)-1)
        ax.set_xticklabels([f'{time}' for time in self.times],
            fontsize=6)
        util.publication_spines(ax)

        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_visible(False)
        ax.set_yticklabels([])
        ax.yaxis.set_ticks_position('none')
        ax.tick_params(axis='y', which='both', bottom=False, 
                       top=False, labelbottom=False)
        ax.set_xlabel('exoskeleton torque peak time\n(% gait cycle)')

        fig.tight_layout()
        fig.savefig(target[0], dpi=600)


# Statistics
# ----------

class TaskCreateCenterOfMassStatisticsTables(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, times, rise, fall):
        super(TaskCreateCenterOfMassStatisticsTables, self).__init__(study)
        self.name = f'create_center_of_mass_statistics_tables_{rise}_fall{fall}'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.subjects = subjects
        self.times = times
        self.rise = rise
        self.fall = fall
        self.gravity = 9.81
        self.torques = study.plot_torques
        self.subtalars = study.plot_subtalars

        deps = list()
        self.multiindex_tuples = list()
        self.multiindex_tuples_muscles = list()
        self.multiindex_tuples_torques = list()
        self.label_dict = dict()
        ilabel = 0
        for isubj, subject in enumerate(subjects):

            # Unperturbed solutions
            # ---------------------
            deps.append(
                os.path.join(
                    self.study.config['results_path'], 
                    'unperturbed', subject,
                    'center_of_mass_unperturbed.sto'))

            self.label_dict[f'{subject}_unperturbed'] = ilabel
            ilabel += 1

            for actu in [False, True]:
                torque_act = '_torque_actuators' if actu else ''
                subpath = 'torque_actuators' if actu else 'perturbed'
                actu_name = 'torques' if actu else 'muscles'

                for itime, time in enumerate(self.times):

                    # Unperturbed time-stepping solutions
                    # -----------------------------------
                    label = (f'perturbed_torque0_time{time}'
                             f'_rise{self.rise}_fall{self.fall}{torque_act}')
                    deps.append(
                        os.path.join(
                            self.study.config['results_path'], 
                            subpath, label, subject,
                            f'center_of_mass_{label}.sto'))

                    self.label_dict[f'{subject}_unperturbed_time{time}{torque_act}'] = ilabel
                    ilabel += 1

                    if actu and not itime:
                        self.multiindex_tuples.append((subject, 'unperturbed'))
                        self.multiindex_tuples_muscles.append((subject, 'unperturbed', 'muscles'))
                        self.multiindex_tuples_torques.append((subject, 'unperturbed', 'torques'))

                    for torque, subtalar in zip(self.torques, self.subtalars):

                         # Perturbed solutions
                         # -------------------
                        label = (f'perturbed_torque{torque}_time{time}'
                                f'_rise{self.rise}_fall{self.fall}{subtalar}{torque_act}')
                        deps.append(
                            os.path.join(
                                self.study.config['results_path'], 
                                subpath, label, subject,
                                f'center_of_mass_{label}.sto')
                            )

                        self.label_dict[f'{subject}_{label}'] = ilabel
                        ilabel += 1

                        if actu and not itime:
                            self.multiindex_tuples.append((
                                subject,
                                f'perturbed_torque{torque}{subtalar}'
                                ))
                            self.multiindex_tuples_muscles.append((
                                subject,
                                f'perturbed_torque{torque}{subtalar}',
                                'muscles'
                                ))
                            self.multiindex_tuples_torques.append((
                                subject,
                                f'perturbed_torque{torque}{subtalar}',
                                'torques'
                                ))

        targets = list()
        for actu in ['muscles', 'torques']:
            for time in self.times:
                for kin in ['pos', 'vel', 'acc']:
                    for direc in ['x', 'y', 'z']:
                            targets += [os.path.join(study.config['statistics_path'], 
                                        'center_of_mass', 'tables',
                                        f'com_stats_time{time}_{kin}_{direc}_{actu}.csv')]

        for time in self.times:
            for kin in ['pos', 'vel', 'acc']:
                for direc in ['x', 'y', 'z']:
                        targets += [os.path.join(study.config['statistics_path'], 
                                    'center_of_mass', 'tables',
                                    f'com_stats_time{time}_{kin}_{direc}_diff.csv')]

        self.add_action(deps, targets, self.create_com_stats_table)

    def create_com_stats_table(self, file_dep, target):

        # Aggregate data
        # --------------
        import collections
        com_dict = collections.defaultdict(dict)
        time_dict = collections.defaultdict(dict)
        com_height_dict = dict()
        duration_dict = dict()

        for isubj, subject in enumerate(self.subjects):

            # Unperturbed center-of-mass trajectory
            # -------------------------------------
            unperturb_index = self.label_dict[f'{subject}_unperturbed']
            tableTemp = osim.TimeSeriesTable(file_dep[unperturb_index])
            com_height_dict[subject] = np.mean(tableTemp.getDependentColumn(
                                               '/|com_position_y').to_numpy())
            timeTemp = np.array(tableTemp.getIndependentColumn())
            duration_dict[subject] = timeTemp[-1] - timeTemp[0]
            for actu in [False, True]:
                torque_act = '_torque_actuators' if actu else ''

                for time in self.times:

                    # Unperturbed center-of-mass trajectory
                    # -------------------------------------
                    label = f'{subject}_unperturbed_time{time}{torque_act}'
                    unperturb_index = self.label_dict[label]
                    unpTable = osim.TimeSeriesTable(file_dep[unperturb_index])
                    unpTimeVec = unpTable.getIndependentColumn()
                    unpTable_np = np.zeros((len(unpTimeVec), 9))
                    unp_pos_x = unpTable.getDependentColumn(
                        '/|com_position_x').to_numpy()
                    unp_pos_y = unpTable.getDependentColumn(
                        '/|com_position_y').to_numpy()
                    unp_pos_z = unpTable.getDependentColumn(
                        '/|com_position_z').to_numpy()
                    unpTable_np[:, 0] = unp_pos_x - unp_pos_x[0] 
                    unpTable_np[:, 1] = unp_pos_y - unp_pos_y[0] 
                    unpTable_np[:, 2] = unp_pos_z - unp_pos_z[0] 
                    unpTable_np[:, 3] = unpTable.getDependentColumn(
                        '/|com_velocity_x').to_numpy() 
                    unpTable_np[:, 4] = unpTable.getDependentColumn(
                        '/|com_velocity_y').to_numpy() 
                    unpTable_np[:, 5] = unpTable.getDependentColumn(
                        '/|com_velocity_z').to_numpy() 
                    unpTable_np[:, 6] = unpTable.getDependentColumn(
                        '/|com_acceleration_x').to_numpy()
                    unpTable_np[:, 7] = unpTable.getDependentColumn(
                        '/|com_acceleration_y').to_numpy()
                    unpTable_np[:, 8] = unpTable.getDependentColumn(
                        '/|com_acceleration_z').to_numpy()

                    com_dict[subject][label] = unpTable_np
                    time_dict[subject][label] = unpTimeVec

                    for torque, subtalar in zip(self.torques, self.subtalars):

                        # Perturbed center-of-mass trajectory
                        # -----------------------------------
                        label = (f'{subject}_perturbed_torque{torque}_time{time}'
                                 f'_rise{self.rise}_fall{self.fall}{subtalar}{torque_act}')
                        perturb_index = self.label_dict[label]
                        table = osim.TimeSeriesTable(file_dep[perturb_index])
                        timeVec = table.getIndependentColumn()
                        N = len(timeVec)
                        table_np = np.zeros((N, 9))
                        pos_x = table.getDependentColumn(
                            '/|com_position_x').to_numpy()
                        pos_y = table.getDependentColumn(
                            '/|com_position_y').to_numpy()
                        pos_z = table.getDependentColumn(
                            '/|com_position_z').to_numpy()
                        table_np[:, 0] = pos_x - pos_x[0] 
                        table_np[:, 1] = pos_y - pos_y[0] 
                        table_np[:, 2] = pos_z - pos_z[0] 
                        table_np[:, 3] = table.getDependentColumn(
                            '/|com_velocity_x').to_numpy()
                        table_np[:, 4] = table.getDependentColumn(
                            '/|com_velocity_y').to_numpy()
                        table_np[:, 5] = table.getDependentColumn(
                            '/|com_velocity_z').to_numpy() 
                        table_np[:, 6] = table.getDependentColumn(
                            '/|com_acceleration_x').to_numpy()
                        table_np[:, 7] = table.getDependentColumn(
                            '/|com_acceleration_y').to_numpy()
                        table_np[:, 8] = table.getDependentColumn(
                            '/|com_acceleration_z').to_numpy()

                        # Compute difference between perturbed and unperturbed
                        # trajectories for this subject. We don't need to interpolate
                        # here since the perturbed and unperturbed trajectories contain
                        # the same time points (up until the end of the perturbation).
                        com_dict[subject][label] = table_np
                        time_dict[subject][label] = timeVec

        # Create tables
        # -------------
        com_value_dict = collections.defaultdict(dict)
        for iactu, actu in enumerate([False, True]):
            torque_act = '_torque_actuators' if actu else ''
            actu_name = 'torques' if actu else 'muscles'

            for itime, time in enumerate(self.times):
                com_values = np.zeros((9, len(self.torques)+1, len(self.subjects)))

                for isubj, subject in enumerate(self.subjects):

                    # Compute the closet time index to the current peak 
                    # perturbation time. 
                    duration = duration_dict[subject]
                    l_max = com_height_dict[subject]
                    v_max = np.sqrt(self.gravity * l_max)

                    # Unperturbed
                    label = f'{subject}_unperturbed_time{time}{torque_act}'
                    timeVec = time_dict[subject][label]
                    time_at_peak = timeVec[0] + (duration * (time / 100.0))
                    index_peak = np.argmin(np.abs(timeVec - time_at_peak))
                    index_fall = -1
                    com = com_dict[subject][label]
                    com_values[0, 0, isubj] = com[index_fall, 0] / l_max
                    com_values[1, 0, isubj] = com[index_fall, 1] / l_max
                    com_values[2, 0, isubj] = com[index_fall, 2] / l_max
                    com_values[3, 0, isubj] = com[index_fall, 3] / v_max
                    com_values[4, 0, isubj] = com[index_fall, 4] / v_max
                    com_values[5, 0, isubj] = com[index_fall, 5] / v_max
                    com_values[6, 0, isubj] = com[index_peak, 6] / self.gravity
                    com_values[7, 0, isubj] = com[index_peak, 7] / self.gravity
                    com_values[8, 0, isubj] = com[index_peak, 8] / self.gravity

                    # Perturbed
                    for iperturb, (torque, subtalar) in enumerate(zip(self.torques, self.subtalars)):
                        label = (f'{subject}_perturbed_torque{torque}_time{time}'
                                 f'_rise{self.rise}_fall{self.fall}{subtalar}{torque_act}')
                        timeVec = time_dict[subject][label]
                        time_at_peak = timeVec[0] + (duration * (time / 100.0))
                        index_peak = np.argmin(np.abs(timeVec - time_at_peak))
                        index_fall = -1
                        com = com_dict[subject][label]
                        com_values[0, iperturb+1, isubj] = com[index_fall, 0] / l_max
                        com_values[1, iperturb+1, isubj] = com[index_fall, 1] / l_max
                        com_values[2, iperturb+1, isubj] = com[index_fall, 2] / l_max
                        com_values[3, iperturb+1, isubj] = com[index_fall, 3] / v_max
                        com_values[4, iperturb+1, isubj] = com[index_fall, 4] / v_max
                        com_values[5, iperturb+1, isubj] = com[index_fall, 5] / v_max
                        com_values[6, iperturb+1, isubj] = com[index_peak, 6] / self.gravity
                        com_values[7, iperturb+1, isubj] = com[index_peak, 7] / self.gravity
                        com_values[8, iperturb+1, isubj] = com[index_peak, 8] / self.gravity

                com_value_dict[actu_name][time] = com_values

                pos_x = com_values[0, :, :].T.reshape(-1, 1)
                pos_y = com_values[1, :, :].T.reshape(-1, 1)
                pos_z = com_values[2, :, :].T.reshape(-1, 1)
                vel_x = com_values[3, :, :].T.reshape(-1, 1)
                vel_y = com_values[4, :, :].T.reshape(-1, 1)
                vel_z = com_values[5, :, :].T.reshape(-1, 1)
                acc_x = com_values[6, :, :].T.reshape(-1, 1)
                acc_y = com_values[7, :, :].T.reshape(-1, 1)
                acc_z = com_values[8, :, :].T.reshape(-1, 1)

                index = pd.MultiIndex.from_tuples(self.multiindex_tuples,
                    names=['subject', 'perturbation'])

                df_list = list()
                df_list.append(pd.DataFrame(pos_x, index=index, columns=['com']))
                df_list.append(pd.DataFrame(pos_y, index=index, columns=['com']))
                df_list.append(pd.DataFrame(pos_z, index=index, columns=['com']))
                df_list.append(pd.DataFrame(vel_x, index=index, columns=['com']))
                df_list.append(pd.DataFrame(vel_y, index=index, columns=['com']))
                df_list.append(pd.DataFrame(vel_z, index=index, columns=['com']))
                df_list.append(pd.DataFrame(acc_x, index=index, columns=['com']))
                df_list.append(pd.DataFrame(acc_y, index=index, columns=['com']))
                df_list.append(pd.DataFrame(acc_z, index=index, columns=['com']))

                offset = len(df_list)*(itime + iactu*len(self.times))
                for index in np.arange(len(df_list)):
                    target_dir = os.path.dirname(target[offset + index])
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                    with open(target[offset + index], 'w') as f:
                        df_list[index].to_csv(f, line_terminator='\n')

        com_diff_dict = collections.defaultdict(dict)
        for itime, time in enumerate(self.times):
            
            # Muscles
            com_muscles = com_value_dict['muscles'][time]
            
            index = pd.MultiIndex.from_tuples(self.multiindex_tuples_muscles,
                    names=['subject', 'perturbation', 'actuator'])
            df_muscles = list()
            df_muscles.append(pd.DataFrame(com_muscles[0, :, :].T.reshape(-1, 1), index=index, columns=['com']))
            df_muscles.append(pd.DataFrame(com_muscles[1, :, :].T.reshape(-1, 1), index=index, columns=['com']))
            df_muscles.append(pd.DataFrame(com_muscles[2, :, :].T.reshape(-1, 1), index=index, columns=['com']))
            df_muscles.append(pd.DataFrame(com_muscles[3, :, :].T.reshape(-1, 1), index=index, columns=['com']))
            df_muscles.append(pd.DataFrame(com_muscles[4, :, :].T.reshape(-1, 1), index=index, columns=['com']))
            df_muscles.append(pd.DataFrame(com_muscles[5, :, :].T.reshape(-1, 1), index=index, columns=['com']))
            df_muscles.append(pd.DataFrame(com_muscles[6, :, :].T.reshape(-1, 1), index=index, columns=['com']))
            df_muscles.append(pd.DataFrame(com_muscles[7, :, :].T.reshape(-1, 1), index=index, columns=['com']))
            df_muscles.append(pd.DataFrame(com_muscles[8, :, :].T.reshape(-1, 1), index=index, columns=['com']))

            for df in df_muscles:
                for subject in self.subjects:
                    df.loc[subject]['com'] -= df.loc[subject, 'unperturbed']['com']
                    df.drop(labels=(subject, 'unperturbed'), inplace=True)

            # Torques
            com_torques = com_value_dict['torques'][time]

            index = pd.MultiIndex.from_tuples(self.multiindex_tuples_torques,
                    names=['subject', 'perturbation', 'actuator'])
            df_torques = list()
            df_torques.append(pd.DataFrame(com_torques[0, :, :].T.reshape(-1, 1), index=index, columns=['com']))
            df_torques.append(pd.DataFrame(com_torques[1, :, :].T.reshape(-1, 1), index=index, columns=['com']))
            df_torques.append(pd.DataFrame(com_torques[2, :, :].T.reshape(-1, 1), index=index, columns=['com']))
            df_torques.append(pd.DataFrame(com_torques[3, :, :].T.reshape(-1, 1), index=index, columns=['com']))
            df_torques.append(pd.DataFrame(com_torques[4, :, :].T.reshape(-1, 1), index=index, columns=['com']))
            df_torques.append(pd.DataFrame(com_torques[5, :, :].T.reshape(-1, 1), index=index, columns=['com']))
            df_torques.append(pd.DataFrame(com_torques[6, :, :].T.reshape(-1, 1), index=index, columns=['com']))
            df_torques.append(pd.DataFrame(com_torques[7, :, :].T.reshape(-1, 1), index=index, columns=['com']))
            df_torques.append(pd.DataFrame(com_torques[8, :, :].T.reshape(-1, 1), index=index, columns=['com']))

            for df in df_torques:
                for subject in self.subjects:
                    df.loc[subject]['com'] -= df.loc[subject, 'unperturbed']['com']
                    df.drop(labels=(subject, 'unperturbed'), inplace=True)

            df_list = list()
            for df_m, df_t in zip(df_muscles, df_torques):
                df_list.append(pd.concat([df_m, df_t]))

            offset = len(df_list)*itime + 18*len(self.times)
            for index in np.arange(len(df_list)):
                target_dir = os.path.dirname(target[offset + index])
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                with open(target[offset + index], 'w') as f:
                    df_list[index].to_csv(f, line_terminator='\n')


class TaskAggregateCenterOfMassStatistics(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, times, rise, fall):
        super(TaskAggregateCenterOfMassStatistics, self).__init__(study)
        self.name = f'aggregate_com_statistics_results_{rise}_fall{fall}'
        self.results_path = os.path.join(study.config['statistics_path'], 
            'center_of_mass', 'results')
        self.aggregate_path = os.path.join(study.config['statistics_path'],
            'center_of_mass', 'aggregate')
        if not os.path.exists(self.aggregate_path): 
            os.makedirs(self.aggregate_path)
        self.times = times
        self.rise = rise
        self.fall = fall
        self.torques = study.plot_torques
        self.subtalars = study.plot_subtalars

        self.multiindex_tuples = list()
        for itime, time in enumerate(self.times):
            for torque, subtalar in zip(self.torques, self.subtalars):
                self.multiindex_tuples.append((
                    time,
                    f'perturbed_torque{torque}{subtalar}'
                    ))


        deps = list()
        for actu in ['muscles', 'torques', 'diff']:
            for kin in ['pos', 'vel', 'acc']:
                for direc in ['x', 'y', 'z']:
                    for time in self.times:
                            deps += [os.path.join(self.results_path,
                                     f'com_stats_time{time}_{kin}_{direc}_{actu}_comparisons.csv')]

        targets = list()
        for actu in ['muscles', 'torques', 'diff']:
            for kin in ['pos', 'vel', 'acc']:
                for direc in ['x', 'y', 'z']:
                        targets += [os.path.join(self.aggregate_path,
                                    f'com_stats_{kin}_{direc}_{actu}.csv')]


        self.add_action(deps, 
                        targets, 
                        self.aggregate_com_stats)

    def aggregate_com_stats(self, file_dep, target):

        idep = 0
        itarget = 0

        # Did all of the perturbations change the center-of-mass kinematics?
        for actu in ['muscles', 'torques']:
            for kin in ['pos', 'vel', 'acc']:
                for direc in ['x', 'y', 'z']:
                    significances = list()
                    for itime, time in enumerate(self.times):
                        df = pd.read_csv(file_dep[idep])
                        idep += 1
                            
                        contrasts = df['contrast']
                        p_values = df['adj.p.value']
                        for torque, subtalar in zip(self.torques, self.subtalars):
                            label = f'unperturbed - perturbed_torque{torque}{subtalar}'

                            iperturb = 0
                            foundContrast = False
                            for icontrast, contrast in enumerate(contrasts):
                                if label == contrast:
                                    iperturb = icontrast
                                    foundContrast = True
                                    break

                            if not foundContrast:
                                raise Exception(f'Did not find statistics contrast {label}')

                            p_value = p_values[iperturb]
                            significant = p_value < 0.05
                            significances.append(significant)

                    index = pd.MultiIndex.from_tuples(self.multiindex_tuples,
                        names=['time', 'perturbation'])
                    df_sig = pd.DataFrame(significances, index=index, columns=['significant'])

                    target_dir = os.path.dirname(target[itarget])
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                    with open(target[itarget], 'w') as f:
                        df_sig.to_csv(f, line_terminator='\n')

                    itarget += 1

        # Were the torque-driven perturbations different from the muscle-driven perturbations?
        for kin in ['pos', 'vel', 'acc']:
            for direc in ['x', 'y', 'z']:
                significances = list()
                for itime, time in enumerate(self.times):
                    df = pd.read_csv(file_dep[idep])
                    idep += 1
                        
                    contrasts = df['contrast']
                    p_values = df['adj.p.value']
                    for torque, subtalar in zip(self.torques, self.subtalars):
                        label = (f'perturbed_torque{torque}{subtalar} muscles - '
                                 f'perturbed_torque{torque}{subtalar} torques')

                        iperturb = 0
                        foundContrast = False
                        for icontrast, contrast in enumerate(contrasts):
                            if label == contrast:
                                iperturb = icontrast
                                foundContrast = True
                                break

                        if not foundContrast:
                            label = (f'perturbed_torque{torque}{subtalar} torques - '
                                     f'perturbed_torque{torque}{subtalar} muscles')

                            for icontrast, contrast in enumerate(contrasts):
                                if label == contrast:
                                    iperturb = icontrast
                                    foundContrast = True
                                    break

                            if not foundContrast:
                                raise Exception(f'Did not find statistics contrast {label}')

                        p_value = p_values[iperturb]
                        significant = p_value < 0.05
                        significances.append(significant)

                index = pd.MultiIndex.from_tuples(self.multiindex_tuples,
                    names=['time', 'perturbation'])
                df_sig = pd.DataFrame(significances, index=index, columns=['significant'])

                target_dir = os.path.dirname(target[itarget])
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                with open(target[itarget], 'w') as f:
                    df_sig.to_csv(f, line_terminator='\n')

                itarget += 1


class TaskCreateCenterOfPressureStatisticsTables(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, times, rise, fall):
        super(TaskCreateCenterOfPressureStatisticsTables, self).__init__(study)
        self.name = f'create_center_of_pressure_statistics_tables_rise{rise}_fall{fall}'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.subjects = subjects
        self.times = times
        self.rise = rise
        self.fall = fall
        self.gravity = 9.81
        self.torques = study.plot_torques
        self.subtalars = study.plot_subtalars

        deps = list()
        self.multiindex_tuples = list()
        self.multiindex_tuples_muscles = list()
        self.multiindex_tuples_torques = list()
        self.label_dict = dict()
        self.models = list()
        ilabel = 0
        for isubj, subject in enumerate(subjects):

            # Model
            # -----
            self.models.append(
                os.path.join(
                    self.study.config['results_path'], 
                    'unperturbed', subject,
                    'model_unperturbed.osim'))

            # Unperturbed solutions
            # ---------------------
            deps.append(
                os.path.join(
                    self.study.config['results_path'], 
                    'unperturbed', subject,
                    'unperturbed_grfs.sto'))

            self.label_dict[f'{subject}_unperturbed'] = ilabel
            ilabel += 1

            for actu in [False, True]:
                torque_act = '_torque_actuators' if actu else ''
                subpath = 'torque_actuators' if actu else 'perturbed'
                actu_name = 'torques' if actu else 'muscles'

                for itime, time in enumerate(self.times):

                    # Unperturbed time-stepping solutions
                    # -----------------------------------
                    label = (f'perturbed_torque0_time{time}'
                             f'_rise{self.rise}_fall{self.fall}{torque_act}')
                    deps.append(
                        os.path.join(
                            self.study.config['results_path'], 
                            subpath, label, subject,
                            f'{label}_grfs.sto'))

                    self.label_dict[f'{subject}_unperturbed_time{time}{torque_act}'] = ilabel
                    ilabel += 1

                    if actu and not itime:
                        self.multiindex_tuples.append((subject, 'unperturbed'))
                        self.multiindex_tuples_muscles.append((subject, 'unperturbed', 'muscles'))
                        self.multiindex_tuples_torques.append((subject, 'unperturbed', 'torques'))

                    for torque, subtalar in zip(self.torques, self.subtalars):

                         # Perturbed solutions
                         # -------------------
                        label = (f'perturbed_torque{torque}_time{time}'
                                f'_rise{self.rise}_fall{self.fall}{subtalar}{torque_act}')
                        deps.append(
                            os.path.join(
                                self.study.config['results_path'], 
                                subpath, label, subject,
                                f'{label}_grfs.sto')
                            )

                        self.label_dict[f'{subject}_{label}'] = ilabel
                        ilabel += 1

                        if actu and not itime:
                            self.multiindex_tuples.append((
                                subject,
                                f'perturbed_torque{torque}{subtalar}'
                                ))
                            self.multiindex_tuples_muscles.append((
                                subject,
                                f'perturbed_torque{torque}{subtalar}',
                                'muscles'
                                ))
                            self.multiindex_tuples_torques.append((
                                subject,
                                f'perturbed_torque{torque}{subtalar}',
                                'torques'
                                ))

        targets = list()
        for actu in ['muscles', 'torques']:
            for time in self.times:
                for kin in ['pos']:
                    for direc in ['x', 'z']:
                            targets += [os.path.join(study.config['statistics_path'], 
                                        'center_of_pressure', 'tables',
                                        f'cop_stats_time{time}_{kin}_{direc}_{actu}.csv')]

        for time in self.times:
            for kin in ['pos']:
                for direc in ['x', 'z']:
                        targets += [os.path.join(study.config['statistics_path'], 
                                    'center_of_pressure', 'tables',
                                    f'cop_stats_time{time}_{kin}_{direc}_diff.csv')]

        self.add_action(deps, targets, self.create_cop_stats_table)

    def create_cop_stats_table(self, file_dep, target):

        # Aggregate data
        # --------------
        import collections
        cop_dict = collections.defaultdict(dict)
        time_dict = collections.defaultdict(dict)
        duration_dict = dict()
        for isubj, subject in enumerate(self.subjects):

            # Model
            # -----
            model = osim.Model(self.models[isubj])
            model.initSystem()

            # Unperturbed center-of-pressure trajectory
            # -----------------------------------------
            unperturb_index = self.label_dict[f'{subject}_unperturbed']
            tableTemp = osim.TimeSeriesTable(file_dep[unperturb_index])
            timeTemp = np.array(tableTemp.getIndependentColumn())
            duration_dict[subject] = timeTemp[-1] - timeTemp[0]
            for actu in [False, True]:
                torque_act = '_torque_actuators' if actu else ''

                for time in self.times:

                    # Unperturbed center-of-mass trajectory
                    # -------------------------------------
                    label = f'{subject}_unperturbed_time{time}{torque_act}'
                    unperturb_index = self.label_dict[label]
                    unpTable = osim.TimeSeriesTable(file_dep[unperturb_index])
                    unpTimeVec = unpTable.getIndependentColumn()
                    unpTable_np = np.zeros((len(unpTimeVec), 3))
                    unpTable_np[:, 0] = unpTable.getDependentColumn(
                        'ground_force_r_px').to_numpy() 
                    unpTable_np[:, 1] = unpTable.getDependentColumn(
                        'ground_force_r_py').to_numpy()  
                    unpTable_np[:, 2] = unpTable.getDependentColumn(
                        'ground_force_r_pz').to_numpy() 

                    cop_dict[subject][label] = unpTable_np
                    time_dict[subject][label] = unpTimeVec

                    for torque, subtalar in zip(self.torques, self.subtalars):

                        # Perturbed center-of-mass trajectory
                        # -----------------------------------
                        label = (f'{subject}_perturbed_torque{torque}_time{time}'
                                 f'_rise{self.rise}_fall{self.fall}{subtalar}{torque_act}')
                        perturb_index = self.label_dict[label]
                        table = osim.TimeSeriesTable(file_dep[perturb_index])
                        timeVec = table.getIndependentColumn()
                        table_np = np.zeros((len(timeVec), 3))
                        table_np[:, 0] = table.getDependentColumn(
                            'ground_force_r_px').to_numpy() 
                        table_np[:, 1] = table.getDependentColumn(
                            'ground_force_r_py').to_numpy()  
                        table_np[:, 2] = table.getDependentColumn(
                            'ground_force_r_pz').to_numpy() 

                        # Compute difference between perturbed and unperturbed
                        # trajectories for this subject. We don't need to interpolate
                        # here since the perturbed and unperturbed trajectories contain
                        # the same time points (up until the end of the perturbation).
                        cop_dict[subject][label] = table_np
                        time_dict[subject][label] = timeVec

        # Create tables
        # -------------
        cop_value_dict = collections.defaultdict(dict)
        for iactu, actu in enumerate([False, True]):
            torque_act = '_torque_actuators' if actu else ''
            actu_name = 'torques' if actu else 'muscles'

            for itime, time in enumerate(self.times):
                cop_values = np.zeros((2, len(self.torques)+1, len(self.subjects)))

                for isubj, subject in enumerate(self.subjects):

                    # Compute the closet time index to the current peak 
                    # perturbation time. 
                    duration = duration_dict[subject]

                    # Unperturbed
                    label = f'{subject}_unperturbed_time{time}{torque_act}'
                    timeVec = time_dict[subject][label]
                    time_at_peak = timeVec[0] + (duration * (time / 100.0))
                    index_peak = np.argmin(np.abs(timeVec - time_at_peak))
                    cop = cop_dict[subject][label]
                    cop_values[0, 0, isubj] = cop[index_peak, 0]
                    cop_values[1, 0, isubj] = cop[index_peak, 2]

                    # Perturbed
                    for iperturb, (torque, subtalar) in enumerate(zip(self.torques, self.subtalars)):
                        label = (f'{subject}_perturbed_torque{torque}_time{time}'
                                 f'_rise{self.rise}_fall{self.fall}{subtalar}{torque_act}')
                        timeVec = time_dict[subject][label]
                        time_at_peak = timeVec[0] + (duration * (time / 100.0))
                        index_peak = np.argmin(np.abs(timeVec - time_at_peak))
                        cop = cop_dict[subject][label]
                        cop_values[0, iperturb+1, isubj] = cop[index_peak, 0]
                        cop_values[1, iperturb+1, isubj] = cop[index_peak, 2]

                cop_value_dict[actu_name][time] = cop_values

                pos_x = cop_values[0, :, :].T.reshape(-1, 1)
                pos_z = cop_values[1, :, :].T.reshape(-1, 1)

                index = pd.MultiIndex.from_tuples(self.multiindex_tuples,
                    names=['subject', 'perturbation'])

                df_list = list()
                df_list.append(pd.DataFrame(pos_x, index=index, columns=['cop']))
                df_list.append(pd.DataFrame(pos_z, index=index, columns=['cop']))

                offset = len(df_list)*(itime + iactu*len(self.times))
                for index in np.arange(len(df_list)):
                    target_dir = os.path.dirname(target[offset + index])
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                    with open(target[offset + index], 'w') as f:
                        df_list[index].to_csv(f, line_terminator='\n')

        cop_diff_dict = collections.defaultdict(dict)
        for itime, time in enumerate(self.times):
            
            # Muscles
            cop_muscles = cop_value_dict['muscles'][time]
            
            index = pd.MultiIndex.from_tuples(self.multiindex_tuples_muscles,
                    names=['subject', 'perturbation', 'actuator'])
            df_muscles = list()
            df_muscles.append(pd.DataFrame(cop_muscles[0, :, :].T.reshape(-1, 1), index=index, columns=['cop']))
            df_muscles.append(pd.DataFrame(cop_muscles[1, :, :].T.reshape(-1, 1), index=index, columns=['cop']))

            for df in df_muscles:
                for subject in self.subjects:
                    df.loc[subject]['cop'] -= df.loc[subject, 'unperturbed']['cop']
                    df.drop(labels=(subject, 'unperturbed'), inplace=True)

            # Torques
            cop_torques = cop_value_dict['torques'][time]

            index = pd.MultiIndex.from_tuples(self.multiindex_tuples_torques,
                    names=['subject', 'perturbation', 'actuator'])
            df_torques = list()
            df_torques.append(pd.DataFrame(cop_torques[0, :, :].T.reshape(-1, 1), index=index, columns=['cop']))
            df_torques.append(pd.DataFrame(cop_torques[1, :, :].T.reshape(-1, 1), index=index, columns=['cop']))

            for df in df_torques:
                for subject in self.subjects:
                    df.loc[subject]['cop'] -= df.loc[subject, 'unperturbed']['cop']
                    df.drop(labels=(subject, 'unperturbed'), inplace=True)

            df_list = list()
            for df_m, df_t in zip(df_muscles, df_torques):
                df_list.append(pd.concat([df_m, df_t]))

            offset = len(df_list)*itime + 4*len(self.times)
            for index in np.arange(len(df_list)):
                target_dir = os.path.dirname(target[offset + index])
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                with open(target[offset + index], 'w') as f:
                    df_list[index].to_csv(f, line_terminator='\n')


class TaskAggregateCenterOfPressureStatistics(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, times, rise, fall):
        super(TaskAggregateCenterOfPressureStatistics, self).__init__(study)
        self.name = f'aggregate_cop_statistics_results_rise{rise}_fall{fall}'
        self.results_path = os.path.join(study.config['statistics_path'], 
            'center_of_pressure', 'results')
        self.aggregate_path = os.path.join(study.config['statistics_path'],
            'center_of_pressure', 'aggregate')
        if not os.path.exists(self.aggregate_path): 
            os.makedirs(self.aggregate_path)
        self.times = times
        self.rise = rise
        self.fall = fall
        self.torques = study.plot_torques
        self.subtalars = study.plot_subtalars

        self.multiindex_tuples = list()
        for itime, time in enumerate(self.times):
            for torque, subtalar in zip(self.torques, self.subtalars):
                self.multiindex_tuples.append((
                    time,
                    f'perturbed_torque{torque}{subtalar}'
                    ))


        deps = list()
        for actu in ['muscles', 'torques', 'diff']:
            for kin in ['pos']:
                for direc in ['x', 'z']:
                    for time in self.times:
                            deps += [os.path.join(self.results_path,
                                     f'cop_stats_time{time}_{kin}_{direc}_{actu}_comparisons.csv')]

        targets = list()
        for actu in ['muscles', 'torques', 'diff']:
            for kin in ['pos']:
                for direc in ['x', 'z']:
                        targets += [os.path.join(self.aggregate_path,
                                    f'cop_stats_{kin}_{direc}_{actu}.csv')]


        self.add_action(deps, 
                        targets, 
                        self.aggregate_cop_stats)

    def aggregate_cop_stats(self, file_dep, target):

        idep = 0
        itarget = 0

        # Did all of the perturbations change the center-of-mass kinematics?
        for actu in ['muscles', 'torques']:
            for kin in ['pos']:
                for direc in ['x', 'z']:
                    significances = list()
                    for itime, time in enumerate(self.times):
                        df = pd.read_csv(file_dep[idep])
                        idep += 1
                            
                        contrasts = df['contrast']
                        p_values = df['adj.p.value']
                        for torque, subtalar in zip(self.torques, self.subtalars):
                            label = f'unperturbed - perturbed_torque{torque}{subtalar}'

                            iperturb = 0
                            foundContrast = False
                            for icontrast, contrast in enumerate(contrasts):
                                if label == contrast:
                                    iperturb = icontrast
                                    foundContrast = True
                                    break

                            if not foundContrast:
                                raise Exception(f'Did not find statistics contrast {label}')

                            p_value = p_values[iperturb]
                            significant = p_value < 0.05
                            significances.append(significant)

                    index = pd.MultiIndex.from_tuples(self.multiindex_tuples,
                        names=['time', 'perturbation'])
                    df_sig = pd.DataFrame(significances, index=index, columns=['significant'])

                    target_dir = os.path.dirname(target[itarget])
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                    with open(target[itarget], 'w') as f:
                        df_sig.to_csv(f, line_terminator='\n')

                    itarget += 1

        # Were the torque-driven perturbations different from the muscle-driven perturbations?
        for kin in ['pos']:
            for direc in ['x', 'z']:
                significances = list()
                for itime, time in enumerate(self.times):
                    df = pd.read_csv(file_dep[idep])
                    idep += 1
                        
                    contrasts = df['contrast']
                    p_values = df['adj.p.value']
                    for torque, subtalar in zip(self.torques, self.subtalars):
                        label = (f'perturbed_torque{torque}{subtalar} muscles - '
                                 f'perturbed_torque{torque}{subtalar} torques')

                        iperturb = 0
                        foundContrast = False
                        for icontrast, contrast in enumerate(contrasts):
                            if label == contrast:
                                iperturb = icontrast
                                foundContrast = True
                                break

                        if not foundContrast:
                            label = (f'perturbed_torque{torque}{subtalar} torques - '
                                     f'perturbed_torque{torque}{subtalar} muscles')

                            for icontrast, contrast in enumerate(contrasts):
                                if label == contrast:
                                    iperturb = icontrast
                                    foundContrast = True
                                    break

                            if not foundContrast:
                                raise Exception(f'Did not find statistics contrast {label}')

                        p_value = p_values[iperturb]
                        significant = p_value < 0.05
                        significances.append(significant)

                index = pd.MultiIndex.from_tuples(self.multiindex_tuples,
                    names=['time', 'perturbation'])
                df_sig = pd.DataFrame(significances, index=index, columns=['significant'])

                target_dir = os.path.dirname(target[itarget])
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                with open(target[itarget], 'w') as f:
                    df_sig.to_csv(f, line_terminator='\n')

                itarget += 1


# Center-of-mass
# --------------

class TaskPlotCenterOfMassVector(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, times, rise, fall):
        super(TaskPlotCenterOfMassVector, self).__init__(study)
        self.name = f'plot_center_of_mass_vector_rise{rise}_fall{fall}'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.figures_path = os.path.join(study.config['figures_path'])
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'center_of_mass_vector',  f'rise{rise}_fall{fall}')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)
        self.walking_speed = study.walking_speed
        self.gravity = 9.81
        self.subjects = subjects
        self.times = times
        self.rise = rise
        self.fall = fall
        self.torques = study.plot_torques
        self.subtalars = study.plot_subtalars
        self.colors = study.plot_colors

        self.kinematic_levels = ['pos', 'vel', 'acc']
        self.planes = ['sagittal','transverse']

        blank = ''
        self.legend_labels = [f'{blank}\neversion\n{blank}', 
                              'plantarflexion\n+\neversion', 
                              f'{blank}\nplantarflexion\n{blank}', 
                              'plantarflexion\n+\ninversion', 
                              f'{blank}\ninversion\n{blank}']

        deps = list()
        self.label_dict = dict()
        ilabel = 0
        for isubj, subject in enumerate(subjects):

            # Unperturbed solutions
            # ---------------------
            deps.append(
                os.path.join(
                    self.study.config['results_path'], 
                    'unperturbed', subject,
                    'center_of_mass_unperturbed.sto'))

            self.label_dict[f'{subject}_unperturbed'] = ilabel
            ilabel += 1

            for actu in [False, True]:
                torque_act = '_torque_actuators' if actu else ''
                subpath = 'torque_actuators' if actu else 'perturbed'

                for time in self.times:

                    # Unperturbed time-stepping solutions
                    # -----------------------------------
                    label = (f'perturbed_torque0_time{time}'
                             f'_rise{self.rise}_fall{self.fall}{torque_act}')
                    deps.append(
                        os.path.join(
                            self.study.config['results_path'], 
                            subpath, label, subject,
                            f'center_of_mass_{label}.sto'))

                    self.label_dict[f'{subject}_unperturbed_time{time}{torque_act}'] = ilabel
                    ilabel += 1

                    for torque, subtalar in zip(self.torques, self.subtalars):

                         # Perturbed solutions
                         # -------------------
                        label = (f'perturbed_torque{torque}_time{time}'
                                f'_rise{self.rise}_fall{self.fall}{subtalar}{torque_act}')
                        deps.append(
                            os.path.join(
                                self.study.config['results_path'], 
                                subpath, label, subject,
                                f'center_of_mass_{label}.sto')
                            )

                        self.label_dict[f'{subject}_{label}'] = ilabel
                        ilabel += 1

        targets = list()
        for kin in self.kinematic_levels:
            for plane in self.planes:
                targets += [os.path.join(self.analysis_path, 
                            f'com_vector_{kin}_{plane}.png')]

        targets += [os.path.join(self.figures_path, 'figureS5', 'figureS5.png')]
        targets += [os.path.join(self.figures_path, 'figureS4', 'figureS4.png')]
        targets += [os.path.join(self.figures_path, 'figure3', 'figure3.png')]
        targets += [os.path.join(self.figures_path, 'figure2', 'figure2.png')]
        targets += [os.path.join(self.figures_path, 'figureS3', 'figureS3.png')]
        targets += [os.path.join(self.figures_path, 'figureS2', 'figureS2.png')]

        self.add_action(deps, targets, self.plot_com_vectors)

    def plot_com_vectors(self, file_dep, target):

        # Globals
        # -------
        tick_fs = 6

        # Initialize figures
        # ------------------
        figs = list()
        axes = list()
        for kin in self.kinematic_levels:
            for iplane, plane in enumerate(self.planes):
                these_axes = list()

                if plane == 'transverse':
                    fig = plt.figure(figsize=(7, 8.5))
                    for itorque, torque in enumerate(self.torques):
                        ax = fig.add_subplot(1, len(self.torques), itorque + 1)
                        ax.set_aspect('equal')

                        ax.grid(axis='y', color='gray', alpha=0.5, linewidth=0.5, 
                                zorder=-10, clip_on=False)
                        ax.axvline(x=0, color='gray', linestyle='-',
                                linewidth=0.5, alpha=0.5, zorder=-1)
                        util.publication_spines(ax)

                        if not itorque:
                            ax.spines['left'].set_position(('outward', 10))
                        else:
                            ax.spines['left'].set_visible(False)
                            ax.set_yticklabels([])
                            ax.yaxis.set_ticks_position('none')
                            ax.tick_params(axis='y', which='both', bottom=False, 
                                           top=False, labelbottom=False)

                        ax.spines['bottom'].set_position(('outward', 65))
                        ax.set_title(self.legend_labels[itorque], pad=30, fontsize=8)
                        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                        ax.tick_params(which='minor', axis='x', direction='in')
                        these_axes.append(ax)

                elif plane == 'sagittal':
                    fig = plt.figure(figsize=(8, 7))
                    for itorque, torque in enumerate(self.torques):
                        ax = fig.add_subplot(len(self.torques), 1, itorque + 1)
                        ax.set_aspect('equal')

                        ax_r = ax.twinx()
                        ax_r.set_ylabel(self.legend_labels[itorque], 
                            rotation=270, labelpad=35, fontsize=8)

                        ax.grid(axis='x', color='gray', alpha=0.5, linewidth=0.5, 
                                zorder=-10, clip_on=False)
                        ax.axhline(y=0, color='gray', linestyle='-',
                                linewidth=0.5, alpha=0.5, zorder=-1)
                        util.publication_spines(ax)

                        if itorque == len(self.torques)-1:
                            ax.spines['bottom'].set_position(('outward', 10))
                        else:
                            ax.spines['bottom'].set_visible(False)
                            ax.set_xticklabels([])
                            ax.xaxis.set_ticks_position('none')
                            ax.tick_params(axis='x', which='both', bottom=False, 
                                           top=False, labelbottom=False)

                        ax.spines['left'].set_position(('outward', 30))
                        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                        ax.tick_params(which='minor', axis='y', direction='in')
                        these_axes.append(ax)

                        # Turn off decorations for dummy axis
                        ax_r.spines['top'].set_visible(False)
                        ax_r.spines['bottom'].set_visible(False)
                        ax_r.spines['left'].set_visible(False)
                        ax_r.spines['right'].set_visible(False)
                        ax_r.set_yticklabels([])
                        ax_r.xaxis.set_ticks_position('none')
                        ax_r.yaxis.set_ticks_position('none')
                        ax_r.tick_params(axis='x', which='both', bottom=False, 
                                       top=False, labelbottom=False)
                        ax_r.tick_params(axis='y', which='both', bottom=False, 
                                       top=False, labelbottom=False)

                axes.append(these_axes)
                figs.append(fig)       

        # Aggregate data
        # --------------
        import collections
        com_dict = collections.defaultdict(dict)
        time_dict = collections.defaultdict(dict)
        com_height_dict = dict()
        duration_dict = dict()

        for isubj, subject in enumerate(self.subjects):

            # Unperturbed center-of-mass trajectory
            # -------------------------------------
            unperturb_index = self.label_dict[f'{subject}_unperturbed']
            tableTemp = osim.TimeSeriesTable(file_dep[unperturb_index])
            com_height_dict[subject] = np.mean(tableTemp.getDependentColumn(
                                               '/|com_position_y').to_numpy())
            timeTemp = np.array(tableTemp.getIndependentColumn())
            duration_dict[subject] = timeTemp[-1] - timeTemp[0]

            for actu in [False, True]:
                torque_act = '_torque_actuators' if actu else ''

                for time in self.times:

                    # Unperturbed center-of-mass trajectory
                    # -------------------------------------
                    unperturb_index = self.label_dict[f'{subject}_unperturbed_time{time}{torque_act}']
                    unpTable = osim.TimeSeriesTable(file_dep[unperturb_index])
                    unpTimeVec = unpTable.getIndependentColumn()
                    unpTable_np = np.zeros((len(unpTimeVec), 9))
                    unp_pos_x = unpTable.getDependentColumn(
                        '/|com_position_x').to_numpy()
                    unp_pos_y = unpTable.getDependentColumn(
                        '/|com_position_y').to_numpy()
                    unp_pos_z = unpTable.getDependentColumn(
                        '/|com_position_z').to_numpy()
                    unpTable_np[:, 0] = unp_pos_x - unp_pos_x[0] 
                    unpTable_np[:, 1] = unp_pos_y - unp_pos_y[0] 
                    unpTable_np[:, 2] = unp_pos_z - unp_pos_z[0] 
                    unpTable_np[:, 3] = unpTable.getDependentColumn(
                        '/|com_velocity_x').to_numpy() 
                    unpTable_np[:, 4] = unpTable.getDependentColumn(
                        '/|com_velocity_y').to_numpy() 
                    unpTable_np[:, 5] = unpTable.getDependentColumn(
                        '/|com_velocity_z').to_numpy() 
                    unpTable_np[:, 6] = unpTable.getDependentColumn(
                        '/|com_acceleration_x').to_numpy()
                    unpTable_np[:, 7] = unpTable.getDependentColumn(
                        '/|com_acceleration_y').to_numpy()
                    unpTable_np[:, 8] = unpTable.getDependentColumn(
                        '/|com_acceleration_z').to_numpy()

                    for torque, subtalar in zip(self.torques, self.subtalars):

                        # Perturbed center-of-mass trajectory
                        # -----------------------------------
                        label = (f'{subject}_perturbed_torque{torque}_time{time}'
                                 f'_rise{self.rise}_fall{self.fall}{subtalar}{torque_act}')
                        perturb_index = self.label_dict[label]
                        table = osim.TimeSeriesTable(file_dep[perturb_index])
                        timeVec = table.getIndependentColumn()
                        N = len(timeVec)
                        table_np = np.zeros((N, 9))
                        pos_x = table.getDependentColumn(
                            '/|com_position_x').to_numpy()
                        pos_y = table.getDependentColumn(
                            '/|com_position_y').to_numpy()
                        pos_z = table.getDependentColumn(
                            '/|com_position_z').to_numpy()
                        table_np[:, 0] = pos_x - pos_x[0] 
                        table_np[:, 1] = pos_y - pos_y[0] 
                        table_np[:, 2] = pos_z - pos_z[0] 
                        table_np[:, 3] = table.getDependentColumn(
                            '/|com_velocity_x').to_numpy()
                        table_np[:, 4] = table.getDependentColumn(
                            '/|com_velocity_y').to_numpy()
                        table_np[:, 5] = table.getDependentColumn(
                            '/|com_velocity_z').to_numpy() 
                        table_np[:, 6] = table.getDependentColumn(
                            '/|com_acceleration_x').to_numpy()
                        table_np[:, 7] = table.getDependentColumn(
                            '/|com_acceleration_y').to_numpy()
                        table_np[:, 8] = table.getDependentColumn(
                            '/|com_acceleration_z').to_numpy()

                        # Compute difference between perturbed and unperturbed
                        # trajectories for this subject. We don't need to interpolate
                        # here since the perturbed and unperturbed trajectories contain
                        # the same time points (up until the end of the perturbation).
                        com_dict[subject][label] = table_np - unpTable_np
                        time_dict[subject][label] = np.array(timeVec)

        # Plot helper functions
        # ---------------------
        def set_arrow_patch_sagittal(ax, x, y, dx, dy, actu, color):
            if 'muscles' in actu:
                arrowstyle = patches.ArrowStyle.CurveFilledB(head_length=0.25, 
                    head_width=0.1)
                lw = 2.0
            elif 'torques' in actu: 
                arrowstyle = patches.ArrowStyle.CurveB()
                lw = 0.75

            arrow = patches.FancyArrowPatch((x, y), (x + dx, y + dy),
                    arrowstyle=arrowstyle, mutation_scale=10, shrinkA=0, shrinkB=0,
                    capstyle='round', joinstyle='miter', 
                    color=color, clip_on=False, zorder=2.5, lw=lw)
            ax.add_patch(arrow)
            return arrow

        def set_arrow_patch_transverse(ax, x, y, dx, dy, actu, color):
            if 'muscles' in actu:
                arrowstyle = patches.ArrowStyle.CurveFilledB(head_length=0.25, 
                    head_width=0.1)
                lw = 2.0
            elif 'torques' in actu: 
                arrowstyle = patches.ArrowStyle.CurveB()
                lw = 0.75

            arrow = patches.FancyArrowPatch((x, y), (x + dx, y + dy),
                    arrowstyle=arrowstyle, mutation_scale=10, shrinkA=0, shrinkB=0,
                    capstyle='round', joinstyle='miter', 
                    color=color, clip_on=False, zorder=2.5, lw=lw)
            ax.add_patch(arrow)
            return arrow

        # Compute changes in center-of-mass kinematics
        # --------------------------------------------
        pos_x_diff = np.zeros(len(self.subjects))
        pos_y_diff = np.zeros(len(self.subjects))
        pos_z_diff = np.zeros(len(self.subjects))
        vel_x_diff = np.zeros(len(self.subjects))
        vel_y_diff = np.zeros(len(self.subjects))
        vel_z_diff = np.zeros(len(self.subjects))
        acc_x_diff = np.zeros(len(self.subjects))
        acc_y_diff = np.zeros(len(self.subjects))
        acc_z_diff = np.zeros(len(self.subjects))
        pos_x_diff_mean = dict()
        pos_y_diff_mean = dict()
        pos_z_diff_mean = dict()
        vel_x_diff_mean = dict()
        vel_y_diff_mean = dict()
        vel_z_diff_mean = dict()
        acc_x_diff_mean = dict()
        acc_y_diff_mean = dict()
        acc_z_diff_mean = dict()
        for actu in [False, True]:
            torque_act = '_torque_actuators' if actu else ''
            actu_key = 'torques' if actu else 'muscles'
            pos_x_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            pos_y_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            pos_z_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            vel_x_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            vel_y_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            vel_z_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            acc_x_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            acc_y_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            acc_z_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))

            for itime, time in enumerate(self.times):
                zipped = zip(self.torques, self.subtalars, self.colors)
                for iperturb, (torque, subtalar, color) in enumerate(zipped):
                    for isubj, subject in enumerate(self.subjects):

                        label = (f'{subject}_perturbed_torque{torque}_time{time}'
                                 f'_rise{self.rise}_fall{self.fall}{subtalar}{torque_act}')
                        com = com_dict[subject][label]

                        # Compute the closet time index to the current peak 
                        # perturbation time. 
                        duration = duration_dict[subject]
                        timeVec = time_dict[subject][label]
                        time_at_peak = timeVec[0] + (duration * (time / 100.0))
                        index_peak = np.argmin(np.abs(timeVec - time_at_peak))
                        index_fall = -1

                        l_max = com_height_dict[subject]
                        v_max = np.sqrt(self.gravity * l_max)
                        pos_x_diff[isubj] = com[index_fall, 0] / l_max
                        pos_y_diff[isubj] = com[index_fall, 1] / l_max
                        pos_z_diff[isubj] = com[index_fall, 2] / l_max
                        vel_x_diff[isubj] = com[index_fall, 3] / v_max
                        vel_y_diff[isubj] = com[index_fall, 4] / v_max
                        vel_z_diff[isubj] = com[index_fall, 5] / v_max
                        acc_x_diff[isubj] = com[index_peak, 6] / self.gravity
                        acc_y_diff[isubj] = com[index_peak, 7] / self.gravity
                        acc_z_diff[isubj] = com[index_peak, 8] / self.gravity

                    pos_x_diff_mean[actu_key][itime, iperturb] = np.mean(pos_x_diff)
                    pos_y_diff_mean[actu_key][itime, iperturb] = np.mean(pos_y_diff)
                    pos_z_diff_mean[actu_key][itime, iperturb] = np.mean(pos_z_diff)
                    vel_x_diff_mean[actu_key][itime, iperturb] = np.mean(vel_x_diff)
                    vel_y_diff_mean[actu_key][itime, iperturb] = np.mean(vel_y_diff)
                    vel_z_diff_mean[actu_key][itime, iperturb] = np.mean(vel_z_diff)
                    acc_x_diff_mean[actu_key][itime, iperturb] = np.mean(acc_x_diff)
                    acc_y_diff_mean[actu_key][itime, iperturb] = np.mean(acc_y_diff)
                    acc_z_diff_mean[actu_key][itime, iperturb] = np.mean(acc_z_diff)

        # Set plot limits and labels
        # --------------------------
        for iperturb in np.arange(len(self.subtalars)):

            # Sagittal position
            xy_pos_scale = 0.003
            axes[0][iperturb].set_ylim(-xy_pos_scale, xy_pos_scale)
            axes[0][iperturb].set_yticks([-xy_pos_scale, 0, xy_pos_scale])
            axes[0][iperturb].set_yticklabels([-xy_pos_scale, 0, xy_pos_scale], fontsize=tick_fs)
            axes[0][iperturb].set_xticks(xy_pos_scale*np.arange(len(self.times)))
            axes[0][iperturb].set_xlim(0, xy_pos_scale*(len(self.times)-1))
            if iperturb == len(self.subtalars)-1:
                axes[0][iperturb].set_xticklabels([f'{time + 5}' for time in self.times],
                                        fontsize=tick_fs+1)
            else:
                axes[0][iperturb].set_xticklabels([])
                
            axes[0][2].set_ylabel(r'$\Delta$' + ' center of mass position $[-]$',
                fontsize=tick_fs+3)
            axes[0][4].set_xlabel('exoskeleton torque end time\n(% gait cycle)',
                fontsize=tick_fs+3)

            # Transverse position
            xz_pos_scale = 0.001
            axes[1][iperturb].set_xlim(-xz_pos_scale, xz_pos_scale)
            axes[1][iperturb].set_xticks([-xz_pos_scale, 0, xz_pos_scale])
            axes[1][iperturb].set_xticklabels([-xz_pos_scale, 0, xz_pos_scale], fontsize=tick_fs)
            axes[1][iperturb].set_yticks(xz_pos_scale*np.arange(len(self.times)))
            axes[1][iperturb].set_ylim(0, xz_pos_scale*(len(self.times)-1))
            if iperturb:
                axes[1][iperturb].set_yticklabels([])
            else:
                axes[1][iperturb].set_yticklabels([f'{time + 5}' for time in self.times],
                                        fontsize=tick_fs+1)
            axes[1][2].set_xlabel(r'$\Delta$' + ' center of mass position $[-]$',
                fontsize=tick_fs+3)
            axes[1][0].set_ylabel('exoskeleton torque end time\n(% gait cycle)',
                fontsize=tick_fs+3)

            # Sagittal velocity
            xy_vel_scale = 0.02
            axes[2][iperturb].set_ylim(-xy_vel_scale, xy_vel_scale)
            axes[2][iperturb].set_yticks([-xy_vel_scale, 0, xy_vel_scale])
            axes[2][iperturb].set_yticklabels([-xy_vel_scale, 0, xy_vel_scale], fontsize=tick_fs)
            axes[2][iperturb].set_xticks(xy_vel_scale*np.arange(len(self.times)))
            axes[2][iperturb].set_xlim(0, xy_vel_scale*(len(self.times)-1))
            if iperturb == len(self.subtalars)-1:
                axes[2][iperturb].set_xticklabels([f'{time + 5}' for time in self.times],
                                        fontsize=tick_fs+1)
            else:
                axes[2][iperturb].set_xticklabels([])
                
            axes[2][2].set_ylabel(r'$\Delta$' + ' center of mass velocity $[-]$',
                fontsize=tick_fs+3)
            axes[2][4].set_xlabel('exoskeleton torque end time\n(% gait cycle)',
                fontsize=tick_fs+3)

            # Transverse velocity
            xz_vel_scale = 0.007
            axes[3][iperturb].set_xlim(-xz_vel_scale, xz_vel_scale)
            axes[3][iperturb].set_xticks([-xz_vel_scale, 0, xz_vel_scale])
            axes[3][iperturb].set_xticklabels([-xz_vel_scale, 0, xz_vel_scale], fontsize=tick_fs)
            axes[3][iperturb].set_yticks(xz_vel_scale*np.arange(len(self.times)))
            axes[3][iperturb].set_ylim(0, xz_vel_scale*(len(self.times)-1))
            if iperturb:
                axes[3][iperturb].set_yticklabels([])
            else:
                axes[3][iperturb].set_yticklabels([f'{time + 5}' for time in self.times],
                                        fontsize=tick_fs+1)
            axes[3][2].set_xlabel(r'$\Delta$' + ' center of mass velocity $[-]$',
                fontsize=tick_fs+3)
            axes[3][0].set_ylabel('exoskeleton torque end time\n(% gait cycle)',
                fontsize=tick_fs+3)

            # Sagittal acceleration
            xy_acc_scale = 0.07
            axes[4][iperturb].set_ylim(-xy_acc_scale, xy_acc_scale)
            axes[4][iperturb].set_yticks([-xy_acc_scale, 0, xy_acc_scale])
            axes[4][iperturb].set_yticklabels([-xy_acc_scale, 0, xy_acc_scale], fontsize=tick_fs)
            axes[4][iperturb].set_xticks(xy_acc_scale*np.arange(len(self.times)))
            axes[4][iperturb].set_xlim(0, xy_acc_scale*(len(self.times)-1))
            if iperturb == len(self.subtalars)-1:
                axes[4][iperturb].set_xticklabels([f'{time}' for time in self.times],
                                        fontsize=tick_fs+1)
            else:
                axes[4][iperturb].set_xticklabels([])
                
            axes[4][2].set_ylabel(r'$\Delta$' + ' center of mass acceleration $[-]$',
                fontsize=tick_fs+3)
            axes[4][4].set_xlabel('exoskeleton torque peak time\n(% gait cycle)',
                fontsize=tick_fs+3)

            # Transverse acceleration
            xz_acc_scale = 0.03
            axes[5][iperturb].set_xlim(-xz_acc_scale, xz_acc_scale)
            axes[5][iperturb].set_xticks([-xz_acc_scale, 0, xz_acc_scale])
            axes[5][iperturb].set_xticklabels([-xz_acc_scale, 0, xz_acc_scale], fontsize=tick_fs)
            axes[5][iperturb].set_yticks(xz_acc_scale*np.arange(len(self.times)))
            axes[5][iperturb].set_ylim(0, xz_acc_scale*(len(self.times)-1))
            if iperturb:
                axes[5][iperturb].set_yticklabels([])
            else:
                axes[5][iperturb].set_yticklabels([f'{time}' for time in self.times],
                                        fontsize=tick_fs+1)
            axes[5][2].set_xlabel(r'$\Delta$' + ' center of mass acceleration $[-]$',
                fontsize=tick_fs+3)
            axes[5][0].set_ylabel('exoskeleton torque peak time\n(% gait cycle)',
                fontsize=tick_fs+3)
        
        # Plot results
        # ------------
        for actu in [True, False]:
            torque_act = '_torque_actuators' if actu else ''
            actu_key = 'torques' if actu else 'muscles'

            for itime, time in enumerate(self.times):
                zipped = zip(self.torques, self.subtalars, self.colors)
                for iperturb, (torque, subtalar, color) in enumerate(zipped):

                    # Position vectors
                    # ----------------
                    pos_x_diff = pos_x_diff_mean[actu_key][itime, iperturb]
                    pos_y_diff = pos_y_diff_mean[actu_key][itime, iperturb]
                    pos_z_diff = pos_z_diff_mean[actu_key][itime, iperturb]
                    set_arrow_patch_sagittal(axes[0][iperturb], xy_pos_scale*itime, 0, pos_x_diff, pos_y_diff, actu_key, color)
                    set_arrow_patch_transverse(axes[1][iperturb], 0, xz_pos_scale*itime, pos_z_diff, pos_x_diff, actu_key, color)

                    # Velocity vectors
                    # ----------------
                    vel_x_diff = vel_x_diff_mean[actu_key][itime, iperturb]
                    vel_y_diff = vel_y_diff_mean[actu_key][itime, iperturb]
                    vel_z_diff = vel_z_diff_mean[actu_key][itime, iperturb]
                    set_arrow_patch_sagittal(axes[2][iperturb], xy_vel_scale*itime, 0, vel_x_diff, vel_y_diff, actu_key, color)
                    set_arrow_patch_transverse(axes[3][iperturb], 0, xz_vel_scale*itime, vel_z_diff, vel_x_diff, actu_key, color)

                    # Acceleration vectors
                    # --------------------
                    acc_x_diff = acc_x_diff_mean[actu_key][itime, iperturb]
                    acc_y_diff = acc_y_diff_mean[actu_key][itime, iperturb]
                    acc_z_diff = acc_z_diff_mean[actu_key][itime, iperturb]
                    set_arrow_patch_sagittal(axes[4][iperturb], xy_acc_scale*itime, 0, acc_x_diff, acc_y_diff, actu_key, color)
                    set_arrow_patch_transverse(axes[5][iperturb], 0, xz_acc_scale*itime, acc_z_diff, acc_x_diff, actu_key, color)

        left = 0.4225
        right = 0.9
        bottom = 0.1
        top = 0.95
        hspace = 0.3
        figs[0].subplots_adjust(left=left, right=right, bottom=bottom, top=top, hspace=hspace)
        figs[2].subplots_adjust(left=left, right=right, bottom=bottom, top=top, hspace=hspace)
        figs[4].subplots_adjust(left=left, right=right, bottom=bottom, top=top, hspace=hspace)

        left = 0.1
        right = 0.95
        bottom = 0.425
        top = 0.9
        wspace = 0.3
        figs[1].subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace)
        figs[3].subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace)
        figs[5].subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace)
        
        import cv2
        def add_sagittal_image(fig):
            side = 0.35
            offset = 0.01
            l = -0.01
            b = ((1.0 - side) / 2.0) + offset
            w = side
            h = side
            ax = fig.add_axes([l, b, w, h], projection=None, polar=False)
            image = cv2.imread(
                os.path.join(self.study.config['figures_path'], 'images',
                    'sagittal_with_arrows.tiff'))[..., ::-1]
            ax.imshow(image, interpolation='none', aspect='equal')

            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yticklabels([])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.tick_params(axis='x', which='both', bottom=False, 
                           top=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', bottom=False, 
                           top=False, labelbottom=False)

        def add_transverse_image(fig):
            side = 0.35
            offset = 0.02 
            l = ((1.0 - side) / 2.0) + offset
            b = -0.02
            w = side
            h = side
            ax = fig.add_axes([l, b, w, h], projection=None, polar=False)
            image = cv2.imread(
                os.path.join(self.study.config['figures_path'], 'images',
                    'transverse_with_arrows.tiff'))[..., ::-1]
            ax.imshow(image, interpolation='none', aspect='equal')

            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yticklabels([])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.tick_params(axis='x', which='both', bottom=False, 
                           top=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', bottom=False, 
                           top=False, labelbottom=False)

        def add_legend(fig, x, y, transverse=False):
            w = 0.1
            h = 0.02
            ax = fig.add_axes([x, y, w, h], projection=None, polar=False)

            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yticklabels([])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.tick_params(axis='x', which='both', bottom=False, 
                           top=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', bottom=False, 
                           top=False, labelbottom=False)

            set_arrow_patch_transverse(ax, 0, 1.25, 0.5, 0, 'muscles', 'black')
            ax.text(0.6, 1.1, 'muscle-driven model')

            if transverse:
                y_arrow = 0.2
                y_text = 0
            else:
                y_arrow = 0
                y_text = -0.15
            set_arrow_patch_transverse(ax, 0, y_arrow, 0.5, 0, 'torques', 'black')
            ax.text(0.6, y_text, 'torque-driven model')

        add_sagittal_image(figs[0])
        add_sagittal_image(figs[2])
        add_sagittal_image(figs[4])
        x = 0.075
        y = 0.15
        add_legend(figs[0], x, y)
        add_legend(figs[2], x, y)
        add_legend(figs[4], x, y)

        add_transverse_image(figs[1])
        add_transverse_image(figs[3])
        add_transverse_image(figs[5])
        x = 0.675
        y = 0.225
        add_legend(figs[1], x, y, transverse=True)
        add_legend(figs[3], x, y, transverse=True)
        add_legend(figs[5], x, y, transverse=True)

        for ifig, fig in enumerate(figs):
            fig.savefig(target[ifig], dpi=600)

        for ifig, fig in enumerate(figs):
            fig.savefig(target[ifig + len(figs)], dpi=600)
 
        plt.close()


class TaskPlotInstantaneousCenterOfMass(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, times, rise, fall):
        super(TaskPlotInstantaneousCenterOfMass, self).__init__(study)
        self.name = f'plot_instantaneous_center_of_mass_rise{rise}_fall{fall}'
        self.figures_path = os.path.join(study.config['figures_path']) 
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.aggregate_path = os.path.join(study.config['statistics_path'],
            'center_of_mass', 'aggregate')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'center_of_mass_instantaneous',  f'rise{rise}_fall{fall}')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)
        self.subjects = subjects
        self.times = times
        self.rise = rise
        self.fall = fall
        self.gravity = 9.81
        self.torques = study.plot_torques
        self.subtalars = study.plot_subtalars
        self.colors = study.plot_colors
        self.hatches = [None, None, None, None, None]
        self.edgecolor = 'black'
        self.width = 0.15
        N = len(self.subtalars)
        min_width = -self.width*((N-1)/2)
        max_width = -min_width
        self.shifts = np.linspace(min_width, max_width, N)
        self.legend_labels = ['eversion', 
                              'plantarflexion + eversion', 
                              'plantarflexion', 
                              'plantarflexion + inversion', 
                              'inversion']
        deps = list()
        self.label_dict = dict()
        ilabel = 0
        for isubj, subject in enumerate(subjects):

            # Unperturbed solutions
            # ---------------------
            deps.append(
                os.path.join(
                    self.study.config['results_path'], 
                    'unperturbed', subject,
                    'center_of_mass_unperturbed.sto'))

            self.label_dict[f'{subject}_unperturbed'] = ilabel
            ilabel += 1

            for actu in [False, True]:
                torque_act = '_torque_actuators' if actu else ''
                subpath = 'torque_actuators' if actu else 'perturbed'

                for time in self.times:

                    # Unperturbed time-stepping solutions
                    # -----------------------------------
                    label = (f'perturbed_torque0_time{time}'
                             f'_rise{self.rise}_fall{self.fall}{torque_act}')
                    deps.append(
                        os.path.join(
                            self.study.config['results_path'], 
                            subpath, label, subject,
                            f'center_of_mass_{label}.sto'))

                    self.label_dict[f'{subject}_unperturbed_time{time}{torque_act}'] = ilabel
                    ilabel += 1

                    for torque, subtalar in zip(self.torques, self.subtalars):

                         # Perturbed solutions
                         # -------------------
                        label = (f'perturbed_torque{torque}_time{time}'
                                f'_rise{self.rise}_fall{self.fall}{subtalar}{torque_act}')
                        deps.append(
                            os.path.join(
                                self.study.config['results_path'], 
                                subpath, label, subject,
                                f'center_of_mass_{label}.sto')
                            )

                        self.label_dict[f'{subject}_{label}'] = ilabel
                        ilabel += 1

        # Statistics results
        for actu in ['muscles', 'torques', 'diff']:
            for kin in ['pos', 'vel', 'acc']:
                for direc in ['x', 'y', 'z']:
                    label = f'com_stats_{kin}_{direc}_{actu}'
                    deps.append(os.path.join(self.aggregate_path, f'{label}.csv'))

                    self.label_dict[label] = ilabel
                    ilabel += 1


        targets = list()
        for kin in ['pos', 'vel', 'acc']:
            targets += [os.path.join(self.analysis_path, 
                        f'instant_com_{kin}.png')]

        targets += [os.path.join(self.figures_path, 'figureS8', 'figureS8.png')]
        targets += [os.path.join(self.figures_path, 'figureS7', 'figureS7.png')]
        targets += [os.path.join(self.figures_path, 'figureS6', 'figureS6.png')]

        targets += [os.path.join(self.analysis_path, 'com_height.txt')]

        self.add_action(deps, targets, self.plot_instantaneous_com)

    def plot_instantaneous_com(self, file_dep, target):

        # Initialize figures
        # ------------------
        from collections import defaultdict
        figs = list()
        axes = defaultdict(list)
        for kin in ['pos', 'vel', 'acc']:
            fig = plt.figure(figsize=(9, 10))
            for iactu, actu in enumerate(['muscles', 'torques']):
                for idirec, direc in enumerate(['AP', 'SI', 'ML']):
                    index = 2*idirec + iactu + 1 
                    ax = fig.add_subplot(3, 2, index)
                    ax.axhline(y=0, color='black', linestyle='-',
                            linewidth=0.1, alpha=1.0, zorder=2.5)
                    ax.spines['left'].set_position(('outward', 30))
                    ax.set_xticks(np.arange(len(self.times)))
                    ax.set_xlim(0, len(self.times)-1)
                    ax.grid(color='gray', linestyle='--', linewidth=0.4,
                        clip_on=False, alpha=0.75, zorder=0)
                    util.publication_spines(ax)

                    if not direc == 'ML':
                        ax.spines['bottom'].set_visible(False)
                        ax.set_xticklabels([])
                        ax.xaxis.set_ticks_position('none')
                        ax.tick_params(axis='x', which='both', bottom=False, 
                                       top=False, labelbottom=False)
                    else:
                        ax.spines['bottom'].set_position(('outward', 10))
                        if kin == 'pos' or kin == 'vel':
                            ax.set_xticklabels([f'{time + 5}' for time in self.times])
                            ax.set_xlabel('exoskeleton torque end time\n(% gait cycle)')
                        else:
                            ax.set_xticklabels([f'{time}' for time in self.times])
                            ax.set_xlabel('exoskeleton torque peak time\n(% gait cycle)')

                    if actu == 'torques':
                        ax.spines['left'].set_visible(False)
                        ax.set_yticklabels([])
                        ax.yaxis.set_ticks_position('none')
                        ax.tick_params(axis='y', which='both', left=False, 
                                       right=False, labelleft=False)


                    axes[actu].append(ax)

            figs.append(fig)

        # Aggregate data
        # --------------
        import collections
        com_dict = collections.defaultdict(dict)
        time_dict = collections.defaultdict(dict)
        com_height_dict = dict()
        duration_dict = dict()
        for isubj, subject in enumerate(self.subjects):

            # Unperturbed center-of-mass trajectory
            # -------------------------------------
            unperturb_index = self.label_dict[f'{subject}_unperturbed']
            tableTemp = osim.TimeSeriesTable(file_dep[unperturb_index])
            com_height_dict[subject] = np.mean(tableTemp.getDependentColumn(
                                               '/|com_position_y').to_numpy())
            timeTemp = np.array(tableTemp.getIndependentColumn())
            duration_dict[subject] = timeTemp[-1] - timeTemp[0]
            for actu in [False, True]:
                torque_act = '_torque_actuators' if actu else ''

                for time in self.times:

                    # Unperturbed center-of-mass trajectory
                    # -------------------------------------
                    unperturb_index = self.label_dict[f'{subject}_unperturbed_time{time}{torque_act}']
                    unpTable = osim.TimeSeriesTable(file_dep[unperturb_index])
                    unpTimeVec = unpTable.getIndependentColumn()
                    unpTable_np = np.zeros((len(unpTimeVec), 9))
                    unp_pos_x = unpTable.getDependentColumn(
                        '/|com_position_x').to_numpy()
                    unp_pos_y = unpTable.getDependentColumn(
                        '/|com_position_y').to_numpy()
                    unp_pos_z = unpTable.getDependentColumn(
                        '/|com_position_z').to_numpy()
                    unpTable_np[:, 0] = unp_pos_x - unp_pos_x[0] 
                    unpTable_np[:, 1] = unp_pos_y - unp_pos_y[0] 
                    unpTable_np[:, 2] = unp_pos_z - unp_pos_z[0] 
                    unpTable_np[:, 3] = unpTable.getDependentColumn(
                        '/|com_velocity_x').to_numpy() 
                    unpTable_np[:, 4] = unpTable.getDependentColumn(
                        '/|com_velocity_y').to_numpy() 
                    unpTable_np[:, 5] = unpTable.getDependentColumn(
                        '/|com_velocity_z').to_numpy() 
                    unpTable_np[:, 6] = unpTable.getDependentColumn(
                        '/|com_acceleration_x').to_numpy()
                    unpTable_np[:, 7] = unpTable.getDependentColumn(
                        '/|com_acceleration_y').to_numpy()
                    unpTable_np[:, 8] = unpTable.getDependentColumn(
                        '/|com_acceleration_z').to_numpy()

                    for torque, subtalar in zip(self.torques, self.subtalars):

                        # Perturbed center-of-mass trajectory
                        # -----------------------------------
                        label = (f'{subject}_perturbed_torque{torque}_time{time}'
                                 f'_rise{self.rise}_fall{self.fall}{subtalar}{torque_act}')
                        perturb_index = self.label_dict[label]
                        table = osim.TimeSeriesTable(file_dep[perturb_index])
                        timeVec = table.getIndependentColumn()
                        N = len(timeVec)
                        table_np = np.zeros((N, 9))
                        pos_x = table.getDependentColumn(
                            '/|com_position_x').to_numpy()
                        pos_y = table.getDependentColumn(
                            '/|com_position_y').to_numpy()
                        pos_z = table.getDependentColumn(
                            '/|com_position_z').to_numpy()
                        table_np[:, 0] = pos_x - pos_x[0] 
                        table_np[:, 1] = pos_y - pos_y[0] 
                        table_np[:, 2] = pos_z - pos_z[0] 
                        table_np[:, 3] = table.getDependentColumn(
                            '/|com_velocity_x').to_numpy()
                        table_np[:, 4] = table.getDependentColumn(
                            '/|com_velocity_y').to_numpy()
                        table_np[:, 5] = table.getDependentColumn(
                            '/|com_velocity_z').to_numpy() 
                        table_np[:, 6] = table.getDependentColumn(
                            '/|com_acceleration_x').to_numpy()
                        table_np[:, 7] = table.getDependentColumn(
                            '/|com_acceleration_y').to_numpy()
                        table_np[:, 8] = table.getDependentColumn(
                            '/|com_acceleration_z').to_numpy()

                        # Compute difference between perturbed and unperturbed
                        # trajectories for this subject. We don't need to interpolate
                        # here since the perturbed and unperturbed trajectories contain
                        # the same time points (up until the end of the perturbation).
                        com_dict[subject][label] = table_np - unpTable_np
                        time_dict[subject][label] = timeVec

        # Plotting
        # --------
        pos_x_diff = np.zeros(len(self.subjects))
        pos_y_diff = np.zeros(len(self.subjects))
        pos_z_diff = np.zeros(len(self.subjects))
        vel_x_diff = np.zeros(len(self.subjects))
        vel_y_diff = np.zeros(len(self.subjects))
        vel_z_diff = np.zeros(len(self.subjects))
        acc_x_diff = np.zeros(len(self.subjects))
        acc_y_diff = np.zeros(len(self.subjects))
        acc_z_diff = np.zeros(len(self.subjects))
        pos_x_diff_mean = dict()
        pos_y_diff_mean = dict()
        pos_z_diff_mean = dict()
        vel_x_diff_mean = dict()
        vel_y_diff_mean = dict()
        vel_z_diff_mean = dict()
        acc_x_diff_mean = dict()
        acc_y_diff_mean = dict()
        acc_z_diff_mean = dict()
        pos_x_diff_std = dict()
        pos_y_diff_std = dict()
        pos_z_diff_std = dict()
        vel_x_diff_std = dict()
        vel_y_diff_std = dict()
        vel_z_diff_std = dict()
        acc_x_diff_std = dict()
        acc_y_diff_std = dict()
        acc_z_diff_std = dict()
        for actu in [False, True]:
            torque_act = '_torque_actuators' if actu else ''
            actu_key = 'torques' if actu else 'muscles'
            pos_x_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            pos_y_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            pos_z_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            vel_x_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            vel_y_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            vel_z_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            acc_x_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            acc_y_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            acc_z_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            pos_x_diff_std[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            pos_y_diff_std[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            pos_z_diff_std[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            vel_x_diff_std[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            vel_y_diff_std[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            vel_z_diff_std[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            acc_x_diff_std[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            acc_y_diff_std[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            acc_z_diff_std[actu_key] = np.zeros((len(self.times), len(self.subtalars)))

            for itime, time in enumerate(self.times):
                zipped = zip(self.torques, self.subtalars, self.colors, self.shifts)
                for isubt, (torque, subtalar, color, shift) in enumerate(zipped):
                    for isubj, subject in enumerate(self.subjects):
                        label = (f'{subject}_perturbed_torque{torque}_time{time}'
                             f'_rise{self.rise}_fall{self.fall}{subtalar}{torque_act}')
                        com = com_dict[subject][label]

                        # Compute the closet time index to the current peak 
                        # perturbation time. 
                        timeVec = time_dict[subject][label]
                        duration = duration_dict[subject]
                        time_at_peak = timeVec[0] + (duration * (time / 100.0))
                        index_peak = np.argmin(np.abs(timeVec - time_at_peak))
                        index_fall = -1

                        l_max = com_height_dict[subject]
                        v_max = np.sqrt(self.gravity * l_max)
                        pos_x_diff[isubj] = com[index_fall, 0] / l_max
                        pos_y_diff[isubj] = com[index_fall, 1] / l_max
                        pos_z_diff[isubj] = com[index_fall, 2] / l_max
                        vel_x_diff[isubj] = com[index_fall, 3] / v_max
                        vel_y_diff[isubj] = com[index_fall, 4] / v_max
                        vel_z_diff[isubj] = com[index_fall, 5] / v_max
                        acc_x_diff[isubj] = com[index_peak, 6] / self.gravity
                        acc_y_diff[isubj] = com[index_peak, 7] / self.gravity
                        acc_z_diff[isubj] = com[index_peak, 8] / self.gravity

                    pos_x_diff_mean[actu_key][itime, isubt] = np.mean(pos_x_diff)
                    pos_y_diff_mean[actu_key][itime, isubt] = np.mean(pos_y_diff)
                    pos_z_diff_mean[actu_key][itime, isubt] = np.mean(pos_z_diff)
                    vel_x_diff_mean[actu_key][itime, isubt] = np.mean(vel_x_diff)
                    vel_y_diff_mean[actu_key][itime, isubt] = np.mean(vel_y_diff)
                    vel_z_diff_mean[actu_key][itime, isubt] = np.mean(vel_z_diff)
                    acc_x_diff_mean[actu_key][itime, isubt] = np.mean(acc_x_diff)
                    acc_y_diff_mean[actu_key][itime, isubt] = np.mean(acc_y_diff)
                    acc_z_diff_mean[actu_key][itime, isubt] = np.mean(acc_z_diff)
                    pos_x_diff_std[actu_key][itime, isubt] = np.std(pos_x_diff)
                    pos_y_diff_std[actu_key][itime, isubt] = np.std(pos_y_diff)
                    pos_z_diff_std[actu_key][itime, isubt] = np.std(pos_z_diff)
                    vel_x_diff_std[actu_key][itime, isubt] = np.std(vel_x_diff)
                    vel_y_diff_std[actu_key][itime, isubt] = np.std(vel_y_diff)
                    vel_z_diff_std[actu_key][itime, isubt] = np.std(vel_z_diff)
                    acc_x_diff_std[actu_key][itime, isubt] = np.std(acc_x_diff)
                    acc_y_diff_std[actu_key][itime, isubt] = np.std(acc_y_diff)
                    acc_z_diff_std[actu_key][itime, isubt] = np.std(acc_z_diff)


        def get_offsets(means, stds, lim, shift=0.0):
            min_value = 0
            max_value = 0

            if np.any(means < 0):
                min_value = np.min(means[means < 0] - stds[means < 0])

            if np.any(means > 0):
                max_value = np.max(means[means > 0] + stds[means > 0])

            lim_range = lim[1] - lim[0]

            offsets = np.zeros_like(means)

            if np.any(means < 0):
                offsets[means < 0] = min_value - (0.06 + shift)*lim_range
            if np.any(means > 0):
                offsets[means > 0] = max_value - (0.01 - 6*shift)*lim_range

            return offsets

        pos_x_step = 0.001
        pos_y_step = 0.001
        pos_z_step = 0.0005
        vel_x_step = 0.005
        vel_y_step = 0.01
        vel_z_step = 0.002
        acc_x_step = 0.02
        acc_y_step = 0.02
        acc_z_step = 0.01
        pos_x_lim = [0.0, 0.0]
        pos_y_lim = [0.0, 0.0]
        pos_z_lim = [0.0, 0.0]
        vel_x_lim = [0.0, 0.0]
        vel_y_lim = [0.0, 0.0]
        vel_z_lim = [0.0, 0.0]
        acc_x_lim = [0.0, 0.0]
        acc_y_lim = [0.0, 0.0]
        acc_z_lim = [0.0, 0.0]
        for actu in ['muscles', 'torques']:
            update_lims(pos_x_diff_mean[actu]-pos_x_diff_std[actu], pos_x_step, pos_x_lim, mirror=True)
            update_lims(pos_y_diff_mean[actu]-pos_y_diff_std[actu], pos_y_step, pos_y_lim)
            update_lims(pos_z_diff_mean[actu]-pos_z_diff_std[actu], pos_z_step, pos_z_lim)
            update_lims(vel_x_diff_mean[actu]-vel_x_diff_std[actu], vel_x_step, vel_x_lim, mirror=True)
            update_lims(vel_y_diff_mean[actu]-vel_y_diff_std[actu], vel_y_step, vel_y_lim)
            update_lims(vel_z_diff_mean[actu]-vel_z_diff_std[actu], vel_z_step, vel_z_lim)
            update_lims(acc_x_diff_mean[actu]-acc_x_diff_std[actu], acc_x_step, acc_x_lim, mirror=True)
            update_lims(acc_y_diff_mean[actu]-acc_y_diff_std[actu], acc_y_step, acc_y_lim)
            update_lims(acc_z_diff_mean[actu]-acc_z_diff_std[actu], acc_z_step, acc_z_lim)        
            update_lims(pos_x_diff_mean[actu]+pos_x_diff_std[actu], pos_x_step, pos_x_lim, mirror=True)
            update_lims(pos_y_diff_mean[actu]+pos_y_diff_std[actu], pos_y_step, pos_y_lim)
            update_lims(pos_z_diff_mean[actu]+pos_z_diff_std[actu], pos_z_step, pos_z_lim)
            update_lims(vel_x_diff_mean[actu]+vel_x_diff_std[actu], vel_x_step, vel_x_lim, mirror=True)
            update_lims(vel_y_diff_mean[actu]+vel_y_diff_std[actu], vel_y_step, vel_y_lim)
            update_lims(vel_z_diff_mean[actu]+vel_z_diff_std[actu], vel_z_step, vel_z_lim)
            update_lims(acc_x_diff_mean[actu]+acc_x_diff_std[actu], acc_x_step, acc_x_lim, mirror=True)
            update_lims(acc_y_diff_mean[actu]+acc_y_diff_std[actu], acc_y_step, acc_y_lim)
            update_lims(acc_z_diff_mean[actu]+acc_z_diff_std[actu], acc_z_step, acc_z_lim)

        diff_stats_pos_x = pd.read_csv(file_dep[self.label_dict['com_stats_pos_x_diff']], index_col=[0, 1])
        diff_stats_pos_y = pd.read_csv(file_dep[self.label_dict['com_stats_pos_y_diff']], index_col=[0, 1])
        diff_stats_pos_z = pd.read_csv(file_dep[self.label_dict['com_stats_pos_z_diff']], index_col=[0, 1])
        diff_stats_vel_x = pd.read_csv(file_dep[self.label_dict['com_stats_vel_x_diff']], index_col=[0, 1])
        diff_stats_vel_y = pd.read_csv(file_dep[self.label_dict['com_stats_vel_y_diff']], index_col=[0, 1])
        diff_stats_vel_z = pd.read_csv(file_dep[self.label_dict['com_stats_vel_z_diff']], index_col=[0, 1])
        diff_stats_acc_x = pd.read_csv(file_dep[self.label_dict['com_stats_acc_x_diff']], index_col=[0, 1])
        diff_stats_acc_y = pd.read_csv(file_dep[self.label_dict['com_stats_acc_y_diff']], index_col=[0, 1])
        diff_stats_acc_z = pd.read_csv(file_dep[self.label_dict['com_stats_acc_z_diff']], index_col=[0, 1])       
        for actu in ['muscles', 'torques']:
            stats_pos_x = pd.read_csv(file_dep[self.label_dict[f'com_stats_pos_x_{actu}']], index_col=[0, 1])
            stats_pos_y = pd.read_csv(file_dep[self.label_dict[f'com_stats_pos_y_{actu}']], index_col=[0, 1])
            stats_pos_z = pd.read_csv(file_dep[self.label_dict[f'com_stats_pos_z_{actu}']], index_col=[0, 1])
            stats_vel_x = pd.read_csv(file_dep[self.label_dict[f'com_stats_vel_x_{actu}']], index_col=[0, 1])
            stats_vel_y = pd.read_csv(file_dep[self.label_dict[f'com_stats_vel_y_{actu}']], index_col=[0, 1])
            stats_vel_z = pd.read_csv(file_dep[self.label_dict[f'com_stats_vel_z_{actu}']], index_col=[0, 1])
            stats_acc_x = pd.read_csv(file_dep[self.label_dict[f'com_stats_acc_x_{actu}']], index_col=[0, 1])
            stats_acc_y = pd.read_csv(file_dep[self.label_dict[f'com_stats_acc_y_{actu}']], index_col=[0, 1])
            stats_acc_z = pd.read_csv(file_dep[self.label_dict[f'com_stats_acc_z_{actu}']], index_col=[0, 1])

            handles_pos = list()
            handles_vel = list()
            handles_acc = list()
            for itime, time in enumerate(self.times):
                pos_x_offsets = get_offsets(pos_x_diff_mean[actu][itime, :], pos_x_diff_std[actu][itime, :], pos_x_lim)
                pos_y_offsets = get_offsets(pos_y_diff_mean[actu][itime, :], pos_y_diff_std[actu][itime, :], pos_y_lim)
                pos_z_offsets = get_offsets(pos_z_diff_mean[actu][itime, :], pos_z_diff_std[actu][itime, :], pos_z_lim)
                vel_x_offsets = get_offsets(vel_x_diff_mean[actu][itime, :], vel_x_diff_std[actu][itime, :], vel_x_lim)
                vel_y_offsets = get_offsets(vel_y_diff_mean[actu][itime, :], vel_y_diff_std[actu][itime, :], vel_y_lim)
                vel_z_offsets = get_offsets(vel_z_diff_mean[actu][itime, :], vel_z_diff_std[actu][itime, :], vel_z_lim)
                acc_x_offsets = get_offsets(acc_x_diff_mean[actu][itime, :], acc_x_diff_std[actu][itime, :], acc_x_lim)
                acc_y_offsets = get_offsets(acc_y_diff_mean[actu][itime, :], acc_y_diff_std[actu][itime, :], acc_y_lim)
                acc_z_offsets = get_offsets(acc_z_diff_mean[actu][itime, :], acc_z_diff_std[actu][itime, :], acc_z_lim)
                diff_pos_x_offsets = get_offsets(pos_x_diff_mean[actu][itime, :], pos_x_diff_std[actu][itime, :], pos_x_lim, shift=0.01)
                diff_pos_y_offsets = get_offsets(pos_y_diff_mean[actu][itime, :], pos_y_diff_std[actu][itime, :], pos_y_lim, shift=0.01)
                diff_pos_z_offsets = get_offsets(pos_z_diff_mean[actu][itime, :], pos_z_diff_std[actu][itime, :], pos_z_lim, shift=0.01)
                diff_vel_x_offsets = get_offsets(vel_x_diff_mean[actu][itime, :], vel_x_diff_std[actu][itime, :], vel_x_lim, shift=0.01)
                diff_vel_y_offsets = get_offsets(vel_y_diff_mean[actu][itime, :], vel_y_diff_std[actu][itime, :], vel_y_lim, shift=0.01)
                diff_vel_z_offsets = get_offsets(vel_z_diff_mean[actu][itime, :], vel_z_diff_std[actu][itime, :], vel_z_lim, shift=0.01)
                diff_acc_x_offsets = get_offsets(acc_x_diff_mean[actu][itime, :], acc_x_diff_std[actu][itime, :], acc_x_lim, shift=0.01)
                diff_acc_y_offsets = get_offsets(acc_y_diff_mean[actu][itime, :], acc_y_diff_std[actu][itime, :], acc_y_lim, shift=0.01)
                diff_acc_z_offsets = get_offsets(acc_z_diff_mean[actu][itime, :], acc_z_diff_std[actu][itime, :], acc_z_lim, shift=0.01) 

                zipped = zip(self.torques, self.subtalars, self.colors, self.shifts, self.hatches)
                for isubt, (torque, subtalar, color, shift, hatch) in enumerate(zipped):
                    perturbation = f'perturbed_torque{torque}{subtalar}'

                    # Set the x-position for these bar chart entries.
                    x = itime + shift
                    lw = 0.2

                    # Instantaneous positions
                    # -----------------------
                    plot_errorbar(axes[actu][0], x, pos_x_diff_mean[actu][itime, isubt], pos_x_diff_std[actu][itime, isubt])
                    h_pos = axes[actu][0].bar(x, pos_x_diff_mean[actu][itime, isubt], self.width, color=color, clip_on=False, 
                        hatch=hatch, edgecolor=self.edgecolor, lw=lw, zorder=2.5)
                    if stats_pos_x.loc[(time, perturbation)]['significant']:
                        axes[actu][0].text(x, pos_x_offsets[isubt], '*', ha='center', fontsize=10)
                        if (actu == 'torques') and diff_stats_pos_x.loc[(time, perturbation)]['significant']:
                            axes[actu][0].text(x, diff_pos_x_offsets[isubt], r'$ \diamond $', ha='center', fontsize=8)
                    handles_pos.append(h_pos)
                    
                    plot_errorbar(axes[actu][1], x, pos_y_diff_mean[actu][itime, isubt], pos_y_diff_std[actu][itime, isubt])
                    axes[actu][1].bar(x, pos_y_diff_mean[actu][itime, isubt], self.width, color=color, clip_on=False, 
                        hatch=hatch, edgecolor=self.edgecolor , lw=lw, zorder=2.5)
                    if stats_pos_y.loc[(time, perturbation)]['significant']:
                        axes[actu][1].text(x, pos_y_offsets[isubt], '*', ha='center', fontsize=10)
                        if (actu == 'torques') and diff_stats_pos_y.loc[(time, perturbation)]['significant']:
                            axes[actu][1].text(x, diff_pos_y_offsets[isubt], r'$ \diamond $', ha='center', fontsize=8)

                    plot_errorbar(axes[actu][2], x, pos_z_diff_mean[actu][itime, isubt], pos_z_diff_std[actu][itime, isubt])
                    axes[actu][2].bar(x, pos_z_diff_mean[actu][itime, isubt], self.width, color=color, clip_on=False, 
                        hatch=hatch, edgecolor=self.edgecolor , lw=lw, zorder=2.5)
                    if stats_pos_z.loc[(time, perturbation)]['significant']:
                        axes[actu][2].text(x, pos_z_offsets[isubt], '*', ha='center', fontsize=10)
                        if (actu == 'torques') and diff_stats_pos_z.loc[(time, perturbation)]['significant']:
                            axes[actu][2].text(x, diff_pos_z_offsets[isubt], r'$ \diamond $', ha='center', fontsize=8)
                        
                    # Instantaneous velocities
                    # ------------------------
                    plot_errorbar(axes[actu][3], x, vel_x_diff_mean[actu][itime, isubt], vel_x_diff_std[actu][itime, isubt])
                    h_vel = axes[actu][3].bar(x, vel_x_diff_mean[actu][itime, isubt], self.width, color=color, clip_on=False, 
                        hatch=hatch, edgecolor=self.edgecolor, lw=lw, zorder=2.5)
                    if stats_vel_x.loc[(time, perturbation)]['significant']:
                        axes[actu][3].text(x, vel_x_offsets[isubt], '*', ha='center', fontsize=10)
                        if (actu == 'torques') and diff_stats_vel_x.loc[(time, perturbation)]['significant']:
                            axes[actu][3].text(x, diff_vel_x_offsets[isubt], r'$ \diamond $', ha='center', fontsize=8)
                    handles_vel.append(h_vel)

                    plot_errorbar(axes[actu][4], x, vel_y_diff_mean[actu][itime, isubt], vel_y_diff_std[actu][itime, isubt])
                    axes[actu][4].bar(x, vel_y_diff_mean[actu][itime, isubt], self.width, color=color, clip_on=False, 
                        hatch=hatch, edgecolor=self.edgecolor, lw=lw, zorder=2.5)
                    if stats_vel_y.loc[(time, perturbation)]['significant']:
                        axes[actu][4].text(x, vel_y_offsets[isubt], '*', ha='center', fontsize=10)
                        if (actu == 'torques') and diff_stats_vel_y.loc[(time, perturbation)]['significant']:
                            axes[actu][4].text(x, diff_vel_y_offsets[isubt], r'$ \diamond $', ha='center', fontsize=8)

                    plot_errorbar(axes[actu][5], x, vel_z_diff_mean[actu][itime, isubt], vel_z_diff_std[actu][itime, isubt])
                    axes[actu][5].bar(x, vel_z_diff_mean[actu][itime, isubt], self.width, color=color, clip_on=False, 
                        hatch=hatch, edgecolor=self.edgecolor, lw=lw, zorder=2.5)
                    if stats_vel_z.loc[(time, perturbation)]['significant']:
                        axes[actu][5].text(x, vel_z_offsets[isubt], '*', ha='center', fontsize=10) 
                        if (actu == 'torques') and diff_stats_vel_z.loc[(time, perturbation)]['significant']:
                            axes[actu][5].text(x, diff_vel_z_offsets[isubt], r'$ \diamond $', ha='center', fontsize=8)    

                    # Instantaneous accelerations
                    # ---------------------------
                    plot_errorbar(axes[actu][6], x, acc_x_diff_mean[actu][itime, isubt], acc_x_diff_std[actu][itime, isubt])
                    h_acc = axes[actu][6].bar(x, acc_x_diff_mean[actu][itime, isubt], self.width, color=color, clip_on=False, 
                        hatch=hatch, edgecolor=self.edgecolor, lw=lw, zorder=2.5)
                    if stats_acc_x.loc[(time, perturbation)]['significant']:
                        axes[actu][6].text(x, acc_x_offsets[isubt], '*', ha='center', fontsize=10)
                        if (actu == 'torques') and diff_stats_acc_x.loc[(time, perturbation)]['significant']:
                            axes[actu][6].text(x, diff_acc_x_offsets[isubt], r'$ \diamond $', ha='center', fontsize=8)
                    handles_acc.append(h_acc)
                    
                    plot_errorbar(axes[actu][7], x, acc_y_diff_mean[actu][itime, isubt], acc_y_diff_std[actu][itime, isubt])
                    axes[actu][7].bar(x, acc_y_diff_mean[actu][itime, isubt], self.width, color=color, clip_on=False,
                        hatch=hatch, edgecolor=self.edgecolor, lw=lw, zorder=2.5)
                    if stats_acc_y.loc[(time, perturbation)]['significant']:
                        axes[actu][7].text(x, acc_y_offsets[isubt], '*', ha='center', fontsize=10)
                        if (actu == 'torques') and diff_stats_acc_y.loc[(time, perturbation)]['significant']:
                            axes[actu][7].text(x, diff_acc_y_offsets[isubt], r'$ \diamond $', ha='center', fontsize=8)
                    
                    plot_errorbar(axes[actu][8], x, acc_z_diff_mean[actu][itime, isubt], acc_z_diff_std[actu][itime, isubt])
                    axes[actu][8].bar(x, acc_z_diff_mean[actu][itime, isubt], self.width, color=color, clip_on=False, 
                        hatch=hatch, edgecolor=self.edgecolor, lw=lw, zorder=2.5)
                    if stats_acc_z.loc[(time, perturbation)]['significant']:
                        axes[actu][8].text(x, acc_z_offsets[isubt], '*', ha='center', fontsize=10)   
                        if (actu == 'torques') and diff_stats_acc_z.loc[(time, perturbation)]['significant']:
                            axes[actu][8].text(x, diff_acc_z_offsets[isubt], r'$ \diamond $', ha='center', fontsize=8)   

            axes[actu][0].set_ylim(pos_x_lim)
            axes[actu][0].set_yticks(get_ticks_from_lims(pos_x_lim, pos_x_step))
            axes[actu][1].set_ylim(pos_y_lim)
            axes[actu][1].set_yticks(get_ticks_from_lims(pos_y_lim, pos_y_step))
            axes[actu][2].set_ylim(pos_z_lim)
            axes[actu][2].set_yticks(get_ticks_from_lims(pos_z_lim, pos_z_step))
            axes[actu][3].set_ylim(vel_x_lim)
            axes[actu][3].set_yticks(get_ticks_from_lims(vel_x_lim, vel_x_step))
            axes[actu][4].set_ylim(vel_y_lim)
            axes[actu][4].set_yticks(get_ticks_from_lims(vel_y_lim, vel_y_step))
            axes[actu][5].set_ylim(vel_z_lim)
            axes[actu][5].set_yticks(get_ticks_from_lims(vel_z_lim, vel_z_step))
            axes[actu][6].set_ylim(acc_x_lim)
            axes[actu][6].set_yticks(get_ticks_from_lims(acc_x_lim, acc_x_step))
            axes[actu][7].set_ylim(acc_y_lim)
            axes[actu][7].set_yticks(get_ticks_from_lims(acc_y_lim, acc_y_step))
            axes[actu][8].set_ylim(acc_z_lim)
            axes[actu][8].set_yticks(get_ticks_from_lims(acc_z_lim, acc_z_step))   

            if 'muscles' in actu:
                axes[actu][0].legend(handles_pos, self.legend_labels, loc='upper left', 
                    frameon=True, fontsize=8)   
                axes[actu][3].legend(handles_vel, self.legend_labels, loc='upper left', 
                    frameon=True, fontsize=8)
                axes[actu][6].legend(handles_acc, self.legend_labels, loc='upper left', 
                    frameon=True, fontsize=8)
                axes[actu][0].set_ylabel(r'$\Delta$' + ' fore-aft COM position $[-]$')
                axes[actu][1].set_ylabel(r'$\Delta$' + ' vertical COM position $[-]$')
                axes[actu][2].set_ylabel(r'$\Delta$' + ' medio-lateral COM position $[-]$')
                axes[actu][3].set_ylabel(r'$\Delta$' + ' fore-aft COM velocity $[-]$')
                axes[actu][4].set_ylabel(r'$\Delta$' + ' vertical COM velocity $[-]$')
                axes[actu][5].set_ylabel(r'$\Delta$' + ' medio-lateral COM velocity $[-]$')
                axes[actu][6].set_ylabel(r'$\Delta$' + ' fore-aft COM acceleration $[-]$')
                axes[actu][7].set_ylabel(r'$\Delta$' + ' vertical COM acceleration $[-]$')
                axes[actu][8].set_ylabel(r'$\Delta$' + ' medio-lateral COM acceleration $[-]$')


        import cv2
        def add_muscles_image(fig):
            side = 0.175
            l = 0.23
            b = 0.81
            w = side
            h = side
            ax = fig.add_axes([l, b, w, h], projection=None, polar=False)
            image = cv2.imread(
                os.path.join(self.study.config['figures_path'], 'images',
                    'sagittal_muscles.tiff'))[..., ::-1]
            ax.imshow(image, interpolation='none', aspect='equal')

            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yticklabels([])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.tick_params(axis='x', which='both', bottom=False, 
                           top=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', bottom=False, 
                           top=False, labelbottom=False)

        def add_torques_image(fig):
            side = 0.175
            l = 0.680
            b = 0.81
            w = side
            h = side
            ax = fig.add_axes([l, b, w, h], projection=None, polar=False)
            image = cv2.imread(
                os.path.join(self.study.config['figures_path'], 'images',
                    'sagittal_torques.tiff'))[..., ::-1]
            ax.imshow(image, interpolation='none', aspect='equal')

            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yticklabels([])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.tick_params(axis='x', which='both', bottom=False, 
                           top=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', bottom=False, 
                           top=False, labelbottom=False)

        for ifig, fig in enumerate(figs):
            fig.subplots_adjust(left=0.125, right=0.95, bottom=0.075, top=0.8, wspace=0.2)
            add_muscles_image(fig)
            add_torques_image(fig)
            fig.savefig(target[ifig], dpi=600)
            plt.close()

        for ifig, fig in enumerate(figs):
            fig.savefig(target[ifig + len(figs)], dpi=600)
            plt.close()

        com_heights = list()
        for subject in self.subjects:
            com_heights.append(com_height_dict[subject])

        with open(target[6], 'w') as f:
            f.write('Center-of-mass height, mean +/- std across subjects\n')
            f.write('\n')
            f.write(f'COM height: {np.mean(com_heights):.2f} +/- {np.std(com_heights):.2f} [m]\n')
            f.write('\n')


class TaskComputeCenterOfMassTimesteppingError(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, times, rise, fall):
        super(TaskComputeCenterOfMassTimesteppingError, self).__init__(study)
        self.name = f'compute_center_of_mass_timestepping_error_rise{rise}_fall{fall}'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.validate_path = os.path.join(study.config['validate_path'],
            'center_of_mass_error',  f'rise{rise}_fall{fall}')
        if not os.path.exists(self.validate_path): 
            os.makedirs(self.validate_path)
        self.subjects = subjects
        self.times = times
        self.rise = rise
        self.fall = fall
        self.gravity = 9.81

        deps = list()
        self.label_dict = dict()
        ilabel = 0
        for isubj, subject in enumerate(subjects):

            # Unperturbed solutions
            # ---------------------
            deps.append(
                os.path.join(
                    self.study.config['results_path'], 
                    'unperturbed', subject,
                    'center_of_mass_unperturbed.sto'))

            self.label_dict[f'{subject}_unperturbed'] = ilabel
            ilabel += 1

            for actu in [False, True]:
                torque_act = '_torque_actuators' if actu else ''
                subpath = 'torque_actuators' if actu else 'perturbed'

                for time in self.times:

                    # Unperturbed time-stepping solutions
                    # -----------------------------------
                    label = (f'perturbed_torque0_time{time}'
                             f'_rise{self.rise}_fall{self.fall}{torque_act}')
                    deps.append(
                        os.path.join(
                            self.study.config['results_path'], 
                            subpath, label, subject,
                            f'center_of_mass_{label}.sto'))

                    self.label_dict[f'{subject}_unperturbed_time{time}{torque_act}'] = ilabel
                    ilabel += 1

        self.add_action(deps, 
                        [os.path.join(self.validate_path, 
                         f'com_timestepping_error.txt')], 
                        self.compute_com_error)

    def compute_com_error(self, file_dep, target):

        # Aggregate data
        # --------------
        import collections
        com_dict = collections.defaultdict(dict)
        time_dict = collections.defaultdict(dict)

        for isubj, subject in enumerate(self.subjects):

            # Unperturbed center-of-mass trajectory
            # -------------------------------------
            unperturb_index = self.label_dict[f'{subject}_unperturbed']
            table = osim.TimeSeriesTable(file_dep[unperturb_index])
            timeVec = np.array(table.getIndependentColumn())
            duration = timeVec[-1] - timeVec[0]

            table_np = np.zeros((len(timeVec), 9))
            pos_x = table.getDependentColumn(
                '/|com_position_x').to_numpy()
            pos_y = table.getDependentColumn(
                '/|com_position_y').to_numpy()
            pos_z = table.getDependentColumn(
                '/|com_position_z').to_numpy()
            l_max = np.mean(pos_y)
            v_max = np.sqrt(self.gravity * l_max)

            table_np[:, 0] = (pos_x - pos_x[0]) / l_max
            table_np[:, 1] = (pos_y - pos_y[0]) / l_max
            table_np[:, 2] = (pos_z - pos_z[0]) / l_max
            table_np[:, 3] = table.getDependentColumn(
                '/|com_velocity_x').to_numpy() / v_max
            table_np[:, 4] = table.getDependentColumn(
                '/|com_velocity_y').to_numpy() / v_max
            table_np[:, 5] = table.getDependentColumn(
                '/|com_velocity_z').to_numpy() / v_max
            table_np[:, 6] = table.getDependentColumn(
                '/|com_acceleration_x').to_numpy() / self.gravity
            table_np[:, 7] = table.getDependentColumn(
                '/|com_acceleration_y').to_numpy() / self.gravity
            table_np[:, 8] = table.getDependentColumn(
                '/|com_acceleration_z').to_numpy() / self.gravity
        
            for actu in [False, True]:
                torque_act = '_torque_actuators' if actu else ''

                for time in self.times:

                    # Unperturbed center-of-mass trajectory
                    # -------------------------------------
                    label = f'{subject}_unperturbed_time{time}{torque_act}'
                    unperturb_index = self.label_dict[label]
                    tableTS = osim.TimeSeriesTable(file_dep[unperturb_index])
                    timeVecTS = np.array(tableTS.getIndependentColumn())
                    tableTS_np = np.zeros((len(timeVecTS), 9))
                    posTS_x = tableTS.getDependentColumn(
                        '/|com_position_x').to_numpy()
                    posTS_y = tableTS.getDependentColumn(
                        '/|com_position_y').to_numpy()
                    posTS_z = tableTS.getDependentColumn(
                        '/|com_position_z').to_numpy()
                    tableTS_np[:, 0] = (posTS_x - posTS_x[0]) / l_max 
                    tableTS_np[:, 1] = (posTS_y - posTS_y[0]) / l_max 
                    tableTS_np[:, 2] = (posTS_z - posTS_z[0]) / l_max 
                    tableTS_np[:, 3] = tableTS.getDependentColumn(
                        '/|com_velocity_x').to_numpy() / v_max
                    tableTS_np[:, 4] = tableTS.getDependentColumn(
                        '/|com_velocity_y').to_numpy() / v_max
                    tableTS_np[:, 5] = tableTS.getDependentColumn(
                        '/|com_velocity_z').to_numpy() / v_max
                    tableTS_np[:, 6] = tableTS.getDependentColumn(
                        '/|com_acceleration_x').to_numpy() / self.gravity
                    tableTS_np[:, 7] = tableTS.getDependentColumn(
                        '/|com_acceleration_y').to_numpy() / self.gravity
                    tableTS_np[:, 8] = tableTS.getDependentColumn(
                        '/|com_acceleration_z').to_numpy() / self.gravity

                    time_at_rise = timeVec[0] + (duration * ((time - 10) / 100.0))
                    time_at_fall = timeVec[0] + (duration * ((time + 5) / 100.0))
                    index_riseTS = np.argmin(np.abs(timeVecTS - time_at_rise))
                    index_fallTS = -1 
                    index_rise = np.argmin(np.abs(timeVec - timeVecTS[index_riseTS]))
                    index_fall = np.argmin(np.abs(timeVec - timeVecTS[index_fallTS]))

                    com_dict[subject][label] = \
                        table_np[index_rise:index_fall] - \
                        tableTS_np[index_riseTS:-1]

                    time_dict[subject][label] = timeVecTS[index_riseTS:index_fallTS]

        # Compute errors
        # --------------
        def calc_rms_error(errors):
            N = len(errors)
            sq_errors = np.square(errors)
            sumsq_errors = np.sum(sq_errors)
            return np.sqrt(sumsq_errors / N) 
        
        def integrated_rms_error(vec, time):
            interval = time[-1] - time[0]
            N = len(time)

            sse = np.zeros(N)
            for i in range(N):
                sse[i] = np.sum(vec[i, :]**2)

            # Trapezoidal rule for uniform grid:
            #      dt / 2 (f_0 + 2f_1 + 2f_2 + 2f_3 + ... + 2f_{N-1} + f_N)
            isse = interval / 2.0 * (np.sum(sse) + np.sum(sse[1:-1]))

            return np.sqrt(isse / interval / 3.0)

        pos_error = np.zeros(len(self.subjects))
        vel_error = np.zeros(len(self.subjects))
        acc_error = np.zeros(len(self.subjects))
        pos_error_mean = dict()
        vel_error_mean = dict()
        acc_error_mean = dict()
        pos_error_std = dict()
        vel_error_std = dict()
        acc_error_std = dict()
        for actu in [False, True]:
            torque_act = '_torque_actuators' if actu else ''
            actu_key = 'torques' if actu else 'muscles'
            pos_error_mean[actu_key] = np.zeros(len(self.times))
            vel_error_mean[actu_key] = np.zeros(len(self.times))
            acc_error_mean[actu_key] = np.zeros(len(self.times))
            pos_error_std[actu_key] = np.zeros(len(self.times))
            vel_error_std[actu_key] = np.zeros(len(self.times))
            acc_error_std[actu_key] = np.zeros(len(self.times))

            for itime, time in enumerate(self.times):
                for isubj, subject in enumerate(self.subjects):
                    label = f'{subject}_unperturbed_time{time}{torque_act}'
                    com = com_dict[subject][label]
                    timeVec = time_dict[subject][label]

                    # pos_error[isubj] = integrated_rms_error(com[:, 0:3], timeVec)
                    # vel_error[isubj] = integrated_rms_error(com[:, 3:6], timeVec)
                    # acc_error[isubj] = integrated_rms_error(com[:, 6:9], timeVec)
                    pos_error[isubj] = calc_rms_error(com[:, 0:3])
                    vel_error[isubj] = calc_rms_error(com[:, 3:6])
                    acc_error[isubj] = calc_rms_error(com[:, 6:9])

            pos_error_mean[actu_key] = np.mean(pos_error)
            vel_error_mean[actu_key] = np.mean(vel_error)
            acc_error_mean[actu_key] = np.mean(acc_error)
            pos_error_std[actu_key] = np.std(pos_error)
            vel_error_std[actu_key] = np.std(vel_error)
            acc_error_std[actu_key] = np.std(acc_error)

        with open(target[0], 'w') as f:
            f.write('Center-of-mass timestepping error, mean +/- std across subjects\n')
            f.write('\n')
            for actu in [False, True]:
                actu_key = 'torques' if actu else 'muscles'
                if actu:
                    f.write('torque-driven\n')
                else:
                    f.write('muscle-driven\n')
                f.write('-------------\n')
                f.write(f'-- position: {pos_error_mean[actu_key]:.2E} +/- {pos_error_std[actu_key]:.2E} [-]\n')
                f.write(f'-- velocity: {vel_error_mean[actu_key]:.2E} +/- {vel_error_std[actu_key]:.2E} [-]\n')
                f.write(f'-- acceleration: {acc_error_mean[actu_key]:.2E} +/- {acc_error_std[actu_key]:.2E} [-]\n')
                f.write('\n')


class TaskPlotCOMVersusCOP(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, times, rise, fall):
        super(TaskPlotCOMVersusCOP, self).__init__(study)
        self.name = f'plot_com_versus_cop_rise{rise}_fall{fall}'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.aggregate_path = os.path.join(study.config['statistics_path'],
            'center_of_mass', 'aggregate')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'com_versus_cop',  f'rise{rise}_fall{fall}')
        self.figures_path = os.path.join(study.config['figures_path'])

        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)
        self.subjects = subjects
        self.times = times
        self.rise = rise
        self.fall = fall
        self.gravity = 9.81
        self.torques = [study.plot_torques[2]]
        self.subtalars = [study.plot_subtalars[2]]
        self.colors = [study.plot_colors[2]]
        self.legend_labels = ['plantarflexion']
        self.models = list()
        deps = list()
        self.com_label_dict = dict()
        self.cop_label_dict = dict()
        ilabel = 0
        for isubj, subject in enumerate(subjects):

            # Model
            # -----
            self.models.append(os.path.join(
                self.study.config['results_path'], 'unperturbed', 
                subject, 'model_unperturbed.osim'))

            # Unperturbed solutions
            # ---------------------
            deps.append(
                os.path.join(
                    self.study.config['results_path'], 
                    'unperturbed', subject,
                    'center_of_mass_unperturbed.sto'))
            self.com_label_dict[f'{subject}_unperturbed'] = ilabel
            ilabel += 1

            deps.append(
                os.path.join(
                    self.study.config['results_path'], 
                    'unperturbed', subject,
                    'unperturbed_grfs.sto'))
            self.cop_label_dict[f'{subject}_unperturbed'] = ilabel
            ilabel += 1

            for actu in [False, True]:
                torque_act = '_torque_actuators' if actu else ''
                subpath = 'torque_actuators' if actu else 'perturbed'

                for time in self.times:

                    # Unperturbed time-stepping solutions
                    # -----------------------------------
                    label = (f'perturbed_torque0_time{time}'
                             f'_rise{self.rise}_fall{self.fall}{torque_act}')
                    deps.append(
                        os.path.join(
                            self.study.config['results_path'], 
                            subpath, label, subject,
                            f'center_of_mass_{label}.sto'))
                    self.com_label_dict[f'{subject}_unperturbed_time{time}{torque_act}'] = ilabel
                    ilabel += 1

                    deps.append(
                        os.path.join(
                            self.study.config['results_path'], 
                            subpath, label, subject,
                            f'{label}_grfs.sto'))
                    self.cop_label_dict[f'{subject}_unperturbed_time{time}{torque_act}'] = ilabel
                    ilabel += 1

                    for torque, subtalar in zip(self.torques, self.subtalars):

                         # Perturbed solutions
                         # -------------------
                        label = (f'perturbed_torque{torque}_time{time}'
                                f'_rise{self.rise}_fall{self.fall}{subtalar}{torque_act}')
                        deps.append(
                            os.path.join(
                                self.study.config['results_path'], 
                                subpath, label, subject,
                                f'center_of_mass_{label}.sto')
                            )
                        self.com_label_dict[f'{subject}_{label}'] = ilabel
                        ilabel += 1

                        deps.append(
                            os.path.join(
                                self.study.config['results_path'], 
                                subpath, label, subject,
                                f'{label}_grfs.sto')
                            )
                        self.cop_label_dict[f'{subject}_{label}'] = ilabel
                        ilabel += 1

        targets = list()
        targets += [os.path.join(self.analysis_path, 'com_versus_cop.png')]
        targets += [os.path.join(self.figures_path, 'figureS14', 'figureS14.png')]

        self.add_action(deps, targets, self.com_versus_cop)

    def com_versus_cop(self, file_dep, target):

        # Initialize figures
        # ------------------
        from collections import defaultdict
        axes = defaultdict(list)
        fig = plt.figure(figsize=(8, 4))
        for iactu, actu in enumerate(['muscles', 'torques']):
            ax = fig.add_subplot(1, 2, iactu + 1 )
            ax.spines['left'].set_position(('outward', 10))
            # ax.set_xticks(np.arange(len(self.times)))
            # ax.set_xlim(0, len(self.times)-1)
            ax.grid(color='gray', linestyle='--', linewidth=0.25,
                clip_on=False, alpha=0.75, zorder=0)
            ax.axhline(y=0, color='black', linestyle='-',
                    linewidth=0.5, alpha=1.0, zorder=2.5)
            util.publication_spines(ax)

            ax.spines['bottom'].set_position(('outward', 10))

            if actu == 'torques':
                ax.spines['left'].set_visible(False)
                ax.set_yticklabels([])
                ax.yaxis.set_ticks_position('none')
                ax.tick_params(axis='y', which='both', left=False, 
                               right=False, labelleft=False)

            axes[actu].append(ax)

        # Aggregate data
        # --------------
        import collections
        com_dict = collections.defaultdict(dict)
        cop_dict = collections.defaultdict(dict)
        time_dict = collections.defaultdict(dict)
        com_height_dict = dict()
        duration_dict = dict()
        for isubj, subject in enumerate(self.subjects):

            # Model
            # -----
            model = osim.Model(self.models[isubj])
            model.initSystem()

            # Unperturbed center-of-mass trajectory
            # -------------------------------------
            unperturb_index = self.com_label_dict[f'{subject}_unperturbed']
            tableTemp = osim.TimeSeriesTable(file_dep[unperturb_index])
            com_height_dict[subject] = np.mean(tableTemp.getDependentColumn(
                                               '/|com_position_y').to_numpy())
            timeTemp = np.array(tableTemp.getIndependentColumn())
            duration_dict[subject] = timeTemp[-1] - timeTemp[0]
            for actu in [False, True]:
                torque_act = '_torque_actuators' if actu else ''

                for time in self.times:

                    # Unperturbed center-of-mass trajectory
                    # -------------------------------------
                    unperturb_index = self.com_label_dict[f'{subject}_unperturbed_time{time}{torque_act}']
                    unpTableCOM = osim.TimeSeriesTable(file_dep[unperturb_index])
                    unpTimeVecCOM = unpTableCOM.getIndependentColumn()
                    unpTableCOM_np = np.zeros((len(unpTimeVecCOM), 9))
                    unpTableCOM_np[:, 0] = unpTableCOM.getDependentColumn(
                        '/|com_position_x').to_numpy()
                    unpTableCOM_np[:, 1] = unpTableCOM.getDependentColumn(
                        '/|com_position_y').to_numpy()
                    unpTableCOM_np[:, 2] = unpTableCOM.getDependentColumn(
                        '/|com_position_z').to_numpy()
                    unpTableCOM_np[:, 3] = unpTableCOM.getDependentColumn(
                        '/|com_velocity_x').to_numpy() 
                    unpTableCOM_np[:, 4] = unpTableCOM.getDependentColumn(
                        '/|com_velocity_y').to_numpy() 
                    unpTableCOM_np[:, 5] = unpTableCOM.getDependentColumn(
                        '/|com_velocity_z').to_numpy() 
                    unpTableCOM_np[:, 6] = unpTableCOM.getDependentColumn(
                        '/|com_acceleration_x').to_numpy()
                    unpTableCOM_np[:, 7] = unpTableCOM.getDependentColumn(
                        '/|com_acceleration_y').to_numpy()
                    unpTableCOM_np[:, 8] = unpTableCOM.getDependentColumn(
                        '/|com_acceleration_z').to_numpy()

                    # Unperturbed center-of-pressure trajectory
                    # -----------------------------------------
                    unperturb_index = self.cop_label_dict[f'{subject}_unperturbed_time{time}{torque_act}']
                    unpTableCOP = osim.TimeSeriesTable(file_dep[unperturb_index])
                    unpTimeVecCOP = unpTableCOP.getIndependentColumn()
                    unpTableCOP_np = np.zeros((len(unpTimeVecCOP), 3))
                    unpTableCOP_np[:, 0] = unpTableCOP.getDependentColumn(
                        'ground_force_r_px').to_numpy() 
                    unpTableCOP_np[:, 1] = unpTableCOP.getDependentColumn(
                        'ground_force_r_py').to_numpy()  
                    unpTableCOP_np[:, 2] = unpTableCOP.getDependentColumn(
                        'ground_force_r_pz').to_numpy() 

                    for torque, subtalar in zip(self.torques, self.subtalars):

                        # Perturbed center-of-mass trajectory
                        # -----------------------------------
                        label = (f'{subject}_perturbed_torque{torque}_time{time}'
                                 f'_rise{self.rise}_fall{self.fall}{subtalar}{torque_act}')
                        perturb_index = self.com_label_dict[label]
                        tableCOM = osim.TimeSeriesTable(file_dep[perturb_index])
                        timeVecCOM = tableCOM.getIndependentColumn()
                        tableCOM_np = np.zeros((len(timeVecCOM), 9))
                        tableCOM_np[:, 0] = tableCOM.getDependentColumn(
                            '/|com_position_x').to_numpy()
                        tableCOM_np[:, 1] = tableCOM.getDependentColumn(
                            '/|com_position_y').to_numpy()
                        tableCOM_np[:, 2] = tableCOM.getDependentColumn(
                            '/|com_position_z').to_numpy()
                        tableCOM_np[:, 3] = tableCOM.getDependentColumn(
                            '/|com_velocity_x').to_numpy()
                        tableCOM_np[:, 4] = tableCOM.getDependentColumn(
                            '/|com_velocity_y').to_numpy()
                        tableCOM_np[:, 5] = tableCOM.getDependentColumn(
                            '/|com_velocity_z').to_numpy() 
                        tableCOM_np[:, 6] = tableCOM.getDependentColumn(
                            '/|com_acceleration_x').to_numpy()
                        tableCOM_np[:, 7] = tableCOM.getDependentColumn(
                            '/|com_acceleration_y').to_numpy()
                        tableCOM_np[:, 8] = tableCOM.getDependentColumn(
                            '/|com_acceleration_z').to_numpy()

                        # Perturbed center-of-pressure trajectory
                        # ---------------------------------------
                        label = (f'{subject}_perturbed_torque{torque}_time{time}'
                                 f'_rise{self.rise}_fall{self.fall}{subtalar}{torque_act}')
                        perturb_index = self.cop_label_dict[label]
                        tableCOP = osim.TimeSeriesTable(file_dep[perturb_index])
                        timeVecCOP = tableCOP.getIndependentColumn()
                        tableCOP_np = np.zeros((len(timeVecCOP), 3))
                        tableCOP_np[:, 0] = tableCOP.getDependentColumn(
                            'ground_force_r_px').to_numpy() 
                        tableCOP_np[:, 1] = tableCOP.getDependentColumn(
                            'ground_force_r_py').to_numpy()  
                        tableCOP_np[:, 2] = tableCOP.getDependentColumn(
                            'ground_force_r_pz').to_numpy()  

                        # Compute difference between perturbed and unperturbed
                        # trajectories for this subject. We don't need to interpolate
                        # here since the perturbed and unperturbed trajectories contain
                        # the same time points (up until the end of the perturbation).
                        com_dict[subject][label] = tableCOM_np
                        com_dict[subject][label][:, 3:9] -= unpTableCOM_np[:, 3:9]
                        cop_dict[subject][label] = tableCOP_np
                        time_dict[subject][label] = timeVecCOM

        # Plotting
        # --------
        pos_vs_cop_x = np.zeros(len(self.subjects))
        vel_x_diff = np.zeros(len(self.subjects))
        pos_vs_cop_x_mean = dict()
        vel_x_diff_mean = dict()
        vel_x_diff_std = dict()
        for actu in [False, True]:
            torque_act = '_torque_actuators' if actu else ''
            actu_key = 'torques' if actu else 'muscles'
            pos_vs_cop_x_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            vel_x_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            vel_x_diff_std[actu_key] = np.zeros((len(self.times), len(self.subtalars)))

            for itime, time in enumerate(self.times):
                zipped = zip(self.torques, self.subtalars, self.colors)
                for isubt, (torque, subtalar, color) in enumerate(zipped):
                    for isubj, subject in enumerate(self.subjects):
                        label = (f'{subject}_perturbed_torque{torque}_time{time}'
                             f'_rise{self.rise}_fall{self.fall}{subtalar}{torque_act}')
                        com = com_dict[subject][label]
                        cop = cop_dict[subject][label]
 
                        timeVec = time_dict[subject][label]
                        duration = duration_dict[subject]
                        time_at_peak = timeVec[0] + (duration * (time / 100.0))
                        index_peak = np.argmin(np.abs(timeVec - time_at_peak))
                        index_fall = -1

                        hCOM = com_height_dict[subject]
                        vFr = np.sqrt(self.gravity * hCOM)
                        pos_vs_cop_x[isubj] = (com[index_fall, 0] - cop[index_peak, 0]) / hCOM
                        vel_x_diff[isubj] = com[index_fall, 3] / vFr
                      
                    pos_vs_cop_x_mean[actu_key][itime, isubt] = np.mean(pos_vs_cop_x)
                    vel_x_diff_mean[actu_key][itime, isubt] = np.mean(vel_x_diff)
                    vel_x_diff_std[actu_key][itime, isubt] = np.std(vel_x_diff)

        pos_vs_cop_x_step = 0.05
        vel_x_step = 0.005
        pos_vs_cop_x_lim = [0.0, 0.0]
        vel_x_lim = [0.0, 0.0]
        for actu in ['muscles', 'torques']:
            update_lims(pos_vs_cop_x_mean[actu], pos_vs_cop_x_step, pos_vs_cop_x_lim)
            update_lims(vel_x_diff_mean[actu]-vel_x_diff_std[actu], vel_x_step, vel_x_lim)
            update_lims(vel_x_diff_mean[actu]+vel_x_diff_std[actu], vel_x_step, vel_x_lim)       
      
        for actu in ['muscles', 'torques']:
            for itime, time in enumerate(self.times): 

                zipped = zip(self.torques, self.subtalars, self.colors)
                for isubt, (torque, subtalar, color) in enumerate(zipped):
                    perturbation = f'perturbed_torque{torque}{subtalar}'
       
                    # Instantaneous velocities
                    # ------------------------
                    ple, cle, ble = axes[actu][0].errorbar(
                        pos_vs_cop_x_mean[actu][itime, isubt], 
                        vel_x_diff_mean[actu][itime, isubt], 
                        yerr=vel_x_diff_std[actu][itime, isubt],
                        fmt='none', ecolor='black', lw=0.25,
                        elinewidth=0.4, markeredgewidth=0.4)
                    for cl in cle:
                        cl.set_marker('_')
                        cl.set_markersize(4)

                    axes[actu][0].plot(pos_vs_cop_x_mean[actu][itime, isubt], 
                        vel_x_diff_mean[actu][itime, isubt], marker='o',
                        markersize=3,
                        color=color, clip_on=False, zorder=2.5)

            axes[actu][0].set_xlim(pos_vs_cop_x_lim)
            axes[actu][0].set_xticks(get_ticks_from_lims(pos_vs_cop_x_lim, pos_vs_cop_x_step))
            axes[actu][0].set_ylim(vel_x_lim)
            axes[actu][0].set_yticks(get_ticks_from_lims(vel_x_lim, vel_x_step))
            axes[actu][0].set_xlabel('fore-aft COM position\nrelative to fore-aft COP position' + r' $[-]$')

            if 'muscles' in actu:
                handles = list()
                for color, label in zip(self.colors, self.legend_labels):
                    handles.append(patches.Patch(color=color, label=label))

                axes[actu][0].legend(handles=handles, loc='upper left', 
                    frameon=True, fontsize=6)
                axes[actu][0].set_ylabel(r'$\Delta$ fore-aft COM velocity $[-]$')

        import cv2
        def add_muscles_image(fig):
            side = 0.32
            l = 0.135
            b = 0.67
            w = side
            h = side
            ax = fig.add_axes([l, b, w, h], projection=None, polar=False)
            image = cv2.imread(
                os.path.join(self.study.config['figures_path'], 'images',
                    'sagittal_muscles.tiff'))[..., ::-1]
            ax.imshow(image, interpolation='none', aspect='equal')

            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yticklabels([])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.tick_params(axis='x', which='both', bottom=False, 
                           top=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', bottom=False, 
                           top=False, labelbottom=False)

        def add_torques_image(fig):
            side = 0.32
            l = 0.60
            b = 0.67
            w = side
            h = side
            ax = fig.add_axes([l, b, w, h], projection=None, polar=False)
            image = cv2.imread(
                os.path.join(self.study.config['figures_path'], 'images',
                    'sagittal_torques.tiff'))[..., ::-1]
            ax.imshow(image, interpolation='none', aspect='equal')

            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yticklabels([])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.tick_params(axis='x', which='both', bottom=False, 
                           top=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', bottom=False, 
                           top=False, labelbottom=False)

        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.175, top=0.65, 
            wspace=0.2, hspace=0.1)
        add_muscles_image(fig)
        add_torques_image(fig)

        fig.savefig(target[0], dpi=600)
        fig.savefig(target[1], dpi=600)
        plt.close()

# Center-of-pressure
# ------------------

class TaskPlotCenterOfPressureVector(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, times, rise, fall):
        super(TaskPlotCenterOfPressureVector, self).__init__(study)
        self.name = f'plot_center_of_pressure_vector_rise{rise}_fall{fall}'
        self.figures_path = os.path.join(study.config['figures_path']) 
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'center_of_pressure_vector',  f'rise{rise}_fall{fall}')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)
        self.walking_speed = study.walking_speed
        self.gravity = 9.81
        self.subjects = subjects
        self.times = times
        self.rise = rise
        self.fall = fall
        self.torques = study.plot_torques
        self.subtalars = study.plot_subtalars
        self.colors = study.plot_colors

        blank = ''
        self.legend_labels = [f'{blank}\neversion\n{blank}', 
                              'plantarflexion\n+\neversion', 
                              f'{blank}\nplantarflexion\n{blank}', 
                              'plantarflexion\n+\ninversion', 
                              f'{blank}\ninversion\n{blank}']

        deps = list()
        self.label_dict = dict()
        self.models = list()
        ilabel = 0
        for isubj, subject in enumerate(subjects):

            # Model
            # -----
            self.models.append(os.path.join(
                self.study.config['results_path'], 'unperturbed', 
                subject, 'model_unperturbed.osim'))

            # Unperturbed solutions
            # ---------------------
            deps.append(
                os.path.join(
                    self.study.config['results_path'], 
                    'unperturbed', subject,
                    'unperturbed_grfs.sto'))

            self.label_dict[f'{subject}_unperturbed'] = ilabel
            ilabel += 1

            for actu in [False, True]:
                torque_act = '_torque_actuators' if actu else ''
                subpath = 'torque_actuators' if actu else 'perturbed'

                for time in self.times:

                    # Unperturbed time-stepping solutions
                    # -----------------------------------
                    label = (f'perturbed_torque0_time{time}'
                             f'_rise{self.rise}_fall{self.fall}{torque_act}')
                    deps.append(
                        os.path.join(
                            self.study.config['results_path'], 
                            subpath, label, subject,
                            f'{label}_grfs.sto'))

                    self.label_dict[f'{subject}_unperturbed_time{time}{torque_act}'] = ilabel
                    ilabel += 1

                    for torque, subtalar in zip(self.torques, self.subtalars):

                         # Perturbed solutions
                         # -------------------
                        label = (f'perturbed_torque{torque}_time{time}'
                                f'_rise{self.rise}_fall{self.fall}{subtalar}{torque_act}')
                        deps.append(
                            os.path.join(
                                self.study.config['results_path'], 
                                subpath, label, subject,
                                f'{label}_grfs.sto')
                            )

                        self.label_dict[f'{subject}_{label}'] = ilabel
                        ilabel += 1

        self.add_action(deps, 
                [os.path.join(self.analysis_path, 
                    'cop_vector.png'),
                 os.path.join(self.figures_path, 
                    'figure4', 'figure4.png')], 
            self.plot_cop_vectors)

    def plot_cop_vectors(self, file_dep, target):

        # Globals
        # -------
        tick_fs = 6

        # Initialize figure
        # -----------------
        axes = list()
        fig = plt.figure(figsize=(7, 7.5))
        for itorque, torque in enumerate(self.torques):
            ax = fig.add_subplot(1, len(self.torques), itorque + 1)
            ax.grid(axis='y', color='gray', alpha=0.5, linewidth=0.5, 
                    zorder=-10, clip_on=False)
            ax.axvline(x=0, color='gray', linestyle='-',
                    linewidth=0.5, alpha=0.5, zorder=-1)
            util.publication_spines(ax)
            ax.set_aspect('equal')

            if not itorque:
                ax.spines['left'].set_position(('outward', 15))
            else:
                ax.spines['left'].set_visible(False)
                ax.set_yticklabels([])
                ax.yaxis.set_ticks_position('none')
                ax.tick_params(axis='y', which='both', bottom=False, 
                               top=False, labelbottom=False)

            ax.spines['bottom'].set_position(('outward', 20))
            ax.set_title(self.legend_labels[itorque], pad=15, fontsize=8)
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(which='minor', axis='x', direction='in')
            axes.append(ax)

        # Aggregate data
        # --------------
        import collections
        cop_dict = collections.defaultdict(dict)
        time_dict = collections.defaultdict(dict)
        duration_dict = dict()
        for isubj, subject in enumerate(self.subjects):

            # Model
            # -----
            model = osim.Model(self.models[isubj])
            model.initSystem()

            # Unperturbed center-of-pressure
            # ------------------------------
            unperturb_index = self.label_dict[f'{subject}_unperturbed']
            tableTemp = osim.TimeSeriesTable(file_dep[unperturb_index])
            timeTemp = np.array(tableTemp.getIndependentColumn())
            duration_dict[subject] = timeTemp[-1] - timeTemp[0]
            for actu in [False, True]:
                torque_act = '_torque_actuators' if actu else ''

                for time in self.times:

                    # Unperturbed center-of-pressure
                    # ------------------------------
                    unperturb_index = self.label_dict[f'{subject}_unperturbed_time{time}{torque_act}']
                    unpTable = osim.TimeSeriesTable(file_dep[unperturb_index])
                    unpTimeVec = unpTable.getIndependentColumn()
                    unpTable_np = np.zeros((len(unpTimeVec), 3))
                    unpTable_np[:, 0] = unpTable.getDependentColumn(
                        'ground_force_r_px').to_numpy() 
                    unpTable_np[:, 1] = unpTable.getDependentColumn(
                        'ground_force_r_py').to_numpy()  
                    unpTable_np[:, 2] = unpTable.getDependentColumn(
                        'ground_force_r_pz').to_numpy()  

                    for torque, subtalar in zip(self.torques, self.subtalars):

                        # Perturbed center-of-mass trajectory
                        # -----------------------------------
                        label = (f'{subject}_perturbed_torque{torque}_time{time}'
                                 f'_rise{self.rise}_fall{self.fall}{subtalar}{torque_act}')
                        perturb_index = self.label_dict[label]
                        table = osim.TimeSeriesTable(file_dep[perturb_index])
                        timeVec = table.getIndependentColumn()
                        table_np = np.zeros((len(timeVec), 3))
                        table_np[:, 0] = table.getDependentColumn(
                            'ground_force_r_px').to_numpy() 
                        table_np[:, 1] = table.getDependentColumn(
                            'ground_force_r_py').to_numpy()  
                        table_np[:, 2] = table.getDependentColumn(
                            'ground_force_r_pz').to_numpy()  

                        # Compute difference between perturbed and unperturbed
                        # trajectories for this subject. We don't need to interpolate
                        # here since the perturbed and unperturbed trajectories contain
                        # the same time points (up until the end of the perturbation).
                        cop_dict[subject][label] = table_np - unpTable_np
                        time_dict[subject][label] = np.array(timeVec)

        # Plot helper functions
        # ---------------------
        def set_arrow_patch_transverse(ax, x, y, dx, dy, actu, color):
            if 'muscles' in actu:
                arrowstyle = patches.ArrowStyle.CurveFilledB(head_length=0.25, 
                    head_width=0.1)
                lw = 2.0
            elif 'torques' in actu: 
                arrowstyle = patches.ArrowStyle.CurveB()
                lw = 0.75

            arrow = patches.FancyArrowPatch((x, y), (x + dx, y + dy),
                    arrowstyle=arrowstyle, mutation_scale=10, shrinkA=0, shrinkB=0,
                    capstyle='round', joinstyle='miter', 
                    color=color, clip_on=False, zorder=2.5, lw=lw)
            ax.add_patch(arrow)
            return arrow

        # Compute changes in center-of-pressure
        # -------------------------------------
        cop_x_diff = np.zeros(len(self.subjects))
        cop_y_diff = np.zeros(len(self.subjects))
        cop_z_diff = np.zeros(len(self.subjects))
        cop_x_diff_mean = dict()
        cop_y_diff_mean = dict()
        cop_z_diff_mean = dict()
        for actu in [False, True]:
            torque_act = '_torque_actuators' if actu else ''
            actu_key = 'torques' if actu else 'muscles'
            cop_x_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            cop_y_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            cop_z_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))

            for itime, time in enumerate(self.times):
                zipped = zip(self.torques, self.subtalars, self.colors)
                for iperturb, (torque, subtalar, color) in enumerate(zipped):
                    for isubj, subject in enumerate(self.subjects):

                        label = (f'{subject}_perturbed_torque{torque}_time{time}'
                                 f'_rise{self.rise}_fall{self.fall}{subtalar}{torque_act}')
                        cop = cop_dict[subject][label]

                        # Compute the closet time index to the current peak 
                        # perturbation time. 
                        duration = duration_dict[subject]
                        timeVec = time_dict[subject][label]
                        time_at_peak = timeVec[0] + (duration * (time / 100.0))
                        index_peak = np.argmin(np.abs(timeVec - time_at_peak))

                        cop_x_diff[isubj] = 100.0*cop[index_peak, 0]
                        cop_y_diff[isubj] = 100.0*cop[index_peak, 1]
                        cop_z_diff[isubj] = 100.0*cop[index_peak, 2]

                    cop_x_diff_mean[actu_key][itime, iperturb] = np.mean(cop_x_diff)
                    cop_y_diff_mean[actu_key][itime, iperturb] = np.mean(cop_y_diff)
                    cop_z_diff_mean[actu_key][itime, iperturb] = np.mean(cop_z_diff)

        # Set plot limits and labels
        # --------------------------
        for iperturb in np.arange(len(self.subtalars)):
            # Transverse position
            xz_cop_scale = 1.5
            axes[iperturb].set_xlim(-xz_cop_scale, xz_cop_scale)
            axes[iperturb].set_xticks([-xz_cop_scale, 0, xz_cop_scale])
            axes[iperturb].set_xticklabels([-xz_cop_scale, 0, xz_cop_scale], fontsize=tick_fs)
            axes[iperturb].set_yticks(xz_cop_scale*np.arange(len(self.times)))
            axes[iperturb].set_ylim(0, xz_cop_scale*(len(self.times)-1))
            if iperturb:
                axes[iperturb].set_yticklabels([])
            else:
                axes[iperturb].set_yticklabels([f'{time}' for time in self.times],
                                        fontsize=tick_fs+1)
            axes[2].set_xlabel(r'$\Delta$' + ' center of pressure position $[cm]$',
                fontsize=tick_fs+3)
            axes[0].set_ylabel('exoskeleton torque peak time\n(% gait cycle)',
                fontsize=tick_fs+3)
      
        # Plot results
        # ------------
        for actu in [True, False]:
            torque_act = '_torque_actuators' if actu else ''
            actu_key = 'torques' if actu else 'muscles'

            for itime, time in enumerate(self.times):
                zipped = zip(self.torques, self.subtalars, self.colors)
                for iperturb, (torque, subtalar, color) in enumerate(zipped):

                    # Position vectors
                    # ----------------
                    cop_x_diff = cop_x_diff_mean[actu_key][itime, iperturb]
                    cop_y_diff = cop_y_diff_mean[actu_key][itime, iperturb]
                    cop_z_diff = cop_z_diff_mean[actu_key][itime, iperturb]
                    set_arrow_patch_transverse(axes[iperturb], 0, xz_cop_scale*itime, 
                        cop_z_diff, cop_x_diff, actu_key, color)

        left = 0.12
        right = 0.95
        bottom = 0.40
        top = 0.92
        wspace = 0.3
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace)

        import cv2
        def add_transverse_image(fig):
            side = 0.35
            offset = 0.0275
            l = ((1.0 - side) / 2.0) + offset
            b = -0.005
            w = side
            h = side
            ax = fig.add_axes([l, b, w, h], projection=None, polar=False)
            image = cv2.imread(
                os.path.join(self.study.config['figures_path'], 'images',
                    'foot_with_arrows.tiff'))[..., ::-1]
            ax.imshow(image, interpolation='none', aspect='equal')

            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yticklabels([])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.tick_params(axis='x', which='both', bottom=False, 
                           top=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', bottom=False, 
                           top=False, labelbottom=False)

        def add_legend(fig, x, y):
            w = 0.1
            h = 0.02
            ax = fig.add_axes([x, y, w, h], projection=None, polar=False)

            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yticklabels([])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.tick_params(axis='x', which='both', bottom=False, 
                           top=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', bottom=False, 
                           top=False, labelbottom=False)

            set_arrow_patch_transverse(ax, 0, 1.25, 0.5, 0, 'muscles', 'black')
            ax.text(0.6, 1.1, 'muscle-driven model')
            set_arrow_patch_transverse(ax, 0, 0, 0.5, 0, 'torques', 'black')
            ax.text(0.6, -0.15, 'torque-driven model')

        add_transverse_image(fig)
        x = 0.675
        y = 0.25
        add_legend(fig, x, y)

        fig.savefig(target[0], dpi=600)
        fig.savefig(target[1], dpi=600)
        plt.close()         


class TaskPlotInstantaneousCenterOfPressure(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, times, rise, fall):
        super(TaskPlotInstantaneousCenterOfPressure, self).__init__(study)
        self.name = f'plot_instantaneous_center_of_pressure_rise{rise}_fall{fall}'
        self.figures_path = os.path.join(study.config['figures_path']) 
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.aggregate_path = os.path.join(study.config['statistics_path'],
            'center_of_pressure', 'aggregate')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'center_of_pressure_instantaneous',  f'rise{rise}_fall{fall}')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)
        self.subjects = subjects
        self.times = times
        self.rise = rise
        self.fall = fall
        self.gravity = 9.81
        self.torques = study.plot_torques
        self.subtalars = study.plot_subtalars
        self.colors = study.plot_colors
        self.hatches = [None, None, None, None, None]
        self.edgecolor = 'black'
        self.width = 0.15
        N = len(self.subtalars)
        min_width = -self.width*((N-1)/2)
        max_width = -min_width
        self.shifts = np.linspace(min_width, max_width, N)
        self.legend_labels = ['eversion', 
                              'plantarflexion + eversion', 
                              'plantarflexion', 
                              'plantarflexion + inversion', 
                              'inversion']

        deps = list()
        self.label_dict = dict()
        self.models = list()
        ilabel = 0
        for isubj, subject in enumerate(subjects):

            # Model
            # -----
            self.models.append(os.path.join(
                self.study.config['results_path'], 'unperturbed', 
                subject, 'model_unperturbed.osim'))

            # Unperturbed solutions
            # ---------------------
            deps.append(
                os.path.join(
                    self.study.config['results_path'], 
                    'unperturbed', subject,
                    'unperturbed_grfs.sto'))

            self.label_dict[f'{subject}_unperturbed'] = ilabel
            ilabel += 1

            for actu in [False, True]:
                torque_act = '_torque_actuators' if actu else ''
                subpath = 'torque_actuators' if actu else 'perturbed'

                for time in self.times:

                    # Unperturbed time-stepping solutions
                    # -----------------------------------
                    label = (f'perturbed_torque0_time{time}'
                             f'_rise{self.rise}_fall{self.fall}{torque_act}')
                    deps.append(
                        os.path.join(
                            self.study.config['results_path'], 
                            subpath, label, subject,
                            f'{label}_grfs.sto'))

                    self.label_dict[f'{subject}_unperturbed_time{time}{torque_act}'] = ilabel
                    ilabel += 1

                    for torque, subtalar in zip(self.torques, self.subtalars):

                         # Perturbed solutions
                         # -------------------
                        label = (f'perturbed_torque{torque}_time{time}'
                                f'_rise{self.rise}_fall{self.fall}{subtalar}{torque_act}')
                        deps.append(
                            os.path.join(
                                self.study.config['results_path'], 
                                subpath, label, subject,
                                f'{label}_grfs.sto')
                            )

                        self.label_dict[f'{subject}_{label}'] = ilabel
                        ilabel += 1

        # Statistics results
        for actu in ['muscles', 'torques', 'diff']:
            for kin in ['pos']:
                for direc in ['x', 'z']:
                    label = f'cop_stats_{kin}_{direc}_{actu}'
                    deps.append(os.path.join(self.aggregate_path, f'{label}.csv'))

                    self.label_dict[label] = ilabel
                    ilabel += 1


        targets = list()
        targets += [os.path.join(self.analysis_path, 'instant_cop_pos.png')]
        targets += [os.path.join(self.figures_path, 'figureS9', 'figureS9.png')]

        self.add_action(deps, targets, self.plot_instantaneous_cop)

    def plot_instantaneous_cop(self, file_dep, target):

        # Initialize figures
        # ------------------
        from collections import defaultdict
        figs = list()
        axes = defaultdict(list)
        for kin in ['pos']:
            fig = plt.figure(figsize=(9, 8))
            for iactu, actu in enumerate(['muscles', 'torques']):
                for idirec, direc in enumerate(['AP', 'ML']):
                    index = 2*idirec + iactu + 1 
                    ax = fig.add_subplot(2, 2, index)
                    ax.axhline(y=0, color='black', linestyle='-',
                            linewidth=0.1, alpha=1.0, zorder=2.5)
                    ax.spines['left'].set_position(('outward', 30))
                    ax.set_xticks(np.arange(len(self.times)))
                    ax.set_xlim(0, len(self.times)-1)
                    ax.grid(color='gray', linestyle='--', linewidth=0.4,
                        clip_on=False, alpha=0.75, zorder=0)
                    util.publication_spines(ax)

                    if not direc == 'ML':
                        ax.spines['bottom'].set_visible(False)
                        ax.set_xticklabels([])
                        ax.xaxis.set_ticks_position('none')
                        ax.tick_params(axis='x', which='both', bottom=False, 
                                       top=False, labelbottom=False)
                    else:
                        ax.spines['bottom'].set_position(('outward', 10))
                        ax.set_xticklabels([f'{time}' for time in self.times])
                        ax.set_xlabel('exoskeleton torque peak time\n(% gait cycle)')

                    if actu == 'torques':
                        ax.spines['left'].set_visible(False)
                        ax.set_yticklabels([])
                        ax.yaxis.set_ticks_position('none')
                        ax.tick_params(axis='y', which='both', left=False, 
                                       right=False, labelleft=False)


                    axes[actu].append(ax)

            figs.append(fig)

        # Aggregate data
        # --------------
        import collections
        cop_dict = collections.defaultdict(dict)
        time_dict = collections.defaultdict(dict)
        duration_dict = dict()
        for isubj, subject in enumerate(self.subjects):

            # Model
            # -----
            model = osim.Model(self.models[isubj])
            model.initSystem()

            # Unperturbed center-of-mass trajectory
            # -------------------------------------
            unperturb_index = self.label_dict[f'{subject}_unperturbed']
            tableTemp = osim.TimeSeriesTable(file_dep[unperturb_index])
            timeTemp = np.array(tableTemp.getIndependentColumn())
            duration_dict[subject] = timeTemp[-1] - timeTemp[0]
            for actu in [False, True]:
                torque_act = '_torque_actuators' if actu else ''

                for time in self.times:

                    # Unperturbed center-of-mass trajectory
                    # -------------------------------------
                    unperturb_index = self.label_dict[f'{subject}_unperturbed_time{time}{torque_act}']
                    unpTable = osim.TimeSeriesTable(file_dep[unperturb_index])
                    unpTimeVec = unpTable.getIndependentColumn()
                    unpTable_np = np.zeros((len(unpTimeVec), 3))
                    unpTable_np[:, 0] = unpTable.getDependentColumn(
                        'ground_force_r_px').to_numpy() 
                    unpTable_np[:, 1] = unpTable.getDependentColumn(
                        'ground_force_r_py').to_numpy()  
                    unpTable_np[:, 2] = unpTable.getDependentColumn(
                        'ground_force_r_pz').to_numpy()

                    for torque, subtalar in zip(self.torques, self.subtalars):

                        # Perturbed center-of-mass trajectory
                        # -----------------------------------
                        label = (f'{subject}_perturbed_torque{torque}_time{time}'
                                 f'_rise{self.rise}_fall{self.fall}{subtalar}{torque_act}')
                        perturb_index = self.label_dict[label]
                        table = osim.TimeSeriesTable(file_dep[perturb_index])
                        timeVec = table.getIndependentColumn()
                        table_np = np.zeros((len(timeVec), 3))
                        table_np[:, 0] = table.getDependentColumn(
                            'ground_force_r_px').to_numpy() 
                        table_np[:, 1] = table.getDependentColumn(
                            'ground_force_r_py').to_numpy()  
                        table_np[:, 2] = table.getDependentColumn(
                            'ground_force_r_pz').to_numpy()

                        # Compute difference between perturbed and unperturbed
                        # trajectories for this subject. We don't need to interpolate
                        # here since the perturbed and unperturbed trajectories contain
                        # the same time points (up until the end of the perturbation).
                        cop_dict[subject][label] = table_np - unpTable_np
                        time_dict[subject][label] = np.array(timeVec)

        # Plotting
        # --------
        cop_x_diff = np.zeros(len(self.subjects))
        cop_z_diff = np.zeros(len(self.subjects))
        cop_x_diff_mean = dict()
        cop_z_diff_mean = dict()
        cop_x_diff_std = dict()
        cop_z_diff_std = dict()
        for actu in [False, True]:
            torque_act = '_torque_actuators' if actu else ''
            actu_key = 'torques' if actu else 'muscles'
            cop_x_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            cop_z_diff_mean[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            cop_x_diff_std[actu_key] = np.zeros((len(self.times), len(self.subtalars)))
            cop_z_diff_std[actu_key] = np.zeros((len(self.times), len(self.subtalars)))

            for itime, time in enumerate(self.times):
                zipped = zip(self.torques, self.subtalars, self.colors, self.shifts)
                for isubt, (torque, subtalar, color, shift) in enumerate(zipped):
                    for isubj, subject in enumerate(self.subjects):
                        label = (f'{subject}_perturbed_torque{torque}_time{time}'
                             f'_rise{self.rise}_fall{self.fall}{subtalar}{torque_act}')
                        cop = cop_dict[subject][label]

                        # Compute the closet time index to the current peak 
                        # perturbation time. 
                        timeVec = time_dict[subject][label]
                        duration = duration_dict[subject]
                        time_at_peak = timeVec[0] + (duration * (time / 100.0))
                        index_peak = np.argmin(np.abs(timeVec - time_at_peak))

                        cop_x_diff[isubj] = 100.0 * cop[index_peak, 0]
                        cop_z_diff[isubj] = 100.0 * cop[index_peak, 2]

                    cop_x_diff_mean[actu_key][itime, isubt] = np.mean(cop_x_diff)
                    cop_z_diff_mean[actu_key][itime, isubt] = np.mean(cop_z_diff)
                    cop_x_diff_std[actu_key][itime, isubt] = np.std(cop_x_diff)
                    cop_z_diff_std[actu_key][itime, isubt] = np.std(cop_z_diff)


        def get_offsets(means, stds, lim, shift=0.0):
            min_value = 0
            max_value = 0

            if np.any(means < 0):
                min_value = np.min(means[means < 0] - stds[means < 0])

            if np.any(means > 0):
                max_value = np.max(means[means > 0] + stds[means > 0])

            lim_range = lim[1] - lim[0]

            offsets = np.zeros_like(means)

            if np.any(means < 0):
                offsets[means < 0] = min_value - (0.06 + shift)*lim_range
            if np.any(means > 0):
                offsets[means > 0] = max_value - (0.01 - 6*shift)*lim_range

            return offsets

        cop_x_step = 0.5
        cop_z_step = 0.5
        cop_x_lim = [0.0, 0.0]
        cop_z_lim = [0.0, 0.0]
        for actu in ['muscles', 'torques']:
            update_lims(cop_x_diff_mean[actu]-cop_x_diff_std[actu], cop_x_step, cop_x_lim, mirror=True)
            update_lims(cop_z_diff_mean[actu]-cop_z_diff_std[actu], cop_z_step, cop_z_lim)       
            update_lims(cop_x_diff_mean[actu]+cop_x_diff_std[actu], cop_x_step, cop_x_lim, mirror=True)
            update_lims(cop_z_diff_mean[actu]+cop_z_diff_std[actu], cop_z_step, cop_z_lim)

        diff_stats_cop_x = pd.read_csv(file_dep[self.label_dict['cop_stats_pos_x_diff']], index_col=[0, 1])
        diff_stats_cop_z = pd.read_csv(file_dep[self.label_dict['cop_stats_pos_z_diff']], index_col=[0, 1])    
        for actu in ['muscles', 'torques']:
            stats_cop_x = pd.read_csv(file_dep[self.label_dict[f'cop_stats_pos_x_{actu}']], index_col=[0, 1])
            stats_cop_z = pd.read_csv(file_dep[self.label_dict[f'cop_stats_pos_z_{actu}']], index_col=[0, 1])

            handles_cop = list()
            for itime, time in enumerate(self.times):
                cop_x_offsets = get_offsets(cop_x_diff_mean[actu][itime, :], cop_x_diff_std[actu][itime, :], cop_x_lim)
                cop_z_offsets = get_offsets(cop_z_diff_mean[actu][itime, :], cop_z_diff_std[actu][itime, :], cop_z_lim)
                diff_cop_x_offsets = get_offsets(cop_x_diff_mean[actu][itime, :], cop_x_diff_std[actu][itime, :], cop_x_lim, shift=0.01)
                diff_cop_z_offsets = get_offsets(cop_z_diff_mean[actu][itime, :], cop_z_diff_std[actu][itime, :], cop_z_lim, shift=0.01)

                zipped = zip(self.torques, self.subtalars, self.colors, self.shifts, self.hatches)
                for isubt, (torque, subtalar, color, shift, hatch) in enumerate(zipped):
                    perturbation = f'perturbed_torque{torque}{subtalar}'

                    # Set the x-position for these bar chart entries.
                    x = itime + shift
                    lw = 0.2

                    # Instantaneous positions
                    # -----------------------
                    plot_errorbar(axes[actu][0], x, cop_x_diff_mean[actu][itime, isubt], cop_x_diff_std[actu][itime, isubt])
                    h_cop = axes[actu][0].bar(x, cop_x_diff_mean[actu][itime, isubt], self.width, color=color, clip_on=False, 
                        hatch=hatch, edgecolor=self.edgecolor, lw=lw, zorder=2.5)
                    if stats_cop_x.loc[(time, perturbation)]['significant']:
                        axes[actu][0].text(x, cop_x_offsets[isubt], '*', ha='center', fontsize=10)
                        if (actu == 'torques') and diff_stats_cop_x.loc[(time, perturbation)]['significant']:
                            axes[actu][0].text(x, diff_cop_x_offsets[isubt], r'$ \diamond $', ha='center', fontsize=8)
                    handles_cop.append(h_cop)
                    
                    plot_errorbar(axes[actu][1], x, cop_z_diff_mean[actu][itime, isubt], cop_z_diff_std[actu][itime, isubt])
                    axes[actu][1].bar(x, cop_z_diff_mean[actu][itime, isubt], self.width, color=color, clip_on=False, 
                        hatch=hatch, edgecolor=self.edgecolor , lw=lw, zorder=2.5)
                    if stats_cop_z.loc[(time, perturbation)]['significant']:
                        axes[actu][1].text(x, cop_z_offsets[isubt], '*', ha='center', fontsize=10)
                        if (actu == 'torques') and diff_stats_cop_z.loc[(time, perturbation)]['significant']:
                            axes[actu][1].text(x, diff_cop_z_offsets[isubt], r'$ \diamond $', ha='center', fontsize=8)
                     
            axes[actu][0].set_ylim(cop_x_lim)
            axes[actu][0].set_yticks(get_ticks_from_lims(cop_x_lim, cop_x_step))
            axes[actu][1].set_ylim(cop_z_lim)
            axes[actu][1].set_yticks(get_ticks_from_lims(cop_z_lim, cop_z_step))

            if 'muscles' in actu:
                axes[actu][0].legend(handles_cop, self.legend_labels, loc='lower left', 
                    frameon=True, fontsize=8)   
                axes[actu][0].set_ylabel(r'$\Delta$' + ' fore-aft COP position $[cm]$')
                axes[actu][1].set_ylabel(r'$\Delta$' + ' medio-lateral COP position $[cm]$')

        import cv2
        def add_muscles_image(fig):
            side = 0.175
            l = 0.23
            b = 0.81
            w = side
            h = side
            ax = fig.add_axes([l, b, w, h], projection=None, polar=False)
            image = cv2.imread(
                os.path.join(self.study.config['figures_path'], 'images',
                    'sagittal_muscles.tiff'))[..., ::-1]
            ax.imshow(image, interpolation='none', aspect='equal')

            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yticklabels([])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.tick_params(axis='x', which='both', bottom=False, 
                           top=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', bottom=False, 
                           top=False, labelbottom=False)

        def add_torques_image(fig):
            side = 0.175
            l = 0.680
            b = 0.81
            w = side
            h = side
            ax = fig.add_axes([l, b, w, h], projection=None, polar=False)
            image = cv2.imread(
                os.path.join(self.study.config['figures_path'], 'images',
                    'sagittal_torques.tiff'))[..., ::-1]
            ax.imshow(image, interpolation='none', aspect='equal')

            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yticklabels([])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.tick_params(axis='x', which='both', bottom=False, 
                           top=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', bottom=False, 
                           top=False, labelbottom=False)

        for ifig, fig in enumerate(figs):
            fig.subplots_adjust(left=0.125, right=0.95, bottom=0.075, top=0.8, wspace=0.2)
            add_muscles_image(fig)
            add_torques_image(fig)
            fig.savefig(target[ifig], dpi=600)
            plt.close()

        for ifig, fig in enumerate(figs):
            fig.savefig(target[ifig + len(figs)], dpi=600)
            plt.close()


# Device torques and powers
# -------------------------

class TaskCreatePerturbationPowersTable(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, torque_actuators=False):
        super(TaskCreatePerturbationPowersTable, self).__init__(study)
        self.subjects = subjects
        self.times = study.times
        self.torques = study.plot_torques
        self.subtalars = study.plot_subtalars
        self.rise = study.rise
        self.fall = study.fall
        self.actu = '_torque_actuators' if torque_actuators else ''
        self.subdir = 'torque_actuators' if torque_actuators else 'perturbed'
        self.suffix = '_torques' if torque_actuators else '_muscles'
        self.name = f'create_perturbation_powers_table{self.suffix}'

        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'perturbation_powers')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)

        deps = list()
        self.unperturbed_map = dict()
        self.solution_map = dict()
        self.perturbation_map = dict()
        self.multiindex_tuples = list()

        index = 0
        for isubj, subject in enumerate(subjects):

            deps.append(
                os.path.join(
                    self.study.config['results_path'], 
                    'unperturbed', subject,
                    'center_of_mass_unperturbed.sto'))

            self.unperturbed_map[f'{subject}_unperturbed'] = index
            index += 1

            for time in self.times:
                for torque, subtalar in zip(self.torques, self.subtalars):
                    label = (f'torque{torque}_time{time}'
                             f'_rise{self.rise}_fall{self.fall}{subtalar}')

                    deps.append(
                        os.path.join(self.study.config['results_path'], 
                            self.subdir, f'perturbed_{label}{self.actu}', subject, 
                            f'perturbed_{label}{self.actu}.sto'))
                    self.solution_map[label] = index
                    index += 1

                    deps.append(
                        os.path.join(self.study.config['results_path'], 
                            self.subdir, f'perturbed_{label}{self.actu}', subject, 
                            'ankle_perturbation_curve.sto'))
                    self.perturbation_map[label] = index
                    index += 1

                    self.multiindex_tuples.append((
                            subject,
                            f'time{time}', 
                            f'torque{torque}{subtalar}'
                            ))

        self.add_action(deps, 
                        [os.path.join(self.analysis_path, 
                            f'perturbation_powers_mean{self.suffix}.csv'),
                        os.path.join(self.analysis_path, 
                            f'perturbation_powers_std{self.suffix}.csv')], 
                        self.create_perturbation_powers_table)

    def create_perturbation_powers_table(self, file_dep, target):

        from scipy.interpolate import interp1d
        def compute_power(torque, speed, torqueTime, speedTime, onsetTime, offsetTime):
            istart = np.argmin(np.abs(speedTime - onsetTime))
            iend = np.argmin(np.abs(speedTime - offsetTime))

            torque_interp = interp1d(torqueTime, torque)
            torque_sampled = torque_interp(speedTime[istart:iend])

            return np.multiply(speed[istart:iend], torque_sampled)

        from collections import OrderedDict
        peak_pos_power = OrderedDict()
        peak_neg_power = OrderedDict()
        avg_pos_power = OrderedDict()
        avg_neg_power = OrderedDict()
        for odict in [peak_pos_power, peak_neg_power, avg_pos_power, avg_neg_power]:
            odict['ankle'] = list()
            odict['subtalar'] = list()

        for subject in self.subjects:
            for time in self.times:
                onset_pgc = time - 10
                offset_pgc = time + 5
                for torque, subtalar in zip(self.torques, self.subtalars):
                    unperturbed = osim.TimeSeriesTable(
                        file_dep[self.unperturbed_map[f'{subject}_unperturbed']])
                    timeUnp = np.array(unperturbed.getIndependentColumn())
                    pgcSol = 100 * (timeUnp - timeUnp[0]) / (timeUnp[-1] - timeUnp[0])
                    istart = np.argmin(np.abs(pgcSol - onset_pgc))
                    iend = np.argmin(np.abs(pgcSol - offset_pgc))
                    onsetTime = timeUnp[istart]
                    offsetTime = timeUnp[iend]
        
                    label = (f'torque{torque}_time{time}'
                             f'_rise{self.rise}_fall{self.fall}{subtalar}')

                    solution = osim.TimeSeriesTable(
                        file_dep[self.solution_map[label]])
                    timeSol = np.array(solution.getIndependentColumn())

                    perturbation = osim.TimeSeriesTable(
                        file_dep[self.perturbation_map[label]])
                    timePerturb = np.array(perturbation.getIndependentColumn())

                    ankleSpeed = solution.getDependentColumn(
                        '/jointset/ankle_r/ankle_angle_r/speed').to_numpy()
                    ankleTorque = perturbation.getDependentColumn(
                        '/forceset/perturbation_ankle_angle_r').to_numpy()
                    anklePower = compute_power(ankleTorque, ankleSpeed, 
                                               timePerturb, timeSol,
                                               onsetTime, offsetTime)

                    if torque:
                        if np.any(anklePower > 0):
                            peak_pos_power['ankle'].append(np.max(anklePower[anklePower > 0]))
                            avg_pos_power['ankle'].append(np.mean(anklePower[anklePower > 0]))
                        else:
                            peak_pos_power['ankle'].append(0)
                            avg_pos_power['ankle'].append(0)

                        if np.any(anklePower < 0):
                            peak_neg_power['ankle'].append(np.min(anklePower[anklePower < 0]))
                            avg_neg_power['ankle'].append(np.mean(anklePower[anklePower < 0]))
                        else:
                            peak_neg_power['ankle'].append(0)
                            avg_neg_power['ankle'].append(0)

                    else:
                        peak_pos_power['ankle'].append(np.nan)
                        peak_neg_power['ankle'].append(np.nan)
                        avg_pos_power['ankle'].append(np.nan)
                        avg_neg_power['ankle'].append(np.nan)

                    if 'subtalar' in subtalar:
                        subtalarTorque = perturbation.getDependentColumn(
                            '/forceset/perturbation_subtalar_angle_r').to_numpy()
                        subtalarSpeed = solution.getDependentColumn(
                            '/jointset/subtalar_r/subtalar_angle_r/speed').to_numpy()
                        subtalarPower = compute_power(subtalarTorque, subtalarSpeed, 
                                                      timePerturb, timeSol,
                                                      onsetTime, offsetTime)

                        if np.any(subtalarPower > 0):
                            peak_pos_power['subtalar'].append(
                                np.max(subtalarPower[subtalarPower > 0]))
                            avg_pos_power['subtalar'].append(
                                np.mean(subtalarPower[subtalarPower > 0]))
                        else:
                            peak_pos_power['subtalar'].append(0)
                            avg_pos_power['subtalar'].append(0)
                       
                        if np.any(subtalarPower < 0):
                            peak_neg_power['subtalar'].append(
                                np.min(subtalarPower[subtalarPower < 0]))
                            avg_neg_power['subtalar'].append(
                                np.mean(subtalarPower[subtalarPower < 0]))
                        else:
                            peak_neg_power['subtalar'].append(0)
                            avg_neg_power['subtalar'].append(0)

                    else:
                        peak_pos_power['subtalar'].append(np.nan)
                        peak_neg_power['subtalar'].append(np.nan)
                        avg_pos_power['subtalar'].append(np.nan)
                        avg_neg_power['subtalar'].append(np.nan)

        index = pd.MultiIndex.from_tuples(self.multiindex_tuples,
                names=['subject', 'time', 'perturbation'])

        df_peak_pos_power = pd.DataFrame(peak_pos_power, index=index)
        df_peak_neg_power = pd.DataFrame(peak_neg_power, index=index)
        df_avg_pos_power = pd.DataFrame(avg_pos_power, index=index)
        df_avg_neg_power = pd.DataFrame(avg_neg_power, index=index)

        peak_pos_power_mean = df_peak_pos_power.groupby(level=['time', 'perturbation']).mean()
        peak_pos_power_std = df_peak_pos_power.groupby(level=['time', 'perturbation']).std()

        peak_neg_power_mean = df_peak_neg_power.groupby(level=['time', 'perturbation']).mean()
        peak_neg_power_std = df_peak_neg_power.groupby(level=['time', 'perturbation']).std()

        avg_pos_power_mean = df_avg_pos_power.groupby(level=['time', 'perturbation']).mean()
        avg_pos_power_std = df_avg_pos_power.groupby(level=['time', 'perturbation']).std()

        avg_neg_power_mean = df_avg_neg_power.groupby(level=['time', 'perturbation']).mean()
        avg_neg_power_std = df_avg_neg_power.groupby(level=['time', 'perturbation']).std()

        column_tuples = [('peak positive', 'ankle'), 
                         ('peak positive', 'subtalar'),
                         ('peak negative', 'ankle'),
                         ('peak negative', 'subtalar'),
                         ('average positive', 'ankle'),
                         ('average positive', 'subtalar'),
                         ('average negative', 'ankle'),
                         ('average negative', 'subtalar')]

        columns = pd.MultiIndex.from_tuples(column_tuples)

        power_mean_dfs = [peak_pos_power_mean, peak_neg_power_mean, 
                          avg_pos_power_mean, avg_neg_power_mean]
        power_mean = pd.concat(power_mean_dfs, axis=1)
        power_mean.columns = columns

        power_std_dfs = [peak_pos_power_std, peak_neg_power_std, 
                         avg_pos_power_std, avg_neg_power_std]
        power_std = pd.concat(power_std_dfs, axis=1)
        power_std.columns = columns

        target_dir = os.path.dirname(target[0])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with open(target[0], 'w') as f:
            f.write('mean power (W/kg) across subjects\n')
            power_mean.to_csv(f, line_terminator='\n')

        target_dir = os.path.dirname(target[1])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with open(target[1], 'w') as f:
            f.write('power standard deviation (W/kg) across subjects\n')
            power_std.to_csv(f, line_terminator='\n')


class TaskPlotPerturbationPowers(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects):
        super(TaskPlotPerturbationPowers, self).__init__(study)
        self.name = 'plot_perturbation_powers'
        self.subjects = subjects
        self.times = study.times
        self.torques = study.plot_torques
        self.subtalars = study.plot_subtalars
        self.colors = study.plot_colors
        self.edgecolor = 'black'
        self.legend_labels = ['eversion', 
                              'plantarflexion + eversion', 
                              'plantarflexion', 
                              'plantarflexion + inversion', 
                              'inversion']
        self.perturb_labels = ['torque0_subtalar-10',
                               'torque10_subtalar-10',
                               'torque10',
                               'torque10_subtalar10',
                               'torque0_subtalar10']

        self.figures_path = os.path.join(study.config['figures_path'])
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'perturbation_powers')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)

        self.add_action([os.path.join(self.analysis_path,
                                'perturbation_powers_mean_muscles.csv'),
                         os.path.join(self.analysis_path, 
                                'perturbation_powers_std_muscles.csv'),
                         os.path.join(self.analysis_path,
                                'perturbation_powers_mean_torques.csv'),
                         os.path.join(self.analysis_path, 
                                'perturbation_powers_std_torques.csv')],
                        [os.path.join(self.analysis_path, 
                                'ankle_perturbation_powers.png'),
                         os.path.join(self.analysis_path, 
                                'subtalar_perturbation_powers.png'),
                         os.path.join(self.figures_path,
                                'figureS13', 'figureS13.png')],
                        self.plot_perturbation_powers)

    def plot_perturbation_powers(self, file_dep, target):

        # Initialize figures
        # ------------------
        figs = list()
        import collections
        axes = collections.defaultdict(list)
        for icoord, coord in enumerate(['ankle', 'subtalar']):
            fig = plt.figure(figsize=(6, 6))
            iplot = 1
            for ipower, power in enumerate(['positive', 'negative']):
                for iactu, actu in enumerate(['muscles', 'torques']):
                    ax = fig.add_subplot(2, 2, iplot)
                    ax.grid(color='gray', linestyle='--', linewidth=0.5,
                            clip_on=False, alpha=1.0, zorder=0)
                    ax.set_xticks(np.arange(len(self.times)))
                    ax.set_xlim(0, len(self.times)-1)
                    ax.grid(color='gray', linestyle='--', linewidth=0.25,
                        clip_on=False, alpha=0.75, zorder=0)
                    ax.axhline(y=0, color='black', linestyle='-',
                            linewidth=0.1, alpha=1.0, zorder=2.5)
                    util.publication_spines(ax)

                    if power == 'positive':
                        ax.spines['bottom'].set_visible(False)
                        ax.set_xticklabels([])
                        ax.xaxis.set_ticks_position('none')
                        ax.tick_params(axis='x', which='both', bottom=False, 
                                       top=False, labelbottom=False)
                        if coord == 'ankle':
                            # ax.set_ylim(0, 0.35)
                            # ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
                            ax.set_ylim(0, 0.15)
                            ax.set_yticks([0, 0.03, 0.06, 0.09, 0.12, 0.15])
                        else:
                            ax.set_ylim(0, 0.2)
                            ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2])

                        if actu == 'muscles': 
                            ax.set_ylabel(r'average positive power $[W/kg]$')
            
                    else:
                        ax.spines['bottom'].set_position(('outward', 10))
                        ax.set_xticklabels([f'{time}' for time in self.times])
                        ax.set_xlabel('exoskeleton torque peak time\n(% gait cycle)')
                        if coord == 'ankle':
                            # ax.set_ylim(-0.08, 0)
                            # ax.set_yticks([-0.08, -0.06, -0.04, -0.02, 0])
                            ax.set_ylim(-0.05, 0)
                            ax.set_yticks([-0.05, -0.04, -0.03, -0.02, -0.01, 0])
                        else:
                            ax.set_ylim(-0.03, 0)
                            ax.set_yticks([-0.03, -0.025, -0.02, -0.015, -0.01, -0.005, 0])

                        if actu == 'muscles': 
                            ax.set_ylabel(r'average negative power $[W/kg]$')   

                    if actu == 'torques':
                        ax.spines['left'].set_visible(False)
                        ax.set_yticklabels([])
                        ax.yaxis.set_ticks_position('none')
                        ax.tick_params(axis='y', which='both', left=False, 
                                       right=False, labelleft=False)
                    else:
                        ax.spines['left'].set_position(('outward', 20))

                    axes[coord].append(ax)
                    iplot += 1

            figs.append(fig)

        # Plotting
        # --------
        lw = 0.2
        for iactu, actu in enumerate(['muscles', 'torques']):
            df_powers_mean = pd.read_csv(file_dep[0 + 2*iactu], index_col=[0, 1], 
                header=[0,1], skiprows=1)
            df_powers_std = pd.read_csv(file_dep[1 + 2*iactu], index_col=[0, 1], 
                header=[0,1], skiprows=1)

            ankle_indices = [1, 2, 3]
            ankle_labels = [self.legend_labels[i] for i in ankle_indices]
            subtalar_indices = [0, 1, 3, 4]
            subtalar_labels = [self.legend_labels[i] for i in subtalar_indices]

            ax_pos_ankle = axes['ankle'][0 + iactu]
            ax_neg_ankle = axes['ankle'][2 + iactu]
            df_ankle_pos_powers_mean = df_powers_mean['average positive']['ankle']
            df_ankle_pos_powers_std = df_powers_std['average positive']['ankle']
            df_ankle_neg_powers_mean = df_powers_mean['average negative']['ankle']
            df_ankle_neg_powers_std = df_powers_std['average negative']['ankle']
            width = 0.25
            N = len(ankle_indices)
            min_width = -width*((N-1)/2)
            max_width = -min_width
            offsets = np.linspace(min_width, max_width, N)
            handles = list()
            for iankle, offset in zip(ankle_indices, offsets):
                perturb = self.perturb_labels[iankle]
                color = self.colors[iankle]
                x = np.arange(len(self.times)) + offset

                df_this_ankle_pos_powers_mean = df_ankle_pos_powers_mean.xs(
                    perturb, level=1, drop_level=False)
                df_this_ankle_pos_powers_std = df_ankle_pos_powers_std.xs(
                    perturb, level=1, drop_level=False)

                h = ax_pos_ankle.bar(x, df_this_ankle_pos_powers_mean, width, color=color,
                    clip_on=False, zorder=2.5, edgecolor=self.edgecolor, lw=lw)
                handles.append(h)
                plot_errorbar(ax_pos_ankle, x, df_this_ankle_pos_powers_mean,
                    df_this_ankle_pos_powers_std)

                df_this_ankle_neg_powers_mean = df_ankle_neg_powers_mean.xs(
                    perturb, level=1, drop_level=False)
                df_this_ankle_neg_powers_std = df_ankle_neg_powers_std.xs(
                    perturb, level=1, drop_level=False)

                ax_neg_ankle.bar(x, df_this_ankle_neg_powers_mean, width, color=color,
                    clip_on=False, zorder=2.5, edgecolor=self.edgecolor, lw=lw)
                plot_errorbar(ax_neg_ankle, x, df_this_ankle_neg_powers_mean,
                    df_this_ankle_neg_powers_std)

            if actu == 'muscles':
                axes['ankle'][0].legend(handles, ankle_labels, 
                    loc='upper left', frameon=True, fontsize=8) 

            ax_pos_subtalar = axes['subtalar'][0 + iactu]
            ax_neg_subtalar = axes['subtalar'][2 + iactu]
            df_subtalar_pos_powers_mean = df_powers_mean['average positive']['subtalar']
            df_subtalar_pos_powers_std = df_powers_std['average positive']['subtalar']
            df_subtalar_neg_powers_mean = df_powers_mean['average negative']['subtalar']
            df_subtalar_neg_powers_std = df_powers_std['average negative']['subtalar']
            width = 0.2
            N = len(subtalar_indices)
            min_width = -width*((N-1)/2)
            max_width = -min_width
            offsets = np.linspace(min_width, max_width, N)
            handles = list()
            for isubtalar, offset in zip(subtalar_indices, offsets):
                perturb = self.perturb_labels[isubtalar]
                color = self.colors[isubtalar]
                x = np.arange(len(self.times)) + offset

                df_this_subtalar_pos_powers_mean = df_subtalar_pos_powers_mean.xs(
                    perturb, level=1, drop_level=False)
                df_this_subtalar_pos_powers_std = df_subtalar_pos_powers_std.xs(
                    perturb, level=1, drop_level=False)

                h = ax_pos_subtalar.bar(x, df_this_subtalar_pos_powers_mean, width, color=color,
                    clip_on=False, zorder=2.5, edgecolor=self.edgecolor, lw=lw)
                handles.append(h)
                plot_errorbar(ax_pos_subtalar, x, df_this_subtalar_pos_powers_mean,
                    df_this_subtalar_pos_powers_std)

                df_this_subtalar_neg_powers_mean = df_subtalar_neg_powers_mean.xs(
                    perturb, level=1, drop_level=False)
                df_this_subtalar_neg_powers_std = df_subtalar_neg_powers_std.xs(
                    perturb, level=1, drop_level=False)

                ax_neg_subtalar.bar(x, df_this_subtalar_neg_powers_mean, width, color=color,
                    clip_on=False, zorder=2.5, edgecolor=self.edgecolor, lw=lw)
                plot_errorbar(ax_neg_subtalar, x, df_this_subtalar_neg_powers_mean,
                    df_this_subtalar_neg_powers_std)

            if actu == 'muscles':
                axes['subtalar'][0].legend(handles, subtalar_labels, 
                    loc='upper left', frameon=True, fontsize=8) 

        import cv2
        def add_muscles_image(fig):
            side = 0.175
            l = 0.245
            b = 0.81
            w = side
            h = side
            ax = fig.add_axes([l, b, w, h], projection=None, polar=False)
            image = cv2.imread(
                os.path.join(self.study.config['figures_path'], 'images',
                    'sagittal_muscles.tiff'))[..., ::-1]
            ax.imshow(image, interpolation='none', aspect='equal')

            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yticklabels([])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.tick_params(axis='x', which='both', bottom=False, 
                           top=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', bottom=False, 
                           top=False, labelbottom=False)

        def add_torques_image(fig):
            side = 0.175
            l = 0.680
            b = 0.81
            w = side
            h = side
            ax = fig.add_axes([l, b, w, h], projection=None, polar=False)
            image = cv2.imread(
                os.path.join(self.study.config['figures_path'], 'images',
                    'sagittal_torques.tiff'))[..., ::-1]
            ax.imshow(image, interpolation='none', aspect='equal')

            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yticklabels([])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.tick_params(axis='x', which='both', bottom=False, 
                           top=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', bottom=False, 
                           top=False, labelbottom=False)

        for ifig, fig in enumerate(figs):
            fig.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.8, hspace=0.2)
            add_muscles_image(fig)
            add_torques_image(fig)
            fig.savefig(target[ifig], dpi=600)

        figs[0].savefig(target[2], dpi=600)
        plt.close()