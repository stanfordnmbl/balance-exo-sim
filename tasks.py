import os

import numpy as np
import pylab as pl
import pandas as pd
# import math
# from scipy.io import loadmat
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

from tracking_problem import TrackingProblem, TrackingConfig
from timestepping_problem import TimeSteppingProblem, TimeSteppingConfig

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

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
                 critically_damped_cutoff_frequency=10,
                 gaussian_smoothing_sigma=10):
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
            '%s.osim' % self.subject.name)
        self.scaled_param_model_fpath = os.path.join(
            self.subject.results_exp_path, 
            '%s_scaled_Fmax.osim' % self.subject.name)
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


class TaskAdjustScaledModel(osp.SubjectTask):
    REGISTRY = []
    def __init__(self, subject, marker_adjustments, treadmill=False):
        super(TaskAdjustScaledModel, self).__init__(subject)
        self.subject = subject
        self.study = subject.study
        self.mass = subject.mass
        self.name = '%s_adjust_scaled_model' % self.subject.name
        self.doc = 'Make adjustments to model marker post-scale'
        self.scaled_model_fpath = os.path.join(
            self.subject.results_exp_path, 
            '%s_scaled_Fmax.osim' % self.subject.name)
        if treadmill:
            self.final_model_fpath = os.path.join(
                self.subject.results_exp_path, 
                '%s_scaled_Fmax_markers.osim' % self.subject.name)
        else:
            self.final_model_fpath = os.path.join(
                self.subject.results_exp_path, 
                '%s_final.osim' % self.subject.name)
        self.marker_adjustments = marker_adjustments

        self.add_action([self.scaled_model_fpath],
                        [self.final_model_fpath],
                        self.adjust_model_markers)

    def adjust_model_markers(self, file_dep, target):
        print('Adjusting scaled model marker locations... ')
        import opensim as osm
        model = osm.Model(file_dep[0])
        markerSet = model.updMarkerSet()
        for name, adj in self.marker_adjustments.items():
            marker = markerSet.get(name)
            loc = marker.get_location()
            loc.set(adj[0], adj[1])
            marker.set_location(loc)

        # print('Adding mtp passive stiffness and damping...')
        # for coordName in ['mtp_angle_r', 'mtp_angle_l']:
        #     sgf = osim.SpringGeneralizedForce(coordName)
        #     sgf.setName(f'passive_stiffness_{coordName}')
        #     sgf.setStiffness(0.4 * self.mass)
        #     sgf.setViscosity(2.0)
        #     model.addForce(sgf)

        # upper_stiffness = 0.10 * mass 
        # lower_stiffness = 0.25 * mass
        # lower_limit = 5
        # upper_limit = 120
        # damping = 0.25
        # transition = 10
        # clf = osim.CoordinateLimitForce('knee_angle_r', 
        #     upper_limit, upper_stiffness, 
        #     lower_limit, lower_stiffness, 
        #     damping, transition)
        # model.addForce(clf)

        # clf = osim.CoordinateLimitForce('knee_angle_l', 
        #     upper_limit, upper_stiffness, 
        #     lower_limit, lower_stiffness, 
        #     damping, transition)
        # model.addForce(clf)

        model.finalizeConnections()
        model.printToXML(target[0])


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


# Validate inverse results
# ------------------------

class TaskValidateMarkerErrors(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, cond_names=['walk1','walk2','walk3','walk4']):
        super(TaskValidateMarkerErrors, self).__init__(study)
        suffix = ''
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction 
        self.name = 'validate_marker_errors%s' % suffix
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
                    'marker_error.csv'))
                
        val_fname = os.path.join(self.validate_path, 'marker_errors.txt')
        self.add_action(errors_fpaths, [val_fname],
                        self.validate_marker_errors)

    def validate_marker_errors(self, file_dep, target):
        if not os.path.isdir(self.validate_path): os.mkdir(self.validate_path)

        all_errors = pd.DataFrame()
        for file in file_dep:
            df = pd.read_csv(file)
            df = df.drop(columns=['Unnamed: 0', 'time'])
            all_errors = pd.concat([all_errors, df])

        numMarkers = all_errors.shape[1]
        all_errors_sq = all_errors.pow(2)
        all_errors_sq_sum_norm = all_errors_sq.sum(axis=1) / float(numMarkers)
        rms_errors = all_errors_sq_sum_norm.pow(0.5)

        peak_rms = rms_errors.max()
        mean_rms = rms_errors.mean()

        with open(target[0],"w") as f:
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
            f.write('Peak RMS marker error: %1.3f cm \n' % (100*peak_rms))
            f.write('Mean RMS marker error: %1.3f cm \n' % (100*mean_rms))


class TaskValidateKinetics(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, cond_name='walk2',
            gen_forces=['hip_flexion_r_moment', 
                        'knee_angle_r_moment',
                        'ankle_angle_r_moment'],
            residual_moments=['pelvis_tilt_moment', 
                              'pelvis_list_moment',
                              'pelvis_rotation_moment'],
            residual_forces=['pelvis_tx_force',
                             'pelvis_ty_force',
                             'pelvis_tz_force'],
            grfs=['ground_force_r_vx',
                  'ground_force_r_vy', 
                  'ground_force_r_vz']):
        super(TaskValidateKinetics, self).__init__(study)
        suffix = ''
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction 
        self.name = 'validate_kinetics%s' % suffix
        self.doc = 'Validate joint moments and residuals across subjects and conditions.'
        self.results_path = study.config['results_path']
        self.validate_path = os.path.join(study.config['validate_path'],
            'kinetics')
        self.figure_path = os.path.join(study.config['figures_path'], 
            'figureS1')
        self.cond_name = cond_name
        self.subjects = study.subjects
        self.gen_forces = gen_forces
        if not len(residual_moments) == 3:
            raise Exception('There must be 3 residual moment components.')
        self.residual_moments = residual_moments
        if not len(residual_forces) == 3:
            raise Exception('There must be 3 residual force components.')
        self.residual_forces = residual_forces
        if not len(grfs) == 3:
            raise Exception('There must be 3 ground reaction force components.')
        self.grfs = grfs

        colors = plt.cm.jet(np.linspace(0, 1, len(study.subjects)))
        masses = list()
        for subject in study.subjects:
            masses.append(subject.mass)

        moments_fpaths_all = list()
        bodykin_fpaths_all = list()
        gait_events_all = list()
        isubjs_all = list()
        grf_fpaths_all = list()

        moments_fpaths = list()
        gait_events = list()
        isubjs = list()
        for isubj, subject in enumerate(study.subjects):
            fpath = os.path.join(self.results_path, 'experiments', 
                subject.name, cond_name, 'id', 'results', 
                '%s_%s_%s_id_solution.sto' % (study.name, subject.name, 
                    cond_name))
            moments_fpaths.append(fpath)
            moments_fpaths_all.append(fpath)

            bodykin_fpath = os.path.join(self.results_path, 'experiments',
                subject.name, cond_name, 'body_kinematics', 'results',
                '%s_%s_%s_body_kinematics_BodyKinematics_pos_global.sto' % (
                    study.name, subject.name, cond_name))
            bodykin_fpaths_all.append(bodykin_fpath)

            condition = subject.get_condition(cond_name)
            trial = condition.get_trial(1)
            cycles = trial.get_cycles()
            gait_events_this_cond = list()
            for cycle in cycles:
                gait_events_this_cond.append((cycle.start, cycle.end))
            gait_events.append(gait_events_this_cond)
            gait_events_all.append(gait_events_this_cond)
            isubjs.append(isubj)
            isubjs_all.append(isubj)

            grf_fpath = os.path.join(self.results_path, 'experiments', 
                subject.name, cond_name, 'expdata', 'ground_reaction.mot')
            grf_fpaths_all.append(grf_fpath)

        joint_moments_fname = os.path.join(self.validate_path, 
                'joint_moments_%s.pdf' % cond_name)
        joint_moments_figname = os.path.join(self.figure_path, 
                'figureS1.pdf')
        self.add_action(moments_fpaths, 
                        [joint_moments_fname, joint_moments_figname], 
                        self.validate_joint_moments, gait_events, isubjs, 
                        colors, masses)

        residuals_fname = os.path.join(self.validate_path, 'residuals.txt')
        moments_fname = os.path.join(self.validate_path, 'moments.csv')
        forces_fname = os.path.join(self.validate_path, 'forces.csv')
        self.add_action(moments_fpaths_all, 
                        [residuals_fname, moments_fname, forces_fname],
                        self.validate_residuals, gait_events_all, grf_fpaths_all,
                        bodykin_fpaths_all)

    def validate_joint_moments(self, file_dep, target, gait_events, isubjs,
            colors, masses):
        if not os.path.isdir(self.validate_path): os.mkdir(self.validate_path)
        if not os.path.isdir(self.figure_path): os.mkdir(self.figure_path)

        nice_gen_force_names = {'hip_flexion_r_moment': 'hip flexion moment',
                                'knee_angle_r_moment': 'knee flexion moment',
                                'ankle_angle_r_moment': 'ankle plantarflexion moment'}

        y_lim_dict = {'hip_flexion_r_moment': (-1.2, 1.2),
                       'knee_angle_r_moment': (-1.2, 1.2),
                       'ankle_angle_r_moment': (-0.5, 2.0)}

        subject_labels = ['subject 1', 'subject 2', 'subject 3', 'subject 4',
                          'subject 5']

        import matplotlib
        fig = pl.figure(figsize=(6, 8))
        ind_axes = list()
        ind_handles = list()
        mean_axes = list()
        dfs = list()
        for iforce, gen_force in enumerate(self.gen_forces):
            idx = iforce + 1
            ind_axes.append(fig.add_subplot(len(self.gen_forces), 2, 2*idx-1))
            mean_axes.append(fig.add_subplot(len(self.gen_forces), 2, 2*idx))
            dfs.append(pd.DataFrame())

        for file, gait_events_cond, isubj in zip(file_dep, gait_events, isubjs):
            id = pp.storage2numpy(file)
            time = id['time']
            for ige, gait_event in enumerate(gait_events_cond):
                start = np.argmin(abs(time-gait_event[0]))
                end = np.argmin(abs(time-gait_event[1]))
                new_time = np.linspace(time[start], time[end], 101)
                pgc = np.linspace(0, 100, 101)
                for iforce, gen_force in enumerate(self.gen_forces):
                    force = id[gen_force][start:end] / masses[isubj]                    
                    force_interp = np.interp(new_time, time[start:end], force)
                    dfs[iforce] = pd.concat([dfs[iforce], 
                            pd.DataFrame(force_interp)], axis=1)

                    sign = -1 if gen_force=='ankle_angle_r_moment' else 1
                    h, = ind_axes[iforce].plot(pgc, sign*force_interp, color=colors[isubj])
                    if not ige and not iforce:
                        ind_handles.append(h)
                    ind_axes[iforce].set_ylabel(
                        '%s (N-m/kg)' % nice_gen_force_names[gen_force])
                    ind_axes[iforce].set_ylim(y_lim_dict[gen_force])
                    ind_axes[iforce].set_xlim((0, 100))
                    ind_axes[iforce].axhline(ls='--', color='lightgray', zorder=0)
                    ind_axes[iforce].spines['top'].set_visible(False)
                    ind_axes[iforce].spines['right'].set_visible(False)
                    ind_axes[iforce].tick_params(direction='in')


        ind_axes[0].legend(ind_handles, subject_labels, fancybox=False,
                frameon=False, prop={'size': 8}, loc=2)                    

        for iforce, gen_force in enumerate(self.gen_forces):
            sign = -1 if gen_force=='ankle_angle_r_moment' else 1
            force_mean = sign*dfs[iforce].mean(axis=1)
            force_std = dfs[iforce].std(axis=1)
            mean_axes[iforce].fill_between(pgc, force_mean-force_std, 
                force_mean+force_std, color='black', alpha=0.2)
            std_h = matplotlib.patches.Patch(color='black', alpha=0.2)
            mean_h, = mean_axes[iforce].plot(pgc, force_mean, color='black')
            mean_axes[iforce].set_ylim(y_lim_dict[gen_force])
            mean_axes[iforce].set_xlim((0, 100))
            mean_axes[iforce].axhline(ls='--', color='lightgray', zorder=0)
            mean_axes[iforce].spines['top'].set_visible(False)
            mean_axes[iforce].spines['right'].set_visible(False)
            mean_axes[iforce].tick_params(direction='in')
            mean_axes[iforce].set_yticklabels([])
            if iforce == len(self.gen_forces)-1:
                mean_axes[iforce].set_xlabel('gait cycle (%)')
                ind_axes[iforce].set_xlabel('gait cycle (%)')
            else:
                ind_axes[iforce].set_xticklabels([])
                mean_axes[iforce].set_xticklabels([])

        mean_axes[0].legend([mean_h, std_h], ['mean', 'std'], fancybox=False,
                frameon=False, prop={'size': 8}, loc=2)  

        fig.tight_layout()
        fig.savefig(target[0])
        fig.savefig(target[0].replace('.pdf', '.png'), dpi=600)
        fig.savefig(target[1])
        fig.savefig(target[1].replace('.pdf', '.png'), dpi=600)
        pl.close(fig)

    def validate_residuals(self, file_dep, target, gait_events, grf_fpaths,
            bodykin_fpaths):
        if not os.path.isdir(self.validate_path): os.mkdir(self.validate_path)

        df_moments = pd.DataFrame()
        df_forces = pd.DataFrame()
        df_grfs = pd.DataFrame()
        df_ycom = pd.DataFrame()
        for file, gait_events_cond, grf_fpath, bodykin_fpath in zip(file_dep, 
                gait_events, grf_fpaths, bodykin_fpaths):
            id = pp.storage2numpy(file)
            time = id['time']
            grfs = pp.storage2numpy(grf_fpath)
            grf_time = grfs['time']
            bodykin = pp.storage2numpy(bodykin_fpath)
            bodykin_time = bodykin['time']
            for gait_event in gait_events_cond:
                start = np.argmin(abs(time-gait_event[0]))
                end = np.argmin(abs(time-gait_event[1]))
                new_time = np.linspace(time[start], time[end], 101)
                for residual_moment in self.residual_moments:
                    moment = id[residual_moment][start:end] 
                    moment_interp = np.interp(new_time, time[start:end], moment)
                    df_moments = pd.concat([df_moments, 
                        pd.DataFrame(moment_interp)], axis=1)

                for residual_force in self.residual_forces:
                    force = id[residual_force][start:end]
                    force_interp = np.interp(new_time, time[start:end], force)
                    df_forces = pd.concat([df_forces, 
                        pd.DataFrame(force_interp)], axis=1)

                grf_start = np.argmin(abs(grf_time-gait_event[0]))
                grf_end = np.argmin(abs(grf_time-gait_event[1]))
                new_grf_time = np.linspace(grf_time[grf_start], 
                        grf_time[grf_end], 101)
                for grf in self.grfs:
                    reaction = grfs[grf][grf_start:grf_end]
                    reaction_interp = np.interp(new_grf_time, 
                        grf_time[grf_start:grf_end], reaction)
                    df_grfs = pd.concat([df_grfs, pd.DataFrame(reaction_interp)], 
                        axis=1)

                bodykin_start = np.argmin(abs(bodykin_time-gait_event[0]))
                bodykin_end = np.argmin(abs(bodykin_time-gait_event[1]))
                new_bodykin_time = np.linspace(bodykin_time[bodykin_start],
                    bodykin_time[bodykin_end], 101)
                ycom = bodykin['center_of_mass_Y'][bodykin_start:bodykin_end]
                ycom_interp = np.interp(new_bodykin_time, 
                    bodykin_time[bodykin_start:bodykin_end], ycom)
                df_ycom = pd.concat([df_ycom, pd.DataFrame(ycom_interp)], axis=1)

        df_moments = pd.DataFrame(np.vstack(np.split(df_moments, 3*len(file_dep), 
            axis=1)))
        df_forces = pd.DataFrame(np.vstack(np.split(df_forces, 3*len(file_dep), 
            axis=1)))
        df_grfs = pd.DataFrame(np.vstack(np.split(df_grfs, 3*len(file_dep), 
            axis=1)))
        df_ycom = pd.DataFrame(np.vstack(np.split(df_ycom, 3*len(file_dep), 
            axis=1)))

        mag_moments = np.linalg.norm(df_moments, axis=1)
        mag_forces = np.linalg.norm(df_forces, axis=1)
        mag_grfs = np.linalg.norm(df_grfs, axis=1)
        peak_grfs = mag_grfs.max()
        avg_ycom = df_ycom.mean()[0]

        peak_moments = 100*mag_moments.max() / (peak_grfs * avg_ycom)
        rms_moments = 100*np.sqrt(np.mean(np.square(mag_moments))) / (
            peak_grfs * avg_ycom)
        peak_forces = 100*mag_forces.max() / peak_grfs
        rms_forces = 100*np.sqrt(np.mean(np.square(mag_forces))) / peak_grfs

        with open(target[0],"w") as f:
            f.write('subjects: ')
            for isubj, subject in enumerate(self.subjects):
                if isubj:
                    f.write(', %s' % subject.name)
                else:
                    f.write('%s' % subject.name)
            f.write('\n')
            f.write('condition: %s', self.cond_name)
            f.write('\n')
            f.write('Peak residual moments (%% GRF): %1.3f \n' % peak_moments) 
            f.write('RMS residual moments (%% GRF): %1.3f \n' % rms_moments)
            f.write('Peak residual forces (%% GRF): %1.3f \n' % peak_forces) 
            f.write('RMS residual forces (%% GRF): %1.3f \n' % rms_forces)

        target_dir = os.path.dirname(target[1])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with open(target[1], 'w') as f:
            f.write('residual moments\n')
            df_moments.to_csv(f, line_terminator='\n')

        target_dir = os.path.dirname(target[2])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with open(target[2], 'w') as f:
            f.write('residual forces\n')
            df_forces.to_csv(f, line_terminator='\n')


class TaskValidateKinematics(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, cond_name='walk2',
            joint_angles=['hip_flexion_r', 
                          'knee_angle_r',
                          'ankle_angle_r']):
        super(TaskValidateKinematics, self).__init__(study)
        suffix = ''
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction 
        self.name = 'validate_kinematics%s' % suffix
        self.doc = 'Validate joint angles across subjects and conditions.'
        self.results_path = study.config['results_path']
        self.validate_path = os.path.join(study.config['validate_path'],
            'kinematics')
        self.figure_path = os.path.join(study.config['figures_path'], 
            'figureS2')
        self.cond_name = cond_name
        self.subjects = study.subjects
        self.joint_angles = joint_angles

        colors = plt.cm.jet(np.linspace(0, 1, len(study.subjects)))

        angles_fpaths = list()
        gait_events = list()
        isubjs = list()
        for isubj, subject in enumerate(study.subjects):
            fpath = os.path.join(self.results_path, 'experiments', 
                subject.name, cond_name, 'ik', 
                '%s_%s_%s_ik_solution.mot' % (study.name, subject.name, 
                    cond_name))
            angles_fpaths.append(fpath)

            condition = subject.get_condition(cond_name)
            trial = condition.get_trial(1)
            cycles = trial.get_cycles()
            gait_events_this_cond = list()
            for cycle in cycles:
                gait_events_this_cond.append((cycle.start, cycle.end))
            gait_events.append(gait_events_this_cond)
            isubjs.append(isubj)

        joint_angles_fname = os.path.join(self.validate_path, 
                'joint_angles_%s.pdf' % cond_name)
        joint_angles_figname = os.path.join(self.figure_path, 
                'figureS2.pdf')
        self.add_action(angles_fpaths, 
                        [joint_angles_fname, joint_angles_figname], 
                        self.validate_joint_angles, gait_events, isubjs, 
                        colors)

    def validate_joint_angles(self, file_dep, target, gait_events, isubjs,
            colors):
        if not os.path.isdir(self.validate_path): os.mkdir(self.validate_path)
        if not os.path.isdir(self.figure_path): os.mkdir(self.figure_path)

        nice_joint_angle_names = {'hip_flexion_r': 'hip flexion angle',
                                  'knee_angle_r': 'knee flexion angle',
                                  'ankle_angle_r': 'ankle plantarflexion angle'}

        y_lim_dict = {'hip_flexion_r': (-40, 40),
                       'knee_angle_r': (-10, 80),
                       'ankle_angle_r': (-30, 30)}

        subject_labels = ['subject 1', 'subject 2', 'subject 3', 'subject 4',
                          'subject 5']

        import matplotlib
        fig = pl.figure(figsize=(6, 8))
        ind_axes = list()
        ind_handles = list()
        mean_axes = list()
        dfs = list()
        for iangle, joint_angle in enumerate(self.joint_angles):
            idx = iangle + 1
            ind_axes.append(fig.add_subplot(len(self.joint_angles), 2, 2*idx-1))
            mean_axes.append(fig.add_subplot(len(self.joint_angles), 2, 2*idx))
            dfs.append(pd.DataFrame())

        for file, gait_events_cond, isubj in zip(file_dep, gait_events, isubjs):
            ik = pp.storage2numpy(file)
            time = ik['time']
            for ige, gait_event in enumerate(gait_events_cond):
                start = np.argmin(abs(time-gait_event[0]))
                end = np.argmin(abs(time-gait_event[1]))
                new_time = np.linspace(time[start], time[end], 101)
                pgc = np.linspace(0, 100, 101)
                for iangle, joint_angle in enumerate(self.joint_angles):
                    angle = ik[joint_angle][start:end]                   
                    angle_interp = np.interp(new_time, time[start:end], angle)
                    dfs[iangle] = pd.concat([dfs[iangle], 
                            pd.DataFrame(angle_interp)], axis=1)

                    h, = ind_axes[iangle].plot(pgc, angle_interp, color=colors[isubj])
                    if not ige and not iangle:
                        ind_handles.append(h)
                    ind_axes[iangle].set_ylabel(
                        '%s (degrees)' % nice_joint_angle_names[joint_angle])
                    ind_axes[iangle].set_ylim(y_lim_dict[joint_angle])
                    ind_axes[iangle].set_xlim((0, 100))
                    ind_axes[iangle].axhline(ls='--', color='lightgray', zorder=0)
                    ind_axes[iangle].spines['top'].set_visible(False)
                    ind_axes[iangle].spines['right'].set_visible(False)
                    ind_axes[iangle].tick_params(direction='in')


        ind_axes[0].legend(ind_handles, subject_labels, fancybox=False,
                frameon=False, prop={'size': 8}, loc=9)                    

        for iangle, joint_angle in enumerate(self.joint_angles):
            angle_mean = dfs[iangle].mean(axis=1)
            angle_std = dfs[iangle].std(axis=1)
            mean_axes[iangle].fill_between(pgc, angle_mean-angle_std, 
                angle_mean+angle_std, color='black', alpha=0.2)
            std_h = matplotlib.patches.Patch(color='black', alpha=0.2)
            mean_h, = mean_axes[iangle].plot(pgc, angle_mean, color='black')
            mean_axes[iangle].set_ylim(y_lim_dict[joint_angle])
            mean_axes[iangle].set_xlim((0, 100))
            mean_axes[iangle].axhline(ls='--', color='lightgray', zorder=0)
            mean_axes[iangle].spines['top'].set_visible(False)
            mean_axes[iangle].spines['right'].set_visible(False)
            mean_axes[iangle].tick_params(direction='in')
            mean_axes[iangle].set_yticklabels([])
            if iangle == len(self.joint_angles)-1:
                mean_axes[iangle].set_xlabel('gait cycle (%)')
                ind_axes[iangle].set_xlabel('gait cycle (%)')
            else:
                ind_axes[iangle].set_xticklabels([])
                mean_axes[iangle].set_xticklabels([])

        mean_axes[0].legend([mean_h, std_h], ['mean', 'std'], fancybox=False,
                frameon=False, prop={'size': 8}, loc=9)  

        fig.tight_layout()
        fig.savefig(target[0])
        fig.savefig(target[0].replace('.pdf', '.png'), dpi=600)
        fig.savefig(target[1])
        fig.savefig(target[1].replace('.pdf', '.png'), dpi=600)
        pl.close(fig)


# Sensitivity analysis
# --------------------

class TaskMocoSensitivityAnalysis(osp.TrialTask):
    REGISTRY = []
    def __init__(self, trial, initial_time, final_time, mesh_interval=0.02,
                tolerance=1e-2, walking_speed=1.25, guess_fpath=None,
                mesh_or_tol_analysis='mesh', randomize_guess=False, **kwargs):
        super(TaskMocoSensitivityAnalysis, self).__init__(trial)
        suffix = f'mesh{int(1000*mesh_interval)}_tol{int(1e4*tolerance)}'
        config_name = f'unperturbed_sensitivity_{suffix}_{mesh_or_tol_analysis}'
        self.config_name = config_name
        self.name = trial.subject.name + '_moco_' + config_name
        self.initial_time = initial_time
        self.final_time = final_time
        self.mesh_interval = mesh_interval
        self.tolerance = tolerance
        self.walking_speed = walking_speed
        self.weights = trial.study.weights
        self.root_dir = trial.study.config['doit_path']
        self.guess_fpath = guess_fpath
        self.randomize_guess = randomize_guess

        expdata_dir = os.path.join(trial.results_exp_path, 'tracking_data', 'expdata')
        extloads_dir = os.path.join(trial.results_exp_path, 'tracking_data', 'extloads')
        self.tracking_coordinates_fpath = os.path.join(expdata_dir, 'coordinates.sto')
        self.coordinates_std_fpath = os.path.join(trial.results_exp_path, 
            f'{trial.id}_joint_angle_standard_deviations.csv')
        self.tracking_extloads_fpath = os.path.join(extloads_dir, 'external_loads.xml')
        self.tracking_grfs_fpath = os.path.join(expdata_dir, 'ground_reaction.mot')

        self.result_fpath = os.path.join(self.study.config['results_path'],
            'sensitivity', trial.subject.name, suffix)
        if not os.path.exists(self.result_fpath): os.makedirs(self.result_fpath)

        self.archive_fpath = os.path.join(self.study.config['results_path'],
            'sensitivity', trial.subject.name, suffix, 'archive')
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
        config = TrackingConfig(
            self.config_name, self.config_name, 'black', self.weights,
            constrain_average_speed=False,
            periodic=True,
            guess=self.guess_fpath,
            randomize_guess=self.randomize_guess
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
            self.walking_speed,
            [config]
        )

        result.generate_results()
        result.report_results()


class TaskPlotSensitivityResults(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects):
        super(TaskPlotSensitivityResults, self).__init__(study)
        self.name = 'plot_sensitivity_analysis_results'
        self.results_path = os.path.join(study.config['results_path'], 
            'sensitivity')
        self.validate_path = os.path.join(study.config['validate_path'],
            'sensitivity')
        if not os.path.exists(self.validate_path): 
            os.makedirs(self.validate_path)
        self.subjects = subjects
        self.meshes = [0.05, 0.04, 0.03, 0.02, 0.01]
        self.tols = [1e1, 1e0, 1e-1, 1e-2]
        self.tol_exps = [1, 0, -1, -2]

        deps_mesh = list()
        for subject in subjects:
            for mesh in self.meshes:
                suffix = f'mesh{int(1000*mesh)}_tol1000'
                result = f'unperturbed_sensitivity_{suffix}_mesh.sto'
                result_fpath = os.path.join(self.results_path,
                    subject, suffix, result)
                deps_mesh.append(result_fpath)

        self.add_action(deps_mesh, 
                        [os.path.join(self.validate_path, 
                            'sensitivity_mesh.png')], 
                        self.plot_sensitivity_mesh)

        deps_tol = list()
        for subject in subjects:
            for tol in self.tols:
                suffix = f'mesh10_tol{int(1e4*tol)}'
                result = f'unperturbed_sensitivity_{suffix}_tol.sto'
                result_fpath = os.path.join(
                    self.results_path,
                    subject, suffix, result)
                deps_tol.append(result_fpath)

        self.add_action(deps_tol, 
                        [os.path.join(self.validate_path, 
                            'sensitivity_tol.png')], 
                        self.plot_sensitivity_tol)

    def plot_sensitivity_mesh(self, file_dep, target):
        
        idep = 0
        objectives = np.zeros([len(self.subjects), len(self.meshes)])
        norm_objectives = np.zeros_like(objectives)
        for isubj, subject in enumerate(self.subjects):
            for imesh, mesh in enumerate(self.meshes):
                table = osim.TimeSeriesTable(file_dep[idep])
                objective = float(table.getTableMetaDataAsString('objective'))
                objectives[isubj, imesh] = objective
                idep = idep + 1

        for isubj, subject in enumerate(self.subjects):
            norm_objectives[isubj, :] = objectives[isubj, :] / objectives[isubj, -1]

        norm_obj_mean = np.mean(norm_objectives[:, :-1], axis=0)
        norm_obj_std = np.std(norm_objectives[:, :-1], axis=0)   

        fig = pl.figure(figsize=(3.5, 2.5))
        ax = fig.add_subplot(1,1,1)
        x = np.arange(len(norm_obj_mean))
        ax.bar(x, norm_obj_mean)
        plotline, caplines, barlinecols = ax.errorbar(
                x, norm_obj_mean, yerr=norm_obj_std, color='black', fmt='none',
                capsize=0, solid_capstyle='projecting', lw=1, lolims=True)
        caplines[0].set_marker('_')
        caplines[0].set_markersize(8)
        tick_labels = [f'{1000*mesh}' for mesh in self.meshes[:-1]]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(1.0, 1.5)
        ax.set_yticks([1, 1.25, 1.5])
        ax.set_ylabel('normalized objective')
        ax.set_xlabel('mesh interval (ms)')
        ax.axhline(y=1.0, color='gray', linewidth=0.5, ls='--', zorder=0)

        fig.tight_layout()
        fig.savefig(target[0], dpi=600)
        pl.close(fig)

    def plot_sensitivity_tol(self, file_dep, target):

        idep = 0
        objectives = np.zeros([len(self.subjects), len(self.tols)])
        norm_objectives = np.zeros_like(objectives)
        for isubj, subject in enumerate(self.subjects):
            for itol, tol in enumerate(self.tols):
                table = osim.TimeSeriesTable(file_dep[idep])
                objective = float(table.getTableMetaDataAsString('objective'))
                objectives[isubj, itol] = objective
                idep = idep + 1

        for isubj, subject in enumerate(self.subjects):
            norm_objectives[isubj, :] = objectives[isubj, :] / objectives[isubj, -1]

        norm_obj_mean = np.mean(norm_objectives[:, :-1], axis=0)
        norm_obj_std = np.std(norm_objectives[:, :-1], axis=0)   

        fig = pl.figure(figsize=(3.5, 2.5))
        ax = fig.add_subplot(1,1,1)
        x = np.arange(len(norm_obj_mean))
        ax.bar(x, norm_obj_mean)
        plotline, caplines, barlinecols = ax.errorbar(
                x, norm_obj_mean, yerr=norm_obj_std, color='black', fmt='none',
                capsize=0, solid_capstyle='projecting', lw=1, lolims=True)
        caplines[0].set_marker('_')
        caplines[0].set_markersize(8)
        tick_labels = [f'$10^{i}$' for i in self.tol_exps[:-1]]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(1.0, 1.05)
        ax.set_yticks([1, 1.025, 1.05])
        ax.set_ylabel('normalized objective')
        ax.set_xlabel('convergence tolerance')
        ax.axhline(y=1.0, color='gray', linewidth=0.5, ls='--', zorder=0)

        fig.tight_layout()
        fig.savefig(target[0], dpi=600)
        pl.close(fig)


# Unperturbed walking
# -------------------

class TaskMocoUnperturbedWalkingGuess(osp.TrialTask):
    REGISTRY = []
    def __init__(self, trial, initial_time, final_time, mesh_interval=0.02,
                walking_speed=1.25, constrain_average_speed=True, guess_fpath=None, 
                constrain_initial_state=False, periodic=True, cost_scale=1.0,
                costs_enabled=True, pelvis_boundary_conditions=True,
                reserve_strength=0, **kwargs):
        super(TaskMocoUnperturbedWalkingGuess, self).__init__(trial)
        config_name = (f'unperturbed_guess_mesh{int(1000*mesh_interval)}'
                       f'_scale{int(cost_scale)}')
        if not costs_enabled: config_name += f'_costsDisabled'
        if reserve_strength: config_name += f'_reserve{reserve_strength}'
        if periodic: config_name += f'_periodic'
        self.config_name = config_name
        self.name = trial.subject.name + '_moco_' + config_name
        self.initial_time = initial_time
        self.final_time = final_time
        self.mesh_interval = mesh_interval
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
            weights[weight_name] /= self.cost_scale

        config = TrackingConfig(
            self.config_name, self.config_name, 'black', weights,
            constrain_average_speed=self.constrain_average_speed,
            constrain_initial_state=self.constrain_initial_state,
            pelvis_boundary_conditions=self.pelvis_boundary_conditions,
            periodic=self.periodic,
            guess=self.guess_fpath,
            effort_enabled=self.costs_enabled,
            tracking_enabled=self.costs_enabled
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
            self.walking_speed,
            [config],
            reserves=bool(self.reserve_strength),
            reserve_strength=self.reserve_strength
        )

        result.generate_results()
        result.report_results()


class TaskMocoUnperturbedWalking(osp.TrialTask):
    REGISTRY = []
    def __init__(self, trial, initial_time, final_time, mesh_interval=0.02,
                 walking_speed=1.25, guess_fpath=None, periodic=True,
                 lumbar_stiffness=1.0, **kwargs):
        super(TaskMocoUnperturbedWalking, self).__init__(trial)
        suffix = ''
        if not lumbar_stiffness == 1.0:
            suffix = f'_lumbar{lumbar_stiffness}'
        self.config_name = f'unperturbed{suffix}'
        self.name = f'{trial.subject.name}_moco_{self.config_name}'
        self.initial_time = initial_time
        self.final_time = final_time
        self.mesh_interval = mesh_interval
        self.walking_speed = walking_speed
        self.guess_fpath = guess_fpath
        self.root_dir = trial.study.config['doit_path']
        self.periodic = periodic
        self.lumbar_stiffness = lumbar_stiffness
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
            self.study.config['results_path'], self.config_name, 
            trial.subject.name)
        if not os.path.exists(self.result_fpath): 
            os.makedirs(self.result_fpath)

        self.archive_fpath = os.path.join(
            self.study.config['results_path'], self.config_name, 
            trial.subject.name, 'archive')
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
            guess=self.guess_fpath
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
            self.walking_speed,
            [config])

        result.generate_results()
        result.report_results()


class TaskPlotUnperturbedResults(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, masses, times, time_colors,
            weld_lumbar_joint=False):
        super(TaskPlotUnperturbedResults, self).__init__(study)
        suffix = '_lumbar_welded' if weld_lumbar_joint else ''
        self.config_name = f'unperturbed{suffix}'
        self.name = f'plot_{self.config_name}_results'
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
        self.models = list()
        self.times = times
        self.time_colors = time_colors

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
                            f'{self.config_name}_coordinates.png')], 
                        self.plot_unperturbed_coordinates)

        self.add_action(unperturbed_grf_fpaths + grf_fpaths, 
                        [os.path.join(self.validate_path, 
                            f'{self.config_name}_grfs.png'),
                        os.path.join(self.validate_path, 
                            f'{self.config_name}_grf_reference.png')], 
                        self.plot_unperturbed_grfs)

        self.add_action(unperturbed_fpaths + emg_fpaths, 
                        [os.path.join(self.validate_path, 
                            f'{self.config_name}_muscle_activity.png')], 
                        self.plot_unperturbed_muscle_activity)

        self.add_action(unperturbed_com_fpaths + experiment_com_fpaths, 
                        [os.path.join(self.validate_path, 
                            f'{self.config_name}_center_of_mass.png')], 
                        self.plot_unperturbed_center_of_mass)

        self.add_action(unperturbed_fpaths, 
                        [os.path.join(self.validate_path, 
                            f'{self.config_name}_step_widths.png')], 
                        self.plot_unperturbed_step_widths)

    def plot_unperturbed_coordinates(self, file_dep, target): 

        numSubjects = len(self.subjects)
        N = 100
        pgc = np.linspace(0, 100, N)
        labels = ['hip flexion', 'hip adduction', 'knee flexion', 'ankle dorsiflexion']
        coordinates = ['hip_flexion', 'hip_adduction', 'knee_angle', 'ankle_angle']
        joints = ['hip', 'hip', 'walker_knee', 'ankle']
        bounds = [[-20, 40],
                  [-20, 10],
                  [0, 80],
                  [-20, 30]]
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
        fig = plt.figure(figsize=(5, 8))
        gs = gridspec.GridSpec(len(coordinates), 2)
        for ic, coord in enumerate(coordinates):
            for iside, side in enumerate(['l','r']):
                key = f'/jointset/{joints[ic]}_{side}/{coord}_{side}/value'
                ax = fig.add_subplot(gs[ic, iside])
                exp_mean = np.mean(experiment_dict[key], axis=1)
                exp_std = np.std(experiment_dict[key], axis=1)
                unp_mean = np.mean(unperturbed_dict[key], axis=1)
                unp_std = np.std(unperturbed_dict[key], axis=1)

                h_exp, = ax.plot(pgc, exp_mean, color=self.exp_color, lw=2.5)
                ax.fill_between(pgc, exp_mean + exp_std, exp_mean - exp_std, color=self.exp_color,
                    alpha=0.3, linewidth=0.0, edgecolor='none')
                h_unp, = ax.plot(pgc, unp_mean, color=self.unp_color, lw=2)
                ax.fill_between(pgc, unp_mean + unp_std, unp_mean - unp_std, color=self.unp_color,
                    alpha=0.3, linewidth=0.0, edgecolor='none')
                # ax.axhline(y=0, color='black', alpha=0.4, linestyle='--', zorder=0, lw=0.75)
                ax.set_ylim(bounds[ic][0], bounds[ic][1])
                ax.set_xlim(0, 100)
                ax.grid(color='black', zorder=0, alpha=0.2, lw=0.5, ls='--')
                util.publication_spines(ax)

                if not ic and not iside:
                    ax.legend([h_exp, h_unp], ['experiment', 'unperturbed'],
                        fancybox=False, frameon=True)

                if ic == len(coordinates)-1:
                    ax.set_xlabel('time (% gait cycle)')
                    ax.spines['bottom'].set_position(('outward', 10))
                else:
                    ax.spines['bottom'].set_visible(False)
                    ax.set_xticklabels([])
                    ax.tick_params(axis='x', which='both', bottom=False, 
                                top=False, labelbottom=False)

                if not iside:
                    ax.spines['left'].set_position(('outward', 10))
                    ax.set_ylabel(f'{labels[ic]} (deg)')
                else:
                    ax.spines['left'].set_visible(False)
                    ax.set_yticklabels([])
                    ax.tick_params(axis='y', which='both', left=False, 
                                right=False, labelbottom=False)

        fig.tight_layout()
        fig.savefig(target[0], dpi=600)
        plt.close()


    def plot_unperturbed_grfs(self, file_dep, target):

        numSubjects = len(self.subjects)
        N = 101
        pgc = np.linspace(0, 100, N)
        labels = ['anterior-posterior', 'vertical', 'medio-lateral']
        forces = ['vx', 'vy', 'vz']
        bounds = [[-0.2, 0.2],
                  [0, 1.25],
                  [-0.1, 0.1]]
        tick_intervals = [0.1, 0.25, 0.05]
        unperturbed_dict = dict()
        experiment_dict = dict()
        for force in forces:
            unperturbed_dict[f'ground_force_r_{force}'] = np.zeros((N, numSubjects))
            experiment_dict[f'ground_force_r_{force}'] = np.zeros((N, numSubjects))

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
                label = f'ground_force_r_{force}'
                unperturbed_col = util.simtk2numpy(unperturbed.getDependentColumn(label)) / BW
                experiment_col = util.simtk2numpy(
                    experiment.getDependentColumn(label))[istart:iend+1] / BW

                unperturbed_dict[label][:, isubj] = np.interp(
                    utime_interp, utime, unperturbed_col)
                experiment_dict[label][:, isubj] = np.interp(
                    etime_interp, etime[istart:iend+1], experiment_col)

        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(4, 7))
        gs = gridspec.GridSpec(len(forces), 1)
        for iforce, force in enumerate(forces):
            label = f'ground_force_r_{force}'
            ax = fig.add_subplot(gs[iforce, 0])

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
            # ax.axhline(y=0, color='black', alpha=0.4, linestyle='--', zorder=0, lw=0.75)
            ax.set_ylim(bounds[iforce][0], bounds[iforce][1])
            ax.set_yticks(np.linspace(bounds[iforce][0], bounds[iforce][1], 
                int((bounds[iforce][1] - bounds[iforce][0]) / tick_intervals[iforce]) + 1))
            ax.set_xlim(0, 100)
            ax.grid(color='black', zorder=0, alpha=0.2, lw=0.5, ls='--')
            util.publication_spines(ax)

            if not iforce:
                ax.legend([h_exp, h_unp], ['experiment', 'unperturbed'],
                    fancybox=False, frameon=True)

            if iforce == len(forces)-1:
                ax.set_xlabel('time (% gait cycle)')
                ax.spines['bottom'].set_position(('outward', 10))
            else:
                ax.spines['bottom'].set_visible(False)
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', bottom=False, 
                            top=False, labelbottom=False)

            ax.spines['left'].set_position(('outward', 10))
            ax.set_ylabel(f'{labels[iforce]} (BW)')

        fig.tight_layout()
        fig.savefig(target[0], dpi=600)
        plt.close()

        fig = plt.figure(figsize=(4, 2))
        new_forces = list()
        new_forces.append(forces[1])
        gs = gridspec.GridSpec(len(new_forces), 1)
        for iforce, force in enumerate(new_forces):
            label = f'ground_force_r_{force}'
            ax = fig.add_subplot(gs[iforce, 0])
            
            start_index = 0
            end_index = 70
            unp_mean = np.mean(unperturbed_dict[label], axis=1)[start_index:end_index+1]
            unp_std = np.std(unperturbed_dict[label], axis=1)[start_index:end_index+1]

            h_unp, = ax.plot(pgc[start_index:end_index+1], unp_mean, color='black', lw=2)
            ax.fill_between(pgc[start_index:end_index+1], 
                unp_mean + unp_std, unp_mean - unp_std, color='black',
                alpha=0.3, linewidth=0.0, edgecolor='none')
            # ax.axhline(y=0, color='black', alpha=0.4, linestyle='--', zorder=0, lw=0.75)
            ax.set_ylim(bounds[1][0], bounds[1][1])
            ax.set_yticks(np.linspace(bounds[1][0], bounds[1][1], 
                int((bounds[1][1] - bounds[1][0]) / tick_intervals[1]) + 1))
            ax.set_xlim(start_index, end_index)
            ax.grid(color='black', zorder=0, alpha=0.2, lw=0.5, ls='--')
            ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
            ax.set_xticklabels([0, 10, 20, 30, 40, 50, 60, 70])
            util.publication_spines(ax)

            # zipped = zip(self.times, self.time_colors)
            # for i, (time, color) in enumerate(zipped):
            #     ax.axvline(x=time, color=color, alpha=1.0, 
            #         linestyle='--', lw=1.5)

            # if iforce == len(forces[:-1])-1:
            ax.set_xlabel('time (% gait cycle)')
            ax.spines['bottom'].set_position(('outward', 10))
                # zipped = zip(self.times, self.time_colors)
                # for i, (time, color) in enumerate(zipped):
                #     ax.get_xticklabels()[i+5].set_color(color)
            # else:
            #     ax.spines['bottom'].set_visible(False)
            #     ax.set_xticklabels([])
            #     ax.tick_params(axis='x', which='both', bottom=False, 
            #                 top=False, labelbottom=False)

            ax.spines['left'].set_position(('outward', 10))
            ax.set_ylabel(f'{labels[1]} (BW)')

        fig.tight_layout()
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
            unperturbed_dict[f'{activation}_r'] = np.zeros((N, numSubjects))
            experiment_dict[f'{activation}_r'] = np.zeros((N, numSubjects))

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
                label = f'{activation}_r'
                unperturbed_col = util.simtk2numpy(unperturbed.getDependentColumn(
                    f'/forceset/{label}/activation'))
                experiment_col = util.simtk2numpy(
                    experiment.getDependentColumn(self.emg_map[label]))[istart:iend+1]

                max_emg = np.max(experiment_col)
                experiment_col_temp = experiment_col - 0.05
                experiment_col_rescale = experiment_col_temp * (max_emg / np.max(experiment_col_temp))

                unperturbed_dict[label][:, isubj] = np.interp(
                    utime_interp, utime, unperturbed_col)
                experiment_dict[label][:, isubj] = np.interp(
                    etime_interp, etime[istart:iend+1], experiment_col_rescale)

        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(4, 12))
        gs = gridspec.GridSpec(len(activations), 1)
        for iact, activation in enumerate(activations):
            label = f'{activation}_r'
            ax = fig.add_subplot(gs[iact, 0])
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
            # ax.axhline(y=0, color='black', alpha=0.4, linestyle='--', zorder=0, lw=0.75)
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.5, 1])
            ax.set_xlim(0, 100)
            ax.grid(color='black', zorder=0, alpha=0.2, lw=0.5, ls='--')
            util.publication_spines(ax)

            if not iact:
                ax.legend([h_exp, h_unp], ['experiment', 'unperturbed'],
                    fancybox=False, frameon=True)

            if iact == len(activations)-1:
                ax.set_xlabel('time (% gait cycle)')
                ax.spines['bottom'].set_position(('outward', 10))
            else:
                ax.spines['bottom'].set_visible(False)
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', bottom=False, 
                            top=False, labelbottom=False)

            ax.spines['left'].set_position(('outward', 10))
            ax.set_ylabel(f'{labels[iact]}')

        fig.tight_layout()
        fig.savefig(target[0], dpi=600)
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
        ax.set_xlabel('medio-lateral position (m)')
        ax.set_ylabel('anterior-posterior position (m)')

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
        for i in np.arange(numSubjects):

            time_r = 22
            time_l = 72
            pos_z_r = unperturbed_dict['pos_z_r'][time_r, i]
            pos_z_l = unperturbed_dict['pos_z_l'][time_l, i]
            width = pos_z_r - pos_z_l

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


# Perturbed walking
# -----------------

class TaskMocoPerturbedWalking(osp.TrialTask):
    REGISTRY = []
    def __init__(self, trial, initial_time, final_time, right_strikes, 
                 left_strikes, unperturbed_fpath=None, walking_speed=1.25,
                 side='right', torque_parameters=[0.5, 0.5, 0.25, 0.1],
                 subtalar_torque_perturbation=False, subtalar_peak_torque=0):
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
        self.name = f'{trial.subject.name}_moco_{self.config_name}'
        self.walking_speed = walking_speed
        self.mesh_interval = 0.01
        self.unperturbed_fpath = unperturbed_fpath
        self.root_dir = trial.study.config['doit_path']
        self.weights = trial.study.weights
        self.initial_time = initial_time
        self.final_time = final_time
        self.right_strikes = right_strikes
        self.left_strikes = left_strikes
        self.model_fpath = trial.subject.sim_model_fpath #.replace('final', 'temp')
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
            self.study.config['results_path'], self.config_name, 
            trial.subject.name)
        if not os.path.exists(self.result_fpath): 
            os.makedirs(self.result_fpath)

        self.archive_fpath = os.path.join(
            self.study.config['results_path'], self.config_name, 
            trial.subject.name, 'archive')
        if not os.path.exists(self.archive_fpath): 
            os.makedirs(self.archive_fpath)

        self.solution_fpath = os.path.join(self.result_fpath, 
            f'{self.config_name}.sto')

        self.unperturbed_result_fpath = os.path.join(
            self.study.config['results_path'], 'unperturbed', 
            trial.subject.name)
        shutil.copyfile(
            os.path.join(self.unperturbed_result_fpath, 
                'unperturbed_experiment_states.sto'),
            os.path.join(self.result_fpath, 
                'unperturbed_experiment_states.sto'))
        shutil.copyfile(
            os.path.join(self.unperturbed_result_fpath, 
                'unperturbed_experiment_states.sto'),
            os.path.join(self.result_fpath, 
                f'{self.config_name}_experiment_states.sto'))

        self.add_action([self.model_fpath,
                         self.tracking_coordinates_fpath,
                         self.coordinates_std_fpath,
                         self.tracking_extloads_fpath,
                         self.tracking_grfs_fpath,
                         self.emg_fpath,
                         self.unperturbed_fpath], 
                         [self.solution_fpath],
                        self.run_tracking_problem)

    def run_tracking_problem(self, file_dep, target):

        config = TimeSteppingConfig(
            self.config_name, self.config_name, 'black', self.weights,
            unperturbed_fpath=file_dep[6],
            ankle_torque_perturbation=True,
            ankle_torque_parameters=self.ankle_torque_parameters,
            subtalar_torque_perturbation=self.subtalar_torque_perturbation,
            subtalar_peak_torque=self.subtalar_peak_torque)

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
        self.ankle_torque_parameters = \
            generate_task.ankle_torque_parameters
        self.initial_time = generate_task.initial_time
        self.final_time = generate_task.final_time
        self.right_strikes = generate_task.right_strikes
        self.left_strikes = generate_task.left_strikes
        self.config_name = generate_task.config_name
        self.subtalar_torque_perturbation = generate_task.subtalar_torque_perturbation
        self.subtalar_peak_torque = generate_task.subtalar_peak_torque

        # Copy over unperturbed solution so we can plot against the
        # perturbed solution
        self.unperturbed_result_fpath = os.path.join(
            self.study.config['results_path'], 'unperturbed', 
            trial.subject.name)
        shutil.copyfile(
            os.path.join(self.unperturbed_result_fpath, 'unperturbed.sto'),
            os.path.join(self.result_fpath, 'unperturbed.sto'))
        shutil.copyfile(
            os.path.join(self.unperturbed_result_fpath, 
                         'unperturbed_grfs.sto'),
            os.path.join(self.result_fpath, 
                         'unperturbed_grfs.sto'))

        self.output_fpath = os.path.join(self.result_fpath, 
            f'tracking_walking_unperturbed_{self.config_name}_report.pdf')
        
        self.add_action([self.model_fpath,
                         self.tracking_coordinates_fpath,
                         self.coordinates_std_fpath,
                         self.tracking_extloads_fpath,
                         self.tracking_grfs_fpath,
                         self.emg_fpath,
                         self.unperturbed_fpath],
                        [self.output_fpath],
                        self.run_tracking_problem)

    def run_tracking_problem(self, file_dep, target):

        configs = list()
        config = TimeSteppingConfig(
            'unperturbed', 'unperturbed', 'black', self.weights)
        configs.append(config)

        config = TimeSteppingConfig(
            self.config_name, self.config_name, 'red', self.weights,
            ankle_torque_perturbation=True,
            ankle_torque_parameters=self.ankle_torque_parameters,
            subtalar_torque_perturbation=self.subtalar_torque_perturbation,
            subtalar_peak_torque=self.subtalar_peak_torque,
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
                    f'perturbed_{label}', subject, 
                    f'model_perturbed_{label}.osim'))
                perturbed_fpaths.append(os.path.join(
                    self.study.config['results_path'], 
                    f'perturbed_{label}', subject, 
                    f'perturbed_{label}.sto'))
                torque_fpaths.append(
                    os.path.join(self.study.config['results_path'], 
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

# Analysis
# --------

class TaskPlotAnkleTorques(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subject, time, torques, rise, fall, color):
        super(TaskPlotAnkleTorques, self).__init__(study)
        self.name = f'plot_ankle_torques_time{time}_rise{rise}_fall{fall}'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'perturbation_torques', 
            f'time{time}_rise{rise}_fall{fall}_{subject}')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)

        self.subject = subject
        self.time = time
        self.torques = torques
        self.rise = rise
        self.fall = fall
        self.color = color

        self.alphas = list()
        deps = list()

        for torque in self.torques:
            label = (f'torque{torque}_time{self.time}'
                     f'_rise{self.rise}_fall{self.fall}')
            deps.append(
                os.path.join(self.study.config['results_path'], 
                    f'perturbed_{label}', subject, 
                    f'perturbed_{label}.sto'))
            deps.append(
                os.path.join(self.study.config['results_path'], 
                    f'perturbed_{label}', subject, 
                    'ankle_perturbation_curve_right.sto'))
            self.alphas.append(torque / study.torques[-1])

        # Model 
        self.model = os.path.join(self.study.config['results_path'], 
            'unperturbed', subject, 'model_unperturbed.osim')

        self.add_action(deps, 
                        [os.path.join(self.analysis_path, 
                            'ankle_perturbation_torques.png')], 
                        self.plot_ankle_perturbation_torques)

    def plot_ankle_perturbation_torques(self, file_dep, target):

        fig = plt.figure(figsize=(5.5, 2.5))
        gs = fig.add_gridspec(1, 3)
        ax_l = fig.add_subplot(gs[0, :-1])
        ax_r = fig.add_subplot(gs[0, 2])
        lw = 2.5

        model = osim.Model(self.model)
        state = model.initSystem()
        mass = model.getTotalMass(state)
        
        for i, alpha in enumerate(self.alphas):
            torqueTable = osim.TimeSeriesTable(file_dep[2*i + 1])
            time = np.array(torqueTable.getIndependentColumn())
            pgc = 100 * (time - time[0]) / (time[-1] - time[0])
            torque = torqueTable.getDependentColumn(
                '/forceset/perturbation_ankle_angle_r').to_numpy()

            ax_l.plot(pgc, -torque, color=self.color, 
                alpha=alpha, linewidth=lw, clip_on=False,
                solid_capstyle='round')
            ax_l.set_ylabel('perturbation torque (N-m/kg)')
            ax_l.set_ylim([0.0, 0.25])
            ax_l.set_yticks([0.0, 0.25])
            ax_l.set_xlim(0, 100)
            util.publication_spines(ax_l, True)
            ax_l.set_xlabel('time (% gait cycle)')
            onset_time = self.time - 10
            peak_time = self.time
            offset_time = self.time + 5
            ax_l.fill_betweenx([-5, 5], onset_time, offset_time, alpha=0.1, 
                color='gray', edgecolor=None, zorder=0, lw=None)
            ax_l.axvline(x=peak_time, color=self.color, linestyle='--',
                linewidth=0.4, alpha=0.8, zorder=0) 

        ax_r.spines['right'].set_visible(False)
        ax_r.spines['top'].set_visible(False)
        ax_r.spines['bottom'].set_visible(False)
        ax_r.spines['left'].set_visible(False)
        ax_r.tick_params(axis='x', which='both', bottom=False, 
                         top=False, labelbottom=False)
        ax_r.tick_params(axis='y', which='both', left=False, 
                         top=False, labelleft=False)

        fig.tight_layout()
        fig.savefig(target[0], dpi=600)
        plt.close()


class TaskPlotCenterOfMass(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, times, torques, rise, fall,
                 ):
        super(TaskPlotCenterOfMass, self).__init__(study)
        suffix = f'rise{rise}_fall{fall}'
        if len(times) == 1:
            suffix += f'_time{times[0]}'
        self.name = f'plot_center_of_mass_{suffix}'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'center_of_mass', suffix)
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)
        self.subjects = subjects
        self.times = times
        self.torques = torques
        self.rise = rise
        self.fall = fall
        self.subtalars = study.subtalar_suffixes
        self.subtalar_colors = study.subtalar_colors

        self.labels = list()
        self.alphas = list()
        self.colors = list()
        self.linewidths = list()

        self.labels.append('unperturbed')
        self.alphas.append(1.0)
        self.colors.append('gray')
        self.linewidths.append(1)
        deps = list()

        for isubj, subject in enumerate(subjects):
            # Unperturbed solution
            deps.append(
                os.path.join(
                    self.study.config['results_path'], 
                    'unperturbed', subject,
                    'center_of_mass_unperturbed.sto'))

            # Perturbed solutions
            for time in self.times:
                for torque in self.torques:
                    for subtalar, color in zip(self.subtalars, self.subtalar_colors):
                        label = (f'perturbed_torque{torque}_time{time}'
                                 f'_rise{self.rise}_fall{self.fall}{subtalar}')
                        deps.append(
                            os.path.join(
                                self.study.config['results_path'], 
                                label, subject,
                                f'center_of_mass_{label}.sto')
                            )

                        if not isubj:
                            self.labels.append(label)
                            self.alphas.append(torque / study.torques[-1])
                            self.linewidths.append(1)
                            self.colors.append(color)

        targets = [os.path.join(self.analysis_path, 
                    'com_position_APvsML.png'), 
                   os.path.join(self.analysis_path, 
                    'com_position_SIvsAP.png'),
                   os.path.join(self.analysis_path, 
                    'com_velocity_APvsML.png'), 
                   os.path.join(self.analysis_path, 
                    'com_velocity_SIvsAP.png'),
                   os.path.join(self.analysis_path, 
                    'com_acceleration_APvsML.png'), 
                   os.path.join(self.analysis_path, 
                    'com_acceleration_SIvsAP.png')]

        self.add_action(deps, targets, self.plot_com_tracking_errors)

    def plot_com_tracking_errors(self, file_dep, target):

        # Initialize figures
        # ------------------
        numTorques = len(self.torques)
        numTimes = len(self.times)
        size = 2.5
        fig0 = plt.figure(figsize=(size*numTorques, size*numTimes))
        fig1 = plt.figure(figsize=(size*numTorques, size*numTimes))
        fig2 = plt.figure(figsize=(size*numTorques, size*numTimes))
        fig3 = plt.figure(figsize=(size*numTorques, size*numTimes))
        fig4 = plt.figure(figsize=(size*numTorques, size*numTimes))
        fig5 = plt.figure(figsize=(size*numTorques, size*numTimes))
        ax_xz_pos = list()
        ax_xy_pos = list()
        ax_xz_vel = list()
        ax_xy_vel = list()
        ax_xz_acc = list()
        ax_xy_acc = list()
        for itime in np.arange(numTimes):
            this_xz_pos = list()
            this_xy_pos = list()
            this_xz_vel = list()
            this_xy_vel = list()
            this_xz_acc = list()
            this_xy_acc = list()
            for itorque in np.arange(numTorques):
                i = numTorques*itime + itorque + 1
                this_xz_pos.append(fig0.add_subplot(numTimes, numTorques, i))
                this_xy_pos.append(fig1.add_subplot(numTimes, numTorques, i))
                this_xz_vel.append(fig2.add_subplot(numTimes, numTorques, i))
                this_xy_vel.append(fig3.add_subplot(numTimes, numTorques, i))
                this_xz_acc.append(fig4.add_subplot(numTimes, numTorques, i))
                this_xy_acc.append(fig5.add_subplot(numTimes, numTorques, i))

            ax_xz_pos.append(this_xz_pos)
            ax_xy_pos.append(this_xy_pos)
            ax_xz_vel.append(this_xz_vel)
            ax_xy_vel.append(this_xy_vel)
            ax_xz_acc.append(this_xz_acc)
            ax_xy_acc.append(this_xy_acc)

        axes = list()
        axes.append(ax_xz_pos)
        axes.append(ax_xy_pos)
        axes.append(ax_xz_vel)
        axes.append(ax_xy_vel)
        axes.append(ax_xz_acc)
        axes.append(ax_xy_acc)

        # Aggregate data
        # --------------
        numLabels = len(self.labels)
        import collections
        com_dict = collections.defaultdict(dict)
        time_dict = dict()
        for isubj, subject in enumerate(self.subjects):
            # Unperturbed center-of-mass trajectories
            unpTable = osim.TimeSeriesTable(file_dep[isubj*numLabels])
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
            time_dict[subject] = unpTimeVec

            for i, label in enumerate(self.labels):
                # Perturbed center-of-mass trajectories
                table = osim.TimeSeriesTable(file_dep[i + isubj*numLabels])
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

                com_dict[subject][label] = table_np - unpTable_np[0:N, :]
               
        # Plotting
        # --------
        def update_lims(data, step, lim):
            if np.min(data) < lim[0]:
                lim[0] = step * np.floor(np.min(data) / step)
            if np.max(data) > lim[1]:
                lim[1] = step * np.ceil(np.max(data) / step)

        def get_ticks_from_lims(lims, interval):
            N = int(np.around((lims[1] - lims[0]) / interval, decimals=3)) + 1
            ticks = np.linspace(lims[0], lims[1], N)
            return ticks

        pos_step = 0.05 # cm
        vel_step = 0.01
        acc_step = 0.1

        pos_xz_xlim = [[0.0, 0.0] for i in np.arange(len(self.torques))]
        pos_xz_ylim = [[0.0, 0.0] for i in np.arange(len(self.times))]
        vel_xz_xlim = [[0.0, 0.0] for i in np.arange(len(self.torques))]
        vel_xz_ylim = [[0.0, 0.0] for i in np.arange(len(self.times))]
        acc_xz_xlim = [[0.0, 0.0] for i in np.arange(len(self.torques))]
        acc_xz_ylim = [[0.0, 0.0] for i in np.arange(len(self.times))]

        pos_xy_xlim = [[0.0, 0.0] for i in np.arange(len(self.torques))]
        pos_xy_ylim = [[0.0, 0.0] for i in np.arange(len(self.times))]
        vel_xy_xlim = [[0.0, 0.0] for i in np.arange(len(self.torques))]
        vel_xy_ylim = [[0.0, 0.0] for i in np.arange(len(self.times))]
        acc_xy_xlim = [[0.0, 0.0] for i in np.arange(len(self.torques))]
        acc_xy_ylim = [[0.0, 0.0] for i in np.arange(len(self.times))]
        for itime, time in enumerate(self.times):
            for itorque, torque in enumerate(self.torques):
                for subtalar, color in zip(self.subtalars, self.subtalar_colors):
                    label = (f'perturbed_torque{torque}_time{time}'
                             f'_rise{self.rise}_fall{self.fall}{subtalar}')
                    for isubj, subject in enumerate(self.subjects):
                        com = com_dict[subject][label]

                        # Compute the closet time index to the current peak 
                        # perturbation time. 
                        #
                        # TODO: A given peak perturbation time (e.g 50% of the 
                        # gait cycle) may not lie exactly on a time point of 
                        # from the simulation. The interval between time points
                        # is 5ms, meaning that the time index could be up to 2.5ms
                        # away from the actual perturbation peak time. 
                        timeVec = np.array(time_dict[subject])
                        duration = timeVec[-1] - timeVec[0]
                        time_at_rise = timeVec[0] + (duration * ((time - self.rise) / 100.0))
                        time_at_peak = timeVec[0] + (duration * (time / 100.0))

                        irise = np.argmin(np.abs(timeVec - time_at_rise))
                        ipeak = np.argmin(np.abs(timeVec - time_at_peak))

                        pos_x = 100 * com[irise:ipeak+1, 0]
                        pos_y = 100 * com[irise:ipeak+1, 1]
                        pos_z = 100 * com[irise:ipeak+1, 2]
                        vel_x = com[irise:ipeak+1, 3]
                        vel_y = com[irise:ipeak+1, 4]
                        vel_z = com[irise:ipeak+1, 5]
                        acc_x = com[irise:ipeak+1, 6]
                        acc_y = com[irise:ipeak+1, 7]
                        acc_z = com[irise:ipeak+1, 8]

                        update_lims(pos_z, pos_step, pos_xz_xlim[itorque])
                        update_lims(pos_x, pos_step, pos_xz_ylim[itime])
                        
                        update_lims(pos_x, pos_step, pos_xy_xlim[itorque])
                        update_lims(pos_y, pos_step, pos_xy_ylim[itime])

                        update_lims(vel_z, vel_step, vel_xz_xlim[itorque])
                        update_lims(vel_x, vel_step, vel_xz_ylim[itime])
                        
                        update_lims(vel_x, vel_step, vel_xy_xlim[itorque])
                        update_lims(vel_y, vel_step, vel_xy_ylim[itime])

                        update_lims(acc_z, acc_step, acc_xz_xlim[itorque])
                        update_lims(acc_x, acc_step, acc_xz_ylim[itime])
                        
                        update_lims(acc_x, acc_step, acc_xy_xlim[itorque])
                        update_lims(acc_y, acc_step, acc_xy_ylim[itime])
                        
                        lw = 1.25
              
                        # Position: anterior-posterior vs medio-lateral
                        ax_xz_pos[itime][itorque].plot(pos_z, pos_x, color=color, lw=lw,
                            zorder=2,solid_capstyle='round', clip_on=False)
                        
                        # Position: superior-inferior vs anterior-posterior
                        ax_xy_pos[itime][itorque].plot(pos_x, pos_y, color=color, lw=lw, 
                            zorder=2, solid_capstyle='round', clip_on=False)

                        # Velocity: anterior-posterior vs medio-lateral
                        ax_xz_vel[itime][itorque].plot(vel_z, vel_x, color=color, lw=lw, 
                            zorder=2,solid_capstyle='round', clip_on=False)

                        # Velocity: superior-inferior vs anterior-posterior
                        ax_xy_vel[itime][itorque].plot(vel_x, vel_y, color=color, lw=lw, 
                            zorder=2, solid_capstyle='round', clip_on=False)

                        # Acceleration: anterior-posterior vs medio-lateral
                        ax_xz_acc[itime][itorque].plot(acc_z, acc_x, color=color, lw=lw, 
                            zorder=2, solid_capstyle='round', clip_on=False)
                        
                        # Acceleration: superior-inferior vs anterior-posterior
                        ax_xy_acc[itime][itorque].plot(acc_x, acc_y, color=color, lw=lw, 
                            zorder=2, solid_capstyle='round', clip_on=False)
         

        for itime, time in enumerate(self.times):
            for itorque, torque in enumerate(self.torques):
                peak_torque = torque / 100.0
                for ax in axes:
                    ax[itime][itorque].grid(color='black', zorder=0, 
                        alpha=0.2, lw=0.4, ls='--', clip_on=False)
                    util.publication_spines(ax[itime][itorque], True)
                    if not itime:
                        if torque:
                            ax[itime][itorque].set_title(
                                f'peak ankle torque: {-peak_torque:1.2f} ' 
                                    + r'$[N \cdot m/kg]$', 
                                fontsize=8)
                        else:
                            ax[itime][itorque].set_title('zero ankle torque')

                    if itorque:
                        ax[itime][itorque].spines['left'].set_visible(False)
                        ax[itime][itorque].set_yticklabels([])
                        ax[itime][itorque].yaxis.set_ticks_position('none')

                    if not time == self.times[-1]:
                        ax[itime][itorque].spines['bottom'].set_visible(False)
                        ax[itime][itorque].set_xticklabels([])
                        ax[itime][itorque].xaxis.set_ticks_position('none')

                # Position: anterior-posterior vs medio-lateral (centimeters)
                ax_xz_pos[itime][itorque].set_xlim(pos_xz_xlim[itorque])
                ax_xz_pos[itime][itorque].set_xticks(get_ticks_from_lims(pos_xz_xlim[itorque], pos_step))
                ax_xz_pos[itime][itorque].set_ylim(pos_xz_ylim[itime])
                ax_xz_pos[itime][itorque].set_yticks(get_ticks_from_lims(pos_xz_ylim[itime], pos_step))
                if time == self.times[-1]:
                    ax_xz_pos[itime][itorque].set_xlabel(r'$\Delta$ medio-lateral position ($cm$)')
                if not itorque:
                    ax_xz_pos[itime][itorque].set_ylabel(r'$\Delta$ fore-aft position ($cm$)')

                # Position: superior-inferior vs anterior-posterior (centimeters)
                ax_xy_pos[itime][itorque].set_xlim(pos_xy_xlim[itorque])
                ax_xy_pos[itime][itorque].set_xticks(get_ticks_from_lims(pos_xy_xlim[itorque], pos_step))
                ax_xy_pos[itime][itorque].set_ylim(pos_xy_ylim[itime])
                ax_xy_pos[itime][itorque].set_yticks(get_ticks_from_lims(pos_xy_ylim[itime], pos_step))
                if time == self.times[-1]:
                    ax_xy_pos[itime][itorque].set_xlabel(r'$\Delta$ fore-aft position ($cm$)')
                if not itorque:
                    ax_xy_pos[itime][itorque].set_ylabel(r'$\Delta$ vertical position ($cm$)')

                # Velocity: anterior-posterior vs medio-lateral
                ax_xz_vel[itime][itorque].set_xlim(vel_xz_xlim[itorque])
                ax_xz_vel[itime][itorque].set_xticks(get_ticks_from_lims(vel_xz_xlim[itorque], vel_step))
                ax_xz_vel[itime][itorque].set_ylim(vel_xz_ylim[itime])
                ax_xz_vel[itime][itorque].set_yticks(get_ticks_from_lims(vel_xz_ylim[itime], vel_step))
                if time == self.times[-1]:
                    ax_xz_vel[itime][itorque].set_xlabel(r'$\Delta$ medio-lateral velocity ($m/s$)')
                if not itorque:
                    ax_xz_vel[itime][itorque].set_ylabel(r'$\Delta$ fore-aft velocity ($m/s$)')

                # Velocity: superior-inferior vs anterior-posterior
                ax_xy_vel[itime][itorque].set_xlim(vel_xy_xlim[itorque])
                ax_xy_vel[itime][itorque].set_xticks(get_ticks_from_lims(vel_xy_xlim[itorque], vel_step))
                ax_xy_vel[itime][itorque].set_ylim(vel_xy_ylim[itime])
                ax_xy_vel[itime][itorque].set_yticks(get_ticks_from_lims(vel_xy_ylim[itime], vel_step))
                if time == self.times[-1]:
                    ax_xy_vel[itime][itorque].set_xlabel(r'$\Delta$ fore-aft velocity ($m/s$)')
                if not itorque:
                    ax_xy_vel[itime][itorque].set_ylabel(r'$\Delta$ vertical velocity ($m/s$)')

                # Acceleration: anterior-posterior vs medio-lateral
                ax_xz_acc[itime][itorque].set_xlim(acc_xz_xlim[itorque])
                ax_xz_acc[itime][itorque].set_xticks(get_ticks_from_lims(acc_xz_xlim[itorque], acc_step))
                ax_xz_acc[itime][itorque].set_ylim(acc_xz_ylim[itime])
                ax_xz_acc[itime][itorque].set_yticks(get_ticks_from_lims(acc_xz_ylim[itime], acc_step))
                if time == self.times[-1]:
                    ax_xz_acc[itime][itorque].set_xlabel(r'$\Delta$ medio-lateral acceleration ($m/s^2$)')
                if not itorque:
                    ax_xz_acc[itime][itorque].set_ylabel(r'$\Delta$ fore-aft acceleration ($m/s^2$)')

                # Acceleration: superior-inferior vs anterior-posterior
                ax_xy_acc[itime][itorque].set_xlim(acc_xy_xlim[itorque])
                ax_xy_acc[itime][itorque].set_xticks(get_ticks_from_lims(acc_xy_xlim[itorque], acc_step))
                ax_xy_acc[itime][itorque].set_ylim(acc_xy_ylim[itime])
                ax_xy_acc[itime][itorque].set_yticks(get_ticks_from_lims(acc_xy_ylim[itime], acc_step))
                if time == self.times[-1]:
                    ax_xy_acc[itime][itorque].set_xlabel(r'$\Delta$ fore-aft acceleration ($m/s^2$)')
                if not itorque:
                    ax_xy_acc[itime][itorque].set_ylabel(r'$\Delta$ vertical acceleration ($m/s^2$)')
        
        fig0.tight_layout()
        fig0.savefig(target[0], dpi=600)
        plt.close()
        fig1.tight_layout()
        fig1.savefig(target[1], dpi=600)
        plt.close()
        fig2.tight_layout()
        fig2.savefig(target[2], dpi=600)
        plt.close()
        fig3.tight_layout()
        fig3.savefig(target[3], dpi=600)
        plt.close()
        fig4.tight_layout()
        fig4.savefig(target[4], dpi=600)
        plt.close()
        fig5.tight_layout()
        fig5.savefig(target[5], dpi=600)
        plt.close()


class TaskPlotInstantaneousCenterOfMass(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, times, torque, rise, fall):
        super(TaskPlotInstantaneousCenterOfMass, self).__init__(study)
        self.name = f'plot_instantaneous_center_of_mass_torque{torque}_rise{rise}_fall{fall}'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'center_of_mass_instantaneous',  f'torque{torque}_rise{rise}_fall{fall}')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)
        self.subjects = subjects
        self.torque = torque
        self.times = times
        self.rise = rise
        self.fall = fall
        self.subtalars = study.subtalar_suffixes
        self.subtalar_colors = study.subtalar_colors
        self.width = 0.2
        N = len(study.subtalar_peak_torques)
        min_width = -self.width*((N-1)/2)
        max_width = -min_width
        self.subtalar_shifts = np.linspace(min_width, max_width, N)
        self.labels = list()
        self.times_list = list()

        self.labels.append('unperturbed')
        self.times_list.append(100)
        deps = list()

        for isubj, subject in enumerate(subjects):
            # Unperturbed solution
            deps.append(
                os.path.join(
                    self.study.config['results_path'], 
                    'unperturbed', subject,
                    'center_of_mass_unperturbed.sto'))

            # Perturbed solutions
            for time in self.times:
                for subtalar in self.subtalars:
                    label = (f'perturbed_torque{torque}_time{time}'
                            f'_rise{self.rise}_fall{self.fall}{subtalar}')
                    deps.append(
                        os.path.join(
                            self.study.config['results_path'], 
                            label, subject,
                            f'center_of_mass_{label}.sto')
                        )

                    if not isubj:
                        self.labels.append(
                            (f'torque{self.torque}_time{time}'
                             f'_rise{self.rise}_fall{self.fall}{subtalar}'))
                        self.times_list.append(time)

        targets = list()
        for kin in ['pos', 'vel', 'acc']:
            for direc in ['AP', 'SI', 'ML']:
                targets += [os.path.join(self.analysis_path, 
                            f'instant_com_{direc}{kin}.png')]

        self.add_action(deps, targets, self.plot_instantaneous_com)

    def plot_instantaneous_com(self, file_dep, target):

        # Initialize figures
        # ------------------
        figs = list()
        axes = list()
        for kin in ['pos', 'vel', 'acc']:
            for direc in ['AP', 'SI', 'ML']:
                fig = plt.figure(figsize=(5, 2.5))
                ax = fig.add_subplot(1, 1, 1)
                figs.append(fig)
                axes.append(ax)

        # Aggregate data
        # --------------
        numLabels = len(self.labels)
        import collections
        com_dict = collections.defaultdict(dict)
        time_dict = dict()
        for isubj, subject in enumerate(self.subjects):
            # Unperturbed center-of-mass trajectories
            unpTable = osim.TimeSeriesTable(file_dep[isubj*numLabels])
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
            time_dict[subject] = unpTimeVec

            for i, (label, time) in enumerate(zip(self.labels, self.times_list)):
                # Perturbed center-of-mass trajectories
                table = osim.TimeSeriesTable(file_dep[i + isubj*numLabels])
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
                com_dict[subject][label] = table_np - unpTable_np[0:N, :]

        # Plotting
        # --------
        def update_lims(data, step, lim):
            if np.min(data) < lim[0]:
                lim[0] = step * np.floor(np.min(data) / step)
                lim[1] = -lim[0]
            if np.max(data) > lim[1]:
                lim[1] = step * np.ceil(np.max(data) / step)
                lim[0] = -lim[1]

        def get_ticks_from_lims(lims, interval):
            N = int(np.around((lims[1] - lims[0]) / interval, decimals=3)) + 1
            ticks = np.linspace(lims[0], lims[1], N)
            return ticks

        def plot_errorbar(ax, x, y, yerr):
            lolims = y > 0
            uplims = y < 0
            ple, cle, ble = ax.errorbar(x, y, yerr=yerr, 
                color=color, fmt='none', ecolor='black', 
                capsize=0, solid_capstyle='projecting', lw=0.25, 
                zorder=0, clip_on=False, lolims=lolims, uplims=uplims,
                elinewidth=0.4, markeredgewidth=0.4)
            for cl in cle:
                cl.set_marker('_')
                cl.set_markersize(2)

        pos_x_diff = np.zeros(len(self.subjects))
        pos_y_diff = np.zeros(len(self.subjects))
        pos_z_diff = np.zeros(len(self.subjects))
        vel_x_diff = np.zeros(len(self.subjects))
        vel_y_diff = np.zeros(len(self.subjects))
        vel_z_diff = np.zeros(len(self.subjects))
        acc_x_diff = np.zeros(len(self.subjects))
        acc_y_diff = np.zeros(len(self.subjects))
        acc_z_diff = np.zeros(len(self.subjects))
        pos_x_diff_mean = np.zeros((len(self.times), len(self.subtalars)))
        pos_y_diff_mean = np.zeros((len(self.times), len(self.subtalars)))
        pos_z_diff_mean = np.zeros((len(self.times), len(self.subtalars)))
        vel_x_diff_mean = np.zeros((len(self.times), len(self.subtalars)))
        vel_y_diff_mean = np.zeros((len(self.times), len(self.subtalars)))
        vel_z_diff_mean = np.zeros((len(self.times), len(self.subtalars)))
        acc_x_diff_mean = np.zeros((len(self.times), len(self.subtalars)))
        acc_y_diff_mean = np.zeros((len(self.times), len(self.subtalars)))
        acc_z_diff_mean = np.zeros((len(self.times), len(self.subtalars)))
        pos_x_diff_std = np.zeros((len(self.times), len(self.subtalars)))
        pos_y_diff_std = np.zeros((len(self.times), len(self.subtalars)))
        pos_z_diff_std = np.zeros((len(self.times), len(self.subtalars)))
        vel_x_diff_std = np.zeros((len(self.times), len(self.subtalars)))
        vel_y_diff_std = np.zeros((len(self.times), len(self.subtalars)))
        vel_z_diff_std = np.zeros((len(self.times), len(self.subtalars)))
        acc_x_diff_std = np.zeros((len(self.times), len(self.subtalars)))
        acc_y_diff_std = np.zeros((len(self.times), len(self.subtalars)))
        acc_z_diff_std = np.zeros((len(self.times), len(self.subtalars)))
        for itime, time in enumerate(self.times):
            zipped = zip(self.subtalars, self.subtalar_colors, self.subtalar_shifts)
            for isubt, (subtalar, color, shift) in enumerate(zipped):

                label = (f'torque{self.torque}_time{time}'
                         f'_rise{self.rise}_fall{self.fall}{subtalar}')
                for isubj, subject in enumerate(self.subjects):
                    com = com_dict[subject][label]

                    # Compute the closet time index to the current peak 
                    # perturbation time. 
                    #
                    # TODO: A given peak perturbation time (e.g 50% of the 
                    # gait cycle) may not lie exactly on a time point of 
                    # from the simulation. The interval between time points
                    # is 5ms, meaning that the time index could be up to 2.5ms
                    # away from the actual perturbation peak time. 
                    timeVec = np.array(time_dict[subject])
                    duration = timeVec[-1] - timeVec[0]
                    time_at_peak = timeVec[0] + (duration * (time / 100.0))
                    index = np.argmin(np.abs(timeVec - time_at_peak))

                    pos_x_diff[isubj] = 100*com[index, 0]
                    pos_y_diff[isubj] = 100*com[index, 1]
                    pos_z_diff[isubj] = 100*com[index, 2]
                    vel_x_diff[isubj] = com[index, 3]
                    vel_y_diff[isubj] = com[index, 4]
                    vel_z_diff[isubj] = com[index, 5]
                    acc_x_diff[isubj] = com[index, 6]
                    acc_y_diff[isubj] = com[index, 7]
                    acc_z_diff[isubj] = com[index, 8]

                pos_x_diff_mean[itime, isubt] = np.mean(pos_x_diff)
                pos_y_diff_mean[itime, isubt] = np.mean(pos_y_diff)
                pos_z_diff_mean[itime, isubt] = np.mean(pos_z_diff)
                vel_x_diff_mean[itime, isubt] = np.mean(vel_x_diff)
                vel_y_diff_mean[itime, isubt] = np.mean(vel_y_diff)
                vel_z_diff_mean[itime, isubt] = np.mean(vel_z_diff)
                acc_x_diff_mean[itime, isubt] = np.mean(acc_x_diff)
                acc_y_diff_mean[itime, isubt] = np.mean(acc_y_diff)
                acc_z_diff_mean[itime, isubt] = np.mean(acc_z_diff)
                pos_x_diff_std[itime, isubt] = np.std(pos_x_diff)
                pos_y_diff_std[itime, isubt] = np.std(pos_y_diff)
                pos_z_diff_std[itime, isubt] = np.std(pos_z_diff)
                vel_x_diff_std[itime, isubt] = np.std(vel_x_diff)
                vel_y_diff_std[itime, isubt] = np.std(vel_y_diff)
                vel_z_diff_std[itime, isubt] = np.std(vel_z_diff)
                acc_x_diff_std[itime, isubt] = np.std(acc_x_diff)
                acc_y_diff_std[itime, isubt] = np.std(acc_y_diff)
                acc_z_diff_std[itime, isubt] = np.std(acc_z_diff)


        pos_step = 0.1 # cm
        vel_step = 0.01
        acc_step = 0.1
        pos_x_lim = [0.0, 0.0]
        pos_y_lim = [0.0, 0.0]
        pos_z_lim = [0.0, 0.0]
        vel_x_lim = [0.0, 0.0]
        vel_y_lim = [0.0, 0.0]
        vel_z_lim = [0.0, 0.0]
        acc_x_lim = [0.0, 0.0]
        acc_y_lim = [0.0, 0.0]
        acc_z_lim = [0.0, 0.0]
        update_lims(pos_x_diff_mean-pos_x_diff_std, pos_step, pos_x_lim)
        update_lims(pos_y_diff_mean-pos_y_diff_std, pos_step, pos_y_lim)
        update_lims(pos_z_diff_mean-pos_z_diff_std, pos_step, pos_z_lim)
        update_lims(vel_x_diff_mean-vel_x_diff_std, vel_step, vel_x_lim)
        update_lims(vel_y_diff_mean-vel_y_diff_std, vel_step, vel_y_lim)
        update_lims(vel_z_diff_mean-vel_z_diff_std, vel_step, vel_z_lim)
        update_lims(acc_x_diff_mean-acc_x_diff_std, acc_step, acc_x_lim)
        update_lims(acc_y_diff_mean-acc_y_diff_std, acc_step, acc_y_lim)
        update_lims(acc_z_diff_mean-acc_z_diff_std, acc_step, acc_z_lim)        
        update_lims(pos_x_diff_mean+pos_x_diff_std, pos_step, pos_x_lim)
        update_lims(pos_y_diff_mean+pos_y_diff_std, pos_step, pos_y_lim)
        update_lims(pos_z_diff_mean+pos_z_diff_std, pos_step, pos_z_lim)
        update_lims(vel_x_diff_mean+vel_x_diff_std, vel_step, vel_x_lim)
        update_lims(vel_y_diff_mean+vel_y_diff_std, vel_step, vel_y_lim)
        update_lims(vel_z_diff_mean+vel_z_diff_std, vel_step, vel_z_lim)
        update_lims(acc_x_diff_mean+acc_x_diff_std, acc_step, acc_x_lim)
        update_lims(acc_y_diff_mean+acc_y_diff_std, acc_step, acc_y_lim)
        update_lims(acc_z_diff_mean+acc_z_diff_std, acc_step, acc_z_lim)
        for itime, time in enumerate(self.times):
            zipped = zip(self.subtalars, self.subtalar_colors, self.subtalar_shifts)
            for isubt, (subtalar, color, shift) in enumerate(zipped):

                # Set the x-position for these bar chart entries.
                x = itime + shift

                # Instantaneous positions
                # -----------------------
                plot_errorbar(axes[0], x, pos_x_diff_mean[itime, isubt], pos_x_diff_std[itime, isubt])
                axes[0].bar(x, pos_x_diff_mean[itime, isubt], self.width, color=color, clip_on=False)
                axes[0].set_ylabel(r'$\Delta$' + ' fore-aft position $[cm]$')
                
                plot_errorbar(axes[1], x, pos_y_diff_mean[itime, isubt], pos_y_diff_std[itime, isubt])
                axes[1].bar(x, pos_y_diff_mean[itime, isubt], self.width, color=color, clip_on=False)
                axes[1].set_ylabel(r'$\Delta$' + ' vertical position $[cm]$')

                plot_errorbar(axes[2], x, pos_z_diff_mean[itime, isubt], pos_z_diff_std[itime, isubt])
                axes[2].bar(x, pos_z_diff_mean[itime, isubt], self.width, color=color, clip_on=False)
                axes[2].set_ylabel(r'$\Delta$' + ' medio-lateral position $[cm]$')

                # Instantaneous velocities
                # ------------------------
                plot_errorbar(axes[3], x, vel_x_diff_mean[itime, isubt], vel_x_diff_std[itime, isubt])
                axes[3].bar(x, vel_x_diff_mean[itime, isubt], self.width, color=color, clip_on=False)
                axes[3].set_ylabel(r'$\Delta$' + ' fore-aft velocity $[m/s]$')

                plot_errorbar(axes[4], x, vel_y_diff_mean[itime, isubt], vel_y_diff_std[itime, isubt])
                axes[4].bar(x, vel_y_diff_mean[itime, isubt], self.width, color=color, clip_on=False)
                axes[4].set_ylabel(r'$\Delta$' + ' vertical velocity $[m/s]$')

                plot_errorbar(axes[5], x, vel_z_diff_mean[itime, isubt], vel_z_diff_std[itime, isubt])
                axes[5].bar(x, vel_z_diff_mean[itime, isubt], self.width, color=color, clip_on=False)
                axes[5].set_ylabel(r'$\Delta$' + ' medio-lateral velocity $[m/s]$')

                # Instantaneous accelerations
                # ---------------------------
                plot_errorbar(axes[6], x, acc_x_diff_mean[itime, isubt], acc_x_diff_std[itime, isubt])
                axes[6].bar(x, acc_x_diff_mean[itime, isubt], self.width, color=color, clip_on=False)
                axes[6].set_ylabel(r'$\Delta$' + ' fore-aft acceleration $[m/s^2]$')

                plot_errorbar(axes[7], x, acc_y_diff_mean[itime, isubt], acc_y_diff_std[itime, isubt])
                axes[7].bar(x, acc_y_diff_mean[itime, isubt], self.width, color=color, clip_on=False)
                axes[7].set_ylabel(r'$\Delta$' + ' vertical acceleration $[m/s^2]$')

                plot_errorbar(axes[8], x, acc_z_diff_mean[itime, isubt], acc_z_diff_std[itime, isubt])
                axes[8].bar(x, acc_z_diff_mean[itime, isubt], self.width, color=color, clip_on=False)
                axes[8].set_ylabel(r'$\Delta$' + ' medio-lateral acceleration $[m/s^2]$')

        axes[0].set_ylim(pos_x_lim)
        axes[0].set_yticks(get_ticks_from_lims(pos_x_lim, pos_step))
        axes[1].set_ylim(pos_y_lim)
        axes[1].set_yticks(get_ticks_from_lims(pos_y_lim, pos_step))
        axes[2].set_ylim(pos_z_lim)
        axes[2].set_yticks(get_ticks_from_lims(pos_z_lim, pos_step))
        axes[3].set_ylim(vel_x_lim)
        axes[3].set_yticks(get_ticks_from_lims(vel_x_lim, vel_step))
        axes[4].set_ylim(vel_y_lim)
        axes[4].set_yticks(get_ticks_from_lims(vel_y_lim, vel_step))
        axes[5].set_ylim(vel_z_lim)
        axes[5].set_yticks(get_ticks_from_lims(vel_z_lim, vel_step))
        axes[6].set_ylim(acc_x_lim)
        axes[6].set_yticks(get_ticks_from_lims(acc_x_lim, acc_step))
        axes[7].set_ylim(acc_y_lim)
        axes[7].set_yticks(get_ticks_from_lims(acc_y_lim, acc_step))
        axes[8].set_ylim(acc_z_lim)
        axes[8].set_yticks(get_ticks_from_lims(acc_z_lim, acc_step))

        for ax in axes:
            ax.axhline(y=0, color='black', linestyle='-',
                    linewidth=0.1, alpha=1.0, zorder=-1)
            ax.spines['bottom'].set_position(('outward', 10))
            ax.spines['left'].set_position(('outward', 20))
            ax.set_xticks(np.arange(len(self.times)))
            ax.set_xlim(0, len(self.times)-1)
            ax.set_xticklabels([f'{time}' for time in self.times])
            ax.set_xlabel('peak perturbation time\n(% gait cycle)')
            util.publication_spines(ax)

        for ifig, fig in enumerate(figs):
            fig.tight_layout()
            fig.savefig(target[ifig], dpi=600)
            plt.close()


class TaskPlotCenterOfMassVector(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, times, torque, rise, fall):
        super(TaskPlotCenterOfMassVector, self).__init__(study)
        self.name = f'plot_center_of_mass_vector_torque{torque}_rise{rise}_fall{fall}'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'center_of_mass_vector',  f'torque{torque}_rise{rise}_fall{fall}')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)
        self.subjects = subjects
        self.torque = torque
        self.times = times
        self.rise = rise
        self.fall = fall
        self.subtalars = list(reversed(study.subtalar_suffixes))
        self.subtalar_colors = list(reversed(study.subtalar_colors))
        self.subtalar_peak_torques = list(
            reversed([torque / 100.0 for torque in study.subtalar_peak_torques]))
        self.width = 0.1
        self.subtalar_shifts = list()
        for i in reversed(np.arange(len(study.subtalar_peak_torques))):
            self.subtalar_shifts.append(-self.width*(i+1))
        self.subtalar_shifts.append(0)
        for i in np.arange(len(study.subtalar_peak_torques)):
            self.subtalar_shifts.append(self.width*(i+1))
        self.labels = list()
        self.times_list = list()

        self.labels.append('unperturbed')
        self.times_list.append(100)
        deps = list()

        for isubj, subject in enumerate(subjects):
            # Unperturbed solution
            deps.append(
                os.path.join(
                    self.study.config['results_path'], 
                    'unperturbed', subject,
                    'center_of_mass_unperturbed.sto'))

            # Perturbed solutions
            for time in self.times:
                for subtalar in self.subtalars:
                    label = (f'perturbed_torque{torque}_time{time}'
                            f'_rise{self.rise}_fall{self.fall}{subtalar}')
                    deps.append(
                        os.path.join(
                            self.study.config['results_path'], 
                            label, subject,
                            f'center_of_mass_{label}.sto')
                        )

                    if not isubj:
                        self.labels.append(
                            (f'torque{self.torque}_time{time}'
                             f'_rise{self.rise}_fall{self.fall}{subtalar}'))
                        self.times_list.append(time)

        targets = list()
        for kin in ['pos', 'vel', 'acc']:
            for plane in ['sagittal', 'coronal', 'frontal']:
                targets += [os.path.join(self.analysis_path, 
                            f'com_vector_{plane}_{kin}.png')]

        self.add_action(deps, targets, self.plot_com_vectors)

    def plot_com_vectors(self, file_dep, target):

        # Initialize figures
        # ------------------
        figs = list()
        axes = list()
        fontsize = 5.5
        linewidth = 0.5
        for kin in ['pos', 'vel', 'acc']:
            for plane in ['sagittal', 'coronal', 'frontal']:
                scale = 1.0
                fig = plt.figure(figsize=(3*scale, 2*scale))
                gs = fig.add_gridspec(2*len(self.subtalars), 10)
                sub_axes = list()
                for isub, peak_sub in enumerate(self.subtalar_peak_torques):
                    ax = fig.add_subplot(gs[(2*isub):(2*isub + 2), 0:8])
                    ax.axhline(y=0, color='black', linestyle='-',
                        linewidth=0.25, alpha=1.0, zorder=1,
                        clip_on=False, solid_capstyle='round')
                    ax.spines['bottom'].set_position(('outward', 5))
                    ax.spines['bottom'].set_linewidth(linewidth)
                    ax.spines['left'].set_position(('outward', 10))
                    ax.spines['left'].set_linewidth(linewidth)

                    ax.set_xticks(np.arange(len(self.times)))
                    xticklabels = list()
                    for itime, time in enumerate(self.times):
                        # if not itime % 2:
                        xticklabels.append(f'{time}')
                        # else:
                            # xticklabels.append('')
                    ax.set_xticklabels(xticklabels)
                    util.publication_spines(ax)
                    
                    ax.tick_params(axis='x', length=1.5, width=linewidth, 
                        labelsize=fontsize-1, pad=1.5)
                    ax.tick_params(axis='y', length=1.5, width=linewidth, 
                        labelsize=fontsize-2, pad=1.5)
                    margin = 0
                    ax.set_xlim(-margin, len(self.times)-1 + margin)

                    if isub == len(self.subtalars)-1:
                        ax.set_xlabel('peak perturbation time\n(% gait cycle)',
                            fontsize=fontsize)
                    else:
                        ax.spines['bottom'].set_visible(False)
                        ax.set_xticklabels([])
                        ax.xaxis.set_ticks_position('none')

                    fig.subplots_adjust(left=0.25, right=0.75, top=0.95, bottom=0.25, hspace=0.6)
                    sub_axes.append(ax)

                cax = plt.axes([0.7, 0.35, 0.025, 0.35])
                cmap = plt.get_cmap('plasma')
                sm = plt.cm.ScalarMappable(cmap=cmap)
                ticks = np.linspace(0, 1.0, len(self.subtalars)) 
                sm.set_array(ticks)
                cbar = plt.colorbar(sm, cax=cax, ticks=ticks)
                cbar.set_ticklabels(self.subtalar_peak_torques)
                cbar.ax.tick_params(axis='both', length=1.25, width=linewidth, 
                        labelsize=fontsize-1, pad=1)
                cbar.ax.get_yaxis().labelpad = 10
                cbar.ax.set_ylabel('subtalar peak torque ' + r'$[N \cdot m/kg]$', rotation=270,
                    fontsize=fontsize-0.5)
                cbar.outline.set_linewidth(linewidth)

                figs.append(fig)
                axes.append(sub_axes)         

        # Aggregate data
        # --------------
        numLabels = len(self.labels)
        import collections
        com_dict = collections.defaultdict(dict)
        time_dict = dict()
        for isubj, subject in enumerate(self.subjects):
            # Unperturbed center-of-mass trajectories
            unpTable = osim.TimeSeriesTable(file_dep[isubj*numLabels])
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
            time_dict[subject] = unpTimeVec

            for i, (label, time) in enumerate(zip(self.labels, self.times_list)):
                # Perturbed center-of-mass trajectories
                table = osim.TimeSeriesTable(file_dep[i + isubj*numLabels])
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
                com_dict[subject][label] = table_np - unpTable_np[0:N, :]

       # Plotting
        # --------
        def update_lims(data, step, lim):
            if np.min(data) < lim[0]:
                lim[0] = step * np.floor(np.min(data) / step)
                lim[1] = -lim[0]
            if np.max(data) > lim[1]:
                lim[1] = step * np.ceil(np.max(data) / step)
                lim[0] = -lim[1]

        def get_ticks_from_lims(lims, interval):
            # N = int(np.around((lims[1] - lims[0]) / interval, decimals=3)) + 1
            # ticks = np.linspace(lims[0], lims[1], N)
            ticks = [lims[0], 0, lims[1]]
            return ticks

        def get_x_scale(ax, x, y):
            point1 = ax.transData.transform((x, 0))
            point2 = ax.transData.transform((x + y, y))
            delta = point2 - point1
            scale = delta[1] / delta[0]
            return scale

        pos_x_diff = np.zeros(len(self.subjects))
        pos_y_diff = np.zeros(len(self.subjects))
        pos_z_diff = np.zeros(len(self.subjects))
        vel_x_diff = np.zeros(len(self.subjects))
        vel_y_diff = np.zeros(len(self.subjects))
        vel_z_diff = np.zeros(len(self.subjects))
        acc_x_diff = np.zeros(len(self.subjects))
        acc_y_diff = np.zeros(len(self.subjects))
        acc_z_diff = np.zeros(len(self.subjects))
        pos_x_diff_mean = np.zeros((len(self.times), len(self.subtalars)))
        pos_y_diff_mean = np.zeros((len(self.times), len(self.subtalars)))
        pos_z_diff_mean = np.zeros((len(self.times), len(self.subtalars)))
        vel_x_diff_mean = np.zeros((len(self.times), len(self.subtalars)))
        vel_y_diff_mean = np.zeros((len(self.times), len(self.subtalars)))
        vel_z_diff_mean = np.zeros((len(self.times), len(self.subtalars)))
        acc_x_diff_mean = np.zeros((len(self.times), len(self.subtalars)))
        acc_y_diff_mean = np.zeros((len(self.times), len(self.subtalars)))
        acc_z_diff_mean = np.zeros((len(self.times), len(self.subtalars)))
        for itime, time in enumerate(self.times):
            zipped = zip(self.subtalars, self.subtalar_colors, self.subtalar_shifts)
            for isubt, (subtalar, color, shift) in enumerate(zipped):

                label = (f'torque{self.torque}_time{time}'
                         f'_rise{self.rise}_fall{self.fall}{subtalar}')
                for isubj, subject in enumerate(self.subjects):
                    com = com_dict[subject][label]

                    # Compute the closet time index to the current peak 
                    # perturbation time. 
                    #
                    # TODO: A given peak perturbation time (e.g 50% of the 
                    # gait cycle) may not lie exactly on a time point of 
                    # from the simulation. The interval between time points
                    # is 5ms, meaning that the time index could be up to 2.5ms
                    # away from the actual perturbation peak time. 
                    timeVec = np.array(time_dict[subject])
                    duration = timeVec[-1] - timeVec[0]
                    time_at_peak = timeVec[0] + (duration * (time / 100.0))
                    index = np.argmin(np.abs(timeVec - time_at_peak))

                    pos_x_diff[isubj] = 100.0*com[index, 0]
                    pos_y_diff[isubj] = 100.0*com[index, 1]
                    pos_z_diff[isubj] = 100.0*com[index, 2]
                    vel_x_diff[isubj] = com[index, 3]
                    vel_y_diff[isubj] = com[index, 4]
                    vel_z_diff[isubj] = com[index, 5]
                    acc_x_diff[isubj] = com[index, 6]
                    acc_y_diff[isubj] = com[index, 7]
                    acc_z_diff[isubj] = com[index, 8]

                pos_x_diff_mean[itime, isubt] = np.mean(pos_x_diff)
                pos_y_diff_mean[itime, isubt] = np.mean(pos_y_diff)
                pos_z_diff_mean[itime, isubt] = np.mean(pos_z_diff)
                vel_x_diff_mean[itime, isubt] = np.mean(vel_x_diff)
                vel_y_diff_mean[itime, isubt] = np.mean(vel_y_diff)
                vel_z_diff_mean[itime, isubt] = np.mean(vel_z_diff)
                acc_x_diff_mean[itime, isubt] = np.mean(acc_x_diff)
                acc_y_diff_mean[itime, isubt] = np.mean(acc_y_diff)
                acc_z_diff_mean[itime, isubt] = np.mean(acc_z_diff)


        pos_step = 0.05 # cm
        vel_step = 0.005
        acc_step = 0.05
        pos_sagittal_lim = [0.0, 0.0]
        pos_coronal_lim = [0.0, 0.0]
        pos_frontal_lim = [0.0, 0.0]
        vel_sagittal_lim = [0.0, 0.0]
        vel_coronal_lim = [0.0, 0.0]
        vel_frontal_lim = [0.0, 0.0]
        acc_sagittal_lim = [0.0, 0.0]
        acc_coronal_lim = [0.0, 0.0]
        acc_frontal_lim = [0.0, 0.0]

        for isubt in np.arange(len(self.subtalars)):
            update_lims(pos_y_diff_mean, pos_step, pos_sagittal_lim)
            axes[0][isubt].set_ylim(pos_sagittal_lim)
            axes[0][isubt].set_yticks(get_ticks_from_lims(pos_sagittal_lim, pos_step))

            update_lims(pos_x_diff_mean, pos_step, pos_coronal_lim)
            axes[1][isubt].set_ylim(pos_coronal_lim)
            axes[1][isubt].set_yticks(get_ticks_from_lims(pos_coronal_lim, pos_step))

            update_lims(pos_y_diff_mean, pos_step, pos_frontal_lim)
            axes[2][isubt].set_ylim(pos_frontal_lim)
            axes[2][isubt].set_yticks(get_ticks_from_lims(pos_frontal_lim, pos_step))

            update_lims(vel_y_diff_mean, vel_step, vel_sagittal_lim)
            axes[3][isubt].set_ylim(vel_sagittal_lim)
            axes[3][isubt].set_yticks(get_ticks_from_lims(vel_sagittal_lim, vel_step))

            update_lims(vel_x_diff_mean, vel_step, vel_coronal_lim)
            axes[4][isubt].set_ylim(vel_coronal_lim)
            axes[4][isubt].set_yticks(get_ticks_from_lims(vel_coronal_lim, vel_step))

            update_lims(vel_y_diff_mean, vel_step, vel_frontal_lim)
            axes[5][isubt].set_ylim(vel_frontal_lim)
            axes[5][isubt].set_yticks(get_ticks_from_lims(vel_frontal_lim, vel_step))

            update_lims(acc_y_diff_mean, acc_step, acc_sagittal_lim)
            axes[6][isubt].set_ylim(acc_sagittal_lim)
            axes[6][isubt].set_yticks(get_ticks_from_lims(acc_sagittal_lim, acc_step))

            update_lims(acc_x_diff_mean, acc_step, acc_coronal_lim)
            axes[7][isubt].set_ylim(acc_coronal_lim)
            axes[7][isubt].set_yticks(get_ticks_from_lims(acc_coronal_lim, acc_step))

            update_lims(acc_y_diff_mean, acc_step, acc_frontal_lim)
            axes[8][isubt].set_ylim(acc_frontal_lim)
            axes[8][isubt].set_yticks(get_ticks_from_lims(acc_frontal_lim, acc_step))



        ilabel = (len(self.subtalars)-1) / 2
        for itime, time in enumerate(self.times):
            zipped = zip(self.subtalars, self.subtalar_colors, self.subtalar_shifts)
            for isubt, (subtalar, color, shift) in enumerate(zipped):

                lw = 1.2

                pos_x_diff = pos_x_diff_mean[itime, isubt]
                pos_y_diff = pos_y_diff_mean[itime, isubt]
                pos_z_diff = pos_z_diff_mean[itime, isubt]
                vel_x_diff = vel_x_diff_mean[itime, isubt]
                vel_y_diff = vel_y_diff_mean[itime, isubt]
                vel_z_diff = vel_z_diff_mean[itime, isubt]
                acc_x_diff = acc_x_diff_mean[itime, isubt]
                acc_y_diff = acc_y_diff_mean[itime, isubt]
                acc_z_diff = acc_z_diff_mean[itime, isubt]

                # Position vectors
                # ----------------
                scale = get_x_scale(axes[0][isubt], itime, pos_y_diff)
                x = [itime, itime + scale * pos_x_diff]
                y = [0, pos_y_diff]
                axes[0][isubt].plot(x, y, color=color, clip_on=False, lw=lw, solid_capstyle='round')

                scale = get_x_scale(axes[1][isubt], itime, pos_x_diff)
                x = [itime, itime + scale * pos_z_diff]
                y = [0, pos_x_diff]
                axes[1][isubt].plot(x, y, color=color, clip_on=False, lw=lw, solid_capstyle='round')

                scale = get_x_scale(axes[2][isubt], itime, pos_y_diff)
                x = [itime, itime + scale * pos_z_diff]
                y = [0, pos_y_diff]
                axes[2][isubt].plot(x, y, color=color, clip_on=False, lw=lw, solid_capstyle='round')

                # Velocity vectors
                # ----------------
                scale = get_x_scale(axes[3][isubt], itime, vel_y_diff)
                x = [itime, itime + scale * vel_x_diff]
                y = [0, vel_y_diff]
                axes[3][isubt].plot(x, y, color=color, clip_on=False, lw=lw, solid_capstyle='round')

                scale = get_x_scale(axes[4][isubt], itime, vel_x_diff)
                x = [itime, itime + scale * vel_z_diff]
                y = [0, vel_x_diff]
                axes[4][isubt].plot(x, y, color=color, clip_on=False, lw=lw, solid_capstyle='round')

                scale = get_x_scale(axes[5][isubt], itime, vel_y_diff)
                x = [itime, itime + scale * vel_z_diff]
                y = [0, vel_y_diff]
                axes[5][isubt].plot(x, y, color=color, clip_on=False, lw=lw, solid_capstyle='round')

                # Acceleration vectors
                # --------------------
                scale = get_x_scale(axes[6][isubt], itime, acc_y_diff)
                x = [itime, itime + scale * acc_x_diff]
                y = [0, acc_y_diff]
                axes[6][isubt].plot(x, y, color=color, clip_on=False, lw=lw, solid_capstyle='round')

                scale = get_x_scale(axes[7][isubt], itime, acc_x_diff)
                x = [itime, itime + scale * acc_z_diff]
                y = [0, acc_x_diff]
                axes[7][isubt].plot(x, y, color=color, clip_on=False, lw=lw, solid_capstyle='round')

                scale = get_x_scale(axes[8][isubt], itime, acc_y_diff)
                x = [itime, itime + scale * acc_z_diff]
                y = [0, acc_y_diff]
                axes[8][isubt].plot(x, y, color=color, clip_on=False, lw=lw, solid_capstyle='round')
                

                if isubt == ilabel:
                    axes[0][isubt].set_ylabel(r'$\Delta$' + ' position $[cm]$', fontsize=fontsize)
                    axes[1][isubt].set_ylabel(r'$\Delta$' + ' position $[cm]$', fontsize=fontsize)
                    axes[2][isubt].set_ylabel(r'$\Delta$' + ' position $[cm]$', fontsize=fontsize)
                    axes[3][isubt].set_ylabel(r'$\Delta$' + ' velocity $[m/s]$', fontsize=fontsize)
                    axes[4][isubt].set_ylabel(r'$\Delta$' + ' velocity $[m/s]$', fontsize=fontsize)
                    axes[5][isubt].set_ylabel(r'$\Delta$' + ' velocity $[m/s]$', fontsize=fontsize)
                    axes[6][isubt].set_ylabel(r'$\Delta$' + ' acceleration $[m/s^2]$', fontsize=fontsize)
                    axes[7][isubt].set_ylabel(r'$\Delta$' + ' acceleration $[m/s^2]$', fontsize=fontsize)
                    axes[8][isubt].set_ylabel(r'$\Delta$' + ' acceleration $[m/s^2]$',fontsize=fontsize)


        for ifig, fig in enumerate(figs):
            # fig.tight_layout()
            fig.savefig(target[ifig], dpi=600)
            plt.close()


class TaskPlotGroundReactions(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, time, torques, rise, fall, color, 
            APbox=[0.0, 0.0, 0.0, 0.0],
            MLbox=[0.0, 0.0, 0.0, 0.0],
            SIbox=[0.0, 0.0, 0.0, 0.0]):
        super(TaskPlotGroundReactions, self).__init__(study)
        self.name = f'plot_ground_reactions_time{time}_rise{rise}_fall{fall}'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'ground_reactions', f'time{time}_rise{rise}_fall{fall}')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)

        self.subjects = subjects
        self.time = time
        self.torques = torques
        self.rise = rise
        self.fall = fall
        self.color = color
        self.APbox = APbox
        self.MLbox = MLbox
        self.SIbox = SIbox

        self.ref_index = 0
        self.labels = list()
        self.colors = list()
        self.alphas = list()
        self.labels.append('unperturbed')
        self.colors.append('black')
        self.alphas.append(1.0)
        deps = list()

        self.models = list()

        for isubj, subject in enumerate(subjects):
            # Model
            self.models.append(os.path.join(
                self.study.config['results_path'], 
                'unperturbed', subject, 'model_unperturbed.osim'))

            # Unperturbed grfs
            deps.append(os.path.join(self.study.config['results_path'], 
                'unperturbed', subject, 'unperturbed_grfs.sto'))

            for torque in self.torques:
                label = (f'torque{torque}_time{self.time}'
                         f'_rise{self.rise}_fall{self.fall}')
                deps.append(
                    os.path.join(self.study.config['results_path'], 
                        f'perturbed_{label}', subject, 
                        f'perturbed_{label}_grfs.sto'))

                if not isubj:
                    self.labels.append(label)
                    self.colors.append(color)
                    self.alphas.append(torque / study.torques[-1]) 


        self.add_action(deps, 
                        [os.path.join(self.analysis_path, 
                            'ground_reactions.png'),
                         os.path.join(self.analysis_path, 
                            'ground_reaction_AP_diffs.png'),
                         os.path.join(self.analysis_path, 
                            'ground_reaction_SI_diffs.png'),
                         os.path.join(self.analysis_path, 
                            'ground_reaction_ML_diffs.png')], 
                        self.plot_ground_reactions)

    def plot_ground_reactions(self, file_dep, target):

        # Initialize figures
        # ------------------
        fig0 = plt.figure(figsize=(4, 6))
        rgrfx_ax = fig0.add_subplot(3, 1, 1)
        rgrfy_ax = fig0.add_subplot(3, 1, 2)
        rgrfz_ax = fig0.add_subplot(3, 1, 3)

        fig2 = plt.figure(figsize=(8, 3))
        gs = fig2.add_gridspec(1, 3)
        ax_f21 = fig2.add_subplot(gs[0, :-1])
        ax_f22 = fig2.add_subplot(gs[0, 2])

        fig3 = plt.figure(figsize=(8, 3))
        gs = fig3.add_gridspec(1, 3)
        ax_f31 = fig3.add_subplot(gs[0, :-1])
        ax_f32 = fig3.add_subplot(gs[0, 2])

        fig4 = plt.figure(figsize=(8, 3))
        gs = fig4.add_gridspec(1, 3)
        ax_f41 = fig4.add_subplot(gs[0, :-1])
        ax_f42 = fig4.add_subplot(gs[0, 2])

        # Helper functions
        # ----------------
        def get_ticks_from_lims(lims, interval):
            N = int(np.around((lims[1] - lims[0]) / interval, decimals=3)) + 1
            ticks = np.linspace(lims[0], lims[1], N)
            return ticks

        def get_ytext_from_lims(lims, shift):
            width = lims[1] - lims[0]
            ytext = width * (shift + 1.0) + lims[0]
            return ytext


        def update_ylims(ylim, interval, avg_diff, peak_diff, max_diff):
            min_avg = np.min(avg_diff)
            min_peak = np.min(peak_diff)
            min_max = np.min(max_diff)
            min_overall = np.min([min_avg, min_peak, min_max])

            if min_overall < ylim[0]:
                ylim[0] = np.floor(min_overall / interval) * interval

            max_avg = np.max(avg_diff)
            max_peak = np.max(peak_diff)
            max_max = np.max(max_diff)
            max_overall = np.max([max_avg, max_peak, max_max])

            if max_overall > ylim[1]:
                ylim[1] = np.ceil(max_overall / interval) * interval

            N = int((ylim[1] - ylim[0]) / interval) + 1
            yticks = np.linspace(ylim[0], ylim[1], N)

            return ylim, yticks

        # Aggregate data
        # --------------
        numSubjects = len(self.subjects)
        numLabels = len(self.labels)
        grfs = ['grfx', 'grfy', 'grfz']

        import collections
        grf_dict = collections.defaultdict(dict)
        for grf in grfs:
            for label in self.labels:
                if 'unperturbed' in label:
                    N = 101
                else:
                    N = self.time + self.fall + 11
                grf_dict[label][grf] = \
                    np.zeros((N, numSubjects))

        for isubj, subject in enumerate(self.subjects):
            model = osim.Model(self.models[isubj])
            state = model.initSystem()
            mass = model.getTotalMass(state)
            BW = abs(model.getGravity()[1]) * mass

            plot_zip = zip(self.labels, self.colors, self.alphas)
            for i, (label, color, alpha) in enumerate(plot_zip):
                table = osim.TimeSeriesTable(file_dep[i + isubj*numLabels])
                time = table.getIndependentColumn()
                if 'unperturbed' in label:
                    N = 101
                else:
                    N = self.time + self.fall + 11
                time_interp = np.linspace(time[0], time[-1], N)
                rgrfx = table.getDependentColumn('ground_force_r_vx').to_numpy() / BW
                rgrfy = table.getDependentColumn('ground_force_r_vy').to_numpy() / BW
                rgrfz = table.getDependentColumn('ground_force_r_vz').to_numpy() / BW

                grf_dict[label]['grfx'][:, isubj] = np.interp(
                            time_interp, time, rgrfx)
                grf_dict[label]['grfy'][:, isubj] = np.interp(
                            time_interp, time, rgrfy)
                grf_dict[label]['grfz'][:, isubj] = np.interp(
                            time_interp, time, rgrfz)

        plot_zip = zip(self.labels, self.colors, self.alphas)
        for i, (label, color, alpha) in enumerate(plot_zip):

            if 'unperturbed' in label:
                N = 101
                pgc = np.linspace(0, 100, N)
            else:
                N = self.time + self.fall + 11
                pgc = np.linspace(0, self.time + self.fall + 10, N)

            onset_index = self.time - self.rise
            offset_index = self.time + self.fall
            start_index = 0
            end_index = self.time + self.fall + 10

            pgc = pgc[start_index:end_index+1]
            rgrfx = np.mean(grf_dict[label]['grfx'], axis=1)[start_index:end_index+1]
            rgrfy = np.mean(grf_dict[label]['grfy'], axis=1)[start_index:end_index+1]
            rgrfz = np.mean(grf_dict[label]['grfz'], axis=1)[start_index:end_index+1]
            rgrfx_std = np.std(grf_dict[label]['grfx'], axis=1)[start_index:end_index+1]
            rgrfy_std = np.std(grf_dict[label]['grfy'], axis=1)[start_index:end_index+1]
            rgrfz_std = np.std(grf_dict[label]['grfz'], axis=1)[start_index:end_index+1]

            if 'unperturbed' in label:
                lw = 3
            else:
                lw = 2

            text = 'torque applied'
            text_shift = 0.02
            rgrfx_lim = [-0.2, 0.25]
            rgrfx_ticks = get_ticks_from_lims(rgrfx_lim, 0.05)
            ytext_rgrfx = get_ytext_from_lims(rgrfx_lim, text_shift)
            rgrfy_lim = [0.0, 1.5]
            rgrfy_ticks = get_ticks_from_lims(rgrfy_lim, 0.5)
            ytext_rgrfy = get_ytext_from_lims(rgrfy_lim, text_shift)
            rgrfz_lim = [-0.075, 0.05]
            rgrfz_ticks = get_ticks_from_lims(rgrfz_lim, 0.025)
            ytext_rgrfz = get_ytext_from_lims(rgrfz_lim, text_shift)
            for iax, ax in enumerate([ax_f21, ax_f22, rgrfx_ax]): 
                ax.plot(pgc, rgrfx, 
                    label=label, color=color, 
                    alpha=alpha, linewidth=lw)
                ax.fill_between(pgc, rgrfx-rgrfx_std, rgrfx+rgrfx_std, 
                     color=color, alpha=0.2*alpha, 
                     edgecolor=None, lw=None)
                ax.set_ylabel('fore-aft ground reaction (BW)')
                ax.set_ylim(rgrfx_lim)
                ax.set_yticks(rgrfx_ticks)
                ax.set_xlim(pgc[0], pgc[-1])
                util.publication_spines(ax)
                ax.spines['left'].set_position(('outward', 10))
                if iax == 0:
                    ax.spines['bottom'].set_position(('outward', 10))
                    ax.set_xlabel('time (% gait cycle)')
                    ax.text(self.time - 7, ytext_rgrfx, text, 
                        fontstyle='italic', color='gray', alpha=0.8,
                        fontsize=6, fontfamily='serif')
                    width = self.APbox[1] - self.APbox[0]
                    height = self.APbox[3] - self.APbox[2]
                    rect = patches.Rectangle((self.APbox[0], self.APbox[2]), 
                        width, height, 
                        linewidth=0.4, edgecolor='k', facecolor='none',
                        zorder=99, alpha=0.5)
                    ax.add_patch(rect)
                elif iax == 1:
                    ax.set_xlim(self.APbox[0:2])
                    ax.set_ylim(self.APbox[2:4])
                    ax.set_yticks(get_ticks_from_lims(self.APbox[2:4], 0.02))
                    ax.set_xlabel('time (% gait cycle)')
                elif iax == 2: 
                    ax.set_xticklabels([])
                    ax.spines['bottom'].set_visible(False)
                    ax.tick_params(axis='x', which='both', bottom=False, 
                                   top=False, labelbottom=False)
                    ax.text(self.time - 7, ytext_rgrfx, text, 
                        fontstyle='italic', color='gray', alpha=0.8,
                        fontsize=6, fontfamily='serif')
                      
            for iax, ax in enumerate([ax_f31, ax_f32, rgrfy_ax]):
                ax.plot(pgc, rgrfy, 
                        label=label, color=color, 
                        alpha=alpha, linewidth=lw)
                ax.fill_between(pgc, rgrfy-rgrfy_std, rgrfy+rgrfy_std, 
                     color=color, alpha=0.2*alpha, 
                     edgecolor=None, lw=None)
                ax.set_ylabel('vertical ground reaction (BW)')
                ax.set_ylim(rgrfy_lim)
                ax.set_yticks(rgrfy_ticks)
                ax.set_xlim(pgc[0], pgc[-1])
                util.publication_spines(ax)
                ax.spines['left'].set_position(('outward', 10))
                if iax == 0:
                    ax.spines['bottom'].set_position(('outward', 10))
                    ax.set_xlabel('time (% gait cycle)')
                    ax.text(self.time - 7, ytext_rgrfy, text, 
                        fontstyle='italic', color='gray', alpha=0.8,
                        fontsize=6, fontfamily='serif') 
                    width = self.SIbox[1] - self.SIbox[0]
                    height = self.SIbox[3] - self.SIbox[2]
                    rect = patches.Rectangle((self.SIbox[0], self.SIbox[2]), 
                        width, height, 
                        linewidth=0.4, edgecolor='k', facecolor='none',
                        zorder=99, alpha=0.5)
                    ax.add_patch(rect)
                elif iax == 1:
                    ax.set_xlim(self.SIbox[0:2])
                    ax.set_ylim(self.SIbox[2:4])
                    ax.set_yticks(get_ticks_from_lims(self.SIbox[2:4], 0.05))
                    ax.set_xlabel('time (% gait cycle)')
                elif iax == 2: 
                    ax.set_xticklabels([])
                    ax.spines['bottom'].set_visible(False)
                    ax.tick_params(axis='x', which='both', bottom=False, 
                                   top=False, labelbottom=False)

            for iax, ax in enumerate([ax_f41, ax_f42, rgrfz_ax]):
                ax.plot(pgc, rgrfz, 
                        label=label, color=color, 
                        alpha=alpha, linewidth=lw)
                ax.fill_between(pgc, rgrfz-rgrfz_std, rgrfz+rgrfz_std, 
                     color=color, alpha=0.2*alpha, 
                     edgecolor=None, lw=None)
                ax.set_xlabel('time (% gait cycle)')
                ax.set_ylabel('medio-lateral ground reaction (BW)')
                ax.set_ylim(rgrfz_lim)
                ax.set_yticks(rgrfz_ticks)
                ax.set_xlim(pgc[0], pgc[-1])
                util.publication_spines(ax)
                ax.spines['left'].set_position(('outward', 10))
                ax.spines['bottom'].set_position(('outward', 10))
                if iax == 0:
                    width = self.MLbox[1] - self.MLbox[0]
                    height = self.MLbox[3] - self.MLbox[2]
                    rect = patches.Rectangle((self.MLbox[0], self.MLbox[2]), 
                        width, height, 
                        linewidth=0.4, edgecolor='k', facecolor='none',
                        zorder=99, alpha=0.5)
                    ax.add_patch(rect)
                    ax.text(self.time - 7, ytext_rgrfz, text, 
                        fontstyle='italic', color='gray', alpha=0.8,
                        fontsize=6, fontfamily='serif')
                elif iax == 1:
                    ax.set_xlim(self.MLbox[0:2])
                    ax.set_ylim(self.MLbox[2:4])
                    ax.set_yticks(get_ticks_from_lims(self.MLbox[2:4], 0.01))

        for ax in [rgrfx_ax, rgrfy_ax, rgrfz_ax, ax_f21, ax_f31, ax_f41,
                   ax_f22, ax_f32, ax_f42]:
            ax.axhline(y=0, color='gray', linestyle='--',
                linewidth=0.5, alpha=0.5, zorder=0)
            ax.fill_betweenx([-5, 5], onset_index, offset_index, alpha=0.3, 
                color='gray', edgecolor=None, zorder=0, lw=None)
            ax.axvline(x=self.time, color=self.color, linestyle='--',
                linewidth=0.4, alpha=0.8, zorder=0) 

        fig0.tight_layout()
        fig0.savefig(target[0], dpi=600)
        plt.close()

        util.publication_spines(ax_f22, True)
        fig2.tight_layout()
        fig2.savefig(target[1], dpi=600)
        plt.close()

        util.publication_spines(ax_f32, True)
        fig3.tight_layout()
        fig3.savefig(target[2], dpi=600)
        plt.close()

        util.publication_spines(ax_f42, True)
        fig4.tight_layout()
        fig4.savefig(target[3], dpi=600)
        plt.close()


class TaskPlotInstantaneousGroundReactions(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, times, torque, rise, fall):
        super(TaskPlotInstantaneousGroundReactions, self).__init__(study)
        self.name = f'plot_instantaneous_ground_reactions_torque{torque}_rise{rise}_fall{fall}'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'ground_reactions_instantaneous', f'torque{torque}_rise{rise}_fall{fall}')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)

        self.subjects = subjects
        self.times = times
        self.torque = torque
        self.rise = rise
        self.fall = fall
        self.subtalars = study.subtalar_suffixes
        self.subtalar_colors = study.subtalar_colors
        self.width = 0.2
        self.width = 0.2
        N = len(study.subtalar_peak_torques)
        min_width = -self.width*((N-1)/2)
        max_width = -min_width
        self.subtalar_shifts = np.linspace(min_width, max_width, N)
        self.labels = list()
        self.labels.append('unperturbed')
        self.times_list = list()
        self.times_list.append(100)
        deps = list()

        self.models = list()

        for isubj, subject in enumerate(subjects):
            # Model
            self.models.append(os.path.join(
                self.study.config['results_path'], 
                'unperturbed', subject, 'model_unperturbed.osim'))

            # Unperturbed grfs
            deps.append(os.path.join(self.study.config['results_path'], 
                'unperturbed', subject, 'unperturbed_grfs.sto'))

            for time in self.times:
                for subtalar in self.subtalars:
                    label = (f'torque{self.torque}_time{time}'
                             f'_rise{self.rise}_fall{self.fall}{subtalar}')
                    deps.append(
                        os.path.join(self.study.config['results_path'], 
                            f'perturbed_{label}', subject, 
                            f'perturbed_{label}_grfs.sto'))

                    if not isubj:
                        self.labels.append(label)
                        self.times_list.append(time)

        targets = list()
        for direc in ['AP', 'SI', 'ML']:
            targets += [os.path.join(self.analysis_path, 
                        f'instant_grfs_{direc}')]

        self.add_action(deps, targets, 
                        self.plot_instantaneous_ground_reactions)

    def plot_instantaneous_ground_reactions(self, file_dep, target):

        # Initialize figures
        # ------------------
        figs = list()
        axes = list()
        for direc in ['AP', 'SI', 'ML']:
            fig = plt.figure(figsize=(5, 2))
            ax = fig.add_subplot(1, 1, 1)
            figs.append(fig)
            axes.append(ax)

        # Aggregate data
        # --------------
        numLabels = len(self.labels)
        import collections
        grf_dict = collections.defaultdict(dict)
        time_dict = dict()
        for isubj, subject in enumerate(self.subjects):
            model = osim.Model(self.models[isubj])
            state = model.initSystem()
            mass = model.getTotalMass(state)
            BW = abs(model.getGravity()[1]) * mass
            unpTable = osim.TimeSeriesTable(file_dep[isubj*numLabels])
            unpTimeVec = unpTable.getIndependentColumn()
            unpTable_np = np.zeros((len(unpTimeVec), 3))
            rgrfx = unpTable.getDependentColumn('ground_force_r_vx').to_numpy() / BW
            rgrfy = unpTable.getDependentColumn('ground_force_r_vy').to_numpy() / BW
            rgrfz = unpTable.getDependentColumn('ground_force_r_vz').to_numpy() / BW
            lgrfx = unpTable.getDependentColumn('ground_force_l_vx').to_numpy() / BW
            lgrfy = unpTable.getDependentColumn('ground_force_l_vy').to_numpy() / BW
            lgrfz = unpTable.getDependentColumn('ground_force_l_vz').to_numpy() / BW

            time_dict[subject] = unpTimeVec
            unpTable_np[:, 0] = rgrfx + lgrfx
            unpTable_np[:, 1] = rgrfy + lgrfy
            unpTable_np[:, 2] = rgrfz + lgrfz

            for i, (label, time) in enumerate(zip(self.labels, self.times_list)):
                table = osim.TimeSeriesTable(file_dep[i + isubj*numLabels])
                timeVec = table.getIndependentColumn()
                N = len(timeVec)
                table_np = np.zeros((N, 3))
                rgrfx = table.getDependentColumn('ground_force_r_vx').to_numpy() / BW
                rgrfy = table.getDependentColumn('ground_force_r_vy').to_numpy() / BW
                rgrfz = table.getDependentColumn('ground_force_r_vz').to_numpy() / BW
                lgrfx = table.getDependentColumn('ground_force_l_vx').to_numpy() / BW
                lgrfy = table.getDependentColumn('ground_force_l_vy').to_numpy() / BW
                lgrfz = table.getDependentColumn('ground_force_l_vz').to_numpy() / BW

                table_np[:, 0] = rgrfx + lgrfx
                table_np[:, 1] = rgrfy + lgrfy
                table_np[:, 2] = rgrfz + lgrfz
                grf_dict[subject][label] = table_np - unpTable_np[0:N, :]

        # Plotting
        # --------
        def update_lims(data, step, lim):
            if np.min(data) < lim[0]:
                lim[0] = step * np.floor(np.min(data) / step)
                lim[1] = -lim[0]
            if np.max(data) > lim[1]:
                lim[1] = step * np.ceil(np.max(data) / step)
                lim[0] = -lim[1]

        def get_ticks_from_lims(lims, interval):
            N = int(np.around((lims[1] - lims[0]) / interval, decimals=3)) + 1
            ticks = np.linspace(lims[0], lims[1], N)
            return ticks

        def plot_errorbar(ax, x, y, yerr):
            lolims = y > 0
            uplims = y < 0
            ple, cle, ble = ax.errorbar(x, y, yerr=yerr, 
                color=color, fmt='none', ecolor='black', 
                capsize=0, solid_capstyle='projecting', lw=0.25, 
                zorder=0, clip_on=False, lolims=lolims, uplims=uplims,
                elinewidth=0.4, markeredgewidth=0.4)
            for cl in cle:
                cl.set_marker('_')
                cl.set_markersize(4)

        grf_x_diff = np.zeros(len(self.subjects))
        grf_y_diff = np.zeros(len(self.subjects))
        grf_z_diff = np.zeros(len(self.subjects))
        grf_x_diff_mean = np.zeros((len(self.times), len(self.subtalars)))
        grf_y_diff_mean = np.zeros((len(self.times), len(self.subtalars)))
        grf_z_diff_mean = np.zeros((len(self.times), len(self.subtalars)))
        grf_x_diff_std = np.zeros((len(self.times), len(self.subtalars)))
        grf_y_diff_std = np.zeros((len(self.times), len(self.subtalars)))
        grf_z_diff_std = np.zeros((len(self.times), len(self.subtalars)))
        for itime, time in enumerate(self.times): 
            zipped = zip(self.subtalars, self.subtalar_colors, self.subtalar_shifts)
            for isubt, (subtalar, color, shift) in enumerate(zipped):

                label = (f'torque{self.torque}_time{time}'
                         f'_rise{self.rise}_fall{self.fall}{subtalar}')
                for isubj, subject in enumerate(self.subjects):
                    grf = grf_dict[subject][label]

                    # Compute the closet time index to the current peak 
                    # perturbation time. 
                    #
                    # TODO: A given peak perturbation time (e.g 50% of the 
                    # gait cycle) may not lie exactly on a time point of 
                    # from the simulation. The interval between time points
                    # is 5ms, meaning that the time index could be up to 2.5ms
                    # away from the actual perturbation peak time. 
                    timeVec = np.array(time_dict[subject])
                    duration = timeVec[-1] - timeVec[0]
                    time_at_peak = timeVec[0] + (duration * (time / 100.0))
                    index = np.argmin(np.abs(timeVec - time_at_peak))

                    grf_x_diff[isubj] = grf[index, 0]
                    grf_y_diff[isubj] = grf[index, 1]
                    grf_z_diff[isubj] = grf[index, 2]

                grf_x_diff_mean[itime, isubt] = np.mean(grf_x_diff)
                grf_y_diff_mean[itime, isubt] = np.mean(grf_y_diff)
                grf_z_diff_mean[itime, isubt] = np.mean(grf_z_diff)

                grf_x_diff_std[itime, isubt] = np.std(grf_x_diff)
                grf_y_diff_std[itime, isubt] = np.std(grf_y_diff)
                grf_z_diff_std[itime, isubt] = np.std(grf_z_diff)


        grf_x_step = 0.01 # cm
        grf_y_step = 0.01
        grf_z_step = 0.01
        grf_x_lim = [0.0, 0.0]
        grf_y_lim = [0.0, 0.0]
        grf_z_lim = [0.0, 0.0]
        update_lims(grf_x_diff_mean-grf_x_diff_std, grf_x_step, grf_x_lim)
        update_lims(grf_x_diff_mean+grf_x_diff_std, grf_x_step, grf_x_lim)
        update_lims(grf_y_diff_mean-grf_y_diff_std, grf_y_step, grf_y_lim)
        update_lims(grf_y_diff_mean+grf_y_diff_std, grf_y_step, grf_y_lim)
        update_lims(grf_z_diff_mean-grf_z_diff_std, grf_z_step, grf_z_lim)
        update_lims(grf_z_diff_mean+grf_z_diff_std, grf_z_step, grf_z_lim)
        for itime, time in enumerate(self.times):
            zipped = zip(self.subtalars, self.subtalar_colors, self.subtalar_shifts)
            for isubt, (subtalar, color, shift) in enumerate(zipped):

                # Set the x-position for these bar chart entries.
                x = itime + shift

                plot_errorbar(axes[0], x, grf_x_diff_mean[itime, isubt], grf_x_diff_std[itime, isubt])
                axes[0].bar(x, grf_x_diff_mean[itime, isubt], self.width, color=color, clip_on=False)
                axes[0].set_ylabel(r'$\Delta$ fore-aft ground reaction $[BW]$')

                plot_errorbar(axes[1], x, grf_y_diff_mean[itime, isubt], grf_y_diff_std[itime, isubt])
                axes[1].bar(x, grf_y_diff_mean[itime, isubt], self.width, color=color, clip_on=False)
                axes[1].set_ylabel(r'$\Delta$ vertical ground reaction $[BW]$')

                plot_errorbar(axes[2], x, grf_z_diff_mean[itime, isubt], grf_z_diff_std[itime, isubt])
                axes[2].bar(x, grf_z_diff_mean[itime, isubt], self.width, color=color, clip_on=False)
                axes[2].set_ylabel(r'$\Delta$ medio-lateral ground reaction $[BW]$')

        axes[0].set_ylim(grf_x_lim)
        axes[0].set_yticks(get_ticks_from_lims(grf_x_lim, grf_x_step))
        axes[1].set_ylim(grf_y_lim)
        axes[1].set_yticks(get_ticks_from_lims(grf_y_lim, grf_y_step))
        axes[2].set_ylim(grf_z_lim)
        axes[2].set_yticks(get_ticks_from_lims(grf_z_lim, grf_z_step))

        for ax in axes:
            ax.axhline(y=0, color='black', linestyle='-',
                    linewidth=0.1, alpha=1.0, zorder=-1)
            ax.spines['bottom'].set_position(('outward', 10))
            ax.spines['left'].set_position(('outward', 25))
            ax.set_xticks(np.arange(len(self.times)))
            ax.set_xlim(0, len(self.times)-1)
            ax.set_xticklabels([f'{time}' for time in self.times])
            ax.set_xlabel('peak perturbation time\n(% gait cycle)')
            util.publication_spines(ax)

        for ifig, fig in enumerate(figs):
            fig.tight_layout()
            fig.savefig(target[ifig], dpi=600)
            plt.close()


class TaskPlotGroundReactionBreakdown(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, time, torque, rise, fall, 
            APbox=[0.0, 0.0, 0.0, 0.0],
            MLbox=[0.0, 0.0, 0.0, 0.0],
            SIbox=[0.0, 0.0, 0.0, 0.0]):
        super(TaskPlotGroundReactionBreakdown, self).__init__(study)
        self.name = (f'plot_ground_reaction_breakdown'
                     f'_time{time}_torque{torque}_rise{rise}_fall{fall}')
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'ground_reaction_breakdown', 
            f'time{time}_torque{torque}_rise{rise}_fall{fall}')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)

        self.subjects = subjects
        self.time = time
        self.torque = torque
        self.rise = rise
        self.fall = fall
        self.APbox = APbox
        self.MLbox = MLbox
        self.SIbox = SIbox

        self.forces = ['contactHeel_r',
                       'contactLateralRearfoot_r',
                       'contactLateralMidfoot_r',
                       'contactMedialMidfoot_r',
                       'contactLateralToe_r',
                       'contactMedialToe_r']
        self.force_colors = ['darkred', 'darkorange', 'gold',
                             'darkgreen', 'darkblue', 'indigo']
        self.force_labels = ['heel', 'rearfoot', 'lat. midfoot',
                             'med. midfoot', 'lat. toe', 'med. toe']

        self.ref_index = 0
        deps = list()

        self.models = list()
        for isubj, subject in enumerate(subjects):
            # Model
            self.models.append(os.path.join(
                self.study.config['results_path'], 
                'unperturbed', subject, 'model_unperturbed.osim'))

            label = (f'torque{self.torque}_time{self.time}'
                     f'_rise{self.rise}_fall{self.fall}')
            deps.append(
                os.path.join(self.study.config['results_path'], 
                    f'perturbed_{label}', subject, 
                    f'perturbed_{label}.sto'))

        self.add_action(deps, 
                        [os.path.join(self.analysis_path, 
                            'ground_reaction_breakdown_AP.png'),
                        os.path.join(self.analysis_path, 
                            'ground_reaction_breakdown_ML.png'),
                        os.path.join(self.analysis_path, 
                            'ground_reaction_breakdown_SI.png')], 
                        self.plot_ground_reaction_breakdown)

    def plot_ground_reaction_breakdown(self, file_dep, target):

        # Initialize figures
        # ------------------
        fig1 = plt.figure(figsize=(8, 3))
        gs = fig1.add_gridspec(1, 3)
        ax_f11 = fig1.add_subplot(gs[0, :-1])
        ax_f12 = fig1.add_subplot(gs[0, 2])

        fig2 = plt.figure(figsize=(8, 3))
        gs = fig2.add_gridspec(1, 3)
        ax_f21 = fig2.add_subplot(gs[0, :-1])
        ax_f22 = fig2.add_subplot(gs[0, 2])

        fig3 = plt.figure(figsize=(8, 3))
        gs = fig3.add_gridspec(1, 3)
        ax_f31 = fig3.add_subplot(gs[0, :-1])
        ax_f32 = fig3.add_subplot(gs[0, 2])

        # Helper functions
        # ----------------
        def get_ticks_from_lims(lims, interval):
            N = int(np.around((lims[1] - lims[0]) / interval, decimals=3)) + 1
            ticks = np.linspace(lims[0], lims[1], N)
            return ticks

        def get_ytext_from_lims(lims, shift):
            width = lims[1] - lims[0]
            ytext = width * (shift + 1.0) + lims[0]
            return ytext

        def update_ylims(ylim, interval, avg_diff, peak_diff, max_diff):
            min_avg = np.min(avg_diff)
            min_peak = np.min(peak_diff)
            min_max = np.min(max_diff)
            min_overall = np.min([min_avg, min_peak, min_max])

            if min_overall < ylim[0]:
                ylim[0] = np.floor(min_overall / interval) * interval

            max_avg = np.max(avg_diff)
            max_peak = np.max(peak_diff)
            max_max = np.max(max_diff)
            max_overall = np.max([max_avg, max_peak, max_max])

            if max_overall > ylim[1]:
                ylim[1] = np.ceil(max_overall / interval) * interval

            N = int((ylim[1] - ylim[0]) / interval) + 1
            yticks = np.linspace(ylim[0], ylim[1], N)

            return ylim, yticks

        # Aggregate data
        # --------------
        numSubjects = len(self.subjects)
        N = self.time + self.fall + 1
        pgc = np.linspace(0, self.time + self.fall, N)
        onset_index = self.time - self.rise
        offset_index = self.time + self.fall
        start_index = 0
        end_index = self.time + self.fall

        direcs = ['x', 'y', 'z']
    
        from collections import defaultdict
        grf_dict = defaultdict(dict)
        
        for direc in direcs:
            for force in self.forces:
                grf_dict[force][direc] = \
                    np.zeros((N, numSubjects))
            grf_dict['total'][direc] = \
                np.zeros((N, numSubjects))

        for isubj, subject in enumerate(self.subjects):
            model = osim.Model(self.models[isubj])
            state = model.initSystem()
            mass = model.getTotalMass(state)
            BW = abs(model.getGravity()[1]) * mass

            mocoTraj = osim.MocoTrajectory(file_dep[isubj])
            time = mocoTraj.getTimeMat()
            time_interp = np.linspace(time[0], time[-1], N)

            for force in self.forces:
                sshsForce = osim.SmoothSphereHalfSpaceForce.safeDownCast(
                    model.getComponent(f'/forceset/{force}'))

                grfx = np.zeros_like(time)
                grfy = np.zeros_like(time)
                grfz = np.zeros_like(time)
                statesTraj = mocoTraj.exportToStatesTrajectory(model)
                for istate in np.arange(statesTraj.getSize()):
                    state = statesTraj.get(int(istate))
                    model.realizeVelocity(state)
                    forceValues = sshsForce.getRecordValues(state)

                    grfx[istate] = forceValues.get(0) / BW
                    grfy[istate] = forceValues.get(1) / BW
                    grfz[istate] = forceValues.get(2) / BW

                grf_dict[force]['x'][:, isubj] = np.interp(
                            time_interp, time, grfx)
                grf_dict[force]['y'][:, isubj] = np.interp(
                            time_interp, time, grfy)
                grf_dict[force]['z'][:, isubj] = np.interp(
                            time_interp, time, grfz)

        for force in self.forces:
            for direc in direcs:
                grf_dict['total'][direc] += grf_dict[force][direc]

        # Plotting
        # --------
        text = 'torque applied'
        text_shift = 0.02
        grfx_lim = [-0.2, 0.25]
        grfx_ticks = get_ticks_from_lims(grfx_lim, 0.05)
        ytext_grfx = get_ytext_from_lims(grfx_lim, text_shift)
        grfy_lim = [0.0, 1.5]
        grfy_ticks = get_ticks_from_lims(grfy_lim, 0.5)
        ytext_rgrfy = get_ytext_from_lims(grfy_lim, text_shift)
        grfz_lim = [-0.075, 0.05]
        grfz_ticks = get_ticks_from_lims(grfz_lim, 0.025)
        ytext_grfz = get_ytext_from_lims(grfz_lim, text_shift)
        for iax, ax in enumerate([ax_f11, ax_f12]): 
            handles = list()
            grfx = np.mean(grf_dict['total']['x'], axis=1)
            h, = ax.plot(pgc, grfx, color='black', linewidth=3)
            handles.append(h)
            for force, color in zip(self.forces, self.force_colors):
                grfx = np.mean(grf_dict[force]['x'], axis=1)
                h, = ax.plot(pgc, grfx, color=color, linewidth=2)
                handles.append(h)

            ax.set_ylabel('anterior-posterior force (BW)')
            ax.set_ylim(grfx_lim)
            ax.set_yticks(grfx_ticks)
            ax.set_xlim(pgc[0], pgc[-1])
            util.publication_spines(ax)
            ax.spines['left'].set_position(('outward', 10))
            if iax == 0:
                ax.spines['bottom'].set_position(('outward', 10))
                ax.set_xlabel('time (% gait cycle)')
                ax.text(self.time - 5, ytext_grfx, text, 
                    fontstyle='italic', color='gray', alpha=0.8,
                    fontsize=6, fontfamily='serif')
                width = self.APbox[1] - self.APbox[0]
                height = self.APbox[3] - self.APbox[2]
                rect = patches.Rectangle((self.APbox[0], self.APbox[2]), 
                    width, height, 
                    linewidth=0.4, edgecolor='k', facecolor='none',
                    zorder=99, alpha=0.5)
                ax.add_patch(rect)

                labels = ['total'] + self.force_labels
                ax.legend(handles, labels, fancybox=False,
                    frameon=False, prop={'size': 6}, loc='upper left')

            elif iax == 1:
                ax.set_xlim(self.APbox[0:2])
                ax.set_ylim(self.APbox[2:4])
                ax.set_yticks(get_ticks_from_lims(self.APbox[2:4], 0.02))
                ax.set_xlabel('time (% gait cycle)')

                  
        for iax, ax in enumerate([ax_f21, ax_f22]):
            handles = list()
            grfy = np.mean(grf_dict['total']['y'], axis=1)
            h, = ax.plot(pgc, grfy, color='black', linewidth=3)
            handles.append(h)
            for force, color in zip(self.forces, self.force_colors):
                grfy = np.mean(grf_dict[force]['y'], axis=1)
                h, = ax.plot(pgc, grfy, color=color, linewidth=2)
                handles.append(h)

            ax.set_ylabel('vertical force (BW)')
            ax.set_ylim(grfy_lim)
            ax.set_yticks(grfy_ticks)
            ax.set_xlim(pgc[0], pgc[-1])
            util.publication_spines(ax)
            ax.spines['left'].set_position(('outward', 10))
            if iax == 0:
                ax.spines['bottom'].set_position(('outward', 10))
                ax.set_xlabel('time (% gait cycle)')
                ax.text(self.time - 5, ytext_rgrfy, text, 
                    fontstyle='italic', color='gray', alpha=0.8,
                    fontsize=6, fontfamily='serif') 
                width = self.SIbox[1] - self.SIbox[0]
                height = self.SIbox[3] - self.SIbox[2]
                rect = patches.Rectangle((self.SIbox[0], self.SIbox[2]), 
                    width, height, 
                    linewidth=0.4, edgecolor='k', facecolor='none',
                    zorder=99, alpha=0.5)
                ax.add_patch(rect)

                labels = ['total'] + self.force_labels
                ax.legend(handles, labels, fancybox=False,
                    frameon=False, prop={'size': 6}, loc='upper left')

            elif iax == 1:
                ax.set_xlim(self.SIbox[0:2])
                ax.set_ylim(self.SIbox[2:4])
                ax.set_yticks(get_ticks_from_lims(self.SIbox[2:4], 0.05))
                ax.set_xlabel('time (% gait cycle)')

        for iax, ax in enumerate([ax_f31, ax_f32]):
            handles = list()
            grfz = np.mean(grf_dict['total']['z'], axis=1)
            h, = ax.plot(pgc, grfz, color='black', linewidth=3)
            handles.append(h)
            for force, color in zip(self.forces, self.force_colors):
                grfz = np.mean(grf_dict[force]['z'], axis=1)
                h, = ax.plot(pgc, grfz, color=color, linewidth=2)
                handles.append(h)

            ax.set_xlabel('time (% gait cycle)')
            ax.set_ylabel('medio-lateral force (BW)')
            ax.set_ylim(grfz_lim)
            ax.set_yticks(grfz_ticks)
            ax.set_xlim(pgc[0], pgc[-1])
            util.publication_spines(ax)
            ax.spines['left'].set_position(('outward', 10))
            ax.spines['bottom'].set_position(('outward', 10))
            if iax == 0:
                width = self.MLbox[1] - self.MLbox[0]
                height = self.MLbox[3] - self.MLbox[2]
                rect = patches.Rectangle((self.MLbox[0], self.MLbox[2]), 
                    width, height, 
                    linewidth=0.4, edgecolor='k', facecolor='none',
                    zorder=99, alpha=0.5)
                ax.add_patch(rect)
                ax.text(self.time - 5, ytext_grfz, text, 
                    fontstyle='italic', color='gray', alpha=0.8,
                    fontsize=6, fontfamily='serif')
                labels = ['total'] + self.force_labels
                ax.legend(handles, labels, fancybox=False,
                    frameon=False, prop={'size': 6})

            elif iax == 1:
                ax.set_xlim(self.MLbox[0:2])
                ax.set_ylim(self.MLbox[2:4])
                ax.set_yticks(get_ticks_from_lims(self.MLbox[2:4], 0.02))

        for ax in [ax_f11, ax_f21, ax_f31, ax_f12, ax_f22, ax_f32]:
            ax.axhline(y=0, color='gray', linestyle='--',
                linewidth=0.5, alpha=0.5, zorder=0)
            ax.fill_betweenx([-5, 5], onset_index, offset_index, alpha=0.3, 
                color='gray', edgecolor=None, zorder=0, lw=None)
            ax.axvline(x=self.time, color='black', linestyle='--',
                linewidth=0.4, alpha=0.8, zorder=0) 

        util.publication_spines(ax_f12, True)
        fig1.tight_layout()
        fig1.savefig(target[0], dpi=600)
        plt.close()

        util.publication_spines(ax_f22, True)
        fig2.tight_layout()
        fig2.savefig(target[1], dpi=600)
        plt.close()

        util.publication_spines(ax_f32, True)
        fig3.tight_layout()
        fig3.savefig(target[2], dpi=600)
        plt.close()


