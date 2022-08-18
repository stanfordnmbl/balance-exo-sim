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


# Validate inverse kinematics
# ---------------------------

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
    def __init__(self, trial, initial_time, final_time, mesh_interval=0.01,
                tolerance=1e-2, walking_speed=1.25, guess_fpath=None,
                randomize_guess=False, implicit_tendon_dynamics=False, **kwargs):
        super(TaskMocoSensitivityAnalysis, self).__init__(trial)
        self.tolerance = tolerance
        self.config_name = f'sensitivity_tol{self.tolerance}'
        self.name = f'{trial.subject.name}_moco_{self.config_name}'
        self.initial_time = initial_time
        self.final_time = final_time
        self.mesh_interval = mesh_interval
        self.constraint_tolerance = trial.study.constraint_tolerance
        self.tolerance = tolerance
        self.num_max_iterations = 10000
        self.walking_speed = walking_speed
        self.weights = trial.study.weights
        self.root_dir = trial.study.config['doit_path']
        self.guess_fpath = guess_fpath
        self.randomize_guess = randomize_guess
        self.implicit_tendon_dynamics = implicit_tendon_dynamics

        expdata_dir = os.path.join(trial.results_exp_path, 'tracking_data', 'expdata')
        extloads_dir = os.path.join(trial.results_exp_path, 'tracking_data', 'extloads')
        self.tracking_coordinates_fpath = os.path.join(expdata_dir, 'coordinates.sto')
        self.coordinates_std_fpath = os.path.join(trial.results_exp_path, 
            f'{trial.id}_joint_angle_standard_deviations.csv')
        self.tracking_extloads_fpath = os.path.join(extloads_dir, 'external_loads.xml')
        self.tracking_grfs_fpath = os.path.join(expdata_dir, 'ground_reaction.mot')

        self.result_fpath = os.path.join(self.study.config['results_path'],
            'sensitivity', trial.subject.name, f'tol{self.tolerance}')
        if not os.path.exists(self.result_fpath): os.makedirs(self.result_fpath)

        self.archive_fpath = os.path.join(self.study.config['results_path'],
            'sensitivity', trial.subject.name, f'tol{tolerance}', 'archive')
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
                         self.emg_fpath,
                         self.guess_fpath],
                        [os.path.join(self.result_fpath, 
                            self.config_name + '.sto')],
                        self.run_tracking_problem)

    def run_tracking_problem(self, file_dep, target):
        
        config = TrackingConfig(
            self.config_name, self.config_name, 'black', self.weights,
            periodic=True,
            periodic_values=True,
            periodic_speeds=True,
            periodic_actuators=True,
            lumbar_stiffness=1.0,
            guess=file_dep[6],
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
            self.tolerance,
            self.constraint_tolerance,
            self.num_max_iterations,
            self.walking_speed,
            [config],
            implicit_tendon_dynamics=self.implicit_tendon_dynamics
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
        self.tols = [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]
        self.tol_exps = [1, 0, -1, -2, -3, -4]

        deps_tol = list()
        for subject in subjects:
            for tol in self.tols:
                result = f'sensitivity_tol{tol}.sto'
                result_fpath = os.path.join(
                    self.results_path,
                    subject, f'tol{tol}', result)
                deps_tol.append(result_fpath)

        self.add_action(deps_tol, 
                        [os.path.join(self.validate_path, 
                            'sensitivity_analysis.png')], 
                        self.plot_sensitivity_analysis)

    def plot_sensitivity_analysis(self, file_dep, target):

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
            norm_objectives[isubj, :] = 100.0 * ((objectives[isubj, :] / objectives[isubj, -1]) - 1)

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
        ax.set_ylim(0, 5)
        ax.set_yticks([0, 1, 2, 3, 4, 5])
        ax.set_ylabel(r'$\Delta$ ' + r'normalized objective $[\%]$')
        ax.set_xlabel('convergence tolerance')
        # ax.axhline(y=1.0, color='gray', linewidth=0.5, ls='--', zorder=0)

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
                reserve_strength=0, implicit_multibody_dynamics=False,
                implicit_tendon_dynamics=False, 
                create_and_insert_guess=False,
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
        self.convergence_tolerance = 1e-2
        self.constraint_tolerance = 1e-3
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

        weights['state_tracking_weight'] /= self.cost_scale
        weights['state_tracking_weight'] *= 10

        weights['grf_tracking_weight'] /= self.cost_scale

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
                            f'{self.config_name}_grfs.png')], 
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
        labels = ['hip flexion', 'hip adduction', 'knee flexion', 'ankle dorsiflexion', 
                  'subtalar inversion', 'toe dorsiflexion']
        coordinates = ['hip_flexion', 'hip_adduction', 'knee_angle', 'ankle_angle',
                       'subtalar_angle', 'mtp_angle']
        joints = ['hip', 'hip', 'walker_knee', 'ankle', 'subtalar', 'mtp']
        bounds = [[-20, 40],
                  [-20, 10],
                  [0, 80],
                  [-20, 30],
                  [-10, 20],
                  [0, 30]]
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
        fig = plt.figure(figsize=(5, 12))
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
                ax.grid(color='black', zorder=0, alpha=0.2, lw=0.5, ls='--', clip_on=False)
                util.publication_spines(ax)

                if not ic and not iside:
                    ax.legend([h_exp, h_unp], ['experiment', 'simulation'],
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
                if not ic:
                    if 'l' in side:
                        ax.set_title('left leg')
                    elif 'r' in side:
                        ax.set_title('right leg')

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
                # ax.axhline(y=0, color='black', alpha=0.4, linestyle='--', zorder=0, lw=0.75)
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
                    ax.set_ylabel(f'{labels[iforce]} (BW)')
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
                        self.plot_unperturbed_coordinates)

        self.add_action(unperturbed_grf_fpaths + grf_fpaths, 
                        [os.path.join(self.validate_path, 
                            'grf_tracking_errors.txt')], 
                        self.plot_unperturbed_grfs)

    def plot_unperturbed_coordinates(self, file_dep, target): 

        numSubjects = len(self.subjects)
        N = 100
        coordinates = ['hip_flexion', 'hip_adduction', 'knee_angle', 'ankle_angle']
        joints = ['hip', 'hip', 'walker_knee', 'ankle']
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
                    unperturbed_col = unperturbed.getDependentColumn(key).to_numpy()
                    experiment_col = experiment.getDependentColumn(key).to_numpy()[istart:iend+1]
                    unperturbed_dict[key][:, i] = np.interp(
                        utime_interp, utime, unperturbed_col)
                    experiment_dict[key][:, i] = np.interp(
                        etime_interp, etime[istart:iend+1], experiment_col)

        rmse_dict = dict()
        for ic, coord in enumerate(coordinates):
            for iside, side in enumerate(['l','r']):
                key = f'/jointset/{joints[ic]}_{side}/{coord}_{side}/value'
                errors = experiment_dict[key] - unperturbed_dict[key]
                rmse = np.sqrt(np.sum(np.square(errors)) / N)
                rmse_dict[f'{coord}_{side}'] = rmse

        with open(target[0], 'w') as f:
            f.write('Coordinate tracking errors (RMSE)\n')
            f.write('---------------------------------\n')
            for key, value in rmse_dict.items():
                f.write(f' -- {key}: {value:.2f} [rad]\n')

    def plot_unperturbed_grfs(self, file_dep, target):

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

        rmse_dict = dict()
        for iforce, force in enumerate(forces):
            for iside, side in enumerate(['l', 'r']):
                label = f'ground_force_{side}_{force}'
                errors = experiment_dict[label] - unperturbed_dict[label]
                rmse = np.sqrt(np.sum(np.square(errors)) / N)
                rmse_dict[label] = rmse

        with open(target[0], 'w') as f:
            f.write('GRF tracking errors (RMSE)\n')
            f.write('--------------------------\n')
            for key, value in rmse_dict.items():
                f.write(f' -- {key}: {value:.2f} [N] \n')

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

# Analysis
# --------
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
        zorder=0, clip_on=False, lolims=lolims, uplims=uplims,
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
        zorder=0, clip_on=False, xlolims=xlolims, xuplims=xuplims,
        elinewidth=0.4, markeredgewidth=0.4)
    for cl in cle:
        cl.set_marker('|')
        cl.set_markersize(4)


class TaskPlotAnkleTorquesAndPowers(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subject, times, torque, subtalar, rise, fall):
        super(TaskPlotAnkleTorquesAndPowers, self).__init__(study)
        self.subject = subject
        self.times = times
        self.torque = torque
        self.rise = rise
        self.fall = fall

        suffix = f'torque{torque}_rise{rise}_fall{fall}'
        self.subtalar = f'_subtalar{subtalar}' if subtalar else ''
        suffix += self.subtalar

        self.name = f'plot_ankle_torques_and_powers_{suffix}'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'perturbation_torques_and_powers', suffix)
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)

        deps = list()

        for time in self.times:
            label = (f'torque{self.torque}_time{time}'
                     f'_rise{self.rise}_fall{self.fall}{self.subtalar}')
            deps.append(
                os.path.join(self.study.config['results_path'], 
                    'perturbed', f'perturbed_{label}', subject, 
                    f'perturbed_{label}.sto'))
            deps.append(
                os.path.join(self.study.config['results_path'], 
                    'perturbed', f'perturbed_{label}', subject, 
                    'ankle_perturbation_curve.sto'))

        # Model 
        self.model = os.path.join(self.study.config['results_path'], 
            'unperturbed', subject, 'model_unperturbed.osim')

        self.add_action(deps, 
                        [os.path.join(self.analysis_path, 
                            'ankle_perturbation_torques.png')], 
                        self.plot_ankle_torques_and_powers)

    def plot_ankle_torques_and_powers(self, file_dep, target):

        fig = plt.figure(figsize=(2*len(self.times), 8))
        gs = fig.add_gridspec(3, len(self.times))
        torque_axes = list()
        speed_axes = list()
        power_axes = list()

        for itime in range(len(self.times)):
            torque_axes.append(fig.add_subplot(gs[0, itime]))
            speed_axes.append(fig.add_subplot(gs[1, itime]))
            power_axes.append(fig.add_subplot(gs[2, itime]))

        lw = 2.5

        zipped = zip(self.times, torque_axes, speed_axes, power_axes)
        for itime, (time, ax_t, ax_s, ax_p) in enumerate(zipped):
            # Perturbation torque
            # -------------------
            perturbation = osim.TimeSeriesTable(file_dep[2*itime + 1])
        
            timeVec = np.array(perturbation.getIndependentColumn())
            pgc = 100 * (timeVec - timeVec[0]) / (timeVec[-1] - timeVec[0])
            onset_time = time - 10
            offset_time = time + 5
            istart = np.argmin(np.abs(pgc - onset_time))
            iend = np.argmin(np.abs(pgc - offset_time))

            torque = perturbation.getDependentColumn(
                '/forceset/perturbation_ankle_angle_r').to_numpy()

            ax_t.plot(pgc[istart:iend], -torque[istart:iend], color='black', 
                linewidth=lw, clip_on=False,
                solid_capstyle='round')
            ax_t.set_ylim([0.0, 0.2])
            ax_t.set_yticks([0.0, 0.1, 0.2])
            ax_t.set_xlim(onset_time, offset_time)
            
            # Ankle speed
            # -----------
            solution = osim.TimeSeriesTable(file_dep[2*itime])
            speed = solution.getDependentColumn(
                '/jointset/ankle_r/ankle_angle_r/speed').to_numpy()
            timeVecSol = np.array(solution.getIndependentColumn())
            pgc = offset_time * (timeVecSol - timeVecSol[0]) / (timeVecSol[-1] - timeVecSol[0])
            istart = np.argmin(np.abs(pgc - onset_time))
            iend = np.argmin(np.abs(pgc - offset_time))
            ax_s.plot(pgc[istart:iend], -speed[istart:iend], color='black', 
                linewidth=lw, clip_on=False,
                solid_capstyle='round')
            ax_s.set_ylim([-2, 6])
            ax_s.set_yticks([-2, 0, 2, 4, 6])
            ax_s.set_xlim(onset_time, offset_time)

            # Perturbation power
            # ------------------
            from scipy.interpolate import interp1d
            torque_interp = interp1d(timeVec, torque)
            torque_sampled = torque_interp(timeVecSol)

            power = np.multiply(speed, torque_sampled)

            ax_p.plot(pgc[istart:iend], power[istart:iend], color='black', 
                linewidth=lw, clip_on=False,
                solid_capstyle='round')
            ax_p.set_ylim([-0.1, 0.5])
            ax_p.set_yticks([-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
            ax_p.set_xlim(onset_time, offset_time)

            if not itime:
                ax_t.set_ylabel('torque (N-m/kg)')
                ax_s.set_ylabel('ankle speed (rad/s)')
                ax_p.set_ylabel('power (W/kg)')

            util.publication_spines(ax_t, True)
            util.publication_spines(ax_s, True)
            util.publication_spines(ax_p, True)

            ax_t.set_title(f'peak time: {time}%')
            ax_p.set_xlabel('time (% gait cycle)')

            for x in [onset_time, time - 5, time, offset_time]:
                for ax in [ax_t, ax_s, ax_p]:
                    ax.axvline(x, color='gray', 
                      alpha=0.5, linewidth=0.5, 
                      zorder=0, clip_on=False, linestyle='--')

            for ax in [ax_t, ax_s, ax_p]:
                ax.axhline(0, color='gray', linewidth=0.5,
                    zorder=0, clip_on=False, linestyle='--')

        # ax_r.spines['right'].set_visible(False)
        # ax_r.spines['top'].set_visible(False)
        # ax_r.spines['bottom'].set_visible(False)
        # ax_r.spines['left'].set_visible(False)
        # ax_r.tick_params(axis='x', which='both', bottom=False, 
        #                  top=False, labelbottom=False)
        # ax_r.tick_params(axis='y', which='both', left=False, 
        #                  top=False, labelleft=False)

        fig.tight_layout()
        fig.savefig(target[0], dpi=600)
        plt.close()


class TaskComputePerturbationPowers(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects):
        super(TaskComputePerturbationPowers, self).__init__(study)
        self.name = 'compute_perturbation_powers'
        self.subjects = subjects
        self.times = study.times
        self.torques = study.plot_torques
        self.subtalars = study.plot_subtalars
        self.rise = study.rise
        self.fall = study.fall

        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'perturbation_powers')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)

        deps = list()
        self.solution_map = dict()
        self.perturbation_map = dict()
        self.multiindex_tuples = dict()

        index = 0
        for time in self.times:
            for torque, subtalar in zip(self.torques, self.subtalars):
                for subject in self.subjects:
                    label = (f'torque{torque}_time{time}'
                             f'_rise{self.rise}_fall{self.fall}{subtalar}')

                    deps.append(
                        os.path.join(self.study.config['results_path'], 
                            'perturbed', f'perturbed_{label}', subject, 
                            f'perturbed_{label}.sto'))
                    self.solution_map[label] = index
                    index += 1

                    deps.append(
                        os.path.join(self.study.config['results_path'], 
                            'perturbed', f'perturbed_{label}', subject, 
                            'ankle_perturbation_curve.sto'))
                    self.perturbation_map[label] = index
                    index += 1

                    self.multiindex_tuples.append((
                            f'time{time}', 
                            f'torque{torque}{subtalar}',
                            subject))

        self.add_action(deps, 
                        [os.path.join(self.analysis_path, 
                            'perturbation_powers_mean.csv'),
                        os.path.join(self.analysis_path, 
                            'perturbation_powers_std.csv')], 
                        self.compute_perturbation_powers)

    def compute_perturbation_powers(self, file_dep, target):

        from scipy.interpolate import interp1d
        def compute_power(torque, speed, torqueTime, speedTime):
            torque_interp = interp1d(torqueTime, torque)
            torque_sampled = torque_interp(speedTime)

            return np.multiply(speed, torque_sampled)

        from collections import OrderedDict
        peak_pos_power = OrderedDict()
        peak_neg_power = OrderedDict()
        avg_pos_power = OrderedDict()
        avg_neg_power = OrderedDict()
        for odict in [peak_pos_power, peak_neg_power, avg_pos_power, avg_neg_power]:
            odict['ankle'] = list()
            odict['subtalar'] = list()

        for time in self.times:
            onset_time = time - 10
            offset_time = time + 5
            for torque, subtalar in zip(self.torques, self.subtalars):
                for subject in self.subjects:
        
                    label = (f'torque{torque}_time{time}'
                             f'_rise{self.rise}_fall{self.fall}{subtalar}')

                    solution = osim.TimeSeriesTable(
                        file_dep[self.solution_map[label]])
                    timeSol = np.array(solution.getIndependentColumn())
                    pgcSol = 100 * (timeSol - timeSol[0]) / (timeSol[-1] - timeSol[0])
                    istart = np.argmin(np.abs(pgcSol - onset_time))
                    iend = np.argmin(np.abs(pgcSol - offset_time))

                    perturbation = osim.TimeSeriesTable(
                        file_dep[self.perturbation_map[label]])
                    timePerturb = np.array(perturbation.getIndependentColumn())

                    ankleSpeed = solution.getDependentColumn(
                        '/jointset/ankle_r/ankle_angle_r/speed').to_numpy()
                    ankleTorque = perturbation.getDependentColumn(
                        '/forceset/perturbation_ankle_angle_r').to_numpy()
                    anklePower = compute_power(ankleTorque, ankleSpeed, 
                                               timePerturb, timeSol)[istart:iend]

                    peak_pos_power['ankle'].append(np.max(anklePower[anklePower > 0]))
                    peak_neg_power['ankle'].append(np.max(-anklePower[anklePower < 0]))
                    avg_neg_power['ankle'].append(np.mean(anklePower[anklePower > 0]))
                    avg_neg_power['ankle'].append(np.mean(anklePower[anklePower < 0]))

                    if 'subtalar' in subtalar:
                        subtalarTorque = perturbation.getDependentColumn(
                            '/forceset/perturbation_ankle_angle_r').to_numpy()
                        subtalarSpeed = solution.getDependentColumn(
                            '/jointset/ankle_r/ankle_angle_r/speed').to_numpy()
                        subtalarPower = compute_power(subtalarTorque, subtalarSpeed, 
                                                      timePerturb, timeSol)[istart:iend]
                        peak_pos_power['subtalar'].append(
                            np.max(subtalarPower[subtalarPower > 0]))
                        peak_neg_power['subtalar'].append(
                            np.max(-subtalarPower[subtalarPower < 0]))
                        avg_pos_power['subtalar'].append(
                            np.mean(subtalarPower[subtalarPower > 0]))
                        avg_neg_power['subtalar'].append(
                            np.mean(subtalarPower[subtalarPower < 0]))
                    else:
                        peak_pos_power['subtalar'].append(0)
                        peak_neg_power['subtalar'].append(0)
                        avg_pos_power['subtalar'].append(0)
                        avg_neg_power['subtalar'].append(0)

        index = pd.MultiIndex.from_tuples(self.multiindex_tuples,
                names=['time', 'perturbation', 'subject'])

        df_peak_pos_power = pd.DataFrame(peak_pos_power, index=index)
        df_peak_neg_power = pd.DataFrame(peak_neg_power, index=index)
        df_avg_pos_power = pd.DataFrame(avg_pos_power, index=index)
        df_avg_neg_power = pd.DataFrame(avg_neg_power, index=index)

        coords = ['ankle', 'subtalar']
        df_peak_pos_power_by_subj = df_peak_pos_power.groupby(level='subject').mean()
        peak_pos_power_mean = df_peak_pos_power_by_subj.mean()[coords]
        peak_pos_power_std = df_peak_pos_power_by_subj.std()[coords]

        df_peak_neg_power_by_subj = df_peak_neg_power.groupby(level='subject').mean()
        peak_neg_power_mean = df_peak_neg_power_by_subj.mean()[coords]
        peak_neg_power_std = df_peak_neg_power_by_subj.std()[coords]

        df_avg_pos_power_by_subj = df_avg_pos_power.groupby(level='subject').mean()
        avg_pos_power_mean = df_avg_pos_power_by_subj.mean()[coords]
        avg_pos_power_std = df_avg_pos_power_by_subj.std()[coords]

        df_avg_neg_power_by_subj = df_avg_neg_power.groupby(level='subject').mean()
        avg_neg_power_mean = df_avg_neg_power_by_subj.mean()[coords]
        avg_neg_power_std = df_avg_neg_power_by_subj.std()[coords]

        columns = ['peak positive', 'peak negative', 'average positive', 'average negative']

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


class TaskPlotCenterOfMass(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, time, rise, fall):
        super(TaskPlotCenterOfMass, self).__init__(study)
        self.name = f'plot_center_of_mass_time{time}_rise{rise}_fall{fall}'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'center_of_mass', f'time{time}_rise{rise}_fall{fall}')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)

        self.subjects = subjects
        self.time = time
        self.rise = rise
        self.fall = fall
        self.torques = study.plot_torques
        self.subtalars = study.plot_subtalars
        self.colors = ['gray'] + study.plot_colors
        self.legend_labels = ['unperturbed',
                              'eversion', 
                              'plantarflexion + eversion', 
                              'plantarflexion', 
                              'plantarflexion + inversion', 
                              'inversion']
        self.labels = list()
        self.labels.append('unperturbed')
        deps = list()
        for isubj, subject in enumerate(subjects):

            # Unperturbed COM
            deps.append(
                os.path.join(
                    self.study.config['results_path'], 
                    'unperturbed', subject,
                    'center_of_mass_unperturbed.sto'))

            # Perturbed COM
            for torque, subtalar in zip(self.torques, self.subtalars):
                label = (f'perturbed_torque{torque}_time{self.time}'
                        f'_rise{self.rise}_fall{self.fall}{subtalar}')
                deps.append(
                    os.path.join(
                        self.study.config['results_path'], 
                        'perturbed', label, subject,
                        f'center_of_mass_{label}.sto')
                    )

                if not isubj:
                    self.labels.append(
                        (f'torque{torque}_time{self.time}'
                         f'_rise{self.rise}_fall{self.fall}{subtalar}'))

        self.add_action(deps, 
                        [os.path.join(self.analysis_path, 
                            'center_of_mass_AP.png'),
                         os.path.join(self.analysis_path, 
                            'center_of_mass_SI.png'),
                         os.path.join(self.analysis_path, 
                            'center_of_mass_ML.png')], 
                        self.plot_center_of_mass)

    def plot_center_of_mass(self, file_dep, target):

        # Initialize figures
        # ------------------
        fig0 = plt.figure(figsize=(4, 5))
        ax_accx = fig0.add_subplot(2,1,1)
        ax_velx = fig0.add_subplot(2,1,2)

        fig1 = plt.figure(figsize=(4, 5))
        ax_accy = fig1.add_subplot(2,1,1)
        ax_vely = fig1.add_subplot(2,1,2)

        fig2 = plt.figure(figsize=(4, 5))
        ax_accz = fig2.add_subplot(2,1,1)
        ax_velz = fig2.add_subplot(2,1,2)

        # Plot formatting
        # ---------------
        for ax in [ax_velx, ax_vely, ax_velz,
                   ax_accx, ax_accy, ax_accz]:
            ax.axvline(x=self.time, color='gray', linestyle='-',
                linewidth=0.25, alpha=1.0, zorder=0)
            util.publication_spines(ax)
            xlim = [self.time-self.rise, self.time+self.fall]
            ax.set_xlim(xlim)
            ax.set_xticks(get_ticks_from_lims(xlim, 5))
            ax.spines['left'].set_position(('outward', 10))

        for ax in [ax_velx, ax_vely, ax_velz]:
            ax.spines['bottom'].set_position(('outward', 10))
            ax.set_xlabel('time (% gait cycle)')

        for ax in [ax_accx, ax_accy, ax_accz]: 
            ax.set_xticklabels([])
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both', bottom=False, 
                           top=False, labelbottom=False)

        # Aggregate data
        # --------------
        numLabels = len(self.labels)
        import collections
        com_dict = collections.defaultdict(dict)
        N = 1001
        for label in self.labels:
            com_dict[label]['velx'] = np.zeros((N, len(self.subjects)))
            com_dict[label]['vely'] = np.zeros((N, len(self.subjects)))
            com_dict[label]['velz'] = np.zeros((N, len(self.subjects)))
            com_dict[label]['accx'] = np.zeros((N, len(self.subjects)))
            com_dict[label]['accy'] = np.zeros((N, len(self.subjects)))
            com_dict[label]['accz'] = np.zeros((N, len(self.subjects)))

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
                vely = table.getDependentColumn('/|com_velocity_y').to_numpy()[irise:ifall]
                velz = table.getDependentColumn('/|com_velocity_z').to_numpy()[irise:ifall]
                accx = table.getDependentColumn('/|com_acceleration_x').to_numpy()[irise:ifall]
                accy = table.getDependentColumn('/|com_acceleration_y').to_numpy()[irise:ifall]
                accz = table.getDependentColumn('/|com_acceleration_z').to_numpy()[irise:ifall]

                timeInterp = np.linspace(timeVec[irise], timeVec[ifall-1], N)
                
                com_dict[label]['velx'][:, isubj] = np.interp(timeInterp, timeVec[irise:ifall], velx)
                com_dict[label]['vely'][:, isubj] = np.interp(timeInterp, timeVec[irise:ifall], vely)
                com_dict[label]['velz'][:, isubj] = np.interp(timeInterp, timeVec[irise:ifall], velz)
                com_dict[label]['accx'][:, isubj] = np.interp(timeInterp, timeVec[irise:ifall], accx)
                com_dict[label]['accy'][:, isubj] = np.interp(timeInterp, timeVec[irise:ifall], accy)
                com_dict[label]['accz'][:, isubj] = np.interp(timeInterp, timeVec[irise:ifall], accz)


        # Plotting
        # --------
        velx_mean = np.zeros((N, len(self.labels)))
        vely_mean = np.zeros((N, len(self.labels)))
        velz_mean = np.zeros((N, len(self.labels)))
        accx_mean = np.zeros((N, len(self.labels)))
        accy_mean = np.zeros((N, len(self.labels)))
        accz_mean = np.zeros((N, len(self.labels)))
        for ilabel, label in enumerate(self.labels):
            velx_mean[:, ilabel] = np.mean(com_dict[label]['velx'], axis=1)
            vely_mean[:, ilabel] = np.mean(com_dict[label]['vely'], axis=1)
            velz_mean[:, ilabel] = np.mean(com_dict[label]['velz'], axis=1)
            accx_mean[:, ilabel] = np.mean(com_dict[label]['accx'], axis=1)
            accy_mean[:, ilabel] = np.mean(com_dict[label]['accy'], axis=1)
            accz_mean[:, ilabel] = np.mean(com_dict[label]['accz'], axis=1)

        velx_step = 0.01
        vely_step = 0.01
        velz_step = 0.01
        accx_step = 0.1
        accy_step = 0.1
        accz_step = 0.1
        velx_lim = [np.mean(velx_mean), np.mean(velx_mean)]
        vely_lim = [np.mean(vely_mean), np.mean(vely_mean)]
        velz_lim = [np.mean(velz_mean), np.mean(velz_mean)]
        accx_lim = [np.mean(accx_mean), np.mean(accx_mean)]
        accy_lim = [np.mean(accy_mean), np.mean(accy_mean)]
        accz_lim = [np.mean(accz_mean), np.mean(accz_mean)]
        for ilabel, label in enumerate(self.labels): 
            update_lims(velx_mean[:, ilabel], velx_step, velx_lim)
            update_lims(vely_mean[:, ilabel], vely_step, vely_lim)
            update_lims(velz_mean[:, ilabel], velz_step, velz_lim)
            update_lims(accx_mean[:, ilabel], accx_step, accx_lim)
            update_lims(accy_mean[:, ilabel], accy_step, accy_lim)
            update_lims(accz_mean[:, ilabel], accz_step, accz_lim)

        pgc = np.linspace(self.time - self.rise, self.time + self.fall, N)
        handles = list()
        for ilabel, (label, color) in enumerate(zip(self.labels, self.colors)):
            lw = 3 if 'unperturbed' in label else 2

            h, = ax_velx.plot(pgc, velx_mean[:, ilabel], label=label, color=color, 
                linewidth=lw, clip_on=False, solid_capstyle='round')
            handles.append(h)
            ax_velx.set_ylabel(r'velocity $[m/s]$')
            ax_velx.set_ylim(velx_lim)
            ax_velx.set_yticks(get_ticks_from_lims(velx_lim, velx_step))

            ax_vely.plot(pgc, vely_mean[:, ilabel], label=label, color=color, 
                linewidth=lw, clip_on=False, solid_capstyle='round')
            ax_vely.set_ylabel(r'velocity $[m/s]$')
            ax_vely.set_ylim(vely_lim)
            ax_vely.set_yticks(get_ticks_from_lims(vely_lim, vely_step))

            ax_velz.plot(pgc, velz_mean[:, ilabel], label=label, color=color, 
                linewidth=lw, clip_on=False, solid_capstyle='round')
            ax_velz.set_ylabel(r'velocity $[m/s]$')
            ax_velz.set_ylim(velz_lim)
            ax_velz.set_yticks(get_ticks_from_lims(velz_lim, velz_step))
                
            ax_accx.plot(pgc, accx_mean[:, ilabel], label=label, color=color, 
                linewidth=lw, clip_on=False, solid_capstyle='round')
            ax_accx.set_ylabel(r'acceleration $[m/s]$')
            ax_accx.set_ylim(accx_lim)
            ax_accx.set_yticks(get_ticks_from_lims(accx_lim, accx_step))

            ax_accy.plot(pgc, accy_mean[:, ilabel], label=label, color=color, 
                linewidth=lw, clip_on=False, solid_capstyle='round')
            ax_accy.set_ylabel(r'acceleration $[m/s]$')
            ax_accy.set_ylim(accy_lim)
            ax_accy.set_yticks(get_ticks_from_lims(accy_lim, accy_step))

            ax_accz.plot(pgc, accz_mean[:, ilabel], label=label, color=color, 
                linewidth=lw, clip_on=False, solid_capstyle='round')
            ax_accz.set_ylabel(r'acceleration $[m/s]$')
            ax_accz.set_ylim(accz_lim)
            ax_accz.set_yticks(get_ticks_from_lims(accz_lim, accz_step))

        for ax in [ax_accx, ax_accy, ax_accz]:
            ax.legend(handles, self.legend_labels, loc='best', 
                frameon=True, prop={'size': 5})

        fig0.tight_layout()
        fig0.savefig(target[0], dpi=600)
        plt.close()

        fig1.tight_layout()
        fig1.savefig(target[1], dpi=600)
        plt.close()

        fig2.tight_layout()
        fig2.savefig(target[2], dpi=600)
        plt.close()


class TaskPlotMethodsFigure(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects):
        super(TaskPlotMethodsFigure, self).__init__(study)
        self.name = 'plot_methods_figure'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'methods_figure')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)

        self.subjects = subjects
        self.time = 35
        self.rise = study.rise
        self.fall = study.fall
        self.labels = ['unperturbed', 'perturbed']
        self.colors = ['dimgrey', 'darkred']
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
                        self.plot_center_of_mass)

    def plot_center_of_mass(self, file_dep, target):

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
                    arrowstyle=arrowstyle, mutation_scale=7.5, shrinkA=2.5, shrinkB=2.5,
                    joinstyle='miter', color='black', clip_on=False, zorder=2.5, lw=lw)
            ax.add_patch(arrow)

        def set_arrow_patch_coronal(ax, x, y, dx, dy, actu):
            point1 = ax.transData.transform((x, y))
            point2 = ax.transData.transform((x + dx, y + dx))
            delta = point2 - point1
            scale = delta[0] / delta[1]

            if 'muscles' in actu:
                arrowstyle = patches.ArrowStyle.CurveFilledB(head_length=0.4, 
                    head_width=0.15)
                lw = 2.0
            elif 'torques' in actu: 
                arrowstyle = patches.ArrowStyle.CurveB()
                lw = 0.75

            arrow = patches.FancyArrowPatch((x, y), (x + dx, y + scale*dy),
                    arrowstyle=arrowstyle, mutation_scale=10, shrinkA=0, shrinkB=0,
                    capstyle='round', joinstyle='miter', 
                    color=color, clip_on=False, zorder=2.5, lw=lw)
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
        ax_vel.set_xticklabels(['torque\nonset', '', 'peak\ntorque', 'torque\noffset'],
            fontsize=xfs)

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
        ax_tau.plot(pgc, tauInterp, color='darkred', linewidth=2, 
            clip_on=False, solid_capstyle='round')
        ax_tau.set_ylabel(r'perturbation torque $[\frac{N\cdot m}{kg}]$', fontsize=yfs)
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
            ax_vel.set_ylabel('center-of-mass\nvelocity', fontsize=yfs)
            ax_vel.set_ylim(vel_lim)
            ax_vel.set_yticks(get_ticks_from_lims(vel_lim, vel_step))
                
            ax_acc.plot(pgc, acc_mean[:, ilabel], label=label, color=color, 
                linewidth=lw, clip_on=False, solid_capstyle='round')
            ax_acc.set_ylabel('center-of-mass\nacceleration', fontsize=yfs)
            ax_acc.set_ylim(acc_lim)
            ax_acc.set_yticks(get_ticks_from_lims(acc_lim, acc_step))


        # Plot decorations
        ax_acc.text(self.time-6.5, -0.195, 'unperturbed', fontsize=6,
            color=self.colors[0], fontweight='bold')
        ax_acc.text(self.time-3.5, -0.475, 'perturbed', fontsize=6,
            color=self.colors[1], fontweight='bold')

        index = np.argmin(np.abs(pgc-self.time))
        set_arrow_patch_vertical(ax_acc, self.time, 
            acc_mean[index, 0], acc_mean[index, 1])

        index = np.argmin(np.abs(pgc-(self.time+self.fall)))
        set_arrow_patch_vertical(ax_vel, self.time+self.fall, 
            vel_mean[index, 0], vel_mean[index, 1])

        ax_acc.text(self.time+0.25, -0.21, r'$\Delta a_{COM}$', fontsize=7,
            color='black', fontweight='bold')

        vloc = (vel_mean[index, 0] + vel_mean[index, 1]) / 2.0
        ax_vel.text(self.time+self.fall-2.5, vloc, r'$\Delta v_{COM}$', fontsize=7,
            color='black', fontweight='bold')

        fig0.tight_layout()
        fig0.savefig(target[0], dpi=600)
        plt.close()


class TaskPlotInstantaneousCenterOfMass(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, times, rise, fall):
        super(TaskPlotInstantaneousCenterOfMass, self).__init__(study)
        self.name = f'plot_instantaneous_center_of_mass_rise{rise}_fall{fall}'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'center_of_mass_instantaneous',  f'rise{rise}_fall{fall}')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)
        self.subjects = subjects
        self.times = times
        self.rise = rise
        self.fall = fall
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
                for torque, subtalar in zip(self.torques, self.subtalars):
                    label = (f'perturbed_torque{torque}_time{time}'
                            f'_rise{self.rise}_fall{self.fall}{subtalar}')
                    deps.append(
                        os.path.join(
                            self.study.config['results_path'], 
                            'perturbed', label, subject,
                            f'center_of_mass_{label}.sto')
                        )

                    if not isubj:
                        self.labels.append(
                            (f'torque{torque}_time{time}'
                             f'_rise{self.rise}_fall{self.fall}{subtalar}'))
                        self.times_list.append(time)

        targets = list()
        for kin in ['pos', 'vel', 'acc']:
            # for direc in ['AP', 'SI', 'ML']:
            targets += [os.path.join(self.analysis_path, 
                        f'instant_com_{kin}.png')]

        self.add_action(deps, targets, self.plot_instantaneous_com)

    def plot_instantaneous_com(self, file_dep, target):

        # Initialize figures
        # ------------------
        figs = list()
        axes = list()
        for kin in ['pos', 'vel', 'acc']:
            fig = plt.figure(figsize=(6, 9))
            for idirec, direc in enumerate(['AP', 'SI', 'ML']):
                ax = fig.add_subplot(3, 1, idirec + 1)
                ax.axhline(y=0, color='black', linestyle='-',
                        linewidth=0.1, alpha=1.0, zorder=-1)
                ax.spines['left'].set_position(('outward', 30))
                ax.set_xticks(np.arange(len(self.times)))
                ax.set_xlim(0, len(self.times)-1)
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
                    ax.set_xlabel('peak perturbation time\n(% gait cycle)')

                axes.append(ax)
            figs.append(fig)

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
            zipped = zip(self.torques, self.subtalars, self.colors, self.shifts)
            for isubt, (torque, subtalar, color, shift) in enumerate(zipped):

                label = (f'torque{torque}_time{time}'
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


        pos_x_step = 0.02 # cm
        pos_y_step = 0.05 # cm
        pos_z_step = 0.02 # cm
        vel_x_step = 0.005
        vel_y_step = 0.01
        vel_z_step = 0.005
        acc_x_step = 0.1
        acc_y_step = 0.1
        acc_z_step = 0.1
        pos_x_lim = [0.0, 0.0]
        pos_y_lim = [0.0, 0.0]
        pos_z_lim = [0.0, 0.0]
        vel_x_lim = [0.0, 0.0]
        vel_y_lim = [0.0, 0.0]
        vel_z_lim = [0.0, 0.0]
        acc_x_lim = [0.0, 0.0]
        acc_y_lim = [0.0, 0.0]
        acc_z_lim = [0.0, 0.0]
        update_lims(pos_x_diff_mean-pos_x_diff_std, pos_x_step, pos_x_lim, mirror=True)
        update_lims(pos_y_diff_mean-pos_y_diff_std, pos_y_step, pos_y_lim)
        update_lims(pos_z_diff_mean-pos_z_diff_std, pos_z_step, pos_z_lim)
        update_lims(vel_x_diff_mean-vel_x_diff_std, vel_x_step, vel_x_lim, mirror=True)
        update_lims(vel_y_diff_mean-vel_y_diff_std, vel_y_step, vel_y_lim)
        update_lims(vel_z_diff_mean-vel_z_diff_std, vel_z_step, vel_z_lim)
        update_lims(acc_x_diff_mean-acc_x_diff_std, acc_x_step, acc_x_lim, mirror=True)
        update_lims(acc_y_diff_mean-acc_y_diff_std, acc_y_step, acc_y_lim)
        update_lims(acc_z_diff_mean-acc_z_diff_std, acc_z_step, acc_z_lim)        
        update_lims(pos_x_diff_mean+pos_x_diff_std, pos_x_step, pos_x_lim, mirror=True)
        update_lims(pos_y_diff_mean+pos_y_diff_std, pos_y_step, pos_y_lim)
        update_lims(pos_z_diff_mean+pos_z_diff_std, pos_z_step, pos_z_lim)
        update_lims(vel_x_diff_mean+vel_x_diff_std, vel_x_step, vel_x_lim, mirror=True)
        update_lims(vel_y_diff_mean+vel_y_diff_std, vel_y_step, vel_y_lim)
        update_lims(vel_z_diff_mean+vel_z_diff_std, vel_z_step, vel_z_lim)
        update_lims(acc_x_diff_mean+acc_x_diff_std, acc_x_step, acc_x_lim, mirror=True)
        update_lims(acc_y_diff_mean+acc_y_diff_std, acc_y_step, acc_y_lim)
        update_lims(acc_z_diff_mean+acc_z_diff_std, acc_z_step, acc_z_lim)
        handles_pos = list()
        handles_vel = list()
        handles_acc = list()
        for itime, time in enumerate(self.times):
            zipped = zip(self.torques, self.subtalars, self.colors, self.shifts, self.hatches)
            for isubt, (torque, subtalar, color, shift, hatch) in enumerate(zipped):

                # Set the x-position for these bar chart entries.
                x = itime + shift
                lw = 0.1

                # Instantaneous positions
                # -----------------------
                plot_errorbar(axes[0], x, pos_x_diff_mean[itime, isubt], pos_x_diff_std[itime, isubt])
                h_pos = axes[0].bar(x, pos_x_diff_mean[itime, isubt], self.width, color=color, clip_on=False, 
                    hatch=hatch, edgecolor=self.edgecolor, lw=lw)
                handles_pos.append(h_pos)
                axes[0].set_ylabel(r'$\Delta$' + ' fore-aft position $[cm]$')
                
                plot_errorbar(axes[1], x, pos_y_diff_mean[itime, isubt], pos_y_diff_std[itime, isubt])
                axes[1].bar(x, pos_y_diff_mean[itime, isubt], self.width, color=color, clip_on=False, 
                    hatch=hatch, edgecolor=self.edgecolor , lw=lw)
                axes[1].set_ylabel(r'$\Delta$' + ' vertical position $[cm]$')

                plot_errorbar(axes[2], x, pos_z_diff_mean[itime, isubt], pos_z_diff_std[itime, isubt])
                axes[2].bar(x, pos_z_diff_mean[itime, isubt], self.width, color=color, clip_on=False, 
                    hatch=hatch, edgecolor=self.edgecolor , lw=lw)
                axes[2].set_ylabel(r'$\Delta$' + ' medio-lateral position $[cm]$')

                # Instantaneous velocities
                # ------------------------
                plot_errorbar(axes[3], x, vel_x_diff_mean[itime, isubt], vel_x_diff_std[itime, isubt])
                h_vel = axes[3].bar(x, vel_x_diff_mean[itime, isubt], self.width, color=color, clip_on=False, 
                    hatch=hatch, edgecolor=self.edgecolor, lw=lw)
                handles_vel.append(h_vel)
                axes[3].set_ylabel(r'$\Delta$' + ' fore-aft velocity $[m/s]$')

                plot_errorbar(axes[4], x, vel_y_diff_mean[itime, isubt], vel_y_diff_std[itime, isubt])
                axes[4].bar(x, vel_y_diff_mean[itime, isubt], self.width, color=color, clip_on=False, 
                    hatch=hatch, edgecolor=self.edgecolor, lw=lw)
                axes[4].set_ylabel(r'$\Delta$' + ' vertical velocity $[m/s]$')

                plot_errorbar(axes[5], x, vel_z_diff_mean[itime, isubt], vel_z_diff_std[itime, isubt])
                axes[5].bar(x, vel_z_diff_mean[itime, isubt], self.width, color=color, clip_on=False, 
                    hatch=hatch, edgecolor=self.edgecolor, lw=lw)
                axes[5].set_ylabel(r'$\Delta$' + ' medio-lateral velocity $[m/s]$')

                # Instantaneous accelerations
                # ---------------------------
                plot_errorbar(axes[6], x, acc_x_diff_mean[itime, isubt], acc_x_diff_std[itime, isubt])
                h_acc = axes[6].bar(x, acc_x_diff_mean[itime, isubt], self.width, color=color, clip_on=False, 
                    hatch=hatch, edgecolor=self.edgecolor, lw=lw)
                handles_acc.append(h_acc)
                axes[6].set_ylabel(r'$\Delta$' + ' fore-aft acceleration $[m/s^2]$')

                plot_errorbar(axes[7], x, acc_y_diff_mean[itime, isubt], acc_y_diff_std[itime, isubt])
                axes[7].bar(x, acc_y_diff_mean[itime, isubt], self.width, color=color, clip_on=False,
                    hatch=hatch, edgecolor=self.edgecolor, lw=lw)
                axes[7].set_ylabel(r'$\Delta$' + ' vertical acceleration $[m/s^2]$')

                plot_errorbar(axes[8], x, acc_z_diff_mean[itime, isubt], acc_z_diff_std[itime, isubt])
                axes[8].bar(x, acc_z_diff_mean[itime, isubt], self.width, color=color, clip_on=False, 
                    hatch=hatch, edgecolor=self.edgecolor, lw=lw)
                axes[8].set_ylabel(r'$\Delta$' + ' medio-lateral acceleration $[m/s^2]$')

        axes[0].set_ylim(pos_x_lim)
        axes[0].set_yticks(get_ticks_from_lims(pos_x_lim, pos_x_step))
        axes[0].legend(handles_pos, self.legend_labels, loc='upper left', 
            title='perturbation', frameon=True)
        axes[1].set_ylim(pos_y_lim)
        axes[1].set_yticks(get_ticks_from_lims(pos_y_lim, pos_y_step))
        axes[2].set_ylim(pos_z_lim)
        axes[2].set_yticks(get_ticks_from_lims(pos_z_lim, pos_z_step))
        axes[3].set_ylim(vel_x_lim)
        axes[3].set_yticks(get_ticks_from_lims(vel_x_lim, vel_x_step))
        axes[3].legend(handles_vel, self.legend_labels, loc='upper left', 
            title='perturbation', frameon=True)
        axes[4].set_ylim(vel_y_lim)
        axes[4].set_yticks(get_ticks_from_lims(vel_y_lim, vel_y_step))
        axes[5].set_ylim(vel_z_lim)
        axes[5].set_yticks(get_ticks_from_lims(vel_z_lim, vel_z_step))
        axes[6].set_ylim(acc_x_lim)
        axes[6].set_yticks(get_ticks_from_lims(acc_x_lim, acc_x_step))
        axes[6].legend(handles_acc, self.legend_labels, loc='upper left', 
            title='perturbation', frameon=True)
        axes[7].set_ylim(acc_y_lim)
        axes[7].set_yticks(get_ticks_from_lims(acc_y_lim, acc_y_step))
        axes[8].set_ylim(acc_z_lim)
        axes[8].set_yticks(get_ticks_from_lims(acc_z_lim, acc_z_step))      

        for ifig, fig in enumerate(figs):
            fig.tight_layout()
            fig.savefig(target[ifig], dpi=600)
            plt.close()


class TaskPlotInstantaneousCenterOfMassLumbarStiffness(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, times, torque, rise, fall):
        super(TaskPlotInstantaneousCenterOfMassLumbarStiffness, self).__init__(study)
        self.name = f'plot_instantaneous_center_of_mass_lumbar_stiffness_torque{torque}_rise{rise}_fall{fall}'
        self.results_path = os.path.join(study.config['results_path'], 'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'lumbar_stiffness',  f'lumbar_stiffness_torque{torque}_rise{rise}_fall{fall}')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)
        self.subjects = subjects
        self.torque = torque
        self.times = times
        self.rise = rise
        self.fall = fall
        self.subtalars = study.subtalar_suffixes
        self.width = 0.25
        self.lumbars = list()
        for stiffness in study.lumbar_stiffnesses:
            lumbar = ''
            if not stiffness == 1.0:
                lumbar = f'lumbar{stiffness}'
            self.lumbars.append(lumbar)
        min_width = -self.width*((len(self.lumbars)-1)/2)
        max_width = -min_width
        self.lumbar_shifts = np.linspace(min_width, max_width, len(self.lumbars))
        cmap = plt.get_cmap('viridis')
        indices = np.linspace(0, 1.0, len(self.lumbars)) 
        self.lumbar_colors = [cmap(idx) for idx in indices]
        self.labels = list()
        self.times_list = list()

        deps = list()
        for isubj, subject in enumerate(subjects):
            for lumbar in self.lumbars:
                # Unperturbed solutions
                deps.append(
                    os.path.join(
                        self.study.config['results_path'], 
                        lumbar, 'unperturbed', subject,
                        f'center_of_mass_unperturbed{lumbar}.sto'))

                self.labels.append(f'unperturbed{lumbar}')
                self.times_list.append(100)

                # Perturbed solutions
                for time in self.times:
                    for subtalar in self.subtalars:
                        label = (f'perturbed_torque{torque}_time{time}'
                                f'_rise{self.rise}_fall{self.fall}{subtalar}_{lumbar}')
                        deps.append(
                            os.path.join(
                                self.study.config['results_path'], 
                                lumbar, label, subject,
                                f'center_of_mass_{label}.sto')
                            )

                        if not isubj:
                            self.labels.append(
                                (f'torque{self.torque}_time{time}'
                                 f'_rise{self.rise}_fall{self.fall}{subtalar}_{lumbar}'))
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
                fig = plt.figure(figsize=(4, 2.5*len(self.subtalars)))
                these_axes = list()
                for isubt, subt in enumerate(self.subtalars):
                    ax = fig.add_subplot(3, 1, isubt+1)
                    these_axes.append(ax)

                figs.append(fig)
                axes.append(these_axes)

        # Aggregate data
        # --------------
        import collections
        com_dict = collections.defaultdict(dict)
        time_dict = dict()
        index = 0
        for isubj, subject in enumerate(self.subjects):
            for ilumbar, lumbar in enumerate(self.lumbars):
                # Unperturbed center-of-mass trajectories
                # iunp = isubj*numLumbars + ilumbar*(numTimes*numSubtalars + 1)
                unpTable = osim.TimeSeriesTable(file_dep[index])
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

                index = index + 1
                for time in self.times:
                    for subtalar in self.subtalars:
                        label = (f'perturbed_torque{self.torque}_time{time}'
                                 f'_rise{self.rise}_fall{self.fall}{subtalar}_{lumbar}')

                        # Perturbed center-of-mass trajectories
                        table = osim.TimeSeriesTable(file_dep[index])
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

                        index = index + 1

        # Plotting
        # --------
        for isubt, subtalar in enumerate(self.subtalars):

            pos_x_diff = np.zeros(len(self.subjects))
            pos_y_diff = np.zeros(len(self.subjects))
            pos_z_diff = np.zeros(len(self.subjects))
            vel_x_diff = np.zeros(len(self.subjects))
            vel_y_diff = np.zeros(len(self.subjects))
            vel_z_diff = np.zeros(len(self.subjects))
            acc_x_diff = np.zeros(len(self.subjects))
            acc_y_diff = np.zeros(len(self.subjects))
            acc_z_diff = np.zeros(len(self.subjects))
            pos_x_diff_mean = np.zeros((len(self.times), len(self.lumbars)))
            pos_y_diff_mean = np.zeros((len(self.times), len(self.lumbars)))
            pos_z_diff_mean = np.zeros((len(self.times), len(self.lumbars)))
            vel_x_diff_mean = np.zeros((len(self.times), len(self.lumbars)))
            vel_y_diff_mean = np.zeros((len(self.times), len(self.lumbars)))
            vel_z_diff_mean = np.zeros((len(self.times), len(self.lumbars)))
            acc_x_diff_mean = np.zeros((len(self.times), len(self.lumbars)))
            acc_y_diff_mean = np.zeros((len(self.times), len(self.lumbars)))
            acc_z_diff_mean = np.zeros((len(self.times), len(self.lumbars)))
            pos_x_diff_std = np.zeros((len(self.times), len(self.lumbars)))
            pos_y_diff_std = np.zeros((len(self.times), len(self.lumbars)))
            pos_z_diff_std = np.zeros((len(self.times), len(self.lumbars)))
            vel_x_diff_std = np.zeros((len(self.times), len(self.lumbars)))
            vel_y_diff_std = np.zeros((len(self.times), len(self.lumbars)))
            vel_z_diff_std = np.zeros((len(self.times), len(self.lumbars)))
            acc_x_diff_std = np.zeros((len(self.times), len(self.lumbars)))
            acc_y_diff_std = np.zeros((len(self.times), len(self.lumbars)))
            acc_z_diff_std = np.zeros((len(self.times), len(self.lumbars)))
            for itime, time in enumerate(self.times):
                zipped = zip(self.lumbars, self.lumbar_colors, self.lumbar_shifts)
                for ilumbar, (lumbar, color, shift) in enumerate(zipped):

                    label = (f'perturbed_torque{self.torque}_time{time}'
                             f'_rise{self.rise}_fall{self.fall}{subtalar}_{lumbar}')
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

                    pos_x_diff_mean[itime, ilumbar] = np.mean(pos_x_diff)
                    pos_y_diff_mean[itime, ilumbar] = np.mean(pos_y_diff)
                    pos_z_diff_mean[itime, ilumbar] = np.mean(pos_z_diff)
                    vel_x_diff_mean[itime, ilumbar] = np.mean(vel_x_diff)
                    vel_y_diff_mean[itime, ilumbar] = np.mean(vel_y_diff)
                    vel_z_diff_mean[itime, ilumbar] = np.mean(vel_z_diff)
                    acc_x_diff_mean[itime, ilumbar] = np.mean(acc_x_diff)
                    acc_y_diff_mean[itime, ilumbar] = np.mean(acc_y_diff)
                    acc_z_diff_mean[itime, ilumbar] = np.mean(acc_z_diff)
                    pos_x_diff_std[itime, ilumbar] = np.std(pos_x_diff)
                    pos_y_diff_std[itime, ilumbar] = np.std(pos_y_diff)
                    pos_z_diff_std[itime, ilumbar] = np.std(pos_z_diff)
                    vel_x_diff_std[itime, ilumbar] = np.std(vel_x_diff)
                    vel_y_diff_std[itime, ilumbar] = np.std(vel_y_diff)
                    vel_z_diff_std[itime, ilumbar] = np.std(vel_z_diff)
                    acc_x_diff_std[itime, ilumbar] = np.std(acc_x_diff)
                    acc_y_diff_std[itime, ilumbar] = np.std(acc_y_diff)
                    acc_z_diff_std[itime, ilumbar] = np.std(acc_z_diff)

            pos_step = 0.2 # cm
            vel_step = 0.02
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
                zipped = zip(self.lumbars, self.lumbar_colors, self.lumbar_shifts)
                for ilumbar, (lumbar, color, shift) in enumerate(zipped):

                    # Set the x-position for these bar chart entries.
                    x = itime + shift

                    # Instantaneous positions
                    # -----------------------
                    plot_errorbar(axes[0][isubt], x, pos_x_diff_mean[itime, ilumbar], pos_x_diff_std[itime, ilumbar])
                    axes[0][isubt].bar(x, pos_x_diff_mean[itime, ilumbar], self.width, color=color, clip_on=False)
                    axes[0][isubt].set_ylabel(r'$\Delta$' + ' fore-aft position $[cm]$')
                    
                    plot_errorbar(axes[1][isubt], x, pos_y_diff_mean[itime, ilumbar], pos_y_diff_std[itime, ilumbar])
                    axes[1][isubt].bar(x, pos_y_diff_mean[itime, ilumbar], self.width, color=color, clip_on=False)
                    axes[1][isubt].set_ylabel(r'$\Delta$' + ' vertical position $[cm]$')

                    plot_errorbar(axes[2][isubt], x, pos_z_diff_mean[itime, ilumbar], pos_z_diff_std[itime, ilumbar])
                    axes[2][isubt].bar(x, pos_z_diff_mean[itime, ilumbar], self.width, color=color, clip_on=False)
                    axes[2][isubt].set_ylabel(r'$\Delta$' + ' medio-lateral position $[cm]$')

                    # Instantaneous velocities
                    # ------------------------
                    plot_errorbar(axes[3][isubt], x, vel_x_diff_mean[itime, ilumbar], vel_x_diff_std[itime, ilumbar])
                    axes[3][isubt].bar(x, vel_x_diff_mean[itime, ilumbar], self.width, color=color, clip_on=False)
                    axes[3][isubt].set_ylabel(r'$\Delta$' + ' fore-aft velocity $[m/s]$')

                    plot_errorbar(axes[4][isubt], x, vel_y_diff_mean[itime, ilumbar], vel_y_diff_std[itime, ilumbar])
                    axes[4][isubt].bar(x, vel_y_diff_mean[itime, ilumbar], self.width, color=color, clip_on=False)
                    axes[4][isubt].set_ylabel(r'$\Delta$' + ' vertical velocity $[m/s]$')

                    plot_errorbar(axes[5][isubt], x, vel_z_diff_mean[itime, ilumbar], vel_z_diff_std[itime, ilumbar])
                    axes[5][isubt].bar(x, vel_z_diff_mean[itime, ilumbar], self.width, color=color, clip_on=False)
                    axes[5][isubt].set_ylabel(r'$\Delta$' + ' medio-lateral velocity $[m/s]$')

                    # Instantaneous accelerations
                    # ---------------------------
                    plot_errorbar(axes[6][isubt], x, acc_x_diff_mean[itime, ilumbar], acc_x_diff_std[itime, ilumbar])
                    axes[6][isubt].bar(x, acc_x_diff_mean[itime, ilumbar], self.width, color=color, clip_on=False)
                    axes[6][isubt].set_ylabel(r'$\Delta$' + ' fore-aft acceleration $[m/s^2]$')

                    plot_errorbar(axes[7][isubt], x, acc_y_diff_mean[itime, ilumbar], acc_y_diff_std[itime, ilumbar])
                    axes[7][isubt].bar(x, acc_y_diff_mean[itime, ilumbar], self.width, color=color, clip_on=False)
                    axes[7][isubt].set_ylabel(r'$\Delta$' + ' vertical acceleration $[m/s^2]$')

                    plot_errorbar(axes[8][isubt], x, acc_z_diff_mean[itime, ilumbar], acc_z_diff_std[itime, ilumbar])
                    axes[8][isubt].bar(x, acc_z_diff_mean[itime, ilumbar], self.width, color=color, clip_on=False)
                    axes[8][isubt].set_ylabel(r'$\Delta$' + ' medio-lateral acceleration $[m/s^2]$')

            axes[0][isubt].set_ylim(pos_x_lim)
            axes[0][isubt].set_yticks(get_ticks_from_lims(pos_x_lim, pos_step))
            axes[1][isubt].set_ylim(pos_y_lim)
            axes[1][isubt].set_yticks(get_ticks_from_lims(pos_y_lim, pos_step))
            axes[2][isubt].set_ylim(pos_z_lim)
            axes[2][isubt].set_yticks(get_ticks_from_lims(pos_z_lim, pos_step))
            axes[3][isubt].set_ylim(vel_x_lim)
            axes[3][isubt].set_yticks(get_ticks_from_lims(vel_x_lim, vel_step))
            axes[4][isubt].set_ylim(vel_y_lim)
            axes[4][isubt].set_yticks(get_ticks_from_lims(vel_y_lim, vel_step))
            axes[5][isubt].set_ylim(vel_z_lim)
            axes[5][isubt].set_yticks(get_ticks_from_lims(vel_z_lim, vel_step))
            axes[6][isubt].set_ylim(acc_x_lim)
            axes[6][isubt].set_yticks(get_ticks_from_lims(acc_x_lim, acc_step))
            axes[7][isubt].set_ylim(acc_y_lim)
            axes[7][isubt].set_yticks(get_ticks_from_lims(acc_y_lim, acc_step))
            axes[8][isubt].set_ylim(acc_z_lim)
            axes[8][isubt].set_yticks(get_ticks_from_lims(acc_z_lim, acc_step))

        for these_axes in axes:
            for ax in these_axes:
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
    def __init__(self, study, subjects, times, rise, fall):
        super(TaskPlotCenterOfMassVector, self).__init__(study)
        self.name = f'plot_center_of_mass_vector_rise{rise}_fall{fall}'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
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
        self.planes = ['sagittal', 'transverse']

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

        self.add_action(deps, targets, self.plot_com_vectors)

    def plot_com_vectors(self, file_dep, target):

        # Globals
        # -------
        tick_fs = 7

        # Initialize figures
        # ------------------
        figs = list()
        axes = list()
        for kin in self.kinematic_levels:
            for iplane, plane in enumerate(self.planes):
                these_axes = list()
                if plane == 'transverse':
                    fig = plt.figure(figsize=(7, 5))
                    for itorque, torque in enumerate(self.torques):
                        ax = fig.add_subplot(1, len(self.torques), itorque + 1)
                        ax.grid(axis='y', color='gray', alpha=0.5, linewidth=0.5, 
                                zorder=-10, clip_on=False)
                        ax.axvline(x=0, color='gray', linestyle='-',
                                linewidth=0.5, alpha=0.5, zorder=-1)
                        ax.set_yticks(np.arange(len(self.times)))
                        ax.set_ylim(0, len(self.times)-1)
                        util.publication_spines(ax)

                        if not itorque:
                            ax.spines['left'].set_position(('outward', 10))
                            if kin == 'pos' or kin == 'vel':
                                ax.set_yticklabels([f'{time + 5}' for time in self.times],
                                    fontsize=tick_fs)
                            else:
                                ax.set_yticklabels([f'{time}' for time in self.times],
                                    fontsize=tick_fs)
                        else:
                            ax.spines['left'].set_visible(False)
                            ax.set_yticklabels([])
                            ax.yaxis.set_ticks_position('none')
                            ax.tick_params(axis='y', which='both', bottom=False, 
                                           top=False, labelbottom=False)

                        ax.spines['bottom'].set_position(('outward', 40))
                        ax.set_title(self.legend_labels[itorque], pad=25, fontsize=8)
                        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                        ax.tick_params(which='minor', axis='x', direction='in')
                        these_axes.append(ax)

                elif plane == 'sagittal':
                    fig = plt.figure(figsize=(5, 7))
                    for itorque, torque in enumerate(self.torques):
                        ax = fig.add_subplot(len(self.torques), 1, itorque + 1)
                        ax_r = ax.twinx()
                        ax_r.set_ylabel(self.legend_labels[itorque], 
                            rotation=270, labelpad=35, fontsize=8)

                        ax.grid(axis='x', color='gray', alpha=0.5, linewidth=0.5, 
                                zorder=-10, clip_on=False)
                        ax.axhline(y=0, color='gray', linestyle='-',
                                linewidth=0.5, alpha=0.5, zorder=-1)
                        ax.set_xticks(np.arange(len(self.times)))
                        ax.set_xlim(0, len(self.times)-1)
                        util.publication_spines(ax)

                        if itorque == len(self.torques)-1:
                            ax.spines['bottom'].set_position(('outward', 10))
                            if kin == 'pos' or kin == 'vel':
                                ax.set_xticklabels([f'{time + 5}' for time in self.times],
                                    fontsize=tick_fs)
                            else:
                                ax.set_xticklabels([f'{time}' for time in self.times],
                                    fontsize=tick_fs)
                        else:
                            ax.spines['bottom'].set_visible(False)
                            ax.set_xticklabels([])
                            ax.xaxis.set_ticks_position('none')
                            ax.tick_params(axis='x', which='both', bottom=False, 
                                           top=False, labelbottom=False)

                        ax.spines['left'].set_position(('outward', 20))
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
        def set_arrow_patch_sagittal(ax, x, y, dx, dy, actu):
            point1 = ax.transData.transform((x, y))
            point2 = ax.transData.transform((x + dy, y + dy))
            delta = point2 - point1
            scale = delta[1] / delta[0]

            if 'muscles' in actu:
                arrowstyle = patches.ArrowStyle.CurveFilledB(head_length=0.4, 
                    head_width=0.15)
                lw = 2.0
            elif 'torques' in actu: 
                arrowstyle = patches.ArrowStyle.CurveB()
                lw = 0.75

            arrow = patches.FancyArrowPatch((x, y), (x + scale*dx, y + dy),
                    arrowstyle=arrowstyle, mutation_scale=10, shrinkA=0, shrinkB=0,
                    capstyle='round', joinstyle='miter', 
                    color=color, clip_on=False, zorder=2.5, lw=lw)
            ax.add_patch(arrow)

        def set_arrow_patch_coronal(ax, x, y, dx, dy, actu):
            point1 = ax.transData.transform((x, y))
            point2 = ax.transData.transform((x + dx, y + dx))
            delta = point2 - point1
            scale = delta[0] / delta[1]

            if 'muscles' in actu:
                arrowstyle = patches.ArrowStyle.CurveFilledB(head_length=0.4, 
                    head_width=0.15)
                lw = 2.0
            elif 'torques' in actu: 
                arrowstyle = patches.ArrowStyle.CurveB()
                lw = 0.75

            arrow = patches.FancyArrowPatch((x, y), (x + dx, y + scale*dy),
                    arrowstyle=arrowstyle, mutation_scale=10, shrinkA=0, shrinkB=0,
                    capstyle='round', joinstyle='miter', 
                    color=color, clip_on=False, zorder=2.5, lw=lw)
            ax.add_patch(arrow)

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
                        #
                        # TODO: A given peak perturbation time (e.g 50% of the 
                        # gait cycle) may not lie exactly on a time point of 
                        # from the simulation. The interval between time points
                        # is 5ms, meaning that the time index could be up to 2.5ms
                        # away from the actual perturbation peak time. 
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
            scale = 0.005
            axes[0][iperturb].set_ylim(-scale, scale)
            axes[0][iperturb].set_yticks([-scale, 0, scale])
            axes[0][iperturb].set_yticklabels([-scale, 0, scale], fontsize=tick_fs)
            axes[0][2].set_ylabel(r'$\Delta$' + ' center-of-mass position $[-]$')
            axes[0][4].set_xlabel('perturbation offset time\n(% gait cycle)')

            # Transverse position
            scale = 0.002
            axes[1][iperturb].set_xlim(-scale, scale)
            axes[1][iperturb].set_xticks([-scale, 0, scale])
            axes[1][iperturb].set_xticklabels([-scale, 0, scale], fontsize=tick_fs-1)
            axes[1][2].set_xlabel(r'$\Delta$' + ' center-of-mass position $[-]$')
            axes[1][0].set_ylabel('perturbation offset time\n(% gait cycle)')

            # Sagittal velocity
            scale = 0.02
            axes[2][iperturb].set_ylim(-scale, scale)
            axes[2][iperturb].set_yticks([-scale, 0, scale])
            axes[2][iperturb].set_yticklabels([-scale, 0, scale], fontsize=tick_fs)
            axes[2][2].set_ylabel(r'$\Delta$' + ' center-of-mass velocity $[-]$')
            axes[2][4].set_xlabel('perturbation offset time\n(% gait cycle)')

            # Transverse velocity
            scale = 0.01
            axes[3][iperturb].set_xlim(-scale, scale)
            axes[3][iperturb].set_xticks([-scale, 0, scale])
            axes[3][iperturb].set_xticklabels([-scale, 0, scale], fontsize=tick_fs-1)
            axes[3][2].set_xlabel(r'$\Delta$' + ' center-of-mass velocity $[-]$')
            axes[3][0].set_ylabel('perturbation offset time\n(% gait cycle)')

            # Sagittal acceleration
            scale = 0.075
            axes[4][iperturb].set_ylim(-scale, scale)
            axes[4][iperturb].set_yticks([-scale, 0, scale])
            axes[4][iperturb].set_yticklabels([-scale, 0, scale], fontsize=tick_fs)
            axes[4][2].set_ylabel(r'$\Delta$' + ' center-of-mass acceleration $[-]$')
            axes[4][4].set_xlabel('perturbation peak time\n(% gait cycle)')

            # Transverse acceleration
            scale = 0.03
            axes[5][iperturb].set_xlim(-scale, scale)
            axes[5][iperturb].set_xticks([-scale, 0, scale])
            axes[5][iperturb].set_xticklabels([-scale, 0, scale], fontsize=tick_fs-1)
            axes[5][2].set_xlabel(r'$\Delta$' + ' center-of-mass acceleration $[-]$')
            axes[5][0].set_ylabel('perturbation peak time\n(% gait cycle)')
        
        scale = 0.01
        axes[2][0].set_ylim(-scale, scale)
        axes[2][4].set_ylim(-scale, scale)
        axes[2][0].set_yticks([-scale, 0, scale])
        axes[2][4].set_yticks([-scale, 0, scale])
        axes[2][0].set_yticklabels([-scale, 0, scale], fontsize=tick_fs)
        axes[2][4].set_yticklabels([-scale, 0, scale], fontsize=tick_fs)

        scale = 0.025
        axes[4][0].set_ylim(-scale, scale)
        axes[4][4].set_ylim(-scale, scale)
        axes[4][0].set_yticks([-scale, 0, scale])
        axes[4][4].set_yticks([-scale, 0, scale])
        axes[4][0].set_yticklabels([-scale, 0, scale], fontsize=tick_fs)
        axes[4][4].set_yticklabels([-scale, 0, scale], fontsize=tick_fs)
            

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
                    set_arrow_patch_sagittal(axes[0][iperturb], itime, 0, pos_x_diff, pos_y_diff, actu_key)
                    set_arrow_patch_coronal(axes[1][iperturb], 0, itime, pos_z_diff, pos_x_diff, actu_key)

                    # Velocity vectors
                    # ----------------
                    vel_x_diff = vel_x_diff_mean[actu_key][itime, iperturb]
                    vel_y_diff = vel_y_diff_mean[actu_key][itime, iperturb]
                    vel_z_diff = vel_z_diff_mean[actu_key][itime, iperturb]
                    set_arrow_patch_sagittal(axes[2][iperturb], itime, 0, vel_x_diff, vel_y_diff, actu_key)
                    set_arrow_patch_coronal(axes[3][iperturb], 0, itime, vel_z_diff, vel_x_diff, actu_key)

                    # Acceleration vectors
                    # --------------------
                    acc_x_diff = acc_x_diff_mean[actu_key][itime, iperturb]
                    acc_y_diff = acc_y_diff_mean[actu_key][itime, iperturb]
                    acc_z_diff = acc_z_diff_mean[actu_key][itime, iperturb]
                    set_arrow_patch_sagittal(axes[4][iperturb], itime, 0, acc_x_diff, acc_y_diff, actu_key)
                    set_arrow_patch_coronal(axes[5][iperturb], 0, itime, acc_z_diff, acc_x_diff, actu_key)


        figs[0].subplots_adjust(left=0.2, right=0.85, bottom=0.1, top=0.95, hspace=0.3)
        figs[0].savefig(target[0], dpi=600)

        figs[1].subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.8, wspace=0.3)
        figs[1].savefig(target[1], dpi=600)

        figs[2].subplots_adjust(left=0.2, right=0.85, bottom=0.1, top=0.95, hspace=0.3)
        figs[2].savefig(target[2], dpi=600)

        figs[3].subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.8, wspace=0.3)
        figs[3].savefig(target[3], dpi=600)

        figs[4].subplots_adjust(left=0.2, right=0.85, bottom=0.1, top=0.95, hspace=0.3)
        figs[4].savefig(target[4], dpi=600)

        figs[5].subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.8, wspace=0.3)
        figs[5].savefig(target[5], dpi=600)
 
        plt.close()


class TaskPlotGroundReactions(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, time, rise, fall):
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
        self.rise = rise
        self.fall = fall
        self.torques = study.plot_torques
        self.subtalars = study.plot_subtalars
        self.colors = ['dimgray'] + study.plot_colors
        self.legend_labels = ['unperturbed',
                              'eversion', 
                              'plantarflexion + eversion', 
                              'plantarflexion', 
                              'plantarflexion + inversion', 
                              'inversion']

        self.labels = list()
        self.labels.append('unperturbed')
        self.models = list()
        deps = list()
        for isubj, subject in enumerate(subjects):
            # Model
            self.models.append(os.path.join(
                self.study.config['results_path'], 
                'unperturbed', subject, 'model_unperturbed.osim'))

            # Unperturbed GRFs
            deps.append(
                os.path.join(
                    self.study.config['results_path'], 
                    'unperturbed', subject,
                    'unperturbed_grfs.sto'))

            # Perturbed GRFs
            for torque, subtalar in zip(self.torques, self.subtalars):
                label = (f'perturbed_torque{torque}_time{self.time}'
                        f'_rise{self.rise}_fall{self.fall}{subtalar}')
                deps.append(
                    os.path.join(
                        self.study.config['results_path'], 
                        'perturbed', label, subject,
                        f'{label}_grfs.sto')
                    )

                if not isubj:
                    self.labels.append(
                        (f'torque{torque}_time{self.time}'
                         f'_rise{self.rise}_fall{self.fall}{subtalar}'))

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
        fig0 = plt.figure(figsize=(5, 9))
        ax_f0_grfx = fig0.add_subplot(3, 1, 1)
        ax_f0_grfy = fig0.add_subplot(3, 1, 2)
        ax_f0_grfz = fig0.add_subplot(3, 1, 3)

        fig2 = plt.figure(figsize=(5, 3))
        ax_grfx = fig2.add_subplot(111)

        fig3 = plt.figure(figsize=(5, 3))
        ax_grfy = fig3.add_subplot(111)

        fig4 = plt.figure(figsize=(5, 3))
        ax_grfz = fig4.add_subplot(111)

        # Plot formatting
        # ---------------
        for ax in [ax_f0_grfx, ax_f0_grfy, ax_f0_grfz, ax_grfx, ax_grfy, ax_grfz]:
            ax.axvline(x=self.time, color='gray', linestyle='-',
                linewidth=0.25, alpha=1.0, zorder=0)
            util.publication_spines(ax)
            xlim = [self.time-self.rise, self.time+self.fall]
            ax.set_xlim(xlim)
            ax.set_xticks(get_ticks_from_lims(xlim, 5))
            ax.spines['left'].set_position(('outward', 10))

        for ax in [ax_f0_grfz, ax_grfx, ax_grfy, ax_grfz]:
            ax.set_xlabel('time (% gait cycle)')
            ax.spines['left'].set_position(('outward', 10))
            ax.spines['bottom'].set_position(('outward', 10))

        for ax in [ax_f0_grfx, ax_f0_grfy]:
            ax.set_xticklabels([])
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both', bottom=False, 
                           top=False, labelbottom=False)

        # Aggregate data
        # --------------
        numLabels = len(self.labels)
        import collections
        grf_dict = collections.defaultdict(dict)
        N = 1001
        for label in self.labels:
            grf_dict[label]['grfx'] = np.zeros((N, len(self.subjects)))
            grf_dict[label]['grfy'] = np.zeros((N, len(self.subjects)))
            grf_dict[label]['grfz'] = np.zeros((N, len(self.subjects)))

        for isubj, subject in enumerate(self.subjects):
            model = osim.Model(self.models[isubj])
            state = model.initSystem()
            mass = model.getTotalMass(state)
            BW = abs(model.getGravity()[1]) * mass
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

                rgrfx = table.getDependentColumn('ground_force_r_vx').to_numpy()[irise:ifall] / BW
                rgrfy = table.getDependentColumn('ground_force_r_vy').to_numpy()[irise:ifall] / BW
                rgrfz = table.getDependentColumn('ground_force_r_vz').to_numpy()[irise:ifall] / BW

                timeInterp = np.linspace(timeVec[irise], timeVec[ifall-1], N)
                
                grf_dict[label]['grfx'][:, isubj] = np.interp(timeInterp, timeVec[irise:ifall], rgrfx)
                grf_dict[label]['grfy'][:, isubj] = np.interp(timeInterp, timeVec[irise:ifall], rgrfy)
                grf_dict[label]['grfz'][:, isubj] = np.interp(timeInterp, timeVec[irise:ifall], rgrfz)


        # Plotting
        # --------
        rgrfx_mean = np.zeros((N, len(self.labels)))
        rgrfy_mean = np.zeros((N, len(self.labels)))
        rgrfz_mean = np.zeros((N, len(self.labels)))
        for ilabel, label in enumerate(self.labels):
            rgrfx_mean[:, ilabel] = np.mean(grf_dict[label]['grfx'], axis=1)
            rgrfy_mean[:, ilabel] = np.mean(grf_dict[label]['grfy'], axis=1)
            rgrfz_mean[:, ilabel] = np.mean(grf_dict[label]['grfz'], axis=1)

        grfx_step = 0.02
        grfy_step = 0.05
        grfz_step = 0.02
        grfx_lim = [np.mean(rgrfx_mean), np.mean(rgrfx_mean)]
        grfy_lim = [np.mean(rgrfy_mean), np.mean(rgrfy_mean)]
        grfz_lim = [np.mean(rgrfz_mean), np.mean(rgrfx_mean)]
        for ilabel, label in enumerate(self.labels): 
            update_lims(rgrfx_mean[:, ilabel], grfx_step, grfx_lim)
            update_lims(rgrfy_mean[:, ilabel], grfy_step, grfy_lim)
            update_lims(rgrfz_mean[:, ilabel], grfz_step, grfz_lim)

        pgc = np.linspace(self.time - self.rise, self.time + self.fall, N)
        handles = list()
        for ilabel, (label, color) in enumerate(zip(self.labels, self.colors)):
            lw = 3 if 'unperturbed' in label else 1.5

            for iax, ax in enumerate([ax_f0_grfx, ax_grfx]): 
                h, = ax.plot(pgc, rgrfx_mean[:, ilabel], label=label, color=color, 
                    linewidth=lw, clip_on=False, solid_capstyle='round')
                if not iax: handles.append(h)
                ax.set_ylabel(r'fore-aft ground reaction $[BW]$')
                ax.set_ylim(grfx_lim)
                ax.set_yticks(get_ticks_from_lims(grfx_lim, grfx_step))
                
            for iax, ax in enumerate([ax_f0_grfy, ax_grfy]):
                ax.plot(pgc, rgrfy_mean[:, ilabel], label=label, color=color, 
                        linewidth=lw, clip_on=False, solid_capstyle='round')
                ax.set_ylabel('vertical ground reaction $[BW]$')
                ax.set_ylim(grfy_lim)
                ax.set_yticks(get_ticks_from_lims(grfy_lim, grfy_step))

            for iax, ax in enumerate([ax_f0_grfz, ax_grfz]):
                ax.plot(pgc, rgrfz_mean[:, ilabel], label=label, color=color, 
                    linewidth=lw, clip_on=False, solid_capstyle='round')
                ax.set_ylabel('medio-lateral ground reaction $[BW]$')
                ax.set_ylim(grfz_lim)
                ax.set_yticks(get_ticks_from_lims(grfz_lim, grfz_step))

        for ax in [ax_f0_grfx, ax_grfx, ax_grfy, ax_grfz]:
            ax.legend(handles, self.legend_labels, loc='best', 
                frameon=True, prop={'size': 6})

        fig0.tight_layout()
        fig0.savefig(target[0], dpi=600)
        plt.close()

        fig2.tight_layout()
        fig2.savefig(target[1], dpi=600)
        plt.close()

        fig3.tight_layout()
        fig3.savefig(target[2], dpi=600)
        plt.close()

        fig4.tight_layout()
        fig4.savefig(target[3], dpi=600)
        plt.close()


class TaskPlotInstantaneousGroundReactions(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, times, rise, fall):
        super(TaskPlotInstantaneousGroundReactions, self).__init__(study)
        self.name = f'plot_instantaneous_ground_reactions_rise{rise}_fall{fall}'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'ground_reactions_instantaneous', f'rise{rise}_fall{fall}')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)

        self.subjects = subjects
        self.times = times
        self.rise = rise
        self.fall = fall
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
                              'ankle + eversion', 
                              'ankle', 
                              'ankle + inversion', 
                              'inversion']
        self.labels = list()
        self.times_list = list()

        self.labels.append('unperturbed')
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
                for torque, subtalar in zip(self.torques, self.subtalars):
                    label = (f'torque{torque}_time{time}'
                             f'_rise{self.rise}_fall{self.fall}{subtalar}')
                    deps.append(
                        os.path.join(self.study.config['results_path'], 
                            'perturbed', f'perturbed_{label}', subject, 
                            f'perturbed_{label}_grfs.sto'))

                    if not isubj:
                        self.labels.append(label)
                        self.times_list.append(time)

        self.add_action(deps, 
                        [os.path.join(self.analysis_path, 'instant_grfs')],
                        self.plot_instantaneous_ground_reactions)

    def plot_instantaneous_ground_reactions(self, file_dep, target):

        # Initialize figures
        # ------------------
        axes = list()
        fig = plt.figure(figsize=(6, 9))
        for idirec, direc in enumerate(['AP', 'SI', 'ML']):
            ax = fig.add_subplot(3, 1, idirec + 1)
            ax.axhline(y=0, color='black', linestyle='-',
                    linewidth=0.1, alpha=1.0, zorder=-1)
            ax.spines['left'].set_position(('outward', 30))
            ax.set_xticks(np.arange(len(self.times)))
            ax.set_xlim(0, len(self.times)-1)
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
                ax.set_xlabel('peak perturbation time\n(% gait cycle)')

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

            for ilabel, label in enumerate(self.labels):
                table = osim.TimeSeriesTable(file_dep[ilabel + isubj*numLabels])
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
            zipped = zip(self.torques, self.subtalars, self.colors, self.shifts)
            for isubt, (torque, subtalar, color, shift) in enumerate(zipped):

                label = (f'torque{torque}_time{time}'
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
        update_lims(grf_x_diff_mean-grf_x_diff_std, grf_x_step, grf_x_lim, mirror=True)
        update_lims(grf_x_diff_mean+grf_x_diff_std, grf_x_step, grf_x_lim, mirror=True)
        update_lims(grf_y_diff_mean-grf_y_diff_std, grf_y_step, grf_y_lim)
        update_lims(grf_y_diff_mean+grf_y_diff_std, grf_y_step, grf_y_lim)
        update_lims(grf_z_diff_mean-grf_z_diff_std, grf_z_step, grf_z_lim)
        update_lims(grf_z_diff_mean+grf_z_diff_std, grf_z_step, grf_z_lim)
        handles = list()
        for itime, time in enumerate(self.times):
            zipped = zip(self.torques, self.subtalars, self.colors, self.shifts, self.hatches)
            for isubt, (torque, subtalar, color, shift, hatch) in enumerate(zipped):

                # Set the x-position for these bar chart entries.
                x = itime + shift
                lw = 0.2

                plot_errorbar(axes[0], x, grf_x_diff_mean[itime, isubt], grf_x_diff_std[itime, isubt])
                h = axes[0].bar(x, grf_x_diff_mean[itime, isubt], self.width, color=color, clip_on=False,
                    hatch=hatch, edgecolor=self.edgecolor, lw=lw)
                handles.append(h)
                axes[0].set_ylabel(r'$\Delta$ fore-aft ground reaction $[BW]$')

                plot_errorbar(axes[1], x, grf_y_diff_mean[itime, isubt], grf_y_diff_std[itime, isubt])
                axes[1].bar(x, grf_y_diff_mean[itime, isubt], self.width, color=color, clip_on=False,
                    hatch=hatch, edgecolor=self.edgecolor, lw=lw)
                axes[1].set_ylabel(r'$\Delta$ vertical ground reaction $[BW]$')

                plot_errorbar(axes[2], x, grf_z_diff_mean[itime, isubt], grf_z_diff_std[itime, isubt])
                axes[2].bar(x, grf_z_diff_mean[itime, isubt], self.width, color=color, clip_on=False,
                    hatch=hatch, edgecolor=self.edgecolor, lw=lw)
                axes[2].set_ylabel(r'$\Delta$ medio-lateral ground reaction $[BW]$')

        axes[0].set_ylim(grf_x_lim)
        axes[0].set_yticks(get_ticks_from_lims(grf_x_lim, grf_x_step))
        axes[0].legend(handles, self.legend_labels, loc='upper left', 
            title='perturbation', frameon=True)
        axes[1].set_ylim(grf_y_lim)
        axes[1].set_yticks(get_ticks_from_lims(grf_y_lim, grf_y_step))
        axes[2].set_ylim(grf_z_lim)
        axes[2].set_yticks(get_ticks_from_lims(grf_z_lim, grf_z_step))

        fig.tight_layout()
        fig.savefig(target[0], dpi=600)
        plt.close()


class TaskPlotGroundReactionsVersusEffectiveForces(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, times, torque, rise, fall, subtalar, lumbar_stiffness=1.0):
        super(TaskPlotGroundReactionsVersusEffectiveForces, self).__init__(study)
        self.lumbar = ''
        if not lumbar_stiffness == 1.0:
            self.lumbar = f'_lumbar{lumbar_stiffness}'
        self.subtalar = ''
        if not subtalar == 0.0:
            self.subtalar = f'_subtalar{subtalar}'

        self.name = (f'plot_ground_reactions_versus_effective_forces_torque'
                     f'{torque}_rise{rise}_fall{fall}{self.subtalar}{self.lumbar}')
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'ground_reactions_versus_effective_forces_instantaneous', 
            f'torque{torque}_rise{rise}_fall{fall}{self.subtalar}{self.lumbar}')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)

        self.subjects = subjects
        self.times = times
        self.torque = torque
        self.rise = rise
        self.fall = fall
        self.width = 0.25

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

            # Unperturbed center-of-mass
            deps.append(os.path.join(self.study.config['results_path'], 
                'unperturbed', subject, 'center_of_mass_unperturbed.sto'))

            for time in self.times:
                label = (f'torque{self.torque}_time{time}'
                         f'_rise{self.rise}_fall{self.fall}'
                         f'{self.subtalar}{self.lumbar}')

                # Perturbed grfs
                deps.append(
                    os.path.join(self.study.config['results_path'], 
                        'perturbed', f'perturbed_{label}', subject, 
                        f'perturbed_{label}_grfs.sto'))

                # Perturbed center-of-mass
                deps.append(
                    os.path.join(self.study.config['results_path'], 
                        'perturbed', f'perturbed_{label}', subject, 
                        f'center_of_mass_perturbed_{label}.sto'))

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
            fig = plt.figure(figsize=(4, 2.5))
            ax = fig.add_subplot(1, 1, 1)
            figs.append(fig)
            axes.append(ax)

        # Aggregate data
        # --------------
        numTimes = len(self.times)
        import collections
        grf_r_dict = collections.defaultdict(dict)
        grf_l_dict = collections.defaultdict(dict)
        com_dict = collections.defaultdict(dict)
        grf_time_dict = dict()
        com_time_dict = dict()
        for isubj, subject in enumerate(self.subjects):
            model = osim.Model(self.models[isubj])
            state = model.initSystem()
            mass = model.getTotalMass(state)
            BW = abs(model.getGravity()[1]) * mass

            index = isubj*2*(numTimes+1)

            # Unperturbed GRFs
            # ----------------
            unpGRFTable = osim.TimeSeriesTable(file_dep[index])
            unpGRFTimeVec = unpGRFTable.getIndependentColumn()
            unpGRFTable_r_np = np.zeros((len(unpGRFTimeVec), 3))
            unpGRFTable_l_np = np.zeros((len(unpGRFTimeVec), 3))

            rgrfx = unpGRFTable.getDependentColumn('ground_force_r_vx').to_numpy() / BW
            rgrfy = unpGRFTable.getDependentColumn('ground_force_r_vy').to_numpy() / BW
            rgrfz = unpGRFTable.getDependentColumn('ground_force_r_vz').to_numpy() / BW
            lgrfx = unpGRFTable.getDependentColumn('ground_force_l_vx').to_numpy() / BW
            lgrfy = unpGRFTable.getDependentColumn('ground_force_l_vy').to_numpy() / BW
            lgrfz = unpGRFTable.getDependentColumn('ground_force_l_vz').to_numpy() / BW

            grf_time_dict[subject] = unpGRFTimeVec
            unpGRFTable_r_np[:, 0] = rgrfx
            unpGRFTable_r_np[:, 1] = rgrfy
            unpGRFTable_r_np[:, 2] = rgrfz
            unpGRFTable_l_np[:, 0] = lgrfx
            unpGRFTable_l_np[:, 1] = lgrfy
            unpGRFTable_l_np[:, 2] = lgrfz

            # Unperturbed center-of-mass trajectories
            # ---------------------------------------
            unpCOMTable = osim.TimeSeriesTable(file_dep[index + 1])
            unpCOMTimeVec = unpCOMTable.getIndependentColumn()
            unpCOMTable_np = np.zeros((len(unpCOMTimeVec), 3))
            unpCOMTable_np[:, 0] = unpCOMTable.getDependentColumn(
                '/|com_acceleration_x').to_numpy()
            unpCOMTable_np[:, 1] = unpCOMTable.getDependentColumn(
                '/|com_acceleration_y').to_numpy()
            unpCOMTable_np[:, 2] = unpCOMTable.getDependentColumn(
                '/|com_acceleration_z').to_numpy()

            unpCOMTable_np[:, 0] *= mass 
            unpCOMTable_np[:, 1] *= mass 
            unpCOMTable_np[:, 2] *= mass
            unpCOMTable_np[:, 0] /= BW
            unpCOMTable_np[:, 1] /= BW
            unpCOMTable_np[:, 2] /= BW 

            com_time_dict[subject] = unpCOMTimeVec

            for itime, time in enumerate(self.times):

                label = (f'torque{self.torque}_time{time}'
                         f'_rise{self.rise}_fall{self.fall}'
                         f'{self.subtalar}{self.lumbar}')

                # Perturbed GRFs
                # ----------------
                perturbGRFTable = osim.TimeSeriesTable(file_dep[index + 2 + 2*itime])
                perturbGRFTimeVec = perturbGRFTable.getIndependentColumn()
                perturbGRFTable_r_np = np.zeros((len(perturbGRFTimeVec), 3))
                perturbGRFTable_l_np = np.zeros((len(perturbGRFTimeVec), 3))

                rgrfx = perturbGRFTable.getDependentColumn('ground_force_r_vx').to_numpy() / BW
                rgrfy = perturbGRFTable.getDependentColumn('ground_force_r_vy').to_numpy() / BW
                rgrfz = perturbGRFTable.getDependentColumn('ground_force_r_vz').to_numpy() / BW
                lgrfx = perturbGRFTable.getDependentColumn('ground_force_l_vx').to_numpy() / BW
                lgrfy = perturbGRFTable.getDependentColumn('ground_force_l_vy').to_numpy() / BW
                lgrfz = perturbGRFTable.getDependentColumn('ground_force_l_vz').to_numpy() / BW

                perturbGRFTable_r_np[:, 0] = rgrfx
                perturbGRFTable_r_np[:, 1] = rgrfy
                perturbGRFTable_r_np[:, 2] = rgrfz
                perturbGRFTable_l_np[:, 0] = lgrfx
                perturbGRFTable_l_np[:, 1] = lgrfy
                perturbGRFTable_l_np[:, 2] = lgrfz

                grf_r_dict[subject][label] = perturbGRFTable_r_np - unpGRFTable_r_np[0:len(perturbGRFTimeVec), :]
                grf_l_dict[subject][label] = perturbGRFTable_l_np - unpGRFTable_l_np[0:len(perturbGRFTimeVec), :]

                # Perturbed center-of-mass trajectories
                # -------------------------------------
                perturbCOMTable = osim.TimeSeriesTable(file_dep[index + 2 + 2*itime + 1])
                perturbCOMTimeVec = perturbCOMTable.getIndependentColumn()
                perturbCOMTable_np = np.zeros((len(perturbCOMTimeVec), 3))
                perturbCOMTable_np[:, 0] = perturbCOMTable.getDependentColumn(
                    '/|com_acceleration_x').to_numpy()
                perturbCOMTable_np[:, 1] = perturbCOMTable.getDependentColumn(
                    '/|com_acceleration_y').to_numpy()
                perturbCOMTable_np[:, 2] = perturbCOMTable.getDependentColumn(
                    '/|com_acceleration_z').to_numpy()

                perturbCOMTable_np[:, 0] *= mass 
                perturbCOMTable_np[:, 1] *= mass 
                perturbCOMTable_np[:, 2] *= mass
                perturbCOMTable_np[:, 0] /= BW
                perturbCOMTable_np[:, 1] /= BW
                perturbCOMTable_np[:, 2] /= BW 

                com_dict[subject][label] = perturbCOMTable_np - unpCOMTable_np[0:len(perturbCOMTimeVec), :]

        # Plotting
        # --------
        grfr_x_diff = np.zeros(len(self.subjects))
        grfr_y_diff = np.zeros(len(self.subjects))
        grfr_z_diff = np.zeros(len(self.subjects))
        grfr_x_diff_mean = np.zeros(len(self.times))
        grfr_y_diff_mean = np.zeros(len(self.times))
        grfr_z_diff_mean = np.zeros(len(self.times))
        grfr_x_diff_std = np.zeros(len(self.times))
        grfr_y_diff_std = np.zeros(len(self.times))
        grfr_z_diff_std = np.zeros(len(self.times))
        grfl_x_diff = np.zeros(len(self.subjects))
        grfl_y_diff = np.zeros(len(self.subjects))
        grfl_z_diff = np.zeros(len(self.subjects))
        grfl_x_diff_mean = np.zeros(len(self.times))
        grfl_y_diff_mean = np.zeros(len(self.times))
        grfl_z_diff_mean = np.zeros(len(self.times))
        grfl_x_diff_std = np.zeros(len(self.times))
        grfl_y_diff_std = np.zeros(len(self.times))
        grfl_z_diff_std = np.zeros(len(self.times))
        com_x_diff = np.zeros(len(self.subjects))
        com_y_diff = np.zeros(len(self.subjects))
        com_z_diff = np.zeros(len(self.subjects))
        com_x_diff_mean = np.zeros(len(self.times))
        com_y_diff_mean = np.zeros(len(self.times))
        com_z_diff_mean = np.zeros(len(self.times))
        com_x_diff_std = np.zeros(len(self.times))
        com_y_diff_std = np.zeros(len(self.times))
        com_z_diff_std = np.zeros(len(self.times))
        for itime, time in enumerate(self.times): 
            label = (f'torque{self.torque}_time{time}'
                     f'_rise{self.rise}_fall{self.fall}'
                     f'{self.subtalar}{self.lumbar}')
            for isubj, subject in enumerate(self.subjects):
                grfr = grf_r_dict[subject][label]
                grfl = grf_l_dict[subject][label]

                # Compute the closet time index to the current peak 
                # perturbation time. 
                #
                # TODO: A given peak perturbation time (e.g 50% of the 
                # gait cycle) may not lie exactly on a time point of 
                # from the simulation. The interval between time points
                # is 5ms, meaning that the time index could be up to 2.5ms
                # away from the actual perturbation peak time. 
                timeVec = np.array(grf_time_dict[subject])
                duration = timeVec[-1] - timeVec[0]
                time_at_peak = timeVec[0] + (duration * (time / 100.0))
                index = np.argmin(np.abs(timeVec - time_at_peak))

                grfr_x_diff[isubj] = grfr[index, 0]
                grfr_y_diff[isubj] = grfr[index, 1]
                grfr_z_diff[isubj] = grfr[index, 2]
                grfl_x_diff[isubj] = grfl[index, 0]
                grfl_y_diff[isubj] = grfl[index, 1]
                grfl_z_diff[isubj] = grfl[index, 2]

                com = com_dict[subject][label]
                timeVec = np.array(com_time_dict[subject])
                duration = timeVec[-1] - timeVec[0]
                time_at_peak = timeVec[0] + (duration * (time / 100.0))
                index = np.argmin(np.abs(timeVec - time_at_peak))

                com_x_diff[isubj] = com[index, 0]
                com_y_diff[isubj] = com[index, 1]
                com_z_diff[isubj] = com[index, 2]

            grfr_x_diff_mean[itime] = np.mean(grfr_x_diff)
            grfr_y_diff_mean[itime] = np.mean(grfr_y_diff)
            grfr_z_diff_mean[itime] = np.mean(grfr_z_diff)

            grfr_x_diff_std[itime] = np.std(grfr_x_diff)
            grfr_y_diff_std[itime] = np.std(grfr_y_diff)
            grfr_z_diff_std[itime] = np.std(grfr_z_diff)

            grfl_x_diff_mean[itime] = np.mean(grfl_x_diff)
            grfl_y_diff_mean[itime] = np.mean(grfl_y_diff)
            grfl_z_diff_mean[itime] = np.mean(grfl_z_diff)

            grfl_x_diff_std[itime] = np.std(grfl_x_diff)
            grfl_y_diff_std[itime] = np.std(grfl_y_diff)
            grfl_z_diff_std[itime] = np.std(grfl_z_diff)

            com_x_diff_mean[itime] = np.mean(com_x_diff)
            com_y_diff_mean[itime] = np.mean(com_y_diff)
            com_z_diff_mean[itime] = np.mean(com_z_diff)

            com_x_diff_std[itime] = np.std(com_x_diff)
            com_y_diff_std[itime] = np.std(com_y_diff)
            com_z_diff_std[itime] = np.std(com_z_diff)


        x_step = 0.05
        y_step = 0.1
        z_step = 0.01
        x_lim = [0.0, 0.0]
        y_lim = [0.0, 0.0]
        z_lim = [0.0, 0.0]
        update_lims(grfr_x_diff_mean-grfr_x_diff_std, x_step, x_lim)
        update_lims(grfr_x_diff_mean+grfr_x_diff_std, x_step, x_lim)
        update_lims(grfr_y_diff_mean-grfr_y_diff_std, y_step, y_lim)
        update_lims(grfr_y_diff_mean+grfr_y_diff_std, y_step, y_lim)
        update_lims(grfr_z_diff_mean-grfr_z_diff_std, z_step, z_lim)
        update_lims(grfr_z_diff_mean+grfr_z_diff_std, z_step, z_lim)
        update_lims(grfl_x_diff_mean-grfl_x_diff_std, x_step, x_lim)
        update_lims(grfl_x_diff_mean+grfl_x_diff_std, x_step, x_lim)
        update_lims(grfl_y_diff_mean-grfl_y_diff_std, y_step, y_lim)
        update_lims(grfl_y_diff_mean+grfl_y_diff_std, y_step, y_lim)
        update_lims(grfl_z_diff_mean-grfl_z_diff_std, z_step, z_lim)
        update_lims(grfl_z_diff_mean+grfl_z_diff_std, z_step, z_lim)
        update_lims(com_x_diff_mean-com_x_diff_std, x_step, x_lim)
        update_lims(com_x_diff_mean+com_x_diff_std, x_step, x_lim)
        update_lims(com_y_diff_mean-com_y_diff_std, y_step, y_lim)
        update_lims(com_y_diff_mean+com_y_diff_std, y_step, y_lim)
        update_lims(com_z_diff_mean-com_z_diff_std, z_step, z_lim)
        update_lims(com_z_diff_mean+com_z_diff_std, z_step, z_lim)
        for itime, time in enumerate(self.times):

            grf_color = 'orange'
            igrf = itime - self.width
            plot_errorbar(axes[0], igrf, grfr_x_diff_mean[itime], grfr_x_diff_std[itime], grf_color)
            h_grfrx = axes[0].bar(igrf, grfr_x_diff_mean[itime], self.width, color=grf_color, clip_on=False)
            axes[0].set_ylabel(r'$\Delta$ fore-aft force $[BW]$')

            plot_errorbar(axes[1], igrf, grfr_y_diff_mean[itime], grfr_y_diff_std[itime], grf_color)
            h_grfry = axes[1].bar(igrf, grfr_y_diff_mean[itime], self.width, color=grf_color, clip_on=False)
            axes[1].set_ylabel(r'$\Delta$ vertical force $[BW]$')

            plot_errorbar(axes[2], igrf, grfr_z_diff_mean[itime], grfr_z_diff_std[itime], grf_color)
            h_grfrz = axes[2].bar(igrf, grfr_z_diff_mean[itime], self.width, color=grf_color, clip_on=False)
            axes[2].set_ylabel(r'$\Delta$ medio-lateral force $[BW]$')

            grf_color = 'blue'
            igrf = itime
            h_grflx = plot_errorbar(axes[0], igrf, grfl_x_diff_mean[itime], grfl_x_diff_std[itime], grf_color)
            axes[0].bar(igrf, grfl_x_diff_mean[itime], self.width, color=grf_color, clip_on=False)

            h_grfly = plot_errorbar(axes[1], igrf, grfl_y_diff_mean[itime], grfl_y_diff_std[itime], grf_color)
            axes[1].bar(igrf, grfl_y_diff_mean[itime], self.width, color=grf_color, clip_on=False)

            h_grflz = plot_errorbar(axes[2], igrf, grfl_z_diff_mean[itime], grfl_z_diff_std[itime], grf_color)
            axes[2].bar(igrf, grfl_z_diff_mean[itime], self.width, color=grf_color, clip_on=False)

            com_color = 'green'
            icom = itime + self.width
            h_comx = plot_errorbar(axes[0], icom, com_x_diff_mean[itime], com_x_diff_std[itime], com_color)
            axes[0].bar(icom, com_x_diff_mean[itime], self.width, color=com_color, clip_on=False)

            h_comy = plot_errorbar(axes[1], icom, com_y_diff_mean[itime], com_y_diff_std[itime], com_color)
            axes[1].bar(icom, com_y_diff_mean[itime], self.width, color=com_color, clip_on=False)

            h_comz = plot_errorbar(axes[2], icom, com_z_diff_mean[itime], com_z_diff_std[itime], com_color)
            axes[2].bar(icom, com_z_diff_mean[itime], self.width, color=com_color, clip_on=False)

        labels = ['right GRF', 'left GRF', 'COM accel']
        axes[0].set_ylim(x_lim)
        axes[0].set_yticks(get_ticks_from_lims(x_lim, x_step))
        # axes[0].legend([h_grfrx, h_grflx, h_comx], labels)
        axes[1].set_ylim(y_lim)
        axes[1].set_yticks(get_ticks_from_lims(y_lim, y_step))
        # axes[1].legend([h_grfry, h_grfly, h_comy], labels)
        axes[2].set_ylim(z_lim)
        axes[2].set_yticks(get_ticks_from_lims(z_lim, z_step))
        # axes[2].legend([h_grfrz, h_grflz, h_comz], labels)

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
    def __init__(self, study, subjects, time, torque, subtalar, rise, fall):
        super(TaskPlotGroundReactionBreakdown, self).__init__(study)
        self.subtalar = ''
        if not subtalar == 0:
            self.subtalar = f'_subtalar{subtalar}'

        self.name = (f'plot_ground_reaction_breakdown'
                     f'_time{time}_torque{torque}{self.subtalar}_rise{rise}_fall{fall}')
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'ground_reaction_breakdown', 
            f'time{time}_torque{torque}{self.subtalar}_rise{rise}_fall{fall}')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)
        self.subjects = subjects
        self.time = time
        self.torque = torque
        self.rise = rise
        self.fall = fall

        self.forces = ['contactHeel_r',
                       'contactLateralRearfoot_r',
                       'contactLateralMidfoot_r',
                       'contactMedialMidfoot_r',
                       'contactLateralToe_r',
                       'contactMedialToe_r']
        self.colors = ['darkred', 'darkorange', 'gold',
                             'darkgreen', 'darkblue', 'indigo']
        self.labels = ['heel', 'rearfoot', 'lat. midfoot',
                             'med. midfoot', 'lat. toe', 'med. toe']

        self.ref_index = 0
        deps = list()

        self.models = list()
        for isubj, subject in enumerate(subjects):
            # Model
            self.models.append(os.path.join(
                self.study.config['results_path'], 
                'unperturbed', subject, 'model_unperturbed.osim'))

            deps.append(
                os.path.join(self.study.config['results_path'], 
                    'unperturbed', subject, 'unperturbed.sto'))

            label = (f'torque{self.torque}_time{self.time}'
                     f'_rise{self.rise}_fall{self.fall}{self.subtalar}')
            deps.append(
                os.path.join(self.study.config['results_path'], 
                    'perturbed', f'perturbed_{label}', subject, 
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
        fig0 = plt.figure(figsize=(5, 3))
        ax_grfx = fig0.add_subplot(111)

        fig1 = plt.figure(figsize=(5, 3))
        ax_grfy = fig1.add_subplot(111)

        fig2 = plt.figure(figsize=(5, 3))
        ax_grfz = fig2.add_subplot(111)

        # Plot formatting
        # ---------------
        for ax in [ax_grfx, ax_grfy, ax_grfz]:
            ax.axvline(x=self.time, color='gray', linestyle='-',
                linewidth=0.25, alpha=1.0, zorder=0)
            util.publication_spines(ax)
            xlim = [self.time-self.rise, self.time+self.fall]
            ax.set_xlim(xlim)
            ax.set_xticks(get_ticks_from_lims(xlim, 5))
            ax.spines['left'].set_position(('outward', 10))

        for ax in [ax_grfx, ax_grfy, ax_grfz]:
            ax.set_xlabel('time (% gait cycle)')
            ax.spines['left'].set_position(('outward', 10))
            ax.spines['bottom'].set_position(('outward', 10))

        # Aggregate data
        # --------------
        import collections
        unp_dict = collections.defaultdict(dict)
        per_dict = collections.defaultdict(dict)
        N = 1001
        for force in ['total'] + self.forces:
            unp_dict[force]['x'] = np.zeros((N, len(self.subjects)))
            unp_dict[force]['y'] = np.zeros((N, len(self.subjects)))
            unp_dict[force]['z'] = np.zeros((N, len(self.subjects)))
            per_dict[force]['x'] = np.zeros((N, len(self.subjects)))
            per_dict[force]['y'] = np.zeros((N, len(self.subjects)))
            per_dict[force]['z'] = np.zeros((N, len(self.subjects)))

        for isubj, subject in enumerate(self.subjects):
            model = osim.Model(self.models[isubj])
            state = model.initSystem()
            mass = model.getTotalMass(state)
            BW = abs(model.getGravity()[1]) * mass


            unpTraj = osim.MocoTrajectory(file_dep[2*isubj])
            perTraj = osim.MocoTrajectory(file_dep[2*isubj + 1])

            unpTimeVec = unpTraj.getTimeMat()
            perTimeVec = unpTraj.getTimeMat()
            
            duration = unpTimeVec[-1] - unpTimeVec[0]
            time_at_rise = unpTimeVec[0] + (duration * ((self.time - self.rise) / 100.0))
            time_at_fall = unpTimeVec[0] + (duration * ((self.time + self.fall) / 100.0))

            irise = np.argmin(np.abs(unpTimeVec - time_at_rise))
            ifall = np.argmin(np.abs(unpTimeVec - time_at_fall))

            timeInterp = np.linspace(unpTimeVec[irise], unpTimeVec[ifall-1], N)


            for force in self.forces:
                sshsForce = osim.SmoothSphereHalfSpaceForce.safeDownCast(
                    model.getComponent(f'/forceset/{force}'))

                # Unperturbed
                grfx = np.zeros_like(unpTimeVec)
                grfy = np.zeros_like(unpTimeVec)
                grfz = np.zeros_like(unpTimeVec)
                statesTraj = unpTraj.exportToStatesTrajectory(model)
                for istate in np.arange(statesTraj.getSize()):
                    state = statesTraj.get(int(istate))
                    model.realizeVelocity(state)
                    forceValues = sshsForce.getRecordValues(state)

                    grfx[istate] = forceValues.get(0) / BW
                    grfy[istate] = forceValues.get(1) / BW
                    grfz[istate] = forceValues.get(2) / BW

                unp_dict[force]['x'][:, isubj] = np.interp(timeInterp, unpTimeVec[irise:ifall], grfx[irise:ifall])
                unp_dict[force]['y'][:, isubj] = np.interp(timeInterp, unpTimeVec[irise:ifall], grfy[irise:ifall])
                unp_dict[force]['z'][:, isubj] = np.interp(timeInterp, unpTimeVec[irise:ifall], grfz[irise:ifall])

                # Perturbed
                grfx = np.zeros_like(perTimeVec)
                grfy = np.zeros_like(perTimeVec)
                grfz = np.zeros_like(perTimeVec)
                statesTraj = perTraj.exportToStatesTrajectory(model)
                for istate in np.arange(statesTraj.getSize()):
                    state = statesTraj.get(int(istate))
                    model.realizeVelocity(state)
                    forceValues = sshsForce.getRecordValues(state)

                    grfx[istate] = forceValues.get(0) / BW
                    grfy[istate] = forceValues.get(1) / BW
                    grfz[istate] = forceValues.get(2) / BW

                per_dict[force]['x'][:, isubj] = np.interp(timeInterp, perTimeVec[irise:ifall], grfx[irise:ifall])
                per_dict[force]['y'][:, isubj] = np.interp(timeInterp, perTimeVec[irise:ifall], grfy[irise:ifall])
                per_dict[force]['z'][:, isubj] = np.interp(timeInterp, perTimeVec[irise:ifall], grfz[irise:ifall])

        for force in self.forces:
            for direc in ['x', 'y', 'z']:
                unp_dict['total'][direc] += unp_dict[force][direc]
                per_dict['total'][direc] += per_dict[force][direc]

        # Plotting
        # --------
        unp_rgrfx_mean = np.zeros((N, len(self.forces)+1))
        unp_rgrfy_mean = np.zeros((N, len(self.forces)+1))
        unp_rgrfz_mean = np.zeros((N, len(self.forces)+1))
        per_rgrfx_mean = np.zeros((N, len(self.forces)+1))
        per_rgrfy_mean = np.zeros((N, len(self.forces)+1))
        per_rgrfz_mean = np.zeros((N, len(self.forces)+1))
        for iforce, force in enumerate(['total'] + self.forces):
            unp_rgrfx_mean[:, iforce] = np.mean(unp_dict[force]['x'], axis=1)
            unp_rgrfy_mean[:, iforce] = np.mean(unp_dict[force]['y'], axis=1)
            unp_rgrfz_mean[:, iforce] = np.mean(unp_dict[force]['z'], axis=1)
            per_rgrfx_mean[:, iforce] = np.mean(per_dict[force]['x'], axis=1)
            per_rgrfy_mean[:, iforce] = np.mean(per_dict[force]['y'], axis=1)
            per_rgrfz_mean[:, iforce] = np.mean(per_dict[force]['z'], axis=1)

        grfx_step = 0.02
        grfy_step = 0.05
        grfz_step = 0.02
        grfx_lim = [np.mean(unp_rgrfx_mean), np.mean(unp_rgrfx_mean)]
        grfy_lim = [np.mean(unp_rgrfy_mean), np.mean(unp_rgrfy_mean)]
        grfz_lim = [np.mean(unp_rgrfz_mean), np.mean(unp_rgrfx_mean)]
        for iforce, force in enumerate(['total'] + self.forces): 
            update_lims(unp_rgrfx_mean[:, iforce], grfx_step, grfx_lim)
            update_lims(unp_rgrfy_mean[:, iforce], grfy_step, grfy_lim)
            update_lims(unp_rgrfz_mean[:, iforce], grfz_step, grfz_lim)
            update_lims(per_rgrfx_mean[:, iforce], grfx_step, grfx_lim)
            update_lims(per_rgrfy_mean[:, iforce], grfy_step, grfy_lim)
            update_lims(per_rgrfz_mean[:, iforce], grfz_step, grfz_lim)

        pgc = np.linspace(self.time - self.rise, self.time + self.fall, N)
        handles = list()
        zipped = zip(['total'] + self.forces, ['black'] + self.colors, ['total'] + self.labels)
        for iforce, (force, color, label) in enumerate(zipped):
            lw = 3 if 'total' in label else 1.5

            h, = ax_grfx.plot(pgc, unp_rgrfx_mean[:, iforce], label=label, color=color, 
                linewidth=lw, clip_on=False, solid_capstyle='round')
            handles.append(h)
            ax_grfx.plot(pgc, per_rgrfx_mean[:, iforce], label=label, color=color, 
                linewidth=lw, ls='--', clip_on=False, solid_capstyle='round')
            ax_grfx.set_ylabel(r'fore-aft ground reaction $[BW]$')
            ax_grfx.set_ylim(grfx_lim)
            ax_grfx.set_yticks(get_ticks_from_lims(grfx_lim, grfx_step))
                
            ax_grfy.plot(pgc, unp_rgrfy_mean[:, iforce], label=label, color=color, 
                    linewidth=lw, clip_on=False, solid_capstyle='round')
            ax_grfy.plot(pgc, per_rgrfy_mean[:, iforce], label=label, color=color, 
                    linewidth=lw, ls='--', clip_on=False, solid_capstyle='round')
            ax_grfy.set_ylabel('vertical ground reaction $[BW]$')
            ax_grfy.set_ylim(grfy_lim)
            ax_grfy.set_yticks(get_ticks_from_lims(grfy_lim, grfy_step))

            ax_grfz.plot(pgc, unp_rgrfz_mean[:, iforce], label=label, color=color, 
                linewidth=lw, clip_on=False, solid_capstyle='round')
            ax_grfz.plot(pgc, per_rgrfz_mean[:, iforce], label=label, color=color, 
                linewidth=lw, ls='--', clip_on=False, solid_capstyle='round')
            ax_grfz.set_ylabel('medio-lateral ground reaction $[BW]$')
            ax_grfz.set_ylim(grfz_lim)
            ax_grfz.set_yticks(get_ticks_from_lims(grfz_lim, grfz_step))

        for ax in [ax_grfx, ax_grfy, ax_grfz]:
            ax.legend(handles, ['total'] + self.labels, loc='best', 
                frameon=True, prop={'size': 6})

        fig0.tight_layout()
        fig0.savefig(target[0], dpi=600)
        plt.close()

        fig1.tight_layout()
        fig1.savefig(target[1], dpi=600)
        plt.close()

        fig2.tight_layout()
        fig2.savefig(target[2], dpi=600)
        plt.close()


class TaskPlotBodyAccelerations(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, time, rise, fall):
        super(TaskPlotBodyAccelerations, self).__init__(study)
        self.name = f'plot_body_accelerations_time{time}_rise{rise}_fall{fall}'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'body_accelerations',  f'time{time}_rise{rise}_fall{fall}')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)
        self.subjects = subjects
        self.time = time
        self.rise = rise
        self.fall = fall
        self.torques = study.plot_torques
        self.subtalars = study.plot_subtalars
        self.colors = study.plot_colors
        self.bodies = ['toes_r', 'talus_r', 'calcn_r', 'tibia_r', 'femur_r', 'pelvis', 'torso']
        self.body_names = ['toes', 'talus', 'calcaneus', 'shank', 'thigh', 'pelvis', 'torso']


        self.hatches = [None, None, None, None, None]
        self.edgecolor = 'black'
        self.width = 0.12
        N = len(self.subtalars)
        min_width = -self.width*((N-1)/2)
        max_width = -min_width
        self.shifts = np.linspace(min_width, max_width, N)
        self.legend_labels = ['eversion', 
                              'plantarflexion + eversion', 
                              'plantarflexion', 
                              'plantarflexion + inversion', 
                              'inversion']
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
            for torque, subtalar in zip(self.torques, self.subtalars):
                label = (f'perturbed_torque{torque}_time{self.time}'
                        f'_rise{self.rise}_fall{self.fall}{subtalar}')
                deps.append(
                    os.path.join(
                        self.study.config['results_path'], 
                        'perturbed', label, subject,
                        f'center_of_mass_{label}.sto')
                    )

                if not isubj:
                    self.labels.append(
                        (f'torque{torque}_time{self.time}'
                         f'_rise{self.rise}_fall{self.fall}{subtalar}'))
                    self.times_list.append(self.time)

        self.add_action(deps, 
            [os.path.join(self.analysis_path, 'body_accelerations.png')], 
            self.plot_body_accelerations)

    def plot_body_accelerations(self, file_dep, target):

        # Initialize figures
        # ------------------
        figs = list()
        axes = list()
        scale = 1.4
        fig = plt.figure(figsize=(4*scale, 3*scale))
        ax = fig.add_subplot(1, 1, 1)

        ax.axvline(x=0, color='gray', linestyle='-',
                linewidth=0.5, alpha=0.75, zorder=-1)
        ax.set_yticks(np.arange(len(self.bodies)))
        ax.set_ylim(0, len(self.bodies)-1)
        util.publication_spines(ax)

        ax.spines['left'].set_position(('outward', 10))
        ax.set_yticklabels([f'{body}' for body in self.body_names])
        # ax.set_ylabel(r'body')

        ax.spines['bottom'].set_position(('outward', 30))
        # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        # ax.tick_params(which='minor', axis='x', direction='in')

        axes.append(ax)
        figs.append(fig)

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
            unpTable_np = np.zeros((len(unpTimeVec), len(self.bodies)))

            for ibody, body in enumerate(self.bodies):
                unpTable_np[:, ibody] = unpTable.getDependentColumn(
                    f'/bodyset/{body}|linear_acceleration_z').to_numpy() 

            time_dict[subject] = unpTimeVec

            for ilabel, label in enumerate(self.labels):
                # Perturbed center-of-mass trajectories
                table = osim.TimeSeriesTable(file_dep[ilabel + isubj*numLabels])
                timeVec = table.getIndependentColumn()
                N = len(timeVec)
                table_np = np.zeros((N, len(self.bodies)))

                for ibody, body in enumerate(self.bodies):
                    table_np[:, ibody] = table.getDependentColumn(
                        f'/bodyset/{body}|linear_acceleration_z').to_numpy() 

                # Compute difference between perturbed and unperturbed
                # trajectories for this subject. We don't need to interpolate
                # here since the perturbed and unperturbed trajectories contain
                # the same time points (up until the end of the perturbation).
                com_dict[subject][label] = table_np - unpTable_np[0:N, :]

        # Plotting
        # --------
        body_acc_diff = np.zeros((len(self.bodies), len(self.subjects)))
        body_acc_diff_mean = np.zeros((len(self.bodies), len(self.subtalars)))
        body_acc_diff_std = np.zeros((len(self.bodies), len(self.subtalars)))
        zipped = zip(self.torques, self.subtalars, self.colors, self.shifts)
        for isubt, (torque, subtalar, color, shift) in enumerate(zipped):

            label = (f'torque{torque}_time{self.time}'
                     f'_rise{self.rise}_fall{self.fall}{subtalar}')
            for isubj, subject in enumerate(self.subjects):
                body_acc = com_dict[subject][label]

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
                time_at_peak = timeVec[0] + (duration * (self.time / 100.0))
                index = np.argmin(np.abs(timeVec - time_at_peak))

                for ibody, body in enumerate(self.bodies):
                    body_acc_diff[ibody, isubj] = body_acc[index, ibody]


            for ibody, body in enumerate(self.bodies):
                body_acc_diff_mean[ibody, isubt] = np.mean(body_acc_diff[ibody, :])
                body_acc_diff_std[ibody, isubt] = np.std(body_acc_diff[ibody, :])

        body_acc_step = 0.5
        body_acc_lim = [0.0, 0.0]
        update_lims(body_acc_diff_mean-body_acc_diff_std, body_acc_step, body_acc_lim, mirror=False)       
        update_lims(body_acc_diff_mean+body_acc_diff_std, body_acc_step, body_acc_lim, mirror=False)
        handles = list()
        for ibody, body in enumerate(self.bodies):
            zipped = zip(self.torques, self.subtalars, self.colors, self.shifts, self.hatches)
            for isubt, (torque, subtalar, color, shift, hatch) in enumerate(zipped):

                # Set the x-position for these bar chart entries.
                y = ibody + shift
                lw = 0.1

                # Medio-lateral body accelerations
                # --------------------------------
                plot_errorbarh(axes[0], y, body_acc_diff_mean[ibody, isubt], body_acc_diff_std[ibody, isubt])
                h = axes[0].barh(y, body_acc_diff_mean[ibody, isubt], self.width, color=color, clip_on=False, 
                    hatch=hatch, edgecolor=self.edgecolor, lw=lw)
                handles.append(h)
                

        axes[0].set_xlabel(r'$\Delta$' + ' medio-lateral acceleration $[m/s^2]$')
        axes[0].set_xlim(body_acc_lim)
        axes[0].set_xticks(get_ticks_from_lims(body_acc_lim, body_acc_step))
        axes[0].legend(handles, self.legend_labels, loc='upper right', 
            title='perturbation', frameon=True)
   
        fig.tight_layout()
        fig.savefig(target[0], dpi=600)
        plt.close()
