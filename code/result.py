import os
import numpy as np
import datetime

from PIL import Image

from abc import ABC, abstractmethod

import opensim as osim

import utilities
from utilities import simtk2numpy

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

class Result(ABC):
    def __init__(self):
        # Global plotting properties
        # --------------------------
        self.exp_color = 'gray'
        self.lw = 2

    @abstractmethod
    def generate_results(self):
        pass

    @abstractmethod
    def report_results(self):
        pass

    def create_valid_path(self, path):
        path = f'{path}'
        path = path.replace(' ', '')
        path = path.replace(':', '_')
        path = path.replace('.', '_')
        return path

    def get_solution_path(self, name):
        return os.path.join(f'{self.result_fpath}',
                            f'{name}.sto')

    def get_experiment_states_path(self, name):
        return os.path.join(f'{self.result_fpath}',
                            f'{name}_experiment_states.sto') 

    def get_solution_archive_path(self, name):
        now = datetime.datetime.now()
        now.strftime('%Y-%m-%dT%H:%M:%S')
        now = self.create_valid_path(now)
        return os.path.join(f'{self.result_fpath}', 'archive',
                            f'{name}_{now}.sto')

    def get_solution_path_grfs(self, name):
        return os.path.join(f'{self.result_fpath}',
                            f'{name}_grfs.sto')

    def get_solution_archive_path_grfs(self, name):
        now = datetime.datetime.now()
        now.strftime('%Y-%m-%dT%H:%M:%S')
        now = self.create_valid_path(now)
        return os.path.join(f'{self.result_fpath}', 'archive',
                            f'{name}_grfs_{now}.sto')

    def load_table(self, table_path):
        num_header_rows = 1
        with open(table_path) as f:
            for line in f:
                if not line.startswith('endheader'):
                    num_header_rows += 1
                else:
                    break
        return np.genfromtxt(table_path, names=True, delimiter='\t',
                             skip_header=num_header_rows)

    def create_model_processor_base(self, config):

        osim.Logger.setLevelString('error')

        # Load model
        # ----------
        model = osim.Model(self.model_fpath)
        state = model.initSystem()
        mass = model.getTotalMass(state)

        coordNames = ['lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
        for coordName in coordNames:
            actu = osim.ActivationCoordinateActuator()
            actu.set_coordinate(coordName)
            actu.setName(f'torque_{coordName}')
            actu.setOptimalForce(mass)
            actu.setMinControl(-1.0)
            actu.setMaxControl(1.0)
            model.addForce(actu)

        stiffnesses = [1.0, 1.5, 0.5] # N-m/rad*kg
        for coordName, stiffness in zip(coordNames, stiffnesses):
            sgf = osim.SpringGeneralizedForce(coordName)
            sgf.setName(f'passive_stiffness_{coordName}')
            sgf.setStiffness(config.lumbar_stiffness * stiffness * mass)
            sgf.setViscosity(2.0)
            model.addForce(sgf)

        if self.reserve_strength:
            coordNames = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz',
                           'pelvis_list', 'pelvis_tilt', 'pelvis_rotation',
                           'hip_adduction_r', 'hip_rotation_r', 'hip_flexion_r',
                           'hip_adduction_l', 'hip_rotation_l', 'hip_flexion_l',
                           'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r',
                           'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l',
                           'mtp_angle_r', 'mtp_angle_l']
            for coordName in coordNames:
                actu = osim.ActivationCoordinateActuator()
                actu.set_coordinate(coordName)
                actu.setName(f'reserve_{coordName}')
                actu.setOptimalForce(self.reserve_strength)
                actu.setMinControl(-1.0)
                actu.setMaxControl(1.0)
                model.addForce(actu)

        model.finalizeConnections()

        modelProcessor = osim.ModelProcessor(model)
        jointsToWeld = list()
        modelProcessor.append(osim.ModOpReplaceJointsWithWelds(jointsToWeld))
        modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
        modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
        modelProcessor.append(osim.ModOpFiberDampingDGF(0.01))

        # Enable tendon compliance for the ankle plantarflexors.
        # ------------------------------------------------------
        model = modelProcessor.process()
        model.initSystem()
        muscles = model.updMuscles()
        for imusc in np.arange(muscles.getSize()):
            muscle = osim.DeGrooteFregly2016Muscle.safeDownCast(muscles.get(int(imusc)))
            muscName = muscle.getName()

            if ('gas' in muscName) or ('soleus' in muscName):
                muscle.set_ignore_tendon_compliance(False)
                muscle.set_tendon_strain_at_one_norm_force(0.10)
                muscle.set_passive_fiber_strain_at_one_norm_force(2.0)

        # Update contact model properties
        # -------------------------------
        # forces = model.updForceSet()
        # for iforce in np.arange(forces.getSize()):
        #     if 'contact' in forces.get(int(iforce)).getName():
        #         sshs = osim.SmoothSphereHalfSpaceForce.safeDownCast(forces.get(int(iforce)))
        #         sshs.set_transition_velocity(0.1)
        #         sshs.set_static_friction(0.95)

        model.finalizeConnections()
        modelProcessor = osim.ModelProcessor(model)
        if self.implicit_tendon_dynamics:
            modelProcessor.append(
                osim.ModOpUseImplicitTendonComplianceDynamicsDGF())

        osim.Logger.setLevelString('info')

        return modelProcessor

    def calc_negative_muscle_forces_base(self, model, solution):
        model.initSystem()
        outputs = osim.analyze(model, solution.exportToStatesTable(),
            solution.exportToControlsTable(), ['.*\|tendon_force'])
        def simtkmin(simtkvec):
            lowest = np.inf
            for i in range(simtkvec.size()):
                if simtkvec[i] < lowest:
                    lowest = simtkvec[i]
            return lowest

        negforces = list()
        muscle_names = list()
        for imusc in range(model.getMuscles().getSize()):
            musc = model.updMuscles().get(imusc)
            max_iso = musc.get_max_isometric_force()
            force = outputs.getDependentColumn(
                musc.getAbsolutePathString() + "|tendon_force")
            neg = simtkmin(force) / max_iso
            if neg < 0:
                negforces.append(neg)
                muscle_names.append(musc.getName())
                print(f'  {musc.getName()}: {neg} F_iso')
        if len(negforces) == 0:
            print('No negative forces')
        else:
            imin = np.argmin(negforces)
            print(f'Largest negative force: {muscle_names[imin]} '
                  f'with {negforces[imin]} F_iso')
        return min([0] + negforces)

    def calc_muscle_mechanics(self, config, model, solution):
        outputList = list()
        for output in ['normalized_fiber_length', 'normalized_fiber_velocity', 
                       'active_fiber_force', 'passive_fiber_force', 
                       'tendon_force', 'activation', 'cos_pennation_angle', 
                       'active_force_length_multiplier', 
                       'force_velocity_multiplier', 'passive_force_multiplier',
                       'tendon_length', 'tendon_strain']:
            for imusc in range(model.getMuscles().getSize()):
                musc = model.updMuscles().get(imusc)
                outputList.append(f'.*{musc.getName()}.*\|{output}')

        for output in ['length', 'lengthening_speed']:
            outputList.append(f'.*\|{output}')

        outputs = osim.analyze(model, solution.exportToStatesTable(),
            solution.exportToControlsTable(), outputList)

        return outputs


    def calc_negative_muscle_forces(self, config, model, solution):
        print(f'Negative force report for {config.name}:')
        model.initSystem()
        return self.calc_negative_muscle_forces_base(model, solution)


    def savefig(self, fig, filename):
        fig.savefig(filename + ".png", format="png", dpi=600)

        # Load this image into PIL
        png2 = Image.open(filename + ".png")

        # Save as TIFF
        png2.save(filename + ".tiff", compression='tiff_lzw')


    def plot_ground_reactions(self, models):

        # Bodyweight
        # ----------
        state = models[0].initSystem()
        mass = models[0].getTotalMass(state)
        gravity = models[0].getGravity()
        BW = mass * abs(gravity[1])

        # Initialize figures
        # ------------------
        fig_grf = plt.figure(figsize=(6, 4))
        gs = gridspec.GridSpec(2, 3)
        ax_rgrf_x = fig_grf.add_subplot(gs[0, 0])
        ax_rgrf_y = fig_grf.add_subplot(gs[0, 1])
        ax_rgrf_z = fig_grf.add_subplot(gs[0, 2])
        ax_lgrf_x = fig_grf.add_subplot(gs[1, 0])
        ax_lgrf_y = fig_grf.add_subplot(gs[1, 1])
        ax_lgrf_z = fig_grf.add_subplot(gs[1, 2])

        ax_list = list()
        ax_list.append(ax_rgrf_x)
        ax_list.append(ax_rgrf_z)
        ax_list.append(ax_rgrf_y)        
        ax_list.append(ax_lgrf_x)
        ax_list.append(ax_lgrf_z)
        ax_list.append(ax_lgrf_y)

        for ax in ax_list:
            utilities.publication_spines(ax)
            ax.tick_params(axis='x', labelsize=6)
            ax.axhline(y=0, lw=0.75, color='black', ls='--', 
                zorder=0, alpha=0.5)

        # Experimental ground reactions
        # -----------------------------
        grf_table = self.load_table(self.grf_fpath)
        grf_start = np.argmin(abs(grf_table['time']-self.initial_time))
        grf_end = np.argmin(abs(grf_table['time']-self.final_time))

        time_grfs = grf_table['time'][grf_start:grf_end]
        ax_rgrf_x.plot(time_grfs, 
            grf_table['ground_force_r_vx'][grf_start:grf_end] / BW, 
            color=self.exp_color, lw=self.lw+1.0)
        ax_rgrf_y.plot(time_grfs, 
            grf_table['ground_force_r_vy'][grf_start:grf_end] / BW, 
            color=self.exp_color, lw=self.lw+1.0)
        ax_rgrf_z.plot(time_grfs,
            grf_table['ground_force_r_vz'][grf_start:grf_end] / BW,
            color=self.exp_color, lw=self.lw+1.0)
        ax_lgrf_x.plot(time_grfs, 
            grf_table['ground_force_l_vx'][grf_start:grf_end] / BW, 
            color=self.exp_color, lw=self.lw+1.0)
        ax_lgrf_y.plot(time_grfs, 
            grf_table['ground_force_l_vy'][grf_start:grf_end] / BW, 
            color=self.exp_color, lw=self.lw+1.0)
        ax_lgrf_z.plot(time_grfs,
            grf_table['ground_force_l_vz'][grf_start:grf_end] / BW,
            color=self.exp_color, lw=self.lw+1.0)


        # Simulation ground reactions
        # ---------------------------
        for i, config in enumerate(self.configs):
            path = self.get_solution_path(config.name)
            traj = osim.MocoTrajectory(path)

            grf_table = self.load_table(
                self.get_solution_path_grfs(config.name))
            time = traj.getTimeMat()

            ax_rgrf_x.plot(time, grf_table['ground_force_r_vx'] / BW, 
                color=config.color, lw=self.lw)
            ax_rgrf_x.set_title('horizontal GRF')
            ax_rgrf_x.set_ylabel('force (BW)')

            ax_rgrf_y.plot(time, grf_table['ground_force_r_vy'] / BW, 
                color=config.color, lw=self.lw)
            ax_rgrf_y.set_title('vertical GRF')

            ax_rgrf_z.plot(time, grf_table['ground_force_r_vz'] / BW, 
                color=config.color, lw=self.lw)
            ax_rgrf_z.set_title('transverse GRF')

            ax_lgrf_x.plot(time, grf_table['ground_force_l_vx'] / BW, 
                color=config.color, lw=self.lw)
            ax_lgrf_x.set_ylabel('force (BW)')
            ax_lgrf_x.set_xlabel('time (s)')

            ax_lgrf_y.plot(time, grf_table['ground_force_l_vy'] / BW, 
                color=config.color, lw=self.lw)
            ax_lgrf_y.set_xlabel('time (s)')

            ax_lgrf_z.plot(time, grf_table['ground_force_l_vz'] / BW, 
                color=config.color, lw=self.lw)
            ax_lgrf_z.set_xlabel('time (s)')


        # Save figure
        # -----------
        fig_grf.tight_layout()
        fig_grf.savefig(os.path.join(self.result_fpath, 
            'ground_reaction_forces.png'), dpi=600)


    def plot_joint_kinematics(self, models):

        # Initialize figures
        # ------------------
        fig_kin = plt.figure(figsize=(6, 9))
        gs = gridspec.GridSpec(5, 2)
        ax_add_l = fig_kin.add_subplot(gs[0, 0])
        ax_rot_l = fig_kin.add_subplot(gs[1, 0])
        ax_hip_l = fig_kin.add_subplot(gs[2, 0])
        ax_knee_l = fig_kin.add_subplot(gs[3, 0])
        ax_ankle_l = fig_kin.add_subplot(gs[4, 0])
        ax_add_r = fig_kin.add_subplot(gs[0, 1])
        ax_rot_r = fig_kin.add_subplot(gs[1, 1])
        ax_hip_r = fig_kin.add_subplot(gs[2, 1])
        ax_knee_r = fig_kin.add_subplot(gs[3, 1])
        ax_ankle_r = fig_kin.add_subplot(gs[4, 1])

        ax_list = list()
        ax_list.append(ax_add_l)
        ax_list.append(ax_hip_l)
        ax_list.append(ax_rot_l)
        ax_list.append(ax_knee_l)
        ax_list.append(ax_ankle_l)
        ax_list.append(ax_add_r)
        ax_list.append(ax_hip_r)
        ax_list.append(ax_rot_r)
        ax_list.append(ax_knee_r)
        ax_list.append(ax_ankle_r)

        for ax in ax_list:
            utilities.publication_spines(ax)
            ax.tick_params(axis='x', labelsize=6)
            ax.axhline(y=0, lw=0.75, color='black', ls='--', 
                zorder=0, alpha=0.5)

        # Experimental kinematics
        # -----------------------
        kinematics = osim.TimeSeriesTable(
            self.get_experiment_states_path(self.configs[0].name))

        time = np.array(kinematics.getIndependentColumn())
        rad2deg = 180 / np.pi
        hip_add_bounds = [-15, 15]
        hip_rot_bounds = [-15, 50]
        hip_flex_bounds = [-20, 60]
        knee_bounds = [0, 100]
        ankle_bounds = [-30, 30]

        ax_add_l.plot(time, 
            rad2deg * simtk2numpy(
                kinematics.getDependentColumn('/jointset/hip_l/hip_adduction_l/value')),
                color=self.exp_color, lw=self.lw + 1.0)
        ax_add_l.set_ylabel('hip adduc.')
        ax_add_l.set_xticklabels([])
        ax_add_l.set_ylim(hip_add_bounds)

        ax_rot_l.plot(time, 
            rad2deg * simtk2numpy(
                kinematics.getDependentColumn('/jointset/hip_l/hip_rotation_l/value')),
                color=self.exp_color, lw=self.lw + 1.0)
        ax_rot_l.set_ylabel('hip rot.')
        ax_rot_l.set_xticklabels([])
        ax_rot_l.set_ylim(hip_rot_bounds)

        ax_hip_l.plot(time, 
            rad2deg * simtk2numpy(
                kinematics.getDependentColumn('/jointset/hip_l/hip_flexion_l/value')),
                color=self.exp_color, lw=self.lw + 1.0)
        ax_hip_l.set_ylabel('hip flex.')
        ax_hip_l.set_xticklabels([])
        ax_hip_l.set_ylim(hip_flex_bounds)

        ax_knee_l.plot(time, 
            rad2deg * simtk2numpy(
                kinematics.getDependentColumn('/jointset/walker_knee_l/knee_angle_l/value')),
                color=self.exp_color, lw=self.lw + 1.0)
        ax_knee_l.set_ylabel('knee flex.')
        ax_knee_l.set_xticklabels([])
        ax_knee_l.set_ylim(knee_bounds)

        ax_ankle_l.plot(time, 
            rad2deg * simtk2numpy(
                kinematics.getDependentColumn('/jointset/ankle_l/ankle_angle_l/value')),
                color=self.exp_color, lw=self.lw + 1.0)
        ax_ankle_l.set_ylabel('ankle dorsiflex.')
        ax_ankle_l.set_xlabel('time (s)')
        ax_ankle_l.set_ylim(ankle_bounds)

        ax_add_r.plot(time, 
            rad2deg * simtk2numpy(
                kinematics.getDependentColumn('/jointset/hip_r/hip_adduction_r/value')),
                color=self.exp_color, lw=self.lw + 1.0)
        ax_add_r.set_xticklabels([])
        ax_add_r.set_yticklabels([])
        ax_add_r.set_ylim(hip_add_bounds)

        ax_rot_r.plot(time, 
            rad2deg * simtk2numpy(
                kinematics.getDependentColumn('/jointset/hip_r/hip_rotation_r/value')),
                color=self.exp_color, lw=self.lw + 1.0)
        ax_rot_r.set_xticklabels([])
        ax_rot_r.set_yticklabels([])
        ax_rot_r.set_ylim(hip_rot_bounds)

        ax_hip_r.plot(time, 
            rad2deg * simtk2numpy(
                kinematics.getDependentColumn('/jointset/hip_r/hip_flexion_r/value')),
                color=self.exp_color, lw=self.lw + 1.0)
        ax_hip_r.set_xticklabels([])
        ax_hip_r.set_yticklabels([])
        ax_hip_r.set_ylim(hip_flex_bounds)

        ax_knee_r.plot(time, 
            rad2deg * simtk2numpy(
                kinematics.getDependentColumn('/jointset/walker_knee_r/knee_angle_r/value')),
                color=self.exp_color, lw=self.lw + 1.0)
        ax_knee_r.set_xticklabels([])
        ax_knee_r.set_yticklabels([])
        ax_knee_r.set_ylim(knee_bounds)

        ax_ankle_r.plot(time, 
            rad2deg * simtk2numpy(
                kinematics.getDependentColumn('/jointset/ankle_r/ankle_angle_r/value')),
                color=self.exp_color, lw=self.lw + 1.0)
        ax_ankle_r.set_ylim(ankle_bounds)
        ax_ankle_r.set_yticklabels([])

        # Simulation kinematics
        # ---------------------
        for i, config in enumerate(self.configs):
            color = config.color
            traj = osim.MocoTrajectory(self.get_solution_path(config.name))
            time = traj.getTimeMat()

            ax_add_l.plot(time, rad2deg * traj.getStateMat(
                '/jointset/hip_l/hip_adduction_l/value'), color=color, lw=self.lw)
            ax_rot_l.plot(time, rad2deg * traj.getStateMat(
                '/jointset/hip_l/hip_rotation_l/value'), color=color, lw=self.lw)
            ax_hip_l.plot(time, rad2deg * traj.getStateMat(
                '/jointset/hip_l/hip_flexion_l/value'), color=color, lw=self.lw)
            ax_knee_l.plot(time, rad2deg * traj.getStateMat(
                '/jointset/walker_knee_l/knee_angle_l/value'), color=color, lw=self.lw)
            ax_ankle_l.plot(time, rad2deg * traj.getStateMat(
                '/jointset/ankle_l/ankle_angle_l/value'), color=color, lw=self.lw)

            ax_add_r.plot(time, rad2deg * traj.getStateMat(
                '/jointset/hip_r/hip_adduction_r/value'), color=color, lw=self.lw)
            ax_rot_r.plot(time, rad2deg * traj.getStateMat(
                '/jointset/hip_r/hip_rotation_r/value'), color=color, lw=self.lw)
            ax_hip_r.plot(time, rad2deg * traj.getStateMat(
                '/jointset/hip_r/hip_flexion_r/value'), color=color, lw=self.lw)
            ax_knee_r.plot(time, rad2deg * traj.getStateMat(
                '/jointset/walker_knee_r/knee_angle_r/value'), color=color, lw=self.lw)
            ax_ankle_r.plot(time, rad2deg * traj.getStateMat(
                '/jointset/ankle_r/ankle_angle_r/value'), color=color, lw=self.lw)

        # Save figure
        # -----------
        fig_kin.tight_layout()
        fig_kin.savefig(os.path.join(self.result_fpath, 
                'joint_kinematics.png'), dpi=600)


    def plot_muscle_activations(self, models):

        # Initialize figures
        # ------------------
        fig_act = plt.figure(figsize=(5, 12))
        gs = gridspec.GridSpec(10, 2)
        muscles = [
            (fig_act.add_subplot(gs[0, 0]), 'addlong_l', 'add. long.'),
            (fig_act.add_subplot(gs[1, 0]), 'glmax2_l', 'glut. max.'),
            (fig_act.add_subplot(gs[2, 0]), 'glmed1_l', 'glut. med.'),
            (fig_act.add_subplot(gs[3, 0]), 'psoas_l', 'psoas'),
            (fig_act.add_subplot(gs[4, 0]), 'semimem_l', 'semimem.'),
            (fig_act.add_subplot(gs[5, 0]), 'recfem_l', 'rec. fem.'),
            (fig_act.add_subplot(gs[6, 0]), 'vasint_l', 'vas. int.'),
            (fig_act.add_subplot(gs[7, 0]), 'gasmed_l', 'med. gas.'),
            (fig_act.add_subplot(gs[8, 0]), 'soleus_l', 'soleus'),
            (fig_act.add_subplot(gs[9, 0]), 'tibant_l', 'tib. ant.'),
            (fig_act.add_subplot(gs[0, 1]), 'addlong_r', 'add. long.'),
            (fig_act.add_subplot(gs[1, 1]), 'glmax2_r', 'glut. max.'),
            (fig_act.add_subplot(gs[2, 1]), 'glmed1_r', 'glut. med.'),
            (fig_act.add_subplot(gs[3, 1]), 'psoas_r', 'psoas'),
            (fig_act.add_subplot(gs[4, 1]), 'semimem_r', 'semimem.'),
            (fig_act.add_subplot(gs[5, 1]), 'recfem_r', 'rec. fem.',),
            (fig_act.add_subplot(gs[6, 1]), 'vasint_r', 'vas. int.'),
            (fig_act.add_subplot(gs[7, 1]), 'gasmed_r', 'med. gas.'),
            (fig_act.add_subplot(gs[8, 1]), 'soleus_r', 'soleus'),
            (fig_act.add_subplot(gs[9, 1]), 'tibant_r', 'tib. ant.'),
        ]

        ax_list = list()
        for muscle in muscles:
            ax_list.append(muscle[0])

        for ax in ax_list:
            utilities.publication_spines(ax)
            ax.tick_params(axis='x', labelsize=6)
            ax.axhline(y=0, lw=0.75, color='black', ls='--', 
                zorder=0, alpha=0.5)

        # Load EMG data
        # -------------
        emg_table = utilities.storage2numpy(self.emg_fpath)
        emg_map = {
           'tibant_r' : 'tibant_r',
           'soleus_r' : 'soleus_r',
           'gasmed_r' : 'gasmed_r',
           'semimem_r': 'semimem_r',
           'vasint_r' : 'vaslat_r',
           'recfem_r' : 'recfem_r',
           'glmax2_r' : 'glmax2_r',
           'glmed1_r' : 'glmed1_r',
        }
   
        # Plot activations with EMG data
        # ------------------------------
        for i, config in enumerate(self.configs):
            color = config.color
            traj = osim.MocoTrajectory(
                self.get_solution_path(config.name))
            time = traj.getTimeMat()
            
            istart = np.argmin(np.abs(emg_table['time'] - time[0]))
            iend = np.argmin(np.abs(emg_table['time'] - time[-1]))
            emg_time = np.array(emg_table['time'][istart:iend])
            for im, muscle in enumerate(muscles):
                ax = muscle[0]
                activation = traj.getStateMat(f'/forceset/{muscle[1]}/activation')

                # Scale EMG to match peak activations
                if muscle[1] in emg_map and i == 0:
                    max_act = max(activation)
                    emg = emg_table[emg_map[muscle[1]]][istart:iend]
                    max_emg = max(emg)
                    emg_scale = max_act / max_emg
                    ax.fill_between(emg_time, emg_scale*emg, np.zeros_like(emg), 
                        color=self.exp_color, zorder=0)

                ax.plot(time, activation, color=color, lw=self.lw)
                ax.set_ylim(0, 1)
                ax.set_yticks([0, 1])
                utilities.publication_spines(ax)
                if im < 10:
                    ax.set_ylabel(muscle[2])
                if im == 9 or im == 19:
                    ax.set_xlabel('time')
                else: 
                    ax.set_xticklabels([])

        fig_act.tight_layout()
        fig_act.savefig(os.path.join(self.result_fpath, 
                'muscle_activations.png'), dpi=600)
        plt.close('all')


    def plot_muscle_mechanics(self, muscle_mechanics, output):

        # Initialize figures
        # ------------------
        fig = plt.figure(figsize=(5, 12))
        gs = gridspec.GridSpec(10, 2)
        muscles = [
            (fig.add_subplot(gs[0, 0]), 'addlong_l', 'add. long.'),
            (fig.add_subplot(gs[1, 0]), 'glmax2_l', 'glut. max.'),
            (fig.add_subplot(gs[2, 0]), 'glmed1_l', 'glut. med.'),
            (fig.add_subplot(gs[3, 0]), 'psoas_l', 'psoas'),
            (fig.add_subplot(gs[4, 0]), 'semimem_l', 'semimem.'),
            (fig.add_subplot(gs[5, 0]), 'recfem_l', 'rec. fem.'),
            (fig.add_subplot(gs[6, 0]), 'vasint_l', 'vas. int.'),
            (fig.add_subplot(gs[7, 0]), 'gasmed_l', 'med. gas.'),
            (fig.add_subplot(gs[8, 0]), 'soleus_l', 'soleus'),
            (fig.add_subplot(gs[9, 0]), 'tibant_l', 'tib. ant.'),
            (fig.add_subplot(gs[0, 1]), 'addlong_r', 'add. long.'),
            (fig.add_subplot(gs[1, 1]), 'glmax2_r', 'glut. max.'),
            (fig.add_subplot(gs[2, 1]), 'glmed1_r', 'glut. med.'),
            (fig.add_subplot(gs[3, 1]), 'psoas_r', 'psoas'),
            (fig.add_subplot(gs[4, 1]), 'semimem_r', 'semimem.'),
            (fig.add_subplot(gs[5, 1]), 'recfem_r', 'rec. fem.',),
            (fig.add_subplot(gs[6, 1]), 'vasint_r', 'vas. int.'),
            (fig.add_subplot(gs[7, 1]), 'gasmed_r', 'med. gas.'),
            (fig.add_subplot(gs[8, 1]), 'soleus_r', 'soleus'),
            (fig.add_subplot(gs[9, 1]), 'tibant_r', 'tib. ant.'),
        ]

        ax_list = list()
        for muscle in muscles:
            ax_list.append(muscle[0])

        for ax in ax_list:
            utilities.publication_spines(ax)
            ax.tick_params(axis='x', labelsize=6)
            if ('normalized_fiber_length' in output or
                    'force_velocity_multiplier' in output):
                ax.axhline(y=0.75, lw=0.75, color='black', ls='--', 
                    zorder=0, alpha=0.5)
                ax.axhline(y=1.0, lw=0.75, color='black', ls='--', 
                    zorder=0, alpha=0.5)
                ax.axhline(y=1.25, lw=0.75, color='black', ls='--', 
                    zorder=0, alpha=0.5)

            elif 'normalized_fiber_velocity' in output:
                ax.axhline(y=-0.25, lw=0.75, color='black', ls='--', 
                    zorder=0, alpha=0.5)
                ax.axhline(y=0, lw=0.75, color='black', ls='--', 
                    zorder=0, alpha=0.5)
                ax.axhline(y=0.25, lw=0.75, color='black', ls='--', 
                    zorder=0, alpha=0.5)

            elif 'passive_force_multiplier' in output:
                ax.axhline(y=0, lw=0.75, color='black', ls='--', 
                    zorder=0, alpha=0.5)
                ax.axhline(y=0.1, lw=0.75, color='black', ls='--', 
                    zorder=0, alpha=0.5)
                ax.axhline(y=0.2, lw=0.75, color='black', ls='--', 
                    zorder=0, alpha=0.5)

            elif 'active_force_length_multiplier' in output:
                ax.axhline(y=0.5, lw=0.75, color='black', ls='--', 
                    zorder=0, alpha=0.5)
                ax.axhline(y=0.75, lw=0.75, color='black', ls='--', 
                    zorder=0, alpha=0.5)
                ax.axhline(y=1.0, lw=0.75, color='black', ls='--', 
                    zorder=0, alpha=0.5)

   
        # Plot muscle output
        # ------------------
        for i, config in enumerate(self.configs):
            color = config.color
            table = muscle_mechanics[config.name]
            time = np.array(table.getIndependentColumn())
            for im, muscle in enumerate(muscles):
                ax = muscle[0]
                outputVec = simtk2numpy(table.getDependentColumn(f'/forceset/{muscle[1]}|{output}'))
                ax.plot(time, outputVec, color=color, lw=self.lw)
                utilities.publication_spines(ax)
                if im < 10:
                    ax.set_ylabel(muscle[2])
                if im == 9 or im == 19:
                    ax.set_xlabel('time')
                else: 
                    ax.set_xticklabels([])

        fig.tight_layout()
        fig.savefig(os.path.join(self.result_fpath, 
                f'{output}.png'), dpi=600)
        plt.close('all')


    def plot_center_of_mass(self, models):

        # Initialize figures
        # ------------------
        fig = plt.figure(figsize=(12, 12))
        pos = fig.add_subplot(4, 3, 1)
        vel = fig.add_subplot(4, 3, 2)
        acc = fig.add_subplot(4, 3, 3)
        pos_tx = fig.add_subplot(4, 3, 4)
        vel_tx = fig.add_subplot(4, 3, 5)
        acc_tx = fig.add_subplot(4, 3, 6)
        pos_ty = fig.add_subplot(4, 3, 7)
        vel_ty = fig.add_subplot(4, 3, 8)
        acc_ty = fig.add_subplot(4, 3, 9)
        pos_tz = fig.add_subplot(4, 3, 10)
        vel_tz = fig.add_subplot(4, 3, 11)
        acc_tz = fig.add_subplot(4, 3, 12)

        def plot_com(com, name, color):
            com = com.flatten(['_x', '_y', '_z'])
            time = np.array(com.getIndependentColumn())

            com_pos_x = simtk2numpy(com.getDependentColumn('/|com_position_x'))
            com_pos_y = simtk2numpy(com.getDependentColumn('/|com_position_y'))
            com_pos_z = simtk2numpy(com.getDependentColumn('/|com_position_z'))
            com_vel_x = simtk2numpy(com.getDependentColumn('/|com_velocity_x'))
            com_vel_y = simtk2numpy(com.getDependentColumn('/|com_velocity_y'))
            com_vel_z = simtk2numpy(com.getDependentColumn('/|com_velocity_z'))
            com_acc_x = simtk2numpy(com.getDependentColumn('/|com_acceleration_x'))
            com_acc_y = simtk2numpy(com.getDependentColumn('/|com_acceleration_y'))
            com_acc_z = simtk2numpy(com.getDependentColumn('/|com_acceleration_z'))

            lw = 3
            s = 100

            # Position
            # --------
            pos.plot(com_pos_z, com_pos_x, color=color, lw=lw)
            pos.set_xlabel('z-position', fontsize=10)
            pos.set_ylabel('x-position', fontsize=10)
            pos.set_title('position (m)', fontsize=10)
            pos.scatter(com_pos_z[0], com_pos_x[0], s=s, color=color, 
                marker='o')
            pos.scatter(com_pos_z[-1], com_pos_x[-1], s=s, 
                color=color, marker='X')
            utilities.publication_spines(pos)

            pos_tx.plot(time, com_pos_x, color=color, lw=lw)
            pos_tx.set_xticklabels([])
            pos_tx.set_ylabel('x-position', fontsize=10)

            pos_ty.plot(time, com_pos_y, color=color, lw=lw)
            pos_ty.set_xticklabels([])
            pos_ty.set_ylabel('y-position', fontsize=10)

            pos_tz.plot(time, com_pos_z, color=color, lw=lw)
            pos_tz.set_xlabel('time (s)')
            pos_tz.set_ylabel('z-position', fontsize=10)

            # Velocity
            # --------
            vel.plot(com_vel_z, com_vel_x, color=color, lw=lw)
            vel.set_xlabel('z-velocity', fontsize=10)
            vel.set_ylabel('x-velocity', fontsize=10)
            vel.set_title('velocity (m/s)', fontsize=10)
            vel.scatter(com_vel_z[0], com_vel_x[0], s=s, color=color, 
                marker='o')
            vel.scatter(com_vel_z[-1], com_vel_x[-1], s=s, color=color, 
                marker='X')
            utilities.publication_spines(vel)

            vel_tx.plot(time, com_vel_x, color=color, lw=lw)
            vel_tx.set_xticklabels([])
            vel_tx.set_ylabel('x-velocity', fontsize=10)

            vel_ty.plot(time, com_vel_y, color=color, lw=lw)
            vel_ty.set_xticklabels([])
            vel_ty.set_ylabel('y-velocity', fontsize=10)

            vel_tz.plot(time, com_vel_z, color=color, lw=lw)
            vel_tz.set_xlabel('time (s)')
            vel_tz.set_ylabel('z-velocity', fontsize=10)

            # Acceleration
            # ------------
            acc.plot(com_acc_z, com_acc_x, color=color, lw=lw)
            acc.set_xlabel('z-acceleration', fontsize=10)
            acc.set_ylabel('x-acceleration', fontsize=10)
            acc.set_title('acceleration (m/s^2)', fontsize=10)
            acc.scatter(com_acc_z[0], com_acc_x[0], s=s, color=color, 
                marker='o')
            acc.scatter(com_acc_z[-1], com_acc_x[-1], s=s, color=color, 
                marker='X')
            utilities.publication_spines(acc)

            acc_tx.plot(time, com_acc_x, color=color, lw=lw)
            acc_tx.set_xticklabels([])
            acc_tx.set_ylabel('x-acceleration', fontsize=10)

            acc_ty.plot(time, com_acc_y, color=color, lw=lw)
            acc_ty.set_xticklabels([])
            acc_ty.set_ylabel('y-acceleration', fontsize=10)

            acc_tz.plot(time, com_acc_z, color=color, lw=lw)
            acc_tz.set_ylabel('z-acceleration', fontsize=10)
            acc_tz.set_xlabel('time (s)')

            axes = [pos_tx, pos_ty, pos_tz, 
                    vel_tx, vel_ty, vel_tz, 
                    acc_tx, acc_ty, acc_tz]
            for ax in axes:
                utilities.publication_spines(ax)

            fpath = os.path.join(self.result_fpath,
                f'center_of_mass_{name}.sto')
            osim.STOFileAdapter.write(com, fpath)


        experimentStates = osim.TimeSeriesTable(
            self.get_experiment_states_path(self.configs[0].name))
        expTime = experimentStates.getIndependentColumn()
        expStatesTraj = osim.StatesTrajectory().createFromStatesTable(
            models[0], experimentStates, True)
        bodyset = models[0].getBodySet()
        numBodies = bodyset.getSize()
        com = osim.TimeSeriesTableVec3()
        for i in np.arange(expStatesTraj.getSize()):
            row = osim.RowVectorVec3(3 + 3*numBodies, osim.Vec3(0.0))
            state = expStatesTraj.get(int(i))
            com_pos = models[0].calcMassCenterPosition(state)
            com_vel = models[0].calcMassCenterVelocity(state)
            com_acc = models[0].calcMassCenterAcceleration(state)
            row[0] = com_pos
            row[1] = com_vel
            row[2] = com_acc
            for ibody in np.arange(numBodies):
                body = bodyset.get(int(ibody))
                body_pos = body.getPositionInGround(state)
                body_vel = body.getLinearVelocityInGround(state)
                body_acc = body.getLinearAccelerationInGround(state)

                row[int(3 + 3*ibody)] = body_pos
                row[int(3 + 3*ibody + 1)] = body_vel
                row[int(3 + 3*ibody + 2)] = body_acc

            com.appendRow(expTime[int(i)], row)

        colLabels = osim.StdVectorString()
        colLabels.append('/|com_position')
        colLabels.append('/|com_velocity')
        colLabels.append('/|com_acceleration')
        for ibody in np.arange(numBodies):
            body = bodyset.get(int(ibody))
            name = body.getAbsolutePathString()
            colLabels.append(f'{name}|position')
            colLabels.append(f'{name}|linear_velocity')
            colLabels.append(f'{name}|linear_acceleration')
        com.setColumnLabels(colLabels)

        plot_com(com, 'experiment', 'darkgray')

        for i, config in enumerate(self.configs):
            solution = osim.MocoTrajectory(
                self.get_solution_path(config.name))
            com = osim.analyzeVec3(models[i], solution.exportToStatesTable(),
                                          solution.exportToControlsTable(),
                                        ['.*com.*', 
                                         '/bodyset/.*position.*',
                                         '/bodyset/.*linear_velocity.*',
                                         '/bodyset/.*linear_acceleration.*'])
            plot_com(com, config.name, config.color)

        fig.tight_layout()
        fig.savefig(os.path.join(self.result_fpath, 
            'center_of_mass.png'), dpi=600)
        plt.close()


    def create_pdf_report(self, models, configs):
        trajectory_filepath = self.get_solution_path(configs[-1].name)
        ref_files = list()
        ref_files.append(self.get_experiment_states_path(configs[-1].name))
        colorlist = list()
        colorlist.append('darkgray')
        report_suffix = configs[-1].name
        for config in configs[:-1]:
            ref_files.append(
                self.get_solution_path(config.name))
            colorlist.append(config.color)
            report_suffix += '_' + config.name
        colorlist.append(configs[-1].color)
        report_output = os.path.join(self.result_fpath, f'{report_suffix}_report.pdf')
        report = osim.report.Report(model=models[0],
                                    trajectory_filepath=trajectory_filepath,
                                    ref_files=ref_files, bilateral=False,
                                    colorlist=colorlist,
                                    output=report_output)
        report.generate()