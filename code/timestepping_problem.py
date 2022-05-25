import os
import numpy as np
import matplotlib.pyplot as plt

import opensim as osim
from result import Result
import copy

import utilities as util

forceNamesRightFoot = ['forceset/contactHeel_r',
                       'forceset/contactLateralRearfoot_r',
                       'forceset/contactLateralMidfoot_r',
                       'forceset/contactLateralToe_r',
                       'forceset/contactMedialToe_r',
                       'forceset/contactMedialMidfoot_r']
forceNamesLeftFoot = ['forceset/contactHeel_l',
                      'forceset/contactLateralRearfoot_l',
                      'forceset/contactLateralMidfoot_l',
                      'forceset/contactLateralToe_l',
                      'forceset/contactMedialToe_l',
                      'forceset/contactMedialMidfoot_l']

class TimeSteppingConfig:
    def __init__(self, name, legend_entry, color, weights, 
                 unperturbed_fpath=None, 
                 ankle_torque_perturbation=False,
                 ankle_torque_parameters=None,
                 subtalar_torque_perturbation=False,
                 subtalar_peak_torque=0,
                 lumbar_stiffness=1.0):

        # Base arguments
        self.name = name
        self.legend_entry = legend_entry
        self.color = color
        self.unperturbed_fpath = unperturbed_fpath

        # Ankle torque perturbation
        self.ankle_torque_perturbation = ankle_torque_perturbation
        self.ankle_torque_parameters = ankle_torque_parameters
        self.ankle_torque_perturbation_start = 0.0
        self.ankle_torque_perturbation_end = 0.0
        self.subtalar_torque_perturbation = subtalar_torque_perturbation
        self.subtalar_peak_torque = subtalar_peak_torque

        # Lumbar stiffness scaling
        self.lumbar_stiffness = lumbar_stiffness

class TimeSteppingProblem(Result):
    def __init__(self, root_dir, result_fpath, model_fpath, coordinates_fpath, 
            coordinates_std_fpath, extloads_fpath, grf_fpath, emg_fpath, 
            initial_time, final_time, cycles, right_strikes, left_strikes, 
            mesh_interval, walking_speed, configs):
        super(TimeSteppingProblem, self).__init__()
        self.root_dir = root_dir
        self.result_fpath = result_fpath
        self.model_fpath = model_fpath
        self.extloads_fpath = extloads_fpath
        self.grf_fpath = grf_fpath
        self.coordinates_fpath = coordinates_fpath
        self.coordinates_std_fpath = coordinates_std_fpath
        self.initial_time = initial_time
        self.final_time = final_time
        self.cycles = cycles
        self.right_strikes = right_strikes
        self.left_strikes = left_strikes
        self.mesh_interval = mesh_interval
        self.walking_speed = walking_speed
        self.emg_fpath = emg_fpath
        self.configs = configs

    def get_perturbation_torque_path(self):
        return os.path.join(self.result_fpath,
                    f'ankle_perturbation_curve.sto')

    def create_model_processor(self, config):

        osim.Logger.setLevelString('error')

        modelProcessor = self.create_model_processor_base(config)
        model = modelProcessor.process()
        state = model.initSystem()
        mass = model.getTotalMass(state)

        # Ankle torque perturbation
        # -------------------------
        if config.ankle_torque_perturbation:       

            def get_torque_curve(initial_time, duration, parameters):
                from scipy.interpolate import CubicSpline
                peak_torque = parameters[0]
                peak_time = parameters[1] * duration + initial_time
                rise_time = parameters[2] * duration
                fall_time = parameters[3] * duration

                x1 = [peak_time - rise_time, peak_time]
                y1 = [0.0, peak_torque]

                spline1 = CubicSpline(x1, y1, bc_type='clamped')
                x1s = np.linspace(x1[0], x1[1], 100)
                y1s = spline1(x1s)

                x2 = [peak_time, peak_time + fall_time]
                y2 = [peak_torque, 0.0]

                spline2 = CubicSpline(x2, y2, bc_type='clamped')
                x2s = np.linspace(x2[0], x2[1], 100)
                y2s = spline2(x2s)

                xis = np.linspace(initial_time, x1[0], 100)
                xfs = np.linspace(x2[1], initial_time + duration, 100)
                yis = np.zeros(100)
                yfs = np.zeros(100)

                xs = np.concatenate((xis[:-1], x1s[:-1], x2s[:-1], xfs[:-1]))
                ys = np.concatenate((yis[:-1], y1s[:-1], y2s[:-1], yfs[:-1]))

                return xs, ys

            time = osim.StdVectorDouble()
            anklePerturbTorque = osim.StdVectorDouble()
            subtalarPerturbTorque = osim.StdVectorDouble()

            duration = 0.0
 
            parameters = config.ankle_torque_parameters
            duration = self.right_strikes[1] - self.right_strikes[0]
            initial_time = self.right_strikes[0]
            peak_time = parameters[1] * duration + initial_time
            rise_time = parameters[2] * duration
            fall_time = parameters[3] * duration
            pgc10 = 0.1 * duration
            config.ankle_torque_perturbation_start = peak_time - rise_time
            config.ankle_torque_perturbation_end = peak_time + fall_time

            xr, yr = get_torque_curve(initial_time, duration, parameters)
            for x, y in zip(xr, yr):
                time.push_back(x)
                anklePerturbTorque.push_back(y)

            subtalar_parameters = copy.deepcopy(config.ankle_torque_parameters)
            subtalar_parameters[0] = config.subtalar_peak_torque
            xr, yr = get_torque_curve(initial_time, duration, subtalar_parameters)
            for x, y in zip(xr, yr):
                subtalarPerturbTorque.push_back(y)

            anklePerturbTable = osim.TimeSeriesTable(time)
            perturbTorqueAnkle = osim.Vector(time.size(), 0.0)
            perturbTorqueSubtalar = osim.Vector(time.size(), 0.0)
            for i in np.arange(time.size()):
                # Flip the sign on the ankle curve to match model convention
                perturbTorqueAnkle.set(int(i), -anklePerturbTorque[int(i)])
                perturbTorqueSubtalar.set(int(i), subtalarPerturbTorque[int(i)])

            anklePerturbTable.appendColumn(
                '/forceset/perturbation_ankle_angle_r', perturbTorqueAnkle)
            if config.subtalar_torque_perturbation:
                anklePerturbTable.appendColumn(
                    '/forceset/perturbation_subtalar_angle_r', perturbTorqueSubtalar)

            sto = osim.STOFileAdapter()
            sto.write(anklePerturbTable, self.get_perturbation_torque_path())

            actu = osim.CoordinateActuator()
            actu.setName('perturbation_ankle_angle_r')
            actu.set_coordinate('ankle_angle_r')
            actu.setOptimalForce(mass)
            actu.setMinControl(-1.0)
            actu.setMaxControl(1.0)
            model.addForce(actu)

            if config.subtalar_torque_perturbation:
                actu = osim.CoordinateActuator()
                actu.setName('perturbation_subtalar_angle_r')
                actu.set_coordinate('subtalar_angle_r')
                actu.setOptimalForce(mass)
                actu.setMinControl(-1.0)
                actu.setMaxControl(1.0)
                model.addForce(actu)

            model.finalizeConnections()

        modelProcessor = osim.ModelProcessor(model)
        osim.Logger.setLevelString('info')

        return modelProcessor

    def run_timestepping_problem(self, config):

        # Create the model for this tracking config
        # -----------------------------------------
        modelProcessor = self.create_model_processor(config)        
        model = modelProcessor.process()
        model.initSystem()

        guessTable = osim.TimeSeriesTable(config.unperturbed_fpath)
        initial_index = guessTable.getNearestRowIndexForTime(
            config.ankle_torque_perturbation_start)
        final_index = guessTable.getNearestRowIndexForTime(
            config.ankle_torque_perturbation_end)-1

        trajectory = osim.MocoTrajectory(config.unperturbed_fpath)
        trajectory.trim(int(initial_index), int(final_index))

        perturbTable = osim.TimeSeriesTable(self.get_perturbation_torque_path())
        trajectory.insertControlsTrajectory(perturbTable)

        osim.prescribeControlsToModel(trajectory, model, 
            'GCVSpline')

        # Add states reporter to the model.
        # ---------------------------------
        statesRep = osim.StatesTrajectoryReporter()
        statesRep.setName('states_reporter')
        statesRep.set_report_time_interval(5e-3)
        model.addComponent(statesRep)

        # Simulate!
        # ---------
        time = trajectory.getTime()
        model.initSystem()
        manager = osim.Manager(model)

        # Set the initial state.
        # ----------------------
        statesTraj = trajectory.exportToStatesTrajectory(model)
        manager.setIntegratorAccuracy(1e-6)
        manager.setIntegratorMinimumStepSize(1e-6)
        manager.setIntegratorMaximumStepSize(1e-2)
        manager.initialize(statesTraj.get(0))
        state = manager.integrate(time[time.size() - 1])

        # Export results from states reporter to a table.
        # -----------------------------------------------
        statesTrajRep = osim.StatesTrajectoryReporter().safeDownCast(
            model.getComponent('/states_reporter')) 
        states = statesTrajRep.getStates().exportToTable(model)
        controls = trajectory.exportToControlsTable()

        # Convert to MocoTrajectory
        statesMatrix = osim.Matrix(states.getNumRows(), states.getNumColumns())
        for irow in np.arange(states.getNumRows()):
            rowVec = states.getRowAtIndex(int(irow))
            for icol in np.arange(states.getNumColumns()):
                statesMatrix.set(int(irow), int(icol), rowVec[int(icol)])

        statesTimes = states.getIndependentColumn()
        timeVec = osim.Vector(int(states.getNumRows()), 0.0)
        for itime in np.arange(len(statesTimes)):
            timeVec[int(itime)] = statesTimes[int(itime)]

        controlSplines = osim.GCVSplineSet(controls)
        controlsMatrix = osim.Matrix(states.getNumRows(), controls.getNumColumns())
        time = osim.Vector(1, 0.0)
        for icol in np.arange(controls.getNumColumns()):
            controlName = controls.getColumnLabel(int(icol))
            controlSpline = controlSplines.get(controlName)
            for irow in np.arange(states.getNumRows()):
                time[0] = timeVec[int(irow)]
                value = controlSpline.calcValue(time)
                controlsMatrix.set(int(irow), int(icol), value)

        solution = osim.MocoTrajectory(
            timeVec,
            states.getColumnLabels(), 
            controls.getColumnLabels(), 
            [], [],
            statesMatrix,
            controlsMatrix,
            osim.Matrix(),
            osim.RowVector())

        solution.write(self.get_solution_path(f'{config.name}_half'))
        solutionTable = osim.TimeSeriesTable(
            self.get_solution_path(f'{config.name}_half'))
        solutionTime = solutionTable.getIndependentColumn()
        solutionLabels = solutionTable.getColumnLabels()

        guessTable = osim.TimeSeriesTable(config.unperturbed_fpath)
        guessTime = guessTable.getIndependentColumn() 

        perturbSplines = osim.GCVSplineSet(perturbTable)
        for label in perturbTable.getColumnLabels():
            perturbSpline = perturbSplines.get(label)
            perturbCol = osim.Vector(len(guessTime), 0.0)
            time = osim.Vector(1, 0.0)
            for itime in np.arange(len(guessTime)):
                time[0] = guessTime[itime]
                perturbCol[int(itime)] = perturbSpline.calcValue(time)

            guessTable.appendColumn(label, perturbCol)

        guessLabels = guessTable.getColumnLabels()
        for label in guessLabels:
            if not label in solutionLabels:
                guessTable.removeColumn(label)

        fullSolutionTable = osim.TimeSeriesTable()
        for irow in np.arange(initial_index):
            fullSolutionTable.appendRow(
                guessTime[int(irow)],
                guessTable.getRowAtIndex(int(irow)))

        for irow in np.arange(len(solutionTime)):
            solutionRow = solutionTable.getRowAtIndex(int(irow))
            solutionTime = solutionTable.getIndependentColumn()
            rowToAppend = osim.RowVector(
                int(guessTable.getNumColumns()), 0.0)

            for iguess in np.arange(guessTable.getNumColumns()):
                guessLabel = guessTable.getColumnLabel(int(iguess))
                for isol in np.arange(solutionTable.getNumColumns()):
                    solutionLabel = solutionTable.getColumnLabel(int(isol))
                    if guessLabel == solutionLabel:
                        rowToAppend[int(iguess)] = solutionRow[int(isol)]

            fullSolutionTable.appendRow(
                solutionTime[int(irow)], rowToAppend)

        fullSolutionTable.setColumnLabels(guessTable.getColumnLabels())
        keys = np.array(solutionTable.getTableMetaDataKeys())
        for key in keys:
            if 'header' in key: 
                continue
            fullSolutionTable.addTableMetaDataString(key, 
                solutionTable.getTableMetaDataAsString(key))    

        osim.STOFileAdapter.write(fullSolutionTable, 
            self.get_solution_path(config.name))

        # Compute ground reaction forces generated by the contact spheres.
        # ----------------------------------------------------------------
        solution = osim.MocoTrajectory(self.get_solution_path(config.name)) 
        externalLoads = osim.createExternalLoadsTableForGait(
                model, solution, forceNamesRightFoot, forceNamesLeftFoot)
        osim.STOFileAdapter.write(externalLoads,
                self.get_solution_path_grfs(config.name))

        # Archive solution
        # ----------------
        solution.write(self.get_solution_archive_path(config.name))
        osim.STOFileAdapter.write(externalLoads,
                self.get_solution_archive_path_grfs(config.name))

    def generate_results(self):
        for config in self.configs:
            self.run_timestepping_problem(config)
   
    def report_results(self):

        # Store a list of models
        # ----------------------
        models = list()
        for config in self.configs:
            modelProcessor = self.create_model_processor(config)
            model = modelProcessor.process()
            model.initSystem()
            model.printToXML(os.path.join(self.result_fpath, 
                f'model_{config.name}.osim'))
            models.append(model)

        # Plot ground reaction forces
        # ---------------------------
        self.plot_ground_reactions(models)

        # Plot joint kinematics
        # ---------------------
        self.plot_joint_kinematics(models)

    
        # Plot muscle activations
        # -----------------------
        self.plot_muscle_activations(models)


        # Calculate muscle outputs and plot joint moment breakdown
        # --------------------------------------------------------
        muscle_mechanics = dict()
        for i, config in enumerate(self.configs):
            # Load solution
            solution = osim.MocoTrajectory(self.get_solution_path(config.name))

            # Calculate muscle mechanics
            muscle_mechanics[config.name] = self.calc_muscle_mechanics(
                config, models[i], solution)
            self.calc_negative_muscle_forces(config, models[i], solution)
            fpath = os.path.join(self.result_fpath,
                    f'muscle_mechanics_{config.name}.sto')
            osim.STOFileAdapter.write(muscle_mechanics[config.name], fpath)

            # Generate joint moment breakdown
            coords = [
                '/jointset/hip_r/hip_flexion_r',
                '/jointset/walker_knee_r/knee_angle_r',
                '/jointset/ankle_r/ankle_angle_r'
            ]
            fig = util.plot_joint_moment_breakdown(models[i], solution, coords)
            fpath = os.path.join(self.result_fpath,
                                 f'joint_moment_breakdown_{config.name}.png')
            fig.savefig(fpath, dpi=600)
            plt.close('all')

        # Plot muscle mechanics
        # ---------------------
        self.plot_muscle_mechanics(muscle_mechanics, 'normalized_fiber_length')

        # Plot center of mass trajectories
        # --------------------------------
        self.plot_center_of_mass(models)

        # Generate PDF report.
        # --------------------
        self.create_pdf_report(models, self.configs)

        