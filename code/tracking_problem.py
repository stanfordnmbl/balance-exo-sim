import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import opensim as osim
from result import Result
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

class TrackingConfig:
    def __init__(self, name, legend_entry, color, weights, guess=None, 
                 effort_enabled=True, tracking_enabled=True,
                 periodic=False,
                 periodic_coordinates_to_include=None, 
                 periodic_actuators=True,
                 periodic_values=True, 
                 periodic_speeds=True,
                 lumbar_stiffness=1.0,
                 randomize_guess=False,
                 create_and_insert_guess=False):

        # Base arguments
        self.name = name
        self.legend_entry = legend_entry
        self.color = color
        self.guess = guess
        self.effort_enabled = effort_enabled
        self.tracking_enabled = tracking_enabled

        # Cost function weights
        self.control_weight = weights['control_weight']
        self.state_tracking_weight = weights['state_tracking_weight'] 
        self.grf_tracking_weight = weights['grf_tracking_weight'] 
        self.torso_orientation_weight = weights['torso_orientation_weight']        
        self.feet_orientation_weight = weights['feet_orientation_weight']        
        self.control_tracking_weight = weights['control_tracking_weight']
        self.aux_deriv_weight = weights['aux_deriv_weight'] 
        self.acceleration_weight = weights['acceleration_weight']

        # Periodicity arguments
        self.periodic = periodic
        self.periodic_coordinates_to_include = periodic_coordinates_to_include
        self.periodic_actuators = periodic_actuators
        self.periodic_values = periodic_values
        self.periodic_speeds = periodic_speeds

        # lumbar stiffness scaling
        self.lumbar_stiffness = lumbar_stiffness

        # Randomize guess
        self.randomize_guess = randomize_guess

        # Enable this flag if the provided guess doesn't
        # exactly match the current MocoProblem, but you 
        # would like to use information that does match
        # anyway.
        self.create_and_insert_guess = create_and_insert_guess

class TrackingProblem(Result):
    def __init__(self, root_dir, result_fpath, model_fpath, coordinates_fpath, 
            coordinates_std_fpath, extloads_fpath, grf_fpath, emg_fpath, 
            initial_time, final_time, cycles, right_strikes, left_strikes, 
            mesh_interval, convergence_tolerance, constraint_tolerance, 
            num_max_iterations, walking_speed, configs,
            reserve_strength=0,
            implicit_multibody_dynamics=False,
            implicit_tendon_dynamics=False):
        super(TrackingProblem, self).__init__()
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
        self.convergence_tolerance = convergence_tolerance
        self.constraint_tolerance = constraint_tolerance
        self.num_max_iterations = num_max_iterations
        self.walking_speed = walking_speed
        self.emg_fpath = emg_fpath
        self.configs = configs
        self.reserve_strength = reserve_strength
        self.implicit_multibody_dynamics = implicit_multibody_dynamics
        self.implicit_tendon_dynamics = implicit_tendon_dynamics

    def create_torso_tracking_reference(self, config):
        modelProcessor = self.create_model_processor(config)
        model = modelProcessor.process()
        model.initSystem()
        nrows = 100
        indVecArray = np.linspace(self.initial_time, self.final_time, num=nrows)
        indVec = osim.StdVectorDouble()
        for indVal in indVecArray:
            indVec.push_back(indVal)
        torsoRef = osim.TimeSeriesTable(indVec)
        stateVars = model.getStateVariableNames()
        for isv in np.arange(stateVars.getSize()):
            stateVar = stateVars.get(int(isv))
            value = 0
            if ('activation' in stateVar) or ('tendon_force' in stateVar):
                value = 0.02
            depVec = osim.Vector(nrows, value)
            torsoRef.appendColumn(stateVar, depVec)

        torsoRef.addTableMetaDataString('inDegrees', 'no')

        osim.STOFileAdapter.write(torsoRef, 
            os.path.join(self.root_dir, 
                'torso_zero_reference.sto'))

    def get_state_bounds(self, config):

        # State bounds
        # ------------
        pi = np.pi
        stateBounds = {
            '/jointset/ground_pelvis/pelvis_tx/value': ([-5, 5], []),
            '/jointset/ground_pelvis/pelvis_tx/speed': ([0.25, 2.5], [], []),
            '/jointset/ground_pelvis/pelvis_ty/value': ([0, 2], []),
            '/jointset/ground_pelvis/pelvis_ty/speed': ([-1, 1], []),
            '/jointset/ground_pelvis/pelvis_tz/value': ([-2, 2], []),
            '/jointset/ground_pelvis/pelvis_tz/speed': ([-2, 2], []),
            '/jointset/ground_pelvis/pelvis_tilt/value': ([-90*pi/180, 90*pi/180], []),
            '/jointset/ground_pelvis/pelvis_list/value': ([-90*pi/180, 90*pi/180], []),
            '/jointset/ground_pelvis/pelvis_rotation/value': ([-90*pi/180, 90*pi/180], []),
            '/jointset/hip_l/hip_flexion_l/value': ([-30*pi/180, 120*pi/180], []),
            '/jointset/hip_r/hip_flexion_r/value': ([-30*pi/180, 120*pi/180], []),
            '/jointset/hip_l/hip_adduction_l/value': ([-50*pi/180, 30*pi/180], []),
            '/jointset/hip_r/hip_adduction_r/value': ([-50*pi/180, 30*pi/180], []),
            '/jointset/hip_l/hip_rotation_l/value': ([-40*pi/180, 40*pi/180], []),
            '/jointset/hip_r/hip_rotation_r/value': ([-40*pi/180, 40*pi/180], []),
            '/jointset/walker_knee_l/knee_angle_l/value': ([0, 120*pi/180], []),
            '/jointset/walker_knee_r/knee_angle_r/value': ([0, 120*pi/180], []),
            '/jointset/patellofemoral_l/knee_angle_l_beta/value': ([-120*pi/180, 120*pi/180], []),
            '/jointset/patellofemoral_r/knee_angle_r_beta/value': ([-120*pi/180, 120*pi/180], []),
            '/jointset/ankle_l/ankle_angle_l/value': ([-40*pi/180, 30*pi/180], []),
            '/jointset/ankle_r/ankle_angle_r/value': ([-40*pi/180, 30*pi/180], []),
            '/jointset/subtalar_l/subtalar_angle_l/value': ([-20*pi/180, 20*pi/180], []),
            '/jointset/subtalar_r/subtalar_angle_r/value': ([-20*pi/180, 20*pi/180], []),
            '/jointset/mtp_l/mtp_angle_l/value': ([-30*pi/180, 30*pi/180], []),
            '/jointset/mtp_r/mtp_angle_r/value': ([-30*pi/180, 30*pi/180], []),
        }

        return stateBounds

    def create_model_processor(self, config):
        return self.create_model_processor_base(config)

    def run_tracking_problem(self, config):

        # Create the model for this tracking config
        # -----------------------------------------
        modelProcessor = self.create_model_processor(config)        
        model = modelProcessor.process()
        model.initSystem()

        # Count the number of Force objects in the model. We'll use this to 
        # normalize the control effort cost.
        numForces = 0
        for actu in model.getComponentsList():
            if (actu.getConcreteClassName().endswith('Muscle') or
                    actu.getConcreteClassName().endswith('Actuator')):
                numForces += 1

        # Create the coordinates tracking reference
        # -----------------------------------------
        coordinatesTable = osim.TimeSeriesTable(self.coordinates_fpath)
        time = coordinatesTable.getIndependentColumn()
        nrows = coordinatesTable.getNumRows()
        pelvis_tx = coordinatesTable.getDependentColumn('pelvis_tx')
        pelvis_tx_new = osim.Vector(nrows, 0.0)
        for i in np.arange(nrows):
            dx = (time[int(i)] - time[0]) * self.walking_speed
            pelvis_tx_new[int(i)] = pelvis_tx[int(i)] + dx

        pelvis_tx_upd = coordinatesTable.updDependentColumn('pelvis_tx')
        for i in np.arange(nrows):
            pelvis_tx_upd[int(i)] = pelvis_tx_new[int(i)]

        for side in ['_l', '_r']:
            coordinatesTable.removeColumn(f'wrist_dev{side}')
            coordinatesTable.removeColumn(f'wrist_flex{side}')
        tableProcessor = osim.TableProcessor(coordinatesTable)
        tableProcessor.append(osim.TabOpLowPassFilter(6))
        tableProcessor.append(osim.TabOpUseAbsoluteStateNames())

        # Load state bounds
        # -----------------
        stateBounds = self.get_state_bounds(config)
        
        # Create state tracking weights
        # -----------------------------
        model = modelProcessor.process()
        model.initSystem()
        coordinates = tableProcessor.process(model)
        paths = coordinates.getColumnLabels()
        coordinates_std = pd.read_csv(self.coordinates_std_fpath, index_col=0)
        stateWeights = osim.MocoWeightSet() 
        for valuePath in paths:
            speedPath = valuePath.replace('/value', '/speed')
            valueWeight = 1.0
            speedWeight = 1.0

            for name in coordinates_std.index:
                std = coordinates_std.loc[name][0] 
                denom = 10.0 * std

                if name in valuePath:
                    if 'lumbar' in name:
                        if config.torso_orientation_weight:
                            valueWeight = 0.0
                            speedWeight = 0.0
                        else:
                            valueWeight = 1.0 / denom
                            speedWeight = 0.01 / denom

                    elif ('beta' in name or 
                          'subtalar' in name or
                          'mtp' in name or 
                          'wrist' in name):  
                        valueWeight = 0.0
                        speedWeight = 0.0

                    elif 'ankle' in name:
                        valueWeight = 2.0 / denom
                        speedWeight = 0.02 / denom

                    elif 'pelvis' in name:
                        if 'pelvis_ty' in name:
                            valueWeight = 0.0
                            speedWeight = 0.01 / denom
                        else:
                            valueWeight = 1.0 / denom
                            speedWeight = 0.01 / denom

                    else:  
                        valueWeight = 1.0  / denom 
                        speedWeight = 0.01 / denom

            stateWeights.cloneAndAppend(
                osim.MocoWeight(valuePath, valueWeight))
            stateWeights.cloneAndAppend(
                osim.MocoWeight(speedPath, speedWeight))

        # Construct the base tracking problem
        # -----------------------------------
        track = osim.MocoTrack()
        track.setName('tracking_walking')
        track.setModel(modelProcessor)
        track.setStatesReference(tableProcessor)
        track.set_states_global_tracking_weight(
            config.state_tracking_weight / model.getNumCoordinates())
        track.set_apply_tracked_states_to_guess(True)           
        track.set_states_weight_set(stateWeights)
        track.set_allow_unused_references(True)
        track.set_track_reference_position_derivatives(True)
        track.set_control_effort_weight(config.control_weight / numForces)
        track.set_initial_time(self.initial_time)
        track.set_final_time(self.final_time)
        track.set_mesh_interval(self.mesh_interval)

        # Customize the base tracking problem
        # -----------------------------------
        study = track.initialize()
        problem = study.updProblem()

        # Copy the tracked states file into the simulation directory
        experimentStates = osim.TableProcessor(
            os.path.join(
                self.root_dir, 
                'tracking_walking_tracked_states.sto')).process()
        experimentStates.trim(self.initial_time, self.final_time)
        osim.STOFileAdapter().write(
            experimentStates, self.get_experiment_states_path(config.name))

        # Set state bounds. 
        # -----------------
        speedBounds = [-20.0, 20.0]
        muscBounds = [0.01, 1.0]
        torqueBounds = [-1.0, 1.0]
        tendonBounds = [0.01, 1.8]
        problem.setStateInfoPattern('/jointset/.*/speed', speedBounds, [], [])
        problem.setControlInfoPattern('/forceset/.*', muscBounds, [], [])
        problem.setControlInfoPattern('/forceset/torque.*', torqueBounds, [], [])
        problem.setControlInfoPattern('/forceset/reserve.*', torqueBounds, [], [])
        problem.setStateInfoPattern('.*/activation', muscBounds, [], [])
        problem.setStateInfoPattern('.*torque.*/activation', torqueBounds, [], [])
        problem.setStateInfoPattern('.*reserve.*/activation', torqueBounds, [], [])
        problem.setStateInfoPattern('.*tendon_force', tendonBounds, [], [])
        for path in stateBounds:
            bounds = stateBounds[path][0]
            initBounds = stateBounds[path][1] if len(stateBounds[path]) > 1 else []
            finalBounds = stateBounds[path][2] if len(stateBounds[path]) > 2 else []
            problem.setStateInfo(path, bounds, initBounds, finalBounds) 

        # Modify the tracking goal
        # ------------------------
        state_tracking = osim.MocoStateTrackingGoal().safeDownCast(
                problem.updGoal('state_tracking'))
        state_tracking.setEnabled(config.tracking_enabled and 
                                  bool(config.state_tracking_weight))
        
        # Modify the control effort goal
        # ------------------------------
        controlEffort = osim.MocoControlGoal().safeDownCast(
                problem.updGoal('control_effort'))
        controlEffort.setDivideByDisplacement(True)
        if not config.control_weight or not config.effort_enabled:
            controlEffort.setEnabled(False)

        # Average speed goal
        # ------------------
        averageSpeed = osim.MocoAverageSpeedGoal()
        averageSpeed.setName('average_speed')
        averageSpeed.set_desired_average_speed(self.walking_speed)
        problem.addGoal(averageSpeed)

        # Torso orientation goal
        # ----------------------
        if config.torso_orientation_weight:

            expStateLabels = experimentStates.getColumnLabels()
            torsoTable = osim.TimeSeriesTable(experimentStates)
            stateVars = model.getStateVariableNames()
            activationVec = osim.Vector(torsoTable.getNumRows(), 0.02)
            for isv in np.arange(stateVars.getSize()):
                stateVar = stateVars.get(int(isv))
                if not stateVar in expStateLabels:
                    torsoTable.appendColumn(stateVar, activationVec)

            torsoOrientationGoal = osim.MocoOrientationTrackingGoal(
                'torso_orientation_goal', 
                config.torso_orientation_weight)
            torsoOrientationGoal.setStatesReference(
                osim.TableProcessor(torsoTable))
            paths = osim.StdVectorString()
            paths.push_back('/bodyset/torso')
            torsoOrientationGoal.setFramePaths(paths)
            torsoOrientationGoal.setEnabled(config.tracking_enabled)
            problem.addGoal(torsoOrientationGoal)

            torsoAngVelGoal = osim.MocoAngularVelocityTrackingGoal(
                'torso_angular_velocity_goal', 
                0.1 * config.torso_orientation_weight)
            torsoAngVelGoal.setStatesReference(
                osim.TableProcessor(torsoTable))
            paths = osim.StdVectorString()
            paths.push_back('/bodyset/torso')
            torsoAngVelGoal.setFramePaths(paths)
            torsoAngVelGoal.setEnabled(config.tracking_enabled)
            problem.addGoal(torsoAngVelGoal)

        # Feet orientation goal
        # ---------------------
        if config.feet_orientation_weight:

            expStateLabels = experimentStates.getColumnLabels()
            feetTable = osim.TimeSeriesTable(experimentStates)
            stateVars = model.getStateVariableNames()
            activationVec = osim.Vector(feetTable.getNumRows(), 0.02)
            for isv in np.arange(stateVars.getSize()):
                stateVar = stateVars.get(int(isv))
                if not stateVar in expStateLabels:
                    feetTable.appendColumn(stateVar, activationVec)

            feetOrientationGoal = osim.MocoOrientationTrackingGoal(
                'feet_orientation_goal', 
                config.feet_orientation_weight)
            feetOrientationGoal.setStatesReference(
                osim.TableProcessor(feetTable))
            paths = osim.StdVectorString()
            paths.push_back('/bodyset/calcn_r')
            paths.push_back('/bodyset/calcn_l')
            feetOrientationGoal.setFramePaths(paths)
            feetOrientationGoal.setEnabled(config.tracking_enabled)
            problem.addGoal(feetOrientationGoal)

            feetAngVelGoal = osim.MocoAngularVelocityTrackingGoal(
                'feet_angular_velocity_goal', 
                0.01 * config.feet_orientation_weight)
            feetAngVelGoal.setStatesReference(
                osim.TableProcessor(feetTable))
            paths = osim.StdVectorString()
            paths.push_back('/bodyset/calcn_r')
            paths.push_back('/bodyset/calcn_l')
            feetAngVelGoal.setFramePaths(paths)
            feetAngVelGoal.setEnabled(config.tracking_enabled)
            problem.addGoal(feetAngVelGoal)
    
        # Distance constraint to prevent intersecting bodies
        # --------------------------------------------------
        distanceConstraint = osim.MocoFrameDistanceConstraint()
        distanceConstraint.setName('distance_constraint')
        footDistance = 0.05
        distanceConstraint.addFramePair(
                osim.MocoFrameDistanceConstraintPair(
                '/bodyset/calcn_l', '/bodyset/calcn_r', footDistance, np.inf))
        distanceConstraint.addFramePair(
                osim.MocoFrameDistanceConstraintPair(
                '/bodyset/toes_l', '/bodyset/toes_r', footDistance, np.inf))
        distanceConstraint.addFramePair(
                osim.MocoFrameDistanceConstraintPair(
                '/bodyset/calcn_l', '/bodyset/toes_r', footDistance, np.inf))
        distanceConstraint.addFramePair(
                osim.MocoFrameDistanceConstraintPair(
                '/bodyset/toes_l', '/bodyset/calcn_r', footDistance, np.inf))

        armDistance = 0.05
        for body in ['humerus', 'ulna', 'radius', 'hand']:
            for side in ['_l', '_r']:
                distanceConstraint.addFramePair(
                        osim.MocoFrameDistanceConstraintPair(
                        '/bodyset/torso', f'/bodyset/{body}{side}', 
                        armDistance, np.inf))

        distanceConstraint.setProjection('vector')
        distanceConstraint.setProjectionVector(osim.Vec3(0, 0, 1))
        problem.addPathConstraint(distanceConstraint)

        # Periodicity constraints
        # -----------------------
        if config.periodic:
            periodic = osim.MocoPeriodicityGoal('periodicity')
            for coord in model.getComponentsList():
                if not type(coord) is osim.Coordinate: continue
                coordName = coord.getName()
                coordValue = coord.getStateVariableNames().get(0)
                coordSpeed = coord.getStateVariableNames().get(1)

                if 'beta' in coordName: continue
                if config.periodic_coordinates_to_include:
                    includeCoord = False
                    for includedCoord in config.periodic_coordinates_to_include:
                        if includedCoord in coordName:
                            includeCoord = True
                    if not includeCoord: continue

                if 'pelvis_tx' in coordName:
                    if config.periodic_speeds:
                        periodic.addStatePair(
                           osim.MocoPeriodicityGoalPair(coordSpeed))
                else:
                    if config.periodic_values:
                        periodic.addStatePair(
                            osim.MocoPeriodicityGoalPair(coordValue))
                    if config.periodic_speeds:
                        periodic.addStatePair(
                           osim.MocoPeriodicityGoalPair(coordSpeed))

            if config.periodic_actuators:
                for actu in model.getComponentsList():
                    if (not (actu.getConcreteClassName().endswith('Muscle') or
                             actu.getConcreteClassName().endswith('Actuator'))): continue
                    if 'reserve' in actu.getAbsolutePathString(): continue

                    periodic.addStatePair(osim.MocoPeriodicityGoalPair(
                            actu.getStateVariableNames().get(0)))
                    periodic.addControlPair(osim.MocoPeriodicityGoalPair(
                            actu.getAbsolutePathString()))
                    # Periodic normalized tendon force
                    if (actu.getConcreteClassName().endswith('Muscle') and
                            actu.getStateVariableNames().getSize() > 1):
                        periodic.addStatePair(osim.MocoPeriodicityGoalPair(
                            actu.getStateVariableNames().get(1)))

            problem.addGoal(periodic)

        # Contact tracking
        # ----------------
        if config.grf_tracking_weight:
            contactTracking = osim.MocoContactTrackingGoal('contact', 
                config.grf_tracking_weight)
            contactTracking.setExternalLoadsFile(self.extloads_fpath)
            rightContactGroup = osim.MocoContactTrackingGoalGroup(
                forceNamesRightFoot, 'Right_GRF', ['/bodyset/toes_r'])
            leftContactGroup = osim.MocoContactTrackingGoalGroup(
                forceNamesLeftFoot, 'Left_GRF', ['/bodyset/toes_l'])
            contactTracking.addContactGroup(rightContactGroup)
            contactTracking.addContactGroup(leftContactGroup)
            contactTracking.setNormalizeTrackingError(True)
            contactTracking.setEnabled(config.tracking_enabled)
            problem.addGoal(contactTracking)        

        # Configure the solver
        # --------------------
        solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
        solver.resetProblem(problem)
        solver.set_optim_constraint_tolerance(self.constraint_tolerance)
        solver.set_optim_convergence_tolerance(self.convergence_tolerance)
        solver.set_num_mesh_intervals(
            int(np.round((self.final_time - self.initial_time) / self.mesh_interval)))
        if self.implicit_multibody_dynamics:
            solver.set_multibody_dynamics_mode('implicit')
            solver.set_minimize_implicit_multibody_accelerations(True)
            solver.set_implicit_multibody_accelerations_weight(config.acceleration_weight)
        else:
            solver.set_multibody_dynamics_mode('explicit')
        solver.set_minimize_implicit_auxiliary_derivatives(
            config.effort_enabled)
        solver.set_implicit_auxiliary_derivatives_weight(
            config.aux_deriv_weight / 6.0)
        solver.set_implicit_auxiliary_derivative_bounds(
            osim.MocoBounds(-100.0, 100.0))
        solver.set_parallel(28)
        solver.set_parameters_require_initsystem(False)
        solver.set_optim_max_iterations(self.num_max_iterations)
        solver.set_scale_variables_using_bounds(True)
        solver.set_optim_finite_difference_scheme('forward')

        # Set the guess
        # -------------
        if config.guess: 

            if config.create_and_insert_guess:
                prevSol = osim.MocoTrajectory(config.guess)
                guess = solver.createGuess()

                statesTable = prevSol.exportToStatesTable()
                controlsTable = prevSol.exportToControlsTable()
                stateLabels = statesTable.getColumnLabels()
                controlLabels = controlsTable.getColumnLabels()

                for label in stateLabels:
                    if 'reserve' in label:
                        statesTable.removeColumn(label)

                for label in controlLabels:
                    if 'reserve' in label:
                        controlsTable.removeColumn(label)

                guess.insertStatesTrajectory(statesTable, True)
                guess.insertControlsTrajectory(controlsTable, True)

            else:
                guess = osim.MocoTrajectory(config.guess)

            if config.randomize_guess:
                guess.randomizeAdd()

            solver.setGuess(guess)
        else:
            # If no guess provided, use the experimental states 
            # trajectory to construct a guess.
            guess = solver.createGuess()
            guess.insertStatesTrajectory(experimentStates, True)
            solver.setGuess(guess)

        # Solve!
        # ------
        # solutionUnsealed = study.solve()
        # solution = solutionUnsealed.unseal()
        # solution.write(self.get_solution_path(config.name))

        # Compute ground reaction forces generated by the contact spheres.
        # ----------------------------------------------------------------
        solution = osim.MocoTrajectory(self.get_solution_path(config.name)) 
        contactForces, copTable = self.create_contact_sphere_force_table(model, solution)
        osim.STOFileAdapter.write(contactForces,
                self.get_solution_path_contacts(config.name))

        externalLoads = self.create_external_loads_table_for_gait(
                model, solution, forceNamesRightFoot, forceNamesLeftFoot,
                copTable)
        osim.STOFileAdapter.write(externalLoads,
                self.get_solution_path_grfs(config.name))

        # Archive solution
        # ----------------
        solution.write(self.get_solution_archive_path(config.name))
        osim.STOFileAdapter.write(externalLoads,
                self.get_solution_archive_path_grfs(config.name))

    def generate_results(self):
        for config in self.configs:
            self.create_torso_tracking_reference(config)
            self.run_tracking_problem(config)

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

            # Muscle-generated moments
            statesTraj = solution.exportToStatesTrajectory(models[i])
            muscle_moments = osim.TimeSeriesTable()

            coordSet = models[i].getCoordinateSet()
            muscleSet = models[i].getMuscles()
            numCoords = coordSet.getSize()
            numMuscles = muscleSet.getSize()

            for istate in range(statesTraj.getSize()):
                state = statesTraj.get(istate)
                time = state.getTime()
                rowVec = osim.RowVector(numCoords, 0.0)

                for icoord in range(numCoords):
                    coord = coordSet.get(icoord)

                    for imusc in range(numMuscles):
                        muscle = muscleSet.get(imusc)
                        tendon_force = muscle_mechanics[config.name].getDependentColumn(
                            f'/forceset/{muscle.getName()}|tendon_force')[istate]
                        moment_arm = muscle.computeMomentArm(state, coord)

                        rowVec[icoord] = rowVec[icoord] + tendon_force * moment_arm

                muscle_moments.appendRow(time, rowVec)

            labels = osim.StdVectorString()
            for icoord in range(numCoords):
                coord = coordSet.get(icoord)
                labels.append(f'/forceset/torque_{coord.getName()}')

            muscle_moments.setColumnLabels(labels)

            for label in labels:
                if (('pelvis' in label) or 
                    ('lumbar' in label) or 
                    ('beta' in label) or 
                    ('elbow' in label) or
                    ('arm' in label) or
                    ('pro_sup' in label)):
                    muscle_moments.removeColumn(label)

            fpath = os.path.join(self.result_fpath,
                    f'muscle_moments_{config.name}.sto')
            osim.STOFileAdapter.write(muscle_moments, fpath)

        # Plot muscle mechanics
        # ---------------------
        self.plot_muscle_mechanics(muscle_mechanics, 'normalized_fiber_length')
        self.plot_muscle_mechanics(muscle_mechanics, 'normalized_fiber_velocity')
        self.plot_muscle_mechanics(muscle_mechanics, 'active_force_length_multiplier')
        self.plot_muscle_mechanics(muscle_mechanics, 'force_velocity_multiplier')
        self.plot_muscle_mechanics(muscle_mechanics, 'passive_fiber_force')
        self.plot_muscle_mechanics(muscle_mechanics, 'active_fiber_force')
        self.plot_muscle_mechanics(muscle_mechanics, 'tendon_strain')
        self.plot_muscle_mechanics(muscle_mechanics, 'tendon_length')

        # Plot center of mass trajectories
        # --------------------------------
        self.plot_center_of_mass(models)

        # Generate PDF report.
        # --------------------
        self.create_pdf_report(models, self.configs)