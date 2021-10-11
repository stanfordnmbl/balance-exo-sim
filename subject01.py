import os

from osimpipeline import utilities as util
import osimpipeline as osp
import tasks
import helpers
import numpy as np

def scale_setup_fcn(util, mset, sset, ikts):
    m = util.Measurement('r_footX', mset)
    m.add_markerpair('RTOE_proj', 'R_AJC_proj')
    m.add_bodyscale('talus_r', 'X')
    m.add_bodyscale('calcn_r', 'X')
    m.add_bodyscale('toes_r', 'X')

    m = util.Measurement('l_footX', mset)
    m.add_markerpair('LTOE_proj', 'L_AJC_proj')
    m.add_bodyscale('talus_l', 'X')
    m.add_bodyscale('calcn_l', 'X')
    m.add_bodyscale('toes_l', 'X')

    m = util.Measurement('r_footZ', mset)
    m.add_markerpair('RLAK', 'RMAK')
    m.add_bodyscale('talus_r', 'Z')
    m.add_bodyscale('calcn_r', 'Z')
    m.add_bodyscale('toes_r', 'Z')

    m = util.Measurement('l_footZ', mset)
    m.add_markerpair('LLAK', 'LMAK')
    m.add_bodyscale('talus_l', 'Z')
    m.add_bodyscale('calcn_l', 'Z')
    m.add_bodyscale('toes_l', 'Z')

    m = util.Measurement('r_tibiaY', mset)
    m.add_markerpair('R_AJC', 'R_KJC')
    m.add_bodyscale('tibia_r', 'Y')

    m = util.Measurement('l_tibiaY', mset)
    m.add_markerpair('L_AJC', 'L_KJC')
    m.add_bodyscale('tibia_l', 'Y')

    m = util.Measurement('r_tibiaXZ', mset)
    m.add_markerpair('RLAK', 'RMAK')
    m.add_bodyscale('tibia_r', 'XZ')

    m = util.Measurement('l_tibiaXZ', mset)
    m.add_markerpair('LLAK', 'LMAK')
    m.add_bodyscale('tibia_l', 'XZ')

    m = util.Measurement('r_femurY', mset)
    m.add_markerpair('R_HJC', 'R_KJC')
    m.add_bodyscale('femur_r', 'Y')

    m = util.Measurement('l_femurY', mset)
    m.add_markerpair('L_HJC', 'L_KJC')
    m.add_bodyscale('femur_l', 'Y')

    m = util.Measurement('r_femurXZ', mset)
    m.add_markerpair('RLKN', 'RMKN')
    m.add_bodyscale('femur_r', 'XZ')
    m.add_bodyscale('patella_r', 'XYZ')

    m = util.Measurement('l_femurXZ', mset)
    m.add_markerpair('LLKN', 'LMKN')
    m.add_bodyscale('femur_l', 'XZ')
    m.add_bodyscale('patella_l', 'XYZ')

    m = util.Measurement('pelvisX', mset)
    m.add_markerpair('midASIS', 'midPSIS')
    m.add_bodyscale('pelvis', 'X')

    m = util.Measurement('pelvisY', mset)
    m.add_markerpair('midHJC', 'midPelvis')
    m.add_bodyscale('pelvis', 'Y')

    m = util.Measurement('pelvisZ', mset)
    m.add_markerpair('R_HJC', 'L_HJC')
    m.add_bodyscale('pelvis', 'Z')

    m = util.Measurement('torsoX', mset)
    m.add_markerpair('C7', 'CLAV')
    m.add_bodyscale('torso', 'X')

    m = util.Measurement('torsoY', mset)
    m.add_markerpair('C7', 'midPSIS')
    m.add_bodyscale('torso', 'Y')

    m = util.Measurement('torsoZ', mset)
    m.add_markerpair('RSHL', 'LSHL')
    m.add_bodyscale('torso', 'Z')

    # m = util.Measurement('r_humerusXZ', mset)
    # m.add_markerpair('RASH', 'RPSH')
    # m.add_bodyscale('humerus_r', 'XZ')

    # m = util.Measurement('l_humerusXZ', mset)
    # m.add_markerpair('LASH', 'LPSH')
    # m.add_bodyscale('humerus_l', 'XZ')

    # m = util.Measurement('r_humerusY', mset)
    # m.add_markerpair('RSHL', 'RLEL')
    # m.add_bodyscale('humerus_r', 'Y')

    # m = util.Measurement('l_humerusY', mset)
    # m.add_markerpair('LSHL', 'LLEL')
    # m.add_bodyscale('humerus_l', 'Y')

    # m = util.Measurement('r_FAXZ', mset)
    # m.add_markerpair('RRAD', 'RULN')
    # m.add_bodyscale('radius_r', 'XZ')
    # m.add_bodyscale('ulna_r', 'XZ')

    # m = util.Measurement('l_FAXZ', mset)
    # m.add_markerpair('LRAD', 'LULN')
    # m.add_bodyscale('radius_l', 'XZ')
    # m.add_bodyscale('ulna_l', 'XZ')

    # m = util.Measurement('r_FAY', mset)
    # m.add_markerpair('RLEL', 'RULN')
    # m.add_bodyscale('radius_r', 'Y')
    # m.add_bodyscale('ulna_r', 'Y')

    # m = util.Measurement('l_FAY', mset)
    # m.add_markerpair('LLEL', 'LULN')
    # m.add_bodyscale('radius_l', 'Y')
    # m.add_bodyscale('ulna_l', 'Y')

    ikts.add_ikmarkertask('C7', True, 10.0)
    ikts.add_ikmarkertask('CLAV', True, 5.0)
    ikts.add_ikmarkertask_bilateral('SHL', True, 5.0)
    ikts.add_ikmarkertask_bilateral('ASI', True, 10.0)
    ikts.add_ikmarkertask_bilateral('PSI', True, 10.0)
    ikts.add_ikmarkertask_bilateral('_HJC', True, 20.0)
    ikts.add_ikmarkertask_bilateral('_KJC', True, 20.0)
    ikts.add_ikmarkertask_bilateral('_AJC', True, 10.0)
    ikts.add_ikmarkertask_bilateral('LKN', True, 10.0)
    ikts.add_ikmarkertask_bilateral('MKN', True, 10.0)
    ikts.add_ikmarkertask_bilateral('LAK', True, 10.0)
    ikts.add_ikmarkertask_bilateral('MAK', True, 10.0)
    ikts.add_ikmarkertask_bilateral('MT5', True, 10.0)
    ikts.add_ikmarkertask_bilateral('CAL', True, 10.0)
    ikts.add_ikmarkertask_bilateral('TOE', True, 10.0)
    ikts.add_ikmarkertask_bilateral('LEL', False, 10.0)
    ikts.add_ikmarkertask_bilateral('ULN', False, 10.0)
    ikts.add_ikmarkertask_bilateral('RAD', False, 10.0)

    ikts.add_ikcoordinatetask_bilateral('knee_angle', True, 0.0, 30.0)

    ikts.add_ikmarkertask_bilateral('PSH', False, 0.0)
    ikts.add_ikmarkertask_bilateral('ASH', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TH1', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TH2', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TH3', False, 0.0)
    ikts.add_ikmarkertask_bilateral('SH1', False, 0.0)
    ikts.add_ikmarkertask_bilateral('SH2', False, 0.0)
    ikts.add_ikmarkertask_bilateral('SH3', False, 0.0)
    ikts.add_ikmarkertask_bilateral('UA1', False, 0.0)
    ikts.add_ikmarkertask_bilateral('UA2', False, 0.0)
    ikts.add_ikmarkertask_bilateral('UA3', False, 0.0)
    ikts.add_ikmarkertask_bilateral('FA', False, 0.0)
    ikts.add_ikmarkertask_bilateral('FA', False, 0.0)
    ikts.add_ikmarkertask_bilateral('MT5_proj', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TOE_proj', False, 0.0)
    ikts.add_ikmarkertask_bilateral('_AJC_proj', False, 0.0)
    ikts.add_ikmarkertask_bilateral('_KJC_proj', False, 0.0)
    ikts.add_ikmarkertask('midASIS', False, 0.0)
    ikts.add_ikmarkertask('midPSIS', False, 0.0)
    ikts.add_ikmarkertask('midHJC', False, 0.0)
    ikts.add_ikmarkertask('midPelvis', False, 0.0)

def add_to_study(study):
    
    # Add subject to study
    # --------------------
    subject = study.add_subject(1, 70.3, 1.7297)

    static = subject.add_condition('static')
    static_trial = static.add_trial(1, omit_trial_dir=True)

    # `os.path.basename(__file__)` should be `subject01.py`.
    scale_setup_task = subject.add_task(osp.TaskScaleSetup,
            init_time=0,
            final_time=0.1, 
            mocap_trial=static_trial,
            edit_setup_function=scale_setup_fcn,
            addtl_file_dep=['dodo.py', os.path.basename(__file__)])

    subject.add_task(osp.TaskScale, scale_setup_task=scale_setup_task,
        ignore_unused_markers=True)

    # Scale max isometric forces based on mass and height
    # ---------------------------------------------------
    subject.add_task(tasks.TaskScaleMuscleMaxIsometricForce)
    
    # Adjust marker locations before inverse kinematics
    # --------------------------------------------------
    marker_adjustments = dict()
    marker_adjustments['RASI'] = (1, 0.03)
    marker_adjustments['LASI'] = (1, 0.03)
    subject.add_task(tasks.TaskAdjustScaledModel, marker_adjustments)
    subject.scaled_model_fpath = os.path.join(subject.results_exp_path,
        '%s_final.osim' % subject.name)

    # unperturbed condition (left foot gait cycle)
    # ---------------------------------------------
    unperturbed = subject.add_condition('unperturbed')
    gait_events = dict()
    gait_events['right_strikes'] = [0.67, 1.76, 2.86, 3.93, 5.02, 6.20, 7.20, 8.30, 9.40]
    gait_events['left_strikes'] = [1.23, 2.30, 3.40, 4.49, 5.58, 6.65, 7.76, 8.86]
    unperturbed_trial = unperturbed.add_trial(1,
            gait_events=gait_events,
            omit_trial_dir=True,
            )
    unperturbed_trial.add_task(osp.TaskGRFGaitLandmarks,
        min_time=0.0, max_time=10.0)

    ik_setup_task, id_setup_task = helpers.generate_main_tasks(
        unperturbed_trial)

    initial_time = 3.93
    final_time = 5.02
    right_strikes = [3.93, 5.02]
    left_strikes = [4.49]
    unperturbed_trial.add_task(tasks.TaskComputeJointAngleStandardDeviations, ik_setup_task)
    unperturbed_trial.add_task(tasks.TaskTrimTrackingData, ik_setup_task, id_setup_task, 
        initial_time, final_time)
    
    # initial guess creation for unperturbed tracking condition
    # ---------------------------------------------------------
    # unperturbed_guess_fpath = os.path.join(study.config['results_path'],
    #     'guess', subject.name, f'unperturbed_guess_mesh10_costsDisabled.sto')
    unperturbed_guess_fpath = None
    unperturbed_trial.add_task(tasks.TaskMocoUnperturbedWalkingGuess,
        initial_time, final_time, mesh_interval=0.035, 
        walking_speed=study.walking_speed,
        guess_fpath=unperturbed_guess_fpath,
        pelvis_boundary_conditions=True,
        costs_enabled=False,
        periodic=True)

    unperturbed_guess_fpath = os.path.join(study.config['results_path'],
        'guess', subject.name, f'unperturbed_guess_mesh35_periodic.sto')
    # unperturbed_guess_fpath = None
    mesh_intervals = [0.035]
    for mesh_interval in mesh_intervals:
        unperturbed_trial.add_task(tasks.TaskMocoUnperturbedWalkingGuess,
            initial_time, final_time, mesh_interval=mesh_interval, 
            walking_speed=study.walking_speed,
            guess_fpath=unperturbed_guess_fpath,
            pelvis_boundary_conditions=False,
            costs_enabled=True,
            periodic=True)
        unperturbed_guess_fpath = os.path.join(study.config['results_path'],
                'guess', subject.name, 
                f'unperturbed_guess_mesh{int(1000*mesh_interval)}_periodic.sto')

    # unperturbed_guess_fpath = os.path.join(study.config['results_path'],
    #             'guess', subject.name, 
    #             f'unperturbed_guess_mesh35_periodic.sto')
    unperturbed_guess_fpath = os.path.join(study.config['results_path'],
                'unperturbed', subject.name, 
                f'unperturbed_mesh35.sto')
    mesh_intervals = [0.035, 0.020, 0.010]
    for mesh_interval in mesh_intervals:    
        unperturbed_trial.add_task(tasks.TaskMocoUnperturbedWalking,
            initial_time, final_time, mesh_interval=mesh_interval, 
            walking_speed=study.walking_speed,
            guess_fpath=unperturbed_guess_fpath,
            periodic=True)
        unperturbed_guess_fpath = os.path.join(study.config['results_path'],
                'unperturbed', subject.name, 
                f'unperturbed_mesh{int(1000*mesh_interval)}.sto')

    unperturbed_fpath = os.path.join(study.config['results_path'],
            'unperturbed', subject.name, f'unperturbed_mesh20.sto')
    unperturbed_trial.add_task(tasks.TaskMocoAnkleTorqueBaselineWalking,
            initial_time, final_time, right_strikes, left_strikes,
            unperturbed_fpath,
            mesh_interval=0.02, 
            walking_speed=study.walking_speed,
            periodic=True)

    # delay = 400
    # for torque in [0.25, 0.5, 0.75, 1.0]:
    #     for time in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    #         torque_parameters = [torque, time, 0.25, 0.1]
    #         unperturbed_trial.add_task(tasks.TaskMocoAnkleTorquePerturbedWalking,
    #             ik_setup_task, id_setup_task, unperturbed_fpath, 
    #             mesh_interval=0.02, 
    #             torque_parameters=torque_parameters,
    #             walking_speed=study.walking_speed,
    #             perturb_response_delay=delay,
    #             side='right')
    #         unperturbed_trial.add_task(
    #             tasks.TaskMocoAnkleTorquePerturbedWalkingPost,
    #             unperturbed_trial.tasks[-1])
    