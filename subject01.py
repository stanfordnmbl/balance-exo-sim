import os

import osimpipeline as osp
import tasks
import helpers

def scale_setup_fcn(util, mset, sset, ikts):
    m = util.Measurement('torso', mset)
    m.add_markerpair('RASI', 'CLAV')
    m.add_markerpair('LASI', 'CLAV')
    m.add_markerpair('LPSI', 'C7')
    m.add_markerpair('RPSI', 'C7')
    m.add_markerpair('RASI',' RACR')
    m.add_markerpair('LASI', 'LACR')
    m.add_bodyscale('torso')

    m = util.Measurement('pelvis_z', mset)
    m.add_markerpair('RPSI', 'LPSI')
    m.add_markerpair('RASI', 'LASI')
    m.add_bodyscale('pelvis', 'Z')

    m = util.Measurement('thigh', mset)
    m.add_markerpair('LHJC', 'LLFC')
    m.add_markerpair('LHJC', 'LMFC')
    m.add_markerpair('RHJC', 'RMFC')
    m.add_markerpair('RHJC', 'RLFC')
    m.add_bodyscale_bilateral('femur')

    m = util.Measurement('shank', mset)
    m.add_markerpair('LLFC', 'LLMAL')
    m.add_markerpair('LMFC', 'LMMAL')
    m.add_markerpair('RLFC', 'RLMAL')
    m.add_markerpair('RMFC', 'RMMAL')
    m.add_bodyscale_bilateral('tibia')

    m = util.Measurement('foot', mset)
    m.add_markerpair('LCAL', 'LMT5')
    m.add_markerpair('LCAL', 'LTOE')
    m.add_markerpair('RCAL', 'RTOE')
    m.add_markerpair('RCAL',' RMT5')
    m.add_bodyscale_bilateral('talus')
    m.add_bodyscale_bilateral('calcn')
    m.add_bodyscale_bilateral('toes')

    # m = util.Measurement('foot_y', mset)
    # m.add_markerpair('LCAL', 'LCAL_proj')
    # m.add_markerpair('LTOE', 'LTOE_proj')
    # m.add_markerpair('LMT5', 'LMT5_proj')
    # m.add_markerpair('RCAL', 'RCAL_proj')
    # m.add_markerpair('RTOE', 'RTOE_proj')
    # m.add_markerpair('RMT5', 'RMT5_proj')
    # m.add_bodyscale_bilateral('talus', 'Y')
    # m.add_bodyscale_bilateral('calcn', 'Y')
    # m.add_bodyscale_bilateral('toes', 'Y')

    # m = util.Measurement('humerus', mset)
    # m.add_markerpair('LSJC', 'LMEL')
    # m.add_markerpair('LSJC', 'LLEL')
    # m.add_markerpair('RSJC', 'RLEL')
    # m.add_markerpair('RSJC', 'RMEL')
    # m.add_bodyscale_bilateral('humerus')

    # m = util.Measurement('radius_ulna', mset)
    # m.add_markerpair('LLEL', 'LFAradius')
    # m.add_markerpair('LMEL', 'LFAulna')
    # m.add_markerpair('RMEL', 'RFAulna')
    # m.add_markerpair('RLEL', 'RFAradius')
    # m.add_bodyscale_bilateral('ulna')
    # m.add_bodyscale_bilateral('radius')
    # m.add_bodyscale_bilateral('hand')

    m = util.Measurement('pelvis_Y', mset)
    m.add_markerpair('LPSI', 'LHJC')
    m.add_markerpair('RPSI', 'RHJC')
    m.add_markerpair('RASI', 'RHJC')
    m.add_markerpair('LASI', 'LHJC')
    m.add_bodyscale('pelvis', 'Y')

    m = util.Measurement('pelvis_X', mset)
    m.add_markerpair('RASI', 'RPSI')
    m.add_markerpair('LASI', 'LPSI')
    m.add_bodyscale('pelvis', 'X')

    ikts.add_ikmarkertask_bilateral('ASI', True, 15.0)
    ikts.add_ikmarkertask_bilateral('PSI', True, 15.0)
    ikts.add_ikmarkertask_bilateral('LFC', True, 10.0)
    ikts.add_ikmarkertask_bilateral('MFC', True, 10.0)
    ikts.add_ikmarkertask_bilateral('LMAL', True, 5.0)
    ikts.add_ikmarkertask_bilateral('MMAL', True, 5.0)
    ikts.add_ikmarkertask_bilateral('CAL', True, 5.0)
    ikts.add_ikmarkertask_bilateral('TOE', True, 5.0)
    ikts.add_ikmarkertask_bilateral('MT5', True, 5.0)
    # ikts.add_ikmarkertask_bilateral('CAL_proj', True, 5.0)
    # ikts.add_ikmarkertask_bilateral('TOE_proj', True, 5.0)
    # ikts.add_ikmarkertask_bilateral('MT5_proj', True, 5.0)
    ikts.add_ikmarkertask_bilateral('ACR', True, 2.0)
    ikts.add_ikmarkertask_bilateral('ASH', True, 2.0)
    ikts.add_ikmarkertask_bilateral('PSH', True, 2.0)
    # ikts.add_ikmarkertask_bilateral('LEL', True, 1.0)
    # ikts.add_ikmarkertask_bilateral('MEL', True, 1.0)
    ikts.add_ikmarkertask_bilateral('HJC', True, 20.0)
    ikts.add_ikmarkertask_bilateral('KJC', True, 10.0)
    ikts.add_ikmarkertask_bilateral('AJC', True, 10.0)
    # ikts.add_ikmarkertask_bilateral('SJC', True, 1.0)
    # ikts.add_ikmarkertask_bilateral('EJC', True, 1.0)
    # ikts.add_ikmarkertask_bilateral('FAsuperior', False, 0.0)
    # ikts.add_ikmarkertask_bilateral('FAradius', False, 0.0)
    # ikts.add_ikmarkertask_bilateral('FAulna', False, 0.0)
    ikts.add_ikmarkertask('CLAV', True, 2.0)
    ikts.add_ikmarkertask('C7', True, 2.0)
    ikts.add_ikmarkertask_bilateral('TH1', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TH2', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TH3', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TB1', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TB2', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TB3', False, 0.0)
    ikts.add_ikmarkertask_bilateral('UA1', False, 0.0)
    ikts.add_ikmarkertask_bilateral('UA2', False, 0.0)
    ikts.add_ikmarkertask_bilateral('UA3', False, 0.0)

def add_to_study(study):
    
    # Add subject to study
    # --------------------
    subject = study.add_subject(1, 72.85, 1.808)

    cond_args = dict()
    subject.cond_args = cond_args

    static = subject.add_condition('static')
    static_trial = static.add_trial(1, omit_trial_dir=True)

    # `os.path.basename(__file__)` should be `subject01.py`.
    scale_setup_task = subject.add_task(osp.TaskScaleSetup,
            init_time=0,
            final_time=0.5, 
            mocap_trial=static_trial,
            edit_setup_function=scale_setup_fcn,
            addtl_file_dep=['dodo.py', os.path.basename(__file__)])

    subject.add_task(osp.TaskScale,
            scale_setup_task=scale_setup_task,
            ignore_unused_markers=True)

    # Scale max isometric forces based on mass and height
    # ---------------------------------------------------
    subject.add_task(tasks.TaskScaleMuscleMaxIsometricForce)

    # Adjust marker locations before inverse kinematics
    # --------------------------------------------------
    marker_adjustments = dict()
    marker_adjustments['RTOE'] = (1, -0.005)
    marker_adjustments['RMT5'] = (1, -0.015)
    marker_adjustments['LTOE'] = (1, -0.005)
    marker_adjustments['LMT5'] = (1, -0.015)
    subject.add_task(tasks.TaskAdjustScaledModel, marker_adjustments)
    subject.scaled_model_fpath = os.path.join(subject.results_exp_path,
        f'{subject.name}_final.osim')
    subject.sim_model_fpath = os.path.join(subject.results_exp_path,
        f'{subject.name}_final.osim')

    # walk2 condition
    # ---------------
    walk2 = subject.add_condition('walk2', metadata={'walking_speed': 1.25})
    
    # Trial to use
    gait_events = dict()
    gait_events['right_strikes'] = [1.18, 2.28, 3.36, 4.49] #, 5.57]
    gait_events['left_toeooffs'] = [1.37, 2.47, 3.58] #, 4.68]
    gait_events['left_strikes'] = [1.73, 2.84, 3.94] #, 5.05]
    gait_events['right_toeoffs'] = [1.93, 3.03, 4.14] #, 5.25]

    walk2_trial = walk2.add_trial(1,
            gait_events=gait_events,
            omit_trial_dir=True,
            )
    walk2_trial.add_task(tasks.TaskUpdateGroundReactionLabels)
    walk2_trial.add_task(tasks.TaskFilterGroundReactions)
    walk2_trial.add_task(osp.TaskGRFGaitLandmarks, min_time=0.5, max_time=5.0)

    # Inverse kinematics and inverse dynamics
    ik_setup_task, id_setup_task = helpers.generate_main_tasks(walk2_trial)

    initial_time = 3.36
    final_time = 4.49
    duration = 1.13
    right_strikes = [3.36, 4.49]
    left_strikes = [3.94]
    walk2_trial.add_task(
        tasks.TaskComputeJointAngleStandardDeviations, 
        ik_setup_task)
    walk2_trial.add_task(
        tasks.TaskTrimTrackingData, 
        ik_setup_task, id_setup_task, 
        initial_time, final_time)

    # sensitivity tasks
    # -----------------
    helpers.generate_sensitivity_tasks(study, subject, walk2_trial,
        initial_time, final_time)

    # unperturbed walking tasks
    # -------------------------
    helpers.generate_unperturbed_tasks(study, subject, walk2_trial, 
        initial_time, final_time)

    # perturbed walking tasks
    # -----------------------
    helpers.generate_perturbed_tasks(study, subject, walk2_trial, 
        initial_time, final_time, right_strikes, left_strikes)
