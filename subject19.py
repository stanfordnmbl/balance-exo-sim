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

    m = util.Measurement('pelvis_x', mset)
    m.add_markerpair('RASI', 'RPSI')
    m.add_markerpair('LASI', 'LPSI')
    m.add_bodyscale('pelvis', 'X')

    m = util.Measurement('pelvis_y', mset)
    m.add_markerpair('LPSI', 'LHJC')
    m.add_markerpair('RPSI', 'RHJC')
    m.add_markerpair('RASI', 'RHJC')
    m.add_markerpair('LASI', 'LHJC')
    m.add_bodyscale('pelvis', 'Y')

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

    m = util.Measurement('humerus', mset)
    m.add_markerpair('LSJC', 'LMEL')
    m.add_markerpair('LSJC', 'LLEL')
    m.add_markerpair('RSJC', 'RLEL')
    m.add_markerpair('RSJC', 'RMEL')
    m.add_bodyscale_bilateral('humerus')

    m = util.Measurement('radius_ulna', mset)
    m.add_markerpair('LLEL', 'LFAradius')
    m.add_markerpair('LMEL', 'LFAulna')
    m.add_markerpair('RMEL', 'RFAulna')
    m.add_markerpair('RLEL', 'RFAradius')
    m.add_bodyscale_bilateral('ulna')
    m.add_bodyscale_bilateral('radius')
    m.add_bodyscale_bilateral('hand')

    ikts.add_ikmarkertask_bilateral('ACR', True, 100.0)
    ikts.add_ikmarkertask('CLAV', True, 250.0)
    ikts.add_ikmarkertask('C7', True, 250.0)
    ikts.add_ikmarkertask_bilateral('ASH', True, 10.0)
    ikts.add_ikmarkertask_bilateral('PSH', True, 10.0)
    ikts.add_ikmarkertask_bilateral('SJC', False, 10.0)
    ikts.add_ikmarkertask_bilateral('LEL', True, 50.0)
    ikts.add_ikmarkertask_bilateral('MEL', True, 50.0)
    ikts.add_ikmarkertask_bilateral('FAsuperior', False, 0.0)
    ikts.add_ikmarkertask_bilateral('FAradius', True, 50.0)
    ikts.add_ikmarkertask_bilateral('FAulna', True, 50.0)

    ikts.add_ikmarkertask_bilateral('ASI', True, 100.0)
    ikts.add_ikmarkertask_bilateral('PSI', True, 50.0)
    ikts.add_ikmarkertask_bilateral('HJC', True, 100.0)
    ikts.add_ikmarkertask_bilateral('LFC', True, 50.0)
    ikts.add_ikmarkertask_bilateral('MFC', True, 50.0)
    ikts.add_ikmarkertask_bilateral('KJC', False, 100.0)
    ikts.add_ikmarkertask_bilateral('LMAL', True, 50.0)
    ikts.add_ikmarkertask_bilateral('MMAL', True, 50.0)
    ikts.add_ikmarkertask_bilateral('AJC', False, 100.0)
    ikts.add_ikmarkertask_bilateral('CAL', True, 25.0)
    ikts.add_ikmarkertask_bilateral('TOE', True, 25.0)
    ikts.add_ikmarkertask_bilateral('MT5', True, 25.0)

    ikts.add_ikmarkertask_bilateral('EJC', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TH1', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TH2', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TH3', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TB1', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TB2', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TB3', False, 0.0)
    ikts.add_ikmarkertask_bilateral('UA1', False, 0.0)
    ikts.add_ikmarkertask_bilateral('UA2', False, 0.0)
    ikts.add_ikmarkertask_bilateral('UA3', False, 0.0)

    ikts.add_ikcoordinatetask('pelvis_tilt', True, 1.0)
    ikts.add_ikcoordinatetask('pelvis_list', True, 1.0)
    ikts.add_ikcoordinatetask_bilateral('hip_flexion', True, 1.0)
    ikts.add_ikcoordinatetask_bilateral('hip_rotation', False, 1.0, manual_value=0.0)
    ikts.add_ikcoordinatetask_bilateral('ankle_angle', True, 1.0, manual_value=0.0)
    ikts.add_ikcoordinatetask('lumbar_bending', True, 0.0)
    ikts.add_ikcoordinatetask('lumbar_rotation', True, 0.0)
    ikts.add_ikcoordinatetask_bilateral('arm_flex', False, 1.0)
    ikts.add_ikcoordinatetask_bilateral('arm_rot', False, 1.0)
    ikts.add_ikcoordinatetask_bilateral('elbow_flex', False, 1.0)
    ikts.add_ikcoordinatetask_bilateral('pro_sup', False, 1.0)

def add_to_study(study):
    subject = study.add_subject(19, 68.50, 1.790)

    cond_args = dict()
    cond_args['walk125'] = (1, '')
    subject.cond_args = cond_args

    static = subject.add_condition('static')
    static_trial = static.add_trial(1, omit_trial_dir=True)

    # `os.path.basename(__file__)` should be `subject19.py`.
    scale_setup_task = subject.add_task(osp.TaskScaleSetup,
            init_time=0.41667,
            final_time=6.14167,
            mocap_trial=static_trial,
            edit_setup_function=scale_setup_fcn,
            addtl_file_dep=['dodo.py', os.path.basename(__file__)])

    subject.add_task(osp.TaskScale,
            scale_setup_task=scale_setup_task,
            ignore_unused_markers=True)

    # Scale max isometric forces based on mass and height
    # ---------------------------------------------------
    subject.add_task(tasks.TaskCopyModelSegmentMasses)
    subject.add_task(tasks.TaskScaleMuscleMaxIsometricForce)  
    subject.scaled_model_fpath = os.path.join(subject.results_exp_path,
        f'{subject.name}_final.osim')
    subject.sim_model_fpath = os.path.join(subject.results_exp_path,
        f'{subject.name}_final.osim')

    # walk2 condition
    # ---------------
    walk2 = subject.add_condition('walk2', metadata={'walking_speed': 1.25})

    # Trial to use
    gait_events = dict()
    gait_events['right_strikes'] = [0.60, 1.65, 2.72, 3.78]
    gait_events['left_toeoffs'] = [0.77, 1.83, 2.89]
    gait_events['left_strikes'] = [1.12, 2.18, 3.25]
    gait_events['right_toeoffs'] = [1.30, 2.36, 3.42]
    walk2_trial = walk2.add_trial(1,
            gait_events=gait_events,
            omit_trial_dir=True,
            )
    walk2_trial.add_task(tasks.TaskUpdateGroundReactionLabels)
    walk2_trial.add_task(tasks.TaskFilterGroundReactions)
    walk2_trial.add_task(osp.TaskGRFGaitLandmarks, min_time=0.5, max_time=5.0)

    # Inverse kinematics and inverse dynamics
    ik_setup_task, id_setup_task = helpers.generate_main_tasks(walk2_trial)

    initial_time = 2.72
    final_time = 3.78
    duration = 1.06
    right_strikes = [2.72, 3.78]
    left_strikes = [3.26]
    walk2_trial.add_task(
        tasks.TaskComputeJointAngleStandardDeviations, 
        ik_setup_task)
    walk2_trial.add_task(
        tasks.TaskTrimTrackingData, 
        ik_setup_task, id_setup_task, 
        initial_time, final_time)

    # unperturbed walking tasks
    # -------------------------
    helpers.generate_unperturbed_tasks(study, subject, walk2_trial, 
        initial_time, final_time)

    # perturbed walking tasks
    # -----------------------
    helpers.generate_perturbed_tasks(study, subject, walk2_trial, 
        initial_time, final_time, right_strikes, left_strikes)

