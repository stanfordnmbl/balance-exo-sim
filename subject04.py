import os

import osimpipeline as osp
import tasks
import helpers

def scale_setup_fcn(pmm, mset, sset, ikts):
    m = pmm.Measurement('torso', mset)
    m.add_markerpair('LPSH', 'LPSI')
    m.add_markerpair('LASH', 'LASI')
    m.add_markerpair('RPSH', 'RPSI')
    m.add_markerpair('RASH', 'RASI')
    m.add_markerpair('C7', 'RASI')
    m.add_markerpair('C7', 'LASI')
    m.add_bodyscale('torso')

    m = pmm.Measurement('pelvis_z', mset)
    m.add_markerpair('LPSI', 'RPSI')
    m.add_markerpair('RASI', 'LASI')
    m.add_markerpair('RHJC', 'LHJC')
    m.add_bodyscale('pelvis', 'Z')

    m = pmm.Measurement('thigh', mset)
    m.add_markerpair('LHJC', 'LLFC')
    m.add_markerpair('LHJC', 'LMFC')
    m.add_markerpair('RHJC', 'RMFC')
    m.add_markerpair('RHJC', 'RLFC')
    m.add_bodyscale_bilateral('femur')

    m = pmm.Measurement('shank', mset)
    m.add_markerpair('LLFC', 'LLMAL')
    m.add_markerpair('LMFC', 'LMMAL')
    m.add_markerpair('RLFC', 'RLMAL')
    m.add_markerpair('RMFC', 'RMMAL')
    m.add_bodyscale_bilateral('tibia', 'XY')

    m = pmm.Measurement('foot', mset)
    m.add_markerpair('LCAL', 'LMT5')
    m.add_markerpair('LCAL', 'LTOE')
    m.add_markerpair('RCAL',' RMT5')
    m.add_markerpair('RCAL', 'RTOE')
    m.add_bodyscale_bilateral('talus')
    m.add_bodyscale_bilateral('calcn')
    m.add_bodyscale_bilateral('toes')

    # m = pmm.Measurement('humerus', mset)
    # m.add_markerpair('LACR', 'LMEL')
    # m.add_markerpair('LACR', 'LLEL')
    # m.add_markerpair('RACR', 'RMEL')
    # m.add_markerpair('RACR', 'RLEL')
    # m.add_bodyscale_bilateral('humerus')

    # m = pmm.Measurement('radius_ulna', mset)
    # m.add_markerpair('LLEL', 'LFAradius')
    # m.add_markerpair('LMEL', 'LFAulna')
    # m.add_markerpair('RMEL', 'RFAulna')
    # m.add_markerpair('RLEL', 'RFAradius')
    # m.add_bodyscale_bilateral('ulna')
    # m.add_bodyscale_bilateral('radius')
    # m.add_bodyscale_bilateral('hand')

    m = pmm.Measurement('pelvis_Y', mset)
    m.add_markerpair('RASI', 'RHJC')
    m.add_markerpair('LASI', 'LHJC')
    m.add_bodyscale('pelvis', 'Y')

    m = pmm.Measurement('pelvis_X', mset)
    m.add_markerpair('RASI', 'RPSI')
    m.add_markerpair('LASI', 'LPSI')
    m.add_bodyscale('pelvis', 'X')

    m = pmm.Measurement('shank_width', mset)
    m.add_markerpair('LLMAL', 'LMMAL')
    m.add_markerpair('RLMAL', 'RMMAL')
    m.add_bodyscale_bilateral('tibia', 'Z')

    # Unused m = pmm.Measurement('thigh_width', mset)
    # Unused m.add_markerpair('LLFC', 'LMFC')
    # Unused m.add_markerpair('RLFC', 'RMFC')

    ikts.add_ikmarkertask_bilateral('ACR', True, 250.0)
    ikts.add_ikmarkertask('C7', True, 500.0)
    ikts.add_ikmarkertask('CLAV', True, 10.0)
    # ikts.add_ikmarkertask_bilateral('LEL', True, 50.0)
    # ikts.add_ikmarkertask_bilateral('MEL', True, 50.0)
    # ikts.add_ikmarkertask_bilateral('FAradius', True, 50.0)
    # ikts.add_ikmarkertask_bilateral('FAulna', True, 50.0)
    ikts.add_ikmarkertask_bilateral('ASI', True, 500.0)
    ikts.add_ikmarkertask_bilateral('PSI', True, 500.0)
    ikts.add_ikmarkertask_bilateral('HJC', True, 1000.0)
    ikts.add_ikmarkertask_bilateral('LFC', True, 500.0)
    ikts.add_ikmarkertask_bilateral('MFC', True, 500.0)
    ikts.add_ikmarkertask_bilateral('LMAL', True, 500.0)
    ikts.add_ikmarkertask_bilateral('MMAL', True, 500.0)
    ikts.add_ikmarkertask_bilateral('CAL', True, 25.0)
    ikts.add_ikmarkertask_bilateral('TOE', True, 25.0)
    ikts.add_ikmarkertask_bilateral('MT5', True, 25.0)

    ikts.add_ikcoordinatetask('pelvis_list', True, 0.0, 1.0)
    ikts.add_ikcoordinatetask_bilateral('knee_angle', True, 0.0, 1.0)
    ikts.add_ikcoordinatetask_bilateral('ankle_angle', True, 0.0, 1.0)

    ikts.add_ikmarkertask_bilateral('ASH', False, 10.0)
    ikts.add_ikmarkertask_bilateral('PSH', False, 10.0)

    # ikts.add_ikmarkertask_bilateral('FAsuperior', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TH1', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TH2', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TH3', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TB1', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TB2', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TB3', False, 0.0)
    ikts.add_ikmarkertask_bilateral('UA1', False, 0.0)
    ikts.add_ikmarkertask_bilateral('UA2', False, 0.0)
    ikts.add_ikmarkertask_bilateral('UA3', False, 0.0)
    ikts.add_ikmarkertask_bilateral('AJC', False, 0.0)
    # ikts.add_ikmarkertask_bilateral('SJC', False, 0.0)
    # ikts.add_ikmarkertask_bilateral('EJC', False, 0.0)
    ikts.add_ikmarkertask_bilateral('KJC', False, 0.0)


def add_to_study(study):
    subject = study.add_subject(4, 80.30, 1.801)

    cond_args = dict()
    subject.cond_args = cond_args

    static = subject.add_condition('static')
    static_trial = static.add_trial(1, omit_trial_dir=True)

    # `os.path.basename(__file__)` should be `subject02.py`.
    scale_setup_task = subject.add_task(osp.TaskScaleSetup,
            init_time=0.73,
            final_time=3.89, 
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
    marker_adjustments['RTOE'] = (1, -0.02)
    marker_adjustments['RMT5'] = (1, -0.03)
    marker_adjustments['LTOE'] = (1, -0.02)
    marker_adjustments['LMT5'] = (1, -0.03)
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
    gait_events['right_strikes'] = [0.27, 1.41, 4.74, 5.90]
    gait_events['left_toeoffs'] = [0.42, 1.57, 4.88]
    gait_events['left_strikes'] = [0.81, 1.95, 5.32]
    gait_events['right_toeoffs'] = [0.97, 2.10, 5.48]
    gait_events['stride_times'] = [1.41-0.27, 2.52-1.41, 5.90-4.74]
    walk2_trial = walk2.add_trial(1,
            gait_events=gait_events,
            omit_trial_dir=True,
            )
    walk2_trial.add_task(tasks.TaskUpdateGroundReactionLabels)
    walk2_trial.add_task(tasks.TaskFilterGroundReactions)
    walk2_trial.add_task(osp.TaskGRFGaitLandmarks, min_time=0.5, max_time=5.0)

    # Inverse kinematics and inverse dynamics
    ik_setup_task, id_setup_task = helpers.generate_main_tasks(walk2_trial)

    initial_time = 1.41
    final_time = 2.52
    right_strikes = [1.41, 2.52]
    left_strikes = [1.95]
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

