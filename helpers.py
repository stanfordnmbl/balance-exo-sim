import osimpipeline as osp
import tasks
import os

def generate_main_tasks(trial):

    # inverse kinematics
    ik_setup_task = trial.add_task(osp.TaskIKSetup)
    trial.add_task(osp.TaskIK, ik_setup_task)
    trial.add_task(osp.TaskIKPost, ik_setup_task,
        error_markers=trial.study.error_markers)

    # inverse dynamics
    id_setup_task = trial.add_task(osp.TaskIDSetup, ik_setup_task)
    trial.add_task(osp.TaskID, id_setup_task)
    # trial.add_task(osp.TaskIDPost, id_setup_task)

    return ik_setup_task, id_setup_task

def generate_unperturbed_tasks(study, subject, trial, 
        initial_time, final_time):

    # Initial guess creation
    # ----------------------
    guess_fpath = ''
    scales             = [0.1, 0.5, 1.0]
    mesh_intervals     = [0.04, 0.03, 0.02]
    reserves           = [200, 20, 0]
    implicit_multibody = [True, True, False]
    implicit_tendons   = [True, True, False]
    periodic_flags     = [False, True, True]
    create_and_insert  = [False, False, True]
    convergence_tols   = [1e-1, 1e-2, 1e-2]
    constraint_tols    = [1e-2, 1e-2, 1e-3]  
    zipped = zip(scales, mesh_intervals, reserves,
                 implicit_multibody, implicit_tendons, 
                 periodic_flags, create_and_insert,
                 convergence_tols, constraint_tols)
    for scale, mesh, reserve, imp_multi, imp_ten, periodic, candi, conv_tol, const_tol in zipped:
        trial.add_task(
            tasks.TaskMocoUnperturbedWalkingGuess,
            initial_time, final_time, 
            mesh_interval=mesh, 
            walking_speed=study.walking_speed,
            periodic=periodic,
            cost_scale=scale,
            reserve_strength=reserve,
            implicit_multibody_dynamics=imp_multi,
            implicit_tendon_dynamics=imp_ten,
            guess_fpath=guess_fpath,
            create_and_insert_guess=candi,
            convergence_tolerance=conv_tol,
            constraint_tolerance=const_tol)

        guess_name = f'unperturbed_guess_mesh{mesh}_scale{scale}_reserve{reserve}'
        if periodic: guess_name += '_periodic'
        guess_fpath = os.path.join(
            study.config['results_path'], 'guess', subject.name, 
            f'{guess_name}.sto')

    # Unperturbed walking
    # -------------------
    if study.config['unperturbed_initial_guess']:
        guess_fpath = os.path.join(
            study.config['results_path'],
            'unperturbed', subject.name, 
            'unperturbed.sto')

    trial.add_task(
        tasks.TaskMocoUnperturbedWalking,
        initial_time, final_time, 
        mesh_interval=0.01, 
        walking_speed=study.walking_speed,
        guess_fpath=guess_fpath,
        periodic=True,
        create_and_insert_guess=False)

    # Unperturbed walking w/ different lumbar stiffnesses
    # ---------------------------------------------------
    # if subject.name == 'subject01':
    #     for lumbar_stiffness in study.lumbar_stiffnesses:
    #         if lumbar_stiffness == 1.0: continue
    #         trial.add_task(
    #             tasks.TaskMocoUnperturbedWalking,
    #             initial_time, final_time, 
    #             mesh_interval=0.01, 
    #             walking_speed=study.walking_speed,
    #             guess_fpath=unperturbed_guess_fpath,
    #             periodic=True,
    #             lumbar_stiffness=lumbar_stiffness)

def generate_perturbed_tasks(study, subject, trial, 
        initial_time, final_time, right_strikes, 
        left_strikes):

    for time in study.times:
        for torque in study.torques:
            for subtalar in study.subtalar_peak_torques:
                torque_parameters = [torque / 100.0, 
                                     time / 100.0, 
                                     study.rise / 100.0, 
                                     study.fall / 100.0]
                subtalar_peak_torque = subtalar / 100.0
                # for lumbar_stiffness in study.lumbar_stiffnesses:
                #     if (not lumbar_stiffness == 1.0) and (not subject.name == 'subject01'): continue
                #     trial.add_task(
                #         tasks.TaskMocoPerturbedWalking,
                #         initial_time, final_time, right_strikes, left_strikes,
                #         torque_parameters=torque_parameters,
                #         walking_speed=study.walking_speed,
                #         side='right',
                #         subtalar_torque_perturbation=bool(subtalar),
                #         subtalar_peak_torque=subtalar_peak_torque,
                #         lumbar_stiffness=lumbar_stiffness)
                #     trial.add_task(
                #         tasks.TaskMocoPerturbedWalkingPost,
                #         trial.tasks[-1])

                for coordact in [False, True]:
                    trial.add_task(
                            tasks.TaskMocoPerturbedWalking,
                            initial_time, final_time, right_strikes, left_strikes,
                            torque_parameters=torque_parameters,
                            walking_speed=study.walking_speed,
                            side='right',
                            subtalar_torque_perturbation=bool(subtalar),
                            subtalar_peak_torque=subtalar_peak_torque,
                            lumbar_stiffness=1.0,
                            use_coordinate_actuators=coordact)
                    if study.config['enable_perturbed_plotting_tasks']:
                        trial.add_task(
                                tasks.TaskMocoPerturbedWalkingPost,
                                trial.tasks[-1])

