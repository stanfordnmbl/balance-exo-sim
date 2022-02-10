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

    # trial.add_task(
    #     tasks.TaskMocoUnperturbedWalkingGuess,
    #     initial_time, final_time, mesh_interval=0.035, 
    #     walking_speed=study.walking_speed,
    #     guess_fpath=None,
    #     costs_enabled=True,
    #     periodic=False,
    #     cost_scale=1e9)

    # unperturbed_guess_fpath = os.path.join(
    #     study.config['results_path'], 'guess', subject.name, 
    #     'unperturbed_guess_mesh35_scale1000000000.sto')
    # trial.add_task(
    #     tasks.TaskMocoUnperturbedWalkingGuess,
    #     initial_time, final_time, mesh_interval=0.035, 
    #     walking_speed=study.walking_speed,
    #     guess_fpath=unperturbed_guess_fpath,
    #     costs_enabled=True,
    #     periodic=False,
    #     cost_scale=1e7)

    # unperturbed_guess_fpath = os.path.join(
    #     study.config['results_path'], 'guess', subject.name, 
    #     'unperturbed_guess_mesh35_scale10000000.sto')
    trial.add_task(
        tasks.TaskMocoUnperturbedWalkingGuess,
        initial_time, final_time, mesh_interval=0.035, 
        walking_speed=study.walking_speed,
        guess_fpath=None,
        costs_enabled=True,
        periodic=False,
        cost_scale=1e5,
        reserve_strength=1000)

    unperturbed_guess_fpath = os.path.join(
        study.config['results_path'], 'guess', subject.name, 
        'unperturbed_guess_mesh35_scale100000_reserve1000.sto')
    trial.add_task(
        tasks.TaskMocoUnperturbedWalkingGuess,
        initial_time, final_time, mesh_interval=0.035, 
        walking_speed=study.walking_speed,
        guess_fpath=unperturbed_guess_fpath,
        costs_enabled=True,
        periodic=False,
        cost_scale=1e3,
        reserve_strength=1000)

    unperturbed_guess_fpath = os.path.join(
        study.config['results_path'], 'guess', subject.name, 
        'unperturbed_guess_mesh35_scale1000_reserve1000.sto')
    trial.add_task(
        tasks.TaskMocoUnperturbedWalkingGuess,
        initial_time, final_time, mesh_interval=0.035, 
        walking_speed=study.walking_speed,
        guess_fpath=unperturbed_guess_fpath,
        costs_enabled=True,
        periodic=False,
        cost_scale=1e2,
        reserve_strength=100)

    unperturbed_guess_fpath = os.path.join(
        study.config['results_path'], 'guess', subject.name, 
        'unperturbed_guess_mesh35_scale100_reserve100.sto')
    trial.add_task(
        tasks.TaskMocoUnperturbedWalkingGuess,
        initial_time, final_time, mesh_interval=0.035, 
        walking_speed=study.walking_speed,
        guess_fpath=unperturbed_guess_fpath,
        costs_enabled=True,
        periodic=False,
        reserve_strength=100)

    unperturbed_guess_fpath = os.path.join(
        study.config['results_path'],
        'guess', subject.name, 
        'unperturbed_guess_mesh35_scale1_reserve100.sto')
    trial.add_task(
        tasks.TaskMocoUnperturbedWalkingGuess,
        initial_time, final_time, mesh_interval=0.035, 
        walking_speed=study.walking_speed,
        guess_fpath=unperturbed_guess_fpath,
        costs_enabled=True,
        periodic=False,
        reserve_strength=10)

    unperturbed_guess_fpath = os.path.join(
        study.config['results_path'],
        'guess', subject.name, 
        'unperturbed_guess_mesh35_scale1_reserve10.sto')
    trial.add_task(
        tasks.TaskMocoUnperturbedWalkingGuess,
        initial_time, final_time, mesh_interval=0.035, 
        walking_speed=study.walking_speed,
        guess_fpath=unperturbed_guess_fpath,
        costs_enabled=True,
        periodic=False)

    unperturbed_guess_fpath = os.path.join(
        study.config['results_path'],
        'guess', subject.name, 
        'unperturbed_guess_mesh35_scale1.sto')
    trial.add_task(
        tasks.TaskMocoUnperturbedWalkingGuess,
        initial_time, final_time, mesh_interval=0.035, 
        walking_speed=study.walking_speed,
        guess_fpath=unperturbed_guess_fpath,
        costs_enabled=True,
        periodic=True)

    unperturbed_guess_fpath = os.path.join(
        study.config['results_path'],
        'guess', subject.name, 
        'unperturbed_guess_mesh35_scale1_periodic.sto')
    trial.add_task(
        tasks.TaskMocoUnperturbedWalkingGuess,
        initial_time, final_time, mesh_interval=0.020, 
        walking_speed=study.walking_speed,
        guess_fpath=unperturbed_guess_fpath,
        costs_enabled=True,
        periodic=True)

    # unperturbed_guess_fpath = os.path.join(
    #     study.config['results_path'],
    #     'guess', subject.name, 
    #     'unperturbed_guess_mesh20_scale1_periodic.sto')
    unperturbed_guess_fpath = os.path.join(
        study.config['results_path'],
        'unperturbed', subject.name, 
        'unperturbed.sto')
    trial.add_task(
        tasks.TaskMocoUnperturbedWalking,
        initial_time, final_time, 
        mesh_interval=0.01, 
        walking_speed=study.walking_speed,
        guess_fpath=unperturbed_guess_fpath,
        periodic=True)

def generate_sensitivity_tasks(study, subject, trial,
        initial_time, final_time):

    initial_guess_fpath = os.path.join(
        study.config['results_path'],
        'unperturbed', subject.name, 
        'unperturbed.sto')

    guess_fpath = initial_guess_fpath
    randomize_guess = True
    for mesh in [0.05, 0.04, 0.03, 0.02, 0.01]:
        trial.add_task(
            tasks.TaskMocoSensitivityAnalysis,
            initial_time, final_time,
            guess_fpath=guess_fpath, 
            mesh_interval=mesh,
            tolerance=1e-1, 
            walking_speed=study.walking_speed,
            mesh_or_tol_analysis='mesh',
            randomize_guess=randomize_guess)
        suffix = f'mesh{int(1000*mesh)}_tol1000'
        label = f'unperturbed_sensitivity_{suffix}_mesh'
        guess_fpath = os.path.join(
            study.config['results_path'], 'sensitivity',
            subject.name, suffix, 
            f'{label}.sto')
        randomize_guess = False
       
    guess_fpath = initial_guess_fpath
    randomize_guess = True
    for tol in [1e1, 1e0, 1e-1, 1e-2, 1e-3]:
        trial.add_task(
            tasks.TaskMocoSensitivityAnalysis,
            initial_time, final_time,
            guess_fpath=guess_fpath, 
            mesh_interval=0.01,
            tolerance=tol, 
            walking_speed=study.walking_speed,
            mesh_or_tol_analysis='tol',
            randomize_guess=randomize_guess)
        suffix = f'mesh10_tol{int(1e4*tol)}'
        label = f'unperturbed_sensitivity_{suffix}_tol'
        guess_fpath = os.path.join(
            study.config['results_path'], 'sensitivity',
            subject.name, suffix, 
            f'{label}.sto')
        randomize_guess = False

def generate_perturbed_tasks(study, subject, trial, 
        initial_time, final_time, right_strikes, 
        left_strikes):

    unperturbed_guess_fpath = os.path.join(
            study.config['results_path'], 'unperturbed', 
            subject.name, 'unperturbed.sto')

    for time in [0.5, 0.6]:
        for torque in [0.25, 0.50, 0.75, 1.0]:
            if torque == 0.25:
                guess_fpath = unperturbed_guess_fpath

            torque_parameters = [torque, time, 0.25, 0.1]
            trial.add_task(
                tasks.TaskMocoPerturbedWalking,
                initial_time, final_time, right_strikes, left_strikes,
                guess_fpath=guess_fpath, 
                mesh_interval=0.01, 
                torque_parameters=torque_parameters,
                walking_speed=study.walking_speed,
                side='right')
            trial.add_task(
                tasks.TaskMocoPerturbedWalkingPost,
                trial.tasks[-1])

            label = (f'perturbed_torque{int(100*torque)}'
                     f'_time{int(100*time)}')
            guess_fpath = os.path.join(
                study.config['results_path'], label,
                subject.name, f'{label}.sto')