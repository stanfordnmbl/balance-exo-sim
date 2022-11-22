import sys
sys.path.insert(1, 'code')
sys.path.insert(1, 'osimpipeline')
sys.path.insert(1, 'osimpipeline/osimpipeline')

import os
import yaml
import numpy as np
with open('config.yaml') as f:
    config = yaml.safe_load(f)
if 'opensim_home' not in config:
    raise Exception('You must define the field `opensim_home` in config.yaml '
                    'to point to the root of your OpenSim 4.0 (or later) '
                    'installation.')
sys.path.insert(1, os.path.join(config['opensim_home'], 'sdk', 'python'))

import opensim as osim


torques = [0, 10, 10, 10, 0]
subtalars = ['_subtalar-10', '_subtalar-10', '', '_subtalar10', '_subtalar10'] 
subjects = ['subject01', 
            'subject02', 
            'subject04', 
            'subject18', 
            'subject19']


time = 35
unperturbed = f'perturbed_torque0_time{time}_rise10_fall5'


values = np.zeros((5, 5))
for isubj, subject in enumerate(subjects):
    model = osim.Model(
                    os.path.join(
                        config['results_path'], 
                        'unperturbed', subject,
                        'model_unperturbed.osim'))
    state = model.initSystem()
    mass = model.getTotalMass(state)


    unp = osim.TimeSeriesTable(
                    os.path.join(
                        config['results_path'], 
                        'unperturbed', subject,
                        'center_of_mass_unperturbed.sto'))

    com_height = np.mean(unp.getDependentColumn('/|com_position_y').to_numpy())

    vmax = np.sqrt(9.81 * com_height)

    label = f'perturbed_torque0_time{time}_rise10_fall5'
    unp_ts = osim.TimeSeriesTable(
                    os.path.join(
                        config['results_path'], 
                        'perturbed', label, subject,
                        f'center_of_mass_{label}.sto'))

    unp_ts_vel_z = unp_ts.getDependentColumn(
                        '/|com_velocity_z').to_numpy() 

    unp_ts_acc_x = unp_ts.getDependentColumn(
                        '/|com_acceleration_x').to_numpy() 
    unp_ts_acc_y = unp_ts.getDependentColumn(
                        '/|com_acceleration_y').to_numpy() 
    unp_ts_acc_z = unp_ts.getDependentColumn(
                        '/|com_acceleration_z').to_numpy() 

    unp_ts_grfs = osim.TimeSeriesTable(
                        os.path.join(
                            config['results_path'], 
                            'perturbed', label, subject,
                            f'{label}_grfs.sto'))

    unp_ts_grf_x = unp_ts_grfs.getDependentColumn('ground_force_r_vx').to_numpy() + unp_ts_grfs.getDependentColumn('ground_force_l_vx').to_numpy()
    unp_ts_grf_y = unp_ts_grfs.getDependentColumn('ground_force_r_vy').to_numpy() + unp_ts_grfs.getDependentColumn('ground_force_l_vy').to_numpy()
    unp_ts_grf_z = unp_ts_grfs.getDependentColumn('ground_force_r_vz').to_numpy() + unp_ts_grfs.getDependentColumn('ground_force_l_vz').to_numpy()


    for iper, (torque, subtalar) in enumerate(zip(torques, subtalars)):
        label = f'perturbed_torque{torque}_time{time}_rise10_fall5{subtalar}'
        per_ts = osim.TimeSeriesTable(
                        os.path.join(
                            config['results_path'], 
                            'perturbed', label, subject,
                            f'center_of_mass_{label}.sto'))

        per_ts_vel_z = per_ts.getDependentColumn(
                    '/|com_velocity_z').to_numpy() 

        values[iper, isubj] = (per_ts_vel_z[-1] - unp_ts_vel_z[-1]) / vmax

        per_ts_acc_x = per_ts.getDependentColumn('/|com_acceleration_x').to_numpy() 
        per_ts_acc_y = per_ts.getDependentColumn('/|com_acceleration_y').to_numpy() 
        per_ts_acc_z = per_ts.getDependentColumn('/|com_acceleration_z').to_numpy()

        per_ts_grfs = osim.TimeSeriesTable(
                        os.path.join(
                            config['results_path'], 
                            'perturbed', label, subject,
                            f'{label}_grfs.sto'))

        per_ts_grf_x = per_ts_grfs.getDependentColumn('ground_force_r_vx').to_numpy() + per_ts_grfs.getDependentColumn('ground_force_l_vx').to_numpy()
        per_ts_grf_y = per_ts_grfs.getDependentColumn('ground_force_r_vy').to_numpy() + per_ts_grfs.getDependentColumn('ground_force_l_vy').to_numpy()
        per_ts_grf_z = per_ts_grfs.getDependentColumn('ground_force_r_vz').to_numpy() + per_ts_grfs.getDependentColumn('ground_force_l_vz').to_numpy()

        delta_acc_x = per_ts_acc_x[-10] - unp_ts_acc_x[-10]
        delta_acc_y = per_ts_acc_y[-10] - unp_ts_acc_y[-10]
        delta_acc_z = per_ts_acc_z[-10] - unp_ts_acc_z[-10]

        delta_grf_x = per_ts_grf_x[-10] - unp_ts_grf_x[-10]
        delta_grf_y = per_ts_grf_y[-10] - unp_ts_grf_y[-10]
        delta_grf_z = per_ts_grf_z[-10] - unp_ts_grf_z[-10]

        print(f'diff x: {delta_acc_x - delta_grf_x / mass}')
        print(f'diff y: {delta_acc_y - delta_grf_y / mass}')
        print(f'diff z: {delta_acc_z - delta_grf_z / mass}')



