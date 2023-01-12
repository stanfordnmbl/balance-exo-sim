# Data pre-processing
# ---------------
# doit ankle_perturb_sim_copy_data
# doit ankle_perturb_sim_copy_generic_model_files
# doit ankle_perturb_sim_apply_markerset_to_generic_model       
# doit subject*_walk2_update_ground_reaction_labels
# doit subject*_walk2_filter_ground_reactions                        
# doit subject*_walk2_gait_landmarks 

# Model scaling
# -------------
# doit subject*_scale_setup
# doit subject*_scale
# doit subject*_scale_max_force

# Inverse kinematics and inverse dynamics
# ---------------------------------------   
# doit subject*_walk2_ik_setup           
# doit subject*_walk2_ik                                  
# doit subject*_walk2_ik_post
# doit subject*_walk2_id_setup    
# doit subject*_walk2_id                                 
# doit subject*_walk2_joint_angle_standard_deviations
# doit subject*_walk2_trim_tracking_data

# Generate results
# ----------------
# doit subject01_moco_unperturbed_guess_mesh0.04_scale0.1_reserve200
# doit subject01_moco_unperturbed_guess_mesh0.03_scale0.5_reserve20_periodic
# doit subject01_moco_unperturbed_guess_mesh0.02_scale1.0_reserve0_periodic
# doit subject01_moco_unperturbed
# doit subject01_moco_perturbed*

# doit subject02_moco_unperturbed_guess_mesh0.04_scale0.1_reserve200
# doit subject02_moco_unperturbed_guess_mesh0.03_scale0.5_reserve20_periodic
# doit subject02_moco_unperturbed_guess_mesh0.02_scale1.0_reserve0_periodic
# doit subject02_moco_unperturbed
# doit subject02_moco_perturbed*

# doit subject04_moco_unperturbed_guess_mesh0.04_scale0.1_reserve200
# doit subject04_moco_unperturbed_guess_mesh0.03_scale0.5_reserve20_periodic
# doit subject04_moco_unperturbed_guess_mesh0.02_scale1.0_reserve0_periodic
# doit subject04_moco_unperturbed
# doit subject04_moco_perturbed*

# doit subject18_moco_unperturbed_guess_mesh0.04_scale0.1_reserve200
# doit subject18_moco_unperturbed_guess_mesh0.03_scale0.5_reserve20_periodic
# doit subject18_moco_unperturbed_guess_mesh0.02_scale1.0_reserve0_periodic
# doit subject18_moco_unperturbed
# doit subject18_moco_perturbed*

# doit subject19_moco_unperturbed_guess_mesh0.04_scale0.1_reserve200
# doit subject19_moco_unperturbed_guess_mesh0.03_scale0.5_reserve20_periodic
# doit subject19_moco_unperturbed_guess_mesh0.02_scale1.0_reserve0_periodic
# doit subject19_moco_unperturbed
# doit subject19_moco_perturbed*

# Validate
# --------
# doit validate_marker_errors
# doit validate_tracking_errors
# doit validate_muscle_activity
# doit validate_accelerations_versus_grfs
# doit validate_accelerations_versus_velocities
# doit compute_center_of_mass_timestepping_error 

# Statistics
# ----------
# doit create_center_of_mass_statistics_tables
# doit create_center_of_pressure_statistics_tables
# doit run_statistics
# doit aggregate_com_statistics_results
# doit aggregate_cop_statistics_results

# Plot results
# ------------
# doit plot_unperturbed_results
# doit plot_center_of_mass_vector
# doit plot_instantaneous_center_of_mass
# doit plot_com_versus_cop
# doit plot_center_of_pressure_vector
# doit plot_instantaneous_center_of_pressure
# doit create_perturbation_powers_table*
# doit plot_perturbation_powers
