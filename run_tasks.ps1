# Data pre-processing
# ---------------
# doit ankle_perturb_sim_copy_data
# doit ankle_perturb_sim_copy_generic_model_files                              

# Model scaling
# -------------
# doit subject*_scale_setup
# doit subject*_scale
# doit subject*_scale_max_force

# Inverse kinematics and inverse dynamics
# --------------------------------------- 
# doit subject*_adjust_scaled_model
# doit subject*_walk2_update_ground_reaction_labels
# doit subject*_walk2_filter_ground_reactions      
# doit subject*_walk2_gait_landmarks 
# doit subject*_walk2_ik_setup           
# doit subject*_walk2_ik                                  
# doit subject*_walk2_ik_post
# doit subject*_walk2_id_setup    
# doit subject*_walk2_id                                 
# doit subject*_walk2_joint_angle_standard_deviations
# doit subject*_walk2_trim_tracking_data

# Sensitivity analysis
# --------------------
# doit subject01_moco_unperturbed_sensitivity_mesh50_tol1000_mesh
# doit subject01_moco_unperturbed_sensitivity_mesh40_tol1000_mesh
# doit subject01_moco_unperturbed_sensitivity_mesh30_tol1000_mesh
# doit subject01_moco_unperturbed_sensitivity_mesh20_tol1000_mesh
# doit subject01_moco_unperturbed_sensitivity_mesh10_tol1000_mesh
# doit subject01_moco_unperturbed_sensitivity_mesh10_tol100000_tol
# doit subject01_moco_unperturbed_sensitivity_mesh10_tol10000_tol
# doit subject01_moco_unperturbed_sensitivity_mesh10_tol1000_tol
# doit subject01_moco_unperturbed_sensitivity_mesh10_tol100_tol
# doit subject01_moco_unperturbed_sensitivity_mesh10_tol10_tol

# doit subject02_moco_unperturbed_sensitivity_mesh50_tol1000_mesh
# doit subject02_moco_unperturbed_sensitivity_mesh40_tol1000_mesh
# doit subject02_moco_unperturbed_sensitivity_mesh30_tol1000_mesh
# doit subject02_moco_unperturbed_sensitivity_mesh20_tol1000_mesh
# doit subject02_moco_unperturbed_sensitivity_mesh10_tol1000_mesh
# doit subject02_moco_unperturbed_sensitivity_mesh10_tol100000_tol
# doit subject02_moco_unperturbed_sensitivity_mesh10_tol10000_tol
# doit subject02_moco_unperturbed_sensitivity_mesh10_tol1000_tol
# doit subject02_moco_unperturbed_sensitivity_mesh10_tol100_tol
# doit subject02_moco_unperturbed_sensitivity_mesh10_tol10_tol

# doit subject04_moco_unperturbed_sensitivity_mesh50_tol1000_mesh
# doit subject04_moco_unperturbed_sensitivity_mesh40_tol1000_mesh
# doit subject04_moco_unperturbed_sensitivity_mesh30_tol1000_mesh
# doit subject04_moco_unperturbed_sensitivity_mesh20_tol1000_mesh
# doit subject04_moco_unperturbed_sensitivity_mesh10_tol1000_mesh
# doit subject04_moco_unperturbed_sensitivity_mesh10_tol100000_tol
# doit subject04_moco_unperturbed_sensitivity_mesh10_tol10000_tol
# doit subject04_moco_unperturbed_sensitivity_mesh10_tol1000_tol
# doit subject04_moco_unperturbed_sensitivity_mesh10_tol100_tol
# doit subject04_moco_unperturbed_sensitivity_mesh10_tol10_tol

# doit subject18_moco_unperturbed_sensitivity_mesh50_tol1000_mesh
# doit subject18_moco_unperturbed_sensitivity_mesh40_tol1000_mesh
# doit subject18_moco_unperturbed_sensitivity_mesh30_tol1000_mesh
# doit subject18_moco_unperturbed_sensitivity_mesh20_tol1000_mesh
# doit subject18_moco_unperturbed_sensitivity_mesh10_tol1000_mesh
# doit subject18_moco_unperturbed_sensitivity_mesh10_tol100000_tol
# doit subject18_moco_unperturbed_sensitivity_mesh10_tol10000_tol
# doit subject18_moco_unperturbed_sensitivity_mesh10_tol1000_tol
# doit subject18_moco_unperturbed_sensitivity_mesh10_tol100_tol
# doit subject18_moco_unperturbed_sensitivity_mesh10_tol10_tol

# doit subject19_moco_unperturbed_sensitivity_mesh50_tol1000_mesh
# doit subject19_moco_unperturbed_sensitivity_mesh40_tol1000_mesh
# doit subject19_moco_unperturbed_sensitivity_mesh30_tol1000_mesh
# doit subject19_moco_unperturbed_sensitivity_mesh20_tol1000_mesh
# doit subject19_moco_unperturbed_sensitivity_mesh10_tol1000_mesh
# doit subject19_moco_unperturbed_sensitivity_mesh10_tol100000_tol
# doit subject19_moco_unperturbed_sensitivity_mesh10_tol10000_tol
# doit subject19_moco_unperturbed_sensitivity_mesh10_tol1000_tol
# doit subject19_moco_unperturbed_sensitivity_mesh10_tol100_tol
# doit subject19_moco_unperturbed_sensitivity_mesh10_tol10_tol

# Unperturbed walking
# -------------------
# doit subject01_moco_unperturbed_guess_mesh35_scale1000000000
# doit subject01_moco_unperturbed_guess_mesh35_scale10000000
# doit subject01_moco_unperturbed_guess_mesh35_scale100000
# doit subject01_moco_unperturbed_guess_mesh35_scale1000
# doit subject01_moco_unperturbed_guess_mesh35_scale1
# doit subject01_moco_unperturbed_guess_mesh35_scale1_periodic
# doit subject01_moco_unperturbed_guess_mesh20_scale1_periodic
# doit subject01_moco_unperturbed

# doit subject02_moco_unperturbed_guess_mesh35_scale1000000000
# doit subject02_moco_unperturbed_guess_mesh35_scale10000000
# doit subject02_moco_unperturbed_guess_mesh35_scale100000
# doit subject02_moco_unperturbed_guess_mesh35_scale1000
# doit subject02_moco_unperturbed_guess_mesh35_scale1
# doit subject02_moco_unperturbed_guess_mesh35_scale1_periodic
# doit subject02_moco_unperturbed_guess_mesh20_scale1_periodic
# doit subject02_moco_unperturbed

# doit subject04_moco_unperturbed_guess_mesh35_scale100000_reserve1000
# doit subject04_moco_unperturbed_guess_mesh35_scale1000_reserve1000
# doit subject04_moco_unperturbed_guess_mesh35_scale100_reserve100
# doit subject04_moco_unperturbed_guess_mesh35_scale1_reserve100
# doit subject04_moco_unperturbed_guess_mesh35_scale1_reserve10
# doit subject04_moco_unperturbed_guess_mesh35_scale1
# doit subject04_moco_unperturbed_guess_mesh35_scale1_periodic
# doit subject04_moco_unperturbed_guess_mesh20_scale1_periodic
# doit subject04_moco_unperturbed

# doit subject18_moco_unperturbed_guess_mesh35_scale1000000000
# doit subject18_moco_unperturbed_guess_mesh35_scale10000000
# doit subject18_moco_unperturbed_guess_mesh35_scale100000
# doit subject18_moco_unperturbed_guess_mesh35_scale1000
# doit subject18_moco_unperturbed_guess_mesh35_scale1
# doit subject18_moco_unperturbed_guess_mesh35_scale1_periodic
# doit subject18_moco_unperturbed_guess_mesh20_scale1_periodic
# doit subject18_moco_unperturbed

# doit subject19_moco_unperturbed_guess_mesh35_scale1000000000
# doit subject19_moco_unperturbed_guess_mesh35_scale10000000
# doit subject19_moco_unperturbed_guess_mesh35_scale100000
# doit subject19_moco_unperturbed_guess_mesh35_scale1000
# doit subject19_moco_unperturbed_guess_mesh35_scale1
# doit subject19_moco_unperturbed_guess_mesh35_scale1_periodic
# doit subject19_moco_unperturbed_guess_mesh20_scale1_periodic
# doit subject19_moco_unperturbed

# doit subject01_moco_unperturbed
# doit subject01_moco_unperturbed_lumbar_welded
# doit subject02_moco_unperturbed
# doit subject02_moco_unperturbed_lumbar_welded
# doit subject04_moco_unperturbed
# doit subject04_moco_unperturbed_lumbar_welded
# doit subject18_moco_unperturbed
# doit subject18_moco_unperturbed_lumbar_welded
# doit subject19_moco_unperturbed
# doit subject19_moco_unperturbed_lumbar_welded
# doit plot_unperturbed_results
# doit plot_unperturbed_lumbar_welded_results

# Perturbed walking
# -----------------
doit subject*_moco_perturbed_torque*_time20_rise*_fall*
doit subject*_moco_perturbed_torque*_time25_rise*_fall*
doit subject*_moco_perturbed_torque*_time30_rise*_fall*
doit subject*_moco_perturbed_torque*_time35_rise*_fall*
doit subject*_moco_perturbed_torque*_time40_rise*_fall*
doit subject*_moco_perturbed_torque*_time45_rise*_fall*
doit subject*_moco_perturbed_torque*_time50_rise*_fall*
doit subject*_moco_perturbed_torque*_time55_rise*_fall*
doit subject*_moco_perturbed_torque*_time60_rise*_fall*
doit subject*_moco_perturbed_torque*_time65_rise*_fall*

