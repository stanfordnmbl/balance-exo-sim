# Data pre-processing
# ---------------
# doit ankle_perturb_sim_copy_data
# doit ankle_perturb_sim_copy_generic_model_files                              

# Model scaling and adjustments
# -----------------------------
# doit subject*_scale_setup
# doit subject*_scale
# doit subject*_scale_max_force
# doit subject*_adjust_scaled_model

# Inverse kinematics and inverse dynamics
# --------------------------------------- 
# doit subject04_walk2_update_ground_reaction_labels
# doit subject04_walk2_filter_ground_reactions      
# doit subject04_walk2_gait_landmarks 
# doit subject04_walk2_ik_setup           
# doit subject04_walk2_ik                                  
# doit subject04_walk2_ik_post
# doit subject04_walk2_id_setup    
# doit subject04_walk2_id                                 
# doit subject04_walk2_joint_angle_standard_deviations
# doit subject04_walk2_trim_tracking_data

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

doit subject01_moco_unperturbed
doit subject02_moco_unperturbed
doit subject04_moco_unperturbed
doit subject18_moco_unperturbed
doit subject19_moco_unperturbed

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

# Perturbed walking
# -------------------
# 50% peak time
# doit subject*_moco_perturbed_torque25_time50
# doit subject*_moco_perturbed_torque25_time50_post
# doit subject*_moco_perturbed_torque50_time50
# doit subject*_moco_perturbed_torque50_time50_post
# doit subject*_moco_perturbed_torque75_time50
# doit subject*_moco_perturbed_torque75_time50_post
# doit subject*_moco_perturbed_torque100_time50
# doit subject*_moco_perturbed_torque100_time50_post
# doit plot_grfs_ankle_perturb_time50_subject01
# doit plot_com_versus_ankle_perturb_torque_time50
# doit plot_ankle_perturb_torques_time50_subject01

# 55% peak time
# doit subject01_moco_perturb_torque25_time55_delay1500
# doit subject01_moco_perturb_torque25_time55_delay1500_post
# doit subject01_moco_perturb_torque50_time55_delay1500
# doit subject01_moco_perturb_torque50_time55_delay1500_post
# doit subject01_moco_perturb_torque75_time55_delay1500
# doit subject01_moco_perturb_torque75_time55_delay1500_post
# doit subject01_moco_perturb_torque100_time55_delay1500
# doit subject01_moco_perturb_torque100_time55_delay1500_post
# doit plot_grfs_ankle_perturb_time55_delay1500_subject01
# doit plot_com_versus_ankle_perturb_torque_time55_delay1500
# doit plot_ankle_perturb_torques_time55_subject01

# 60% peak time
# doit subject01_moco_perturb_torque25_time60_delay1500
# doit subject01_moco_perturb_torque25_time60_delay1500_post
# doit subject01_moco_perturb_torque50_time60_delay1500
# doit subject01_moco_perturb_torque50_time60_delay1500_post
# doit subject01_moco_perturb_torque75_time60_delay1500
# doit subject01_moco_perturb_torque75_time60_delay1500_post
# doit subject01_moco_perturb_torque100_time60_delay1500
# doit subject01_moco_perturb_torque100_time60_delay1500_post
# doit plot_grfs_ankle_perturb_time60_delay1500_subject01
# doit plot_com_versus_ankle_perturb_torque_time60_delay1500
# doit plot_ankle_perturb_torques_time60_subject01

