# Data pre-processing
# ---------------
# doit forget ankle_perturb_sim_copy_data
# doit forget ankle_perturb_sim_copy_generic_model_files     

# Model scaling
# -------------
# doit forget subject01_scale_setup
# doit forget subject01_scale
# doit forget subject01_scale_max_force   

# doit forget subject02_scale_setup
# doit forget subject02_scale
# doit forget subject02_scale_max_force    

# doit forget subject04_scale_setup
# doit forget subject04_scale
# doit forget subject04_scale_max_force       

# doit forget subject18_scale_setup
# doit forget subject18_scale
# doit forget subject18_scale_max_force        

# doit forget subject19_scale_setup
# doit forget subject19_scale
# doit forget subject19_scale_max_force   

# Inverse kinematics and inverse dynamics
# --------------------------------------- 
doit forget subject01_adjust_scaled_model
doit forget subject01_walk2_update_ground_reaction_labels
doit forget subject01_walk2_filter_ground_reactions      
doit forget subject01_walk2_gait_landmarks 
doit forget subject01_walk2_ik_setup           
doit forget subject01_walk2_ik                                  
doit forget subject01_walk2_ik_post
doit forget subject01_walk2_id_setup    
doit forget subject01_walk2_id                                 
doit forget subject01_walk2_joint_angle_standard_deviations
doit forget subject01_walk2_trim_tracking_data

doit forget subject02_adjust_scaled_model
doit forget subject02_walk2_update_ground_reaction_labels
doit forget subject02_walk2_filter_ground_reactions      
doit forget subject02_walk2_gait_landmarks 
doit forget subject02_walk2_ik_setup           
doit forget subject02_walk2_ik                                  
doit forget subject02_walk2_ik_post
doit forget subject02_walk2_id_setup    
doit forget subject02_walk2_id                                 
doit forget subject02_walk2_joint_angle_standard_deviations
doit forget subject02_walk2_trim_tracking_data

doit forget subject04_adjust_scaled_model
doit forget subject04_walk2_update_ground_reaction_labels
doit forget subject04_walk2_filter_ground_reactions      
doit forget subject04_walk2_gait_landmarks 
doit forget subject04_walk2_ik_setup           
doit forget subject04_walk2_ik                                  
doit forget subject04_walk2_ik_post
doit forget subject04_walk2_id_setup    
doit forget subject04_walk2_id                                 
doit forget subject04_walk2_joint_angle_standard_deviations
doit forget subject04_walk2_trim_tracking_data

doit forget subject18_adjust_scaled_model
doit forget subject18_walk2_update_ground_reaction_labels
doit forget subject18_walk2_filter_ground_reactions      
doit forget subject18_walk2_gait_landmarks 
doit forget subject18_walk2_ik_setup           
doit forget subject18_walk2_ik                                  
doit forget subject18_walk2_ik_post
doit forget subject18_walk2_id_setup    
doit forget subject18_walk2_id                                 
doit forget subject18_walk2_joint_angle_standard_deviations
doit forget subject18_walk2_trim_tracking_data

doit forget subject19_adjust_scaled_model
doit forget subject19_walk2_update_ground_reaction_labels
doit forget subject19_walk2_filter_ground_reactions      
doit forget subject19_walk2_gait_landmarks 
doit forget subject19_walk2_ik_setup           
doit forget subject19_walk2_ik                                  
doit forget subject19_walk2_ik_post
doit forget subject19_walk2_id_setup    
doit forget subject19_walk2_id                                 
doit forget subject19_walk2_joint_angle_standard_deviations
doit forget subject19_walk2_trim_tracking_data

# Sensitivity analysis
# --------------------
# doit forget subject01_moco_unperturbed_sensitivity_mesh50_tol1000_mesh
# doit forget subject01_moco_unperturbed_sensitivity_mesh40_tol1000_mesh
# doit forget subject01_moco_unperturbed_sensitivity_mesh30_tol1000_mesh
# doit forget subject01_moco_unperturbed_sensitivity_mesh20_tol1000_mesh
# doit forget subject01_moco_unperturbed_sensitivity_mesh10_tol1000_mesh
# doit forget subject01_moco_unperturbed_sensitivity_mesh10_tol100000_tol
# doit forget subject01_moco_unperturbed_sensitivity_mesh10_tol10000_tol
# doit forget subject01_moco_unperturbed_sensitivity_mesh10_tol1000_tol
# doit forget subject01_moco_unperturbed_sensitivity_mesh10_tol100_tol
# doit forget subject01_moco_unperturbed_sensitivity_mesh10_tol10_tol

# doit forget subject02_moco_unperturbed_sensitivity_mesh50_tol1000_mesh
# doit forget subject02_moco_unperturbed_sensitivity_mesh40_tol1000_mesh
# doit forget subject02_moco_unperturbed_sensitivity_mesh30_tol1000_mesh
# doit forget subject02_moco_unperturbed_sensitivity_mesh20_tol1000_mesh
# doit forget subject02_moco_unperturbed_sensitivity_mesh10_tol1000_mesh
# doit forget subject02_moco_unperturbed_sensitivity_mesh10_tol100000_tol
# doit forget subject02_moco_unperturbed_sensitivity_mesh10_tol10000_tol
# doit forget subject02_moco_unperturbed_sensitivity_mesh10_tol1000_tol
# doit forget subject02_moco_unperturbed_sensitivity_mesh10_tol100_tol
# doit forget subject02_moco_unperturbed_sensitivity_mesh10_tol10_tol

# doit forget subject04_moco_unperturbed_sensitivity_mesh50_tol1000_mesh
# doit forget subject04_moco_unperturbed_sensitivity_mesh40_tol1000_mesh
# doit forget subject04_moco_unperturbed_sensitivity_mesh30_tol1000_mesh
# doit forget subject04_moco_unperturbed_sensitivity_mesh20_tol1000_mesh
# doit forget subject04_moco_unperturbed_sensitivity_mesh10_tol1000_mesh
# doit forget subject04_moco_unperturbed_sensitivity_mesh10_tol100000_tol
# doit forget subject04_moco_unperturbed_sensitivity_mesh10_tol10000_tol
# doit forget subject04_moco_unperturbed_sensitivity_mesh10_tol1000_tol
# doit forget subject04_moco_unperturbed_sensitivity_mesh10_tol100_tol
# doit forget subject04_moco_unperturbed_sensitivity_mesh10_tol10_tol

# doit forget subject18_moco_unperturbed_sensitivity_mesh50_tol1000_mesh
# doit forget subject18_moco_unperturbed_sensitivity_mesh40_tol1000_mesh
# doit forget subject18_moco_unperturbed_sensitivity_mesh30_tol1000_mesh
# doit forget subject18_moco_unperturbed_sensitivity_mesh20_tol1000_mesh
# doit forget subject18_moco_unperturbed_sensitivity_mesh10_tol1000_mesh
# doit forget subject18_moco_unperturbed_sensitivity_mesh10_tol100000_tol
# doit forget subject18_moco_unperturbed_sensitivity_mesh10_tol10000_tol
# doit forget subject18_moco_unperturbed_sensitivity_mesh10_tol1000_tol
# doit forget subject18_moco_unperturbed_sensitivity_mesh10_tol100_tol
# doit forget subject18_moco_unperturbed_sensitivity_mesh10_tol10_tol

# doit forget subject19_moco_unperturbed_sensitivity_mesh50_tol1000_mesh
# doit forget subject19_moco_unperturbed_sensitivity_mesh40_tol1000_mesh
# doit forget subject19_moco_unperturbed_sensitivity_mesh30_tol1000_mesh
# doit forget subject19_moco_unperturbed_sensitivity_mesh20_tol1000_mesh
# doit forget subject19_moco_unperturbed_sensitivity_mesh10_tol1000_mesh
# doit forget subject19_moco_unperturbed_sensitivity_mesh10_tol100000_tol
# doit forget subject19_moco_unperturbed_sensitivity_mesh10_tol10000_tol
# doit forget subject19_moco_unperturbed_sensitivity_mesh10_tol1000_tol
# doit forget subject19_moco_unperturbed_sensitivity_mesh10_tol100_tol
# doit forget subject19_moco_unperturbed_sensitivity_mesh10_tol10_tol

# Unperturbed walking
# -------------------
# doit forget subject01_moco_unperturbed_guess_mesh35_scale1000000000
# doit forget subject01_moco_unperturbed_guess_mesh35_scale10000000
# doit forget subject01_moco_unperturbed_guess_mesh35_scale100000
# doit forget subject01_moco_unperturbed_guess_mesh35_scale1000
# doit forget subject01_moco_unperturbed_guess_mesh35_scale1
# doit forget subject01_moco_unperturbed_guess_mesh35_scale1_periodic
# doit forget subject01_moco_unperturbed_guess_mesh20_scale1_periodic
# doit forget subject01_moco_unperturbed

# doit forget subject02_moco_unperturbed_guess_mesh35_scale1000000000
# doit forget subject02_moco_unperturbed_guess_mesh35_scale10000000
# doit forget subject02_moco_unperturbed_guess_mesh35_scale100000
# doit forget subject02_moco_unperturbed_guess_mesh35_scale1000
# doit forget subject02_moco_unperturbed_guess_mesh35_scale1
# doit forget subject02_moco_unperturbed_guess_mesh35_scale1_periodic
# doit forget subject02_moco_unperturbed_guess_mesh20_scale1_periodic
# doit forget subject02_moco_unperturbed

# doit forget subject04_moco_unperturbed_guess_mesh35_scale100000_reserve1000
# doit forget subject04_moco_unperturbed_guess_mesh35_scale1000_reserve1000
# doit forget subject04_moco_unperturbed_guess_mesh35_scale100_reserve100
# doit forget subject04_moco_unperturbed_guess_mesh35_scale1_reserve100
# doit forget subject04_moco_unperturbed_guess_mesh35_scale1_reserve10
# doit forget subject04_moco_unperturbed_guess_mesh35_scale1
# doit forget subject04_moco_unperturbed_guess_mesh35_scale1_periodic
# doit forget subject04_moco_unperturbed_guess_mesh20_scale1_periodic
# doit forget subject04_moco_unperturbed

# doit forget subject18_moco_unperturbed_guess_mesh35_scale1000000000
# doit forget subject18_moco_unperturbed_guess_mesh35_scale10000000
# doit forget subject18_moco_unperturbed_guess_mesh35_scale100000
# doit forget subject18_moco_unperturbed_guess_mesh35_scale1000
# doit forget subject18_moco_unperturbed_guess_mesh35_scale1
# doit forget subject18_moco_unperturbed_guess_mesh35_scale1_periodic
# doit forget subject18_moco_unperturbed_guess_mesh20_scale1_periodic
# doit forget subject18_moco_unperturbed

# doit forget subject19_moco_unperturbed_guess_mesh35_scale1000000000
# doit forget subject19_moco_unperturbed_guess_mesh35_scale10000000
# doit forget subject19_moco_unperturbed_guess_mesh35_scale100000
# doit forget subject19_moco_unperturbed_guess_mesh35_scale1000
# doit forget subject19_moco_unperturbed_guess_mesh35_scale1
# doit forget subject19_moco_unperturbed_guess_mesh35_scale1_periodic
# doit forget subject19_moco_unperturbed_guess_mesh20_scale1_periodic
# doit forget subject19_moco_unperturbed

doit forget subject01_moco_unperturbed
doit forget subject02_moco_unperturbed
doit forget subject04_moco_unperturbed
doit forget subject18_moco_unperturbed
doit forget subject19_moco_unperturbed
doit forget plot_unperturbed_results

# Perturbed walking
# -------------------
# doit forget subject*_moco_perturbed_torque*_time40_rise10_fall5_post
# doit forget subject*_moco_perturbed_torque*_time42_rise10_fall5
# doit forget subject*_moco_perturbed_torque*_time42_rise10_fall5_post
# doit forget subject*_moco_perturbed_torque*_time44_rise10_fall5
# doit forget subject*_moco_perturbed_torque*_time44_rise10_fall5_post
# doit forget subject*_moco_perturbed_torque*_time46_rise10_fall5
# doit forget subject*_moco_perturbed_torque*_time46_rise10_fall5_post
# doit forget subject*_moco_perturbed_torque*_time48_rise10_fall5
# doit forget subject*_moco_perturbed_torque*_time48_rise10_fall5_post
# doit forget subject*_moco_perturbed_torque*_time50_rise10_fall5
# doit forget subject*_moco_perturbed_torque*_time50_rise10_fall5_post
# doit forget subject*_moco_perturbed_torque*_time52_rise10_fall5
# doit forget subject*_moco_perturbed_torque*_time52_rise10_fall5_post
# doit forget subject*_moco_perturbed_torque*_time54_rise10_fall5
# doit forget subject*_moco_perturbed_torque*_time54_rise10_fall5_post
# doit forget subject*_moco_perturbed_torque*_time56_rise10_fall5
# doit forget subject*_moco_perturbed_torque*_time56_rise10_fall5_post
# doit forget subject*_moco_perturbed_torque*_time58_rise10_fall5
# doit forget subject*_moco_perturbed_torque*_time58_rise10_fall5_post
# doit forget subject*_moco_perturbed_torque*_time60_rise10_fall5
# doit forget subject*_moco_perturbed_torque*_time60_rise10_fall5_post

# doit forget subject*_moco_perturbed_torque*_time40_rise25_fall10
# doit forget subject*_moco_perturbed_torque*_time40_rise25_fall10_post
# doit forget subject*_moco_perturbed_torque*_time42_rise25_fall10
# doit forget subject*_moco_perturbed_torque*_time42_rise25_fall10_post
# doit forget subject*_moco_perturbed_torque*_time44_rise25_fall10
# doit forget subject*_moco_perturbed_torque*_time44_rise25_fall10_post
# doit forget subject*_moco_perturbed_torque*_time46_rise25_fall10
# doit forget subject*_moco_perturbed_torque*_time46_rise25_fall10_post
# doit forget subject*_moco_perturbed_torque*_time48_rise25_fall10
# doit forget subject*_moco_perturbed_torque*_time48_rise25_fall10_post
# doit forget subject*_moco_perturbed_torque*_time50_rise25_fall10
# doit forget subject*_moco_perturbed_torque*_time50_rise25_fall10_post
# doit forget subject*_moco_perturbed_torque*_time52_rise25_fall10
# doit forget subject*_moco_perturbed_torque*_time52_rise25_fall10_post
# doit forget subject*_moco_perturbed_torque*_time54_rise25_fall10
# doit forget subject*_moco_perturbed_torque*_time54_rise25_fall10_post
# doit forget subject*_moco_perturbed_torque*_time56_rise25_fall10
# doit forget subject*_moco_perturbed_torque*_time56_rise25_fall10_post
# doit forget subject*_moco_perturbed_torque*_time58_rise25_fall10
# doit forget subject*_moco_perturbed_torque*_time58_rise25_fall10_post
# doit forget subject*_moco_perturbed_torque*_time60_rise25_fall10
# doit forget subject*_moco_perturbed_torque*_time60_rise25_fall10_post


# 50% peak time
# doit forget subject01_moco_perturbed_torque25_time50
# doit forget subject01_moco_perturbed_torque25_time50_post
# doit forget subject02_moco_perturbed_torque25_time50
# doit forget subject02_moco_perturbed_torque25_time50_post
# doit forget subject04_moco_perturbed_torque25_time50
# doit forget subject04_moco_perturbed_torque25_time50_post
# doit forget subject18_moco_perturbed_torque25_time50
# doit forget subject18_moco_perturbed_torque25_time50_post
# doit forget subject19_moco_perturbed_torque25_time50
# doit forget subject19_moco_perturbed_torque25_time50_post

# doit forget subject01_moco_perturbed_torque50_time50
# doit forget subject01_moco_perturbed_torque50_time50_post
# doit forget subject02_moco_perturbed_torque50_time50
# doit forget subject02_moco_perturbed_torque50_time50_post
# doit forget subject04_moco_perturbed_torque50_time50
# doit forget subject04_moco_perturbed_torque50_time50_post
# doit forget subject18_moco_perturbed_torque50_time50
# doit forget subject18_moco_perturbed_torque50_time50_post
# doit forget subject19_moco_perturbed_torque50_time50
# doit forget subject19_moco_perturbed_torque50_time50_post

# doit forget subject01_moco_perturbed_torque75_time50
# doit forget subject01_moco_perturbed_torque75_time50_post
# doit forget subject02_moco_perturbed_torque75_time50
# doit forget subject02_moco_perturbed_torque75_time50_post
# doit forget subject04_moco_perturbed_torque75_time50
# doit forget subject04_moco_perturbed_torque75_time50_post
# doit forget subject18_moco_perturbed_torque75_time50
# doit forget subject18_moco_perturbed_torque75_time50_post
# doit forget subject19_moco_perturbed_torque75_time50
# doit forget subject19_moco_perturbed_torque75_time50_post

# doit forget subject01_moco_perturbed_torque100_time50
# doit forget subject01_moco_perturbed_torque100_time50_post
# doit forget subject02_moco_perturbed_torque100_time50
# doit forget subject02_moco_perturbed_torque100_time50_post
# doit forget subject04_moco_perturbed_torque100_time50
# doit forget subject04_moco_perturbed_torque100_time50_post
# doit forget subject18_moco_perturbed_torque100_time50
# doit forget subject18_moco_perturbed_torque100_time50_post
# doit forget subject19_moco_perturbed_torque100_time50
# doit forget subject19_moco_perturbed_torque100_time50_post


# # 60% peak time
# doit forget subject01_moco_perturbed_torque25_time60
# doit forget subject01_moco_perturbed_torque25_time60_post
# doit forget subject02_moco_perturbed_torque25_time60
# doit forget subject02_moco_perturbed_torque25_time60_post
# doit forget subject04_moco_perturbed_torque25_time60
# doit forget subject04_moco_perturbed_torque25_time60_post
# doit forget subject18_moco_perturbed_torque25_time60
# doit forget subject18_moco_perturbed_torque25_time60_post
# doit forget subject19_moco_perturbed_torque25_time60
# doit forget subject19_moco_perturbed_torque25_time60_post

# doit forget subject01_moco_perturbed_torque50_time60
# doit forget subject01_moco_perturbed_torque50_time60_post
# doit forget subject02_moco_perturbed_torque50_time60
# doit forget subject02_moco_perturbed_torque50_time60_post
# doit forget subject04_moco_perturbed_torque50_time60
# doit forget subject04_moco_perturbed_torque50_time60_post
# doit forget subject18_moco_perturbed_torque50_time60
# doit forget subject18_moco_perturbed_torque50_time60_post
# doit forget subject19_moco_perturbed_torque50_time60
# doit forget subject19_moco_perturbed_torque50_time60_post

# doit forget subject01_moco_perturbed_torque75_time60
# doit forget subject01_moco_perturbed_torque75_time60_post
# doit forget subject02_moco_perturbed_torque75_time60
# doit forget subject02_moco_perturbed_torque75_time60_post
# doit forget subject04_moco_perturbed_torque75_time60
# doit forget subject04_moco_perturbed_torque75_time60_post
# doit forget subject18_moco_perturbed_torque75_time60
# doit forget subject18_moco_perturbed_torque75_time60_post
# doit forget subject19_moco_perturbed_torque75_time60
# doit forget subject19_moco_perturbed_torque75_time60_post

# doit forget subject01_moco_perturbed_torque100_time60
# doit forget subject01_moco_perturbed_torque100_time60_post
# doit forget subject02_moco_perturbed_torque100_time60
# doit forget subject02_moco_perturbed_torque100_time60_post
# doit forget subject04_moco_perturbed_torque100_time60
# doit forget subject04_moco_perturbed_torque100_time60_post
# doit forget subject18_moco_perturbed_torque100_time60
# doit forget subject18_moco_perturbed_torque100_time60_post
# doit forget subject19_moco_perturbed_torque100_time60
# doit forget subject19_moco_perturbed_torque100_time60_post

# doit forget plot_ankle_torques_time50
# doit forget plot_ankle_torques_time60
# doit forget plot_center_of_mass_time50
# doit forget plot_center_of_mass_time60
# doit forget plot_ground_reactions_time50
# doit forget plot_ground_reactions_time60