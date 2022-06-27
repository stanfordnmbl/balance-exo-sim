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
# doit forget subject01_adjust_scaled_model
# doit forget subject01_walk2_update_ground_reaction_labels
# doit forget subject01_walk2_filter_ground_reactions      
# doit forget subject01_walk2_gait_landmarks 
# doit forget subject01_walk2_ik_setup           
# doit forget subject01_walk2_ik                                  
# doit forget subject01_walk2_ik_post
# doit forget subject01_walk2_id_setup    
# doit forget subject01_walk2_id                                 
# doit forget subject01_walk2_joint_angle_standard_deviations
# doit forget subject01_walk2_trim_tracking_data

# doit forget subject02_adjust_scaled_model
# doit forget subject02_walk2_update_ground_reaction_labels
# doit forget subject02_walk2_filter_ground_reactions      
# doit forget subject02_walk2_gait_landmarks 
# doit forget subject02_walk2_ik_setup           
# doit forget subject02_walk2_ik                                  
# doit forget subject02_walk2_ik_post
# doit forget subject02_walk2_id_setup    
# doit forget subject02_walk2_id                                 
# doit forget subject02_walk2_joint_angle_standard_deviations
# doit forget subject02_walk2_trim_tracking_data

# doit forget subject04_adjust_scaled_model
# doit forget subject04_walk2_update_ground_reaction_labels
# doit forget subject04_walk2_filter_ground_reactions      
# doit forget subject04_walk2_gait_landmarks 
# doit forget subject04_walk2_ik_setup           
# doit forget subject04_walk2_ik                                  
# doit forget subject04_walk2_ik_post
# doit forget subject04_walk2_id_setup    
# doit forget subject04_walk2_id                                 
# doit forget subject04_walk2_joint_angle_standard_deviations
# doit forget subject04_walk2_trim_tracking_data

# doit forget subject18_adjust_scaled_model
# doit forget subject18_walk2_update_ground_reaction_labels
# doit forget subject18_walk2_filter_ground_reactions      
# doit forget subject18_walk2_gait_landmarks 
# doit forget subject18_walk2_ik_setup           
# doit forget subject18_walk2_ik                                  
# doit forget subject18_walk2_ik_post
# doit forget subject18_walk2_id_setup    
# doit forget subject18_walk2_id                                 
# doit forget subject18_walk2_joint_angle_standard_deviations
# doit forget subject18_walk2_trim_tracking_data

# doit forget subject19_adjust_scaled_model
# doit forget subject19_walk2_update_ground_reaction_labels
# doit forget subject19_walk2_filter_ground_reactions      
# doit forget subject19_walk2_gait_landmarks 
# doit forget subject19_walk2_ik_setup           
# doit forget subject19_walk2_ik                                  
# doit forget subject19_walk2_ik_post
# doit forget subject19_walk2_id_setup    
# doit forget subject19_walk2_id                                 
# doit forget subject19_walk2_joint_angle_standard_deviations
# doit forget subject19_walk2_trim_tracking_data

# Unperturbed walking
# -------------------
# doit forget subject01_moco_unperturbed
# doit forget subject01_moco_unperturbed_lumbar0.1
# doit forget subject01_moco_unperturbed_lumbar10.0
# doit forget subject02_moco_unperturbed
# doit forget subject02_moco_unperturbed_lumbar0.1
# doit forget subject02_moco_unperturbed_lumbar10.0
# doit forget subject04_moco_unperturbed
# doit forget subject04_moco_unperturbed_lumbar0.1
# doit forget subject04_moco_unperturbed_lumbar10.0
# doit forget subject18_moco_unperturbed
# doit forget subject18_moco_unperturbed_lumbar0.1
# doit forget subject18_moco_unperturbed_lumbar10.0
# doit forget subject19_moco_unperturbed
# doit forget subject19_moco_unperturbed_lumbar0.1
# doit forget subject19_moco_unperturbed_lumbar10.0
# doit forget plot_unperturbed_results

# Perturbed walking
# -----------------
doit forget subject18_moco_unperturbed
doit forget subject18_moco_perturbed_torque0_time20_rise10_fall5
doit forget subject18_moco_perturbed_torque0_time20_rise10_fall5_subtalar10
doit forget subject18_moco_perturbed_torque0_time20_rise10_fall5_subtalar-10
doit forget subject18_moco_perturbed_torque0_time25_rise10_fall5
doit forget subject18_moco_perturbed_torque0_time25_rise10_fall5_subtalar10
doit forget subject18_moco_perturbed_torque0_time25_rise10_fall5_subtalar-10
doit forget subject18_moco_perturbed_torque0_time30_rise10_fall5
doit forget subject18_moco_perturbed_torque0_time30_rise10_fall5_subtalar10
doit forget subject18_moco_perturbed_torque0_time30_rise10_fall5_subtalar-10
doit forget subject18_moco_perturbed_torque0_time35_rise10_fall5
doit forget subject18_moco_perturbed_torque0_time35_rise10_fall5_subtalar10
doit forget subject18_moco_perturbed_torque0_time35_rise10_fall5_subtalar-10
doit forget subject18_moco_perturbed_torque0_time40_rise10_fall5
doit forget subject18_moco_perturbed_torque0_time40_rise10_fall5_subtalar10
doit forget subject18_moco_perturbed_torque0_time40_rise10_fall5_subtalar-10
doit forget subject18_moco_perturbed_torque0_time45_rise10_fall5
doit forget subject18_moco_perturbed_torque0_time45_rise10_fall5_subtalar10
doit forget subject18_moco_perturbed_torque0_time45_rise10_fall5_subtalar-10
doit forget subject18_moco_perturbed_torque0_time50_rise10_fall5
doit forget subject18_moco_perturbed_torque0_time50_rise10_fall5_subtalar10
doit forget subject18_moco_perturbed_torque0_time50_rise10_fall5_subtalar-10
doit forget subject18_moco_perturbed_torque0_time55_rise10_fall5
doit forget subject18_moco_perturbed_torque0_time55_rise10_fall5_subtalar10
doit forget subject18_moco_perturbed_torque0_time55_rise10_fall5_subtalar-10
doit forget subject18_moco_perturbed_torque0_time60_rise10_fall5
doit forget subject18_moco_perturbed_torque0_time60_rise10_fall5_subtalar10
doit forget subject18_moco_perturbed_torque0_time60_rise10_fall5_subtalar-10
doit forget subject18_moco_perturbed_torque10_time20_rise10_fall5
doit forget subject18_moco_perturbed_torque10_time20_rise10_fall5_subtalar10
doit forget subject18_moco_perturbed_torque10_time20_rise10_fall5_subtalar-10
doit forget subject18_moco_perturbed_torque10_time25_rise10_fall5
doit forget subject18_moco_perturbed_torque10_time25_rise10_fall5_subtalar10
doit forget subject18_moco_perturbed_torque10_time25_rise10_fall5_subtalar-10
doit forget subject18_moco_perturbed_torque10_time30_rise10_fall5
doit forget subject18_moco_perturbed_torque10_time30_rise10_fall5_subtalar10
doit forget subject18_moco_perturbed_torque10_time30_rise10_fall5_subtalar-10
doit forget subject18_moco_perturbed_torque10_time35_rise10_fall5
doit forget subject18_moco_perturbed_torque10_time35_rise10_fall5_subtalar10
doit forget subject18_moco_perturbed_torque10_time35_rise10_fall5_subtalar-10
doit forget subject18_moco_perturbed_torque10_time40_rise10_fall5
doit forget subject18_moco_perturbed_torque10_time40_rise10_fall5_subtalar10
doit forget subject18_moco_perturbed_torque10_time40_rise10_fall5_subtalar-10
doit forget subject18_moco_perturbed_torque10_time45_rise10_fall5
doit forget subject18_moco_perturbed_torque10_time45_rise10_fall5_subtalar10
doit forget subject18_moco_perturbed_torque10_time45_rise10_fall5_subtalar-10
doit forget subject18_moco_perturbed_torque10_time50_rise10_fall5
doit forget subject18_moco_perturbed_torque10_time50_rise10_fall5_subtalar10
doit forget subject18_moco_perturbed_torque10_time50_rise10_fall5_subtalar-10
doit forget subject18_moco_perturbed_torque10_time55_rise10_fall5
doit forget subject18_moco_perturbed_torque10_time55_rise10_fall5_subtalar10
doit forget subject18_moco_perturbed_torque10_time55_rise10_fall5_subtalar-10
doit forget subject18_moco_perturbed_torque10_time60_rise10_fall5
doit forget subject18_moco_perturbed_torque10_time60_rise10_fall5_subtalar10
doit forget subject18_moco_perturbed_torque10_time60_rise10_fall5_subtalar-10