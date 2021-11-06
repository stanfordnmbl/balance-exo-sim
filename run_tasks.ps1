# Data processing
# ---------------
# doit ankle_perturb_sim_copy_data                                
# doit ankle_perturb_sim_transform_experimental_data                 
# doit ankle_perturb_sim_extract_and_filter_emg
# doit ankle_perturb_sim_extract_and_filter_perturbation_forces
# doit ankle_perturb_sim_filter_and_shift_grfs

# Model scaling and adjustments
# -----------------------------
# doit ankle_perturb_sim_copy_generic_model_files
# doit subject01_scale_setup
# doit subject01_scale
# doit subject01_scale_max_force
# doit subject01_adjust_scaled_model

# Inverse kinematics and inverse dynamics
# ---------------------------------------       
# doit subject01_unperturbed_gait_landmarks            
# doit subject01_unperturbed_ik                                  
# doit subject01_unperturbed_ik_post 
# doit subject01_unperturbed_ik_setup
# doit subject01_unperturbed_id                                 
# doit subject01_unperturbed_id_setup   
# doit subject01_unperturbed_joint_angle_standard_deviations
# doit subject01_unperturbed_trim_tracking_data

# Unperturbed walking
# -------------------
# doit subject01_moco_unperturbed_guess_mesh50_costsDisabled
# doit subject01_moco_unperturbed_guess_mesh50_costsDisabled_periodic
# doit subject01_moco_unperturbed_guess_mesh50_periodic
# doit subject01_moco_unperturbed_mesh50
# doit subject01_moco_unperturbed_mesh35
# doit subject01_moco_unperturbed_mesh20
# doit subject01_moco_unperturbed_mesh10

# Perturbed walking
# -------------------
# doit subject01_moco_perturb_torque25_time20_delay1000
# doit subject01_moco_perturb_torque25_time20_delay1000_post
# doit subject01_moco_perturb_torque50_time20_delay1000
# doit subject01_moco_perturb_torque50_time20_delay1000_post
# doit subject01_moco_perturb_torque75_time20_delay1000
# doit subject01_moco_perturb_torque75_time20_delay1000_post
# doit subject01_moco_perturb_torque100_time20_delay1000
# doit subject01_moco_perturb_torque100_time20_delay1000_post
# doit subject01_moco_perturb_torque25_time30_delay1000
# doit subject01_moco_perturb_torque25_time30_delay1000_post
# doit subject01_moco_perturb_torque50_time30_delay1000
# doit subject01_moco_perturb_torque50_time30_delay1000_post
# doit subject01_moco_perturb_torque75_time30_delay1000
# doit subject01_moco_perturb_torque75_time30_delay1000_post
# doit subject01_moco_perturb_torque100_time30_delay1000
# doit subject01_moco_perturb_torque100_time30_delay1000_post
# doit subject01_moco_perturb_torque25_time40_delay1500
# doit subject01_moco_perturb_torque25_time40_delay1500_post
# doit subject01_moco_perturb_torque50_time40_delay1500
# doit subject01_moco_perturb_torque50_time40_delay1500_post
# doit subject01_moco_perturb_torque75_time40_delay1500
# doit subject01_moco_perturb_torque75_time40_delay1500_post
# doit subject01_moco_perturb_torque100_time40_delay1500
# doit subject01_moco_perturb_torque100_time40_delay1500_post
# doit subject01_moco_perturb_torque25_time45_delay1500
# doit subject01_moco_perturb_torque25_time45_delay1500_post
# doit subject01_moco_perturb_torque50_time45_delay1500
# doit subject01_moco_perturb_torque50_time45_delay1500_post
# doit subject01_moco_perturb_torque75_time45_delay1500
# doit subject01_moco_perturb_torque75_time45_delay1500_post
# doit subject01_moco_perturb_torque100_time45_delay1500
# doit subject01_moco_perturb_torque100_time45_delay1500_post
doit subject01_moco_perturb_torque25_time50_delay1500
doit subject01_moco_perturb_torque25_time50_delay1500_post
doit subject01_moco_perturb_torque50_time50_delay1500
doit subject01_moco_perturb_torque50_time50_delay1500_post
doit subject01_moco_perturb_torque75_time50_delay1500
doit subject01_moco_perturb_torque75_time50_delay1500_post
doit subject01_moco_perturb_torque100_time50_delay1500
doit subject01_moco_perturb_torque100_time50_delay1500_post
doit subject01_moco_perturb_torque25_time55_delay1500
doit subject01_moco_perturb_torque25_time55_delay1500_post
doit subject01_moco_perturb_torque50_time55_delay1500
doit subject01_moco_perturb_torque50_time55_delay1500_post
doit subject01_moco_perturb_torque75_time55_delay1500
doit subject01_moco_perturb_torque75_time55_delay1500_post
doit subject01_moco_perturb_torque100_time55_delay1500
doit subject01_moco_perturb_torque100_time55_delay1500_post
doit subject01_moco_perturb_torque25_time60_delay1500
doit subject01_moco_perturb_torque25_time60_delay1500_post
doit subject01_moco_perturb_torque50_time60_delay1500
doit subject01_moco_perturb_torque50_time60_delay1500_post
doit subject01_moco_perturb_torque75_time60_delay1500
doit subject01_moco_perturb_torque75_time60_delay1500_post
doit subject01_moco_perturb_torque100_time60_delay1500
doit subject01_moco_perturb_torque100_time60_delay1500_post