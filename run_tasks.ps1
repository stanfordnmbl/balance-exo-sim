# Data processing
# ---------------
# doit ankle_perturb_sim_copy_data                                
# doit ankle_perturb_sim_copy_generic_model_files
# doit ankle_perturb_sim_transform_experimental_data                 
# doit ankle_perturb_sim_extract_and_filter_emg
# doit ankle_perturb_sim_extract_and_filter_perturbation_forces
# doit ankle_perturb_sim_filter_and_shift_grfs

# Model scaling and adjustments
# -----------------------------
# doit subject*_scale_setup
# doit subject*_scale
# doit subject*_scale_max_force
# doit subject*_adjust_scaled_model_markers
# doit subject*_unperturbed_gait_landmarks

# Inverse kinematics and inverse dynamics
# ---------------------------------------                   
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
# doit subject01_moco_unperturbed_guess_mesh35_costsDisabled_periodic
# doit subject01_moco_unperturbed_guess_mesh35_periodic
doit subject01_moco_unperturbed_mesh35
# doit subject01_moco_unperturbed_mesh20
# doit subject01_moco_unperturbed_mesh10

# Perturbations at 350ms muscle delay
# -----------------------------------
# doit subject01_ankle_torque_perturbed_walking_torque100_time30_delay400
# doit subject01_ankle_torque_perturbed_walking_torque100_time30_delay400_post
# doit subject01_ankle_torque_perturbed_walking_torque100_time35_delay400
# doit subject01_ankle_torque_perturbed_walking_torque100_time35_delay400_post
# doit subject01_ankle_torque_perturbed_walking_torque100_time40_delay400
# doit subject01_ankle_torque_perturbed_walking_torque100_time40_delay400_post
# doit subject01_ankle_torque_perturbed_walking_torque100_time45_delay400
# doit subject01_ankle_torque_perturbed_walking_torque100_time45_delay400_post
# doit subject01_ankle_torque_perturbed_walking_torque100_time50_delay400
# doit subject01_ankle_torque_perturbed_walking_torque100_time50_delay400_post
# doit subject01_ankle_torque_perturbed_walking_torque100_time55_delay400
# doit subject01_ankle_torque_perturbed_walking_torque100_time55_delay400_post
# doit subject01_ankle_torque_perturbed_walking_torque100_time60_delay400
# doit subject01_ankle_torque_perturbed_walking_torque100_time60_delay400_post
# doit subject01_ankle_torque_perturbed_walking_torque75_time40_delay400
# doit subject01_ankle_torque_perturbed_walking_torque75_time40_delay400_post
# doit subject01_ankle_torque_perturbed_walking_torque75_time50_delay400
# doit subject01_ankle_torque_perturbed_walking_torque75_time50_delay400_post
# doit subject01_ankle_torque_perturbed_walking_torque75_time60_delay400
# doit subject01_ankle_torque_perturbed_walking_torque75_time60_delay400_post
# doit subject01_ankle_torque_perturbed_walking_torque50_time40_delay400
# doit subject01_ankle_torque_perturbed_walking_torque50_time40_delay400_post
# doit subject01_ankle_torque_perturbed_walking_torque50_time50_delay400
# doit subject01_ankle_torque_perturbed_walking_torque50_time50_delay400_post
# doit subject01_ankle_torque_perturbed_walking_torque50_time60_delay400
# doit subject01_ankle_torque_perturbed_walking_torque50_time60_delay400_post
# doit subject01_ankle_torque_perturbed_walking_torque25_time40_delay400
# doit subject01_ankle_torque_perturbed_walking_torque25_time40_delay400_post
# doit subject01_ankle_torque_perturbed_walking_torque25_time50_delay400
# doit subject01_ankle_torque_perturbed_walking_torque25_time50_delay400_post
# doit subject01_ankle_torque_perturbed_walking_torque25_time60_delay400
# doit subject01_ankle_torque_perturbed_walking_torque25_time60_delay400_post