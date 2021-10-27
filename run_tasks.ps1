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
# doit subject*_unperturbed_gait_landmarks            
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
# doit forget subject01_moco_unperturbed_mesh20
# doit subject01_moco_unperturbed_mesh20
# doit subject01_moco_unperturbed_mesh10

# Perturbed walking
# -------------------
# doit subject01_moco_perturbed_torque100_time30_delay400
# doit subject01_moco_perturbed_torque100_time30_delay400_post
# doit subject01_moco_perturbed_torque100_time35_delay400
# doit subject01_moco_perturbed_torque100_time35_delay400_post
# doit subject01_moco_perturbed_torque100_time40_delay400
# doit subject01_moco_perturbed_torque100_time40_delay400_post
# doit subject01_moco_perturbed_torque100_time45_delay400
# doit subject01_moco_perturbed_torque100_time45_delay400_post
# doit subject01_moco_perturbed_torque100_time50_delay400
# doit subject01_moco_perturbed_torque100_time50_delay400_post
# doit subject01_moco_perturbed_torque100_time55_delay400
# doit subject01_moco_perturbed_torque100_time55_delay400_post
# doit subject01_moco_perturbed_torque100_time60_delay400
# doit subject01_moco_perturbed_torque100_time60_delay400_post

# Baseline torque, unperturbed and perturbed
# ------------------------------------------
# doit forget subject01_moco_baseline_ankle_torque
# doit subject01_moco_baseline_ankle_torque
# doit subject01_moco_perturbed_from_baseline_ankle_torque20_time50_delay1000
# doit subject01_moco_perturbed_from_baseline_ankle_torque30_time50_delay1000
# doit subject01_moco_perturbed_from_baseline_ankle_torque40_time50_delay1000
# doit subject01_moco_perturbed_from_baseline_ankle_torque60_time50_delay1000
# doit subject01_moco_perturbed_from_baseline_ankle_torque70_time50_delay1000
# doit subject01_moco_perturbed_from_baseline_ankle_torque80_time50_delay1000

# Unperturbed and perturbed: two gait cycles
# ------------------------------------------
doit subject01_moco_baseline_ankle_torque_two_cycles
doit subject01_moco_perturb_from_baseline_two_cycles_torque20_time50_delay400
doit subject01_moco_perturb_from_baseline_two_cycles_torque20_time50_delay400_post
doit subject01_moco_perturb_from_baseline_two_cycles_torque30_time50_delay400
doit subject01_moco_perturb_from_baseline_two_cycles_torque30_time50_delay400_post
doit subject01_moco_perturb_from_baseline_two_cycles_torque40_time50_delay400
doit subject01_moco_perturb_from_baseline_two_cycles_torque40_time50_delay400_post
doit subject01_moco_perturb_from_baseline_two_cycles_torque60_time50_delay400
doit subject01_moco_perturb_from_baseline_two_cycles_torque60_time50_delay400_post
doit subject01_moco_perturb_from_baseline_two_cycles_torque70_time50_delay400
doit subject01_moco_perturb_from_baseline_two_cycles_torque70_time50_delay400_post
doit subject01_moco_perturb_from_baseline_two_cycles_torque80_time50_delay400
doit subject01_moco_perturb_from_baseline_two_cycles_torque80_time50_delay400_post
doit subject01_moco_unperturbed_two_cycles_mesh20
doit subject01_moco_perturb_two_cycles_torque100_time30_delay400
doit subject01_moco_perturb_two_cycles_torque100_time30_delay400_post
doit subject01_moco_perturb_two_cycles_torque100_time40_delay400
doit subject01_moco_perturb_two_cycles_torque100_time40_delay400_post
doit subject01_moco_perturb_two_cycles_torque100_time50_delay400
doit subject01_moco_perturb_two_cycles_torque100_time50_delay400_post
doit subject01_moco_perturb_two_cycles_torque100_time60_delay400
doit subject01_moco_perturb_two_cycles_torque100_time60_delay400_post
doit subject01_moco_perturb_two_cycles_torque25_time50_delay400
doit subject01_moco_perturb_two_cycles_torque25_time50_delay400_post
doit subject01_moco_perturb_two_cycles_torque50_time50_delay400
doit subject01_moco_perturb_two_cycles_torque50_time50_delay400_post
doit subject01_moco_perturb_two_cycles_torque75_time50_delay400
doit subject01_moco_perturb_two_cycles_torque75_time50_delay400_post