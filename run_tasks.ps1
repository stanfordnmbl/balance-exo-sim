# Data pre-processing
# ---------------
# doit ankle_perturb_sim_copy_data
# doit ankle_perturb_sim_copy_generic_model_files
# doit ankle_perturb_sim_apply_markerset_to_generic_model                              

# Model scaling
# -------------
# doit subject01_scale_setup
# doit subject01_scale
doit subject*_scale_max_force

# Inverse kinematics and inverse dynamics
# --------------------------------------- 
doit subject*_walk2_update_ground_reaction_labels
doit subject*_walk2_filter_ground_reactions      
doit subject*_walk2_gait_landmarks 
doit subject*_walk2_ik_setup           
doit subject*_walk2_ik                                  
doit subject*_walk2_ik_post
doit subject*_walk2_id_setup    
doit subject*_walk2_id                                 
doit subject*_walk2_joint_angle_standard_deviations
doit subject*_walk2_trim_tracking_data

# Sensitivity analysis
# --------------------
# doit subject01_moco_sensitivity_tol1000.0
# doit subject01_moco_sensitivity_tol100.0
# doit subject01_moco_sensitivity_tol10.0
# doit subject01_moco_sensitivity_tol1.0
# doit subject01_moco_sensitivity_tol0.1
# doit subject01_moco_sensitivity_tol0.01
# doit subject01_moco_sensitivity_tol0.001
# doit subject01_moco_sensitivity_tol0.0001

# doit subject02_moco_sensitivity*
# doit subject04_moco_sensitivity*
# doit subject18_moco_sensitivity*
# doit subject19_moco_sensitivity*

# Generate results
# ----------------
doit subject01_moco_unperturbed
# doit subject01_moco_unperturbed_lumbar0.1
# doit subject01_moco_unperturbed_lumbar10.0
# doit subject01_moco_perturbed*torque_actuators
# doit subject01_moco_perturbed*torque_actuators_post

doit subject02_moco_unperturbed
# doit subject02_moco_unperturbed_lumbar0.1
# doit subject02_moco_unperturbed_lumbar10.0
# doit subject02_moco_perturbed*torque_actuators
# doit subject02_moco_perturbed*torque_actuators_post

doit subject04_moco_unperturbed
# doit subject04_moco_unperturbed_lumbar0.1
# doit subject04_moco_unperturbed_lumbar10.0
# doit subject04_moco_perturbed*torque_actuators
# doit subject04_moco_perturbed*torque_actuators_post

# doit subject18_moco_unperturbed_guess_mesh0.05_scale0.001_reserve1000
# doit subject18_moco_unperturbed_guess_mesh0.04_scale0.1_reserve100
# doit subject18_moco_unperturbed_guess_mesh0.02_scale1.0_reserve0_periodic
doit subject18_moco_unperturbed
# doit subject18_moco_unperturbed_lumbar0.1
# doit subject18_moco_unperturbed_lumbar10.0
# doit subject18_moco_perturbed*torque_actuators
# doit subject18_moco_perturbed*torque_actuators_post

doit subject19_moco_unperturbed
# doit subject19_moco_unperturbed_lumbar0.1
# doit subject19_moco_unperturbed_lumbar10.0
# doit subject19_moco_perturbed*torque_actuators
# doit subject19_moco_perturbed*torque_actuators_post

doit plot_unperturbed_results
