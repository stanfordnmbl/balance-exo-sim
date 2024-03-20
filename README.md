# balance-exo-sim

Simulations to understand the effect of ankle exoskeleton torques on changes 
in center of mass kinematics during walking.

Models, data, and results from this project can be found on SimTK.org: 
- https://simtk.org/projects/balance-exo-sim

## Publication

Bianco, N. A., Collins, S. H., Liu, K., and Delp, S. L., “Simulating the effect 
of ankle plantarflexion and inversion-eversion exoskeleton torques on center of 
mass kinematics during walking,” PLOS Comp Bio, 2023. 
doi: 10.1371/journal.pcbi.1010712. (2023)

Software requirements
---------------------
- OpenSim 4.4.1 or later
- R 4.2.1 or later (if running the statistics)
- Python 3.8 (an Anaconda environment can be loaded from conda_enviroment.yml)
  - pandas
  - pyyaml==5.4.1
  - doit
  - opencv
  - matplotlib
  - scipy
  - numpy=1.20.2

Config file setup
-----------------
Before running any simulations, you will need create a file called 'config.yml'
and place it in the top directory of the repository. This file will contain
various values and paths needed to run the simulation pipeline. 

The following four entries are used to configure certain aspects of the 
simulation pipeline to be include or exclude. The following settings skip the 
trajectory optimization tasks (solutions are already included in the 
repository) but run the time-stepping integration tasks that produce model 
kinematic changes in response to applied exoskeleton torques:

- unperturbed_initial_guess: True
- skip_tracking_solve: True
- skip_timestepping_solve: False
- enable_perturbed_plotting_tasks: False

The remaining entries are paths that will be specific to your system and 
represent the top-level repository path need by the 'doit' Python package, the
OpenSim installation directory, the directory containing the raw mocap data, and
subdirectories needed for different parts of the simulation pipeline:

- doit_path: Path to the repository root (e.g., C:\Users\Nick\Repos\balance-exo-sim)
- motion_capture_data_path: Path to raw mocap data (download from SimTK project)
- R_exec_path: \path\to\Rscript.exe (R language executable)
- results_path: <doit_path>\results
- analysis_path: <doit_path>\analysis
- validate_path: <doit_path>\validate
- statistics_path: <doit_path>\statistics
- figures_path: <doit_path>\figures

'doit' workflow
---------------
All steps in the simulation pipeline for this study are handled using the Python
package 'doit', which is a task automation tool. 'Doit' keeps track of the tasks
that have been run and which previous tasks need to be run in order to run tasks
later in the pipeline. 

The top-level 'doit' file is called 'dodo.py'; all tasks for the study eminate 
from this file. Subject-specific tasks are defined in 'subjectXX.py' files. These
files instantiate "tasks", which are individual blocks of simulation code that 
'doit' will execute. The core OpenSim pipeline tasks are contained in the 
'osimpipeline' submodule, and study-specific tasks are contained in the 
'tasks.py'. 

Tasks are executed by calling 'doit <task_name>' in PowerShell (or other
terminal environment). Call 'doit list' to see a list of available tasks that 
can be run. The file 'run_tasks.ps1' is a PowerShell script containing the tasks
needed to be run to reproduce the study. The tasks are listed in order, and if 
you run a task before running the necessary preceeding tasks, those tasks will
be run automatically. You can force a task to run without the preceeding tasks
by calling 'doit -s <task_name>'; this can be helpful if the 'doit' cache gets
corrupted and you know that all preceeding tasks have been run already.

Docker container
----------------
A Docker container that includes a Python environment and an OpenSim installation
needed to reproduce the study results can be created from the Dockerfile located
under the subdirectory '/docker'. The text file 'commands.txt' contains a list
of useful commands for building the Docker image, creating a container, 
configuring the 'config.yml' and 'run_tasks.sh' files, and executing the 
simulation workflow.

### NOTE
In a previous version of this project, we refered to ankle exoskeleton
torques as 'perturbations'. The results folders are still organized using the
terminology 'unperturbed' for normal walking simulation results, and 
'perturbed' for results with exoskeleton torques applied. 'ankle_perturb_sim' is
the previous name of the repository, and therefore many files have this label.
