import itertools
import osimpipeline as osp
import tasks
import pandas as pd
import numpy as np
import os

def generate_main_tasks(trial):

    # inverse kinematics
    ik_setup_task = trial.add_task(osp.TaskIKSetup)
    trial.add_task(osp.TaskIK, ik_setup_task)
    trial.add_task(osp.TaskIKPost, ik_setup_task,
        error_markers=trial.study.error_markers)

    # inverse dynamics
    id_setup_task = trial.add_task(osp.TaskIDSetup, ik_setup_task)
    trial.add_task(osp.TaskID, id_setup_task)
    # trial.add_task(osp.TaskIDPost, id_setup_task)

    return ik_setup_task, id_setup_task