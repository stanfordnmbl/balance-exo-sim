import os
import opensim as osim

if __name__ == "__main__":
    import argparse


    root_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=str)

    parser.add_argument('--unperturbed', dest='unperturb', action='store_true')
    parser.add_argument('--ankle-perturbation', dest='ankle_perturb', action='store_true')
    parser.add_argument('--peak-torque', dest='peak_torque', type=int)
    parser.add_argument('--peak-time', dest='peak_time', type=int)
    parser.add_argument('--delay', dest='delay', type=int)
    parser.set_defaults(unperturb=False, ankle_perturb=False, 
                        peak_torque=None, peak_time=None, delay=None)

    args = parser.parse_args()

    subdir = 'ankle_perturb' if args.ankle_perturb else 'moco'
    dir = os.path.join(root_dir, 'results', 'experiments', 'subject01',
        args.trial, subdir)
    if args.peak_torque and args.peak_time and args.delay:
        dir = os.path.join(dir, 
            f'torque{args.peak_torque}_time{args.peak_time}_delay{args.delay}')

    model = osim.Model(os.path.join(dir, 'model.osim'))
    model.initSystem()

    solution_fname = 'perturb.sto'
    solution_fname = 'unperturbed.sto' if args.unperturb else solution_fname
    if args.peak_torque and args.peak_time and args.delay:
        solution_fname = f'perturb_torque{args.peak_torque}_time{args.peak_time}_delay{args.delay}.sto'
    solution = osim.MocoTrajectory(os.path.join(dir, solution_fname))
    osim.VisualizerUtilities.showMotion(model, solution.exportToStatesTable())