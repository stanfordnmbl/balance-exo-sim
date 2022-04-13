import os
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from abc import ABC, abstractmethod

import opensim as osim

import utilities
from utilities import toarray

class Result(ABC):
    def __init__(self):
        self.emg_sensor_names = [
            'SOL', 'GAS', 'TA', 'MH', 'BF', 'VL', 'VM', 'RF', 'GMAX', 'GMED']

    @abstractmethod
    def generate_results(self):
        pass

    @abstractmethod
    def report_results(self):
        pass

    def shift(self, time, y, initial_time=None, final_time=None, starting_time=None):
        if not initial_time:
            initial_time = self.initial_time
        if not final_time:
            final_time = self.final_time
        if not starting_time:
            starting_time = self.initial_time
        return utilities.shift_data_to_cycle(initial_time, final_time,
            starting_time, time, y, cut_off=True)

    def plot(self, ax, time, y, shift=True, fill=False, *args, **kwargs):
        if shift:
            shifted_time, shifted_y = self.shift(time, y)
        else:
            duration = self.final_time - self.initial_time
            shifted_time, shifted_y = self.shift(time, y,
                starting_time=self.initial_time + 0.5 * duration)

        duration = self.final_time - self.initial_time
        if fill:
            return ax.fill_between(
                100.0 * shifted_time / duration,
                shifted_y,
                np.zeros_like(shifted_y),
                *args,
                clip_on=False, **kwargs)
        else:
            return ax.plot(100.0 * shifted_time / duration, shifted_y, *args,
                           clip_on=False, **kwargs)

    def load_electromyography(self, root_dir):
        anc = utilities.ANCFile(
            os.path.join(root_dir, 'data/Rajagopal2016/emg_walk_raw.anc'))
        raw = anc.data
        fields_to_remove = []
        for name in anc.names:
            if name != 'time' and name not in self.emg_sensor_names:
                fields_to_remove.append(name)
        del name

        # We don't actually use the data that is initially in this object. We
        # will overwrite all the data with the filtered data.
        filtered_emg = utilities.remove_fields_from_structured_ndarray(raw,
            fields_to_remove).copy()

        # Debugging.
        emg_fields = list(filtered_emg.dtype.names)
        emg_fields.remove('time')
        for expected_field in self.emg_sensor_names:
            if expected_field not in emg_fields:
                raise Exception("EMG field {} not found.".format(
                    expected_field))

        # Filter all columns.
        for name in filtered_emg.dtype.names:
            if name != 'time':
                scaled_raw = anc.ranges[name] * 2 / 65536.0 * 0.001 * anc[name]
                filtered_emg[name] = utilities.filter_emg(
                    scaled_raw.copy(), anc.rates[name])
                filtered_emg[name] /= np.max(filtered_emg[name])
        return filtered_emg

    def load_electromyography_PerryBurnfield(self, root_dir):
        return np.genfromtxt(os.path.join(root_dir, 'data',
                                          'PerryBurnfieldElectromyography.csv'),
                             names=True,
                             delimiter=',')

    def calc_negative_muscle_forces_base(self, model, solution):
        model.initSystem()
        outputs = osim.analyze(model, solution.exportToStatesTable(),
            solution.exportToControlsTable(), ['.*\|tendon_force'])
        def simtkmin(simtkvec):
            lowest = np.inf
            for i in range(simtkvec.size()):
                if simtkvec[i] < lowest:
                    lowest = simtkvec[i]
            return lowest

        negforces = list()
        muscle_names = list()
        for imusc in range(model.getMuscles().getSize()):
            musc = model.updMuscles().get(imusc)
            max_iso = musc.get_max_isometric_force()
            force = outputs.getDependentColumn(
                musc.getAbsolutePathString() + "|tendon_force")
            neg = simtkmin(force) / max_iso
            if neg < 0:
                negforces.append(neg)
                muscle_names.append(musc.getName())
                print(f'  {musc.getName()}: {neg} F_iso')
        if len(negforces) == 0:
            print('No negative forces')
        else:
            imin = np.argmin(negforces)
            print(f'Largest negative force: {muscle_names[imin]} '
                  f'with {negforces[imin]} F_iso')
        return min([0] + negforces)

    def bound_passive_muscle_forces(self, root_dir, model, 
            max_passive_multiplier=0.05):
        # Update passive muscle force parameters so that muscle passive force
        # doesn't exceed a maximum value, assuming a rigid tendon. Muscle-tendon
        # length information was obtained from an OpenSim MuscleAnalysis using
        # the reference coordinate data.
        print(f'Updating muscle passive force parameters...')
        muscleTendonLengths = osim.TimeSeriesTable(os.path.join(root_dir,
            'data/Rajagopal2016/muscle_tendon_lengths.sto'))

        model.initSystem()
        muscles = model.getMuscles()
        for imusc in np.arange(muscles.getSize()):
            muscle = osim.DeGrooteFregly2016Muscle.safeDownCast(
                muscles.get(int(imusc)))
            muscName = muscle.getName()

            tendonSlackLength = muscle.getTendonSlackLength()
            optimalFiberLength = muscle.getOptimalFiberLength()
            maxIsometricForce = muscle.getMaxIsometricForce()
            muscleTendonLength = \
                muscleTendonLengths.getDependentColumn(muscName)

            maxMuscleTendonLength = 0
            for i in range(muscleTendonLengths.getNumRows()):
                if muscleTendonLength[i] > maxMuscleTendonLength:
                    maxMuscleTendonLength = muscleTendonLength[i]

            maxFiberLength = maxMuscleTendonLength - tendonSlackLength
            maxNormFiberLength = maxFiberLength / optimalFiberLength
            currStrain = muscle.get_passive_fiber_strain_at_one_norm_force()
            currMultiplier = \
                muscle.calcPassiveForceMultiplier(maxNormFiberLength)

            if currMultiplier > max_passive_multiplier:
                while currMultiplier > max_passive_multiplier:
                    currStrain *= 1.05
                    muscle.set_passive_fiber_strain_at_one_norm_force(currStrain)
                    currMultiplier = \
                        muscle.calcPassiveForceMultiplier(maxNormFiberLength)

                print(f'  --> Updated {muscName} passive fiber strain at one '
                      f'normalized force to {currStrain} with force '
                      f'{currMultiplier*maxIsometricForce}')
        print('\n')

        return model

    def load_table(self, table_path):
        num_header_rows = 1
        with open(table_path) as f:
            for line in f:
                if not line.startswith('endheader'):
                    num_header_rows += 1
                else:
                    break
        return np.genfromtxt(table_path, names=True, delimiter='\t',
                             skip_header=num_header_rows)

    def calc_reserves(self, config, model, solution):
        output = osim.analyze(model, solution.exportToStatesTable(), 
            solution.exportToControlsTable(), ['.*reserve.*actuation'])
        return output

    def calc_muscle_mechanics(self, config, model, solution):
        outputList = list()
        for output in ['normalized_fiber_length', 'normalized_fiber_velocity', 
                       'active_fiber_force', 'passive_fiber_force', 
                       'tendon_force', 'activation', 'cos_pennation_angle', 
                       'active_force_length_multiplier', 
                       'force_velocity_multiplier', 'passive_force_multiplier']:
            for imusc in range(model.getMuscles().getSize()):
                musc = model.updMuscles().get(imusc)
                outputList.append(f'.*{musc.getName()}.*\|{output}')

        for output in ['length', 'lengthening_speed']:
            outputList.append(f'.*\|{output}')

        outputs = osim.analyze(model, solution.exportToStatesTable(),
            solution.exportToControlsTable(), outputList)

        return outputs

    def calc_negative_muscle_forces(self, config, model, solution):
        print(f'Negative force report for {config.name}:')
        model.initSystem()
        return self.calc_negative_muscle_forces_base(model, solution)

    def calc_max_knee_reaction_force(self, model, solution):
        jr = osim.analyzeSpatialVec(model, solution.exportToStatesTable(),
                                    solution.exportToControlsTable(),
                                    ['.*walker_knee.*reaction_on_parent.*'])
        jr = jr.flatten(['_mx', '_my', '_mz', '_fx', '_fy', '_fz'])
        traj = np.empty(jr.getNumRows())
        max = -np.inf
        for itime in range(jr.getNumRows()):
            for irxn in range(int(jr.getNumColumns() / 6)):
                fx = jr.getDependentColumnAtIndex(6 * irxn + 3)[itime]
                fy = jr.getDependentColumnAtIndex(6 * irxn + 4)[itime]
                fz = jr.getDependentColumnAtIndex(6 * irxn + 5)[itime]
                norm = np.sqrt(fx**2 + fy**2 + fz**2)
                traj[itime] = norm
                max = np.max([norm, max])
        time = jr.getIndependentColumn()
        avg = np.trapz(traj, x=time) / (time[-1] - time[0])
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.plot(time, traj)
        # plt.show()
        g = np.abs(model.get_gravity()[1])
        state = model.initSystem()
        mass = model.getTotalMass(state)
        weight = mass * g
        return max / weight, avg / weight

    def savefig(self, fig, filename):
        fig.savefig(filename + ".png", format="png", dpi=600)

        # Load this image into PIL
        png2 = Image.open(filename + ".png")

        # Save as TIFF
        png2.save(filename + ".tiff", compression='tiff_lzw')
