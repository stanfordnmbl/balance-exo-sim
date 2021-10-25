import os

import numpy as np
import pylab as pl
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import copy
import shutil
import opensim as osim

import osimpipeline as osp
from osimpipeline import utilities as util
from osimpipeline import postprocessing as pp
from matplotlib import colors as mcolors

from tracking_walking import MotionTrackingWalking, MocoTrackConfig

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


class working_directory():
    """Use this to temporarily run code with some directory as a working
    directory and to then return to the original working directory::

        with working_directory('<dir>'):
            pass
    """
    def __init__(self, path):
        self.path = path
        self.original_working_dir = os.getcwd()
    def __enter__(self):
        os.chdir(self.path)
    def __exit__(self, *exc_info):
        os.chdir(self.original_working_dir)


class TaskCopyMotionCaptureData(osp.TaskCopyMotionCaptureData):
    REGISTRY = []
    def __init__(self, study, subjects, unperturbed_trial, prefix='S01DN6'):
        regex_replacements = list()

        subjects = ['subject%02i' % subj for subj in subjects]
        for subject in subjects:
            # Scale static trial
            regex_replacements.append(
                (
                    os.path.join('OpenSim', subject, 'markers',
                        '%sstanding.trc' % prefix).replace('\\', '\\\\'),
                    os.path.join('experiments',
                        subject, 'static', 'expdata', 
                        'marker_trajectories.trc').replace('\\','\\\\')
                    ))
            # for side in ['_left', '_right']:
            # Vicon
            regex_replacements.append(
                (
                    os.path.join('Vicon', subject,
                        '%s%02i.c3d' % (prefix, unperturbed_trial)).replace('\\', '\\\\'),
                    os.path.join('experiments',
                        subject, f'unperturbed', 'expdata', 
                        'vicon.c3d').replace('\\','\\\\')
                    ))
            # EMG
            regex_replacements.append(
                (
                    os.path.join('Vicon', subject,
                        '%s%02i.mat' % (prefix, unperturbed_trial)).replace('\\', '\\\\'),
                    os.path.join('experiments',
                        subject, f'unperturbed', 'expdata', 
                        'emg.mat').replace('\\','\\\\')
                    ))
            # Speedgoat
            regex_replacements.append(
                (
                    os.path.join('Speedgoat', subject,
                        '%s%02i.mat' % (prefix, unperturbed_trial)).replace('\\', '\\\\'),
                    os.path.join('experiments',
                        subject, f'unperturbed', 'expdata', 
                        'speedgoat.mat').replace('\\','\\\\')
                    ))

        super(TaskCopyMotionCaptureData, self).__init__(study,
                regex_replacements)


class TaskTransformExperimentalData(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects):
        super(TaskTransformExperimentalData, self).__init__(study)
        self.name = study.name + '_transform_experimental_data'

        subjects = ['subject%02i' % subj for subj in subjects]
        for subject in subjects:
            # for side in ['_left', '_right']:
            expdata_path = os.path.join(study.config['results_path'], 
                    'experiments', subject, f'unperturbed', 
                    'expdata')
            self.add_action(
                [os.path.join(expdata_path, 
                    'vicon.c3d').replace('\\','\\\\')],
                [os.path.join(expdata_path, 
                    'marker_trajectories.trc').replace('\\','\\\\'),
                 os.path.join(expdata_path, 
                    'ground_reaction_unfiltered.mot').replace('\\','\\\\')],
                self.transform_experimental_data)

    def transform_experimental_data(self, file_dep, target):

        c3d = osim.C3DFileAdapter()
        c3d.setLocationForForceExpression(0)
        tables = c3d.read(file_dep[0])

        # Set the marker and force data into OpenSim tables
        markers = c3d.getMarkersTable(tables)
        forces = c3d.getForcesTable(tables)

        def rotateTable(table, axis, value):
            R = osim.Rotation(np.deg2rad(value), osim.CoordinateAxis(axis))
            for irow in np.arange(table.getNumRows()):
                rowVec = table.getRowAtIndex(int(irow))
                rowVec_rotated = R.multiply(rowVec)
                table.setRowAtIndex(int(irow), rowVec_rotated)
            return table

        markers = rotateTable(markers, 0, -90)
        markers = rotateTable(markers, 1, -90)
        forces = rotateTable(forces, 0, -90)
        forces = rotateTable(forces, 1, -90)

        # Convert forces to meters
        nrows  = forces.getNumRows()
        ncols  = forces.getNumColumns()
        labels = forces.getColumnLabels()
        for i in np.arange(ncols):
            # All force columns will have the 'f' prefix while point
            # and moment columns will have 'p' and 'm' prefixes,
            # respectively.
            if not labels[i].startswith('f'):
                col = forces.updDependentColumnAtIndex(int(i))
                newCol = osim.VectorVec3(col.size(), osim.Vec3(0))
                for j in np.arange(nrows):
                    # Divide by 1000
                    newCol.set(int(j), col[int(j)].scalarDivideEq(1000))

                forces.setDependentColumnAtIndex(int(i), newCol)

        # Write markers to TRC file
        # -------------------------
        trc = osim.TRCFileAdapter()
        trc.write(markers, target[0])

        # Write GRFs to MOT file
        # ----------------------
        # Get the column labels
        labels = forces.getColumnLabels()
        # Make a copy
        updlabels = list(copy.deepcopy(labels))

        # Labels from C3DFileAdapter are f1, p1, m1, f2,...
        # We edit them to be consistent with requirements of viewing
        # forces in the GUI (ground_force_vx, ground_force_px,...)
        for i in np.arange(len(labels)):

            # Get the label as a string
            label = labels[i]

            # Transform the label depending on force, point, or moment
            if label.startswith('f'):
                label = label.replace('f', 'ground_force_')
                label = label + '_v'

            elif label.startswith('p'):
                label = label.replace('p', 'ground_force_')
                label = label + '_p'

            elif label.startswith('m'):
                label = label.replace('m', 'ground_moment_')
                label = label + '_m'

            # update the label name
            updlabels[i] = label

        # set the column labels
        forces.setColumnLabels(updlabels)

        # Flatten the Vec3 force table
        postfix = osim.StdVectorString()
        postfix.append('x')
        postfix.append('y')
        postfix.append('z')
        forces_flat = forces.flatten(postfix)

        # Change the header in the file to meet Storage conditions
        if len(forces_flat.getTableMetaDataKeys()) > 0:
            for i in np.arange(len(forces_flat.getTableMetaDataKeys())):
                # Get the metakey string at index zero. Since the array gets 
                # smaller on each loop, we just need to keep taking the first 
                # one in the array.
                metakey = forces_flat.getTableMetaDataKeys()[0]
                # Remove the key from the meta data
                forces_flat.removeTableMetaDataKey(metakey)

        # Add the column and row data to the meta key
        forces_flat.addTableMetaDataString('nColumns',
                str(forces_flat.getNumColumns()+1))
        forces_flat.addTableMetaDataString('nRows', 
                str(forces_flat.getNumRows()))

        # Write to file
        sto = osim.STOFileAdapter()
        sto.write(forces_flat, target[1])


class TaskFilterAndShiftGroundReactions(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, threshold=35, sample_rate=2000,
                 critically_damped_order=4, 
                 critically_damped_cutoff_frequency=10,
                 gaussian_smoothing_sigma=5, grf_xpos_offset=-1.75, 
                 grf_zpos_offset=0.55):
        super(TaskFilterAndShiftGroundReactions, self).__init__(study)
        self.name = study.name + '_filter_and_shift_grfs'
        
        # Vertical contact force detection threshold (Newtons)
        self.threshold = threshold 

        # Recorded force sample rate (Hz)
        self.sample_rate = sample_rate

        # Critically damped filter order and cutoff frequency
        self.critically_damped_order = critically_damped_order
        self.critically_damped_cutoff_frequency = \
            critically_damped_cutoff_frequency

        # Smoothing factor for Gaussian smoothing process
        self.gaussian_smoothing_sigma = gaussian_smoothing_sigma

        # Manual adjustments for COP locations
        # TODO: ideally these aren't necessary
        self.grf_xpos_offset = grf_xpos_offset
        self.grf_zpos_offset = grf_zpos_offset

        subjects = ['subject%02i' % subj for subj in subjects]
        for subject in subjects:
            # for side in ['_left', '_right']:
            expdata_path = os.path.join(study.config['results_path'], 
                    'experiments', subject, 'unperturbed', 
                    'expdata')
            self.add_action(
                [os.path.join(expdata_path, 
                    'ground_reaction_unfiltered.mot').replace('\\', '\\\\')],
                [os.path.join(expdata_path, 
                    'ground_reaction.mot').replace('\\', '\\\\'),
                 os.path.join(expdata_path, 
                    'ground_reactions_filtered.png').replace('\\', '\\\\')],
                self.filter_and_shift_grfs)

    def filter_and_shift_grfs(self, file_dep, target):

        n_force_plates = 2
        sides = ['l', 'r']

        # Orientation of force plate frames in the motion capture frame 
        # (x forward, y left, z up).
        force_plate_rotation = [
                np.matrix([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]]),
                np.matrix([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]]),
                ]

        # Orientation of OpenSim ground frame (x forward, y up, z right) in 
        # motion capture frame.
        opensim_rotation = np.matrix([[1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [0.0, 0.0, 1.0]])

        grfs = osim.TimeSeriesTable(file_dep[0])
        nrow = grfs.getNumRows()
        f1 = np.empty((nrow, 3))
        p1 = np.empty((nrow, 3))
        m1 = np.empty((nrow, 3))
        f2 = np.empty((nrow, 3))
        p2 = np.empty((nrow, 3))
        m2 = np.empty((nrow, 3))

        f1[:, 0] = util.toarray(grfs.getDependentColumn('ground_force_1_vx'))
        f1[:, 1] = util.toarray(grfs.getDependentColumn('ground_force_1_vy'))
        f1[:, 2] = util.toarray(grfs.getDependentColumn('ground_force_1_vz'))
        p1[:, 0] = util.toarray(grfs.getDependentColumn('ground_force_1_px'))
        p1[:, 1] = util.toarray(grfs.getDependentColumn('ground_force_1_py'))
        p1[:, 2] = util.toarray(grfs.getDependentColumn('ground_force_1_pz'))
        m1[:, 0] = util.toarray(grfs.getDependentColumn('ground_moment_1_mx'))
        m1[:, 1] = util.toarray(grfs.getDependentColumn('ground_moment_1_my'))
        m1[:, 2] = util.toarray(grfs.getDependentColumn('ground_moment_1_mz'))
        f2[:, 0] = util.toarray(grfs.getDependentColumn('ground_force_2_vx'))
        f2[:, 1] = util.toarray(grfs.getDependentColumn('ground_force_2_vy'))
        f2[:, 2] = util.toarray(grfs.getDependentColumn('ground_force_2_vz'))
        p2[:, 0] = util.toarray(grfs.getDependentColumn('ground_force_2_px'))
        p2[:, 1] = util.toarray(grfs.getDependentColumn('ground_force_2_py'))
        p2[:, 2] = util.toarray(grfs.getDependentColumn('ground_force_2_pz'))
        m2[:, 0] = util.toarray(grfs.getDependentColumn('ground_moment_2_mx'))
        m2[:, 1] = util.toarray(grfs.getDependentColumn('ground_moment_2_my'))
        m2[:, 2] = util.toarray(grfs.getDependentColumn('ground_moment_2_mz'))

        # Positions of force plate frame origins (center) from lab origin 
        # (meters).
        force_plate_location = [p1[0], p2[0]]

        grfData = np.concatenate((f1, m1, f2, m2), axis=1)

        # x right, y backward, z up
        grfLabels = ['F1X', 'F1Y', 'F1Z', 'M1X', 'M1Y', 'M1Z',
                     'F2X', 'F2Y', 'F2Z', 'M2X', 'M2Y', 'M2Z']
        time = np.linspace(0, 20.999, num=grfData.shape[0])

        data = pd.DataFrame(data=grfData, columns=grfLabels, dtype=float)
        n_times = data.shape[0]

        def repack_data(data, vecs, colnames):
            for icol, colname in enumerate(colnames):
                data[colname] = vecs[icol]

        # Coordinate transformation.
        # --------------------------
        for fp in range(n_force_plates):
            force_names = ['F%s%s' % (fp + 1, s) for s in ['X', 'Y', 'Z']]
            moment_names = ['M%s%s' % (fp + 1, s) for s in ['X', 'Y', 'Z']]

            # Unpack 3 columns as rows of vectors.
            force_vecs = data[force_names]
            moment_vecs = data[moment_names]

            # Rotate.
            force_vecs = force_vecs.dot(force_plate_rotation[fp])
            moment_vecs = moment_vecs.dot(force_plate_rotation[fp])

            # Compute moments.
            # M_new = r x F + M_orig.
            r = np.tile(force_plate_location[fp], [n_times, 1])
            moment_vecs = np.cross(r, force_vecs) + moment_vecs

            # Pack back into the data.
            repack_data(data, force_vecs, force_names)
            repack_data(data, moment_vecs, moment_names)

        # Combine force plates to generate left foot, right foot data.
        # ------------------------------------------------------------
        forces = {side: np.zeros((n_times, 3)) for side in sides}
        moments = {side: np.zeros((n_times, 3)) for side in sides}

        for iside, side in enumerate(sides):
            force_names = ['F%s%s' % (iside + 1, s) for s in ['X', 'Y', 'Z']]
            moment_names = ['M%s%s' % (iside + 1, s) for s in ['X', 'Y', 'Z']]

            forces[side] += data[force_names].values
            moments[side] += data[moment_names].values

        # Plot raw GRF (before cutting off or filtering)
        fig = pl.figure(figsize=(12, 10))
        ax_FX = fig.add_subplot(6, 1, 1)
        ax_FY = fig.add_subplot(6, 1, 2)
        ax_FZ = fig.add_subplot(6, 1, 3)
        ax_MX = fig.add_subplot(6, 1, 4)
        ax_MY = fig.add_subplot(6, 1, 5)
        ax_MZ = fig.add_subplot(6, 1, 6)
        for iside, side in enumerate(sides):
            ax_FX.plot(time, forces[side][:, 0], lw=2, color='black')
            ax_FY.plot(time, forces[side][:, 1], lw=2, color='black')
            ax_FZ.plot(time, forces[side][:, 2], lw=2, color='black')
            # ax_MX.plot(time, moments[side][:, 0], lw=2, color='black')
            # ax_MY.plot(time, moments[side][:, 2], lw=2, color='black')
            # ax_MZ.plot(time, moments[side][:, 1], lw=2, color='black')

        # Cut-off and smooth with Gaussian filter
        # ---------------------------------------
        # Before filtering, cutoff the GRF using the first coarser cutoff.
        # Use Gaussian filter after cutoff to smooth force transitions at ground
        # contact and liftoff.
        for side in sides:
            # Vertical force is in the y-direction.
            filt = (forces[side][:, 1] < self.threshold)
            for item in [forces, moments]:
                item[side][filt, :] = 0
                for i in np.arange(item[side].shape[1]):
                    item[side][:, i] = gaussian_filter1d(item[side][:, i], 
                        self.gaussian_smoothing_sigma)

        # Critically damped filter (prevents overshoot).
        # TODO may change array size.
        for item in [forces, moments]:
            for side in sides:
                for direc in range(3):
                    item[side][:, direc] = util.filter_critically_damped(
                            item[side][:, direc], self.sample_rate,
                            self.critically_damped_cutoff_frequency,
                            order=self.critically_damped_order)

        # Compute centers of pressure.
        # ---------------------------
        centers_of_pressure = {side: np.zeros((n_times, 3)) for side in sides}
        for side in sides:
            # Only compute when foot is on ground.
            # Time indices corresponding to foot on ground.
            filt = forces[side][:, 1] != 0
            Mx = moments[side][filt, 0]
            My = moments[side][filt, 1]
            Mz = moments[side][filt, 2]
            Fx = forces[side][filt, 0]
            Fy = forces[side][filt, 1]
            Fz = forces[side][filt, 2]
            COPz = -Mx / Fy
            COPx = Mz / Fy
            centers_of_pressure[side][filt, 0] = COPx
            centers_of_pressure[side][filt, 2] = COPz

            # Must have zero when foot is not on ground.
            My_new = np.zeros(n_times)
            My_new[filt] = My - COPz * Fx + COPx * Fz
            moments[side][:, 1] = My_new
            moments[side][:, 0] = 0
            moments[side][:, 2] = 0

        # Apply Gaussian smoothing after computing centers of pressure
        # ------------------------------------------------------------
        for side in sides:
            for item in [forces, moments, centers_of_pressure]:
                for i in np.arange(item[side].shape[1]):
                    item[side][:,i] = gaussian_filter1d(item[side][:,i], 
                        self.gaussian_smoothing_sigma)      

        # Transform from motion capture frame to OpenSim ground frame.
        # ------------------------------------------------------------
        for side in sides:
            for item in [forces, moments]:
                item[side] = item[side] * opensim_rotation
            centers_of_pressure[side] = \
                    centers_of_pressure[side] * opensim_rotation

        # Manual COP adjustments
        # ----------------------
        # TODO: avoid these
        # Shift COP to model feet
        centers_of_pressure['l'][:,0] += self.grf_xpos_offset
        centers_of_pressure['l'][:,2] += -self.grf_zpos_offset
        centers_of_pressure['r'][:,0] += self.grf_xpos_offset
        centers_of_pressure['r'][:,2] += self.grf_zpos_offset

        # Create structured array for MOT file.
        # -------------------------------------
        dtype_names = ['time']
        data_dict = dict()
        for side in sides:
            # Force.
            for idirec, direc in enumerate(['x', 'y', 'z']):
                colname = 'ground_force_%s_v%s' % (side, direc)
                dtype_names.append(colname)
                data_dict[colname] = forces[side][:, idirec].reshape(-1)

            # Center of pressure.
            for idirec, direc in enumerate(['x', 'y', 'z']):
                colname = 'ground_force_%s_p%s' % (side, direc)
                dtype_names.append(colname)
                data_dict[colname] = \
                        centers_of_pressure[side][:, idirec].reshape(-1)

            # Moment.
            for idirec, direc in enumerate(['x', 'y', 'z']):
                colname = 'ground_torque_%s_%s' % (side, direc)
                dtype_names.append(colname)
                data_dict[colname] = \
                        moments[side][:, idirec].reshape(-1)

        mot_data = np.empty(nrow, dtype={'names': dtype_names,
            'formats': len(dtype_names) * ['f8']})
        # TODO discrepancy with Amy.
        mot_data['time'] = time #[[0] + range(nrow-1)]
        for k, v in data_dict.items():
            mot_data[k] = v

        util.ndarray2storage(mot_data, target[0], name='ground reactions')

        # Plot filtered GRFs
        # ------------------
        for side in sides:
            ax_FX.plot(time, forces[side][:, 0], lw=1.5, color='red')
            ax_FY.plot(time, forces[side][:, 1], lw=1.5, color='red')
            ax_FZ.plot(time, forces[side][:, 2], lw=1.5, color='red')
            ax_MX.plot(time, moments[side][:, 0], lw=1.5, color='red')
            ax_MY.plot(time, moments[side][:, 1], lw=1.5, color='red')
            ax_MZ.plot(time, moments[side][:, 2], lw=1.5, color='red')
        ax_FX.set_ylabel('FX')
        ax_FY.set_ylabel('FY')
        ax_FZ.set_ylabel('FZ')
        ax_MX.set_ylabel('MX')
        ax_MY.set_ylabel('MY')
        ax_MZ.set_ylabel('MZ')
        fig.savefig(target[1])
        pl.close()


class TaskExtractAndFilterEMG(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, frequency_band=[10.0, 400.0],
            filter_order=4.0, lowpass_freq=6.0):
        super(TaskExtractAndFilterEMG, self).__init__(study)
        self.name = study.name + '_extract_and_filter_emg'
        self.labels = ['tibant_l', 'soleus_l', 'gasmed_l', 'semitend_l', 
                       'vaslat_l', 'recfem_l','glmax_l','glmed_l', 'tibant_r', 
                       'soleus_r', 'gasmed_r', 'semitend_r', 'vaslat_r', 
                       'recfem_r','glmax_r','glmed_r']
        self.sample_rate = 1000 # Hz
        self.frequency_band = frequency_band
        self.filter_order = filter_order
        self.lowpass_freq = lowpass_freq

        subjects = ['subject%02i' % subj for subj in subjects]
        for subject in subjects:
            # for side in ['_left', '_right']:
            expdata_path = os.path.join(study.config['results_path'], 
                    'experiments', subject, f'unperturbed', 
                    'expdata')
            self.add_action(
                [os.path.join(expdata_path, 
                    'emg.mat').replace('\\','\\\\')],
                [os.path.join(expdata_path, 
                    'emg.sto').replace('\\','\\\\'),
                 os.path.join(expdata_path, 
                    'emg_filtered.png').replace('\\','\\\\')],
                self.extract_and_filter_emg)

    def extract_and_filter_emg(self, file_dep, target):

        from scipy.signal import butter
        from scipy.signal import filtfilt

        mat = loadmat(file_dep[0])
        
        emg = mat['data']['Analog'][0][0][0][0][-1][:,24:40]
        emgFiltered = np.zeros_like(emg)
        emgUnfiltered = np.zeros_like(emg)
        nrows = emg.shape[0]
        ncols = emg.shape[1]

        Wn = [wn / (0.5 * self.sample_rate) for wn in self.frequency_band]
        b_bp, a_bp = butter(self.filter_order, Wn, 'bandpass')
        Wn_low = self.lowpass_freq / (0.5 * self.sample_rate)
        b_low, a_low = butter(self.filter_order, Wn_low, 'low')
        for i in np.arange(ncols):
            
            # bandpass filter
            emgFiltered[:, i] = filtfilt(b_bp, a_bp, emg[:, i])
            emgUnfiltered[:, i] = filtfilt(b_bp, a_bp, emg[:, i])
            
            # demean
            emgFiltered[:, i] = emgFiltered[:, i] - np.mean(emgFiltered[:, i])
            emgUnfiltered[:, i] = emgUnfiltered[:, i] - np.mean(emgUnfiltered[:, i])
            
            # rectify
            emgFiltered[:, i] = np.abs(emgFiltered[:, i])
            emgUnfiltered[:, i] = np.abs(emgUnfiltered[:, i])
                
            # lowpass filter
            emgFiltered[:, i] = filtfilt(b_low, a_low, emgFiltered[:, i])
            
            # normalize
            emgFiltered[:, i] = emgFiltered[:, i] / np.max(emgFiltered[:, i])
            emgUnfiltered[:, i] = emgUnfiltered[:, i] / np.max(emgUnfiltered[:, i])

        # write to file
        time = np.linspace(0, (nrows-1)/1000, num=nrows)
        timeVec = osim.StdVectorDouble()
        for i in np.arange(nrows):
           timeVec.append(time[i])
        
        emgTable = osim.TimeSeriesTable(timeVec)

        for j in np.arange(ncols):
           col = osim.Vector(nrows, 0.0)
           label = self.labels[j]
           for i in np.arange(nrows):
               col.set(int(i), float(emgFiltered[i, j]))
           
           emgTable.appendColumn(label, col)
        
        sto = osim.STOFileAdapter()
        sto.write(emgTable, target[0])

        # plot
        fig = pl.figure(figsize=(12, 12))
        side = np.round(np.sqrt(ncols))

        istart = 8000
        iend = 12000
        for j in np.arange(ncols):
            ax = fig.add_subplot(int(side), int(side), int(j+1))
            ax.plot(time[istart:iend], emgUnfiltered[istart:iend, j], 
                lw=2, color='black')
            ax.plot(time[istart:iend], emgFiltered[istart:iend, j], 
                lw=2, color='red')
            ax.set_title(self.labels[j])

        fig.tight_layout()
        fig.savefig(target[1])
        pl.close()


class TaskExtractAndFilterPerturbationForces(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, lowpass_freq=10.0, 
            filter_order=4.0, threshold=25, gaussian_smoothing_sigma=15):
        super(TaskExtractAndFilterPerturbationForces, self).__init__(study)
        self.name = study.name + '_extract_and_filter_perturbation_forces'
        self.sample_rate = 1000 # Hz
        self.lowpass_freq = lowpass_freq
        self.filter_order = filter_order
        self.threshold = threshold
        self.gaussian_smoothing_sigma = gaussian_smoothing_sigma

        subjects = ['subject%02i' % subj for subj in subjects]
        for subject in subjects:
            # for side in ['_left', '_right']:
            expdata_path = os.path.join(study.config['results_path'], 
                    'experiments', subject, f'unperturbed', 
                    'expdata')
            self.add_action(
                [os.path.join(expdata_path, 
                    'speedgoat.mat').replace('\\','\\\\')],
                [os.path.join(expdata_path, 
                    'perturbation_force.sto').replace('\\','\\\\'),
                 os.path.join(expdata_path, 
                    'perturbation_force_filtered.png').replace('\\','\\\\')],
                self.extract_and_filter_perturbation_force)

    def extract_and_filter_perturbation_force(self, file_dep, target):

        from scipy.signal import butter
        from scipy.signal import filtfilt

        mat = loadmat(file_dep[0])
        perturbForces = mat['speedgoatDataTrimmed'][0][0]['data'][:, 4:8]

        nrows = perturbForces.shape[0]
        ncols = perturbForces.shape[1]

        Wn = self.lowpass_freq / (0.5 * self.sample_rate)
        b, a = butter(self.filter_order, Wn, 'low')
        
        perturbForcesFiltered = np.zeros_like(perturbForces)
        for icol in np.arange(ncols):
            if np.max(perturbForces[:, icol]) < 100: continue
            for irow in np.arange(nrows):
                if perturbForces[irow, icol] > self.threshold:
                    perturbForcesFiltered[irow, icol] = \
                        perturbForces[irow, icol]

            perturbForcesFiltered[:, icol] = gaussian_filter1d(
                perturbForcesFiltered[:, icol], self.gaussian_smoothing_sigma)   
 
        # write to file
        time = np.linspace(0, (nrows-1)/1000, num=nrows)
        timeVec = osim.StdVectorDouble()
        for i in np.arange(nrows):
           timeVec.append(time[i])

        perturbTable = osim.TimeSeriesTable(timeVec)
        for icol in np.arange(ncols):
            perturbForceVec = osim.Vector(nrows, 0.0)
            for irow in np.arange(nrows):
               perturbForceVec.set(int(irow), perturbForcesFiltered[irow, icol])

            perturbTable.appendColumn(f'perturbation_force_{icol}', 
                perturbForceVec)

        sto = osim.STOFileAdapter()
        sto.write(perturbTable, target[0]) 

        # Detect onset of perturbation
        # ----------------------------
        istart = 9000
        iend = 11000

        index_range = range(istart, iend)

        threshold = 40 # Newtons
        def zero(number):
            return abs(number) < threshold

        def onset_times(ordinate):
            onsets = list()
            for i in index_range:
                # 'Skip' first value because we're going to peak back at previous
                # index.
                if zero(ordinate[i - 1]) and (not zero(ordinate[i])):
                    onsets.append(time[i])
            return np.array(onsets)

        onsets = list()
        onsets.append(onset_times(perturbForcesFiltered[:,0]))
        onsets.append(onset_times(perturbForcesFiltered[:,1]))
        onsets.append(onset_times(perturbForcesFiltered[:,2]))
        onsets.append(onset_times(perturbForcesFiltered[:,3]))

        # plot
        fig = pl.figure(figsize=(8, 8))

        istart = 9000
        iend = 11000
        for icol in np.arange(ncols):
            ax = fig.add_subplot(2, 2, icol+1)
            ax.plot(time[istart:iend], perturbForces[istart:iend, icol], 
                lw=2, color='black')
            ax.plot(time[istart:iend], perturbForcesFiltered[istart:iend, icol], 
                lw=2, color='red')
            ax.set_title(f'perturbation_force_{icol}')

            theseOnsets = onsets[icol]
            ones = np.array([1, 1])
            for i, onset in enumerate(theseOnsets):
                if i == 0: kwargs = {'label': 'onsets'}
                else: kwargs = dict()
                pl.plot(onset * ones, ax.get_ylim(), 'b', **kwargs)
                pl.text(onset, .03 * ax.get_ylim()[1], ' %.3f' % round(onset, 3))

        fig.tight_layout()
        fig.savefig(target[1])
        pl.close()


class TaskScaleMuscleMaxIsometricForce(osp.SubjectTask):
    REGISTRY = []
    """The generic model mass and heights are based on the generic Rajagopal
       et al. 2015 model.
    """
    def __init__(self, subject, generic_mass=75.337, generic_height=1.6557):
        super(TaskScaleMuscleMaxIsometricForce, self).__init__(subject)
        self.subject = subject
        self.name = '%s_scale_max_force' % self.subject.name
        self.doc = 'Scale subject muscle Fmax parameters from Handsfield2014'
        self.generic_model_fpath = self.study.source_generic_model_fpath
        self.subject_model_fpath = os.path.join(self.subject.results_exp_path, 
            '%s.osim' % self.subject.name)
        self.scaled_param_model_fpath = os.path.join(
            self.subject.results_exp_path, 
            '%s_scaled_Fmax.osim' % self.subject.name)
        self.generic_mass = generic_mass
        self.generic_height = generic_height

        self.add_action([self.generic_model_fpath, self.subject_model_fpath],
                        [self.scaled_param_model_fpath],
                        self.scale_model_parameters)

    def scale_model_parameters(self, file_dep, target):
        """From Handsfields 2014 figure 5a and from Apoorva's muscle properties
       spreadsheet.
       
       v: volume fraction
       V: total volume
       F: max isometric force
       l: optimal fiber length

       F = v * sigma * V / l

       *_g: generic model.
       *_s: subject-specific model.

       F_g = v * sigma * V_g / l_g
       F_s = v * sigma * V_s / l_s

       F_s = (F_g * l_g / V_g) * V_s / l_s
           = F_g * (V_s / V_g) * (l_g / l_s)

        Author: Chris Dembia 
        Borrowed from mrsdeviceopt GitHub repo:
        https://github.com/chrisdembia/mrsdeviceopt          
       """

        print("Muscle force scaling: "
              "total muscle volume and optimal fiber length.")

        # def total_muscle_volume_regression(mass):
        #     return 91.0*mass + 588.0

        def total_muscle_volume_regression(mass, height):
            return 47.0*mass*height + 1285.0

        generic_TMV = total_muscle_volume_regression(self.generic_mass, 
            self.generic_height)
        subj_TMV = total_muscle_volume_regression(self.subject.mass, 
            self.subject.height)

        import opensim as osm
        generic_model = osm.Model(file_dep[0])
        subj_model = osm.Model(file_dep[1])

        generic_mset = generic_model.getMuscles()
        subj_mset = subj_model.getMuscles()

        for im in range(subj_mset.getSize()):
            muscle_name = subj_mset.get(im).getName()

            generic_muscle = generic_mset.get(muscle_name)
            subj_muscle = subj_mset.get(muscle_name)

            generic_OFL = generic_muscle.get_optimal_fiber_length()
            subj_OFL = subj_muscle.get_optimal_fiber_length()

            scale_factor = (subj_TMV / generic_TMV) * (generic_OFL / subj_OFL)
            print("Scaling '%s' muscle force by %f." % (muscle_name,
                scale_factor))

            generic_force = generic_muscle.getMaxIsometricForce()
            scaled_force = generic_force * scale_factor
            subj_muscle.setMaxIsometricForce(scaled_force)

        subj_model.printToXML(target[0])


class TaskAdjustScaledModel(osp.SubjectTask):
    REGISTRY = []
    def __init__(self, subject, marker_adjustments, treadmill=False):
        super(TaskAdjustScaledModel, self).__init__(subject)
        self.subject = subject
        self.study = subject.study
        self.name = '%s_adjust_scaled_model' % self.subject.name
        self.doc = 'Make adjustments to model marker post-scale'
        self.scaled_model_fpath = os.path.join(
            self.subject.results_exp_path, 
            '%s_scaled_Fmax.osim' % self.subject.name)
        if treadmill:
            self.final_model_fpath = os.path.join(
                self.subject.results_exp_path, 
                '%s_scaled_Fmax_markers.osim' % self.subject.name)
        else:
            self.final_model_fpath = os.path.join(
                self.subject.results_exp_path, 
                '%s_final.osim' % self.subject.name)
        self.marker_adjustments = marker_adjustments

        self.add_action([self.scaled_model_fpath],
                        [self.final_model_fpath],
                        self.adjust_model_markers)

    def adjust_model_markers(self, file_dep, target):
        print('Adjusting scaled model marker locations... ')
        import opensim as osm
        model = osm.Model(file_dep[0])
        markerSet = model.updMarkerSet()
        for name, adj in self.marker_adjustments.items():
            marker = markerSet.get(name)
            loc = marker.get_location()
            loc.set(adj[0], adj[1])
            marker.set_location(loc)

        # print('Adding Bump-em module markers... ')
        # module0 = osim.Marker('module0', model.getGround(), self.study.module0)
        # module1 = osim.Marker('module1', model.getGround(), self.study.module1)
        # module2 = osim.Marker('module2', model.getGround(), self.study.module2)
        # module3 = osim.Marker('module3', model.getGround(), self.study.module3)

        # markerSet.adoptAndAppend(module0)
        # markerSet.adoptAndAppend(module1)
        # markerSet.adoptAndAppend(module2)
        # markerSet.adoptAndAppend(module3)
        # model.finalizeConnections()

        print('Adding markers for ankle exoskeletons...')
        state = model.initSystem()
        markerSet = model.updMarkerSet()
        LCAL = markerSet.get('LCAL')
        RCAL = markerSet.get('RCAL')
        LCAL_ground = LCAL.getLocationInGround(state)
        RCAL_ground = RCAL.getLocationInGround(state)
        LCAL_ground[1] = LCAL_ground[1] + 0.25
        RCAL_ground[1] = RCAL_ground[1] + 0.25

        bodySet = model.getBodySet()
        tibia_l = bodySet.get('tibia_l')
        tibia_r = bodySet.get('tibia_r')

        ground = model.getGround()
        LEXO_tibia_l = ground.findStationLocationInAnotherFrame(
            state, LCAL_ground, tibia_l)
        REXO_tibia_r = ground.findStationLocationInAnotherFrame(
            state, RCAL_ground, tibia_r)

        LEXO = osim.Marker('LEXO', tibia_l, LEXO_tibia_l)
        REXO = osim.Marker('REXO', tibia_r, REXO_tibia_r)
        markerSet.adoptAndAppend(LEXO)
        markerSet.adoptAndAppend(REXO)
        model.finalizeConnections()

        print('Unlocking subtalar joints...')
        coordSet = model.updCoordinateSet()
        for coordName in ['subtalar_angle']:
            for side in ['_l', '_r']:
                coord = coordSet.get(f'{coordName}{side}')
                coord.set_locked(False)
                coord.set_clamped(False)

        model.finalizeConnections()
        model.printToXML(target[0])


class TaskComputeJointAngleStandardDeviations(osp.TrialTask):
    REGISTRY = []
    def __init__(self, trial, ik_setup_task):
        super(TaskComputeJointAngleStandardDeviations, self).__init__(trial)
        self.name = trial.id + '_joint_angle_standard_deviations'
        self.trial = trial
        self.ik_solution_fpath = ik_setup_task.solution_fpath

        self.add_action([self.ik_solution_fpath],
                        [os.path.join(trial.results_exp_path, f'{trial.id}_joint_angle_standard_deviations.csv')],
                        self.compute_joint_angle_standard_deviations)

    def compute_joint_angle_standard_deviations(self, file_dep, target):
        
        kinematics = osim.TimeSeriesTable(file_dep[0])
        labels = kinematics.getColumnLabels()
        angles = np.ndarray(shape=(100, len(labels), len(self.trial.cycles)))     
        for icycle, cycle in enumerate(self.trial.cycles):
            istart = kinematics.getNearestRowIndexForTime(cycle.start)
            iend = kinematics.getNearestRowIndexForTime(cycle.end)+1
            time = kinematics.getIndependentColumn()[istart:iend]
            timeInterp = np.linspace(cycle.start, cycle.end, 100)
            for ilabel, label in enumerate(labels):
                col = kinematics.getDependentColumn(label).to_numpy()[istart:iend]
                colInterp = np.interp(timeInterp, time, col)
                angles[:, ilabel, icycle] = colInterp

        angles_std = np.std(angles, axis=2)
        angles_std_mean = np.mean(angles_std, axis=0)
        df = pd.DataFrame(data=angles_std_mean, index=labels)
        df.to_csv(target[0])


class TaskTrimTrackingData(osp.TrialTask):
    REGISTRY = []
    def __init__(self, trial, ik_setup_task, id_setup_task, initial_time, final_time):
        super(TaskTrimTrackingData, self).__init__(trial)
        self.name = trial.id + '_trim_tracking_data'
        self.trial = trial
        self.ik_solution_fpath = ik_setup_task.solution_fpath
        self.extloads_fpath = id_setup_task.results_extloads_fpath
        self.grf_fpath = os.path.join(trial.results_exp_path, 'expdata', 
            'ground_reaction.mot')
        self.initial_time = initial_time
        self.final_time = final_time

        expdata_dir = os.path.join(trial.results_exp_path, 'tracking_data', 'expdata')
        extloads_dir = os.path.join(trial.results_exp_path, 'tracking_data', 'extloads')
        self.tracking_coordinates_fpath = os.path.join(expdata_dir, 'coordinates.sto')
        self.tracking_extloads_fpath = os.path.join(extloads_dir, 'external_loads.xml')
        self.tracking_grfs_fpath = os.path.join(expdata_dir, 'ground_reaction.mot')

        if not os.path.exists(expdata_dir): os.makedirs(expdata_dir)
        if not os.path.exists(extloads_dir): os.makedirs(extloads_dir)

        self.add_action([self.ik_solution_fpath,
                         self.extloads_fpath,
                         self.grf_fpath],
                        [self.tracking_coordinates_fpath,
                         self.tracking_extloads_fpath,
                         self.tracking_grfs_fpath],
                        self.trim_tracking_data)

    def trim_tracking_data(self, file_dep, target):
        
        self.copy_file([file_dep[1]], [target[1]])

        sto = osim.STOFileAdapter()
        kinematics = osim.TimeSeriesTable(file_dep[0])
        kinematics.trim(self.initial_time, self.final_time)
        sto.write(kinematics, target[0])

        grfs = osim.TimeSeriesTable(file_dep[2])
        grfs.trim(self.initial_time, self.final_time)
        sto.write(grfs, target[2])


class TaskMocoUnperturbedWalkingGuess(osp.TrialTask):
    REGISTRY = []
    def __init__(self, trial, initial_time, final_time, mesh_interval=0.02,
                walking_speed=1.25, constrain_average_speed=True, guess_fpath=None, 
                constrain_initial_state=False, periodic=True, 
                costs_enabled=True, pelvis_boundary_conditions=True, **kwargs):
        super(TaskMocoUnperturbedWalkingGuess, self).__init__(trial)
        config_name = f'unperturbed_guess_mesh{int(1000*mesh_interval)}'
        if not costs_enabled: config_name += f'_costsDisabled'
        if periodic: config_name += f'_periodic'
        self.config_name = config_name
        self.name = trial.subject.name + '_moco_' + config_name
        self.initial_time = initial_time
        self.final_time = final_time
        self.mesh_interval = mesh_interval
        self.walking_speed = walking_speed
        self.root_dir = trial.study.config['doit_path']
        self.constrain_initial_state = constrain_initial_state
        self.constrain_average_speed = constrain_average_speed
        self.periodic = periodic
        self.costs_enabled = costs_enabled
        self.pelvis_boundary_conditions = pelvis_boundary_conditions
        self.guess_fpath = guess_fpath

        expdata_dir = os.path.join(trial.results_exp_path, 'tracking_data', 'expdata')
        extloads_dir = os.path.join(trial.results_exp_path, 'tracking_data', 'extloads')
        self.tracking_coordinates_fpath = os.path.join(expdata_dir, 'coordinates.sto')
        self.coordinates_std_fpath = os.path.join(trial.results_exp_path, 
            f'{trial.id}_joint_angle_standard_deviations.csv')
        self.tracking_extloads_fpath = os.path.join(extloads_dir, 'external_loads.xml')
        self.tracking_grfs_fpath = os.path.join(expdata_dir, 'ground_reaction.mot')

        self.result_fpath = os.path.join(self.study.config['results_path'],
            'guess', trial.subject.name)
        if not os.path.exists(self.result_fpath): os.makedirs(self.result_fpath)

        self.archive_fpath = os.path.join(self.study.config['results_path'],
            'guess', trial.subject.name, 'archive')
        if not os.path.exists(self.archive_fpath): os.makedirs(self.archive_fpath)

        self.grf_fpath = os.path.join(trial.results_exp_path, 'expdata', 
            'ground_reaction.mot')
        self.emg_fpath = os.path.join(trial.results_exp_path, 'expdata', 
            'emg.sto')

        self.add_action([trial.subject.scaled_model_fpath,
                         self.tracking_coordinates_fpath,
                         self.coordinates_std_fpath,
                         self.tracking_extloads_fpath,
                         self.tracking_grfs_fpath,
                         self.emg_fpath],
                        [os.path.join(self.result_fpath, 
                            self.config_name + '.sto')],
                        self.run_tracking_problem)

    def run_tracking_problem(self, file_dep, target):

        weights = {
            'state_tracking_weight'  : 1e-2,
            'control_weight'         : 1e-2,
            'grf_tracking_weight'    : 1e-4,
            'com_tracking_weight'    : 1e-2,
            'base_of_support_weight' : 0,
            'head_accel_weight'      : 0,
            'upright_torso_weight'   : 0,
            'torso_tracking_weight'  : 0,
            'foot_tracking_weight'   : 0, 
            'pelvis_tracking_weight' : 0,
            'aux_deriv_weight'       : 1e-3,
            'metabolics_weight'      : 0,
            'accel_weight'           : 1e-3,
            'regularization_weight'  : 0
        }

        config = MocoTrackConfig(
            self.config_name, self.config_name, 'black', weights,
            constrain_average_speed=self.constrain_average_speed,
            constrain_initial_state=self.constrain_initial_state,
            pelvis_boundary_conditions=self.pelvis_boundary_conditions,
            periodic=self.periodic,
            guess=self.guess_fpath,
            effort_enabled=self.costs_enabled,
            tracking_enabled=self.costs_enabled
            )

        cycles = list()
        for cycle in self.trial.cycles:
            cycles.append([cycle.start, cycle.end])

        result = MotionTrackingWalking(
            self.root_dir, # root directory
            self.result_fpath, # result directory
            file_dep[0], # model file path
            file_dep[1], # IK coordinates path
            file_dep[2], # Coordinates standard deviations
            file_dep[3], # external loads file 
            file_dep[4], # GRF MOT file
            file_dep[5], # EMG data
            self.initial_time,
            self.final_time, 
            cycles,
            self.trial.right_strikes,
            self.trial.left_strikes,
            self.mesh_interval, 
            self.walking_speed,
            [config],
        )

        result.generate_results(self.result_fpath)
        result.report_results(self.result_fpath)


class TaskMocoUnperturbedWalking(osp.TrialTask):
    REGISTRY = []
    def __init__(self, trial, initial_time, final_time, mesh_interval=0.02,
                 walking_speed=1.25, constrain_average_speed=False,
                 guess_fpath=None, track_grfs=True, 
                 constrain_initial_state=False,
                 periodic=True, two_cycles=False, **kwargs):
        super(TaskMocoUnperturbedWalking, self).__init__(trial)
        suffix = '_two_cycles' if two_cycles else ''
        self.config_name = f'unperturbed{suffix}_mesh{int(1000*mesh_interval)}'
        self.name = f'{trial.subject.name}_moco_{self.config_name}'
        self.initial_time = initial_time
        self.final_time = final_time
        self.mesh_interval = mesh_interval
        self.walking_speed = walking_speed
        self.constrain_average_speed = constrain_average_speed
        self.guess_fpath = guess_fpath
        self.track_grfs = track_grfs
        self.root_dir = trial.study.config['doit_path']
        self.constrain_initial_state = constrain_initial_state
        self.periodic = periodic
        self.weights = trial.study.weights

        expdata_dir = os.path.join(
            trial.results_exp_path, 'tracking_data', 'expdata')
        extloads_dir = os.path.join(
            trial.results_exp_path, 'tracking_data', 'extloads')
        self.tracking_coordinates_fpath = os.path.join(
            expdata_dir, 'coordinates.sto')
        self.coordinates_std_fpath = os.path.join(
            trial.results_exp_path, 
            f'{trial.id}_joint_angle_standard_deviations.csv')
        self.tracking_extloads_fpath = os.path.join(
            extloads_dir, 'external_loads.xml')
        self.tracking_grfs_fpath = os.path.join(
            expdata_dir, 'ground_reaction.mot')

        self.result_fpath = os.path.join(
            self.study.config['results_path'], f'unperturbed{suffix}', 
            trial.subject.name)
        if not os.path.exists(self.result_fpath): 
            os.makedirs(self.result_fpath)

        self.archive_fpath = os.path.join(
            self.study.config['results_path'], f'unperturbed{suffix}', 
            trial.subject.name, 'archive')
        if not os.path.exists(self.archive_fpath): 
            os.makedirs(self.archive_fpath)

        self.grf_fpath = os.path.join(
            trial.results_exp_path, 'expdata', 'ground_reaction.mot')
        self.emg_fpath = os.path.join(
            trial.results_exp_path, 'expdata', 'emg.sto')

        self.coordinates_std_fpath = os.path.join(
            trial.results_exp_path, 
            f'{trial.id}_joint_angle_standard_deviations.csv')

        self.add_action([trial.subject.scaled_model_fpath,
                         self.tracking_coordinates_fpath,
                         self.coordinates_std_fpath,
                         self.tracking_extloads_fpath,
                         self.tracking_grfs_fpath,
                         self.emg_fpath],
                        [os.path.join(
                            self.result_fpath, 
                            self.config_name + '.sto')],
                        self.run_tracking_problem)

    def run_tracking_problem(self, file_dep, target):

        weights = copy.deepcopy(self.weights)
        if not self.track_grfs:    
            weights['grf_tracking_weight'] = 0.0

        config = MocoTrackConfig(
            self.config_name, self.config_name, 'black', weights,
            constrain_average_speed=self.constrain_average_speed,
            constrain_initial_state=self.constrain_initial_state,
            periodic=self.periodic,
            guess=self.guess_fpath,
            )

        cycles = list()
        for cycle in self.trial.cycles:
            cycles.append([cycle.start, cycle.end])

        result = MotionTrackingWalking(
            self.root_dir,      # root directory
            self.result_fpath,  # result directory
            file_dep[0],        # model file path
            file_dep[1],        # IK coordinates path
            file_dep[2],        # Coordinates standard deviations
            file_dep[3],        # external loads file 
            file_dep[4],        # GRF MOT file
            file_dep[5],        # EMG data
            self.initial_time,
            self.final_time, 
            cycles,
            self.trial.right_strikes,
            self.trial.left_strikes,
            self.mesh_interval, 
            self.walking_speed,
            [config])

        result.generate_results(self.result_fpath)
        result.report_results(self.result_fpath)


class TaskMocoDoublePeriodicTrajectory(osp.TrialTask):
    REGISTRY = []
    def __init__(self, trial, traj_fpath, **kwargs):
        super(TaskMocoDoublePeriodicTrajectory, self).__init__(trial)
        self.name = f'{trial.id}_moco_double_periodic_trajectory'
        self.traj_fpath = traj_fpath
        self.double_traj_fpath = traj_fpath.replace(
            '.sto', '_doubled.sto')

        self.add_action([self.traj_fpath],
                        [self.double_traj_fpath],
                        self.double_periodic_trajectory)

    def double_periodic_trajectory(self, file_dep, target):
        traj = osim.MocoTrajectory(file_dep[0])
        double_traj = osim.createPeriodicTrajectory(traj)
        double_traj.write(target[0])


class TaskMocoAnkleTorquePerturbedWalking(osp.TrialTask):
    REGISTRY = []
    def __init__(self, trial, initial_time, final_time, right_strikes, 
                 left_strikes, guess_fpath=None, mesh_interval=0.02, 
                 walking_speed=1.25, constrain_initial_state=True, 
                 bound_controls=True, side='right', two_cycles=False,
                 torque_parameters=[0.5, 0.5, 0.25, 0.1],
                 perturb_response_delay=0.0, periodic=False):
        super(TaskMocoAnkleTorquePerturbedWalking, self).__init__(trial)
        torque = int(100*torque_parameters[0])
        time = int(100*torque_parameters[1])
        delay = int(1000*perturb_response_delay)
        suffix = '_two_cycles' if two_cycles else ''
        self.config_name = (f'perturb{suffix}_torque{torque}'
                            f'_time{time}_delay{delay}')
        self.name = f'{trial.subject.name}_moco_{self.config_name}'
        self.suffix = suffix
        self.mesh_interval = mesh_interval
        self.walking_speed = walking_speed
        self.guess_fpath = guess_fpath
        self.root_dir = trial.study.config['doit_path']
        self.constrain_initial_state = constrain_initial_state
        self.bound_controls = bound_controls
        self.perturb_response_delay = perturb_response_delay
        self.weights = trial.study.weights
        self.side = side
        self.initial_time = initial_time
        self.final_time = final_time
        self.right_strikes = right_strikes
        self.left_strikes = left_strikes
        self.model_fpath = trial.subject.scaled_model_fpath
        self.periodic = periodic

        ankle_torque_left_parameters = list()
        ankle_torque_right_parameters = list()
        for rhs in self.right_strikes:
            ankle_torque_right_parameters.append(torque_parameters)
        for lhs in self.left_strikes:
            ankle_torque_left_parameters.append(torque_parameters)

        self.ankle_torque_left_parameters = ankle_torque_left_parameters
        self.ankle_torque_right_parameters = ankle_torque_right_parameters

        expdata_dir = os.path.join(
            trial.results_exp_path, 'tracking_data', 'expdata')
        extloads_dir = os.path.join(
            trial.results_exp_path, 'tracking_data', 'extloads')
        self.tracking_coordinates_fpath = os.path.join(
            expdata_dir, 'coordinates.sto')
        self.coordinates_std_fpath = os.path.join(
            trial.results_exp_path, 
            f'{trial.id}_joint_angle_standard_deviations.csv')
        self.tracking_extloads_fpath = os.path.join(
            extloads_dir, 'external_loads.xml')
        self.tracking_grfs_fpath = os.path.join(
            expdata_dir, 'ground_reaction.mot')
        self.emg_fpath = os.path.join(
            trial.results_exp_path, 'expdata', 'emg.sto')

        self.result_fpath = os.path.join(
            self.study.config['results_path'], self.config_name, 
            trial.subject.name)
        if not os.path.exists(self.result_fpath): 
            os.makedirs(self.result_fpath)

        self.archive_fpath = os.path.join(
            self.study.config['results_path'], self.config_name, 
            trial.subject.name, 'archive')
        if not os.path.exists(self.archive_fpath): 
            os.makedirs(self.archive_fpath)

        self.add_action([self.model_fpath,
                         self.tracking_coordinates_fpath,
                         self.coordinates_std_fpath,
                         self.tracking_extloads_fpath,
                         self.tracking_grfs_fpath,
                         self.emg_fpath,
                         self.guess_fpath], [],
                        self.run_tracking_problem)

    def run_tracking_problem(self, file_dep, target):
        config = MocoTrackConfig(
            self.config_name, self.config_name, 'black', self.weights,
            constrain_initial_state=self.constrain_initial_state,
            use_guess_for_initial_muscle_states=True,
            use_guess_for_initial_kinematic_states=True,
            bound_controls=self.bound_controls,
            control_bound_solution_fpath=file_dep[6],
            perturb_response_delay=self.perturb_response_delay, 
            perturb_start_sim_at_onset=True,
            guess=file_dep[6],
            ankle_torque_perturbation=True,
            ankle_torque_left_parameters=self.ankle_torque_left_parameters,
            ankle_torque_right_parameters=self.ankle_torque_right_parameters,
            ankle_torque_side=self.side,
            ankle_torque_first_cycle_only=True,
            periodic=self.periodic)

        cycles = list()
        for cycle in self.trial.cycles:
            cycles.append([cycle.start, cycle.end])

        result = MotionTrackingWalking(
            self.root_dir,      # root directory
            self.result_fpath,  # result directory
            file_dep[0],        # model file path
            file_dep[1],        # IK coordinates path
            file_dep[2],        # Coordinate STD path
            file_dep[3],        # external loads file 
            file_dep[4],        # GRF MOT file
            file_dep[5],        # EMG data
            self.initial_time,
            self.final_time, 
            cycles,
            self.right_strikes,
            self.left_strikes,
            self.mesh_interval, 
            self.walking_speed,
            [config],
        )

        result.generate_results(self.result_fpath)
        result.report_results(self.result_fpath)


class TaskMocoAnkleTorquePerturbedWalkingPost(osp.TrialTask):
    REGISTRY = []
    def __init__(self, trial, generate_task, **kwargs):
        super(TaskMocoAnkleTorquePerturbedWalkingPost, self).__init__(trial)
        self.name = f'{generate_task.name}_post'
        self.weights = trial.study.weights
        self.root_dir = trial.study.config['doit_path']
        self.walking_speed = generate_task.walking_speed
        self.mesh_interval = generate_task.mesh_interval
        self.guess_fpath = generate_task.guess_fpath
        self.result_fpath = generate_task.result_fpath
        self.archive_fpath = generate_task.archive_fpath
        self.model_fpath = generate_task.model_fpath
        self.tracking_coordinates_fpath = \
            generate_task.tracking_coordinates_fpath
        self.coordinates_std_fpath = generate_task.coordinates_std_fpath
        self.tracking_extloads_fpath = generate_task.tracking_extloads_fpath
        self.tracking_grfs_fpath = generate_task.tracking_grfs_fpath
        self.emg_fpath = generate_task.emg_fpath
        self.guess_fpath = generate_task.guess_fpath
        self.ankle_torque_left_parameters = \
            generate_task.ankle_torque_left_parameters
        self.ankle_torque_right_parameters = \
            generate_task.ankle_torque_right_parameters
        self.side = generate_task.side
        self.initial_time = generate_task.initial_time
        self.final_time = generate_task.final_time
        self.right_strikes = generate_task.right_strikes
        self.left_strikes = generate_task.left_strikes
        self.config_name = generate_task.config_name
        self.suffix = generate_task.suffix

        # Copy over unperturbed solution so we can plot against the
        # perturbed solution
        self.unperturbed_result_fpath = os.path.join(
            self.study.config['results_path'], 'unperturbed', 
            trial.subject.name)
        shutil.copyfile(
            os.path.join(self.unperturbed_result_fpath, 
                         f'unperturbed{self.suffix}_mesh20.sto'),
            os.path.join(self.result_fpath, f'unperturbed{self.suffix}.sto'))
        shutil.copyfile(
            os.path.join(self.unperturbed_result_fpath, 
                         f'unperturbed{self.suffix}_mesh20_grfs.sto'),
            os.path.join(self.result_fpath, 
                         f'unperturbed{self.suffix}_grfs.sto'))
        
        self.add_action([self.model_fpath,
                         self.tracking_coordinates_fpath,
                         self.coordinates_std_fpath,
                         self.tracking_extloads_fpath,
                         self.tracking_grfs_fpath,
                         self.emg_fpath,
                         self.guess_fpath],
                        [],
                        self.run_tracking_problem)

    def run_tracking_problem(self, file_dep, target):

        configs = list()
        config = MocoTrackConfig(
            'unperturbed', 'unperturbed', 'black', self.weights,
            guess=file_dep[5],
            )
        configs.append(config)

        config = MocoTrackConfig(
            self.config_name, self.config_name, 
            'red', self.weights,
            ankle_torque_right_parameters=self.ankle_torque_right_parameters,
            ankle_torque_left_parameters=self.ankle_torque_left_parameters,
            ankle_torque_side=self.side,
            )
        configs.append(config)

        cycles = list()
        for cycle in self.trial.cycles:
            cycles.append([cycle.start, cycle.end])

        result = MotionTrackingWalking(
            self.root_dir,      # root directory
            self.result_fpath,  # result directory
            file_dep[0],        # model file path
            file_dep[1],        # IK coordinates path
            file_dep[2],        # Coordinates STD path
            file_dep[3],        # external loads file 
            file_dep[4],        # GRF MOT file
            file_dep[5],        # EMG data
            self.initial_time,
            self.final_time,
            cycles,
            self.right_strikes,
            self.left_strikes,
            self.mesh_interval, 
            self.walking_speed,
            configs,
        )

        result.report_results(self.result_fpath)


class TaskMocoAnkleTorqueBaselineWalking(osp.TrialTask):
    REGISTRY = []
    def __init__(self, trial, initial_time, final_time, right_strikes, 
                 left_strikes, guess_fpath=None, mesh_interval=0.02, 
                 walking_speed=1.25, side='both',
                 constrain_initial_state=False,
                 torque_parameters=[0.5, 0.5, 0.25, 0.1],
                 periodic=False,
                 two_cycles=False, **kwargs):
        super(TaskMocoAnkleTorqueBaselineWalking, self).__init__(trial)
        suffix = '_two_cycles' if two_cycles else ''
        self.name = f'{trial.subject.name}_moco_baseline_ankle_torque{suffix}'
        self.mesh_interval = mesh_interval
        self.walking_speed = walking_speed
        self.periodic = periodic
        self.guess_fpath = guess_fpath
        self.root_dir = trial.study.config['doit_path']
        self.weights = trial.study.weights
        self.side = side
        self.initial_time = initial_time
        self.final_time = final_time
        self.right_strikes = right_strikes
        self.left_strikes = left_strikes
        self.model_fpath = trial.subject.scaled_model_fpath
        self.constrain_initial_state = constrain_initial_state

        ankle_torque_left_parameters = list()
        ankle_torque_right_parameters = list()
        for rhs in self.right_strikes:
            ankle_torque_right_parameters.append(torque_parameters)
        for lhs in self.left_strikes:
            ankle_torque_left_parameters.append(torque_parameters)

        self.ankle_torque_left_parameters = ankle_torque_left_parameters
        self.ankle_torque_right_parameters = ankle_torque_right_parameters

        expdata_dir = os.path.join(
            trial.results_exp_path, 'tracking_data', 'expdata')
        extloads_dir = os.path.join(
            trial.results_exp_path, 'tracking_data', 'extloads')
        self.tracking_coordinates_fpath = os.path.join(
            expdata_dir, 'coordinates.sto')
        self.coordinates_std_fpath = os.path.join(
            trial.results_exp_path, 
            f'{trial.id}_joint_angle_standard_deviations.csv')
        self.tracking_extloads_fpath = os.path.join(
            extloads_dir, 'external_loads.xml')
        self.tracking_grfs_fpath = os.path.join(
            expdata_dir, 'ground_reaction.mot')

        self.result_fpath = os.path.join(
            self.study.config['results_path'],
            f'baseline_torque{suffix}', trial.subject.name)
        if not os.path.exists(self.result_fpath): 
            os.makedirs(self.result_fpath)

        self.archive_fpath = os.path.join(
            self.study.config['results_path'],
            f'baseline_torque{suffix}', trial.subject.name, 'archive')
        if not os.path.exists(self.archive_fpath): 
            os.makedirs(self.archive_fpath)

        self.grf_fpath = os.path.join(
            trial.results_exp_path, 'expdata', 'ground_reaction.mot')
        self.emg_fpath = os.path.join(
            trial.results_exp_path, 'expdata', 'emg.sto')

        self.config_name = f'baseline_torque{suffix}'
        self.add_action([self.model_fpath,
                         self.tracking_coordinates_fpath,
                         self.coordinates_std_fpath,
                         self.tracking_extloads_fpath,
                         self.tracking_grfs_fpath,
                         self.emg_fpath],
                        [],
                        self.run_tracking_problem)

    def run_tracking_problem(self, file_dep, target):
        
        config = MocoTrackConfig(
            self.config_name, self.config_name, 'black', self.weights,
            guess=self.guess_fpath,
            constrain_initial_state=self.constrain_initial_state,
            use_guess_for_initial_muscle_states=self.constrain_initial_state,
            use_guess_for_initial_kinematic_states=self.constrain_initial_state,
            ankle_torque_perturbation=True,
            ankle_torque_left_parameters=self.ankle_torque_left_parameters,
            ankle_torque_right_parameters=self.ankle_torque_right_parameters,
            ankle_torque_first_cycle_only=False,
            ankle_torque_side=self.side,
            periodic=self.periodic)

        cycles = list()
        for cycle in self.trial.cycles:
            cycles.append([cycle.start, cycle.end])

        result = MotionTrackingWalking(
            self.root_dir,      # root directory
            self.result_fpath,  # result directory
            file_dep[0],        # model file path
            file_dep[1],        # IK coordinates path
            file_dep[2],        # Coord stds
            file_dep[3],        # external loads file 
            file_dep[4],        # GRF MOT file
            file_dep[5],        # EMG data
            self.initial_time,
            self.final_time, 
            cycles,
            self.right_strikes,
            self.left_strikes,
            self.mesh_interval, 
            self.walking_speed,
            [config],
        )

        result.generate_results(self.result_fpath)
        result.report_results(self.result_fpath)


class TaskMocoAnkleTorquePerturbedFromBaselineWalking(osp.TrialTask):
    REGISTRY = []
    def __init__(self, trial, initial_time, final_time, right_strikes, 
                 left_strikes, ankle_torque_left_parameters, 
                 ankle_torque_right_parameters, 
                 guess_fpath=None, mesh_interval=0.02, periodic=False,
                 walking_speed=1.25, side='both', constrain_initial_state=True, 
                 perturb_response_delay=0.0, bound_controls=True,
                 two_cycles=False):
        super(TaskMocoAnkleTorquePerturbedFromBaselineWalking, self).__init__(
            trial)
        parameters = ankle_torque_right_parameters[0]
        torque = int(100*parameters[0])
        time = int(100*parameters[1])
        delay = int(1000*perturb_response_delay)
        suffix = '_two_cycles' if two_cycles else ''
        self.config_name = (f'perturb_from_baseline{suffix}'
                            f'_torque{torque}_time{time}_delay{delay}')
        self.name = f'{trial.subject.name}_moco_{self.config_name}'
        self.mesh_interval = mesh_interval
        self.walking_speed = walking_speed
        self.periodic = periodic
        self.guess_fpath = guess_fpath
        self.root_dir = trial.study.config['doit_path']
        self.weights = trial.study.weights
        self.ankle_torque_left_parameters = ankle_torque_left_parameters
        self.ankle_torque_right_parameters = ankle_torque_right_parameters
        self.side = side
        self.initial_time = initial_time
        self.final_time = final_time
        self.right_strikes = right_strikes
        self.left_strikes = left_strikes
        self.model_fpath = trial.subject.scaled_model_fpath
        self.constrain_initial_state = constrain_initial_state
        self.bound_controls = bound_controls
        self.perturb_response_delay = perturb_response_delay

        expdata_dir = os.path.join(
            trial.results_exp_path, 'tracking_data', 'expdata')
        extloads_dir = os.path.join(
            trial.results_exp_path, 'tracking_data', 'extloads')
        self.tracking_coordinates_fpath = os.path.join(
            expdata_dir, 'coordinates.sto')
        self.coordinates_std_fpath = os.path.join(
            trial.results_exp_path, 
            f'{trial.id}_joint_angle_standard_deviations.csv')
        self.tracking_extloads_fpath = os.path.join(
            extloads_dir, 'external_loads.xml')
        self.tracking_grfs_fpath = os.path.join(
            expdata_dir, 'ground_reaction.mot')

        self.result_fpath = os.path.join(
            self.study.config['results_path'],
            self.config_name, 
            trial.subject.name)
        if not os.path.exists(self.result_fpath): 
            os.makedirs(self.result_fpath)

        self.archive_fpath = os.path.join(
            self.study.config['results_path'],
            self.config_name, 
            trial.subject.name, 'archive')
        if not os.path.exists(self.archive_fpath): 
            os.makedirs(self.archive_fpath)

        self.grf_fpath = os.path.join(
            trial.results_exp_path, 'expdata', 'ground_reaction.mot')
        self.emg_fpath = os.path.join(
            trial.results_exp_path, 'expdata', 'emg.sto')

        self.add_action([self.model_fpath,
                         self.tracking_coordinates_fpath,
                         self.coordinates_std_fpath,
                         self.tracking_extloads_fpath,
                         self.tracking_grfs_fpath,
                         self.emg_fpath],
                        [],
                        self.run_tracking_problem)

    def run_tracking_problem(self, file_dep, target):

        config = MocoTrackConfig(
            self.config_name, self.config_name, 'black', self.weights,
            guess=self.guess_fpath,
            constrain_initial_state=self.constrain_initial_state,
            use_guess_for_initial_muscle_states=True,
            use_guess_for_initial_kinematic_states=True,
            bound_controls=self.bound_controls,
            control_bound_solution_fpath=self.guess_fpath,
            perturb_response_delay=self.perturb_response_delay,
            perturb_start_sim_at_onset=True,
            ankle_torque_perturbation=True,
            ankle_torque_left_parameters=self.ankle_torque_left_parameters,
            ankle_torque_right_parameters=self.ankle_torque_right_parameters,
            ankle_torque_first_cycle_only=False,
            ankle_torque_side=self.side,
            periodic=False)

        cycles = list()
        for cycle in self.trial.cycles:
            cycles.append([cycle.start, cycle.end])

        result = MotionTrackingWalking(
            self.root_dir,      # root directory
            self.result_fpath,  # result directory
            file_dep[0],        # model file path
            file_dep[1],        # IK coordinates path
            file_dep[2],        # Coord stds
            file_dep[3],        # external loads file 
            file_dep[4],        # GRF MOT file
            file_dep[5],        # EMG data
            self.initial_time,
            self.final_time, 
            cycles,
            self.right_strikes,
            self.left_strikes,
            self.mesh_interval, 
            self.walking_speed,
            [config],
        )

        result.generate_results(self.result_fpath)
        result.report_results(self.result_fpath)


class TaskBodyKinematicsSetup(osp.SetupTask):
    REGISTRY = []
    def __init__(self, trial, ik_setup_task, **kwargs):
        super(TaskBodyKinematicsSetup, self).__init__('body_kinematics', 
            trial, **kwargs)
        self.doc = 'Create a setup file for a BodyKinematics analysis.'
        self.ik_setup_task = ik_setup_task
        self.rel_kinematics_fpath = os.path.relpath(
            ik_setup_task.solution_fpath, self.path)
        self.solution_fpath = os.path.join(
            self.path, 'results',
            '%s_%s_body_kinematics_BodyKinematics_pos_global.sto' % (
                self.study.name, trial.id))

        # Fill out setup.xml template and write to results directory
        self.create_setup_action()

    def fill_setup_template(self, file_dep, target,
                            init_time=None, final_time=None):
        with open(file_dep[0]) as ft:
            content = ft.read()
            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@NAME@', self.tricycle.id)
            content = content.replace('@MODEL@',
                os.path.relpath(self.subject.scaled_model_fpath, self.path))
            content = content.replace('@COORDINATES_FILE@',
                self.rel_kinematics_fpath)
            content = content.replace('@INIT_TIME@', '%.4f' % init_time)
            content = content.replace('@FINAL_TIME@', '%.4f' % final_time)

        with open(target[0], 'w') as f:
            f.write(content)


class TaskBodyKinematics(osp.ToolTask):
    REGISTRY = []
    def __init__(self, trial, bodykin_setup_task, **kwargs):
        super(TaskBodyKinematics, self).__init__(bodykin_setup_task, 
            trial, exec_name='analyze', **kwargs)
        self.doc = "Run an OpenSim BodyKinematics analysis."
        self.ik_setup_task = bodykin_setup_task.ik_setup_task
        self.file_dep += [
                self.subject.scaled_model_fpath,
                bodykin_setup_task.results_setup_fpath,
                self.ik_setup_task.solution_fpath
                ]
        self.targets += [
                bodykin_setup_task.solution_fpath
                ]


class TaskValidateMarkerErrors(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, cond_names=['walk1','walk2','walk3','walk4']):
        super(TaskValidateMarkerErrors, self).__init__(study)
        suffix = ''
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction 
        self.name = 'validate_marker_errors%s' % suffix
        self.doc = 'Compute marker errors across subjects and conditions.'
        self.results_path = study.config['results_path']
        self.validate_path = os.path.join(study.config['validate_path'],
            'marker_errors')
        self.cond_names = cond_names
        self.subjects = study.subjects

        errors_fpaths = list()
        for cond_name in cond_names:
            for subject in study.subjects:
                errors_fpaths.append(os.path.join(self.results_path, 
                    'experiments', subject.name, cond_name, 'ik', 
                    'marker_error.csv'))
                
        val_fname = os.path.join(self.validate_path, 'marker_errors.txt')
        self.add_action(errors_fpaths, [val_fname],
                        self.validate_marker_errors)

    def validate_marker_errors(self, file_dep, target):
        if not os.path.isdir(self.validate_path): os.mkdir(self.validate_path)

        all_errors = pd.DataFrame()
        for file in file_dep:
            df = pd.read_csv(file)
            df = df.drop(columns=['Unnamed: 0', 'time'])
            all_errors = pd.concat([all_errors, df])

        numMarkers = all_errors.shape[1]
        all_errors_sq = all_errors.pow(2)
        all_errors_sq_sum_norm = all_errors_sq.sum(axis=1) / float(numMarkers)
        rms_errors = all_errors_sq_sum_norm.pow(0.5)

        peak_rms = rms_errors.max()
        mean_rms = rms_errors.mean()

        with open(target[0],"w") as f:
            f.write('subjects: ')
            for isubj, subject in enumerate(self.subjects):
                if isubj:
                    f.write(', %s' % subject.name)
                else:
                    f.write('%s' % subject.name)
            f.write('\n')
            f.write('conditions: ')
            for icond, cond_name in enumerate(self.cond_names):
                if icond:
                    f.write(', %s' % subject)
                else:
                    f.write('%s' % cond_name)
            f.write('\n')
            f.write('Peak RMS marker error: %1.3f cm \n' % (100*peak_rms))
            f.write('Mean RMS marker error: %1.3f cm \n' % (100*mean_rms))


class TaskValidateKinetics(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, cond_name='walk2',
            gen_forces=['hip_flexion_r_moment', 
                        'knee_angle_r_moment',
                        'ankle_angle_r_moment'],
            residual_moments=['pelvis_tilt_moment', 
                              'pelvis_list_moment',
                              'pelvis_rotation_moment'],
            residual_forces=['pelvis_tx_force',
                             'pelvis_ty_force',
                             'pelvis_tz_force'],
            grfs=['ground_force_r_vx',
                  'ground_force_r_vy', 
                  'ground_force_r_vz']):
        super(TaskValidateKinetics, self).__init__(study)
        suffix = ''
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction 
        self.name = 'validate_kinetics%s' % suffix
        self.doc = 'Validate joint moments and residuals across subjects and conditions.'
        self.results_path = study.config['results_path']
        self.validate_path = os.path.join(study.config['validate_path'],
            'kinetics')
        self.figure_path = os.path.join(study.config['figures_path'], 
            'figureS1')
        self.cond_name = cond_name
        self.subjects = study.subjects
        self.gen_forces = gen_forces
        if not len(residual_moments) == 3:
            raise Exception('There must be 3 residual moment components.')
        self.residual_moments = residual_moments
        if not len(residual_forces) == 3:
            raise Exception('There must be 3 residual force components.')
        self.residual_forces = residual_forces
        if not len(grfs) == 3:
            raise Exception('There must be 3 ground reaction force components.')
        self.grfs = grfs

        colors = plt.cm.jet(np.linspace(0, 1, len(study.subjects)))
        masses = list()
        for subject in study.subjects:
            masses.append(subject.mass)

        moments_fpaths_all = list()
        bodykin_fpaths_all = list()
        gait_events_all = list()
        isubjs_all = list()
        grf_fpaths_all = list()

        moments_fpaths = list()
        gait_events = list()
        isubjs = list()
        for isubj, subject in enumerate(study.subjects):
            fpath = os.path.join(self.results_path, 'experiments', 
                subject.name, cond_name, 'id', 'results', 
                '%s_%s_%s_id_solution.sto' % (study.name, subject.name, 
                    cond_name))
            moments_fpaths.append(fpath)
            moments_fpaths_all.append(fpath)

            bodykin_fpath = os.path.join(self.results_path, 'experiments',
                subject.name, cond_name, 'body_kinematics', 'results',
                '%s_%s_%s_body_kinematics_BodyKinematics_pos_global.sto' % (
                    study.name, subject.name, cond_name))
            bodykin_fpaths_all.append(bodykin_fpath)

            condition = subject.get_condition(cond_name)
            trial = condition.get_trial(1)
            cycles = trial.get_cycles()
            gait_events_this_cond = list()
            for cycle in cycles:
                gait_events_this_cond.append((cycle.start, cycle.end))
            gait_events.append(gait_events_this_cond)
            gait_events_all.append(gait_events_this_cond)
            isubjs.append(isubj)
            isubjs_all.append(isubj)

            grf_fpath = os.path.join(self.results_path, 'experiments', 
                subject.name, cond_name, 'expdata', 'ground_reaction.mot')
            grf_fpaths_all.append(grf_fpath)

        joint_moments_fname = os.path.join(self.validate_path, 
                'joint_moments_%s.pdf' % cond_name)
        joint_moments_figname = os.path.join(self.figure_path, 
                'figureS1.pdf')
        self.add_action(moments_fpaths, 
                        [joint_moments_fname, joint_moments_figname], 
                        self.validate_joint_moments, gait_events, isubjs, 
                        colors, masses)

        residuals_fname = os.path.join(self.validate_path, 'residuals.txt')
        moments_fname = os.path.join(self.validate_path, 'moments.csv')
        forces_fname = os.path.join(self.validate_path, 'forces.csv')
        self.add_action(moments_fpaths_all, 
                        [residuals_fname, moments_fname, forces_fname],
                        self.validate_residuals, gait_events_all, grf_fpaths_all,
                        bodykin_fpaths_all)

    def validate_joint_moments(self, file_dep, target, gait_events, isubjs,
            colors, masses):
        if not os.path.isdir(self.validate_path): os.mkdir(self.validate_path)
        if not os.path.isdir(self.figure_path): os.mkdir(self.figure_path)

        nice_gen_force_names = {'hip_flexion_r_moment': 'hip flexion moment',
                                'knee_angle_r_moment': 'knee flexion moment',
                                'ankle_angle_r_moment': 'ankle plantarflexion moment'}

        y_lim_dict = {'hip_flexion_r_moment': (-1.2, 1.2),
                       'knee_angle_r_moment': (-1.2, 1.2),
                       'ankle_angle_r_moment': (-0.5, 2.0)}

        subject_labels = ['subject 1', 'subject 2', 'subject 3', 'subject 4',
                          'subject 5']

        import matplotlib
        fig = pl.figure(figsize=(6, 8))
        ind_axes = list()
        ind_handles = list()
        mean_axes = list()
        dfs = list()
        for iforce, gen_force in enumerate(self.gen_forces):
            idx = iforce + 1
            ind_axes.append(fig.add_subplot(len(self.gen_forces), 2, 2*idx-1))
            mean_axes.append(fig.add_subplot(len(self.gen_forces), 2, 2*idx))
            dfs.append(pd.DataFrame())

        for file, gait_events_cond, isubj in zip(file_dep, gait_events, isubjs):
            id = pp.storage2numpy(file)
            time = id['time']
            for ige, gait_event in enumerate(gait_events_cond):
                start = np.argmin(abs(time-gait_event[0]))
                end = np.argmin(abs(time-gait_event[1]))
                new_time = np.linspace(time[start], time[end], 101)
                pgc = np.linspace(0, 100, 101)
                for iforce, gen_force in enumerate(self.gen_forces):
                    force = id[gen_force][start:end] / masses[isubj]                    
                    force_interp = np.interp(new_time, time[start:end], force)
                    dfs[iforce] = pd.concat([dfs[iforce], 
                            pd.DataFrame(force_interp)], axis=1)

                    sign = -1 if gen_force=='ankle_angle_r_moment' else 1
                    h, = ind_axes[iforce].plot(pgc, sign*force_interp, color=colors[isubj])
                    if not ige and not iforce:
                        ind_handles.append(h)
                    ind_axes[iforce].set_ylabel(
                        '%s (N-m/kg)' % nice_gen_force_names[gen_force])
                    ind_axes[iforce].set_ylim(y_lim_dict[gen_force])
                    ind_axes[iforce].set_xlim((0, 100))
                    ind_axes[iforce].axhline(ls='--', color='lightgray', zorder=0)
                    ind_axes[iforce].spines['top'].set_visible(False)
                    ind_axes[iforce].spines['right'].set_visible(False)
                    ind_axes[iforce].tick_params(direction='in')


        ind_axes[0].legend(ind_handles, subject_labels, fancybox=False,
                frameon=False, prop={'size': 8}, loc=2)                    

        for iforce, gen_force in enumerate(self.gen_forces):
            sign = -1 if gen_force=='ankle_angle_r_moment' else 1
            force_mean = sign*dfs[iforce].mean(axis=1)
            force_std = dfs[iforce].std(axis=1)
            mean_axes[iforce].fill_between(pgc, force_mean-force_std, 
                force_mean+force_std, color='black', alpha=0.2)
            std_h = matplotlib.patches.Patch(color='black', alpha=0.2)
            mean_h, = mean_axes[iforce].plot(pgc, force_mean, color='black')
            mean_axes[iforce].set_ylim(y_lim_dict[gen_force])
            mean_axes[iforce].set_xlim((0, 100))
            mean_axes[iforce].axhline(ls='--', color='lightgray', zorder=0)
            mean_axes[iforce].spines['top'].set_visible(False)
            mean_axes[iforce].spines['right'].set_visible(False)
            mean_axes[iforce].tick_params(direction='in')
            mean_axes[iforce].set_yticklabels([])
            if iforce == len(self.gen_forces)-1:
                mean_axes[iforce].set_xlabel('gait cycle (%)')
                ind_axes[iforce].set_xlabel('gait cycle (%)')
            else:
                ind_axes[iforce].set_xticklabels([])
                mean_axes[iforce].set_xticklabels([])

        mean_axes[0].legend([mean_h, std_h], ['mean', 'std'], fancybox=False,
                frameon=False, prop={'size': 8}, loc=2)  

        fig.tight_layout()
        fig.savefig(target[0])
        fig.savefig(target[0].replace('.pdf', '.png'), dpi=600)
        fig.savefig(target[1])
        fig.savefig(target[1].replace('.pdf', '.png'), dpi=600)
        pl.close(fig)

    def validate_residuals(self, file_dep, target, gait_events, grf_fpaths,
            bodykin_fpaths):
        if not os.path.isdir(self.validate_path): os.mkdir(self.validate_path)

        df_moments = pd.DataFrame()
        df_forces = pd.DataFrame()
        df_grfs = pd.DataFrame()
        df_ycom = pd.DataFrame()
        for file, gait_events_cond, grf_fpath, bodykin_fpath in zip(file_dep, 
                gait_events, grf_fpaths, bodykin_fpaths):
            id = pp.storage2numpy(file)
            time = id['time']
            grfs = pp.storage2numpy(grf_fpath)
            grf_time = grfs['time']
            bodykin = pp.storage2numpy(bodykin_fpath)
            bodykin_time = bodykin['time']
            for gait_event in gait_events_cond:
                start = np.argmin(abs(time-gait_event[0]))
                end = np.argmin(abs(time-gait_event[1]))
                new_time = np.linspace(time[start], time[end], 101)
                for residual_moment in self.residual_moments:
                    moment = id[residual_moment][start:end] 
                    moment_interp = np.interp(new_time, time[start:end], moment)
                    df_moments = pd.concat([df_moments, 
                        pd.DataFrame(moment_interp)], axis=1)

                for residual_force in self.residual_forces:
                    force = id[residual_force][start:end]
                    force_interp = np.interp(new_time, time[start:end], force)
                    df_forces = pd.concat([df_forces, 
                        pd.DataFrame(force_interp)], axis=1)

                grf_start = np.argmin(abs(grf_time-gait_event[0]))
                grf_end = np.argmin(abs(grf_time-gait_event[1]))
                new_grf_time = np.linspace(grf_time[grf_start], 
                        grf_time[grf_end], 101)
                for grf in self.grfs:
                    reaction = grfs[grf][grf_start:grf_end]
                    reaction_interp = np.interp(new_grf_time, 
                        grf_time[grf_start:grf_end], reaction)
                    df_grfs = pd.concat([df_grfs, pd.DataFrame(reaction_interp)], 
                        axis=1)

                bodykin_start = np.argmin(abs(bodykin_time-gait_event[0]))
                bodykin_end = np.argmin(abs(bodykin_time-gait_event[1]))
                new_bodykin_time = np.linspace(bodykin_time[bodykin_start],
                    bodykin_time[bodykin_end], 101)
                ycom = bodykin['center_of_mass_Y'][bodykin_start:bodykin_end]
                ycom_interp = np.interp(new_bodykin_time, 
                    bodykin_time[bodykin_start:bodykin_end], ycom)
                df_ycom = pd.concat([df_ycom, pd.DataFrame(ycom_interp)], axis=1)

        df_moments = pd.DataFrame(np.vstack(np.split(df_moments, 3*len(file_dep), 
            axis=1)))
        df_forces = pd.DataFrame(np.vstack(np.split(df_forces, 3*len(file_dep), 
            axis=1)))
        df_grfs = pd.DataFrame(np.vstack(np.split(df_grfs, 3*len(file_dep), 
            axis=1)))
        df_ycom = pd.DataFrame(np.vstack(np.split(df_ycom, 3*len(file_dep), 
            axis=1)))

        mag_moments = np.linalg.norm(df_moments, axis=1)
        mag_forces = np.linalg.norm(df_forces, axis=1)
        mag_grfs = np.linalg.norm(df_grfs, axis=1)
        peak_grfs = mag_grfs.max()
        avg_ycom = df_ycom.mean()[0]

        peak_moments = 100*mag_moments.max() / (peak_grfs * avg_ycom)
        rms_moments = 100*np.sqrt(np.mean(np.square(mag_moments))) / (
            peak_grfs * avg_ycom)
        peak_forces = 100*mag_forces.max() / peak_grfs
        rms_forces = 100*np.sqrt(np.mean(np.square(mag_forces))) / peak_grfs

        with open(target[0],"w") as f:
            f.write('subjects: ')
            for isubj, subject in enumerate(self.subjects):
                if isubj:
                    f.write(', %s' % subject.name)
                else:
                    f.write('%s' % subject.name)
            f.write('\n')
            f.write('condition: %s', self.cond_name)
            f.write('\n')
            f.write('Peak residual moments (%% GRF): %1.3f \n' % peak_moments) 
            f.write('RMS residual moments (%% GRF): %1.3f \n' % rms_moments)
            f.write('Peak residual forces (%% GRF): %1.3f \n' % peak_forces) 
            f.write('RMS residual forces (%% GRF): %1.3f \n' % rms_forces)

        target_dir = os.path.dirname(target[1])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with open(target[1], 'w') as f:
            f.write('residual moments\n')
            df_moments.to_csv(f, line_terminator='\n')

        target_dir = os.path.dirname(target[2])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with open(target[2], 'w') as f:
            f.write('residual forces\n')
            df_forces.to_csv(f, line_terminator='\n')


class TaskValidateKinematics(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, cond_name='walk2',
            joint_angles=['hip_flexion_r', 
                          'knee_angle_r',
                          'ankle_angle_r']):
        super(TaskValidateKinematics, self).__init__(study)
        suffix = ''
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction 
        self.name = 'validate_kinematics%s' % suffix
        self.doc = 'Validate joint angles across subjects and conditions.'
        self.results_path = study.config['results_path']
        self.validate_path = os.path.join(study.config['validate_path'],
            'kinematics')
        self.figure_path = os.path.join(study.config['figures_path'], 
            'figureS2')
        self.cond_name = cond_name
        self.subjects = study.subjects
        self.joint_angles = joint_angles

        colors = plt.cm.jet(np.linspace(0, 1, len(study.subjects)))

        angles_fpaths = list()
        gait_events = list()
        isubjs = list()
        for isubj, subject in enumerate(study.subjects):
            fpath = os.path.join(self.results_path, 'experiments', 
                subject.name, cond_name, 'ik', 
                '%s_%s_%s_ik_solution.mot' % (study.name, subject.name, 
                    cond_name))
            angles_fpaths.append(fpath)

            condition = subject.get_condition(cond_name)
            trial = condition.get_trial(1)
            cycles = trial.get_cycles()
            gait_events_this_cond = list()
            for cycle in cycles:
                gait_events_this_cond.append((cycle.start, cycle.end))
            gait_events.append(gait_events_this_cond)
            isubjs.append(isubj)

        joint_angles_fname = os.path.join(self.validate_path, 
                'joint_angles_%s.pdf' % cond_name)
        joint_angles_figname = os.path.join(self.figure_path, 
                'figureS2.pdf')
        self.add_action(angles_fpaths, 
                        [joint_angles_fname, joint_angles_figname], 
                        self.validate_joint_angles, gait_events, isubjs, 
                        colors)

    def validate_joint_angles(self, file_dep, target, gait_events, isubjs,
            colors):
        if not os.path.isdir(self.validate_path): os.mkdir(self.validate_path)
        if not os.path.isdir(self.figure_path): os.mkdir(self.figure_path)

        nice_joint_angle_names = {'hip_flexion_r': 'hip flexion angle',
                                  'knee_angle_r': 'knee flexion angle',
                                  'ankle_angle_r': 'ankle plantarflexion angle'}

        y_lim_dict = {'hip_flexion_r': (-40, 40),
                       'knee_angle_r': (-10, 80),
                       'ankle_angle_r': (-30, 30)}

        subject_labels = ['subject 1', 'subject 2', 'subject 3', 'subject 4',
                          'subject 5']

        import matplotlib
        fig = pl.figure(figsize=(6, 8))
        ind_axes = list()
        ind_handles = list()
        mean_axes = list()
        dfs = list()
        for iangle, joint_angle in enumerate(self.joint_angles):
            idx = iangle + 1
            ind_axes.append(fig.add_subplot(len(self.joint_angles), 2, 2*idx-1))
            mean_axes.append(fig.add_subplot(len(self.joint_angles), 2, 2*idx))
            dfs.append(pd.DataFrame())

        for file, gait_events_cond, isubj in zip(file_dep, gait_events, isubjs):
            ik = pp.storage2numpy(file)
            time = ik['time']
            for ige, gait_event in enumerate(gait_events_cond):
                start = np.argmin(abs(time-gait_event[0]))
                end = np.argmin(abs(time-gait_event[1]))
                new_time = np.linspace(time[start], time[end], 101)
                pgc = np.linspace(0, 100, 101)
                for iangle, joint_angle in enumerate(self.joint_angles):
                    angle = ik[joint_angle][start:end]                   
                    angle_interp = np.interp(new_time, time[start:end], angle)
                    dfs[iangle] = pd.concat([dfs[iangle], 
                            pd.DataFrame(angle_interp)], axis=1)

                    h, = ind_axes[iangle].plot(pgc, angle_interp, color=colors[isubj])
                    if not ige and not iangle:
                        ind_handles.append(h)
                    ind_axes[iangle].set_ylabel(
                        '%s (degrees)' % nice_joint_angle_names[joint_angle])
                    ind_axes[iangle].set_ylim(y_lim_dict[joint_angle])
                    ind_axes[iangle].set_xlim((0, 100))
                    ind_axes[iangle].axhline(ls='--', color='lightgray', zorder=0)
                    ind_axes[iangle].spines['top'].set_visible(False)
                    ind_axes[iangle].spines['right'].set_visible(False)
                    ind_axes[iangle].tick_params(direction='in')


        ind_axes[0].legend(ind_handles, subject_labels, fancybox=False,
                frameon=False, prop={'size': 8}, loc=9)                    

        for iangle, joint_angle in enumerate(self.joint_angles):
            angle_mean = dfs[iangle].mean(axis=1)
            angle_std = dfs[iangle].std(axis=1)
            mean_axes[iangle].fill_between(pgc, angle_mean-angle_std, 
                angle_mean+angle_std, color='black', alpha=0.2)
            std_h = matplotlib.patches.Patch(color='black', alpha=0.2)
            mean_h, = mean_axes[iangle].plot(pgc, angle_mean, color='black')
            mean_axes[iangle].set_ylim(y_lim_dict[joint_angle])
            mean_axes[iangle].set_xlim((0, 100))
            mean_axes[iangle].axhline(ls='--', color='lightgray', zorder=0)
            mean_axes[iangle].spines['top'].set_visible(False)
            mean_axes[iangle].spines['right'].set_visible(False)
            mean_axes[iangle].tick_params(direction='in')
            mean_axes[iangle].set_yticklabels([])
            if iangle == len(self.joint_angles)-1:
                mean_axes[iangle].set_xlabel('gait cycle (%)')
                ind_axes[iangle].set_xlabel('gait cycle (%)')
            else:
                ind_axes[iangle].set_xticklabels([])
                mean_axes[iangle].set_xticklabels([])

        mean_axes[0].legend([mean_h, std_h], ['mean', 'std'], fancybox=False,
                frameon=False, prop={'size': 8}, loc=9)  

        fig.tight_layout()
        fig.savefig(target[0])
        fig.savefig(target[0].replace('.pdf', '.png'), dpi=600)
        fig.savefig(target[1])
        fig.savefig(target[1].replace('.pdf', '.png'), dpi=600)
        pl.close(fig)


class TaskPlotCOMTrackingErrorsAnklePerturb(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, times, colormap, cmap_indices, delay, 
                torques=[25, 50, 75, 100]):
        super(TaskPlotCOMTrackingErrorsAnklePerturb, self).__init__(study)
        self.name = 'plot_com_tracking_errors_ankle_perturb'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'com_tracking_errors', 'ankle_perturb')
        if not os.path.exists(self.analysis_path): os.makedirs(self.analysis_path)
        self.subjects = subjects
        self.torques = torques
        self.times = times

        self.labels = list()
        self.labels.append('unperturbed')
        self.colors = list()
        self.colors.append('gray')
        cmap = plt.get_cmap(colormap)       
        deps = list()
        for subject in subjects:
            # Unperturbed solution
            deps.append(os.path.join(
                self.study.config['results_path'], 'unperturbed', 
                subject, 'tracking_unperturbed_mesh20_center_of_mass.sto'))
            # Perturbed solutions
            for time, cmap_idx in zip(self.times, cmap_indices):
                for torque in self.torques:
                    label = f'torque{torque}_time{time}_delay{int(1000*delay)}'
                    deps.append(
                        os.path.join(self.study.config['results_path'], 
                            f'ankle_perturb_{label}', subject, 
                            f'tracking_perturb_{label}_center_of_mass.sto')
                        )
                    self.labels.append(f'time: {time}%\ntorque: {torque}%')
                    self.colors.append(cmap(cmap_idx))

        self.add_action(deps, 
                        [os.path.join(self.analysis_path, 
                            'absolute_errors.png')], 
                        self.plot_com_tracking_errors)

    def plot_com_tracking_errors(self, file_dep, target):

        def compute_com_rmse(currTable, refTable):
            times = refTable.getIndependentColumn()
            labels = refTable.getColumnLabels()
            sumSquaredError = np.zeros_like(times)
            for itime, time in enumerate(times):
                for label in labels:
                    if 'position' in label:
                        currValue = currTable.getDependentColumn(label)[itime]
                        refValue = refTable.getDependentColumn(label)[itime]
                        error = currValue - refValue
                        sumSquaredError[itime] += error * error

            # Trapezoidal rule
            interval = times[-1] - times[0]
            isse = 0.5 * interval * (sumSquaredError.sum() + 
                                     sumSquaredError[1:-1].sum())

            rmse = np.sqrt(isse / interval / len(labels))  
            return rmse

        numSubjects = len(self.subjects)
        numLabels = len(self.labels)
        errors = np.zeros((numLabels, numSubjects))
        for isubj, subject in enumerate(self.subjects):
            refTable = osim.TimeSeriesTable(file_dep[isubj*numLabels])
            for ilabel, label in enumerate(self.labels):
                currTable = osim.TimeSeriesTable(
                    file_dep[isubj*numLabels + ilabel])
                errors[ilabel, isubj] = compute_com_rmse(currTable, refTable)

        # Remove elements with zero error
        errors = errors[1:, :]

        errors_mean = np.mean(errors, axis=1)
        errors_std = np.std(errors, axis=1)

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.bar(self.labels[1:], errors_mean, 0.75, 
            yerr=errors_std, color=self.colors[1:])
        ax.set_ylabel('center-of-mass tracking error (RMSE)')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', 
            rotation_mode='anchor')
        ax.set_xlabel('ankle perturbation')
        util.publication_spines(ax)
        fig.tight_layout()
        fig.savefig(target[0], dpi=600)
        plt.close()


class TaskPlotNormalizedImpulseAnklePerturb(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, times, colormap, cmap_indices, delay,
                torque=100):
        super(TaskPlotNormalizedImpulseAnklePerturb, self).__init__(study)
        self.name = f'plot_normalized_impulse_ankle_perturb'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'normalized_impulse', 'ankle_perturb')
        if not os.path.exists(self.analysis_path): os.makedirs(self.analysis_path)
        self.subjects = subjects
        self.times = times

        self.labels = list()
        self.labels.append('unperturbed')
        self.colors = list()
        self.colors.append('gray')
        self.start_pgc = list()
        self.stop_pgc = list()
        self.start_pgc.append(0)
        self.stop_pgc.append(100)
        cmap = plt.get_cmap(colormap)
        perturb_fpaths = list()
        for time, cmap_idx in zip(self.times, cmap_indices):
            label = f'torque{torque}_time{time}_delay{int(1000*delay)}'
            perturb_fpaths.append(
                os.path.join('ankle_perturb', label, 
                    f'tracking_perturb_{label}_center_of_mass.sto')
                )
            self.labels.append(f'{time}%')
            self.colors.append(cmap(cmap_idx))
            # self.start_pgc.append(time - 25)
            # self.stop_pgc.append(time + 10)
            self.start_pgc.append(0)
            self.stop_pgc.append(100)

        deps = list()
        for subject in subjects:
            condition_path = os.path.join(self.results_path, subject, 
                'unperturbed')
            # Unperturbed solution
            deps.append(os.path.join(condition_path, 'moco', 
                'tracking_unperturbed_center_of_mass.sto'))
            # Perturbed solutions
            for perturb_fpath in perturb_fpaths:
                deps.append(os.path.join(condition_path, perturb_fpath))

        self.add_action(deps, 
                        [os.path.join(self.analysis_path, 
                            'absolute_errors.png')
                        ], 
                        self.plot_com_tracking_errors)

    def plot_com_tracking_errors(self, file_dep, target):

        def compute_normalized_impulse(table, start_idx, stop_idx):
            times = np.array(table.getIndependentColumn())
            times = times[start_idx:stop_idx]
            pos_x = table.getDependentColumn('/|com_position_x').to_numpy()[start_idx:stop_idx]
            pos_y = table.getDependentColumn('/|com_position_y').to_numpy()[start_idx:stop_idx]
            pos_z = table.getDependentColumn('/|com_position_z').to_numpy()[start_idx:stop_idx]
            acc_x = table.getDependentColumn('/|com_acceleration_x').to_numpy()[start_idx:stop_idx]
            acc_y = table.getDependentColumn('/|com_acceleration_y').to_numpy()[start_idx:stop_idx]
            acc_z = table.getDependentColumn('/|com_acceleration_z').to_numpy()[start_idx:stop_idx]
            normAccel = np.zeros_like(times)
            for i, time in enumerate(times):
                normAccel[i] = np.sqrt(acc_x[i] * acc_x[i] + 
                                       acc_y[i] * acc_y[i] +
                                       acc_z[i] * acc_z[i])

            delta_x = pos_x[-1] - pos_x[0]
            delta_y = pos_y[-1] - pos_y[0]
            delta_z = pos_z[-1] - pos_z[0]
            deltaPos = np.sqrt(delta_x * delta_x +
                               delta_y * delta_y +
                               delta_z * delta_z)

            deltaT = times[-1] - times[0]
            avgVel = deltaPos / deltaT

            # Trapezoidal rule
            intNormAccel = 0.5 * deltaT * (normAccel.sum() + 
                                           normAccel[1:-1].sum())

            return intNormAccel / avgVel

        numSubjects = len(self.subjects)
        numLabels = len(self.labels)
        diff_norm_impulse = np.zeros((numLabels, numSubjects))
        for isubj, subject in enumerate(self.subjects):
            for ilabel, label in enumerate(self.labels):
                ref_table = osim.TimeSeriesTable(file_dep[isubj*numLabels])
                curr_table = osim.TimeSeriesTable(file_dep[isubj*numLabels + ilabel])
                times = np.array(curr_table.getIndependentColumn())
                duration = times[-1] - times[0]
                start_time = times[0] + duration * (self.start_pgc[ilabel] / 100.0)
                stop_time = times[0] + duration * (self.stop_pgc[ilabel] / 100.0)
                start_idx = np.argmin(np.abs(times-start_time))
                stop_idx = np.argmin(np.abs(times-stop_time)) + 1
                ref_norm_impulse = compute_normalized_impulse(ref_table, 
                    start_idx, stop_idx)
                curr_norm_impulse = compute_normalized_impulse(curr_table, 
                    start_idx, stop_idx)
                diff_norm_impulse[ilabel, isubj] = curr_norm_impulse - ref_norm_impulse

        # Remove elements with zero diff
        diff_norm_impulse = diff_norm_impulse[1:, :]

        diff_norm_impulse_mean = np.mean(diff_norm_impulse, axis=1)
        diff_norm_impulse_std = np.std(diff_norm_impulse, axis=1)

        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111)
        ax.bar(self.labels[1:], diff_norm_impulse_mean, 0.8, color=self.colors[1:])
            # yerr=diff_norm_impulse_std)
        ax.set_ylabel('change in normalized impulse')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', 
            rotation_mode='anchor')
        ax.set_xlabel('perturbation time (% gait cycle)')
        ax.axhline(y=0, color='black', alpha=0.2, linestyle='--', 
            zorder=0, lw=0.5)
        ax.set_ylim(-15, 20)
        ax.set_yticks([-15, -10, -5, 0, 5, 10, 15, 20])
        util.publication_spines(ax)
        fig.tight_layout()
        fig.savefig(target[0], dpi=600)
        plt.close()


class TaskPlotCOMVersusAnklePerturbTime(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, torque, times, colormap, cmap_indices,
            delay):
        super(TaskPlotCOMVersusAnklePerturbTime, self).__init__(study)
        self.name = 'plot_com_versus_ankle_perturb_time'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'com_versus_ankle_perturb_time', 'ankle_perturb')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)
        self.subjects = subjects
        self.torque = torque
        self.times = times

        self.labels = list()
        self.labels.append('unperturbed')
        self.colors = list()
        self.colors.append('black')
        cmap = plt.get_cmap(colormap)
        deps = list()
        for subject in subjects:
            # Unperturbed solution
            deps.append(
                os.path.join(
                    self.study.config['results_path'], 'unperturbed', subject,
                    'tracking_unperturbed_mesh20_center_of_mass.sto'))
            # Perturbed solutions
            for time, cmap_idx in zip(self.times, cmap_indices):
                label = f'torque{self.torque}_time{time}_delay{int(1000*delay)}'
                deps.append(
                    os.path.join(
                        self.study.config['results_path'], 
                        f'ankle_perturb_{label}', subject,
                        f'tracking_perturb_{label}_center_of_mass.sto')
                    )
                self.labels.append(f'{time}%')
                self.colors.append(cmap(cmap_idx))

        self.add_action(deps, 
                        [os.path.join(self.analysis_path, 
                            'com_versus_perturb_time_position.png'), 
                         os.path.join(self.analysis_path, 
                            'com_versus_perturb_time_velocity.png'),
                         os.path.join(self.analysis_path, 
                            'com_versus_perturb_time_acceleration.png'),
                         os.path.join(self.analysis_path, 
                            'com_versus_perturb_time_z_pos_vel_acc.png'), 
                         os.path.join(self.analysis_path, 
                            'com_versus_perturb_time.png')], 
                        self.plot_com_tracking_errors)

    def plot_com_tracking_errors(self, file_dep, target):

        # Add an element for the unperturbed trial so the arrays match length
        percents = [50] + self.times
        numSubjects = len(self.subjects)
        numLabels = len(self.labels)
        max_diff_vel = np.zeros((numLabels, 3, numSubjects))
        max_diff_acc = np.zeros((numLabels, 3, numSubjects))
        peak_diff_vel = np.zeros((numLabels, 3, numSubjects))
        peak_diff_acc = np.zeros((numLabels, 3, numSubjects))
        delay_diff_vel = np.zeros((numLabels, 3, numSubjects))
        delay_diff_acc = np.zeros((numLabels, 3, numSubjects))
        fig = plt.figure(figsize=(6, 3*numSubjects))
        fig2 = plt.figure(figsize=(6, 3*numSubjects))
        fig3 = plt.figure(figsize=(6, 3*numSubjects))
        fig4 = plt.figure(figsize=(3.5, 5))

        for isubj, subject in enumerate(self.subjects):
            refTable = osim.TimeSeriesTable(file_dep[isubj*numLabels])
            ref_pos_x = refTable.getDependentColumn('/|com_position_x').to_numpy()
            ref_pos_y = refTable.getDependentColumn('/|com_position_y').to_numpy()
            ref_pos_z = refTable.getDependentColumn('/|com_position_z').to_numpy()
            ref_vel_x = refTable.getDependentColumn('/|com_velocity_x').to_numpy()
            ref_vel_y = refTable.getDependentColumn('/|com_velocity_y').to_numpy()
            ref_vel_z = refTable.getDependentColumn('/|com_velocity_z').to_numpy()
            ref_acc_x = refTable.getDependentColumn('/|com_acceleration_x').to_numpy()
            ref_acc_y = refTable.getDependentColumn('/|com_acceleration_y').to_numpy()
            ref_acc_z = refTable.getDependentColumn('/|com_acceleration_z').to_numpy()
            ax_xz_pos = fig.add_subplot(numSubjects, 2, isubj+1)
            ax_xy_pos = fig.add_subplot(numSubjects, 2, 2*(isubj+1))
            ax_xz_vel = fig2.add_subplot(numSubjects, 2, isubj+1)
            ax_xy_vel = fig2.add_subplot(numSubjects, 2, 2*(isubj+1))
            ax_xz_acc = fig3.add_subplot(numSubjects, 2, isubj+1)
            ax_xy_acc = fig3.add_subplot(numSubjects, 2, 2*(isubj+1))

            ax_pos = fig4.add_subplot(3, numSubjects, isubj+1)
            ax_vel = fig4.add_subplot(3, numSubjects, 2*(isubj+1))
            ax_acc = fig4.add_subplot(3, numSubjects, 3*(isubj+1))

            for i, (label, color, percent) in enumerate(zip(self.labels, self.colors, percents)):
                currTable = osim.TimeSeriesTable(
                    file_dep[isubj*numLabels + i])
                curr_pos_x = currTable.getDependentColumn('/|com_position_x').to_numpy()
                curr_pos_y = currTable.getDependentColumn('/|com_position_y').to_numpy()
                curr_pos_z = currTable.getDependentColumn('/|com_position_z').to_numpy()
                curr_vel_x = currTable.getDependentColumn('/|com_velocity_x').to_numpy()
                curr_vel_y = currTable.getDependentColumn('/|com_velocity_y').to_numpy()
                curr_vel_z = currTable.getDependentColumn('/|com_velocity_z').to_numpy()
                curr_acc_x = currTable.getDependentColumn('/|com_acceleration_x').to_numpy()
                curr_acc_y = currTable.getDependentColumn('/|com_acceleration_y').to_numpy()
                curr_acc_z = currTable.getDependentColumn('/|com_acceleration_z').to_numpy()

                diff_vel_x = curr_vel_x - ref_vel_x 
                diff_vel_y = curr_vel_y - ref_vel_y 
                diff_vel_z = curr_vel_z - ref_vel_z 
                diff_acc_x = curr_acc_x - ref_acc_x 
                diff_acc_y = curr_acc_y - ref_acc_y 
                diff_acc_z = curr_acc_z - ref_acc_z

                time = refTable.getIndependentColumn()
                initial_time = time[0]
                final_time = time[-1]
                duration = final_time - initial_time
                percent_time = initial_time + duration * (percent / 100.0)

                max_diff_vel_x_idx = np.argmax(np.absolute(diff_vel_x)) 
                max_diff_vel_y_idx = np.argmax(np.absolute(diff_vel_y)) 
                max_diff_vel_z_idx = np.argmax(np.absolute(diff_vel_z)) 
                max_diff_acc_x_idx = np.argmax(np.absolute(diff_acc_x)) 
                max_diff_acc_y_idx = np.argmax(np.absolute(diff_acc_y)) 
                max_diff_acc_z_idx = np.argmax(np.absolute(diff_acc_z)) 

                # Compute max change in COM kinematics 
                max_diff_vel[i, 0, isubj] = diff_vel_x[max_diff_vel_x_idx]
                max_diff_vel[i, 1, isubj] = diff_vel_y[max_diff_vel_y_idx]
                max_diff_vel[i, 2, isubj] = diff_vel_z[max_diff_vel_z_idx]
                max_diff_acc[i, 0, isubj] = diff_acc_x[max_diff_acc_x_idx]
                max_diff_acc[i, 1, isubj] = diff_acc_y[max_diff_acc_y_idx]
                max_diff_acc[i, 2, isubj] = diff_acc_z[max_diff_acc_z_idx]

                # Compute change in COM kinematics at peak perturbation
                percent_time = initial_time + duration * (percent / 100.0)
                index = refTable.getNearestRowIndexForTime(percent_time)
                peak_diff_vel[i, 0, isubj] = diff_vel_x[index]
                peak_diff_vel[i, 1, isubj] = diff_vel_y[index]
                peak_diff_vel[i, 2, isubj] = diff_vel_z[index]
                peak_diff_acc[i, 0, isubj] = diff_acc_x[index]
                peak_diff_acc[i, 1, isubj] = diff_acc_y[index]
                peak_diff_acc[i, 2, isubj] = diff_acc_z[index]
                
                plot_pos_x = curr_pos_x - curr_pos_x[0]
                plot_pos_y = curr_pos_y - curr_pos_y[0]
                plot_pos_z = curr_pos_z - curr_pos_z[0]

                # diff_pos_x = np.diff(curr_pos_x)
                # diff_pos_y = np.diff(curr_pos_y)
                # diff_pos_z = np.diff(curr_pos_z)
                # imax = np.argmax(diff_pos_x)
                # diff_pos_x[imax] = (diff_pos_x[imax-1] + diff_pos_x[imax+1]) / 2.0
                # diff_pos_y[imax] = (diff_pos_y[imax-1] + diff_pos_y[imax+1]) / 2.0
                # diff_pos_z[imax] = (diff_pos_z[imax-1] + diff_pos_z[imax+1]) / 2.0

                # diff_vel_x = diff_pos_x / np.diff(time)
                # diff_vel_y = diff_pos_y / np.diff(time)
                # diff_vel_z = diff_pos_z / np.diff(time)

                # temp_diff_vel_x = np.diff(diff_vel_x)
                # temp_diff_vel_y = np.diff(diff_vel_y)
                # temp_diff_vel_z = np.diff(diff_vel_z)
                # imax = np.argmin(temp_diff_vel_x)
                # temp_diff_vel_x[imax] = (temp_diff_vel_x[imax-1] + temp_diff_vel_x[imax+1]) / 2.0
                # temp_diff_vel_y[imax] = (temp_diff_vel_y[imax-1] + temp_diff_vel_y[imax+1]) / 2.0
                # temp_diff_vel_z[imax] = (temp_diff_vel_z[imax-1] + temp_diff_vel_z[imax+1]) / 2.0

                # diff_acc_x = temp_diff_vel_x / np.diff(time)[1:]
                # diff_acc_y = temp_diff_vel_y / np.diff(time)[1:]
                # diff_acc_z = temp_diff_vel_z / np.diff(time)[1:]

                # if i:
                #     import pdb
                #     pdb.set_trace()

                # Position
                ax_xz_pos.plot(plot_pos_z, plot_pos_x, color=color, linewidth=2,
                    zorder=0)
                ax_xz_pos.scatter(plot_pos_z[0], plot_pos_x[0], s=50, color=color, 
                    marker='o')
                ax_xz_pos.scatter(plot_pos_z[-1], plot_pos_x[-1], s=50, color=color, 
                    marker='X')
                util.publication_spines(ax_xz_pos)
                ax_xz_pos.set_xlim(-0.05, 0.05)
                ax_xz_pos.set_xticks([-0.05, -0.025, 0, 0.025, 0.05])
                ax_xz_pos.set_ylim(-0.1, 1.5)
                ax_xz_pos.set_yticks([0, 0.5, 1.0, 1.5])
                ax_xz_pos.set_xlabel('medio-lateral COM position (m)')
                ax_xz_pos.set_ylabel('anterior-posterior COM position (m)')

                ax_xy_pos.plot(plot_pos_x, plot_pos_y, color=color, linewidth=2,
                    zorder=0)
                ax_xy_pos.scatter(plot_pos_x[0], plot_pos_y[0], s=50, color=color, 
                    marker='o')
                ax_xy_pos.scatter(plot_pos_x[-1], plot_pos_y[-1], s=50, color=color, 
                    marker='X')
                util.publication_spines(ax_xy_pos)
                
                ax_xy_pos.set_xlabel('anterior-posterior COM position (m)')
                ax_xy_pos.set_xlim(-0.1, 1.5)
                ax_xy_pos.set_xticks([0, 0.5, 1.0, 1.5])
                ax_xy_pos.set_ylabel('superior-inferior COM position (m)')

                fig.tight_layout()
                fig.savefig(target[0], dpi=600)
                plt.close()

                # Velocity
                ax_xz_vel.plot(curr_vel_z, curr_vel_x, color=color, linewidth=2,
                    zorder=0)
                # ax_xz_vel.plot(diff_vel_z, diff_vel_x, color=color, linewidth=2,
                #     linestyle='--')
                ax_xz_vel.scatter(curr_vel_z[0], curr_vel_x[0], s=50, color=color, 
                    marker='o')
                ax_xz_vel.scatter(curr_vel_z[-1], curr_vel_x[-1], s=50, color=color, 
                    marker='X')
                util.publication_spines(ax_xz_vel)
                # ax_xz_vel.set_xlim(-0.05, 0.05)
                # ax_xz_vel.set_xticks([-0.05, -0.025, 0, 0.025, 0.05])
                # ax_xz_vel.set_ylim(-0.1, 1.5)
                # ax_xz_vel.set_yticks([0, 0.5, 1.0, 1.5])
                ax_xz_vel.set_xlabel('medio-lateral COM velocity (m/s)')
                ax_xz_vel.set_ylabel('anterior-posterior COM velocity (m/s)')

                ax_xy_vel.plot(curr_vel_x, curr_vel_y, color=color, linewidth=2,
                    zorder=0)
                ax_xy_vel.scatter(curr_vel_x[0], curr_vel_y[0], s=50, color=color, 
                    marker='o')
                ax_xy_vel.scatter(curr_vel_x[-1], curr_vel_y[-1], s=50, color=color, 
                    marker='X')
                util.publication_spines(ax_xy_vel)
                
                ax_xy_vel.set_xlabel('anterior-posterior COM velocity (m/s)')
                # ax_xy_vel.set_xlim(-0.1, 1.5)
                # ax_xy_vel.set_xticks([0, 0.5, 1.0, 1.5])
                ax_xy_vel.set_ylabel('superior-inferior COM velocity (m/s)')

                fig2.tight_layout()
                fig2.savefig(target[1], dpi=600)
                plt.close()

                # Acceleration
                ax_xz_acc.plot(curr_acc_z, curr_acc_x, color=color, linewidth=2,
                    zorder=0)
                # ax_xz_acc.plot(diff_acc_z, diff_acc_x, color=color, linewidth=2,
                #     linestyle='--')
                ax_xz_acc.scatter(curr_acc_z[0], curr_acc_x[0], s=50, color=color, 
                    marker='o')
                ax_xz_acc.scatter(curr_acc_z[-1], curr_acc_x[-1], s=50, color=color, 
                    marker='X')
                util.publication_spines(ax_xz_acc)
                # ax_xz_acc.set_xlim(-0.05, 0.05)
                # ax_xz_acc.set_xticks([-0.05, -0.025, 0, 0.025, 0.05])
                # ax_xz_acc.set_ylim(-0.1, 1.5)
                # ax_xz_acc.set_yticks([0, 0.5, 1.0, 1.5])
                ax_xz_acc.set_xlabel('medio-lateral COM acceleration (m/s^2)')
                ax_xz_acc.set_ylabel('anterior-posterior COM acceleration (m/s^2)')

                ax_xy_acc.plot(curr_acc_x, curr_acc_y, color=color, linewidth=2,
                    zorder=0)
                ax_xy_acc.scatter(curr_acc_x[0], curr_acc_y[0], s=50, color=color, 
                    marker='o')
                ax_xy_acc.scatter(curr_acc_x[-1], curr_acc_y[-1], s=50, color=color, 
                    marker='X')
                util.publication_spines(ax_xy_acc)
                
                ax_xy_acc.set_xlabel('anterior-posterior COM acceleration (m/s^2)')
                # ax_xy_acc.set_xlim(-0.1, 1.5)
                # ax_xy_acc.set_xticks([0, 0.5, 1.0, 1.5])
                ax_xy_acc.set_ylabel('superior-inferior COM acceleration (m/s^2)')

                fig3.tight_layout()
                fig3.savefig(target[2], dpi=600)
                plt.close()


                # Position, velocity, acceleration versus time
                pgc = np.linspace(0, 100, len(time))
                ax_pos.plot(pgc, curr_pos_z, color=color, linewidth=1.5)
                ax_pos.set_xticklabels([])
                ax_pos.set_ylabel('position (m)')
                ax_pos.set_xlim(0, 100)
                util.publication_spines(ax_pos)

                ax_vel.plot(pgc, curr_vel_z, color=color, linewidth=1.5)
                ax_vel.set_xticklabels([])
                ax_vel.set_ylabel('velocity (m/s)')
                ax_vel.set_xlim(0, 100)
                util.publication_spines(ax_vel)

                ax_acc.plot(pgc, curr_acc_z, color=color, linewidth=1.5)
                util.publication_spines(ax_acc)
                ax_acc.set_ylabel('acceleration (m/s^2)')
                ax_acc.set_xlabel('time (% gait cycle)')
                ax_acc.set_xlim(0, 100)

                fig4.tight_layout()
                fig4.savefig(target[3], dpi=600)
                plt.close()

        def plot_com_versus_time(axes, diff_vel, diff_acc, marker, 
                colors, label):
            ax_vel_ml = axes[0]
            ax_vel_ap = axes[1]
            ax_acc_ml = axes[2]
            ax_acc_ap = axes[3]

            # Remove elements with zero diffs
            diff_vel = diff_vel[1:, :, :]
            diff_acc = diff_acc[1:, :, :]

            diff_vel_mean = np.mean(diff_vel, axis=2)
            diff_vel_std = np.std(diff_vel, axis=2)
            diff_acc_mean = np.mean(diff_acc, axis=2)
            diff_acc_std = np.std(diff_acc, axis=2)

            ax_vel_ml.bar(self.times,  diff_vel_mean[:, 2], color=colors,
                width=7)
            ax_vel_ml.set_ylabel('medio-lateral')
            # ax_vel_ml.set_ylim(-0.2, 0.2)
            ax_vel_ml.set_xticks(self.times)
            ax_vel_ml.set_xticklabels([])
            ax_vel_ml.legend(fontsize=6, fancybox=False, frameon=False)
            ax_vel_ml.set_title('center-of-mass\nvelocity (m/s)', fontsize=8)

            ax_vel_ap.bar(self.times,  diff_vel_mean[:, 0], color=colors,
                width=7)
            ax_vel_ap.set_ylabel('anterior-posterior')
            # ax_vel_ap.set_ylim(-0.2, 0.2)
            ax_vel_ap.set_xticks(self.times)
            ax_vel_ap.set_xticklabels(self.labels[1:])
            ax_vel_ap.set_xlabel('peak time\n(% gait cycle)')

            ax_acc_ml.bar(self.times,  diff_acc_mean[:, 2], color=colors,
                width=7)
            # ax_acc_ml.set_ylim(-2.0, 2.0)
            ax_acc_ml.set_xticks(self.times)
            ax_acc_ml.set_xticklabels([])
            ax_acc_ml.set_title('center-of-mass\nacceleration (m/s^2)', 
                fontsize=8)

            ax_acc_ap.bar(self.times,  diff_acc_mean[:, 0], color=colors,
                width=7)
            # ax_acc_ap.set_ylim(-2.0, 2.0)
            ax_acc_ap.set_xticks(self.times)
            ax_acc_ap.set_xticklabels(self.labels[1:])
            ax_acc_ap.set_xlabel('peak time\n(% gait cycle)')

            for ax in [ax_vel_ml, ax_vel_ap, ax_acc_ml, ax_acc_ap]:
                util.publication_spines(ax)
                ax.axhline(y=0, color='gray', linestyle='--',
                            linewidth=0.5, alpha=0.7, zorder=0)

        fig = plt.figure(figsize=(4.5, 4.5))
        ax_vel_ml = fig.add_subplot(221)
        ax_vel_ap = fig.add_subplot(223)
        ax_acc_ml = fig.add_subplot(222)
        ax_acc_ap = fig.add_subplot(224)
        axes = [ax_vel_ml, ax_vel_ap, ax_acc_ml, ax_acc_ap]
        # plot_com_versus_time(axes, max_diff_vel, max_diff_acc, 'o', 'orange',
        #     'maximum change')
        plot_com_versus_time(axes, peak_diff_vel, peak_diff_acc, 'o', self.colors[1:],
            'change at peak torque')
        fig.tight_layout()
        fig.savefig(target[4], dpi=600)
        plt.close()


class TaskPlotCOMVersusAnklePerturbTorque(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, time, torques=[25, 50, 75, 100],
            delay=0.400):
        super(TaskPlotCOMVersusAnklePerturbTorque, self).__init__(study)
        self.name = f'plot_com_versus_ankle_perturb_torque_for_time{time}'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            f'com_versus_ankle_perturb_torque_for_time{time}', 
            'ankle_perturb')
        if not os.path.exists(self.analysis_path): os.makedirs(self.analysis_path)
        self.subjects = subjects
        self.torques = torques
        self.time = time

        self.labels = list()
        self.labels.append('unperturbed')
        perturb_fpaths = list()
        for torque in self.torques:
            label = f'torque{torque}_time{self.time}_delay{int(1000*delay)}'
            perturb_fpaths.append(
                os.path.join('ankle_perturb', label, 
                    f'tracking_perturb_{label}_center_of_mass.sto')
                )
            self.labels.append(f'{torque}%')

        deps = list()
        for subject in subjects:
            condition_path = os.path.join(self.results_path, subject, 
                'unperturbed')
            # Unperturbed solution
            deps.append(os.path.join(condition_path, 'moco', 
                'tracking_unperturbed_center_of_mass.sto'))
            # Perturbed solutions
            for perturb_fpath in perturb_fpaths:
                deps.append(os.path.join(condition_path, perturb_fpath))

        self.add_action(deps, 
                        [os.path.join(self.analysis_path, 
                            'com_versus_perturb_torque_max.png'),
                         os.path.join(self.analysis_path, 
                            'com_versus_perturb_torque_peak.png'),
                        ], 
                        self.plot_com_tracking_errors)

    def plot_com_tracking_errors(self, file_dep, target):

        # Add an element for the unperturbed trial so the arrays match length
        numSubjects = len(self.subjects)
        numLabels = len(self.labels)
        max_diff_vel = np.zeros((numLabels, 3, numSubjects))
        max_diff_acc = np.zeros((numLabels, 3, numSubjects))
        peak_diff_vel = np.zeros((numLabels, 3, numSubjects))
        peak_diff_acc = np.zeros((numLabels, 3, numSubjects))
        delay_diff_vel = np.zeros((numLabels, 3, numSubjects))
        delay_diff_acc = np.zeros((numLabels, 3, numSubjects))
        for isubj, subject in enumerate(self.subjects):
            refTable = osim.TimeSeriesTable(file_dep[isubj*numLabels])
            ref_vel_x = refTable.getDependentColumn('/|com_velocity_x').to_numpy()
            ref_vel_y = refTable.getDependentColumn('/|com_velocity_y').to_numpy()
            ref_vel_z = refTable.getDependentColumn('/|com_velocity_z').to_numpy()
            ref_acc_x = refTable.getDependentColumn('/|com_acceleration_x').to_numpy()
            ref_acc_y = refTable.getDependentColumn('/|com_acceleration_y').to_numpy()
            ref_acc_z = refTable.getDependentColumn('/|com_acceleration_z').to_numpy()
            for i, label in enumerate(self.labels):
                currTable = osim.TimeSeriesTable(file_dep[isubj*numLabels + i])
                curr_vel_x = currTable.getDependentColumn('/|com_velocity_x').to_numpy()
                curr_vel_y = currTable.getDependentColumn('/|com_velocity_y').to_numpy()
                curr_vel_z = currTable.getDependentColumn('/|com_velocity_z').to_numpy()
                curr_acc_x = currTable.getDependentColumn('/|com_acceleration_x').to_numpy()
                curr_acc_y = currTable.getDependentColumn('/|com_acceleration_y').to_numpy()
                curr_acc_z = currTable.getDependentColumn('/|com_acceleration_z').to_numpy()

                diff_vel_x = curr_vel_x - ref_vel_x 
                diff_vel_y = curr_vel_y - ref_vel_y 
                diff_vel_z = curr_vel_z - ref_vel_z 
                diff_acc_x = curr_acc_x - ref_acc_x 
                diff_acc_y = curr_acc_y - ref_acc_y 
                diff_acc_z = curr_acc_z - ref_acc_z 

                time = refTable.getIndependentColumn()
                initial_time = time[0]
                final_time = time[-1]
                duration = final_time - initial_time
                percent_time = initial_time + duration * (self.time / 100.0)

                # Only find max values between torque onset and torque peak
                temp_start_time = initial_time + duration * ((self.time - 25) / 100.0)
                temp_start_end = initial_time + duration * ((self.time - 10) / 100.0)
                temp_start = refTable.getNearestRowIndexForTime(temp_start_time)
                temp_end = refTable.getNearestRowIndexForTime(temp_start_end)

                temp_diff_vel_x = diff_vel_x[temp_start:temp_end] 
                temp_diff_vel_y = diff_vel_y[temp_start:temp_end] 
                temp_diff_vel_z = diff_vel_z[temp_start:temp_end] 
                temp_diff_acc_x = diff_acc_x[temp_start:temp_end] 
                temp_diff_acc_y = diff_acc_y[temp_start:temp_end] 
                temp_diff_acc_z = diff_acc_z[temp_start:temp_end] 

                max_diff_vel_x_idx = np.argmax(np.absolute(temp_diff_vel_x)) 
                max_diff_vel_y_idx = np.argmax(np.absolute(temp_diff_vel_y)) 
                max_diff_vel_z_idx = np.argmax(np.absolute(temp_diff_vel_z)) 
                max_diff_acc_x_idx = np.argmax(np.absolute(temp_diff_acc_x)) 
                max_diff_acc_y_idx = np.argmax(np.absolute(temp_diff_acc_y)) 
                max_diff_acc_z_idx = np.argmax(np.absolute(temp_diff_acc_z)) 

                # Compute max change in COM kinematics 
                max_diff_vel[i, 0, isubj] = temp_diff_vel_x[max_diff_vel_x_idx]
                max_diff_vel[i, 1, isubj] = temp_diff_vel_y[max_diff_vel_y_idx]
                max_diff_vel[i, 2, isubj] = temp_diff_vel_z[max_diff_vel_z_idx]
                max_diff_acc[i, 0, isubj] = temp_diff_acc_x[max_diff_acc_x_idx]
                max_diff_acc[i, 1, isubj] = temp_diff_acc_y[max_diff_acc_y_idx]
                max_diff_acc[i, 2, isubj] = temp_diff_acc_z[max_diff_acc_z_idx] 

                # Compute change in COM kinematics at peak perturbation
                perturb_time = initial_time + duration * ((self.time - 15) / 100.0)
                index = refTable.getNearestRowIndexForTime(perturb_time)
                peak_diff_vel[i, 0, isubj] = diff_vel_x[index]
                peak_diff_vel[i, 1, isubj] = diff_vel_y[index]
                peak_diff_vel[i, 2, isubj] = diff_vel_z[index]
                peak_diff_acc[i, 0, isubj] = diff_acc_x[index]
                peak_diff_acc[i, 1, isubj] = diff_acc_y[index]
                peak_diff_acc[i, 2, isubj] = diff_acc_z[index]

        def plot_com_versus_torque(diff_vel, diff_acc, target):
            fig = plt.figure(figsize=(4, 4))
            ax_vel_ml = fig.add_subplot(221)
            ax_vel_ap = fig.add_subplot(223)
            ax_acc_ml = fig.add_subplot(222)
            ax_acc_ap = fig.add_subplot(224)

            # Remove elements with zero diffs
            diff_vel = diff_vel[1:, :, :]
            diff_acc = diff_acc[1:, :, :]

            diff_vel_mean = np.mean(diff_vel, axis=2)
            diff_vel_std = np.std(diff_vel, axis=2)
            diff_acc_mean = np.mean(diff_acc, axis=2)
            diff_acc_std = np.std(diff_acc, axis=2)

            ax_vel_ml.scatter(self.torques,  diff_vel_mean[:, 2])
            ax_vel_ml.set_ylabel('change in ML COM velocity (m/s)')
            ax_vel_ml.set_ylim(-0.2, 0.2)
            ax_vel_ml.set_xticks(self.torques)
            ax_vel_ml.set_xticklabels([])

            ax_vel_ap.scatter(self.torques,  diff_vel_mean[:, 0])
            ax_vel_ap.set_ylabel('change in AP COM velocity (m/s)')
            ax_vel_ap.set_ylim(-0.2, 0.2)
            ax_vel_ap.set_xticks(self.torques)
            ax_vel_ap.set_xticklabels(self.labels[1:])
            ax_vel_ap.set_xlabel('peak torque\n(%*mass N-m)')

            ax_acc_ml.scatter(self.torques,  diff_acc_mean[:, 2])
            ax_acc_ml.set_ylabel('change in ML COM accel (m/s^2)')
            ax_acc_ml.set_ylim(-2.0, 2.0)
            ax_acc_ml.set_xticks(self.torques)
            ax_acc_ml.set_xticklabels([])

            ax_acc_ap.scatter(self.torques,  diff_acc_mean[:, 0])
            ax_acc_ap.set_ylabel('change in AP COM accel (m/s^2)')
            ax_acc_ap.set_ylim(-2.0, 2.0)
            ax_acc_ap.set_xticks(self.torques)
            ax_acc_ap.set_xticklabels(self.labels[1:])
            ax_acc_ap.set_xlabel('peak torque\n(%*mass N-m)')

            for ax in [ax_vel_ml, ax_vel_ap, ax_acc_ml, ax_acc_ap]:
                util.publication_spines(ax)
                ax.axhline(y=0, color='gray', linestyle='--',
                            linewidth=1, zorder=0)

            fig.tight_layout()
            fig.savefig(target, dpi=600)
            plt.close()

        plot_com_versus_torque(max_diff_vel, max_diff_acc, target[0])
        plot_com_versus_torque(peak_diff_vel, peak_diff_acc, target[1])


class TaskPlotCOMVersusAnklePerturbDelay(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, delays, torque, time):
        super(TaskPlotCOMVersusAnklePerturbDelay, self).__init__(study)
        self.name = f'plot_com_versus_ankle_perturb_delay_torque{torque}_time{time}'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            f'com_versus_ankle_perturb_delay_torque{torque}_time{time}', 
            'ankle_perturb')
        if not os.path.exists(self.analysis_path): os.makedirs(self.analysis_path)
        self.subjects = subjects
        self.torque = torque
        self.time = time
        self.delays = delays

        self.labels = list()
        self.labels.append('unperturbed')
        self.colors = list()
        self.colors.append('black')
        cmap = plt.get_cmap('viridis')
        idxs = np.linspace(0, 1, len(self.delays))
        perturb_fpaths = list()
        for delay, idx in zip(self.delays, idxs):

            label = f'torque{self.torque}_time{self.time}_delay{int(1000*delay)}'
            perturb_fpaths.append(
                os.path.join('ankle_perturb', label, 
                    f'tracking_perturb_{label}_center_of_mass.sto')
                )
            self.labels.append(f'{delay}')
            self.colors.append(cmap(idx))

        deps = list()
        for subject in subjects:
            condition_path = os.path.join(self.results_path, subject, 
                'unperturbed')
            # Unperturbed solution
            deps.append(os.path.join(condition_path, 'moco', 
                'tracking_unperturbed_center_of_mass.sto'))
            # Perturbed solutions
            for perturb_fpath in perturb_fpaths:
                deps.append(os.path.join(condition_path, perturb_fpath))

        self.add_action(deps, 
                        [os.path.join(self.analysis_path, 
                            'com_versus_perturb_time_position.png'), 
                        ], 
                        self.plot_com_tracking_errors)

    def plot_com_tracking_errors(self, file_dep, target):

        numSubjects = len(self.subjects)
        numLabels = len(self.labels)
        fig = plt.figure(figsize=(6, 3*numSubjects))

        for isubj, subject in enumerate(self.subjects):
            refTable = osim.TimeSeriesTable(file_dep[isubj*numLabels])
            ref_pos_x = refTable.getDependentColumn('/|com_position_x').to_numpy()
            ref_pos_y = refTable.getDependentColumn('/|com_position_y').to_numpy()
            ref_pos_z = refTable.getDependentColumn('/|com_position_z').to_numpy()
            ref_vel_x = refTable.getDependentColumn('/|com_velocity_x').to_numpy()
            ref_vel_y = refTable.getDependentColumn('/|com_velocity_y').to_numpy()
            ref_vel_z = refTable.getDependentColumn('/|com_velocity_z').to_numpy()
            ref_acc_x = refTable.getDependentColumn('/|com_acceleration_x').to_numpy()
            ref_acc_y = refTable.getDependentColumn('/|com_acceleration_y').to_numpy()
            ref_acc_z = refTable.getDependentColumn('/|com_acceleration_z').to_numpy()
            ax_xz = fig.add_subplot(numSubjects, 2, isubj+1)
            ax_xy = fig.add_subplot(numSubjects, 2, 2*(isubj+1))


            for i, (label, color) in enumerate(zip(self.labels, self.colors)):
                currTable = osim.TimeSeriesTable(
                    file_dep[isubj*numLabels + i])
                curr_pos_x = currTable.getDependentColumn('/|com_position_x').to_numpy()
                curr_pos_y = currTable.getDependentColumn('/|com_position_y').to_numpy()
                curr_pos_z = currTable.getDependentColumn('/|com_position_z').to_numpy()
                curr_vel_x = currTable.getDependentColumn('/|com_velocity_x').to_numpy()
                curr_vel_y = currTable.getDependentColumn('/|com_velocity_y').to_numpy()
                curr_vel_z = currTable.getDependentColumn('/|com_velocity_z').to_numpy()
                curr_acc_x = currTable.getDependentColumn('/|com_acceleration_x').to_numpy()
                curr_acc_y = currTable.getDependentColumn('/|com_acceleration_y').to_numpy()
                curr_acc_z = currTable.getDependentColumn('/|com_acceleration_z').to_numpy()
                
                plot_pos_x = curr_pos_x - curr_pos_x[0]
                plot_pos_y = curr_pos_y - curr_pos_y[0]
                plot_pos_z = curr_pos_z - curr_pos_z[0]

                ax_xz.plot(plot_pos_z, plot_pos_x, color=color, linewidth=2)
                ax_xz.scatter(plot_pos_z[0], plot_pos_x[0], s=50, color=color, 
                    marker='o')
                ax_xz.scatter(plot_pos_z[-1], plot_pos_x[-1], s=50, color=color, 
                    marker='X')
                util.publication_spines(ax_xz)
                ax_xz.set_xlim(-0.05, 0.05)
                ax_xz.set_xticks([-0.05, -0.025, 0, 0.025, 0.05])
                ax_xz.set_ylim(-0.05, 1.5)
                ax_xz.set_yticks([0, 0.5, 1.0, 1.5])
                ax_xz.set_xlabel('medio-lateral COM position (m)')
                ax_xz.set_ylabel('anterior-posterior COM position (m)')
                util.publication_spines(ax_xz)

                ax_xy.plot(plot_pos_x, plot_pos_y, color=color, linewidth=2)
                ax_xy.scatter(plot_pos_x[0], plot_pos_y[0], s=50, color=color, 
                    marker='o')
                ax_xy.scatter(plot_pos_x[-1], plot_pos_y[-1], s=50, color=color, 
                    marker='X')
                util.publication_spines(ax_xy)
                
                ax_xy.set_xlabel('anterior-posterior COM position (m)')
                ax_xy.set_xlim(-0.05, 1.5)
                ax_xy.set_xticks([0, 0.5, 1.0, 1.5])
                ax_xy.set_ylabel('superior-inferior COM position (m)')
                # ax.set_xlim(-0.05, 0.05)
                # ax.set_xticks([-0.05, -0.025, 0, 0.025, 0.05])
                util.publication_spines(ax_xy)

                fig.tight_layout()
                fig.savefig(target[0], dpi=600)
                plt.close()


class TaskPlotCOMTrackingErrorsAnklePerturbDelay(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects, delays, torque, time):
        super(TaskPlotCOMTrackingErrorsAnklePerturbDelay, self).__init__(study)
        self.name = f'plot_com_tracking_errors_ankle_perturb_delay_torque{torque}_time{time}'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            f'com_tracking_errors_perturb_delay_torque{torque}_time{time}', 
            'ankle_perturb')
        if not os.path.exists(self.analysis_path): os.makedirs(self.analysis_path)
        self.subjects = subjects
        self.torque = torque
        self.time = time
        self.delays = delays

        self.labels = list()
        self.labels.append('unperturbed')
        self.colors = list()
        self.colors.append('black')
        cmap = plt.get_cmap('viridis')
        idxs = np.linspace(0, 1, len(self.delays))
        perturb_fpaths = list()
        for delay, idx in zip(self.delays, idxs):
            label = f'torque{torque}_time{time}_delay{int(1000*delay)}'
            perturb_fpaths.append(
                os.path.join('ankle_perturb', label, 
                    f'tracking_perturb_{label}_center_of_mass.sto')
                )
            self.labels.append(f'{int(1000*delay)}')
            self.colors.append(cmap(idx))

        deps = list()
        for subject in subjects:
            condition_path = os.path.join(self.results_path, subject, 
                'unperturbed')
            # Unperturbed solution
            deps.append(os.path.join(condition_path, 'moco', 
                'tracking_unperturbed_center_of_mass.sto'))
            # Perturbed solutions
            for perturb_fpath in perturb_fpaths:
                deps.append(os.path.join(condition_path, perturb_fpath))

        self.add_action(deps, 
                        [os.path.join(self.analysis_path, 
                            'absolute_errors.png')
                        ], 
                        self.plot_com_tracking_errors)

    def plot_com_tracking_errors(self, file_dep, target):

        def compute_com_rmse(currTable, refTable):
            times = refTable.getIndependentColumn()
            labels = refTable.getColumnLabels()
            sumSquaredError = np.zeros_like(times)
            for itime, time in enumerate(times):
                for label in labels:
                    if 'position' in label:
                        currValue = currTable.getDependentColumn(label)[itime]
                        refValue = refTable.getDependentColumn(label)[itime]
                        error = currValue - refValue
                        sumSquaredError[itime] += error * error

            # Trapezoidal rule
            interval = times[-1] - times[0]
            isse = 0.5 * interval * (sumSquaredError.sum() + 
                                     sumSquaredError[1:-1].sum())

            rmse = np.sqrt(isse / interval / len(labels))  
            return rmse

        numSubjects = len(self.subjects)
        numLabels = len(self.labels)
        errors = np.zeros((numLabels, numSubjects))
        for isubj, subject in enumerate(self.subjects):
            refTable = osim.TimeSeriesTable(file_dep[isubj*numLabels])
            for ilabel, label in enumerate(self.labels):
                currTable = osim.TimeSeriesTable(
                    file_dep[isubj*numLabels + ilabel])
                errors[ilabel, isubj] = compute_com_rmse(currTable, refTable)

        # Remove elements with zero error
        errors = errors[1:, :]

        errors_mean = np.mean(errors, axis=1)
        errors_std = np.std(errors, axis=1)

        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(111)
        ax.bar(self.labels[1:], errors_mean, 0.75, 
            yerr=errors_std, color=self.colors[1:])
        ax.set_ylabel('center-of-mass tracking error (RMS)')
        # plt.setp(ax.get_xticklabels(), rotation=30, ha='right', 
        #     rotation_mode='anchor')
        ax.set_xlabel('delay time (ms)')
        util.publication_spines(ax)
        fig.tight_layout()
        fig.savefig(target[0], dpi=600)
        plt.close()


class TaskPlotGroundReactionsAnklePerturb(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subject, times, colormap, cmap_indices, 
            delay, torques=[100]):
        super(TaskPlotGroundReactionsAnklePerturb, self).__init__(study)
        self.name = f'plot_grfs_ankle_perturb'
        self.results_path = os.path.join(study.config['results_path'], 
            'experiments')
        self.analysis_path = os.path.join(study.config['analysis_path'],
            'ground_reactions', f'ankle_perturb_{subject}')
        if not os.path.exists(self.analysis_path): 
            os.makedirs(self.analysis_path)

        self.subject = subject
        self.torques = torques
        self.times = times
        self.cmap_indices = cmap_indices

        self.labels = list()
        # self.labels.append('experiment')
        self.labels.append('unperturbed')

        self.colors = list()
        # self.colors.append('gray')
        self.colors.append('black')

        self.alphas = list()
        # self.alphas.append(1.0)
        self.alphas.append(1.0)

        perturb_fpaths = list()
        cmap = plt.get_cmap(colormap)
        self.times_colors = list()
        for i in np.arange(len(times)):
            self.times_colors.append(cmap(self.cmap_indices[i])) 
        for torque in self.torques:
            for time, cmap_idx in zip(self.times, self.cmap_indices):
                label = f'torque{torque}_time{time}_delay{int(1000*delay)}'
                perturb_fpaths.append(
                    os.path.join(self.study.config['results_path'], 
                        f'ankle_perturb_{label}', 
                        subject, f'perturb_{label}_grfs.sto'))
                self.labels.append(label)
                self.colors.append(cmap(cmap_idx))
                self.alphas.append(torque / 100.0)

        deps = list()
      
        # Model 
        self.model = os.path.join(self.study.config['results_path'], 
            'unperturbed', subject, 'model_unperturbed_mesh20.osim')
        # Experimental grfs
        # deps.append(os.path.join(condition_path, 'expdata', 
        #     'ground_reaction.mot'))
        # Unperturbed grfs
        deps.append(os.path.join(self.study.config['results_path'], 
            'unperturbed', subject, 'unperturbed_mesh20_grfs.sto'))
        # Perturbed grfs
        for perturb_fpath in perturb_fpaths:
            deps.append(perturb_fpath)

        self.add_action(deps, 
                        [os.path.join(self.analysis_path, 
                            'ground_reactions.png'),
                         os.path.join(self.analysis_path, 
                            'ground_reaction_ML_diffs_at_peak.png'),
                         os.path.join(self.analysis_path, 
                            'ground_reaction_SI_diffs_at_peak.png'),
                         os.path.join(self.analysis_path, 
                            'ground_reaction_AP_diffs_at_peak.png')
                        ], 
                        self.plot_com_tracking_errors)

    def plot_com_tracking_errors(self, file_dep, target):

        # Get time range
        percents = [50] + self.times
        grf_temp = osim.TimeSeriesTable(file_dep[-1])
        time_temp = grf_temp.getIndependentColumn()
        initial_time = time_temp[0]
        final_time = time_temp[-1]
        duration = final_time - initial_time

        fig = plt.figure(figsize=(8, 8))
        numLabels = len(self.labels)
        peak_diff_lgrf = np.zeros((numLabels, 3))
        peak_diff_rgrf = np.zeros((numLabels, 3))

        model = osim.Model(self.model)
        state = model.initSystem()
        mass = model.getTotalMass(state)
        BW = abs(model.getGravity()[1]) * mass
        
        lgrfx_ax = fig.add_subplot(3, 2, 1)
        lgrfy_ax = fig.add_subplot(3, 2, 3)
        lgrfz_ax = fig.add_subplot(3, 2, 5)
        rgrfx_ax = fig.add_subplot(3, 2, 2)
        rgrfy_ax = fig.add_subplot(3, 2, 4)
        rgrfz_ax = fig.add_subplot(3, 2, 6)

        plot_zip = zip(self.labels, self.colors, self.alphas, percents)
        for i, (label, color, alpha, percent) in enumerate(plot_zip):
            ref_grfs = osim.TimeSeriesTable(file_dep[0])
            grfs = osim.TimeSeriesTable(file_dep[i])
            time = grfs.getIndependentColumn()
            
            lgrfx = grfs.getDependentColumn('ground_force_l_vx').to_numpy()
            lgrfy = grfs.getDependentColumn('ground_force_l_vy').to_numpy()
            lgrfz = grfs.getDependentColumn('ground_force_l_vz').to_numpy()
            rgrfx = grfs.getDependentColumn('ground_force_r_vx').to_numpy()
            rgrfy = grfs.getDependentColumn('ground_force_r_vy').to_numpy()
            rgrfz = grfs.getDependentColumn('ground_force_r_vz').to_numpy()

            ref_lgrfx = ref_grfs.getDependentColumn('ground_force_l_vx').to_numpy()
            ref_lgrfy = ref_grfs.getDependentColumn('ground_force_l_vy').to_numpy()
            ref_lgrfz = ref_grfs.getDependentColumn('ground_force_l_vz').to_numpy()
            ref_rgrfx = ref_grfs.getDependentColumn('ground_force_r_vx').to_numpy()
            ref_rgrfy = ref_grfs.getDependentColumn('ground_force_r_vy').to_numpy()
            ref_rgrfz = ref_grfs.getDependentColumn('ground_force_r_vz').to_numpy()

            diff_lgrfx = lgrfx - ref_lgrfx 
            diff_lgrfy = lgrfy - ref_lgrfy 
            diff_lgrfz = lgrfz - ref_lgrfz 
            diff_rgrfx = rgrfx - ref_rgrfx 
            diff_rgrfy = rgrfy - ref_rgrfy 
            diff_rgrfz = rgrfz - ref_rgrfz

            percent_time = initial_time + duration * (percent / 100.0)
            index = ref_grfs.getNearestRowIndexForTime(percent_time)
            peak_diff_lgrf[i, 0] = diff_lgrfx[index]
            peak_diff_lgrf[i, 1] = diff_lgrfy[index]
            peak_diff_lgrf[i, 2] = diff_lgrfz[index]
            peak_diff_rgrf[i, 0] = diff_rgrfx[index]
            peak_diff_rgrf[i, 1] = diff_rgrfy[index]
            peak_diff_rgrf[i, 2] = diff_rgrfz[index] 

            if 'experiment' in label:
                initial_index = grfs.getNearestRowIndexForTime(initial_time)
                final_index = grfs.getNearestRowIndexForTime(final_time)
                time = time[initial_index:final_index]
                lgrfx = lgrfx[initial_index:final_index]
                lgrfy = lgrfy[initial_index:final_index]
                lgrfz = lgrfz[initial_index:final_index]
                rgrfx = rgrfx[initial_index:final_index]
                rgrfy = rgrfy[initial_index:final_index]
                rgrfz = rgrfz[initial_index:final_index]

            if 'experiment' in label:
                lw = 4
            elif 'unperturbed'in label:
                lw = 3
            else:
                lw = 2

            pgc_grfs = np.linspace(0, 100, len(time))
            lgrfx_ax.plot(pgc_grfs, lgrfx / BW, label=label, color=color, 
                alpha=alpha, linewidth=lw)
            lgrfx_ax.set_xticklabels([])
            lgrfx_ax.set_ylabel('horizontal force (BW)')
            lgrfx_ax.set_ylim(-0.3, 0.3)
            lgrfx_ax.set_yticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
            lgrfx_ax.set_xlim(0, 100)

            rgrfx_ax.plot(pgc_grfs, rgrfx / BW, label=label, color=color, 
                alpha=alpha, linewidth=lw)
            rgrfx_ax.set_xticklabels([])
            rgrfx_ax.set_ylim(-0.3, 0.3)
            rgrfx_ax.set_yticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
            rgrfx_ax.set_yticklabels([])
            rgrfx_ax.set_xlim(0, 100)

            lgrfy_ax.plot(pgc_grfs, lgrfy / BW, label=label, color=color, 
                alpha=alpha, linewidth=lw)
            lgrfy_ax.set_xticklabels([])
            lgrfy_ax.set_ylabel('vertical force (BW)')
            lgrfy_ax.set_ylim(0, 1.5)
            lgrfy_ax.set_yticks([0, 0.5, 1.0, 1.5])
            lgrfy_ax.set_xlim(0, 100)

            rgrfy_ax.plot(pgc_grfs, rgrfy / BW, label=label, color=color, 
                alpha=alpha, linewidth=lw)
            rgrfy_ax.set_xticklabels([])
            rgrfy_ax.set_ylim(0, 1.5)
            rgrfy_ax.set_yticks([0, 0.5, 1.0, 1.5])
            rgrfy_ax.set_yticklabels([])
            rgrfy_ax.set_xlim(0, 100)

            lgrfz_ax.plot(pgc_grfs, lgrfz / BW, label=label, color=color, 
                alpha=alpha, linewidth=lw)
            lgrfz_ax.set_xlabel('time (% gait cycle)')
            lgrfz_ax.set_ylabel('medio-lateral force (BW)')
            lgrfz_ax.set_ylim(-0.2, 0.2)
            lgrfz_ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
            lgrfz_ax.set_xlim(0, 100)

            rgrfz_ax.plot(pgc_grfs, rgrfz / BW, label=label, color=color, 
                alpha=alpha, linewidth=lw)
            rgrfz_ax.set_xlabel('time (% gait cycle)')
            rgrfz_ax.set_ylim(-0.2, 0.2)
            rgrfz_ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
            rgrfz_ax.set_yticklabels([])
            rgrfz_ax.set_xlim(0, 100)

        for ax in [lgrfx_ax, lgrfy_ax, lgrfz_ax,
                   rgrfx_ax, rgrfy_ax, rgrfz_ax]:
            util.publication_spines(ax)
            for time, color in zip(self.times, self.times_colors):
                time_index = grfs.getNearestRowIndexForTime(
                    initial_time + duration * (time / 100.0))
                ax.axvline(x=time_index, color=color, linestyle='--',
                    linewidth=1, zorder=0)

        fig.tight_layout()
        fig.savefig(target[0], dpi=600)
        plt.close()

        fig2 = plt.figure(figsize=(3, 3))
        ax = fig2.add_subplot(1,1,1)

        ax.bar(self.times,  peak_diff_lgrf[1:, 2], width=7.0,
            color=self.colors[1:])
        ax.set_ylabel('change in medio-lateral GRF (N)')
        ax.set_xticks(self.times)
        ax.set_xlabel('perturbation torque peak time (% gait cycle)')
        util.publication_spines(ax)

        fig2.tight_layout()
        fig2.savefig(target[1], dpi=600)
        plt.close()

        fig3 = plt.figure(figsize=(3, 3))
        ax = fig3.add_subplot(1,1,1)

        ax.bar(self.times,  peak_diff_lgrf[1:, 1], width=7.0,
            color=self.colors[1:])
        ax.set_ylabel('change in vertical GRF (N)')
        ax.set_xticks(self.times)
        ax.set_xlabel('perturbation torque peak time (% gait cycle)')
        util.publication_spines(ax)

        fig3.tight_layout()
        fig3.savefig(target[2], dpi=600)
        plt.close()

        fig4 = plt.figure(figsize=(3, 3))
        ax = fig4.add_subplot(1,1,1)

        ax.bar(self.times,  peak_diff_lgrf[1:, 0], width=7.0,
            color=self.colors[1:])
        ax.set_ylabel('change in horizontal GRF (N)')
        ax.set_xticks(self.times)
        ax.set_xlabel('perturbation torque peak time (% gait cycle)')
        util.publication_spines(ax)

        fig4.tight_layout()
        fig4.savefig(target[3], dpi=600)
        plt.close()
