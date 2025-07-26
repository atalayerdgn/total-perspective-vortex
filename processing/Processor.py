import mne
import numpy as np
import os
# class ButterBandpass:
#     def __init__(self, fs, order=5):
#         self.fs = fs
#         self.order = order
#         self.lower = 8.0
#         self.upper = 30 # from motor imagery paper
#     def butter_bandpass(self):
#         nyq = 0.5 * self.fs
#         low = self.lower / nyq
#         high = self.upper / nyq
#         b, a = butter(self.order, [low, high], btype='band')
#         return b, a
#     def apply_filter(self, raw_mne_object):
#         raw_copy = raw_mne_object.copy()
#         stim_channels = mne.pick_types(raw_copy.info, stim=True)
#         data = raw_copy.get_data()
#         if stim_channels.size > 0:
#             channels_to_filter = np.setdiff1d(np.arange(data.shape[0]), stim_channels)
#         else:
#             channels_to_filter = np.arange(data.shape[0])
#         b, a = self.butter_bandpass()
#         data[channels_to_filter, :] = filtfilt(b, a, data[channels_to_filter, :], axis=1)
#         raw_copy._data[:] = data
#         return raw_copy

import numpy as np
import matplotlib.pyplot as plt


def find_scales(frequency, min_hz, max_hz, w0=5):
    scale_for_max_freq = (w0 * frequency) / (2 * np.pi * max_hz)
    scale_for_min_freq = (w0 * frequency) / (2 * np.pi * min_hz)
    return round(scale_for_max_freq), round(scale_for_min_freq) + 1

class Transform:
    def __init__(self, raw, channel_index=0, min_hz=8, max_hz=30, w0=5):
        self.raw = raw
        self.sfreq = self.raw.info['sfreq']
        self.dt = 1 / self.sfreq
        self.channel_index = channel_index
        self.data = self.raw.get_data(picks=self.channel_index)[0]
        min_scale, max_scale = find_scales(self.sfreq, min_hz, max_hz, w0)
        self.num_scales = max_scale - min_scale
        self.scales = np.arange(min_scale, max_scale)
        self.w0 = w0

    def morlet_wavelet(self, t, w0=5):
        return (np.pi**-0.25) * np.exp(1j * w0 * t) * np.exp(-t**2 / 2)

    def compute_cwt(self):
        signal = self.data
        n = len(signal)
        cwt_matrix = np.zeros((len(self.scales), n), dtype=complex)
        for i, a in enumerate(self.scales):
            width = int(10 * a)
            t_wavelet = np.linspace(-width // 2, width // 2, width)
            wavelet_data = self.morlet_wavelet(t_wavelet / a, w0=self.w0)
            wavelet_data = wavelet_data / np.sqrt(abs(a))

            conv_result = np.convolve(signal, wavelet_data.conj(), mode='same')
            cwt_matrix[i, :] = conv_result

        return cwt_matrix, self.scales
    def compute_signal(self):
        cwt_matrix, _ = self.compute_cwt()
        power = np.abs(cwt_matrix) ** 2
        reconstructed = np.real(np.sum(cwt_matrix, axis=0)) / self.num_scales
        return reconstructed
class EEGPRocessor:
    def __init__(self):
        pass
    @staticmethod
    def set_montage(raw_mne_object, montage='standard_1020'):
        raw_mne_object.set_montage(montage, on_missing='ignore')
        return raw_mne_object
    @staticmethod
    def get_events(raw_mne_object, run):
        events, _ = mne.events_from_annotations(raw_mne_object)
        if run in [3, 4, 7, 8, 11, 12]:
            task_id = {
                'left_fist': 1, 
                'right_fist': 2,
            }
        elif run in [5, 6, 9, 10, 13, 14]:
            task_id = {
            'both_fists': 1,
            'both_feet': 2,
            }
        return events, task_id
    @staticmethod
    def apply_cwt(raw_mne_object, channel_index=0, min_hz=8, max_hz=30, w0=5):
        filtered_all = np.zeros_like(raw_mne_object.get_data())
        for ch in range(raw_mne_object.info['nchan']):
            tf = Transform(raw_mne_object, channel_index=ch, min_hz=min_hz, max_hz=max_hz, w0=w0)
            filtered_all[ch, :] = tf.compute_signal()
        return filtered_all

    @staticmethod
    def create_epochs(raw_mne_object, events, task_id, tmin=-1.0, tmax=4.0):
        picks = mne.pick_types(raw_mne_object.info, eeg=True, stim=False)
        epochs = mne.Epochs(raw_mne_object, events, event_id=task_id,
                            tmin=tmin, tmax=tmax, picks=picks,
                            baseline=(None, 0), preload=True)
        return epochs
    @staticmethod
    def save_epochs(epochs, path):
        """        
        Save epochs to a specified path.
        Args:
            epochs (mne.Epochs): The epochs to save.
            path (Path): The path to save the epochs.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        epochs.save(path, overwrite=True)
    def run(subjects, runs):
        path = "../data/MNE-eegbci-data/files/eegmmidb/1.0.0/"
        raws = []
        for subject in subjects:
            for run in runs:
                raw = mne.io.read_raw_edf(f"{path}/S{subject:03d}/S{subject:03d}R{run:02d}.edf", preload=True, stim_channel='auto')
                raws.append(raw)
            raw_concatenated = mne.concatenate_raws(raws)
            raw_concatenated = EEGPRocessor.set_montage(raw_concatenated)
            cwt = EEGPRocessor.apply_cwt(raw_concatenated)
            raw_concatenated._data[:] = cwt
            events, task_id = EEGPRocessor.get_events(raw_concatenated, run)
            epochs = EEGPRocessor.create_epochs(raw_concatenated, events, task_id)
            EEGPRocessor.save_epochs(epochs, f"../data/epochs/S{subject:03d}-epo.fif")
        return subjects

if __name__ == "__main__":
    print("open-close fist events are 3,7,11")
    print("open-close fist or foot events are 5,9,13")
    print("imagine opening-closing fist events are 4,8,12")
    print("imagine opening-closing fist or foot events are 6,10,14")
    print("If you want to process all subjects, enter 'all' for subject scale.")
    runs = input("Enter runs (comma-separated, e.g., 4,6): ")
    subject_scale = input("Enter subject scale (e.g., 1-20 or 'all'): ")
    if subject_scale == 'all':
        subjects = range(1, 110)
    else:
        subjects = []
        for part in subject_scale.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                subjects.extend(range(start, end + 1))
            else:
                subjects.append(int(part))
    runs = [int(run) for run in runs.split(',')]
    subjects = EEGPRocessor.run(subjects, runs)
    with open("../subjects.txt", "w") as f:
        for subject in subjects:
            f.write(f"{subject}\n")
            
    
