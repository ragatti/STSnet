"""Palmar grasp dataset
"""

from moabb.datasets.base import BaseDataset
from mne import create_info, Annotations
from mne.io import RawArray

import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import os


class PGHealthy(BaseDataset):
    def __init__(self):
        self.path_fold = None,
        super().__init__(
            subjects=list(range(1, 17)),
            sessions_per_subject=4,
            events={'Fast20': 1, 'Fast60': 2, 'Slow20': 3, 'Slow60': 4},
            code='PGHealthy',
            interval=[0, 6.5],
            paradigm='imagery',
            doi='10.1088/1741-2560/12/5/056013')

    def set_path(self, path=None):
        self.path_fold = path

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        file_path_list = self.data_path(subject)
        sessions = {}
        # sessions['session_1'] = {}
        for sess, path in enumerate(file_path_list):
            data = loadmat(path)
            raw = self._convert_to_raw(data, sess)
            sessions['session_%d' % (sess + 1)] = {}
            sessions['session_%d' % (sess + 1)]['run_1'] = raw
            # sessions['session_1']['run_%d' % (sess + 1)] = raw

        return sessions

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):

        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        if self.path_fold is None:
            raise(ValueError("Set path_fold"))

        path = []
        for ev in self.event_id:
            path.append('{f}/Subject {s}/{e}MVC.mat'.format(f=self.path_fold,
                                                            s=subject,
                                                            e=ev))

        return path

    def _convert_to_raw(self, data, s):
        fs = 500
        ch_names = ['Fp1', 'F5', 'F3', 'F1', 'Fz', 'FC5', 'FC3', 'FC1',
                    'FCz', 'C5', 'C3', 'C1', 'Cz', 'CP5', 'CP3', 'CP1',
                    'CPz', 'P5', 'P3', 'P1', 'Pz']
        ch_types = ['eeg'] * 21
        info = create_info(ch_names, fs, ch_types)
        info['description'] = 'PGHealthy'
        x = data['Signals'] * 1e-6
        raw = RawArray(x, info)
        epoch_start = np.array(data['Epoch_start']).T
        N = len(epoch_start)
        events = np.c_[epoch_start, np.zeros(N), np.ones(N) * (s + 1)]
        events = events.astype(np.int)
        raw.set_montage('standard_1005')
        mapping = {v: k for k, v in self.event_id.items()}
        onsets = events[:, 0] / raw.info['sfreq']
        durations = np.zeros_like(onsets)  # assumes instantaneous events
        descriptions = [mapping[ev_id] for ev_id in events[:, 2]]
        annot_from_events = Annotations(onset=onsets, duration=durations,
                                        description=descriptions)
        raw.notch_filter(50, verbose=False)
        return raw.set_annotations(annot_from_events)

    def get_subject_data(self, subject):
        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))
        raw_dic = self._get_single_subject_data(subject)
        epoch
        for sess in raw:
            for run in raw[sess].values():
                run
        return run


if __name__ == '__main__':
    import os
    from moabb.paradigms import MotorImagery

    print('TEST')
    dataset = PGHealthy()
    path = os.path.join('..', 'datasets', 'Palmar grasp',
                        'Motor execution - healthy')
    dataset.set_path(path)
    paradigm = MotorImagery(n_classes=4)
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[1])

    print(X.shape)
    print(np.unique(y))
    print(meta)
    '''
    raw = dataset._get_single_subject_data(subject=1)
    for sess in raw:
        print(sess)
        for run in raw[sess].values():
            print(run)
            fig = run.plot(start=0, duration=3, n_channels=1, show=True,
                           scalings='auto', block=False)
            plt.show(fig)
    '''
