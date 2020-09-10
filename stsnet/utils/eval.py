from __future__ import division, print_function, absolute_import
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score


def best_model_file(history, cv):
    best_epoch = np.argmin(history.history['val_loss'])
    best_val_loss = history.history['val_loss'][best_epoch]
    file = 'Model_cv{}-epoch_{:02d}-val_loss_{:.3f}.hdf5'
    return(file.format(cv, (best_epoch + 1), best_val_loss))


def eval_metrics(actual, pred):
    acc = accuracy_score(actual, pred)
    kappa = cohen_kappa_score(actual, pred)
    return acc * 100, kappa
