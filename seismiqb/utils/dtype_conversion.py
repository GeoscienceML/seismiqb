""" Utils for dtype conversion. """

from functools import partial
import numpy as np

def proba_to_int(x, dtype='int8'):
    """ Convert float probability values in interval [0, 1] to integer values of defined type. """
    min_ = np.iinfo(dtype).min
    ptp = np.iinfo(dtype).max - min_
    return (x * ptp + min_).astype(dtype)

def int_to_proba(x):
    """ Convert integer values to float probability values in interval [0, 1]. """
    min_ = np.iinfo(x.dtype).min
    ptp = np.iinfo(x.dtype).max - min_
    return (x.astype('float32') - min_) / ptp

proba_to_int8  = partial(proba_to_int, dtype='int8')
proba_to_int16 = partial(proba_to_int, dtype='int16')
proba_to_int32 = partial(proba_to_int, dtype='int32')

proba_to_uint8  = partial(proba_to_int, dtype='uint8')
proba_to_uint16 = partial(proba_to_int, dtype='uint16')
proba_to_uint32 = partial(proba_to_int, dtype='uint32')
