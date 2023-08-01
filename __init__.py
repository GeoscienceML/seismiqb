"""Init file.
Also disables the OMP warnings, which are produced by Numba or Tensorflow and can't be disabled otherwise.
The change of env variable should be before any imports, relying on it, so we place it on top.
"""
#pylint: disable=wrong-import-position
import os
os.environ['KMP_WARNINGS'] = '0'

from .seismiqb import *
__path__ = [os.path.join(os.path.dirname(__file__), 'seismiqb')]
__version__ = '1.0.0'
