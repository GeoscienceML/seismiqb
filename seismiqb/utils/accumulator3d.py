""" Accumulator for 3d volumes. """
import os
import shutil
from copy import copy
from multiprocessing.shared_memory import SharedMemory

import blosc
import h5pickle as h5py
import hdf5plugin
import numpy as np

from sklearn.linear_model import LinearRegression

from batchflow import Notifier

from .functions import generate_string, triangular_weights_function_nd, take_along_axis



class Accumulator3D:
    """ Base class to aggregate predicted sub-volumes into a larger 3D cube.
    Can accumulate data in memory (Numpy arrays) or on disk (HDF5 datasets).

    Type of aggregation is defined in subclasses, that must implement `__init__`, `_update` and `_aggregate` methods.
    The main result in subclasses should be stored in `data` attribute, which is accessed by the base class.

    Supposed to be used in combination with `:class:.~RegularGrid` and
    `:meth:.~SeismicCropBatch.update_accumulator` in a following manner:
        - `RegularGrid` defines how to split desired cube range into small crops
        - `Accumulator3D` creates necessary placeholders for a desired type of aggregation
        - `update_accumulator` action of pipeline passes individual crops (and their locations) to
        update those placeholders (see `:meth:~.update`)
        - `:meth:~.aggregate` is used to get the resulting volume
        - `:meth:~.clear` can be optionally used to remove array references and HDF5 file from disk

    This class is an alternative to `:meth:.~SeismicDataset.assemble_crops`, but allows to
    greatly reduce memory footprint of crop aggregation by up to `overlap_factor` times.
    Also, as this class updates rely on `location`s of crops, it can take crops in any order.

    Note that not all pixels of placeholders will be updated with data due to removal of dead traces,
    so we have to be careful with initialization!

    Parameters
    ----------
    shape : sequence
        Shape of the placeholder.
    origin : sequence
        The upper left point of the volume: used to shift crop's locations.
    dtype : np.dtype
        Dtype of storage. Must be either integer or float.
    transform : callable, optional
        Additional function to call before storing the crop data.
    path : str or file-like object, optional
        If provided, then we use HDF5 datasets instead of regular Numpy arrays, storing the data directly on disk.
        After the initialization, we keep the file handle in `w-` mode during the update phase.
        After aggregation, we re-open the file to automatically repack it in `r` mode.
    kwargs : dict
        Other parameters are passed to HDF5 dataset creation.
    """
    #pylint: disable=redefined-builtin
    def __init__(self, shape=None, origin=None, orientation=0, dtype=np.float32, transform=None,
                 format=None, path=None, dataset_kwargs=None, **kwargs):
        # Dimensionality and location, corrected on `orientation`
        self.orientation = orientation
        self.shape = self.reorder(shape)
        self.origin = self.reorder(origin)
        self.location = self.reorder([slice(start, start + shape)
                                      for start, shape in zip(self.origin, self.shape)])

        # Properties of storages
        self.dtype = dtype
        self.transform = getattr(self, transform) if isinstance(transform, str) else transform

        # Container definition
        if format is None:
            format = os.path.splitext(path)[1][1:] if path is not None else 'numpy'
        self.type = format

        if self.type in ['hdf5', 'zarr']:
            if isinstance(path, str) and os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            self.path = path
            self.dataset_kwargs = dataset_kwargs or {}

            if self.type == 'hdf5':
                self.file = h5py.File(path, mode='w-')
            else:
                import zarr #pylint: disable=import-outside-toplevel
                self.file = zarr.group(zarr.LMDBStore(path))

        elif self.type == 'shm':
            self.shm_data = {} # placeholder name -> shm_instance, dtype

        self.placeholders = []

        self.aggregated = False
        self.kwargs = kwargs

    def reorder(self, sequence):
        """ Reorder `sequence` with the `orientation` of accumulator. """
        if self.orientation == 1:
            sequence = np.array([sequence[1], sequence[0], sequence[2]])
        if self.orientation == 2:
            sequence = np.array([sequence[2], sequence[0], sequence[1]])
        return sequence


    # Placeholder management
    def create_placeholder(self, name=None, dtype=None, fill_value=None):
        """ Create named storage as a dataset of HDF5 or plain array. """
        if self.type in ['hdf5', 'qhdf5']:
            placeholder = self.file.create_dataset(name, shape=self.shape, dtype=dtype,
                                                   fillvalue=fill_value, **self.dataset_kwargs)
        elif self.type == 'zarr':
            kwargs = {
                'chunks': (1, *self.shape[1:]),
                **self.dataset_kwargs
            }
            placeholder = self.file.create_dataset(name, shape=self.shape, dtype=dtype,
                                                   fill_value=fill_value, **kwargs)

        elif self.type == 'numpy':
            placeholder = np.full(shape=self.shape, fill_value=fill_value, dtype=dtype)

        elif self.type == 'shm':
            size = np.dtype(dtype).itemsize * np.prod(self.shape)
            shm_name = generate_string(size=10)

            shm = SharedMemory(create=True, size=size, name=shm_name)
            placeholder = np.ndarray(buffer=shm.buf, shape=self.shape, dtype=dtype)
            placeholder[:] = fill_value

            self.shm_data[name] = [shm, dtype]

        self.placeholders.append(name)
        setattr(self, name, placeholder)

    def remove_placeholder(self, name=None, unlink=False):
        """ Remove created placeholder. """
        if self.type in ['hdf5', 'qhdf5', 'zarr']:
            del self.file[name]
        elif self.type == 'shm':
            shm = self.shm_data[name][0]
            shm.close()
            if unlink:
                shm.unlink()
            self.shm_data.pop(name)

        self.placeholders.remove(name)
        setattr(self, name, None)

    def clear(self, unlink=False):
        """ Remove placeholders from memory and disk. """
        if self.type in ['hdf5', 'qhdf5', 'zarr']:
            os.remove(self.path)

        if self.type == 'shm':
            for name in self.placeholders:
                self.remove_placeholder(name, unlink=unlink)

    def __getstate__(self):
        """ Store state of an instance. Remove file handlers and shared memory objects. """
        state = copy(self.__dict__)
        if self.type in ['hdf5', 'qhdf5', 'zarr']:
            for name in self.placeholders:
                state[name] = None

        elif self.type == 'shm':
            shm_data = {}
            for name in self.placeholders:
                shm_instance, dtype = self.shm_data[name]
                shm_data[name] = [shm_instance.name, dtype]
                state[name] = None
            state['shm_data'] = shm_data

        elif self.type == 'numpy':
            for name in self.placeholders:
                array = state[name]
                compressed = blosc.compress_ptr(array.__array_interface__['data'][0], array.size, array.dtype.itemsize)
                state[name] = (array.dtype, array.shape, compressed)
        return state

    def __setstate__(self, state):
        """ Re-create an instance from state. Re-open file handers and shared memory objects. """
        self.__dict__ = state
        if self.type in ['hdf5', 'qhdf5', 'zarr']:
            for name in self.placeholders:
                setattr(self, name, self.file[name])

        elif self.type == 'shm':
            for name in self.placeholders:
                shm_name, dtype = self.shm_data[name]
                shm = SharedMemory(name=shm_name)
                placeholder = np.ndarray(buffer=shm.buf, shape=self.shape, dtype=dtype)
                self.shm_data[name][0] = shm
                setattr(self, name, placeholder)

        elif self.type == 'numpy':
            for name in self.placeholders:
                dtype, shape, compressed = state[name]
                placeholder = np.frombuffer(blosc.decompress(compressed, True), dtype=dtype).reshape(shape)
                setattr(self, name, placeholder)

    def __del__(self):
        if self.type == 'shm':
            self.clear()


    # Store data in accumulator
    def update(self, crop, location):
        """ Update underlying storages in supplied `location` with data from `crop`. """
        if self.aggregated:
            raise RuntimeError('Aggregated data has been already computed!')

        # Check all shapes for compatibility
        for s, slc in zip(crop.shape, location):
            if slc.step and slc.step != 1:
                raise ValueError(f"Invalid step in location {location}")

            if s < slc.stop - slc.start:
                raise ValueError(f"Inconsistent crop_shape {crop.shape} and location {location}")

        # Correct orientation
        location = self.reorder(location)
        if self.orientation == 1:
            crop = crop.transpose(1, 0, 2)
        elif self.orientation == 2:
            crop = crop.transpose(2, 0, 1)

        # Compute correct shapes
        loc, loc_crop = [], []
        for xmin, slc, xmax in zip(self.origin, location, self.shape):
            loc.append(slice(max(0, slc.start - xmin), min(xmax, slc.stop - xmin)))
            loc_crop.append(slice(max(0, xmin - slc.start), min(xmax + xmin - slc.start , slc.stop - slc.start)))
        loc, loc_crop = tuple(loc), tuple(loc_crop)

        # Actual update
        crop = self.transform(crop[loc_crop]) if self.transform is not None else crop[loc_crop]
        self._update(crop, loc)

    def _update(self, crop, location):
        """ Update placeholders with data from `crop` at `locations`. """
        _ = crop, location
        raise NotImplementedError

    def aggregate(self):
        """ Finalize underlying storages to create required aggregation. """
        if self.aggregated:
            raise RuntimeError('All data in the container has already been cleared!')
        self._aggregate()

        # Re-open the HDF5 file to force flush changes and release disk space from deleted datasets
        # Also add alias to `data` dataset, so the resulting cube can be opened by `Geometry`
        # TODO: open resulting HDF5 file with `Geometry` and return it instead?
        self.aggregated = True
        if self.type in ['hdf5', 'qhdf5']:
            if self.orientation == 0:
                projection_name = 'projection_i'
            elif self.orientation == 1:
                projection_name = 'projection_x'
            else:
                projection_name = 'projection_d'

            self.file[projection_name] = self.file['data']
            self.file.close()
            self.file = h5py.File(self.path, 'r+')
            self.data = self.file['data']
        elif self.type == 'zarr':
            self.file.store.flush()
        else:
            if self.orientation == 1:
                self.data = self.data.transpose(1, 0, 2)
            elif self.orientation == 2:
                self.data = self.data.transpose(1, 2, 0)
        return self.data

    def _aggregate(self):
        """ Aggregate placeholders into resulting array. Changes `data` placeholder inplace. """
        raise NotImplementedError

    @property
    def result(self):
        """ Reference to the aggregated result. """
        if not self.aggregated:
            self.aggregate()
        return self.data


    # Utilify methods
    def export_to_hdf5(self, path=None, projections=(0,), pbar='t', dtype=None, transform=None, dataset_kwargs=None):
        """ Export `data` attribute to a file. """
        if self.type != 'numpy' or self.orientation != 0:
            raise NotImplementedError('`export_to_hdf5` works only with `numpy` accumulators with `orientation=0`!')

        # Parse parameters
        from ..geometry.conversion_mixin import ConversionMixin #pylint: disable=import-outside-toplevel
        if isinstance(path, str) and os.path.exists(path):
            os.remove(path)

        dtype = dtype or self.dtype
        transform = transform or (lambda array: array)
        dataset_kwargs = dataset_kwargs or dict(hdf5plugin.Blosc(cname='lz4hc', clevel=6, shuffle=0))

        data = self.data

        with h5py.File(path, mode='w-') as file:
            with Notifier(pbar, total=sum(data.shape[axis] for axis in projections)) as progress_bar:
                for axis in projections:
                    projection_name = ConversionMixin.PROJECTION_NAMES[axis]
                    projection_transposition = ConversionMixin.TO_PROJECTION_TRANSPOSITION[axis]
                    projection_shape = np.array(data.shape)[projection_transposition]

                    dataset_kwargs_ = {'chunks': (1, *projection_shape[1:]), **dataset_kwargs}
                    projection = file.create_dataset(projection_name, shape=projection_shape, dtype=self.dtype,
                                                    **dataset_kwargs_)

                    for i in range(data.shape[axis]):
                        projection[i] = transform(take_along_axis(data, i, axis=axis))
                        progress_bar.update()
        return h5py.File(path, mode='r')


    # Pre-defined transforms
    @staticmethod
    def prediction_to_int8(array):
        """ Convert a float array with values in [0.0, 1.0] to an int8 array with values in [-128, +127]. """
        array *= 255
        array -= 128
        return array.astype(np.int8)

    @staticmethod
    def int8_to_prediction(array):
        """ Convert an int8 array with values in [-128, +127] to a float array with values in [0.0, 1.0]. """
        array = array.astype(np.float32)
        array += 128
        array /= 255
        return array

    @staticmethod
    def prediction_to_uint8(array):
        """ Convert a float array with values in [0.0, 1.0] to an uint8 array with values in [0, 255]. """
        array *= 255
        return array.astype(np.uint8)

    @staticmethod
    def uint8_to_prediction(array):
        """ Convert an uint8 array with values in [0, 255] to a float array with values in [0.0, 1.0]. """
        array = array.astype(np.float32)
        array /= 255
        return array

    @staticmethod
    def prediction_to_uint16(array):
        """ Convert a float array with values in [0.0, 1.0] to an uint16 array with values in [0, 255].
        Useful for accumulators that need to keep track of sum of values.
        """
        array *= 255
        return array.astype(np.uint16)


    # Alternative constructors
    @classmethod
    def from_aggregation(cls, aggregation='max', shape=None, origin=None, dtype=np.float32, fill_value=None,
                         transform=None, format=None, path=None, dataset_kwargs=None, **kwargs):
        """ Initialize chosen type of accumulator aggregation. """
        class_to_aggregation = {
            NoopAccumulator3D: [None, False, 'noop'],
            MaxAccumulator3D: ['max', 'maximum'],
            MeanAccumulator3D: ['mean', 'avg', 'average'],
            StdAccumulator3D: ['std'],
            GMeanAccumulator3D: ['gmean', 'geometric'],
            WeightedSumAccumulator3D: ['weighted'],
            ModeAccumulator3D: ['mode']
        }
        aggregation_to_class = {alias: class_ for class_, lst in class_to_aggregation.items()
                                for alias in lst}

        return aggregation_to_class[aggregation](shape=shape, origin=origin, dtype=dtype, fill_value=fill_value,
                                                 transform=transform, format=format, path=path,
                                                 dataset_kwargs=dataset_kwargs, **kwargs)

    @classmethod
    def from_grid(cls, grid, aggregation='max', dtype=np.float32, fill_value=None, transform=None,
                  format=None, path=None, dataset_kwargs=None, **kwargs):
        """ Infer necessary parameters for accumulator creation from a passed grid. """
        return cls.from_aggregation(aggregation=aggregation, dtype=dtype, fill_value=fill_value,
                                    shape=grid.shape, origin=grid.origin, orientation=grid.orientation,
                                    transform=transform, format=format, path=path, dataset_kwargs=dataset_kwargs,
                                    **kwargs)


class NoopAccumulator3D(Accumulator3D):
    """ Accumulator that applies no aggregation of overlapping crops. """
    def __init__(self, shape=None, origin=None, dtype=np.float32, transform=None, path=None, **kwargs):
        super().__init__(shape=shape, origin=origin, dtype=dtype, transform=transform, path=path, **kwargs)

        self.create_placeholder(name='data', dtype=self.dtype, fill_value=0.0)

    def _update(self, crop, location):
        self.data[location] = crop

    def _aggregate(self):
        pass


class MaxAccumulator3D(Accumulator3D):
    """ Accumulator that takes maximum value of overlapping crops. """
    def __init__(self, shape=None, origin=None, dtype=np.float32, fill_value=None, transform=None, path=None, **kwargs):
        super().__init__(shape=shape, origin=origin, dtype=dtype, transform=transform, path=path, **kwargs)

        min_value = np.finfo(dtype).min if 'float' in dtype.__name__ else np.iinfo(dtype).min
        self.fill_value = fill_value if fill_value is not None else min_value
        self.create_placeholder(name='data', dtype=self.dtype, fill_value=self.fill_value)

    def _update(self, crop, location):
        self.data[location] = np.maximum(crop, self.data[location])

    def _aggregate(self):
        pass


class MeanAccumulator3D(Accumulator3D):
    """ Accumulator that takes mean value of overlapping crops. """
    def __init__(self, shape=None, origin=None, dtype=np.float32, transform=None, path=None, **kwargs):
        if dtype in [np.int8, np.uint8]:
            raise NotImplementedError('`mean` accumulation is unavailable for one-byte dtypes.')
        super().__init__(shape=shape, origin=origin, dtype=dtype, transform=transform, path=path, **kwargs)

        self.create_placeholder(name='data', dtype=self.dtype, fill_value=0)
        self.create_placeholder(name='counts', dtype=np.uint8, fill_value=0)

    def _update(self, crop, location):
        self.data[location] += crop
        self.counts[location] += 1

    def _aggregate(self):
        #pylint: disable=access-member-before-definition
        if self.type == 'hdf5':
            # Amortized updates for HDF5
            for i in range(self.data.shape[0]):
                counts = self.counts[i]
                counts[counts == 0] = 1
                if np.issubdtype(self.dtype, np.floating):
                    self.data[i] /= counts
                else:
                    self.data[i] //= counts

        elif self.type in ['numpy', 'shm']:
            self.counts[self.counts == 0] = 1
            if np.issubdtype(self.dtype, np.floating):
                self.data /= self.counts
            else:
                self.data //= self.counts

        # Cleanup
        self.remove_placeholder('counts', unlink=True)


class StdAccumulator3D(Accumulator3D):
    """ Accumulator that takes std value of overlapping crops. """
    def __init__(self, shape=None, origin=None, dtype=np.float32, transform=None, path=None, **kwargs):
        if dtype not in [np.float32, np.float64]:
            raise ValueError('Dtype should be float32 or float64 for `std` accumulator!')
        super().__init__(shape=shape, origin=origin, dtype=dtype, transform=transform, path=path, **kwargs)

        self.create_placeholder(name='data', dtype=self.dtype, fill_value=0)                 # sum of squared values
        self.create_placeholder(name='sum', dtype=self.dtype, fill_value=0)                  # sum of values
        self.create_placeholder(name='counts', dtype=np.uint8, fill_value=0)

    def _update(self, crop, location):
        self.data[location] += crop ** 2
        self.sum[location] += crop
        self.counts[location] += 1

    def _aggregate(self):
        #pylint: disable=access-member-before-definition
        if self.type == 'hdf5':
            # Amortized updates for HDF5
            for i in range(self.data.shape[0]):
                counts = self.counts[i]
                counts[counts == 0] = 1
                self.data[i] /= counts
                self.sum[i] /= counts

                self.data[i] -= self.sum[i] ** 2
                self.data[i] **= 1/2

        elif self.type in ['numpy', 'shm']:
            self.counts[self.counts == 0] = 1
            self.data /= self.counts
            self.sum /= self.counts

            self.data -= self.sum ** 2
            self.data **= 1/2

        # Cleanup
        self.remove_placeholder('counts', unlink=True)
        self.remove_placeholder('sum', unlink=True)


class GMeanAccumulator3D(Accumulator3D):
    """ Accumulator that takes geometric mean value of overlapping crops. """
    def __init__(self, shape=None, origin=None, dtype=np.float32, transform=None, path=None, **kwargs):
        super().__init__(shape=shape, origin=origin, dtype=dtype, transform=transform, path=path, **kwargs)

        self.create_placeholder(name='data', dtype=self.dtype, fill_value=1)
        self.create_placeholder(name='counts', dtype=np.uint8, fill_value=0)

    def _update(self, crop, location):
        self.data[location] *= crop
        self.counts[location] += 1

    def _aggregate(self):
        #pylint: disable=access-member-before-definition
        if self.type == 'hdf5':
            # Amortized updates for HDF5
            for i in range(self.data.shape[0]):
                counts = self.counts[i]
                counts[counts == 0] = 1

                counts = counts.astype(np.float32)
                counts **= -1
                self.data[i] **= counts

        elif self.type in ['numpy', 'shm']:
            self.counts[self.counts == 0] = 1

            counts = self.counts.astype(np.float32)
            counts **= -1
            self.data **= self.counts

        # Cleanup
        self.remove_placeholder('counts')


class ModeAccumulator3D(Accumulator3D):
    """ Accumulator that takes mode value in overlapping crops. """
    def __init__(self, shape=None, origin=None, dtype=np.float32,
                 n_classes=2, transform=None, path=None, **kwargs):
        # Create placeholder with counters for each class
        self.fill_value = 0
        self.n_classes = n_classes

        shape = (*shape, n_classes)
        origin = (*origin, 0)

        super().__init__(shape=shape, origin=origin, dtype=dtype, transform=transform, path=path, **kwargs)

        self.create_placeholder(name='data', dtype=self.dtype, fill_value=self.fill_value)

    def _update(self, crop, location):
        # Update class counters in location
        crop = np.eye(self.n_classes)[crop]
        self.data[location] += crop

    def _aggregate(self):
        # Choose the most frequently seen class value
        if self.type == 'hdf5':
            for i in range(self.data.shape[0]):
                self.data[i] = np.argmax(self.data[i], axis=-1)

        elif self.type in ['numpy', 'shm']:
            self.data = np.argmax(self.data, axis=-1)


class WeightedSumAccumulator3D(Accumulator3D):
    """ Accumulator that takes weighted sum of overlapping crops. Accepts `weights_function`
    for making weights for each crop into the initialization.

    NOTE: add support of weights incoming along with a data-crop.
    """
    def __init__(self, shape=None, origin=None, dtype=np.float32, transform=None, path=None,
                 weights_function=triangular_weights_function_nd, **kwargs):
        if dtype in [np.int8, np.uint8]:
            raise NotImplementedError('`weighted` accumulation is unavailable for one-byte dtypes.')
        super().__init__(shape=shape, origin=origin, dtype=dtype, transform=transform, path=path, **kwargs)

        self.create_placeholder(name='data', dtype=self.dtype, fill_value=0)
        self.create_placeholder(name='weights', dtype=np.float32, fill_value=0)
        self.weights_function = weights_function
        self.crop_weights = None

    def _update(self, crop, location):
        # Weights matrix for the incoming crop
        if self.crop_weights is None or self.crop_weights.shape != crop.shape:
            self.crop_weights = self.weights_function(crop)

        self.data[location] = ((self.crop_weights * crop + self.data[location] * self.weights[location]) /
                               (self.crop_weights + self.weights[location]))
        self.weights[location] += self.crop_weights

    def _aggregate(self):
        # Cleanup
        self.remove_placeholder('weights', unlink=True)


class RegressionAccumulator(Accumulator3D):
    """ Accumulator that fits least-squares regression to scale values of
    each incoming crop to match values of the overlap. In doing so, ignores nan-values.
    For aggregation uses weighted sum of crops. Weights-making for crops is controlled by
    `weights_function`-parameter.

    Parameters
    ----------
    shape : sequence
        Shape of the placeholder.
    origin : sequence
        The upper left point of the volume: used to shift crop's locations.
    dtype : np.dtype
        Dtype of storage. Must be either integer or float.
    transform : callable, optional
        Additional function to call before storing the crop data.
    path : str or file-like object, optional
        If provided, then we use HDF5 datasets instead of regular Numpy arrays, storing the data directly on disk.
        After the initialization, we keep the file handle in `w-` mode during the update phase.
        After aggregation, we re-open the file to automatically repack it in `r` mode.
    weights_function : callable
        Function that accepts a crop and returns matrix with weights of the same shape. Default scheme
        involves using larger weights in the crop-centre and lesser weights closer to the crop borders.
    rsquared_lower_bound : float
        Can be a number between 0 and 1 or `None`. If set to `None`, we use each incoming crop with
        predictions to update the assembled array. Otherwise, we use only those crops, that fit already
        filled data well enough, requiring r-squared of linear regression to be larger than the supplied
        parameter.
    regression_target : str
        Can be either 'assembled' (same as 'accumulated') or 'crop' (same as 'incoming'). If set to
        'assembled', the regression considers new crop as a regressor and already filled overlap as a target.
        If set to 'crop', incoming crop is the target in the regression. The choice of 'assembled'
        should yield more stable results.

    NOTE: As of now, relies on the order in which crops with data arrive. When the order of
    supplied crops is different, the result of aggregation might differ as well.
    """
    def __init__(self, shape=None, origin=None, dtype=np.float32, transform=None, path=None,
                 weights_function=triangular_weights_function_nd, rsquared_lower_bound=.2,
                 regression_target='assembled', **kwargs):
        super().__init__(shape=shape, origin=origin, dtype=dtype, transform=transform, path=path, **kwargs)

        # Fill both placeholders with nans: in order to fit the regression
        # it is important to understand what overlap values are already filled.
        # NOTE: perhaps rethink and make weighted regression.
        self.create_placeholder(name='data', dtype=self.dtype, fill_value=np.nan)
        self.create_placeholder(name='weights', dtype=np.float32, fill_value=np.nan)

        self.weights_function = weights_function
        self.rsquared_lower_bound = rsquared_lower_bound or -1

        if regression_target in ('assembled', 'accumulated'):
            self.regression_target = 'assembled'
        elif regression_target in ('crop', 'incoming'):
            self.regression_target = 'crop'
        else:
            raise ValueError(f'Unknown regression target {regression_target}.')

    def _update(self, crop, location):
        # Scale incoming crop to better fit already filled data.
        # Fit is done via least-squares regression.
        overlap_data = self.data[location]
        overlap_weights = self.weights[location]
        crop_weights = self.weights_function(crop)

        # If some of the values are already filled, use regression to fit new crop
        # to what's filled.
        overlap_indices = np.where((~np.isnan(overlap_data)) & (~np.isnan(crop)))
        new_indices = np.where(np.isnan(overlap_data))

        if len(overlap_indices[0]) > 0:
            # Take overlap values from data-placeholder and the crop.
            # Select regression/target according to supplied parameter `regression_target`.
            if self.regression_target == 'assembled':
                xs, ys = crop[overlap_indices], overlap_data[overlap_indices]
            else:
                xs, ys = overlap_data[overlap_indices], crop[overlap_indices]

            # Fit new crop to already existing data and transform the crop.
            model = LinearRegression()
            model.fit(xs.reshape(-1, 1), ys.reshape(-1))

            # Calculating the r-squared of the fitted regression.
            a, b = model.coef_[0], model.intercept_
            xs, ys = xs.reshape(-1), ys.reshape(-1)
            rsquared = 1 - ((a * xs + b - ys) ** 2).mean() / ((ys - ys.mean()) ** 2).mean()

            # If the fit is bad (r-squared is too small), ignore the incoming crop.
            # If it is of acceptable quality, use it to update the assembled-array.
            if rsquared > self.rsquared_lower_bound:
                if self.regression_target == 'assembled':
                    crop = a * crop + b
                else:
                    crop = (crop - b) / a

                # Update location-slice with weighted average.
                overlap_data[overlap_indices] = ((overlap_weights[overlap_indices] * overlap_data[overlap_indices]
                                                + crop_weights[overlap_indices] * crop[overlap_indices]) /
                                                (overlap_weights[overlap_indices] + crop_weights[overlap_indices]))

                # Update weights over overlap.
                overlap_weights[overlap_indices] += crop_weights[overlap_indices]

                # Use values from crop to update the region covered by the crop and not yet filled.
                self.data[location][new_indices] = crop[new_indices]
                self.weights[location][new_indices] = crop_weights[new_indices]
        else:
            self.data[location] = crop
            self.weights[location] = crop_weights

    def _aggregate(self):
        # Clean-up
        self.remove_placeholder('weights')
