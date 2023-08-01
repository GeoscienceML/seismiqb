""" Converted geometry: optimized storage. """

import numpy as np
import h5pickle as h5py

from .base import Geometry
from ..utils import repack_hdf5



class GeometryHDF5(Geometry):
    """ Class to work with cubes in HDF5 format.
    We expect a certain structure to the file: mostly, this file should be created by :meth:`ConversionMixin.convert`.

    The file should contain data in one or more projections. When the data is requested, we choose the fastest one
    to actually perform the data reading step.
    Some of the projections may be missing — in this case, other (possibly, slower) projections are used to load data.

    Additional meta attributes like coordinates, SEG-Y parameters, etc, can be also in the same file.

    Refer to the documentation of the base class :class:`Geometry` for more information about attributes and parameters.
    """
    FILE_OPENER = h5py.File

    def init(self, path, mode='r', **kwargs):
        """ Init for HDF5 geometry. The sequence of actions:
            - open file handler
            - check available projections in the file
            - add attributes from file: meta and info about shapes/dtypes.

        Default mode is r to multiple opens for reading.
        If you want to allow other file opens for both read/write, provide 'r+' mode.
        """
        # Open the file
        self.file = self.FILE_OPENER(path, mode, swmr=True)

        # Check available projections
        self.available_axis = [axis for axis, name in self.PROJECTION_NAMES.items()
                               if name in self.file]
        self.available_names = [self.PROJECTION_NAMES[axis] for axis in self.available_axis]

        # Save projection handlers to instance
        self.axis_to_projection = {}
        for axis in self.available_axis:
            name = self.PROJECTION_NAMES[axis]
            projection = self.file[name]

            self.axis_to_projection[axis] = projection

        # Parse attributes from meta / set defaults
        self.add_attributes(**kwargs)

    def add_attributes(self, **kwargs):
        """ Add attributes from the file. """
        # Innate attributes of converted geometry
        self.index_headers = ('INLINE_3D', 'CROSSLINE_3D')
        self.index_length = 2
        self.converted = True

        # Infer attributes from the available projections
        axis = self.available_axis[0]
        projection = self.axis_to_projection[axis]

        shape = np.array(projection.shape)[self.FROM_PROJECTION_TRANSPOSITION[axis]]
        self.shape = shape
        *self.lengths, self.depth = shape

        self.dtype = projection.dtype
        self.quantized = (projection.dtype == np.int8)

        # Get from meta / set defaults
        required_attributes = self.PRESERVED + self.PRESERVED_LAZY + self.PRESERVED_LAZY_CACHED
        meta_exists_and_has_attributes = self.meta_storage.exists and self.meta_storage.has_items(required_attributes)

        if meta_exists_and_has_attributes:
            self.load_meta(keys=self.PRESERVED)
            self.has_stats = True
        else:
            self.set_default_index_attributes(**kwargs)
            self.has_stats = False

    def set_default_index_attributes(self, **kwargs):
        """ Set default values for seismic attributes. """
        self.n_traces = np.prod(self.shape[:2])
        self.delay, self.sample_interval, self.sample_rate = 0.0, 1.0, 1000
        self.compute_dead_traces()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def compute_dead_traces(self, frequency=100):
        """ Fallback for dead traces matrix computation, if no full stats are collected. """
        slides = []

        for idx in range(0, self.depth, frequency):
            slides.append(self.load_slide_native(index=idx, axis=2))

        std_matrix = np.std(slides, axis=0)

        self.dead_traces_matrix = (std_matrix == 0).astype(np.bool_)
        self.n_dead_traces = np.sum(self.dead_traces_matrix)
        self.n_alive_traces = np.prod(self.lengths) - self.n_dead_traces

    # General utilities
    def get_optimal_axis(self, locations=None, shape=None):
        """ Choose the fastest axis from available projections, based on shape. """
        shape = shape or self.locations_to_shape(locations)

        for axis in np.argsort(shape):
            if axis in self.available_axis:
                return axis
        return None


    # Load data: 2D
    def load_slide_native(self, index, axis=0, limits=None, buffer=None, safe=False):
        """ Load slide with public or private API of `h5py`. """
        if safe or buffer is None or buffer.dtype != self.dtype:
            buffer = self.load_slide_native_safe(index=index, axis=axis, limits=limits, buffer=buffer)
        else:
            self.load_slide_native_unsafe(index=index, axis=axis, limits=limits, buffer=buffer)
        return buffer

    def load_slide_native_safe(self, index, axis=0, limits=None, buffer=None):
        """ Load slide with public API of `h5py`. Requires an additional copy to put data into buffer. """
        # Prepare locations
        loading_axis = axis if axis in self.available_axis else self.available_axis[0]
        to_projection_transposition, from_projection_transposition = self.compute_axis_transpositions(loading_axis)

        locations = self.make_slide_locations(index=index, axis=axis)
        locations = [locations[idx] for idx in to_projection_transposition]

        if limits is not None:
            locations[-1] = self.process_limits(limits)
        locations = tuple(locations)

        # Load data
        slide = self.axis_to_projection[loading_axis][locations]

        # Re-order and squeeze the requested axis
        slide = slide.transpose(from_projection_transposition)
        slide = slide.squeeze(axis)

        # Write back to buffer
        if buffer is not None:
            buffer[:] = slide
        else:
            buffer = slide
        return buffer

    def load_slide_native_unsafe(self, index, axis=0, limits=None, buffer=None):
        """ Load slide with private API of `h5py`. Reads data directly into buffer. """
        # Prepare locations
        loading_axis = axis if axis in self.available_axis else self.available_axis[0]
        to_projection_transposition, from_projection_transposition = self.compute_axis_transpositions(loading_axis)

        locations = self.make_slide_locations(index=index, axis=axis)
        locations = [locations[idx] for idx in to_projection_transposition]

        if limits is not None:
            locations[-1] = self.process_limits(limits)
        locations = tuple(locations)

        # View buffer in projections ordering
        buffer = np.expand_dims(buffer, axis)
        buffer = buffer.transpose(to_projection_transposition)

        # Load data
        self.axis_to_projection[loading_axis].read_direct(buffer, locations)

        # View buffer in original ordering
        buffer = buffer.transpose(from_projection_transposition)
        buffer = buffer.squeeze(axis)
        return buffer


    # Load data: 3D
    def load_crop_native(self, locations, axis=None, buffer=None, safe=False):
        """ Load crop with public or private API of `h5py`. """
        axis = axis or self.get_optimal_axis(locations=locations)
        if axis not in self.available_axis:
            raise ValueError(f'Axis={axis} is not available!')

        if safe or axis == 2 or buffer is None or buffer.dtype != self.dtype:
            buffer = self.load_crop_native_safe(locations=locations, axis=axis, buffer=buffer)
        else:
            self.load_crop_native_unsafe(locations=locations, axis=axis, buffer=buffer)
        return buffer

    def load_crop_native_safe(self, locations, axis=None, buffer=None):
        """ Load slide with public API of `h5py`. Requires an additional copy to put data into buffer. """
        # Prepare locations
        to_projection_transposition, from_projection_transposition = self.compute_axis_transpositions(axis)

        locations = [locations[idx] for idx in to_projection_transposition]
        locations = tuple(locations)

        # Load data
        crop = self.axis_to_projection[axis][locations]

        # Re-order back from projections' ordering
        crop = crop.transpose(from_projection_transposition)

        # Write back to buffer
        if buffer is not None:
            buffer[:] = crop
        else:
            buffer = crop
        return buffer

    def load_crop_native_unsafe(self, locations, axis=None, buffer=None):
        """ Load slide with private API of `h5py`. Reads data directly into buffer. """
        # Prepare locations
        to_projection_transposition, from_projection_transposition = self.compute_axis_transpositions(axis)
        locations = [locations[idx] for idx in to_projection_transposition]
        locations = tuple(locations)

        # View buffer in projections ordering
        buffer = buffer.transpose(to_projection_transposition)

        # Load data
        self.axis_to_projection[axis].read_direct(buffer, locations)

        # View buffer in original ordering
        buffer = buffer.transpose(from_projection_transposition)
        return buffer

    def repack_hdf5(self, dst_path=None, projections = (0, ), transform=None, dtype='float32', pbar='t', inplace=False,
                    **dataset_kwargs):
        """ Recreate hdf5 file with conversion and compression. """
        repack_hdf5(self.path, dst_path=dst_path, projections=projections, transform=transform, dtype=dtype, pbar=pbar,
                    inplace=inplace, **dataset_kwargs)
