""" Seismic Crop Batch. """
import os
import traceback
from warnings import warn
from functools import wraps
from inspect import signature

import numpy as np
import cv2
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.signal import butter, sosfiltfilt

from batchflow import DatasetIndex, Batch
from batchflow import action, any_action_failed, SkipBatchException
from batchflow import apply_parallel as apply_parallel_decorator

from .visualization_batch import VisualizationMixin
from ..labels import Horizon
from ..utils import to_list, groupby_all
from .. import functional

from ..labels.fault import skeletonize



def add_methods(method_names):
    """ Add augmentations to batch class. """
    def _add_methods(cls):
        def create_batch_method(method_name):
            method = getattr(functional, method_name)
            requires_rng = 'rng' in signature(method).parameters

            @wraps(method)
            def wrapper(self, _, buffer, *args, src=None, dst=None, **kwargs):
                _ = src, dst
                buffer[:] = method(buffer, *args, **kwargs)
            wrapper = cls.use_apply_parallel(wrapper, requires_rng=requires_rng)
            return wrapper

        for method_name in method_names:
            setattr(cls, method_name, create_batch_method(method_name))
        return cls
    return _add_methods

@add_methods(['rotate_2d', 'rotate_3d', 'scale_2d', 'scale_3d',
              'affine_transform', 'perspective_transform', 'elastic_transform',
              'compute_instantaneous_amplitude', 'compute_instantaneous_phase', 'compute_instantaneous_frequency'])
class SeismicCropBatch(Batch, VisualizationMixin):
    """ Batch with ability to generate 3d-crops of various shapes.

    The first action in any pipeline with this class should be `make_locations` to transform batch index from
    individual cubes into crop-based indices. The transformation uses randomly generated postfix (see `:meth:.salt`)
    to obtain unique elements.
    """
    apply_defaults = {
        'init': 'preallocating_init',
        'post': 'noop_post',
        'target': 'for',
    }

    # Inner workings
    @action
    def add_components(self, components, init=None):
        """ Add new components, checking that attributes of the same name are not present in dataset.

        Parameters
        ----------
        components : str or list
            new component names
        init : array-like
            initial component data

        Raises
        ------
        ValueError
            If a component or an attribute with the given name already exists in batch or dataset.
        """
        for component in to_list(components):
            if hasattr(self.dataset, component):
                msg = f"Component with `{component}` name cannot be added to batch, "\
                      "since attribute with this name is already present in dataset."
                raise ValueError(msg)
        super().add_components(components=components, init=init)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if isinstance(value, np.ndarray) and value.ndim == 4 and name not in self.name_to_order:
            self.name_to_order[name] = np.array(['i', 'x', 'd'])

    def get(self, item=None, component=None):
        """ Custom access for batch attributes.
        If `component` is one of the registered components, get it, optionally indexed with `item`.
        Otherwise, try to get `component` as the attribute of a field, corresponding to `item`.
        """
        if component in self.components:
            # Faster, than `getattr(self, component)`, as we already know that the component exists
            data = self._data[component]
            if item is not None:
                return data[item]

        elif component in self.__dict__:
            data = self.__dict__[component]
            if item is not None:
                return data[item]

        else: # retrieve from `dataset`
            if item is not None:
                field_name = self._data['field_names'][item]
                field = self.dataset.fields[field_name]
                if component == 'fields':
                    return field
                return getattr(field, component)

            # Not often used, mainly for introspection/debug. Not optimized
            data = getattr(self.dataset, component)

        return data

    def deepcopy(self, preserve=False):
        """ Create a copy of a batch with the same `dataset` and `pipeline` references. """
        #pylint: disable=protected-access
        new = super().deepcopy()

        if preserve:
            new._dataset = self.dataset
            new.pipeline = self.pipeline
        return new


    # Batch action parallelization
    def preallocating_init(self, src=None, dst=None, buffer_type=None, return_indices=True, **kwargs):
        """ Preallocate a buffer. """
        if src is None and dst is None:
            raise ValueError('Specify either `src` or `dst`!.')
        dst = dst if dst is not None else src

        # Check if the operation can be done in-place
        inplace = False
        if src is None:
            inplace = False
        if dst == src:
            inplace = True

        if inplace:
            buffer = self.get(component=src)
        else:
            if src is None or buffer_type is not None:
                dtype = np.float32 # TODO: determine based on geometries
                buffer = (getattr(np, buffer_type))((len(self), *self.crop_shape), dtype=dtype)
            else:
                buffer = self.get(component=src).copy()
            self.add_components(dst, buffer)

        return list(zip(self.indices, buffer)) if return_indices else buffer

    def noop_post(self, all_results, **kwargs):
        """ Check for errors after the action, otherwise does nothing. """
        _ = kwargs

        if any_action_failed(all_results):
            all_errors = self.get_errors(all_results)
            print(all_errors)
            traceback.print_tb(all_errors[0].__traceback__)
            raise RuntimeError("Could not assemble the batch!") from all_errors[0]
        return self

    def normalize_post(self, all_results, func, src=None, mode='meanstd', **kwargs):
        """ Post function to store"""
        self.noop_post(all_results, **kwargs)
        normalization_stats = [item[1] for item in all_results]
        self.add_components(f'normalization_stats_{src}', normalization_stats)
        return self

    # Core actions
    @action
    def make_locations(self, generator, batch_size=None, keep_attributes=None):
        """ Use `generator` to create `batch_size` locations.
        Each location defines position in a cube and can be used to retrieve data/create masks at this place.

        Generator can be either Sampler or Grid to make locations in a random or deterministic fashion.
        `generator` must be a callable and return (batch_size, 9+) array, where the first nine columns should be:
        (field_id, label_id, orientation, i_start, x_start, d_start, i_stop, x_stop, d_stop).
        `generator` must have `to_names` method to convert cube and label ids into actual strings.

        Alternatively, `generator` may be a ready-to-use ndarray with the same structure. In this case, `to_names`
        is called directly from dataset. Should be used with caution and mainly for debugging purposes.

        Field and label ids are transformed into names of actual fields and labels (horizons, faults, facies, etc).
        Then we create a new instance of `SeismicCropBatch`, where the index is set to a enumeration of locations.

        After parsing contents of generated (batch_size, 9+)-shaped array we add following attributes:
            - `locations` with triplets of slices
            - `orientations` with crop orientation: 0 for iline direction, 1 for crossline direction
            - `shapes`
            - `crop_shape`, computed from `shapes`
            - `field_names`
            - `label_names`
            - `generated` with originally generated data
        If `generator` creates more than 9 columns, they are not used, but still stored in the  `generated` attribute.

        Parameters
        ----------
        generator : callable or np.ndarray
            Sampler or Grid to retrieve locations. Must be a callable that works off of a positive integer.
        batch_size : int
            Number of locations to generate.
        keep_attributes : str or sequence of str
            Components to keep in a newly created batch.

        Returns
        -------
        SeismicCropBatch
            A new instance of Batch.
        """
        # Get ndarray with `locations` and `orientations`, convert IDs to names, that are used in dataset
        if callable(generator):
            generated = generator(batch_size)
            field_names, label_names = generator.to_names(generated[:, [0, 1]]).T
        elif isinstance(generator, np.ndarray):
            generated = generator
            field_names, label_names = self.dataset.to_names(generated[:, [0, 1]]).T
        else:
            raise ValueError(f'`generator` should either be callable or ndarray, got {type(generator)} instead!')

        # Locations: 3D slices in the cube coordinates
        locations = [[slice(i_start, i_stop), slice(x_start, x_stop), slice(d_start, d_stop)]
                      for i_start, x_start, d_start, i_stop,  x_stop,  d_stop in generated[:, 3:9]]

        # Additional info
        orientations = generated[:, 2]
        shapes = generated[:, [6, 7, 8]] - generated[:, [3, 4, 5]]
        crop_shape = shapes[0] if orientations[0] == 0 else shapes[0][[1, 0, 2]]

        # Create a new SeismicCropBatch instance
        new_index = np.arange(len(locations), dtype=np.int32)
        new_batch = type(self)(DatasetIndex.from_index(index=new_index))

        # Keep chosen components in the new batch
        if keep_attributes:
            keep_attributes = [keep_attributes] if isinstance(keep_attributes, str) else keep_attributes
            for component in keep_attributes:
                if hasattr(self, component):
                    new_batch.add_components(component, self.get(component=component))

        # Set all freshly computed attributes. Manually keep the reference to the `pipeline`
        # Note: `pipeline` would be set by :meth:`~.Pipeline._exec_one_action` anyway, so this is not necessary.
        new_batch.add_components(('locations', 'generated', 'shapes', 'orientations', 'field_names', 'label_names'),
                                 (locations, generated, shapes, orientations, field_names, label_names))
        new_batch.crop_shape = crop_shape
        new_batch.name_to_order = {}
        new_batch.pipeline = self.pipeline
        return new_batch


    # Loading of cube data and its derivatives
    @apply_parallel_decorator(init='preallocating_init', post='noop_post', buffer_type='empty', target='for')
    def load_seismic(self, ix, buffer, dst, src=None, src_geometry='geometry', **kwargs):
        """ Load data from cube for stored `locations`.

        Parameters
        ----------
        dst : str
            Component of batch to put loaded crops in.
        slicing : str
            If 'custom', use `load_crop` method to make crops.
            if 'native', crop will be loaded as a slice of geometry. Preferred for 3D crops to speed up loading.
        src_geometry : str
            Field attribute with desired geometry.
        """
        field = self.get(ix, 'fields')
        locations = self.get(ix, 'locations')
        orientation = self.get(ix, 'orientations')

        if orientation == 1:
            buffer = buffer.transpose(1, 0, 2)
        field.load_seismic(locations=locations, src=src_geometry, buffer=buffer, **kwargs)

    load_cubes = load_crops = load_seismic


    @apply_parallel_decorator(init='preallocating_init', post='normalize_post', target='for')
    def normalize(self, ix, buffer, src, dst=None, mode='meanstd', stats=None, clip_to_quantiles=None):
        """ Normalize `src` with provided stats.
        Depending on the parameters, stats for normalization will be taken from (in order of priority):
            - supplied `stats`, if provided
            - the field that created this `src`, if `stats=True` or `stats='field'`
            - from `normalization_stats_{stats}` component (each `normalize` action put used statistics into
              normalization_stats_{src} component)
            - computed from `src` data directly

        Parameters
        ----------
        mode : {'mean', 'std', 'meanstd', 'minmax'}, callable or None
            If str, then normalization description.
            If callable, then it will be called on `src` data with additional `stats` argument.
            If None, `mode` from normalizer instance will be used.
        stats : dict or str, optional
            If provided, then used to get statistics for normalization.
            If dict, stats for each field.
            If 'field', field normalization statistics will be used.
            If other str, `normalization_stats_{stats}` will be used.
            If None, item statistics will be used.
        clip_to_quantiles : bool
            Whether to clip the data to quantiles, specified by `q` parameter.
            Quantile values are taken from `stats`, provided by either of the ways.
        """
        field = self.get(ix, 'fields')

        # Prepare normalization stats
        if isinstance(stats, dict):
            if field.short_name in stats:
                stats = stats[field.short_name]
        elif stats in {'field', True}:
            stats = field.normalization_stats
        elif isinstance(stats, str):
            stats = getattr(self, f'normalization_stats_{stats}')[ix]

        buffer, stats = field.normalizer.normalize(buffer, normalization_stats=stats, mode=mode,
                                                   return_stats=True, inplace=True)
        return buffer, stats

    @apply_parallel_decorator(init='preallocating_init', post='noop_post', target='for')
    def denormalize(self, ix, buffer, src, dst=None, mode=None, stats=None):
        """ Denormalize images using provided statistics.

        Parameters
        ----------
        mode : {'mean', 'std', 'meanstd', 'minmax'}, callable or None
            If str, then normalization description.
            If callable, then it will be called on `src` data with additional `stats` argument.
            If None, `mode` from normalizer instance will be used.
        stats : dict or str, optional
            If provided, then used to get statistics for normalization.
            If dict, stats for each field.
            If 'field', field normalization statistics will be used.
            If other str, `normalization_stats_{stats}` will be used.
        """
        field = self.get(ix, 'fields')

        # Prepare normalization stats
        if isinstance(stats, dict):
            if field.short_name in stats:
                stats = stats[field.short_name]
        elif stats in {'field', True}:
            stats = field.normalization_stats
        elif isinstance(stats, str):
            stats = getattr(self, f'normalization_stats_{stats}')[ix]

        buffer = field.normalizer.denormalize(buffer, normalization_stats=stats, mode=mode, inplace=True)
        return buffer

    @apply_parallel_decorator(init='preallocating_init', post='noop_post', target='for')
    def quantize(self, ix, buffer, src, dst=None):
        """ Quantize image. """
        field = self.get(ix, 'fields')
        buffer[:] = field.quantizer.quantize(buffer)
        return buffer

    @apply_parallel_decorator(init='preallocating_init', post='noop_post', target='for')
    def dequantize(self, ix, buffer, src, dst=None):
        """ Dequantize image (lossy). """
        field = self.get(ix, 'fields')
        buffer[:] = field.quantizer.dequantize(buffer)
        return buffer

    @apply_parallel_decorator(init='indices', post='_assemble', target='for')
    def compute_attribute(self, ix, dst, src='images', attribute='semblance', window=10, stride=1, device='cpu'):
        """ Compute geological attribute.

        Parameters
        ----------
        dst : str
            Destination batch component
        src : str, optional
            Source batch component, by default 'images'
        attribute : str, optional
            Attribute to compute, by default 'semblance'
        window : int or tuple, optional
            Window to compute attribute, by default 10 (for each axis)
        stride : int, optional
            Stride for windows, by default 1 (for each axis)
        device : str, optional
            Device to compute attribute, by default 'cpu'

        Returns
        -------
        SeismicCropBatch
            Batch with loaded masks in desired components.
        """
        from ..utils.layers import compute_attribute #pylint: disable=import-outside-toplevel
        image = self.get(ix, src)
        result = compute_attribute(image, window, device, attribute)
        return result


    # Loading of labels
    @action
    @apply_parallel_decorator(init='preallocating_init', post='noop_post', buffer_type='zeros', target='for')
    def create_masks(self, ix, buffer, dst, src=None, indices='all', width=3, src_labels='labels',
                     sparse=False, **kwargs):
        """ Create masks from labels in stored `locations`.

        Parameters
        ----------
        dst : str
            Component of batch to put loaded masks in.
        indices : str, int or sequence of ints
            Which labels to use in mask creation.
            If 'all', then use all labels.
            If 'single' or `random`, then use one random label.
            If int or array-like, then element(s) are interpreted as indices of desired labels.
        width : int
            Width of the resulting label.
        src_labels : str
            Dataset attribute with labels dict.
        sparse : bool, optional
            Whether create sparse mask (only on labeled slides) or not, by default False. Unlabeled
            slides will be filled with -1.
        """
        field = self.get(ix, 'fields')
        locations = self.get(ix, 'locations')
        orientation = self.get(ix, 'orientations')

        if orientation == 1:
            buffer = buffer.transpose(1, 0, 2)
        field.make_mask(locations=locations, orientation=orientation, buffer=buffer,
                        width=width, indices=indices, src=src_labels, sparse=sparse, **kwargs)


    @action
    @apply_parallel_decorator(init='indices', post='_assemble', target='for')
    def create_regression_masks(self, ix, dst, src=None, indices='all', src_labels='labels', scale=False):
        """ Create masks with relative depth. """
        field = self.get(ix, 'fields')
        location = self.get(ix, 'locations')
        return field.make_regression_mask(location=location, indices=indices, src=src_labels, scale=scale)


    @action
    @apply_parallel_decorator(init='indices', post='_assemble', target='for')
    def compute_label_attribute(self, ix, dst, src='amplitudes', atleast_3d=True, dtype=np.float32, **kwargs):
        """ Compute requested attribute along label surface. Target labels are defined by sampled locations.

        Parameters
        ----------
        src : str
            Keyword that defines label attribute to compute.
        atleast_3d : bool
            Whether add one more dimension to 2d result or not.
        dtype : valid dtype compatible with requested attribute
            A dtype that result must have.
        kwargs : misc
            Passed directly to one of attribute-evaluating methods.

        Notes
        -----
        Correspondence between the attribute and the method that computes it
        is defined by :attr:`~Horizon.ATTRIBUTE_TO_METHOD`.

        TODO: can be improved with `preallocating_init`
        """
        field = self.get(ix, 'fields')
        location = self.get(ix, 'locations')
        label_index = self.get(ix, 'generated')[1]
        src = src.replace('*', str(label_index))

        result = field.load_attribute(src=src, location=location, atleast_3d=atleast_3d, dtype=dtype, **kwargs)
        return result


    # Rebatch and its callables
    @action
    def rebatch_on_condition(self, src=None, condition='area', threshold=None, keep_attributes=None, **kwargs):
        """ Compute a condition on each item of `src`, keep only elements that returned value bigger than `threshold`.
        Modifies (slices) all of the components in a batch instance, as well as its index.

        Parameters
        ----------
        src : str
            Name of the component to use as input for `condition`.
        condition : callable, {'area', 'discontinuity_size'}
            If callable, then applied to each item of `src`.
            If 'area', then labeled area of a mask is computed, using the specified axis for projection.
            If 'discontinuity_size', then we compute the biggest discontinuity size.
        threshold : number
            A value to compare computed conditions against.
        keep_attributes : sequence, optional
            Additional batch attributes to slice with the new index.
        kwargs : dict
            Passed directly to `condition` function.
        """
        # Select correct function to compute
        if callable(condition):
            pass
        elif condition == 'area':
            condition = self._compute_mask_area
        elif condition == 'discontinuity_size':
            condition = self._compute_discontinuity_size
        elif condition == 'crop_area':
            condition = self._compute_crop_area

        # Compute indices to keep
        data = self.get(component=src)
        indices = np.array([i for i, item in enumerate(data)
                            if condition(item, **kwargs) > threshold])

        if len(indices) > 0:
            self.index = DatasetIndex.from_index(index=indices)
        else:
            raise SkipBatchException

        # Re-index components and additional attributes passed
        keep_attributes = keep_attributes = keep_attributes or []
        keep_attributes += list(self.components or [])
        keep_attributes = list(set(keep_attributes))

        for component in keep_attributes:
            component_data = self.get(component=component)
            if isinstance(component_data, np.ndarray):
                component_data = component_data[indices]
            else:
                component_data = [component_data[i] for i in indices]
            setattr(self, component, component_data)
        return self

    @staticmethod
    def _compute_mask_area(array, axis=-1, **kwargs):
        """ Compute the area of a projection (along the `axis`, by default depth), of a horizon mask. """
        _ = kwargs
        labeled_traces = array.max(axis=axis)
        area = labeled_traces.sum() / labeled_traces.size
        return area

    @staticmethod
    def _compute_crop_area(array, axis=(1, 2), **kwargs):
        """ Compute the area of a projection (along the `axis`, by default depth), of a horizon mask. """
        _ = kwargs
        return 1 - np.isnan(array).sum(axis=axis) / array.size

    @staticmethod
    def _compute_discontinuity_size(array, **kwargs):
        """ Compute the size of the biggest discontinuity (allegedly, fault) in the horizon mask.
        Assumes the array in (inline, crossline, depth) orientation. Tested mostly for 2D crops.
        """
        _ = kwargs

        # Get point cloud of labeled points. For each trace (for each (i, x) pixel) compute depth stats
        points = np.array(np.nonzero(array)).T                        # (iline, xline, depth) point cloud
        points = groupby_all(points)                                  # (iline, xline, _, min_depth, max_depth, _)
        condition = points[:-1, 1] == points[1:, 1] - 1               # get only sequential traces

        # Upper/lower bounds
        mins = points[:-1, 3][condition]
        mins_next = points[1:, 3][condition]
        upper = np.max(np.array([mins, mins_next]), axis=0)           # maximum values of `min_depth` for each pixel

        maxs = points[:-1, 4][condition]
        maxs_next = points[1:, 4][condition]
        lower = np.min(np.array([maxs, maxs_next]), axis=0)           # minimum values of `max_depth` for each pixel

        return (upper - lower).max()


    # Methods to work with (mostly, horizon) masks
    @apply_parallel_decorator(init='preallocating_init', post='noop_post', target='for')
    def filter_sides(self, _, buffer, ratio, side, axis=0, src=None, dst=None):
        """ Filter out left or right side of a crop.
        Assumes the array in (inline, crossline, depth) orientation. Tested mostly for 2D crops.

        Parameters
        ----------
        ratio : float
            The ratio of the crop lines to be filtered out.
        side : str
            Which side to filter out. Possible options are 'left' or 'right'.
        """
        if not 0 <= ratio <= 1:
            raise ValueError(f"Invalid value ratio={ratio}: must be a float in [0, 1] interval.")

        # Get the amount of crop lines and kept them on the chosen crop part
        max_len = buffer.shape[axis]
        length = round(max_len * (1 - ratio))

        locations = [slice(None)] * 3
        locations[axis] = slice(0, max_len-length) if side == 'left' else slice(length, max_len)
        buffer[tuple(locations)] = 0


    @apply_parallel_decorator(init='data', post='_assemble', target='for')
    def shift_masks(self, crop, n_segments=3, max_shift=4, min_len=5, max_len=10, src=None, dst=None):
        """ Randomly shift parts of the crop up or down.

        Parameters
        ----------
        n_segments : int
            Number of segments to shift.
        max_shift : int
            Max size of shift along vertical axis.
        min_len : int
            Min size of shift along horizontal axis.
        max_len : int
            Max size of shift along horizontal axis.
        """
        crop = np.copy(crop)
        for _ in range(n_segments):
            # Point of starting the distortion, its length and size
            begin = np.random.randint(0, crop.shape[1])
            length = np.random.randint(min_len, max_len)
            shift = np.random.randint(-max_shift, max_shift)

            # Apply shift
            if shift != 0:
                segment_to_shift = crop[:, begin:min(begin + length, crop.shape[1]), :]
                shifted_segment = np.roll(segment_to_shift, shift=shift, axis=-1)
                crop[:, begin:min(begin + length, crop.shape[1]), :] = shifted_segment
        return crop

    @apply_parallel_decorator(init='data', post='_assemble', target='for')
    def bend_masks(self, crop, angle=10, src=None, dst=None):
        """ Rotate part of the mask on a given angle.
        Must be used for crops in (xlines, depths, inlines) format.

        Parameters
        ----------
        angle : float
            Rotation angle in degrees.
        """
        shape = crop.shape
        point_x = np.random.randint(0, shape[0])
        point_d = int(np.argmax(crop[point_x, :, :]))

        if np.sum(crop[point_x, point_d, :]) == 0.0:
            return crop

        matrix = cv2.getRotationMatrix2D((point_d, point_x), angle, 1)
        rotated = cv2.warpAffine(crop, matrix, (shape[1], shape[0])).reshape(shape)

        combined = np.zeros_like(crop)
        if point_x >= shape[0]//2:
            combined[:point_x, :, :] = crop[:point_x, :, :]
            combined[point_x:, :, :] = rotated[point_x:, :, :]
        else:
            combined[point_x:, :, :] = crop[point_x:, :, :]
            combined[:point_x, :, :] = rotated[:point_x, :, :]
        return combined

    @apply_parallel_decorator(init='data', post='_assemble', target='for')
    def linearize_masks(self, crop, n=3, shift=0, kind='random', width=None, src=None, dst=None):
        """ Sample `n` points from the original mask and create a new mask by interpolating them.

        Parameters
        ----------
        n : int
            Number of points to sample.
        shift : int
            Maximum amplitude of random shift along the depths axis.
        kind : {'random', 'linear', 'slinear', 'quadratic', 'cubic', 'previous', 'next'}
            Type of interpolation to use. If 'random', then chosen randomly for each crop.
        width : int
            Width of interpolated lines.
        """
        # Parse arguments
        if kind == 'random':
            kind = np.random.choice(['linear', 'slinear', 'quadratic', 'cubic', 'previous', 'next'])
        if width is None:
            width = np.sum(crop, axis=2)
            width = int(np.round(np.mean(width[width!=0])))

        # Choose the anchor points
        axis = 1 - np.argmin(crop.shape)
        *nz, _ = np.nonzero(crop)
        min_, max_ = nz[axis][0], nz[axis][-1]
        idx = [min_, max_]

        step = (max_ - min_) // n
        for i in range(0, max_-step, step):
            idx.append(np.random.randint(i, i + step))

        # Put anchors into new mask
        mask_ = np.zeros_like(crop)
        slc = (idx if axis == 0 else slice(None),
               idx if axis == 1 else slice(None),
               slice(None))
        mask_[slc] = crop[slc]
        *nz, y = np.nonzero(mask_)

        # Shift depths randomly
        x = nz[axis]
        y += np.random.randint(-shift, shift + 1, size=y.shape)

        # Sort and keep only unique values, based on `x` to remove width of original mask
        sort_indices = np.argsort(x)
        x, y = x[sort_indices], y[sort_indices]
        _, unique_indices = np.unique(x, return_index=True)
        x, y = x[unique_indices], y[unique_indices]

        # Interpolate points; put into mask
        interpolator = interp1d(x, y, kind=kind)
        indices = np.arange(min_, max_, dtype=np.int32)
        depths = interpolator(indices).astype(np.int32)

        slc = (indices if axis == 0 else indices * 0,
               indices if axis == 1 else indices * 0,
               np.clip(depths, 0, crop.shape[2]-1))
        mask_ = np.zeros_like(crop)
        mask_[slc] = 1

        # Make horizon wider
        structure = np.ones((1, width), dtype=np.uint8)
        shape = mask_.shape
        mask_ = mask_.reshape((mask_.shape[axis], mask_.shape[2]))
        mask_ = cv2.dilate(mask_, kernel=structure, iterations=1).reshape(shape)
        return mask_


    @apply_parallel_decorator(init='data', post='_assemble', target='for')
    def smooth_labels(self, crop, eps=0.05, src=None, dst=None):
        """ Smooth labeling for segmentation mask:
            - change `1`'s to `1 - eps`
            - change `0`'s to `eps`
        Assumes that the mask is binary.
        """
        label_mask = crop == 1
        crop[label_mask] = 1 - eps
        crop[~label_mask] = eps
        return crop


    # Predictions
    @action
    @apply_parallel_decorator(init='indices', post=None, target='for')
    def update_accumulator(self, ix, src, accumulator, dst=None):
        """ Update accumulator with data from crops.
        Allows to gradually accumulate predictions in a single instance, instead of
        keeping all of them and assembling later.

        Parameters
        ----------
        src : str
            Component with crops.
        accumulator : Accumulator3D
            Container for aggregation.
        """
        crop = self.get(ix, src)
        location = self.get(ix, 'locations')
        if self.get(ix, 'orientations') == 1:
            crop = crop.transpose(1, 0, 2)
        elif self.get(ix, 'orientations') == 2:
            crop = crop.transpose(1, 2, 0)
        accumulator.update(crop, location)
        return self

    @action
    @apply_parallel_decorator(init='indices', post='_masks_to_horizons_post', target='for')
    def masks_to_horizons(self, ix, src, dst, threshold=0.5, mode='mean', minsize=0, prefix='predict'):
        """ Convert predicted segmentation mask to a list of Horizon instances.

        Parameters
        ----------
        src_masks : str
            Component of batch that stores masks.
        dst : str/object
            Component of batch to store the resulting horizons.
        threshold, mode, minsize, mean_threshold, adjacency, prefix
            Passed directly to :meth:`Horizon.from_mask`.
        """
        _ = dst

        # Threshold the mask, transpose and rotate the mask if needed
        mask = self.get(ix, src)
        if self.get(ix, 'orientations'):
            mask = mask.transpose(1, 0, 2)

        field = self.get(ix, 'fields')
        origin = [self.get(ix, 'locations')[k].start for k in range(3)]
        horizons = Horizon.from_mask(mask, field=field, origin=origin, threshold=threshold,
                                     mode=mode, minsize=minsize, prefix=prefix)
        return horizons

    def _masks_to_horizons_post(self, horizons_lists, dst=None, **kwargs):
        """ Flatten list of lists of horizons, recieved from each worker. """
        _ = kwargs
        if dst is None:
            raise ValueError('Specify `dst`!')

        # Check for errors, flatten lists
        self.noop_post(horizons_lists, **kwargs)
        setattr(self, dst, [horizon for horizon_list in horizons_lists for horizon in horizon_list])
        return self


    @action
    @apply_parallel_decorator(init='indices', target='for')
    def save_masks(self, ix, src, dst=None, save_to=None, savemode='numpy',
                   threshold=0.5, mode='mean', minsize=0, prefix='predict'):
        """ Save extracted horizons to disk. """
        os.makedirs(save_to, exist_ok=True)

        # Get correct mask
        mask = self.get(ix, src)
        if self.get(ix, 'orientations'):
            mask = np.transpose(mask, (1, 0, 2))

        # Get meta parameters of the mask
        field = self.get(ix, 'fields')
        origin = [self.get(ix, 'locations')[k].start for k in range(3)]
        endpoint = [self.get(ix, 'locations')[k].stop for k in range(3)]

        # Extract surfaces
        horizons = Horizon.from_mask(mask, field=field, origin=origin, mode=mode,
                                    threshold=threshold, minsize=minsize, prefix=prefix)

        if horizons and len(horizons[-1]) > minsize:
            horizon = horizons[-1]
            str_location = '__'.join([f'{start}-{stop}' for start, stop in zip(origin, endpoint)])
            savepath = os.path.join(save_to, f'{prefix}_{str_location}')

            if savemode in ['numpy', 'np', 'npy']:
                np.save(savepath, horizon.points)

            elif savemode in ['dump']:
                horizon.dump(savepath)

        return self


    # Actions to work with components
    @action
    def concat_components(self, src, dst, axis=-1):
        """ Concatenate a list of components and save results to `dst` component.

        Parameters
        ----------
        src : array-like
            List of components to concatenate of length more than one.
        dst : str
            Component of batch to put results in.
        axis : int
            The axis along which the arrays will be joined.
        """
        if len(src) == 1:
            warn("Since `src` contains only one component, concatenation not needed.")

        items = [self.get(None, attr) for attr in src]

        concat_axis_size = sum(item.shape[axis] for item in items)
        shape = list(items[0].shape)
        shape[axis] = concat_axis_size

        buffer = np.empty(shape, dtype=np.float32)

        size_counter = 0
        slicing = [slice(None) for _ in range(axis + 1)]
        for item in items:
            item_size = item.shape[axis]
            slicing[-1] = slice(size_counter, size_counter + item_size)
            buffer[tuple(slicing)] = item
            size_counter += item_size
        setattr(self, dst, buffer)
        return self

    @action
    def transpose(self, src, order, dst=None):
        """ Change order of axis. """
        #pylint: disable=access-member-before-definition
        if src is None:
            src = list(self.name_to_order.keys())

        dst = dst or src
        src = [src] if isinstance(src, str) else src
        dst = [dst] if isinstance(dst, str) else dst

        for src_, dst_ in zip(src, dst):
            current_order = self.name_to_order[src_]
            data = self.get(component=src_)

            # Select correct order of axis
            if order == 'channels_last':
                order_ = np.argsort(data.shape[1:])[::-1]
            elif isinstance(order, str):
                order_ = [current_order.tolist().index(item) for item in order]
            else:
                order_ = list(order)

            # Update meta, transpose data with corrected on batch dimension order
            self.name_to_order[src_] = current_order[list(order_)]
            setattr(self, dst_, data.transpose(0, *(i+1 for i in order_)))
        return self


    @action
    def adaptive_expand(self, src, dst=None, axis=1, symbol='c'):
        """ Add channels dimension to 4D components if needed.
        If component data has shape `(batch_size, 1, n_x, n_d)`, the same shape is kept
        If component data has shape `(batch_size, n_i, n_x, n_d)` and `n_i > 1`, an axis at `axis` position is created.
        """
        dst = dst or src
        src = [src] if isinstance(src, str) else src
        dst = [dst] if isinstance(dst, str) else dst

        for src_, dst_ in zip(src, dst):
            data = self.get(component=src_)
            if data.ndim == 4 and data.shape[1] != 1:
                data = np.expand_dims(data, axis=axis)
                self.name_to_order[src_] = np.insert(self.name_to_order[src_], axis-1, symbol)
            setattr(self, dst_, data)
        return self

    @action
    def adaptive_squeeze(self, src, dst=None, axis=1):
        """ Remove channels dimension from 5D components if needed.
        If component data has shape `(batch_size, n_c, n_i, n_x, n_d)` and `axis=1
                           or shape `(batch_size, n_i, n_x, n_d, n_c)` and `axis=-1`
        and `n_c == 1` , axis at position `axis` will be squeezed.
        """
        dst = dst or src
        src = [src] if isinstance(src, str) else src
        dst = [dst] if isinstance(dst, str) else dst

        for src_, dst_ in zip(src, dst):
            data = self.get(component=src_)
            if data.ndim == 5 and data.shape[axis] == 1:
                data = np.squeeze(data, axis=axis)
                self.name_to_order[src_] = np.delete(self.name_to_order[src_], axis-1)
            setattr(self, dst_, data)
        return self


    # Augmentations: values
    @apply_parallel_decorator(init='preallocating_init', post='noop_post', target='for')
    def additive_noise(self, _, buffer, scale, **kwargs):
        """ Add random value to each entry of crop. Added values are centered at 0.

        Parameters
        ----------
        scale : float
            Standard deviation of normal distribution.
        """
        buffer += scale * self.random.standard_normal(dtype=np.float32, size=buffer.shape)

    @apply_parallel_decorator(init='preallocating_init', post='noop_post', target='for')
    def multiplicative_noise(self,  _, buffer, scale, **kwargs):
        """ Multiply each entry of crop by random value, centered at 1.

        Parameters
        ----------
        scale : float
            Standard deviation of normal distribution.
        """
        buffer *= 1 + scale * self.random.standard_normal(dtype=np.float32, size=buffer.shape)


    @apply_parallel_decorator(init='preallocating_init', post='noop_post', target='for')
    def translate(self, _, buffer, shift=5, scale=0.0, **kwargs):
        """ Add and multiply values by uniformly sampled values. """
        shift = self.random.uniform(-shift, shift)
        scale = self.random.uniform(1 - scale, 1 + scale)

        buffer += np.float32(shift)
        buffer *= np.float32(scale)

    @apply_parallel_decorator(init='preallocating_init', post='noop_post', target='for')
    def invert(self, _, buffer, **kwargs):
        """ Change sign of values. """
        buffer *= -1

    @apply_parallel_decorator(init='preallocating_init', post='noop_post', target='for')
    def equalize(self, _, buffer, mode='default', **kwargs):
        """ Apply histogram equalization. """
        #pylint: disable=import-outside-toplevel
        import torch
        import kornia

        tensor = torch.from_numpy(buffer)

        if mode == 'default':
            tensor = kornia.enhance.equalize(tensor)
        else:
            tensor = kornia.enhance.equalize_clahe(tensor)

        buffer[:] = tensor.numpy()


    @apply_parallel_decorator(init='preallocating_init', post='noop_post', target='for')
    def binarize(self, _, buffer, threshold=0.5, **kwargs):
        """ Binarize image by threshold. """
        buffer[:] = buffer > threshold


    # Augmentations: geometric. `rotate_2d/3d`, `scale_2d/3d`,
    # 'affine_transform', 'perspective_transform' and 'elastic_transform' are added by decorator
    @apply_parallel_decorator(init='preallocating_init', post='noop_post', target='for')
    def flip(self, _, buffer, axis=0, **kwargs):
        """ Flip crop along the given axis.

        Parameters
        ----------
        axis : int
            Axis to flip along
        """
        locations = [slice(None)] * buffer.ndim
        locations[axis] = slice(None, None, -1)
        buffer[:] = buffer[tuple(locations)]

    @apply_parallel_decorator(init='data', post='_assemble')
    def center_crop(self, crop, shape, **kwargs):
        """ Central crop of defined shape. """
        return functional.center_crop(crop, shape)

    @apply_parallel_decorator(init='data', post='_assemble', target='for')
    def resize(self, crop, size=None, factor=2, interpolation=1, **kwargs):
        """ Resize image. By default uses a bilinear interpolation."""
        if size is None:
            # for 2D crop
            if crop.shape[0] == 1:
                h, w = int(crop.shape[1] // factor), int(crop.shape[2] // factor)
            # for 3D crop
            else:
                h, w = int(crop.shape[0] // factor), int(crop.shape[1] // factor)
            size = (h, w)
        return functional.resize(array=crop, size=size, interpolation=interpolation)

    @apply_parallel_decorator(init='data', post='_assemble', target='for')
    def skeletonize_seismic(self, crop, smooth=True, axis=0, width=3, sigma=3, **kwargs):
        """ Perform skeletonize of seismic on 2D slide """
        if smooth:
            crop = gaussian_filter(crop, sigma=sigma, mode='nearest')
        crop = crop.squeeze()
        skeletonized_max = skeletonize(crop, axis=axis, width=width)
        skeletonized_min = skeletonize(-crop, axis=axis, width=width)
        skeletonized = skeletonized_max - skeletonized_min
        return skeletonized.reshape(1, *skeletonized.shape)


    # Augmentations: geologic. `compute_instantaneous_amplitude/phase/frequency` are added by decorator
    @apply_parallel_decorator(init='preallocating_init', post='noop_post', target='for')
    def sign_transform(self, _, buffer, **kwargs):
        """ Element-wise indication of the sign of a number. """
        buffer[:] = np.sign(buffer)

    @action
    @apply_parallel_decorator(init='indices', post='_assemble', target='for')
    def bandpass_filter(self, ix, src, dst, lowcut=None, highcut=None, axis=1, order=4, sign=True):
        """ Keep only frequencies between `lowcut` and `highcut`. Frequency bounds `lowcut` and `highcut`
        are measured in Hz.

        NOTE: use action `SeismicCropBatch.plot_frequencies` to look at the component's spectrum. The action
        shows power spectrum in the same units as required here by parameters `lowcut` and `highcut`.

        Parameters
        ----------
        lowcut : float
            Lower bound for frequencies kept.
        highcut : float
            Upper bound for frequencies kept.
        order : int
            Filtering order.
        sign : bool
            Whether to keep only signs of resulting image.
        """
        field = self.get(ix, 'fields')
        sampling_frequency = field.sample_rate
        crop = self.get(ix, src)

        sos = butter(order, [lowcut, highcut], btype='band', output='sos', fs=sampling_frequency)
        filtered = sosfiltfilt(sos, crop, axis=axis)
        if sign:
            filtered = np.sign(filtered)
        return filtered


    # Augmentations: misc
    @apply_parallel_decorator(init='preallocating_init', post='noop_post', target='for')
    def gaussian_filter(self, _, buffer, axis=1, sigma=2, order=0, **kwargs):
        """ Apply a gaussian filter along specified axis. """
        buffer[:] = gaussian_filter1d(buffer, sigma=sigma, axis=axis, order=order)


    @apply_parallel_decorator(init='preallocating_init', post='noop_post', target='for', requires_rng=True)
    def cutout_2d(self, _, buffer, patch_shape, n_patches, fill_value=0, rng=None, **kwargs):
        """ Change patches of data to zeros.

        Parameters
        ----------
        patch_shape : int or array-like
            Shape of patches along each axis. If int, square patches will be generated. If array of length 2,
            patch will be the same for all channels.
        n_patches : number
            Number of patches to cut.
        fill_value : number
            Value to fill patches with.
        """
        # Parse arguments
        if isinstance(patch_shape, (int, np.integer)):
            patch_shape = np.array([patch_shape, patch_shape, buffer.shape[-1]])
        if len(patch_shape) == 2:
            patch_shape = np.array([*patch_shape, buffer.shape[-1]])

        patch_shape = np.array(patch_shape).astype(np.int32)
        upper_bounds = np.clip(np.array(buffer.shape) - np.array(patch_shape), a_min=1, a_max=buffer.shape)

        # Generate locations for erasing
        for _ in range(int(n_patches)):
            starts = rng.integers(upper_bounds)
            stops = starts + patch_shape

            slices = [slice(start, stop) for start, stop in zip(starts, stops)]
            buffer[tuple(slices)] = fill_value


    @action
    def fill_bounds(self, src, dst=None, margin=0.05, fill_value=0):
        """ Fill bounds of crops with `fill_value`. To remove predictions on bounds. """
        if (np.array(margin) == 0).all():
            return self

        dst = dst or src
        src = [src] if isinstance(src, str) else src
        dst = [dst] if isinstance(dst, str) else dst

        if isinstance(margin, (int, float)):
            margin = (margin, margin, margin)

        for src_, dst_ in zip(src, dst):
            crop = self.get(component=src_).copy()
            pad = [int(np.floor(s) * m) if isinstance(m, float) else m for m, s in zip(margin, crop.shape[1:])]
            pad = [m if s > 1 else 0 for m, s in zip(pad, crop.shape[1:])]
            pad = [(item // 2, item - item // 2) for item in pad]
            for i in range(3):
                slices = [slice(None), slice(None), slice(None), slice(None)]
                slices[i+1] = slice(pad[i][0])
                crop[slices] = fill_value

                slices[i+1] = slice(crop.shape[i+1] - pad[i][1], None)
                crop[slices] = fill_value
            setattr(self, dst_, crop)
        return self
