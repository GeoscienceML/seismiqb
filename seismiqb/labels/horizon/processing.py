""" Mixin for horizon processing. """
from math import isnan
from functools import partialmethod
import numpy as np
from numba import njit, prange

from cv2 import inpaint as cv2_inpaint
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation, binary_erosion

from ...functional import make_gaussian_kernel
from ...utils import make_bezier_figure

class ProcessingMixin:
    """ Methods for horizon processing.

    Contains methods for:
        - Removing or adding points to the horizon surface.
        - Smoothing out the horizon surface.
        - Cutting shapes (holes or carcasses) from the horizon surface.

    Note, almost all of these methods can change horizon surface inplace or create a new instance.
    In either case they return a filtered horizon instance.
    """
    # Filtering methods
    def filter(self, filtering_matrix=None, margin=0, inplace=False, add_prefix=True, **kwargs):
        """ Remove points that correspond to 1's in `filtering_matrix` from the horizon surface.

        Note, this method may change horizon inplace or create a new instance. By default creates a new instance.
        In either case it returns a processed horizon instance.

        Parameters
        ----------
        filtering_matrix : None, str or np.ndarray
            Mask of points to cut out from the horizon.
            If None, then remove points corresponding to zero traces.
            If str, then remove points corresponding to the `filtering_matrix` attribute.
            If np.ndarray, then used as filtering mask.
        margin : int
            Amount of traces to cut out near to boundaries considering `filtering_matrix` appliance.
        inplace : bool
            Whether to apply operation inplace or return a new Horizon object.
        add_prefix : bool
            If True and not inplace, adds prefix to the horizon name.
        kwargs : dict
            Arguments to be passed in the loading attribute method in case when filtering_matrix is a str.

        Returns
        -------
        :class:`~.Horizon`
            Processed horizon instance. A new instance if `inplace` is False, `self` otherwise.
        """
        if filtering_matrix is None:
            filtering_matrix = self.field.dead_traces_matrix
        elif isinstance(filtering_matrix, str):
            filtering_matrix = self.load_attribute(filtering_matrix, **kwargs)
            filtering_matrix[np.abs(filtering_matrix) > 1] = 1

        if not issubclass(filtering_matrix.dtype.type, bool):
            filtering_matrix = filtering_matrix > 0

        if margin > 0:
            filtering_matrix = binary_dilation(filtering_matrix, iterations=margin)

            filtering_matrix[:margin, :] = 1
            filtering_matrix[:, :margin] = 1
            filtering_matrix[-margin:, :] = 1
            filtering_matrix[:, -margin:] = 1

        mask = filtering_matrix[self.points[:, 0], self.points[:, 1]]
        points = self.points[mask == 0]

        if inplace:
            self.points = points
            self.reset_storage('matrix')
            return self

        name = 'filtered_' + self.name if add_prefix else self.name
        return type(self)(storage=points, field=self.field, name=name)

    despike = partialmethod(filter, filtering_matrix='spikes')

    def filter_disconnected_regions(self, erosion_rate=0, inplace=False):
        """ Remove regions, not connected to the largest component of a horizon.

        Note, this method may change horizon inplace or create a new instance. By default creates a new instance.
        In either case it returns a processed horizon instance.
        """
        if erosion_rate > 0:
            structure = np.ones((3, 3))
            matrix = binary_erosion(self.mask, structure, iterations=erosion_rate)
        else:
            matrix = self.mask

        labeled = label(matrix)
        values, counts = np.unique(labeled, return_counts=True)
        counts = counts[values != 0]
        values = values[values != 0]

        object_id = values[np.argmax(counts)]

        filtering_matrix = np.zeros_like(self.mask)
        filtering_matrix[labeled == object_id] = 1

        if erosion_rate > 0:
            filtering_matrix = binary_dilation(filtering_matrix, structure, iterations=erosion_rate)

        filtering_matrix = filtering_matrix == 0

        return self.filter(filtering_matrix, inplace=inplace)


    # Horizon surface transformations
    def smooth_out(self, mode='convolve', iters=1,
                   kernel_size=(3, 3), sigma_spatial=0.8, kernel=None, sigma_range=2.0,
                   max_depth_difference=5, inplace=False, add_prefix=True, dtype=None):
        """ Smooth out the horizon surface.

        Smoothening is applied without absent points changing.

        This method supports two types of smoothening:
            - if `mode='convolve'`, then the method uses a convolution with a given or a gaussian kernel.
            - if `mode='bilateral'`, then the method applies a bilateral filtering with a given or a gaussian kernel.
            Bilateral filtering is an edge-preserving smoothening, which ignores areas with faults.
            Be careful with `sigma_range` value:
                - The higher the `sigma_range` value, the more 'bilateral' result looks like a 'convolve' result.
                - If the `sigma_range` too low, then no smoothening applied.

        Note, this method may change horizon inplace or create a new instance. By default creates a new instance.
        In either case it returns a processed horizon instance.

        Note, that the method makes dtype conversion only if the parameter `inplace` is False.

        Parameters
        ----------
        mode : str
            Smoothening type mode. Can be 'convolve' or 'bilateral'.
            If 'convolve', then the method makes a convolution with a given kernel.
            If 'bilateral', then the method applies a bilateral filtering with a given kernel.
        iters : int
            Number of times to apply smoothing.
        kernel_size : int or sequence of ints
            Size of a created gaussian filter if `kernel` is None.
        sigma_spatial : number
            Standard deviation (spread or “width”) for gaussian kernel.
            The lower, the more weight is put into the point itself.
        kernel : ndarray or None
            If passed, then ready-to-use kernel. Otherwise, gaussian kernel will be created.
        sigma_range : number
            Standard deviation for additional weight which smooth differences in depth values.
            The lower, the more weight is put into the depths differences between point in a window.
            Note, if it is too low, then no smoothening is applied.
        max_depth_difference : number
            If the distance between anchor point and the point inside filter is bigger than the threshold,
            then the point is ignored in smoothening.
            Can be used for separate smoothening on sides of discontinuity.
        inplace : bool
            Whether to apply operation inplace or return a new Horizon object.
        add_prefix : bool
            If True and not inplace, adds prefix to the horizon name.
        dtype : type
            Output horizon dtype. Supported only if `inplace` is False.

        Returns
        -------
        :class:`~.Horizon`
            Processed horizon instance. A new instance if `inplace` is False, `self` otherwise.
        """
        if 'conv' in mode:
            smoothening_function, kwargs = _convolve, {}
        else:
            smoothening_function, kwargs = _bilateral_filter, {'sigma_range': sigma_range}

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        kernel = kernel if kernel is not None else make_gaussian_kernel(kernel_size=kernel_size, sigma=sigma_spatial)
        dtype = dtype if dtype is not None else self.dtype

        result = self.matrix.astype(np.float32)
        result[result == self.FILL_VALUE] = np.nan

        # Apply smoothening multiple times. Note that there is no dtype conversion in between
        # Also the method returns a new object
        for _ in range(iters):
            result = smoothening_function(src=result, kernel=kernel,
                                          max_depth_difference=max_depth_difference,
                                          **kwargs)

        result[(self.matrix == self.FILL_VALUE) | np.isnan(result)] = self.FILL_VALUE
        result[self.field.dead_traces_matrix[self.i_min:self.i_max + 1,
                                             self.x_min:self.x_max + 1]] = self.FILL_VALUE

        if dtype == np.int32 or (self.dtype == np.int32 and inplace is True):
            result = np.rint(result).astype(np.int32)

        if inplace:
            self.matrix = result
            self.reset_storage('points')
            return self

        name = 'smoothed_' + self.name if add_prefix else self.name
        return type(self)(storage=result, i_min=self.i_min, x_min=self.x_min, field=self.field, name=name, dtype=dtype)

    def interpolate(self, iters=1, kernel_size=(3, 3), sigma=0.8, kernel=None,
                    min_present_neighbors=0, max_depth_ptp=None, inplace=False, add_prefix=True):
        """ Interpolate horizon surface on the regions with missing traces.

        Under the hood, we fill missing traces with weighted neighbor values.

        Note, this method may change horizon inplace or create a new instance. By default creates a new instance.
        In either case it returns a processed horizon instance.

        Parameters
        ----------
        iters : int
            Number of interpolation iterations to perform.
        kernel_size : int or sequence of ints
            If the kernel is not provided, shape of the square gaussian kernel.
        sigma : number
            Standard deviation (spread or “width”) for gaussian kernel.
            The lower, the more weight is put into the point itself.
        kernel : ndarray or None
            Interpolation weights kernel.
        min_present_neighbors : int
            Minimal amount of non-missing neighboring points in a window to interpolate a central point.
        max_depth_ptp : number
            A maximum distance between values in a squared window for which we apply interpolation.
        inplace : bool
            Whether to apply operation inplace or return a new Horizon object.
        add_prefix : bool
            If True and not inplace, adds prefix to the horizon name.

        Returns
        -------
        :class:`~.Horizon`
            Processed horizon instance. A new instance if `inplace` is False, `self` otherwise.
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        kernel = kernel if kernel is not None else make_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)

        result = self.matrix.astype(np.float32)
        result[self.matrix == self.FILL_VALUE] = np.nan

        # Apply `_interpolate` multiple times. Note that there is no dtype conversion in between
        # Also the method returns a new object
        for _ in range(iters):
            result = _interpolate(src=result, kernel=kernel, min_present_neighbors=min_present_neighbors,
                                  max_depth_ptp=max_depth_ptp)

        result[np.isnan(result)] = self.FILL_VALUE
        result[self.field.dead_traces_matrix[self.i_min:self.i_max + 1,
                                             self.x_min:self.x_max + 1]] = self.FILL_VALUE

        if self.dtype == np.int32:
            result = np.rint(result).astype(np.int32)

        if inplace:
            self.matrix = result
            self.reset_storage('points')
            return self

        name = 'interpolated_' + self.name if add_prefix else self.name
        return type(self)(storage=result, i_min=self.i_min, x_min=self.x_min, field=self.field, name=name)

    def inpaint(self, inpaint_radius=1, neighbors_radius=1, method=0, inplace=False, add_prefix=True):
        """ Inpaint horizon surface on the regions with missing traces.

        Under the hood, the method uses the inpainting method from OpenCV.

        Note, this method may change horizon inplace or create a new instance. By default creates a new instance.
        In either case it returns a processed horizon instance.

        Parameters
        ----------
        inpaint_radius : int
            Radius of traces to inpaint near horizon boundaries.
            When the surface has huge missing regions, we don't want to fill them completely, because
            inpainting too far from existing horizon traces can be made with huge errors.
        neighbors_radius : int
            Parameter passed to the :meth:`cv2.inpaint` as `inpaintRadius`.
            Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.
        method : int
            Parameter passed to the :meth:`cv2.inpaint` as `flags`. Can be 0 or 1.
            If 0, then Navier-Stokes algorithm is used.
            If 1, then Telea algorithm is used.
        inplace : bool
            Whether to apply operation inplace or return a new Horizon object.
        add_prefix : bool
            If True and not inplace, adds prefix to the horizon name.

        Returns
        -------
        :class:`~.Horizon`
            Processed horizon instance. A new instance if `inplace` is False, `self` otherwise.
        """
        image = self.matrix.astype(np.uint16) # dtype conversion for compatibility with the OpenCV method
        dead_traces_matrix = self.field.dead_traces_matrix[self.i_min:self.i_max + 1,
                                                           self.x_min:self.x_max + 1]

        # We use all empty traces as inpainting mask because it is important for correct boundary conditions
        # in differential equations, that are used in the inpainting method
        holes_mask = (self.matrix == self.FILL_VALUE).astype(np.uint8)
        holes_mask[dead_traces_matrix == 1] = 1

        result = cv2_inpaint(src=image, inpaintMask=holes_mask, inpaintRadius=neighbors_radius, flags=method)
        result = result.astype(self.dtype)

        # Filtering mask to remove traces, that are too far from existing traces
        too_far_traces_mask = binary_erosion(holes_mask, iterations=inpaint_radius).astype(int)

        # Filter traces with anomalies (can be caused by boundary conditions in equations on horizon borders)
        anomalies_mask = (result > self.d_max + 10) | (result < self.d_min - 10)

        result[(too_far_traces_mask == 1) | (anomalies_mask == 1) | (dead_traces_matrix == 1)] = self.FILL_VALUE

        if inplace:
            self.matrix = result
            self.reset_storage('points')
            return self

        name = 'inpainted_' + self.name if add_prefix else self.name
        return type(self)(storage=result, i_min=self.i_min, x_min=self.x_min, field=self.field, name=name)

    # Horizon distortions
    def thin_out(self, factor=1, threshold=256, inplace=False, add_prefix=True):
        """ Thin out the horizon by keeping only each `factor`-th line.

        Note, this method may change horizon inplace or create a new instance. By default creates a new instance.
        In either case it returns a processed horizon instance.

        Parameters
        ----------
        factor : integer or sequence of two integers
            Frequency of lines to keep along ilines and xlines direction.
        threshold : integer
            Minimal amount of points in a line to keep.
        inplace : bool
            Whether to apply operation inplace or return a new Horizon object.
        add_prefix : bool
            If True and not inplace, adds prefix to the horizon name.

        Returns
        -------
        :class:`~.Horizon`
            Processed horizon instance. A new instance if `inplace` is False, `self` otherwise.
        """
        if isinstance(factor, int):
            factor = (factor, factor)

        uniques, counts = np.unique(self.points[:, 0], return_counts=True)
        mask_i = np.isin(self.points[:, 0], uniques[counts > threshold][::factor[0]])

        uniques, counts = np.unique(self.points[:, 1], return_counts=True)
        mask_x = np.isin(self.points[:, 1], uniques[counts > threshold][::factor[1]])

        points = self.points[mask_i + mask_x]

        if inplace:
            self.points = points
            self.reset_storage('matrix')
            return self

        name = 'thinned_' + self.name if add_prefix else self.name
        return type(self)(storage=points, field=self.field, name=name)

    def make_carcass(self, frequencies=100, margin=50, interpolate=False, add_prefix=True, inplace=False, **kwargs):
        """ Cut carcass out of a horizon. Returns a new instance.

        Parameters
        ----------
        frequencies : int or sequence of two ints
            Frequencies of carcass lines along inline/crossline axis.
        margin : int
            Margin from geometry edges to exclude from carcass.
        interpolate : bool
            Whether to interpolate the result.
        kwargs : dict
            Other parameters for grid creation, see `:meth:~.Geometry.make_quality_grid`.
        """
        #pylint: disable=import-outside-toplevel
        carcass = self if inplace else self.copy(add_prefix=add_prefix)
        carcass.name = carcass.name.replace('copy', 'carcass')

        grid_matrix = self.field.geometry.get_grid(frequency=frequencies, margin=margin)
        carcass.filter(filtering_matrix=1-grid_matrix, inplace=True)
        if interpolate:
            carcass.interpolate(inplace=True)
        return carcass

    def generate_holes_matrix(self, n=10, scale=1.0, max_scale=.25,
                              max_angles_amount=4, max_sharpness=5.0, locations=None,
                              points_proportion=1e-5, points_shape=1,
                              noise_level=0, seed=None):
        """ Create matrix of random holes for horizon.

        Holes can be bezier-like figures or points-like.
        We can control bezier-like and points-like holes amount by `n` and `points_proportion` parameters respectively.
        We also do some noise amplifying with `noise_level` parameter.

        Parameters
        ----------
        n : int
            Amount of bezier-like holes on horizon.
        points_proportion : float
            Proportion of point-like holes on the horizon. A number between 0 and 1.
        points_shape : int or sequence of int
            Shape of point-like holes.
        noise_level : int
            Radius of noise scattering near the borders of holes.
        scale : float or sequence of float
            If float, each bezier-like hole will have a random scale from exponential distribution with parameter scale.
            If sequence, each bezier-like hole will have a provided scale.
        max_scale : float
            Maximum bezier-like hole scale.
        max_angles_amount : int
            Maximum amount of angles in each bezier-like hole.
        max_sharpness : float
            Maximum value of bezier-like holes sharpness.
        locations : ndarray
            If provided, an array of desired locations of bezier-like holes.
        seed : int, optional
            Seed the random numbers generator.
        """
        rng = np.random.default_rng(seed)
        filtering_matrix = np.zeros_like(self.full_matrix)

        # Generate bezier-like holes
        # Generate figures scales
        if isinstance(scale, float):
            scales = []
            sampling_scale = int(
                np.ceil(1.0 / (1 - np.exp(-scale * max_scale)))
            ) # inverse probability of scales < max_scales
            while len(scales) < n:
                new_scales = rng.exponential(scale, size=sampling_scale*(n - len(scales)))
                new_scales = new_scales[new_scales <= max_scale]
                scales.extend(new_scales)
            scales = scales[:n]
        else:
            scales = scale

        # Generate figures-like holes locations
        if locations is None:
            idxs = rng.choice(len(self), size=n)
            locations = self.points[idxs, :2]

        coordinates = [] # container for all types of holes, represented by their coordinates

        # Generate figures inside the field
        for location, figure_scale in zip(locations, scales):
            n_key_points = rng.integers(2, max_angles_amount + 1)
            radius = rng.random()
            sharpness = rng.random() * rng.integers(1, max_sharpness)

            figure_coordinates = make_bezier_figure(n=n_key_points, radius=radius, sharpness=sharpness,
                                                    scale=figure_scale, shape=self.shape, seed=seed)
            figure_coordinates += location

            # Shift figures if they are out of field bounds
            negative_coords_shift = np.min(np.vstack([figure_coordinates, [0, 0]]), axis=0)
            huge_coords_shift = np.max(np.vstack([figure_coordinates - self.shape, [0, 0]]), axis=0)
            figure_coordinates -= (huge_coords_shift + negative_coords_shift + 1)

            coordinates.append(figure_coordinates)

        # Generate points-like holes
        if points_proportion:
            points_n = int(points_proportion * len(self))
            idxs = rng.choice(len(self), size=points_n)
            locations = self.points[idxs, :2]

            filtering_matrix[locations[:, 0], locations[:, 1]] = 1

            if isinstance(points_shape, int):
                points_shape = (points_shape, points_shape)
            filtering_matrix = binary_dilation(filtering_matrix, np.ones(points_shape))

            coordinates.append(np.argwhere(filtering_matrix > 0))

        coordinates = np.concatenate(coordinates)

        # Add noise and filtering matrix transformations
        if noise_level:
            noise = rng.normal(loc=coordinates,
                               scale=noise_level,
                               size=coordinates.shape)
            coordinates = np.unique(np.vstack([coordinates, noise.astype(int)]), axis=0)

        # Add valid coordinates onto filtering matrix
        idx = np.where((coordinates[:, 0] >= 0) &
                       (coordinates[:, 1] >= 0) &
                       (coordinates[:, 0] < self.i_length) &
                       (coordinates[:, 1] < self.x_length))[0]
        coordinates = coordinates[idx]

        filtering_matrix[coordinates[:, 0], coordinates[:, 1]] = 1

        # Process holes
        filtering_matrix = binary_fill_holes(filtering_matrix)
        filtering_matrix = binary_dilation(filtering_matrix, iterations=4)
        return filtering_matrix

    def make_holes(self, inplace=False, n=10, scale=1.0, max_scale=.25,
                   max_angles_amount=4, max_sharpness=5.0, locations=None,
                   points_proportion=1e-5, points_shape=1,
                   noise_level=0, seed=None):
        """ Make holes on a horizon surface.

        Note, this method may change horizon inplace or create a new instance. By default creates a new instance.
        In either case it returns a processed horizon instance.
        """
        #pylint: disable=self-cls-assignment
        filtering_matrix = self.generate_holes_matrix(n=n, scale=scale, max_scale=max_scale,
                                                      max_angles_amount=max_angles_amount,
                                                      max_sharpness=max_sharpness, locations=locations,
                                                      points_proportion=points_proportion, points_shape=points_shape,
                                                      noise_level=noise_level, seed=seed)

        return self.filter(filtering_matrix, inplace=inplace)

    make_holes.__doc__ += '\n' + '\n'.join(generate_holes_matrix.__doc__.split('\n')[1:])

# Helper functions
@njit(parallel=True)
def _convolve(src, kernel, max_depth_difference):
    """ Jit-accelerated function to apply 2d convolution with special care for nan values. """
    #pylint: disable=too-many-nested-blocks, consider-using-enumerate, not-an-iterable
    k = kernel.shape[0] // 2
    raveled_kernel = kernel.ravel() / np.sum(kernel)

    i_range, x_range = src.shape
    dst = src.copy()

    for iline in prange(0, i_range):
        for xline in range(0, x_range):
            central = src[iline, xline]

            if isnan(central):
                continue

            # Get values in the squared window and apply kernel to them
            element = src[max(0, iline-k):min(iline+k+1, i_range),
                          max(0, xline-k):min(xline+k+1, x_range)].ravel()

            s, sum_weights = np.float32(0), np.float32(0)
            for item, weight in zip(element, raveled_kernel):
                if not isnan(item) and (abs(item - central) <= max_depth_difference):
                    s += item * weight
                    sum_weights += weight

            if sum_weights != 0.0:
                dst[iline, xline] = s / sum_weights
    return dst

@njit(parallel=True)
def _bilateral_filter(src, kernel, max_depth_difference, sigma_range=0.1):
    """ Jit-accelerated function to apply 2d bilateral filtering with special care for nan values.

    The difference between :func:`_convolve` and :func:`_bilateral_filter` is in additional weight multiplier,
    which is a gaussian of difference of convolved elements.
    """
    #pylint: disable=too-many-nested-blocks, consider-using-enumerate, not-an-iterable
    k = kernel.shape[0] // 2
    raveled_kernel = kernel.ravel() / np.sum(kernel)
    sigma_squared = sigma_range**2

    i_range, x_range = src.shape
    dst = src.copy()

    for iline in prange(0, i_range):
        for xline in range(0, x_range):
            central = src[iline, xline]

            if isnan(central):
                continue # Because can't evaluate additional multiplier

            # Get values in the squared window and apply kernel to them
            element = src[max(0, iline-k):min(iline+k+1, i_range),
                          max(0, xline-k):min(xline+k+1, x_range)].ravel()

            s, sum_weights = np.float32(0), np.float32(0)
            for item, weight in zip(element, raveled_kernel):
                if not isnan(item) and (abs(item - central) <= max_depth_difference):
                    weight *= np.exp(-0.5*((item - central)**2)/sigma_squared)

                    s += item * weight
                    sum_weights += weight

            if sum_weights != 0.0:
                dst[iline, xline] = s / sum_weights
    return dst

@njit(parallel=True)
def _interpolate(src, kernel, min_present_neighbors=1, max_depth_ptp=None):
    """ Jit-accelerated function to apply 2d interpolation to nan values. """
    #pylint: disable=too-many-nested-blocks, consider-using-enumerate, not-an-iterable
    k = kernel.shape[0] // 2
    raveled_kernel = kernel.ravel() / np.sum(kernel)

    i_range, x_range = src.shape
    dst = src.copy()

    for iline in prange(0, i_range):
        for xline in range(0, x_range):
            central = src[iline, xline]

            if not isnan(central):
                continue # We interpolate values only to nan points

            # Get neighbors and check whether we can interpolate them
            element = src[max(0, iline-k):min(iline+k+1, i_range),
                          max(0, xline-k):min(xline+k+1, x_range)].ravel()

            filled_neighbors = kernel.size - np.isnan(element).sum()
            if filled_neighbors < min_present_neighbors:
                continue

            # Compare ptp with the max_distance_threshold
            if max_depth_ptp is not None:
                nanmax, nanmin = np.float32(element[0]), np.float32(element[0])

                for item in element:
                    if not isnan(item):
                        if isnan(nanmax):
                            nanmax = item
                            nanmin = item
                        else:
                            nanmax = max(item, nanmax)
                            nanmin = min(item, nanmin)

                if nanmax - nanmin > max_depth_ptp:
                    continue

            # Apply kernel to neighbors to get value for interpolated point
            s, sum_weights = np.float32(0), np.float32(0)
            for item, weight in zip(element, raveled_kernel):
                if not isnan(item):
                    s += item * weight
                    sum_weights += weight

            if sum_weights != 0.0:
                dst[iline, xline] = s / sum_weights
    return dst
