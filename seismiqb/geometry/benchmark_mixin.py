""" Collection of tools for benchmarking and testing array-like geometries. """
import time

import numpy as np

from batchflow import Notifier


class BenchmarkMixin:
    """ Methods for testing and benchmarking the geometry. """
    def equal(self, other, return_explanation=False):
        """ Check if two geometries are equal: have the same shape, headers and values. """
        condition = (self.shape == other.shape).all()
        if condition is False:
            explanation = f'Different shapes, {self.shape}  != {other.shape}'
            return (False, explanation) if return_explanation else False

        condition = (self.headers == other.headers).all().all()
        if condition is False:
            explanation = 'Different `headers` dataframes!'
            return (False, explanation) if return_explanation else False

        condition = (self.mean_matrix == other.mean_matrix).all()
        if condition is False:
            explanation = 'Different `mean_matrix` values!'
            return (False, explanation) if return_explanation else False

        for i in range(self.shape[0]):
            condition = (self[i] == other[i]).all()
            if condition is False:
                explanation = f'Different values in slide={i}!'
                return (False, explanation) if return_explanation else False

        return (True, '') if return_explanation else True


    @staticmethod
    def make_random_slide_locations(bounds, allowed_axis=(0, 1, 2), rng=None):
        """ Create random slide locations along one of the axis. """
        rng = rng or np.random.default_rng(rng)
        axis = rng.choice(a=allowed_axis)
        index = rng.integers(*bounds[axis])

        locations = [slice(None), slice(None), slice(None)]
        locations[axis] = slice(index, index + 1)
        return locations

    @staticmethod
    def make_random_crop_locations(bounds, size_min=10, size_max=100, rng=None):
        """ Create random crop locations. """
        rng = rng or np.random.default_rng(rng)
        if isinstance(size_min, int):
            size_min = (size_min, size_min, size_min)
        if isinstance(size_max, int):
            size_max = (size_max, size_max, size_max)

        point = rng.integers(*bounds)
        shape = rng.integers(low=size_min, high=size_max)
        locations = [slice(start, np.clip(start+size, bound_min, bound_max))
                     for start, size, bound_min, bound_max in zip(point, shape, bounds[0], bounds[1])]
        return locations

    def benchmark(array_like, n_slides=300, slide_allowed_axis=(0, 1, 2),
                  n_crops=300, crop_size_min=(10, 10, 256), crop_size_max=(128, 128, 512), seed=42, pbar=False):
        """ Calculate average loading timings.
        Output is user, system and wall timings in milliseconds for slides and crops.
        TODO: separate timings for each slide axis

        Parameters
        ----------
        array_like : array like
            An object with numpy-like getitem semantics and `shape` attribute.
        n_slides : int
            Number of slides to load.
        slide_allowed_axis : sequence of int
            Allowed projections to generate slides along.
        n_crops : int
            Number of crops to load.
        crop_size_min : int or tuple of int
            A minimum size of generated crops.
            If tuple, then each number corresponds to size along each axis.
        crop_size_max : int or tuple of int
            A maximum size of generated crops.
            If tuple, then each number corresponds to size along each axis.
        seed : int
            Seed for the random numbers generator.
        pbar : bool, str
            If bool, then whether to display progress bar.
            If str, then type of progress bar to display: `'t'` for textual, `'n'` for widget.
        """
        #pylint: disable=no-self-argument, import-outside-toplevel
        import psutil

        # Parse parameters
        bbox = np.array([[0, s] for s in array_like.shape])

        rng = np.random.default_rng(seed)
        timings = {}

        # Calculate the average loading slide time
        if n_slides:
            timestamp_start, wall_start = psutil.cpu_times(), time.perf_counter()
            for _ in Notifier(pbar, desc='Slides benchmark')(range(n_slides)):
                slide_locations = BenchmarkMixin.make_random_slide_locations(bounds=bbox, rng=rng,
                                                                             allowed_axis=slide_allowed_axis)
                slide_locations = tuple(slide_locations)
                _ = array_like[slide_locations]
            timestamp_end, wall_end = psutil.cpu_times(), time.perf_counter()

            timings['slide'] = {
                'user': 1000 * (timestamp_end[0] - timestamp_start[0]) / n_slides,
                'system': 1000 * (timestamp_end[2] - timestamp_start[2]) / n_slides,
                'wall': 1000 * (wall_end - wall_start) / n_slides
            }

        # Calculate the average loading crop time
        if n_crops:
            timestamp_start, wall_start = psutil.cpu_times(), time.perf_counter()
            for _ in Notifier(pbar, desc='Crops benchmark')(range(n_crops)):
                crop_locations = BenchmarkMixin.make_random_crop_locations(bbox.T, rng=rng,
                                                                           size_min=crop_size_min,
                                                                           size_max=crop_size_max)
                crop_locations = tuple(crop_locations)
                _ = array_like[crop_locations]
            timestamp_end, wall_end = psutil.cpu_times(), time.perf_counter()

            timings['crop'] = {
                'user': 1000 * (timestamp_end[0] - timestamp_start[0]) / n_crops,
                'system': 1000 * (timestamp_end[2] - timestamp_start[2]) / n_crops,
                'wall': 1000 * (wall_end - wall_start) / n_crops
            }

        return timings
