""" Accumulator for 2d matrices. """
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = np
    CUPY_AVAILABLE = False
from .classes import augmented_np



class Accumulator:
    """ Class to accumulate statistics over streamed matrices.
    An example of usage:
        one can either store matrices and take a mean along desired axis at the end of their generation,
        or sequentially update the `mean` matrix with the new data by using this class.
    Note the latter approach is inherintly slower, but requires O(N) times less memory,
    where N is the number of accumulated matrices.

    This class is intended to be used in the following manner:
        - initialize the instance with desired aggregation
        - iteratively call `update` method with new matrices
        - to get the aggregated result, use `get` method

    NaNs are ignored in all computations.
    This class works with both CPU (`numpy`) and GPU (`cupy`) arrays and automatically detects current device.

    Parameters
    ----------
    agg : str
        Which type of aggregation to use. Currently, following modes are implemented:
            - 'mean' works by storing matrix of sums and non-nan counts.
            To get the mean result, the sum is divided by the counts
            - 'std' works by keeping track of sum of the matrices, sum of squared matrices,
            and non-nan counts. To get the result, we subtract squared mean from mean of squared values
            - 'min', 'max' works by iteratively updating the matrix of minima/maxima values
            - 'argmin', 'argmax' iteratively updates index of the minima/maxima values in the passed matrices
            - 'stack' just stores the matrices and concatenates them along (new) last axis
            - 'mode' stores supplied matrices and computes mode along the last axis during the `get` call
    amortize : bool
        If False, then supplied matrices are stacked into ndarray, and then aggregation is applied.
        If True, then accumulation logic is applied.
        Allows for trade-off between memory usage and speed: `amortize=False` is faster,
        but takes more memory resources.
    total : int or None
        If integer, then total number of matrices to be aggregated.
        Used to reduce the memory footprint if `amortize` is set to False.
    axis : int
        Axis to stack matrices on and to apply aggregation funcitons.
    """
    #pylint: disable=attribute-defined-outside-init
    def __init__(self, agg='mean', amortize=False, total=None, axis=0):
        self.agg = agg
        self.amortize = amortize
        self.total = total
        self.axis = axis

        self.initialized = False


    def init(self, matrix):
        """ Initialize all the containers on first `update`. """
        # No amortization: collect all the matrices and apply reduce afterwards
        self.module = cp.get_array_module(matrix) if CUPY_AVAILABLE else augmented_np
        self.n = 1

        if self.amortize is False or self.agg in ['stack', 'mode']:
            if self.total:
                self.values = self.module.empty((self.total, *matrix.shape))
                self.values[0, ...] = matrix
            else:
                self.values = [matrix]

            self.initialized = True
            return

        # Amortization: init all the containers
        if self.agg in ['mean', 'nanmean']:
            # Sum of values and counts of non-nan
            self.value = matrix
            self.counts = (~self.module.isnan(matrix)).astype(self.module.int32)

        elif self.agg in ['min', 'nanmin', 'max', 'nanmax']:
            self.value = matrix

        elif self.agg in ['std', 'nanstd']:
            # Same as means, but need to keep track of mean of squares and squared mean
            self.means = matrix
            self.squared_means = matrix ** 2
            self.counts = (~self.module.isnan(matrix)).astype(self.module.int32)

        elif self.agg in ['argmin', 'argmax', 'nanargmin', 'nanargmax']:
            # Keep the current maximum/minimum and update indices matrix, if needed
            self.value = matrix
            self.indices = self.module.zeros_like(matrix)

        self.initialized = True
        return


    def update(self, matrix):
        """ Update containers with new matrix. """
        if not self.initialized:
            self.init(matrix.copy())
            return

        # No amortization: just store everything
        if self.amortize is False or self.agg in ['stack', 'mode']:
            if self.total:
                self.values[self.n, ...] = matrix
            else:
                self.values.append(matrix)

            self.n += 1
            return

        # Amortization: update underlying containers
        slc = ~self.module.isnan(matrix)

        if self.agg in ['min', 'nanmin']:
            self.value[slc] = self.module.fmin(self.value[slc], matrix[slc])

        elif self.agg in ['max', 'nanmax']:
            self.value[slc] = self.module.fmax(self.value[slc], matrix[slc])

        elif self.agg in ['mean', 'nanmean']:
            mask = np.logical_and(slc, self.module.isnan(self.value))
            self.value[mask] = 0.0
            self.value[slc] += matrix[slc]
            self.counts[slc] += 1

        elif self.agg in ['std', 'nanstd']:
            mask = np.logical_and(slc, self.module.isnan(self.means))
            self.means[mask] = 0.0
            self.squared_means[mask] = 0.0
            self.means[slc] += matrix[slc]
            self.squared_means[slc] += matrix[slc] ** 2
            self.counts[slc] += 1

        elif self.agg in ['argmin', 'nanargmin']:
            mask = self.module.logical_and(slc, self.module.isnan(self.value))
            self.value[mask] = matrix[mask]
            self.indices[mask] = self.n

            slc_ = matrix < self.value
            self.value[slc_] = matrix[slc_]
            self.indices[slc_] = self.n

        elif self.agg in ['argmax', 'nanargmax']:
            mask = self.module.logical_and(slc, self.module.isnan(self.value))
            self.value[mask] = matrix[mask]
            self.indices[mask] = self.n

            slc_ = matrix > self.value
            self.value[slc_] = matrix[slc_]
            self.indices[slc_] = self.n

        self.n += 1
        return

    def get(self, final=False):
        """ Use stored matrices to get the aggregated result. """
        # No amortization: apply function along the axis to the stacked array
        if self.amortize is False or self.agg in ['stack', 'mode']:
            if self.total:
                stacked = self.values
            else:
                stacked = self.module.stack(self.values, axis=self.axis)

            if final:
                self.values = None

            if self.agg in ['stack']:
                value = stacked

            elif self.agg in ['mode']:
                uniques = self.module.unique(stacked)

                accumulator = Accumulator('argmax')
                for item in uniques[~self.module.isnan(uniques)]:
                    counts = (stacked == item).sum(axis=self.axis)
                    accumulator.update(counts)
                indices = accumulator.get(final=True)
                value = uniques[indices]
                value[self.module.isnan(self.module.max(stacked, axis=self.axis))] = self.module.nan

            else:
                value = getattr(self.module, self.agg)(stacked, axis=self.axis)

            return value

        # Amortization: compute desired aggregation
        if self.agg in ['min', 'nanmin', 'max', 'nanmax']:
            value = self.value

        elif self.agg in ['mean', 'nanmean']:
            slc = self.counts > 0
            value = self.value if final else self.value.copy()
            value[slc] /= self.counts[slc]

        elif self.agg in ['std', 'nanstd']:
            slc = self.counts > 0
            means = self.means if final else self.means.copy()
            means[slc] /= self.counts[slc]

            squared_means = self.squared_means if final else self.squared_means.copy()
            squared_means[slc] /= self.counts[slc]
            value = self.module.sqrt(squared_means - means ** 2)

        elif self.agg in ['argmin', 'argmax', 'nanargmin', 'nanargmax']:
            value = self.indices

        return value
