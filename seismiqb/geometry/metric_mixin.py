""" Collection of tools for evaluating quality of geometries. """
import numpy as np

from batchflow import Notifier



class MetricMixin:
    """ Tools to evaluate geometry quality or compare multiple geometries. """
    def compare(self, other, function, kernel_size=(5, 5), limits=None, chunk_size=(500, 500), pbar='t', **kwargs):
        """ Compare two geometries by iterating over their values in lateral window.

        Parameters
        ----------
        self, other : instances of Geometry
            Geometries to compare.
        function : callable
            Function to compare values in each lateral window. Must take two 2D arrays as inputs.
        kernel_size : sequence of two ints
            Shape of the lateral window.
        limits : sequence of ints, slice, optional
            Slice of the data along the depth (last) axis.
        chunk_size : sequence of two ints
            Size of the data loaded at once.
        pbar : bool, str
            If bool, then whether to display progress bar.
            If str, then type of progress bar to display: `'t'` for textual, `'n'` for widget.
        kwargs : dict
            Passed directly to `function`.
        """
        # Parse parameters
        limits = self.process_limits(limits)
        window = np.array(kernel_size)
        low = window // 2
        high = window - low

        # Compute the shape of `function` output
        array_example = self[0:kernel_size[0], 0:kernel_size[1]].reshape(-1, self.depth)
        size = function(array_example, array_example, **kwargs).size
        metric_matrix = np.full((*self.spatial_shape, size), np.nan, dtype=np.float32)
        total = np.prod(self.shape[:2] - window)

        with Notifier(pbar, total=total) as progress_bar:
            # Iterate over geometries in lateral chunks
            for i_chunk_start in range(0, self.shape[0], chunk_size[0] - window[0]):
                for x_chunk_start in range(0, self.shape[1], chunk_size[1] - window[1]):
                    i_chunk_end = min(i_chunk_start + chunk_size[0], self.shape[0])
                    x_chunk_end = min(x_chunk_start + chunk_size[1], self.shape[1])

                    chunk_locations = [slice(i_chunk_start, i_chunk_end),
                                       slice(x_chunk_start, x_chunk_end),
                                       limits]
                    self_chunk  =  self.load_crop(chunk_locations)
                    other_chunk = other.load_crop(chunk_locations)
                    chunk_shape = self_chunk.shape

                    # Iterate over chunks in lateral kernels
                    for i_anchor in range(low[0], chunk_shape[0] - high[0]):
                        for x_anchor in range(low[1], chunk_shape[1] - high[1]):
                            i_kernel_slice = slice(i_anchor - low[0], i_anchor + high[0])
                            x_kernel_slice = slice(x_anchor - low[1], x_anchor + high[1])

                            self_subset  =  self_chunk[i_kernel_slice, x_kernel_slice].reshape(-1, chunk_shape[-1])
                            other_subset = other_chunk[i_kernel_slice, x_kernel_slice].reshape(-1, chunk_shape[-1])
                            result = function(self_subset, other_subset, **kwargs)

                            metric_matrix[i_chunk_start + i_anchor, x_chunk_start + x_anchor] = result
                            progress_bar.update()
        return metric_matrix
