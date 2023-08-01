""" Common labels mixins. """

import numpy as np

from ..plotters import plot

class VisualizationMixin:
    """ Visualization utilities. """
    def load_slide(self, index, axis=0, width=3):
        """ Create a mask at desired location along supplied axis. """
        axis = self.field.geometry.parse_axis(axis)
        locations = self.field.geometry.make_slide_locations(index, axis=axis)
        shape = self.field.geometry.locations_to_shape(locations)
        width = width or max(5, min(9, shape[-1] // 100))

        mask = np.zeros(shape, dtype=np.float32)
        mask = self.add_to_mask(mask, locations=locations, width=width)
        return np.squeeze(mask)

    def show_slide(self, index, width=None, axis='i', zoom=None, plotter=plot, **kwargs):
        """ Show slide with horizon on it.

        Parameters
        ----------
        index : int, str
            Index of the slide to show.
            If int, then interpreted as the ordinal along the specified axis.
            If `'random'`, then we generate random index along the axis.
            If string of the `'#XXX'` format, then we interpret it as the exact indexing header value.
        width : int
            Horizon thickness. If None given, set to 1% of seismic slide depth.
        axis : int
            Number of axis to load slide along.
        zoom : tuple, None or 'auto'
            Tuple of slices to apply directly to 2d images. If None, slicing is not applied.
            If 'auto', zero traces on bounds will be dropped and image will be centered on label.
        plotter : instance of `plot`
            Plotter instance to use.
            Combined with `positions` parameter allows using subplots of already existing plotter.
        """
        # Make `locations` for slide loading
        axis = self.field.geometry.parse_axis(axis)
        index = self.field.geometry.get_slide_index(index, axis=axis)

        # Load seismic and mask
        seismic_slide = self.field.geometry.load_slide(index=index, axis=axis)
        mask = self.load_slide(index=index, axis=axis, width=width)
        seismic_slide, mask = np.squeeze(seismic_slide), np.squeeze(mask)
        xmin, xmax, ymin, ymax = 0, seismic_slide.shape[0], seismic_slide.shape[1], 0

        if zoom == 'auto':
            zoom = self.compute_auto_zoom(index, axis)

        if zoom is not None:
            seismic_slide = seismic_slide[zoom]
            mask = mask[zoom]
            xmin = zoom[0].start or xmin
            xmax = zoom[0].stop or xmax
            ymin = zoom[1].stop or ymin
            ymax = zoom[1].start or ymax

        # defaults for plotting if not supplied in kwargs
        header = self.field.axis_names[axis]
        total = self.field.shape[axis]

        if axis in [0, 1]:
            xlabel = self.field.index_headers[1 - axis]
            ylabel = 'DEPTH'
        if axis == 2:
            xlabel = self.field.index_headers[0]
            ylabel = self.field.index_headers[1]
            total = self.field.depth

        title = f'{self.__class__.__name__} `{self.name}` on cube'\
                f'`{self.field.short_name}`\n {header} {index} out of {total}'

        kwargs = {
            'cmap': ['Greys_r', 'darkorange'],
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'extent': (xmin, xmax, ymin, ymax),
            'legend': False,
            'labeltop': False,
            'labelright': False,
            'curve_width': width,
            'grid': [False, True],
            'colorbar': [True, False],
            'augment_mask': [False, True],
            **kwargs
        }
        return plotter(data=[seismic_slide, mask], **kwargs)
