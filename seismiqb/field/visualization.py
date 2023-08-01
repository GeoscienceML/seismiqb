""" A mixin with field visualizations. """
#pylint: disable=global-variable-undefined, too-many-statements
import re
from copy import copy
from collections import defaultdict
from itertools import cycle

import numpy as np
from batchflow.plotter.plot import Subplot

from ..functional import compute_instantaneous_amplitude, compute_instantaneous_phase, compute_instantaneous_frequency
from ..utils import DelegatingList, to_list
from ..plotters import plot, show_3d
from ..labels.horizon.attributes import AttributesMixin

COLOR_GENERATOR = iter(Subplot.MASK_COLORS)
NAME_TO_COLOR = {}



class VisualizationMixin:
    """ Methods for field visualization: textual, 2d along various axis, 2d interactive, 3d. """
    # Textual representation
    def __repr__(self):
        return f"""<Field `{self.short_name}` at {hex(id(self))}>"""

    REPR_MAX_LEN = 100
    REPR_MAX_ROWS = 5

    def __str__(self):
        processed_prefix = 'un' if self.geometry.has_stats is False else ''
        labels_prefix = ' and labels:' if self.labels else ''
        msg = f'Field `{self.short_name}` with {processed_prefix}processed geometry{labels_prefix}\n'

        for label_src in self.loaded_labels:
            labels = getattr(self, label_src)
            names = [label.short_name for label in labels]

            labels_msg = ''
            line = f'    - {label_src}: ['
            while names:
                line += names.pop(0)

                if names:
                    line += ', '
                else:
                    labels_msg += line
                    break

                if len(line) > self.REPR_MAX_LEN:
                    labels_msg += line
                    line = '\n         ' + ' ' * len(label_src)

                if len(labels_msg) > self.REPR_MAX_LEN * self.REPR_MAX_ROWS:
                    break

            if names:
                labels_msg += f'\n         {" "*len(label_src)}and {len(names)} more item(s)'
            labels_msg += ']\n'
            msg += labels_msg
        return msg[:-1]

    # 2D along axis
    ATTRIBUTE_TO_ALIASES = {
        compute_instantaneous_amplitude: ['iamplitudes', 'instantaneous_amplitudes'],
        compute_instantaneous_phase: ['iphases', 'instantaneous_phases'],
        compute_instantaneous_frequency: ['ifrequencies', 'instantaneous_frequencies'],
    }
    ALIASES_TO_ATTRIBUTE = {alias: name for name, aliases in ATTRIBUTE_TO_ALIASES.items() for alias in aliases}

    def load_slide(self, index, axis=0, attribute=None, src_geometry='geometry'):
        """ Load one slide of data along specified axis and apply `transform`.
        Refer to the documentation of :meth:`.Geometry.load_slide` for details.

        Parameters
        ----------
        index : int, str
            If int, then interpreted as the ordinal along the specified axis.
            If `'random'`, then we generate random index along the axis.
            If string of the `'#XXX'` format, then we interpret it as the exact indexing header value.
        axis : int
            Axis of the slide.
        attribute : callable or str
            If callable, then directly applied to the loaded data.
            If str, then one of pre-defined aliases for pre-defined geological transforms.
        """
        slide = getattr(self, src_geometry).load_slide(index=index, axis=axis)

        if attribute:
            if isinstance(attribute, str) and attribute in self.ALIASES_TO_ATTRIBUTE:
                attribute = self.ALIASES_TO_ATTRIBUTE[attribute]
            if callable(attribute):
                slide = attribute(slide)
            else:
                raise ValueError(f'Unknown transform={attribute}')
        return slide


    def show_slide(self, index, axis='i', attribute=None, zoom=None, width=9,
                   src_geometry='geometry', src_labels='labels',
                   enumerate_labels=False, indices='all', augment_mask=True, plotter=plot, **kwargs):
        """ Show slide with horizon on it.

        Parameters
        ----------
        index : int, str
            Index of the slide to show.
            If int, then interpreted as the ordinal along the specified axis.
            If `'random'`, then we generate random index along the axis.
            If string of the `'#XXX'` format, then we interpret it as the exact indexing header value.
        axis : int
            Number of axis to load slide along.
        attribute : callable or str
            If callable, then directly applied to the loaded data.
            If str, then one of pre-defined aliases for pre-defined geological transforms.
        width : int
            Horizon thickness. If None given, set to 1% of seismic slide depth.
        zoom : tuple, None or 'auto'
            Tuple of slices to apply directly to 2d images. If None, slicing is not applied.
            If 'auto', zero traces on bounds will be dropped.
        """
        axis = self.geometry.parse_axis(axis)
        index = self.geometry.get_slide_index(index, axis=axis)
        locations = self.geometry.make_slide_locations(index, axis=axis)

        # Load seismic and mask
        seismic_slide = self.load_slide(index=index, axis=axis, attribute=attribute, src_geometry=src_geometry)

        src_labels = src_labels if isinstance(src_labels, (tuple, list)) else [src_labels]
        masks = []
        for src in src_labels:
            masks.append(self.make_mask(locations=locations, orientation=axis, src=src, width=width,
                                        indices=indices, enumerate_labels=enumerate_labels))
        mask = sum(masks)

        seismic_slide, mask = np.squeeze(seismic_slide), np.squeeze(mask)
        xmin, xmax, ymin, ymax = 0, seismic_slide.shape[0], seismic_slide.shape[1], 0

        if zoom == 'auto':
            zoom = self.geometry.compute_auto_zoom(index, axis)
        if zoom:
            seismic_slide = seismic_slide[zoom]
            mask = mask[zoom]
            xmin = zoom[0].start or xmin
            xmax = zoom[0].stop or xmax
            ymin = zoom[1].stop or ymin
            ymax = zoom[1].start or ymax

        # defaults for plotting if not supplied in kwargs
        header = self.geometry.axis_names[axis]
        total = self.geometry.shape[axis]

        if axis in [0, 1]:
            xlabel = self.geometry.index_headers[1 - axis]
            ylabel = 'DEPTH'
        if axis == 2:
            xlabel = self.geometry.index_headers[0]
            ylabel = self.geometry.index_headers[1]
            total = self.geometry.depth

        kwargs = {
            'cmap': ['Greys_r', 'darkorange'],
            'title': f'{header} {index} out of {total}',
            'suptitle':  f'Field `{self.short_name}`',
            'xlabel': xlabel,
            'ylabel': ylabel,
            'extent': (xmin, xmax, ymin, ymax),
            'legend': ', '.join(src_labels),
            'labeltop': False,
            'labelright': False,
            'curve_width': width,
            'grid': [None, 'both'],
            'colorbar': [True, None],
            'augment_mask': augment_mask,
            **kwargs
        }

        return plotter(data=[seismic_slide, mask], **kwargs)

    def show_section(self, locations, zoom=None, plotter=plot, linecolor='gray', linewidth=3, **kwargs):
        """ Show seismic section via desired traces.
        Under the hood relies on :meth:`load_section`, so works with geometries in any formats.

        Parameters
        ----------
        locations : iterable
            Locations of traces to construct section.
        zoom : tuple, None or 'auto'
            Tuple of slices to apply directly to 2d images. If None, slicing is not applied.
            If 'auto', zero traces on bounds will be dropped.
        plotter : instance of `plot`
            Plotter instance to use.
            Combined with `positions` parameter allows using subplots of already existing plotter.
        linecolor : str or None
            Color of line to mark node traces. If None, lines will not be drawn.
        linewidth : int
            With of the line.
        """
        self.geometry.show_section(locations, zoom=zoom, plotter=plotter, linecolor=linecolor,
                                   linewidth=linewidth, **kwargs)

    # 2D depth slice
    def show_points(self, src='labels', plotter=plot, **kwargs):
        """ Plot 2D map of labels points. Meant to be used with spatially disjoint objects (e.g. faults). """
        map_ = np.zeros(self.spatial_shape)
        denum = np.zeros(self.spatial_shape)

        for label in getattr(self, src):
            map_[label.points[:, 0], label.points[:, 1]] += label.points[:, 2]
            denum[label.points[:, 0], label.points[:, 1]] += 1
        denum[denum == 0] = 1
        map_ = map_ / denum
        map_[map_ == 0] = np.nan

        labels_class = type(getattr(self, src)[0]).__name__
        kwargs = {
            'title': f'{labels_class}s on `{self.short_name}`',
            'xlabel': self.index_headers[0],
            'ylabel': self.index_headers[1],
            'cmap': ['Reds', 'black'],
            'colorbar': True,
            'augment_mask': True,
            **kwargs
        }
        return plotter([map_, self.dead_traces_matrix], **kwargs)


    # 2D top-view maps
    def show(self, attributes='snr', mode='image', title_pattern='{attributes} of {label_name}',
             bbox=False, savepath=None, load_kwargs=None, show=True, plotter=plot, **kwargs):
        """ Show one or more field attributes on one figure.

        Parameters
        ----------
        attributes : str, np.ndarray, dict or sequence of them
            Attributes to display.
            If str, then use `:meth:.load_attribute` to load the data. For example, `geometry/snr`, `labels:0/depths`.
            If instead of label index contains `:*`, for example, `labels:*/amplitudes`, then run this method
            for each of the objects in `labels` attribute.
            If np.ndarray, then directly used as data to display.
            If dict, then should define either string or np.ndarray (and used the same as previous types),
            as well as other parameters for `:meth:.load_attribute`.
            If sequence of them, then either should be a list to display loaded entities one over the other,
            or nested list to define separate axis and overlaying for each of them.
            For more details, refer to `:func:plot`.
        mode : 'image' or 'histogram'
            Mode to display images.
        title_pattern : str with key substrings to be replaced by corresponding variables values
            If {src_label} in pattern, replaced by name of labels source (e.g. 'horizons:0').
            If {label_name} in pattern, replaced by label name (e.g. 'predicted_#3.char').
            If {attributes} in pattern, replaced by list of attributes names (e.g. '['depths', 'amplitudes']').
            If multiple labels displayed on single subplot, pattern will be repeated in title for every one of them.
        bbox : bool
            Whether crop horizon by its bounding box or not.
        savepath : str, optional
            Path to save the figure. `**` is changed to a field base directory, `*` is changed to field base name.
        load_kwargs : dict
            Loading parameters common for every requested attribute.
        show : bool
            Whether to show created plot or not.
        plotter : instance of `plot`
            Plotter instance to use.
            Combined with `positions` parameter allows using subplots of already existing plotter.
        kwargs : dict
            Additional parameters for plot creation.

        Examples
        --------
        Simplest possible plot of a geometry-related attribute:
        >>> field.show('mean_matrix')

        Display attribute of a fan over the geometry map:
        >>> field.show(['mean_matrix', 'fans:0/mask'])

        Display attributes on separate axis:
        >>> field.show(['mean_matrix', 'horizons:0/fourier', custom_data_array], combine='separate')

        Use various parameters for each of the plots:
        >>> field.show([{'src': 'labels:0/fourier', 'window': 20, 'normalize': True},
                        {'src': 'labels:0/fourier', 'window': 40, 'n_components': 3}],
                       combine='separate')

        Display amplitudes and gradients for each of the horizons in a field:
        >>> field.show(['horizons:*/amplitudes', 'horizons:*/gradient'], combine='separate')

        Display several attributes on multiple axes with overlays and save it near the cube:
        >>> field.show(['geometry/std_matrix', 'horizons:3/amplitudes',
                        ['horizons:3/instant_phases', 'fans:3/mask'],
                        ['horizons:3/instant_phases', predicted_mask]],
                       savepath='~/IMAGES/complex.png')
        """
        # Wrap given attributes load parameters in a structure that allows applying functions to its nested items
        load_params = DelegatingList(attributes)
        load_params = load_params.map(lambda item: copy(item) if isinstance(item, dict) else item)

        # Prepare data loading params
        load_params = load_params.map(self._make_load_params, common_params=load_kwargs)

        # Extract names of labels sources that require wildcard loading
        detect_wildcard = lambda params: params['src_labels'] if params['label_num'] == '*' else []
        labels_require_wildcard_loading = load_params.map(detect_wildcard).flat

        # If any attributes require wildcard loading, run `show` for every label item
        if any(labels_require_wildcard_loading):
            plotters = []

            reference_labels_source = labels_require_wildcard_loading[0]
            n_items = len(getattr(self, reference_labels_source))
            for label_num in range(n_items):
                #pylint: disable=cell-var-from-loop
                substitutor = lambda params: {**params, 'src': params['src'].replace('*', str(label_num))}
                label_attributes = load_params.map(substitutor)

                plotter_ = self.show(attributes=label_attributes, mode=mode, bbox=bbox, title_pattern=title_pattern,
                                     savepath=savepath, load_kwargs=load_kwargs, show=show, plotter=plotter, **kwargs)
                plotters.append(plotter_)

            return plotters

        data_params = load_params.map(self._load_data)

        # Prepare default plotting parameters
        plot_config = data_params.map(self._make_plot_config, mode=mode).to_dict()
        plot_config = {**plot_config, **kwargs}

        plot_config = {
            'suptitle': f'Field `{self.short_name}`',
            'augment_mask': True,
            **plot_config
        }

        if mode == 'image':
            plot_config['colorbar'] = True
            plot_config['xlabel'] = self.index_headers[0]
            plot_config['ylabel'] = self.index_headers[1]

        if title_pattern and 'title' not in plot_config:
            plot_config['title'] = data_params.map(self._make_title, shallow=True, title_pattern=title_pattern)

        if bbox:
            bboxes_list = data_params.map(lambda params: params['bbox'])
            lims_list = [np.stack([bboxes]).transpose(1, 2, 0) for bboxes in bboxes_list]
            plot_config['xlim'] = [(lims[0, 0].min(), lims[0, 1].max()) for lims in lims_list]
            plot_config['ylim'] = [(lims[1, 1].max(), lims[1, 0].min()) for lims in lims_list]

        if savepath:
            first_label_name = data_params.reference_object['label_name']
            plot_config['savepath'] = self.make_path(savepath, name=first_label_name)

        # Plot image with given params and return resulting figure
        plotter_ = plotter(mode=mode, show=show, **plot_config)
        plotter_.force_show()
        return plotter_

    # Auxilary methods utilized by `show`
    ALIAS_TO_ATTRIBUTE = AttributesMixin.ALIAS_TO_ATTRIBUTE

    def _make_load_params(self, attribute, common_params):
        # Transform load parameters into dict if needed, extract string indicating data source to use
        if isinstance(attribute, str):
            params = {'src': attribute}
        elif isinstance(attribute, np.ndarray):
            params = {'src': 'user data', 'data': attribute}
        elif isinstance(attribute, dict):
            params = copy(attribute)
        else:
            raise TypeError(f'Attribute should be either str, dict or array! Got {type(attribute)} instead.')

        # Extract source labels names and attribute names, detect if any labels sources require wildcard loading,
        # i.e. loading of data for every label stored in requested attribute (e.g. 'horizons:*/depths')
        attribute_name, label_num, src_labels = (re.split(':([0-9, *]+)/', params['src'])[::-1] + ['', 'geometry'])[:3]
        params['attribute_name'] = self.ALIAS_TO_ATTRIBUTE.get(attribute_name, attribute_name)
        params['src_labels'] = src_labels
        params['label_num'] = label_num

        # Make data loading defaults
        default_params = {'dtype': np.float32}

        if params['attribute_name'] in ['instantaneous_amplitudes', 'instantaneous_phases']:
            default_params['channels'] = 'middle'

        if params['attribute_name'] in ['fourier_decomposition', 'wavelet_decomposition']:
            default_params['n_components'] = 1

        if attribute_name in ['mask', 'full_binary_matrix']:
            params['fill_value'] = 0

        # Merge defaults with provided parameters
        params = {**default_params, **(common_params or {}), **params}

        return params

    def _load_data(self, load_params):
        params = {'attribute_name': load_params.pop('attribute_name'),
                  'src_labels': load_params.pop('src_labels'),
                  'label_num': load_params.pop('label_num')}

        postprocess = load_params.pop('postprocess', lambda x: x)

        if 'data' not in load_params:
            data, label = self.load_attribute(_return_label=True, **load_params)
            params['label_name'] = label.short_name
            params['bbox'] = label.bbox[:2]
        else:
            data = load_params['data']
            params['label_name'] = self.short_name
            params['bbox'] = np.array([[0, max] for max in data.shape])

        params['data'] = postprocess(data.squeeze())

        return params

    CMAP_TO_ATTRIBUTE = {
        'Depths': ['full_matrix'],
        'Reds': ['spikes', 'quality_map', 'quality_grid'],
        'Metric': ['metric'],
        'RdYlGn': ['probabilities']
    }
    ATTRIBUTE_TO_CMAP = {attr: cmap for cmap, attributes in CMAP_TO_ATTRIBUTE.items()
                         for attr in attributes}

    def _make_plot_config(self, data_params, mode):
        params = {'data': data_params['data']}

        src_labels = data_params['src_labels']
        attribute_name = data_params['attribute_name']

        # Choose default cmap
        if attribute_name == 'full_binary_matrix' or mode == 'histogram':
            global_name = f"{src_labels}/{attribute_name}"
            if global_name not in NAME_TO_COLOR:
                NAME_TO_COLOR[global_name] = next(COLOR_GENERATOR)
            cmap = NAME_TO_COLOR[global_name]
        else:
            cmap = self.ATTRIBUTE_TO_CMAP.get(attribute_name, 'Seismic')

        params['cmap'] = cmap

        # Choose default alpha
        if attribute_name in ['full_binary_matrix']:
            alpha = 0.7
        else:
            alpha = 1.0

        # Bounds for metrics
        if 'metric' in attribute_name:
            params['vmin'], params['vmax'] = -1.0, 1.0

        params['alpha'] = alpha

        return params

    def _make_title(self, data_params, title_pattern):
        linkage = defaultdict(list)

        for params in to_list(data_params):
            if isinstance(params, list):
                params = params[0]
            src_label = params['src_labels']
            if params['label_num']:
                src_label += ':' + params['label_num']
            label_name = params['label_name']

            linkage[(src_label, label_name)].append(params['attribute_name'])

        title = ''
        for (src_label, label_name), attributes in linkage.items():
            title += '\n' * (title != '')
            part = title_pattern
            part = part.replace('{src_label}', src_label)
            part = part.replace('{label_name}', label_name)
            part = part.replace('{attributes}', ','.join(attributes))
            title += part

        return title

    # 2D interactive
    def viewer(self, figsize=(8, 8), **kwargs):
        """ Interactive field viewer. """
        from .viewer import FieldViewer #pylint: disable=import-outside-toplevel
        return FieldViewer(field=self, figsize=figsize, **kwargs)


    # 3D interactive
    def show_3d(self, src='labels', aspect_ratio=None, zoom=None, n_points=100, threshold=100,
                sticks_step=None, stick_nodes_step=None, sticks=False, stick_orientation=None,
                slides=None, margin=(0, 0, 20), colors=None, **kwargs):
        """ Interactive 3D plot for some elements of a field.
        Roughly, does the following:
            - take some faults and/or horizons
            - select `n` points to represent the horizon surface and `sticks_step` and `stick_nodes_step` for each fault
            - triangulate those points
            - remove some of the triangles on conditions
            - use Plotly to draw the tri-surface
            - draw few slides of the cube if needed

        Parameters
        ----------
        src : str, Horizon-instance or list
            Items to draw, by default, 'labels'. If item of list (or `src` itself) is str, then all items of
            that dataset attribute will be drawn.
        aspect_ratio : None, tuple of floats or Nones
            Aspect ratio for each axis. Each None in the resulting tuple will be replaced by item from
            `(geometry.shape[0] / geometry.shape[1], 1, 1)`.
        zoom : tuple of slices or None
            Crop from cube to show. By default, the whole cube volume will be shown.
        n_points : int
            Number of points for horizon surface creation.
            The more, the better the image is and the slower it is displayed.
        threshold : number
            Threshold to remove triangles with bigger depth differences in vertices.
        sticks_step : int or None
            Number of slides between sticks. If None, fault triangulation (nodes and simplices) will be used.
        stick_nodes_step : int or None
            Distance between stick nodes. If None, fault triangulation (nodes and simplices) will be used.
        sticks : bool
            If True, show fault sticks. If False, show interpolated surface.
        stick_orientation : 0, 1 or 2
            Axis which defines stick_orientation
        slides : list of tuples
            Each tuple is pair of location and axis to load slide from seismic cube.
        margin : tuple of ints
            Added margin for each axis, by default, (0, 0, 20).
        colors : dict, list or str.
            Mapping of label class name to color defined as str, by default, all labels will be shown in green.
            Also can be 'random' to set all label items colors randomly.
        show_axes : bool
            Whether to show axes and their labels.
        width, height : number
            Size of the image.
        savepath : str
            Path to save interactive html to.
        kwargs : dict
            Other arguments of plot creation.
        """
        src = src if isinstance(src, (tuple, list)) else [src]
        coords = []
        simplices = []

        if zoom is None:
            zoom = [slice(0, s) for s in self.shape]
        else:
            zoom = [
                slice(item.start or 0, item.stop or stop) for item, stop in zip(zoom, self.shape)
            ]
        zoom = tuple(zoom)
        triangulation_kwargs = {
            'n_points': n_points,
            'threshold': threshold,
            'sticks_step': sticks_step,
            'stick_nodes_step': stick_nodes_step,
            'slices': zoom,
            'sticks': sticks,
            'stick_orientation': stick_orientation
        }

        labels = [getattr(self, src_) if isinstance(src_, str) else [src_] for src_ in src]
        labels = sum(labels, [])

        if colors == 'random':
            colors = ['rgb(' + ', '.join([str(c) for c in np.random.randint(0, 255, size=3)]) + ')' for _ in labels]
        if isinstance(colors, str):
            colors = [colors]
        if isinstance(colors, list):
            cycled_colors = cycle(colors)
            colors = [next(cycled_colors) for _ in range(len(labels))]

        if colors is None:
            colors = ['green' for _ in labels]
        if isinstance(colors, dict):
            colors = [colors.get(type(label).__name__, colors.get('all', 'green')) for label in labels]

        simplices_colors = []
        for label, color in zip(labels, colors):
            if label is not None:
                x, y, z, simplices_ = label.make_triangulation(**triangulation_kwargs)
                if len(simplices_) == 0:
                    continue
                if x is not None:
                    simplices += [simplices_ + sum(len(item) for item in coords)]
                    simplices_colors += [[color] * len(simplices_)]
                    coords += [np.stack([x, y, z], axis=1)]

        if len(simplices) > 0:
            simplices = np.concatenate(simplices, axis=0)
            coords = np.concatenate(coords, axis=0)
            simplices_colors = np.concatenate(simplices_colors)
        else:
            simplices = None
            coords = np.zeros((0, 3))
            simplices_colors = None
        title = self.short_name

        default_aspect_ratio = (self.shape[0] / self.shape[1], 1, 1)
        aspect_ratio = [None] * 3 if aspect_ratio is None else aspect_ratio
        aspect_ratio = [item or default for item, default in zip(aspect_ratio, default_aspect_ratio)]

        axis_labels = (self.index_headers[0], self.index_headers[1], 'DEPTH')

        images = []
        if slides is not None:
            for loc, axis in slides:
                image = self.geometry.load_slide(loc, axis=axis)
                if axis == 0:
                    image = image[zoom[1:]]
                elif axis == 1:
                    image = image[zoom[0], zoom[-1]]
                else:
                    image = image[zoom[:-1]]
                images += [(image, loc, axis)]

        show_3d(coords[:, 0], coords[:, 1], coords[:, 2], simplices, title, zoom, simplices_colors, margin=margin,
                aspect_ratio=aspect_ratio, axis_labels=axis_labels, images=images, **kwargs)
