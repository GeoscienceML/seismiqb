""" Container for storing seismic data and labels. """
#pylint: disable=too-many-lines, too-many-arguments
from textwrap import indent

import numpy as np
import pandas as pd

from batchflow import DatasetIndex, Dataset, Pipeline

from .field import Field, SyntheticField
from .geometry import Geometry
from .batch import SeismicCropBatch
from .utils import AugmentedDict
from .plotters import plot

class SeismicDataset(Dataset):
    """ Container of fields.

    Getitem is re-defined to index stored fields.
    Getattr is re-defined to return the same attributes from all stored fields, wrapped into `AugmentedDict`.

    Can be initialized with:
        - a nested dictionary, where keys are field-like entities (path to seismic cube, instance of Geometry or Field),
        and values are either:
            - dictionary with keys defining attribute to store loaded labels in and values as
            sequences of label-like entities (path to a label or instance of label class)
            - sequence with label-like entities. This way, labels will be stored in `labels` attribute
            - string to define path(s) to labels (same as those paths wrapped in a list)
            - None as a signal that no labels are provided for a field.
        - a sequence with field-like entities (same as dictionary where every value is None)
        - one field-like entity (same as sequence with only one element)
    Named arguments are passed for each field initialization.
    """
    #pylint: disable=keyword-arg-before-vararg
    def __init__(self, index, batch_class=SeismicCropBatch, *args, **kwargs):
        if args:
            raise TypeError('Positional args are not allowed for `SeismicDataset` initialization!')

        # Convert `index` to a dictionary
        if isinstance(index, (str, Geometry, Field, SyntheticField)):
            index = [index]
        if isinstance(index, (tuple, list, DatasetIndex)):
            index = {item : None for item in index}

        if isinstance(index, dict):
            self.fields = AugmentedDict()
            for field_idx, labels_idx in index.items():
                if isinstance(field_idx, (Field, SyntheticField)):
                    field = field_idx
                    if labels_idx is not None:
                        field.load_labels(labels=labels_idx, **kwargs)
                else:
                    field = Field(geometry=field_idx, labels=labels_idx, **kwargs)

                self.fields[field.short_name] = field
        else:
            raise TypeError('Dataset should be initialized with a string, a ready-to-use Geometry or Field,'
                            f' sequence or a dict, got {type(index)} instead.')

        dataset_index = DatasetIndex(list(self.fields.keys()))
        super().__init__(dataset_index, batch_class=batch_class)
        self._names = None

    @classmethod
    def from_horizon(cls, horizon):
        """ Create dataset from an instance of Horizon. """
        return cls({horizon.field.geometry : {'horizons': [horizon]}})


    # Inner workings
    def __getitem__(self, key):
        """ Index a field with either its name or ordinal. """
        if isinstance(key, (int, np.integer, str)):
            return self.fields[key]
        raise KeyError(f'Unsupported key for subscripting, {key}')


    def get_nested_iterable(self, attribute):
        """ Create an `AugmentedDict` with field ids as keys and their `attribute` as values.
        For example, `dataset.get_nested_iterable('labels')` would
        return an `AugmentedDict` with labels for every field.
        """
        return AugmentedDict({idx : getattr(field, attribute) for idx, field in self.fields.items()})

    def __getattr__(self, key):
        """ Create nested iterables for a key.
        For example, `dataset.labels` would return an `AugmentedDict` with labels for every field.
        """
        if isinstance(key, str) and key not in self.indices:
            return self.get_nested_iterable(key)
        raise AttributeError(f'Unknown attribute {key}')

    @property
    def names(self):
        """ 2D index of available fields and labels. """
        if self._names is None:
            names = {}
            for i, (field_name, field_labels) in enumerate(self.labels.items()):
                for j, label in enumerate(field_labels):
                    names[(i, j)] = field_name, label.short_name
            self._names = names
        return self._names

    def to_names(self, id_array):
        """ Convert the first two columns of sampled locations into field and label string names. """
        return np.array([self.names[tuple(ids)] for ids in id_array])


    def gen_batch(self, batch_size=None, shuffle=False, n_iters=None, n_epochs=None, drop_last=False, **kwargs):
        """ Remove `n_epochs`  and `drop_last` from passed arguments.
        Set default value `batch_size` to the size of current dataset, removing the need to
        pass it to `next_batch` and `run` methods.
        """
        if (n_epochs is not None and n_epochs != 1) or drop_last:
            raise TypeError(f'`SeismicCubeset` does not work with `n_epochs`, `shuffle` or `drop_last`!'
                            f'`{n_epochs}`, `{shuffle}`, `{drop_last}`')

        batch_size = batch_size or len(self)
        return super().gen_batch(batch_size, n_iters=n_iters, shuffle=shuffle, **kwargs)


    # Default pipeline and batch for fast testing / introspection
    def data_pipeline(self, sampler, batch_size=4, width=4):
        """ Pipeline with default actions of creating locations, loading seismic images and corresponding masks. """
        return (self.p
                .make_locations(generator=sampler, batch_size=batch_size)
                .create_masks(dst='masks', width=width)
                .load_cubes(dst='images')
                .adaptive_reshape(src=['images', 'masks'])
                .normalize(src='images'))

    def data_batch(self, sampler, batch_size=4, width=4):
        """ Get one batch of `:meth:.data_pipeline` with `images` and `masks`. """
        return self.data_pipeline(sampler=sampler, batch_size=batch_size, width=width).next_batch()


    # Textual and visual representation of dataset contents
    def __str__(self):
        msg = f'Seismic Dataset with {len(self)} field{"s" if len(self) > 1 else ""}:\n'
        msg += '\n\n'.join([indent(str(field), prefix='    ') for field in self.fields.values()])
        return msg


    def show_slide(self, loc, idx=0, axis='iline', zoom=None, src_labels='labels',
                   indices='all', width=5, plotter=plot, **kwargs):
        """ Show slide of the given cube on the given line.

        Parameters
        ----------
        loc : int
            Number of slide to load.
        idx : str, int
            Number of cube in the index to use.
        axis : int or str
            Number or name of axis to load slide along.
        zoom : tuple of slices, None or 'auto'
            Tuple of slices to apply directly to 2d images. If None, slicing is not applied.
            If 'auto', zero traces on bounds will be dropped.
        src_labels : str
            Dataset components to show as labels.
        indices : str, int or sequence of ints
            Which labels to use in mask creation.
            If 'all', then use all labels.
            If 'single' or `random`, then use one random label.
            If int or array-like, then element(s) are interpreted as indices of desired labels.
        width : int
            Width of the resulting label.
        plotter : instance of `plot`
            Plotter instance to use.
            Combined with `positions` parameter allows using subplots of already existing plotter.
        """
        components = ('images', 'masks') if getattr(self, src_labels)[idx] else ('images',)
        cube_name = self.indices[idx]
        geometry = self.fields[cube_name].geometry
        crop_shape = np.array(geometry.shape)

        axis = geometry.parse_axis(axis)
        crop_shape[axis] = 1

        location = np.zeros((1, 9), dtype=np.int32)
        location[0, 2] = axis
        location[0, axis + 3] = loc
        location[0, axis + 6] = loc
        location[0, [6, 7, 8]] += crop_shape

        # Fake generator with one point only
        generator = lambda batch_size: location
        generator.to_names = lambda array: np.array([[cube_name, 'unknown']])

        pipeline = (Pipeline()
                    .make_locations(generator=generator)
                    .load_cubes(dst='images', src_labels=src_labels)
                    .normalize(src='images'))

        if 'masks' in components:
            labels_pipeline = (Pipeline()
                               .create_masks(src_labels=src_labels, dst='masks', width=width, indices=indices))

            pipeline = pipeline + labels_pipeline

        batch = (pipeline << self).next_batch()
        # TODO: Make every horizon mask creation individual to allow their distinction while plot.
        data = [np.squeeze(getattr(batch, comp)) for comp in components]
        xmin, xmax, ymin, ymax = 0, data[0].shape[0], data[0].shape[1], 0

        if zoom == 'auto':
            zoom = geometry.compute_auto_zoom(loc, axis)
        if zoom:
            data = [image[zoom] for image in data]
            xmin = zoom[0].start or xmin
            xmax = zoom[0].stop or xmax
            ymin = zoom[1].stop or ymin
            ymax = zoom[1].start or ymax

        # Plotting defaults
        header = geometry.axis_names[axis]
        total = geometry.shape[axis]

        if axis in [0, 1]:
            xlabel = geometry.index_headers[1 - axis]
            ylabel = 'DEPTH'
        if axis == 2:
            xlabel = geometry.index_headers[0]
            ylabel = geometry.index_headers[1]

        kwargs = {
            'cmap': ['Greys_r', 'darkorange'],
            'title': f'Data slice on cube `{geometry.short_name}`\n {header} {loc} out of {total}',
            'xlabel': xlabel,
            'ylabel': ylabel,
            'extent': (xmin, xmax, ymin, ymax),
            'legend': src_labels,
            'augment_mask': [False, True],
            **kwargs
        }

        return plotter(data, **kwargs)

    # Facies
    def evaluate_facies(self, src_horizons, src_true=None, src_pred=None, metrics='dice'):
        """ Calculate facies metrics for requested labels of the dataset and return dataframe of results.

        Parameters
        ----------
        scr_horizons : str
            Name of field attribute that contains base horizons.
        src_true : str
            Name of field attribute that contains ground-truth labels.
        src_pred : str
            Name of field attribute that contains predicted labels.
        metrics: str or list of str
            Metrics function(s) to calculate.
        """
        metrics_values = self.fields.evaluate_facies(src_horizons=src_horizons, src_true=src_true,
                                                     src_pred=src_pred, metrics=metrics)
        result = pd.concat(metrics_values.flat)

        return result
