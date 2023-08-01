""" A container for all information about the field: geometry and labels, as well as convenient API. """
import os
import re
import itertools
from glob import glob
from difflib import get_close_matches
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from batchflow.notifier import Notifier

from .visualization import VisualizationMixin
from ..geometry import Geometry
from ..labels import Horizon, Fault
from ..metrics import FaciesMetrics
from ..utils import AugmentedList
from ..utils import CharismaMixin



class Field(CharismaMixin, VisualizationMixin):
    """ A common container for all information about the field: cube geometry and various labels.

    To initialize, one must provide:
        - geometry-like entity, which can be a path to a seismic cube or instance of `:class:Geometry`;
        additional parameters of geometry instantiation can be passed via `geometry_kwargs` parameters.

        - optionally, `labels` in one of the following formats:
            - dictionary with keys defining attribute to store loaded labels in and values as
            sequences of label-like entities (path to a label or instance of label class)
            - sequence with label-like entities. This way, labels will be stored in `labels` attribute
            - string to define path(s) to labels (same as those paths wrapped in a list)
            - None as a signal that no labels are provided for a field.

        - `labels_class` defines the class to use for loading and can be supplied in one of the following formats:
            - dictionary with same keys as in `labels`. Values are either string (e.g. `horizon`) or
            the type to initialize label itself (e.g. `:class:.Horizon`)
            - a single string or type to use for all of the labels
            - if not provided, we try to infer the class from name of the attribute to store the labels in.
            The guess is based on a similarity between passed name and a list of pre-defined label types.
            For example, `horizons` will be threated as `horizon` and loaded as such.
            >>> {'horizons': 'path/to/horizons/*'}
            would be loaded as instances of `:class:.Horizon`.

        - `labels_kwargs` are passed for instantiation of every label.

    Examples
    --------
    Initialize field with only geometry:
    >>> Field(geometry='path/to/cube.sgy')
    >>> Field(geometry=Geometry(...))

    The most complete labels definition:
    >>> Field(geometry=..., labels={'horizons': ['path/to/horizon', Horizon(...)],
                                    'fans': 'paths/to/fans/*',
                                    'faults': ['path/to/fault1', 'path/to/fault2', ],
                                    'lift_geometry': 'path/to/geometry_target.hdf5'})

    Use a `labels_class` instead; this way, all of the labels are stored as `labels` attribute, no matter the class:
    >>> Field(geometry=..., labels='paths/*', labels_class='horizon')
    >>> Field(geometry=..., labels=['paths/1', 'paths/2', 'paths/3'], labels_class='fault')
    """
    #pylint: disable=redefined-builtin
    def __init__(self, geometry, labels=None, labels_class=None, geometry_kwargs=None, labels_kwargs=None, **kwargs):
        # Attributes
        self.labels = []
        self.horizons, self.facies, self.fans, self.channels, self.faults = [], [], [], [], []
        self.loaded_labels = []

        # Geometry: description and convenient API to a seismic cube
        if isinstance(geometry, str):
            geometry_kwargs = geometry_kwargs or {}
            geometry = Geometry.new(geometry, **{**kwargs, **geometry_kwargs})
        self.geometry = geometry

        # Labels: objects on a field
        if labels:
            labels_kwargs = labels_kwargs or {}
            self.load_labels(labels, labels_class, **{**kwargs, **labels_kwargs})


    # Label initialization inner workings
    METHOD_TO_NAMES = {
        '_load_horizons': ['horizon', 'facies', 'fans', 'channels', Horizon],
        '_load_faults': ['fault', Fault],
        '_load_geometries': ['geometries', 'geometry',  Geometry],
    }
    NAME_TO_METHOD = {name: method for method, names in METHOD_TO_NAMES.items() for name in names}

    def load_labels(self, labels=None, labels_class=None, mode='w', **labels_kwargs):
        """ Load labels and store them in the instance. Refer to the class documentation for details. """
        if isinstance(labels, str):
            labels = self.make_path(labels, makedirs=False)
            labels = glob(labels)
        if isinstance(labels, (tuple, list)):
            labels = {'labels': labels}
        if not isinstance(labels, dict):
            raise TypeError(f'Labels type should be `str`, `sequence` or `dict`, got {type(labels)} instead!')

        # Labels class: make a dictionary
        if labels_class is None:
            labels_class_dict = {label_dst : None for label_dst in labels.keys()}
        if isinstance(labels_class, (type, str)):
            labels_class_dict = {label_dst : labels_class for label_dst in labels.keys()}
        if isinstance(labels_class, dict):
            labels_class_dict = labels_class

        for label_dst, label_src in labels.items():
            # Try getting provided `labels_class`, else fallback on NAME_TO_METHOD closest match
            label_class = labels_class_dict.get(label_dst)

            if label_class is None:
                # Roughly equivalent to ``label_class = self.NAME_TO_METHOD.get(label_dst)``
                str_names = [name for name in (self.NAME_TO_METHOD.keys())
                             if isinstance(name, str)]
                matched = get_close_matches(label_dst, str_names, n=1)
                if matched:
                    label_class = matched[0]

            if label_class is None:
                raise TypeError(f"Can't determine the label class for `{label_dst}`!")

            if isinstance(label_class, str):
                method = self.NAME_TO_METHOD[label_class]
                label_class = self.METHOD_TO_NAMES[method][-1]

            # Process paths: get rid of service files
            if isinstance(label_src, str):
                label_src = self.make_path(label_src, makedirs=False)
                label_src = glob(label_src)
            if not isinstance(label_src, (tuple, list)):
                label_src = [label_src]
            label_src = self._filter_paths(label_src)

            # Load desired labels, based on class
            method_name = self.NAME_TO_METHOD[label_class]
            method = getattr(self, method_name)
            result = method(label_src, label_class=label_class, **labels_kwargs)

            if mode == 'w':
                setattr(self, label_dst, result)
                self.loaded_labels.append(label_dst)
            else:
                if not hasattr(self, label_dst):
                    setattr(self, label_dst, AugmentedList())
                else:
                    getattr(self, label_dst).extend(result)

                if label_dst not in self.loaded_labels:
                    self.loaded_labels.append(label_dst)

            if 'labels' not in labels and not self.labels:
                setattr(self, 'labels', result)

    @staticmethod
    def _filter_paths(paths):
        """ Remove paths for service files. """
        return [path for path in paths
                if not isinstance(path, str) or \
                not any(ext in path for ext in ['.dvc', '.gitignore', '.meta', '.sgy_meta'])]

    def _load_horizons(self, paths, max_workers=4, filter=True, interpolate=False, sort=True,
                       label_class=Horizon, **kwargs):
        """ Load horizons from paths or re-use already created ones. """
        # Separate paths from ready-to-use instances
        horizons, paths_to_load = [], []
        for item in paths:
            if isinstance(item, str):
                paths_ = self._filter_paths(glob(item))
                paths_to_load.extend(paths_)

            elif isinstance(item, label_class):
                item.field = self
                horizons.append(item)

        # Load from paths in multiple threads
        with ThreadPoolExecutor(max_workers=min(max_workers, len(paths_to_load) or 1)) as executor:
            function = lambda path: self._load_horizon(path, filter=filter, interpolate=interpolate,
                                                       constructor_class=label_class, **kwargs)
            loaded = list(executor.map(function, paths_to_load))
        horizons.extend(loaded)

        if sort:
            sort = sort if isinstance(sort, str) else 'd_mean'
            horizons.sort(key=lambda label: getattr(label, sort))
        return horizons

    def _load_horizon(self, path, filter=True, interpolate=False, constructor_class=Horizon, **kwargs):
        """ Load a single horizon from path. """
        horizon = constructor_class(path, field=self, **kwargs)
        if filter:
            horizon.filter(inplace=True)
        if interpolate:
            horizon.interpolate(inplace=True)
        return horizon


    def _load_faults(self, paths, max_workers=4, pbar='t', interpolate=False, label_class=Fault, **kwargs):
        """ Load faults from paths. """
        with ThreadPoolExecutor(max_workers=min(max_workers, len(paths) or 1)) as executor:
            function = lambda path: self._load_fault(path, interpolate=interpolate,
                                                     constructor_class=label_class, **kwargs)
            loaded = list(Notifier(pbar, total=len(paths))(executor.map(function, paths)))

        faults = [fault for fault in itertools.chain(*loaded) if len(fault) > 0]
        return faults

    def _load_fault(self, path, interpolate=False, constructor_class=Fault, **kwargs):
        """ Load a single fault from path. """
        if isinstance(path, constructor_class):
            path.field = self
            return [path]

        faults = constructor_class.load(path, self, interpolate=interpolate, **kwargs)
        return faults

    def _load_geometries(self, paths, constructor_class=Geometry.new, **kwargs):
        if isinstance(paths, str):
            path = paths
        if isinstance(paths, (tuple, list)):
            if not paths:
                raise FileNotFoundError('No such file or directory')
            if len(paths) > 1:
                raise ValueError(f'Path for Geometry loading is non-unique!, {paths}')
            path = paths[0]
        return constructor_class(path, **kwargs)


    # Other methods of initialization
    @classmethod
    def from_horizon(cls, horizon):
        """ Create a field from a single horizon. """
        return cls(geometry=horizon.geometry, labels={'horizons': horizon})

    @classmethod
    def from_dvc(cls, tag, dvc_path=''):
        """ Create a field from a dvc tag. """


    # Inner workings
    def __getattr__(self, key):
        """ Redirect calls for missing attributes, properties and methods to `geometry`. """
        if not key.endswith('state__') and 'geometry' in self.__dict__ and hasattr(self.geometry, key):
            return getattr(self.geometry, key)
        raise AttributeError(f'Attribute `{key}` does not exist in either Field or associated Geometry!')

    def __getattribute__(self, key):
        """ Wrap every accessed list with `AugmentedList`.
        The wrapped attribute is re-stored in the instance, so that we return the same object as in the instance. """
        result = super().__getattribute__(key)
        if isinstance(result, list) and not isinstance(result, AugmentedList):
            result = AugmentedList(result)
            if not (key in vars(self.__class__) and isinstance(getattr(self.__class__, key), property)):
                setattr(self, key, result)
        return result


    # Methods to call from Batch
    def load_seismic(self, locations, src='geometry', buffer=None, **kwargs):
        """ Load data from cube.

        Parameters
        ----------
        locations : sequence
            A triplet of slices to specify the location of a subvolume.
        src : str
            Attribute with desired geometry.
        """
        _ = kwargs
        geometry = getattr(self, src)
        if buffer is None:
            shape = geometry.locations_to_shape(locations)
            buffer = np.empty(shape, dtype=np.float32)

        geometry.load_crop(locations, buffer=buffer)
        return buffer

    def make_mask(self, locations, orientation=0, buffer=None, indices='all', width=3, src='labels',
                  sparse=False, enumerate_labels=False, **kwargs):
        """ Create masks from labels.

        Parameters
        ----------
        location : int or sequence
            If integer, then location along specified `axis`.
            Otherwise, a triplet of slices to define exact location in the cube.
        axis : int or str
            Axis identifier. must be provided if `location` is integer.
        indices : str, int or sequence of ints
            Which labels to use in mask creation.
            If 'all', then use all labels.
            If 'single' or `random`, then use one random label.
            If int or array-like, then element(s) are interpreted as indices of desired labels.
        width : int
            Width of the resulting label.
        src : str
            Attribute with desired labels.
        sparse : bool
            Create mask only for labeled slices (for Faults). Unlabeled slices will be marked by -1.
        enumerate_labels : bool
            Whether to use enumeration as mask values for successive labels.
        """
        # Parse buffer
        if buffer is None:
            shape = self.geometry.locations_to_shape(locations)
            buffer = np.zeros(shape, dtype=np.float32)

        if sparse:
            buffer -= 1

        # Parse requested labels
        labels = getattr(self, src)
        labels = [labels] if not isinstance(labels, (tuple, list)) else labels
        if len(labels) == 0:
            return buffer

        indices = [indices] if isinstance(indices, int) else indices
        if isinstance(indices, (tuple, list, np.ndarray)):
            labels = [labels[idx] for idx in indices]

        # Add mask of each component to the buffer
        for i, label in enumerate(labels, start=1):
            alpha = 1 if enumerate_labels is False else i
            label.add_to_mask(buffer, locations=locations, width=width, axis=orientation,
                              sparse=sparse, alpha=alpha, **kwargs)
        return buffer


    def make_regression_mask(self, location, axis=None, indices='all', src='labels', **kwargs):
        """ Create regression mask from labels. Works only with horizons. """
        #
        labels = getattr(self, src)
        labels = [labels] if not isinstance(labels, (tuple, list)) else labels

        if len(labels) > 1:
            pass
        else:
            shape = tuple(slc.stop - slc.start for slc in location[:2])
            mask = np.full((*shape, 1), -1, dtype=np.float32)
            labels[0].add_to_regression_mask(mask=mask[..., 0], locations=location, **kwargs)
        return mask


    # Attribute retrieval
    def load_attribute(self, src, _return_label=False, **kwargs):
        """ Load desired geological attribute from geometry or labels.

        Parameters
        ----------
        src : str
            Identificator of `what` to load and `from where`.
            The part before the slash identifies the instance, for example: `geometry`, `horizons:0`, `faults:123`.
            In general it is `attribute_name:idx`, where `attribute_name` is the attribute to retrieve, and
            optional `idx` can be used to slice it.
            The part after the slash is passed directly to instance's `load_attribute` method.
        kwargs : dict
            Additional parameters for attribute computation.
        """
        # Prepare `src`
        src = src.strip('/')
        if '/' not in src:
            src = 'geometry/' + src

        label_id, *src = src.split('/')
        src = '/'.join(src)

        # Select instance
        if any(sep in label_id for sep in ':-'):
            label_attr, label_idx = re.split(':|-', label_id)

            if label_attr not in self.loaded_labels:
                raise ValueError(f"Can't determine the label attribute for `{label_attr}`!")
            label_idx = int(label_idx)
            label = getattr(self, label_attr)[label_idx]
        else:
            label = getattr(self, label_id)

        data = label.load_attribute(src, **kwargs)

        if _return_label:
            return data, label
        return data

    @property
    def available_attributes(self):
        """ A list of all load-able attributes from a current field. """
        #pylint: disable=unidiomatic-typecheck
        available_names = []

        for name in ['geometry'] + self.loaded_labels:
            labels = getattr(self, name)

            if isinstance(labels, list):
                for idx, label in enumerate(labels):
                    if type(label) is Horizon:
                        available_attributes = ['depths', 'amplitudes', 'metrics',
                                                'instant_amplitudes', 'instant_phases',
                                                'fourier_decomposition', 'wavelet_decomposition']
                    else:
                        available_attributes = []
                    available_names.extend([f'{name}:{idx}/{attr}' for attr in available_attributes])
            else:
                if isinstance(labels, Geometry):
                    available_attributes = ['mean_matrix', 'std_matrix', 'snr']
                available_names.extend([f'{name}/{attr}' for attr in available_attributes])
        return available_names


    # Utility functions
    def make_path(self, path, name=None, makedirs=True):
        """ Make path by mapping some of the symbols into pre-defined strings:
            - `**` or `%` is replaced with basedir of a cube
            - `*` is replaced with `name`

        Parameters
        ----------
        path : str
            Path to process.
        name : str
            Replacement for `*` symbol.
        makedirs : bool
            Whether to make dirs preceding the path.
        """
        basedir = os.path.dirname(self.path)
        name = name or self.short_name

        path = (path.replace('~', basedir)
                    .replace('$', name)
                    .replace('//', '/'))

        if makedirs and os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        return path


    # Cache: introspection and reset
    @property
    def attached_instances(self):
        """ All correctly loaded/added instances. """
        instances = []
        for src in self.loaded_labels:
            item = getattr(self, src)
            if isinstance(item, list):
                instances.extend(item)
            else:
                instances.append(item)
        return instances

    def reset_cache(self):
        """ Clear cached data from underlying entities. """
        self.geometry.reset_cache()
        self.attached_instances.reset_cache()

    @property
    def cache_nbytes(self):
        """ Total nbytes of cached data. """
        nbytes = self.geometry.cache_nbytes

        if len(self.attached_instances) > 0:
            nbytes += sum(self.attached_instances.cache_nbytes)
        return nbytes

    # Facies
    def evaluate_facies(self, src_horizons, src_true=None, src_pred=None, metrics='dice'):
        """ Calculate facies metrics for requested labels of the field and return dataframe of results.

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
        horizons = getattr(self, src_horizons)
        true_labels = getattr(self, src_true) if src_true is not None else None
        pred_labels = getattr(self, src_pred) if src_pred is not None else None

        fm = FaciesMetrics(horizons=horizons, true_labels=true_labels, pred_labels=pred_labels)
        result = fm.evaluate(metrics)

        return result
