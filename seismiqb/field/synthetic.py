""" A wrapper around `SyntheticGenerator` to provide the same API, as regular `:class:.Field`. """
import numpy as np

from batchflow import Config
from ..utils import lru_cache
from ..plotters import plot

class GeometryMock:
    """ Mock for Geometry. """
    def __getattr__(self, _):
        return None


class SyntheticField:
    """ A wrapper around `SyntheticGenerator` to provide the same API, as regular `:class:.Field`.

    The intended use of this class is:
        - define a `param_generator` function, that returns a dictionary with parameters of seismic generation.
        The parameters may be randomized, so the generated data is different each time.
        - initialize an instance of this class.
        - use `get_attribute` method to get synthetic images, horizon/fault masks, velocity models.
        In order to ensure that they match one another, supply the same `locations` (triplet of slices), for example:
        >>> locations = (slice(10, 11), slice(100, 200), slice(1000, 1500))
        >>> synthetic = synthetic_field.get_attribute(locations=locations, attribute='synthetic')
        >>> velocity = synthetic_field.get_attribute(locations=locations, attribute='velocity')
        would make synthetic and velocity images for the same underlying synthetic model.

    Using the same generator for multiple calls with different requested attributes relies on LRU caching of
    the generator instances. Due to this, the `cache_maxsize` should always be bigger than the number of successive
    calls to `get_attribute` with the same `attribute`. If synthetic field is used in any data loading pipelines,
    this traslates to having `cache_maxsize` bigger than the `batch_size`.

    Methods `load_seismic` and `make_masks` are thin wrappers around `get_attribute` to make API of this class
    identical to that of `:class:.Field`.

    Under the hood, we keep track of internal cache (with `locations` as key) to use the same instance of generator
    multiple times. The size of cache is parametrized at initialization and should be bigger than the batch size.
    Other than that, the `locations` is also used to infer shape of requested synthetic data,
    if it is not provided at initialization / from `param_generator`.

    Parameters
    ----------
    param_generator : callable, optional
        If provided, should return a dictionary with parameters to generate synthetic.
        Refer to `:meth:.default_param_generator` for example of implementation.
        Can be omitted if the `data_generator` is supplied instead.
    data_generator : callable, optional
        If provided, then a callable to populate an instance of `SyntheticGenerator` with data.
        Should take `generator` as the only required argument. Disables the `param_generator` option.
        Note that the logic of keeping the same instance of `generator` for multiple calls with the same `locations`
        is performed by class internals and still available in that case.
    attribute : str
        Attribute to get from the generator if `labels` are requested.
    crop_shape : tuple of int
        Default shape of the generated synthetic images.
        If not provided, we use the shape from `param_generator` or `locations`.
    name : str
        Name of the the field. Used to comply with `:class:.Field` API.
    cache_maxsize : int
        Number of cached generators. Should be equal or bigger than the batch size.
    """
    GENERATOR_CONSTRUCTOR = None

    #pylint: disable=method-hidden, protected-access, not-callable
    def __init__(self, param_generator=None, data_generator=None, name='synthetic_field', cache_maxsize=128,
                 default_attribute=None, default_shape=None):
        # Data generation
        self.param_generator = param_generator
        self.data_generator = data_generator
        self._make_generator = lru_cache(maxsize=cache_maxsize)(self._make_generator)
        self._cache_maxsize = cache_maxsize

        # Defaults
        self.default_attribute = default_attribute
        self.default_shape = default_shape

        # String info
        self.path = self.short_path = f'{name}_path'
        self.name = self.short_name = self.short_name = name
        self.index_headers = self.axis_names = ['INLINE_3D', 'CROSSLINE_3D']

        # Attributes to comply with `:class:.Field` API
        self.geometry = GeometryMock()
        self.spatial_shape = (-1, -1)
        self.shape = (-1, -1, -1)
        self.depth = -1
        self.dead_traces_matrix = self.mean_matrix = self.std_matrix = None

        # Properties
        self._normalization_stats = None

    @property
    def labels(self):
        """ Property for sampler creation. Used as a signal that this field is in fact synthetic. """
        return self

    # Generator creation
    def get_generator(self, locations=None, shape=None):
        """ Get a generator with data of a given `shape`.
        If called with the same parameters twice, returns the same instance: `locations` is used as a hash value.
        """
        if locations is None:
            shape = shape if shape is not None else self.default_shape

            if shape is None:
                raise ValueError('Shape is undefined: pass it directly or supply a `default_shape` at initialization!')
            locations = self.shape_to_locations(shape)

        hash_value = self.locations_to_hash(locations)

        generator = self._make_generator(hash_value)
        self._populate_generator(generator=generator, locations=locations) # Works in-place!
        return generator

    # @lru_cache
    def _make_generator(self, hash_value):
        """ Create a generator instance. During initialization, wrapped in `lru_cache`. """
        return self.GENERATOR_CONSTRUCTOR(seed=abs(hash_value))

    def _populate_generator(self, generator, locations=None):
        """ Call `generator` methods to populate it with data: impedance model, horizon surfaces, faults, etc. """
        if hasattr(generator, '_populated'):
            return None

        if self.data_generator is not None:
            self.data_generator(generator)

        else:
            # Generate parameters, use them to populate `generator` in-place
            shape = self.locations_to_shape(locations)
            params = self.param_generator(shape=shape, rng=generator.rng)
            params = Config(params)

            # Compute velocity model, using the velocity vector and horizon matrices
            (generator
             .init_shapes(**params['init_shapes'])
             .make_velocities(**params['make_velocities'])
             .make_horizons(**params['make_horizons'])
             .make_velocity_model(**params['make_velocity_model'])
             )

            # Faults
            for fault_params in params.get('make_fault_2d', []):
                generator.make_fault_2d(**fault_params)

            for fault_params in params.get('make_fault_3d', []):
                generator.make_fault_3d(**fault_params)

            # Finalize synthetic creation
            (generator
             .make_density_model(**params['make_density_model'])
             .make_impedance_model(**params['make_impedance_model'])
             .make_reflectivity_model(**params['make_reflectivity_model'])

             .make_synthetic(**params['make_synthetic'])
             .postprocess_synthetic(**params['postprocess_synthetic'])
             .cleanup(**params['cleanup'])
            )

            generator.params = params

        generator._populated = True
        return None


    # Getting data
    def get_attribute(self, locations=None, shape=None, attribute='synthetic', buffer=None, **kwargs):
        """ Output requested `attribute`.
        If `locations` is not provided, uses `shape` to create a random one.
        For the same `locations` values, uses the same generator instance (with the same velocity model):
        >>> locations = (slice(10, 11), slice(100, 200), slice(1000, 1500))
        >>> synthetic = synthetic_field.get_attribute(locations=locations, attribute='synthetic')
        >>> impedance = synthetic_field.get_attribute(locations=locations, attribute='impedance')
        """
        _ = kwargs

        generator = self.get_generator(locations=locations, shape=shape)

        # Select what is `labels`
        if attribute == 'labels':
            if self.default_attribute is not None:
                attribute = self.default_attribute
            else:
                attribute = generator.params.get('attribute')

            if attribute is None:
                raise ValueError('Attribute `labels` is undefined: use `default_attribute`'
                                 'or a key in `param_generator` to define what to retrieve!')

        # Main: velocity, reflectivity, synthetic
        if attribute in ['synthetic', 'geometry', 'image']:
            result = generator.get_attribute(attribute='synthetic')
        elif 'velocity' in attribute:
            result = generator.get_attribute(attribute='velocity_model')
        elif 'impedance' in attribute:
            result = generator.get_attribute(attribute='impedance_model')
        elif 'reflect' in attribute:
            result = generator.get_attribute(attribute='reflectivity_model')
        elif 'upward' in attribute:
            result = generator.get_increasing_impedance_model()

        # Labels: horizons and faults
        elif 'horizon' in attribute:
            # Remove extra kwargs
            _ = kwargs.pop('orientation', None)
            _ = kwargs.pop('sparse', None)

            kwargs = {
                'indices': 'all',
                'width': 3,
                'format': 'mask',
                **kwargs
            }
            result = generator.get_horizons(**kwargs)
        elif 'amplified' in attribute:
            result = generator.get_horizons(indices='amplified', format='mask', width=kwargs.get('width', 3))
        elif 'fault' in attribute:
            result = generator.get_faults(format='mask', width=kwargs.get('width', 3))

        # Fallback
        else:
            result = generator.get_attribute(attribute=attribute)

        result = result.reshape(shape)

        if buffer is not None:
            buffer[:] = result
        else:
            buffer = result
        return buffer


    def load_seismic(self, locations=None, shape=None, src='synthetic', buffer=None, **kwargs):
        """ Wrapper around `:meth:.get_attribute` to comply with `:class:.Field` API. """
        return self.get_attribute(locations=locations, shape=shape, attribute=src, buffer=buffer, **kwargs)

    def make_mask(self, locations=None, shape=None, src='labels', buffer=None, **kwargs):
        """ Wrapper around `:meth:.get_attribute` to comply with `:class:.Field` API. """
        return self.get_attribute(locations=locations, shape=shape, attribute=src, buffer=buffer, **kwargs)


    # Utilities
    def shape_to_locations(self, shape):
        """ Make a randomized locations with desired shape. """
        starts = np.random.randint((0, 0, 0), (10000, 10000, 10000))
        return tuple(slice(start, start + s)
                     for start, s in zip(starts, shape))

    def locations_to_shape(self, locations):
        """ Compute shape of a given locations. """
        return tuple(slc.stop - slc.start for slc in locations)

    def locations_to_hash(self, locations):
        """ Compute hash value of a given locations. """
        return hash(tuple((slc.start, slc.stop, slc.step) for slc in locations))


    @classmethod
    def velocity_to_seismic(cls, velocity, ricker_width=4.3):
        """ Generate synthetic seismic out of velocity predictions. """
        result = []
        for velocity_array in velocity:
            generator = cls.GENERATOR_CONSTRUCTOR()

            # Generating synthetic out of predicted velocity for all items
            generator.velocity_model = velocity_array
            generator.shape = generator.shape_padded = velocity_array.shape
            generator.depth = generator.depth_padded = velocity_array.shape[-1]

            (generator
                .make_density_model(randomization=None)
                .make_impedance_model()
                .make_reflectivity_model()
                .make_synthetic(ricker_width=ricker_width, ricker_points=100))
            result.append(generator.synthetic)

        return np.stack(result).astype(np.float32)

    # Normalization
    def make_normalization_stats(self, n=100, shape=None, attribute='synthetic'):
        """ Compute normalization stats (`mean`, `std`, `min`, `max`, quantiles) from `n` generated `attributes`. """
        data = [self.get_attribute(shape=shape, attribute=attribute) for _ in range(n)]
        data = np.array(data)

        q01, q05, q95, q99 = np.quantile(data, (0.01, 0.05, 0.95, 0.99))

        normalization_stats = {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'q_01': q01,
            'q_05': q05,
            'q_95': q95,
            'q_99': q99,
        }
        self._normalization_stats = normalization_stats
        return normalization_stats

    @property
    def normalization_stats(self):
        """ Property with default normalization stats for synthetic images. """
        if self._normalization_stats is None:
            self.make_normalization_stats()
        return self._normalization_stats

    # Visualization
    def __repr__(self):
        return f"""<SyntheticField `{self.short_name}` at {hex(id(self))}>"""

    def __str__(self):
        msg = f"SyntheticField `{self.short_name}`"

        if self.param_generator is not None:
            attribute = self.param_generator.get('attribute')
            if attribute is not None:
                msg += f':\n    - labels: attribute `{attribute}`'
        return msg

    def show_slide(self, locations=None, shape=None, **kwargs):
        """ Create one generator and show underlying models, synthetic and masks. """
        generator = self.get_generator(locations=locations, shape=shape)
        self._last_generator = generator
        return generator.show_slide(**kwargs)

    def plot_roll(self, shape=None, attribute='synthetic', n=25, **kwargs):
        """ Show attribute-images for a number of generators. """
        data = [[self.get_attribute(shape=shape, attribute=attribute)[0]] for _ in range(n)]

        # Display images
        plot_config = {
            'suptitle': f'Roll of `{attribute}`',
            'title': list(range(n)),
            'cmap': 'Greys_r',
            'colorbar': True,
            'ncols': 5,
            **kwargs
        }
        return plot(data, **plot_config)
