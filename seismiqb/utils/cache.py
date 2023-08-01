""" Thread-safe lru cache class and cache mixin. """
import os
from copy import copy
from functools import wraps
from inspect import ismethod, signature
import json
from threading import RLock
from collections import Counter, defaultdict, OrderedDict
from weakref import WeakSet

import numpy as np
import pandas as pd


class _GlobalCache:
    """ Methods for global cache management.

    Note, this class controls only objects which use :class:`~.lru_cache`.
    So, for properties you need to use both `property` and `lru_cache` decorators for proper cache introspection.
    """
    #pylint: disable=redefined-builtin
    def __init__(self):
        """ Initialize containers with cache references and instances with cached objects.

        Note, the `cache_references` container is filled on the modules import stage."""
        self.cache_references = {} # for tests and debugging, helps to get cache info such as maxsize or stats
        self.instances_with_cache = WeakSet()

    @property
    def size(self):
        """ Total cache size. """
        return self.get_stats(stats='size', level='total')['size']

    @property
    def nbytes(self):
        """ Total cache nbytes. """
        return self.get_stats(stats='nbytes', level='total')['nbytes']

    def get_size(self, level='total', format='dict'):
        """ Get cache size grouped by level. For more read the doc for :meth:`~.get_attr`"""
        result = self.get_stats(stats='size', level=level, format=format)
        result = result['size'] if (level == 'total' and format == 'default') else result
        return result

    def get_nbytes(self, level='total', format='dict'):
        """ Get cache nbytes grouped by level. For more read the doc for :meth:`~.get_attr`"""
        result = self.get_stats(stats='nbytes', level=level, format=format)
        result = result['nbytes'] if (level == 'total' and format == 'default') else result
        return result

    def get_stats(self, stats='size', level='total', format='default'):
        """ Get cache statistics grouped by level.

        Parameters
        ----------
        stat : str or list of str
            Statistic to get values. Possible options are: 'size', 'nbytes' or both.
        level : {'total', 'class', 'instance'}
            Result groupby level.
            If 'total', then return a total stat value for all instances.
            If 'class', then return a dict with stat value for each class.
            If 'instance', then return a nested dict with stat value for each instance.
        format : {'default', 'dict', 'df'}
            Returned data format.
            If 'default', then return data as it is.
            If 'dict', then convert data to the dictionary.
            If 'df', then convert data to the pandas DataFrame.
        """
        stats = (stats, ) if isinstance(stats, str) else stats

        # Init result accumulator/container
        if level == 'total':
            result = Counter({})
        elif level == 'class':
            result = defaultdict(Counter)
        else:
            result = defaultdict(lambda: defaultdict(Counter))

        # Fill accumulator/container with attribute values from instances with cache
        for instance in self.instances_with_cache:
            instance_stats = {stat: getattr(instance, f'cache_{stat}')  for stat in stats}

            if level == 'total':
                result += instance_stats
            elif level == 'class':
                result[instance.__class__.__name__] += instance_stats
            else:
                result[instance.__class__.__name__][f'id_{id(instance)}'] += instance_stats

        # Prepare output
        if format == 'dict':
            result = json.loads(json.dumps(result, default=lambda x: x.__dict__))

        elif format == 'df':
            if level == 'total':
                result = pd.DataFrame(result, index=['total'])
            elif level == 'class':
                result = pd.DataFrame(result).T
            elif level == 'instance':
                result = pd.concat({k: pd.DataFrame(v).T for k, v in result.items()}, axis=0)

        return result

    def get_cache_repr(self, format='dict'):
        """ Create global cache representation.

        Cache representation consists of names of objects, that use data caching,
        information about cache size, nbytes, and arguments for each method.

        Keys (for 'dict') or index columns (for 'df') are: class name, instance id, method or property name.
        Values are: size, nbytes and arguments.

        Parameters
        ----------
        format : {'dict', 'df'}
            Return value format. 'df' means pandas DataFrame.
        """
        cache_repr_ = {}

        # Extract cache repr for each cached object
        for instance in self.instances_with_cache:
            instance_cache_repr = instance.get_cache_repr()

            if instance_cache_repr is not None:
                class_name = instance.__class__.__name__
                if class_name not in cache_repr_:
                    cache_repr_[class_name] = {}

                cache_repr_[class_name][f'id_{id(instance)}'] = instance_cache_repr

        # Convert to pandas dataframe
        if format == 'df' and len(cache_repr_) > 0:
            # Dataframe index columns are (class_name, instance_id, object_name), expand values for them:
            cache_repr_ = pd.DataFrame.from_dict({
                (class_name, instance_id, object_name): object_data
                    for class_name, class_data in cache_repr_.items()
                    for instance_id, instance_data in class_data.items()
                    for object_name, object_data in instance_data.items()},
            orient='index')

            cache_repr_ = cache_repr_.loc[:, ['size', 'nbytes', 'arguments']] # Columns sort

        return cache_repr_ if len(cache_repr_) > 0 else None

    @property
    def repr(self):
        """ Global cache representation. """
        df = self.get_cache_repr(format='df')
        if df is not None:
            df = df.loc[:, ['size', 'nbytes']]
        return df

    def reset(self):
        """ Clear all cache. """
        for instance in self.instances_with_cache:
            instance.reset_cache()

GlobalCache = _GlobalCache() # Global cache controller, must be the only one instance

class lru_cache:
    """ Thread-safe least recent used cache. Must be applied to a class methods.
    Adds the `use_cache` argument to the decorated method to control whether the caching logic is applied.

    Under the hood, the decorator creates `cache` attribute with dict of all cached elements.
    The `cache` keys are cached objects names and values are OrderedDict with all saved items.

    Parameters
    ----------
    maxsize : int
        Maximum amount of stored values.
    attributes: None, str or sequence of str
        Attributes to get from object and use as additions to key.
    apply_by_default : bool
        Whether the cache logic is on by default.
    copy_on_return : bool
        Whether to copy the object on retrieving from cache.

    Examples
    --------
    Store loaded slides::

    @lru_cache(maxsize=128)
    def load_slide(cube_name, slide_no):
        pass

    Specify cache size on class instantiation::
    def __init__(self, maxsize):
        self.method = lru_cache(maxsize)(self.method)

    Notes
    -----
    All arguments to a decorated method must be hashable.
    """
    #pylint: disable=invalid-name, attribute-defined-outside-init
    def __init__(self, maxsize=128, attributes=None, apply_by_default=True, copy_on_return=False):
        self.maxsize = maxsize
        self.apply_by_default = apply_by_default
        self.copy_on_return = copy_on_return
        self.func_signature = None

        # Parse `attributes`
        if isinstance(attributes, str):
            self.attributes = [attributes]
        elif isinstance(attributes, (tuple, list)):
            self.attributes = attributes
        else:
            self.attributes = False

        self.default = Singleton
        self.lock = RLock()
        self.reset()

    def reset(self, instance=None):
        """ Clear cache and stats. """
        if instance is None:
            self.stats = defaultdict(lambda: {'hit': 0, 'miss': 0})
        else:
            if hasattr(self, 'cache') and (self.cached_attr in self.cache):
                del self.cache[self.cached_attr]

            instance_hash = self.compute_hash(instance)
            self.stats[instance_hash] = {'hit': 0, 'miss': 0}

    def make_key(self, instance, func, args, kwargs):
        """ Create a key from a combination of method args and instance attributes. """
        # pylint: disable=unsupported-membership-test
        # Process args
        args = list(args) if not isinstance(args, list) else args

        if 'self' in self.func_signature:
            args = args[1:]

        args_and_defaults = [name for name in self.func_signature.keys()
                             if (name not in kwargs.keys()) and (name != 'self')]

        # Process default values
        for default_param in args_and_defaults[len(args):]:
            default_value = self.func_signature.get(default_param).default
            args.append(default_value)

        # Create key from args and defaults
        key = list(zip(args_and_defaults, args))

        # Process kwargs
        if kwargs:
            for k, v in sorted(kwargs.items()):
                if isinstance(v, slice):
                    v = (v.start, v.stop, v.step)
                key.append((k, v))

        # Process attributes
        if self.attributes:
            for attr in self.attributes:
                attr_hash = getattr(instance, attr).__hash__()
                key.append(attr_hash)
        return flatten_nested(key)

    @staticmethod
    def compute_hash(obj):
        """ Compute `obj` hash. If not provided by the object, rely on objects identity. """
        #pylint: disable=bare-except
        try:
            result = hash(obj)
        except:
            result = id(obj)
        return result

    def __call__(self, func):
        """ Add the cache to the function. """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # if a bound method, get class instance from function else from arguments
            instance = func.__self__ if self.is_method else args[0]

            use_cache = kwargs.pop('use_cache', self.apply_by_default)
            copy_on_return = kwargs.pop('copy_on_return', self.copy_on_return)

            if os.getenv('SEISMIQB_DISABLE_CACHE', ""):
                use_cache = False

            # Skip the caching logic and evaluate function directly
            if not use_cache:
                result = func(*args, **kwargs)
                return result

            # Init cache and reference on it in the GlobalCache controller
            if not hasattr(instance, 'cache'):
                # Init cache container in the instance
                setattr(instance, 'cache', defaultdict(OrderedDict))

            GlobalCache.instances_with_cache.add(instance)

            key = self.make_key(instance, func, args, kwargs)
            instance_hash = self.compute_hash(instance)

            # If result is already in cache, just retrieve it and update its timings
            instance_cache = instance.cache[self.cached_attr]
            result = instance_cache.get(key, self.default)

            if result is not self.default:
                with self.lock:
                    instance_cache.move_to_end(key)
                    self.stats[instance_hash]['hit'] += 1
                    return copy(result) if copy_on_return else result

            # The result was not found in cache: evaluate function
            result = func(*args, **kwargs)

            # Add the result to cache
            with self.lock:
                self.stats[instance_hash]['miss'] += 1

                if key in instance_cache:
                    pass
                elif len(instance_cache) >= self.maxsize:
                    instance_cache.popitem(last=False)
                    instance_cache[key] = result
                else:
                    instance_cache[key] = result

            return copy(result) if copy_on_return else result

        self.is_method = ismethod(func)
        self.cached_attr = func.__qualname__ # used as a cache key in instances
        self.func_signature = signature(func).parameters

        wrapper.__name__ = func.__name__
        wrapper.stats = lambda: self.stats
        wrapper.reset = self.reset
        wrapper.reset_instance = lambda instance: self.reset(instance=instance)

        GlobalCache.cache_references[func.__qualname__] = self
        return wrapper


class SingletonClass:
    """ There must be only one! """
Singleton = SingletonClass()

def flatten_nested(iterable):
    """ Recursively flatten nested structure of tuples, list and dicts. """
    result = []
    if isinstance(iterable, (tuple, list)):
        for item in iterable:
            result.extend(flatten_nested(item))
    elif isinstance(iterable, dict):
        for key, value in sorted(iterable.items()):
            result.extend((*flatten_nested(key), *flatten_nested(value)))
    else:
        return (iterable,)
    return tuple(result)


class CacheMixin:
    """ Methods for cache management.

    You can use this mixin for cache introspection and cached data cleaning on instance level.
    """
    def get_cache_size(self, name=None):
        """ Get cache size for specified objects.

        Parameters
        ----------
        name: str, optional
            Attribute name. If None, then get total cache size.
        """
        cached_values = self.get_cached_values(name)
        return len(cached_values)

    def get_cache_nbytes(self, name=None):
        """ Get cache nbytes for specified objects.

        Parameters
        ----------
        name: str, optional
            Attribute name. If None, then get total cache nbytes.
        """
        cache_nbytes_accumulator = 0
        cached_values = self.get_cached_values(name)

        # Accumulate nbytes over all cached objects: each term is a nbytes of cached numpy array
        for value in cached_values:
            if isinstance(value, np.ndarray):
                cache_nbytes_accumulator += value.nbytes #/ (1024 ** 3)

        return cache_nbytes_accumulator

    def get_cached_values(self, name=None):
        """  Get cache values for specified objects. """
        cached_values = []
        if hasattr(self, 'cache'):
            names = (name,) if name is not None else self.cache.keys()

            for cached_attr in names:
                cached_values.extend(self.cache[cached_attr].values())
        return cached_values

    @property
    def cache_size(self):
        """ Total amount of cached objects. """
        return self.get_cache_size()

    @property
    def cache_nbytes(self):
        """ Total nbytes of cached objects. """
        return self.get_cache_nbytes()

    def _get_object_cache_repr(self, name):
        """ Make object's cache repr. """
        object_cache_size = self.get_cache_size(name=name)

        if object_cache_size == 0:
            return None

        object_cache_nbytes = self.get_cache_nbytes(name=name)

        cached_data = getattr(self, 'cache', {}).get(name, {})

        # The class saves cache for the same method with different arguments values
        # Get them all in a desired format: list of dicts
        all_arguments = []
        for arguments in cached_data.keys():
            arguments = dict(zip(arguments[::2], arguments[1::2])) # tuple ('name', value, ...) to dict
            all_arguments.append(arguments)

        # Expand extra scopes
        if len(all_arguments) == 1:
            all_arguments = all_arguments[0]

        object_cache_repr = {
            'size': object_cache_size,
            'nbytes': object_cache_nbytes,
            'arguments': all_arguments
        }

        return object_cache_repr

    def get_cache_repr(self, format='dict'):
        """  Create instance cache representation.

        Cache representation consists of names of objects that use data caching,
        information about cache size, nbytes, and arguments for each method.

        Parameters
        ----------
        format : {'dict', 'df'}
            Return value format. 'df' means pandas DataFrame.
        """
        #pylint: disable=redefined-builtin
        cache_repr_ = {}

        # Create cache representation for each object
        for name in self.cache.keys():
            object_cache_repr = self._get_object_cache_repr(name=name)

            if object_cache_repr is not None:
                cache_repr_[name] = object_cache_repr

        # Convert to pandas dataframe
        if format == 'df' and len(cache_repr_) > 0:
            cache_repr_ = pd.DataFrame.from_dict(cache_repr_, orient='index')
            cache_repr_ = cache_repr_.loc[:, ['size', 'nbytes', 'arguments']] # Columns sort

        return cache_repr_ if len(cache_repr_) > 0 else None

    @property
    def cache_repr(self):
        """ DataFrame with cache representation that contains names, cache_size
        and cache_nbytes for each cached object.
        """
        df = self.get_cache_repr(format='df')
        if df is not None:
            df = df.loc[:, ['size', 'nbytes']]
        return df

    def reset_cache(self, name=None):
        """ Clear cached data.

        Parameters
        ----------
        name: str, optional
            Attribute name. If None, then clean cache of all cached objects.
        """
        if hasattr(self, 'cache'):
            if name is not None:
                del self.cache[name]
            else:
                self.cache.clear()
