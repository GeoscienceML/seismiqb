""" Dict-like disk storage, based on HDF5 with compression. """
import os

import numpy as np
import pandas as pd

import h5py
import hdf5plugin



class SQBStorage:
    """ Dict-like disk storage, based on HDF5 with compression.
    Unlike native HDF5, works with sequences, dataframes and np.arrays with dtype object.

    Under the hood, we don't store an open file descriptor and re-open the file in read/write mode at each operation.
    """
    def __init__(self, path, cname='lz4hc', clevel=6, shuffle=0):
        self.path = path
        self.loaded_items = []

        self.dataset_parameters = hdf5plugin.Blosc(cname=cname, clevel=clevel, shuffle=shuffle)

        # Scan all available keys and store them in the instance for faster check
        self.keys = set()
        if self.exists:
            with h5py.File(self.path, mode='r', swmr=True) as src:
                for key in src:
                    self.keys.add(key)

    @property
    def exists(self):
        """ True, if the storage exists on disk and it is HDF5 file. """
        return os.path.exists(self.path) and h5py.is_hdf5(self.path)

    @staticmethod
    def is_storage(path):
        """ True if `path` exists on disk and it is HDF5 file. """
        return os.path.exists(path) and h5py.is_hdf5(path)

    def file_handler(self, mode='r', swmr=True):
        """ A convenient file handler for using in with-statements. """
        return h5py.File(self.path, mode=mode, swmr=swmr)


    # Store
    def store(self, items, overwrite=True):
        """ Store items from dict-like iterator.

        Parameters
        ----------
        items : dict or iterable
            Dictionary or iterable with key-value pairs.
        overwrite : bool
            Whether to overwrite keys in a storage.
        """
        iterator = items.items() if isinstance(items, dict) else items
        for key, value in iterator:
            self.store_item(key=key, value=value, overwrite=overwrite)
    update = store

    def store_item(self, key, value, overwrite=True):
        """ Save one `value` as `key`.
        Unlike native `h5py`, works with sequences, dataframes and arrays with `object` dtype.
        """
        key = (key + '/').replace('//', '/')

        with h5py.File(self.path, mode='a') as dst:
            if overwrite and key in dst:
                del dst[key]

            # Sequence: store length and type separately, then dump each item to its own group
            if isinstance(value, (tuple, list)) or (isinstance(value, np.ndarray) and value.dtype == object):
                dst[key + 'is_sequence'] = 1
                dst[key + 'length'] = len(value)

                types = {tuple: 0, list: 1, np.ndarray: 2}
                type_ = types[type(value)]
                dst[key + 'type'] = type_

                for i, v in enumerate(value):
                    self.store_item(key=key+str(i), value=v)

            # Dictionary: store keys / values separately
            # TODO: current implementation works only with numeric keys and values
            elif isinstance(value, dict):
                dst[key + 'is_dict'] = 1

                key_, value_ = next(iter(value.items()))
                dst[key + 'keys'] = np.fromiter(value.keys(), dtype=np.array(key_).dtype)
                dst[key + 'values'] = np.fromiter(value.values(), dtype=np.array(value_).dtype)

            # Dataframe: store column/index names and values separately
            # TODO: would not work correctly with arbitrary index. Can be improved by dumping index values directly
            elif isinstance(value, pd.DataFrame):
                dst[key + 'is_dataframe'] = 1
                dst.attrs[key + 'columns'] = list(value.columns)

                index_names = list(value.index.names)
                if index_names[0]:
                    dst.attrs[key + 'index_names'] = index_names
                    values_ = value.reset_index().values
                else:
                    values_ = value.values
                self.store_item(key=key+'values', value=values_)

            # String: use ASCII encoding with removed zero bytes
            elif isinstance(value, str):
                dst[key+'is_str'] = 1
                dst[key+'encoded'] = value.encode('ascii').replace(b'\x00', b' ')

            # None: store as flag
            elif value is None:
                dst[key+'is_none'] = 1

            # Numpy array with numerical dtype: compress for efficiency
            elif isinstance(value, np.ndarray):
                dst.create_dataset(key.strip('/'), data=value, **self.dataset_parameters)

            # Fallback for native types: int, float, etc
            else:
                dst[key] = value

            self.keys.add(key)

    def __setitem__(self, key, value):
        self.store_item(key=key, value=value, overwrite=True)


    # Read
    def read(self, keys):
        """ Read keys from a storage. """
        result = {}
        for key in keys:
            result[key] = self.read_item(key)
        return result

    def read_item(self, key):
        """ Read one `key` from storage.
        Unlike native `h5py`, works with sequences, dataframes and arrays with `object` dtype.
        """
        key = (key + '/').replace('//', '/')

        with h5py.File(self.path, mode='r', swmr=True) as src:
            # Sequence: read each element, reconstruct into original type
            if key + 'is_sequence' in src:
                length = src[key + 'length'][()]
                type_ = src[key + 'type'][()]

                value = [self.read_item(key=key + str(i)) for i in range(length)]

                types = {0: tuple, 1: list, 2: np.array}
                value = types[type_](value)

            # Dictionary: read keys/values separately, zip
            elif key + 'is_dict' in src:
                keys = src[key + 'keys'][()]
                values = src[key + 'values'][()]

                value = dict(zip(keys, values))

            # Dataframe: read columns and index separately, impose on values
            elif key + 'is_dataframe' in src:
                values = src[key + 'values'][()]
                columns = src.attrs[key + 'columns']

                value = pd.DataFrame(data=values, columns=columns)

                if key + 'index_names' in src:
                    index_names = src.attrs[key + 'index_names']
                    value.set_index(index_names, inplace=True)

            # String: decode back from ASCII
            elif key + 'is_str' in src:
                value = src[key+'encoded'][()].decode('ascii')

            # None
            elif key + 'is_none' in src:
                value = None

            # Fallback for Numpy arrays and native types
            elif key in src:
                value = src[key][()]

            else:
                raise KeyError(f'Key `{key}` is not in storage!')

        self.loaded_items.append(key)
        return value

    def __getitem__(self, key):
        return self.read_item(key=key)

    def get(self, key, default=None):
        """ Get item, if present in storage, or return `default` otherwise. """
        return self.read_item(key) if self.has_item(key) else default

    # Check for item(s)
    def has_items(self, keys):
        """ Check if all of `keys` are present. """
        if self.exists is False:
            return False
        for key in keys:
            if self.has_item(key) is False:
                return False
        return True

    def has_item(self, key):
        """ Check if `key` is in storage. """
        if self.exists is False:
            return False
        return key in self.keys

    def __contains__(self, key):
        return self.has_item(key=key)

    # Reset
    def reset(self, keys):
        """ Delete `keys` from storage. """
        with h5py.File(self.path, mode='a') as dst:
            for key in keys:
                if key in dst:
                    del dst[key]

    def remove(self):
        """ Remove storage file entirely. """
        os.remove(self.path)


    # Introspection
    def print_tree(self):
        """ Print textual representation of storage. """
        if self.exists:
            with h5py.File(self.path, mode='r', swmr=True) as src:
                src.visititems(self._print_tree)

    def _print_tree(self, name, node):
        """ Print one storage node. """
        if isinstance(node, h5py.Dataset):
            shift = name.count('/') * ' ' * 4
            item_name = name.split('/')[-1]
            shape_ = f'shape={node.shape}' if node.shape != tuple() else 'scalar'
            print(f'{shift}{item_name}: {shape_}, dtype={node.dtype}')
        if isinstance(node, h5py.Group):
            print(name)
