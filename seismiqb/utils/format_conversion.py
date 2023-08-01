""" Utils for format conversion. """

import os
import h5py
import hdf5plugin

from batchflow import Notifier

def repack_hdf5(path, dst_path=None, projections = (0, ), transform=None, dtype='float32', pbar='t', inplace=False,
                **dataset_kwargs):
    """ Recreate hdf5-file with transformation and type conversion. """
    dataset_kwargs = dataset_kwargs or dict(hdf5plugin.Blosc(cname='lz4hc', clevel=6, shuffle=0))
    transform = transform or (lambda array: array)

    with h5py.File(path, mode='r') as src_file:
        from ..geometry.conversion_mixin import ConversionMixin #pylint: disable=import-outside-toplevel

        total = sum(src_file[ConversionMixin.PROJECTION_NAMES[axis]].shape[0] for axis in projections)
        dst_path = dst_path or os.path.splitext(path)[0] + '_tmp.hdf5'
        with h5py.File(dst_path, mode='w-') as dst_file:
            with Notifier(pbar, total=total) as progress_bar:
                for axis in projections:
                    projection_name = ConversionMixin.PROJECTION_NAMES[axis]
                    data = src_file[projection_name]
                    projection_shape = data.shape

                    dataset_kwargs_ = {'chunks': (1, *projection_shape[1:]), **dataset_kwargs}
                    projection = dst_file.create_dataset(projection_name, shape=projection_shape, dtype=dtype,
                                                         **dataset_kwargs_)

                    for i in range(data.shape[0]):
                        projection[i] = transform(data[i])
                        progress_bar.update()
    if inplace:
        os.remove(path)
        os.rename(dst_path, path)
