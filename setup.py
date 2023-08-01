""" seismiQB is a framework for deep learning research on 3d-cubes of seismic data. """

from setuptools import setup, find_packages
import re

with open('__init__.py', 'r') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)


extras = [
    'psutil>=5.6.7',
    'bottleneck>=1.3',
    'numexpr>=2.7',
]

extras_nn = [
    'torch>=1.7.0',
    'torchvision>=0.1.3',
    'cupy>=8.1.0',
]

extras_cupy = [
    'cupy>=8.1.0',
]

extras_vis = [
    'plotly>=4.3.0',
    'ipython>=7.10.0',
    'ipywidgets>=7.0',
]

extras_test = [
    'py-nbtools[nbrun]>=0.9.8',
    'ipywidgets>=7.0',
    'plotly>=4.3.0',
    'pytest>=5.3.1',
]


setup(
    name='seismiQB',
    packages=find_packages(exclude=['tutorials']),
    version=version,
    url='https://github.com/gazprom-neft/seismiqb',
    license='Apache 2.0',
    author='Gazprom Neft DS team',
    author_email='rhudor@gmail.com',
    description='An ML framework for research on volumetric seismic data',
    long_description='',
    zip_safe=False,
    platforms='any',
    install_requires=[
        # General Python libraries
        'dill>=0.3.1.1',
        'tqdm>=4.50.0',

        # Numerical
        'numpy>=1.16.0',
        'numba>=0.43.0',
        'scipy>=1.3.3',
        'scikit-learn>=0.21.3',
        'scikit_image>=0.16.2',
        'connected-components-3d>=3.10.2',

        # Data manipulation
        'pandas>=1.0.0',
        'segyio>=1.8.3',
        'lasio>=0.29',
        'h5py>=2.10.0',
        'h5pickle>=0.2.0',
        'hdf5plugin>=3.3.0',
        'blosc>=1.11',

        # Working with images
        'opencv_python>=4.1.2.30',
        'matplotlib>=3.0.2',

        # Our libraries
        'batchflow>=0.8.0',
    ],
    extras_require={
        'extra': extras,
        'nn': extras_nn,
        'cupy': extras_cupy,
        'test': extras_test,
        'vis': extras_vis,
        'dev': extras + extras_nn + extras_test + extras_vis,
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
    ],
)
