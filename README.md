<div align="center">

# seismiQB

<a href="#installation">Installation</a> •
<a href="#getting-started">Getting Started</a> •
<a href="#citing-seismicpro">Citation</a>


[![License](https://img.shields.io/github/license/analysiscenter/batchflow.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-green.svg)](https://pytorch.org)
[![Status](https://github.com/GeoscienceML/seismiqb/actions/workflows/status.yml/badge.svg?branch=master&event=push)](https://github.com/GeoscienceML/seismiqb/actions/workflows/status.yml)
[![Test installation](https://github.com/GeoscienceML/seismiqb/actions/workflows/test-install.yml/badge.svg?branch=master&event=push)](https://github.com/GeoscienceML/seismiqb/actions/workflows/test-install.yml)
</div>

---


**seismiQB** is a framework for research and deployment of deep learning models on post-stack seismic data.
It covers all main stages of model development and its production usage for seismic interpretation. The main features are:

* Optimized IO for `SEG-Y` storage: by using our library **[segfast](https://github.com/analysiscenter/segfast)**, combined with quantization and compression, we are working with the most popular transport format by an order of magnitude faster;
* Labelling classes, matching core interpretation entities: horizons, faults, facies, wells, and more;
* Pipelines for preparing the model inputs, e.g. patches (2d or 3d) of seismic data and corresponding segmentation masks;
* Geologic transformations, as well as more traditional ML augmentations;
* Advanced primitives for train/inference to saturate even the fastest GPUs;
* Fast and convenient export: all predicted objects are easy to convert to popular formats (SEG-Y, CHARISMA, LAS) for validation by geophysicists;



## Installation
**seismiQB** is compatible with Python 3.8+ and well tested on Ubuntu 20.04.

    # pipenv
    pipenv install git+https://github.com/GeoscienceML/seismiqb.git#egg=seismiqb

    # pip / pip3
    pip3 install git+https://github.com/GeoscienceML/seismiqb.git

    # developer version (add `--depth 1` if needed)
    git clone https://github.com/GeoscienceML/seismiqb.git


## Getting started

After installation just import **seismiQB** into your code. A quick demo of our primitives and methods:
```python
import seismiqb

field = Field('/path/to/cube.sgy')                                    # Initialize field with SEG-Y
field.load_labels('path/to/horizons/*.char', labels_class='horizon')  # Add labeling

# Labels
field.horizons.interpolate()                                          # Fill in small holes
field.horizons.smooth_out()                                           # Smooth out and remove spikes
field.horizons.evaluate()                                             # Compute a quality control metric

# Visualizations
field.geometry.print()                                                # Display key stats about SEG-Y
field.show_slide(index=100, axis=1)                                   # Show 100-th crossline
field.show('horizons:0/metric')                                       # Show QC metric for one horizon

```

Be sure to check out our [tutorials](tutorials) to get more info about the **seismiQB** primitives and usage.



## Citing

Please cite **seismiQB** in your publications if it helps your research.

    Khudorozhkov R., Tsimfer S., Kozhevin A., Koryagin A., Sorokina S., Strievich E. SeismiQB library for seismic interpretation with deep learning. 2023.

```
@misc{seismiQB_2023,
  author       = {R. Khudorozhkov and S. Tsimfer and A. Kozhevin and A. Koryagin and S. Sorokina and E. Strievich},
  title        = {SeismiQB library for seismic interpretation with deep learning},
  year         = 2023
}
```
