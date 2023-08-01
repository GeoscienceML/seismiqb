""" 3D surface plotting. """
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
try:
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from matplotlib.cm import get_cmap



def show_3d(x, y, z, simplices, title, zoom, colors=None, show_axes=True, aspect_ratio=(1, 1, 1),
            axis_labels=None, width=1200, height=1200, margin=(0, 0, 20), savepath=None,
            images=None, bounds=False, resize_factor=2, colorscale='Greys', show=True, camera=None, **kwargs):
    """ Interactive 3D plot for some elements of cube.

    Parameters
    ----------
    x, y, z : numpy.ndarrays
        Triangle vertices.
    simplices : numpy.ndarray
        (N, 3) array where each row represent triangle. Elements of row are indices of points
        that are vertices of triangle.
    title : str
        Title of plot.
    zoom : tuple of slices
        Crop from cube to show.
    colors : list or None
        List of colors for each simplex.
    show_axes : bool
        Whether to show axes and their labels.
    aspect_ratio : tuple of floats.
        Aspect ratio for each axis.
    axis_labels : tuple
        Titel for each axis.
    width, height : number
        Size of the image.
    margin : tuple of ints
        Added margin for each axis, by default, (0, 0, 20).
    savepath : str
        Path to save interactive html to.
    images : list of tuples
        Each tuple is triplet of image, location and axis to load slide from seismic cube.
    bounds : bool or int
        Whether to draw bounds on slides. If int, width of the border.
    resize_factor : float
        Resize factor for seismic slides. Is needed to spedify loading and ploting of seismic slices.
    colorscale : str
        Colormap for seismic slides.
    show : bool
        Whether to show figure.
    camera : dict
        Parameters for initial camera view.
    kwargs : dict
        Other arguments of plot creation.
    """
    #pylint: disable=too-many-arguments
    if not PLOTLY_AVAILABLE:
        raise ImportError('Install `plotly` to use 3d interactive viewer!')

    # Arguments of graph creation
    kwargs = {
        'title': title,
        'colormap': [plt.get_cmap('Depths')(x) for x in np.linspace(0, 1, 10)],
        'edges_color': 'rgb(70, 40, 50)',
        'show_colorbar': False,
        'width': width,
        'height': height,
        'aspectratio': {'x': aspect_ratio[0], 'y': aspect_ratio[1], 'z': aspect_ratio[2]},
        **kwargs
    }
    cmin, cmax = kwargs.pop('cmin', None), kwargs.pop('cmax', None)
    if isinstance(colorscale, str) and colorscale in plt.colormaps():
        cmap = get_cmap(colorscale)
        levels = np.arange(0, 256, 1) / 255
        colorscale = [
            (level, f'rgb({r * 255}, {g * 255}, {b * 255})')
            for (r, g, b, _), level in zip(cmap(levels), levels)
        ]

    if simplices is not None:
        if colors is not None:
            fig = ff.create_trisurf(x=x, y=y, z=z, color_func=colors, simplices=simplices, **kwargs)
        else:
            fig = ff.create_trisurf(x=x, y=y, z=z, simplices=simplices, **kwargs)
    else:
        fig = go.Figure()
    if images is not None:
        for image, loc, axis in images:
            shape = image.shape
            if resize_factor != 1:
                image = cv2.resize(image.astype('float32'), tuple(np.array(shape)[::-1] // resize_factor))
            image = image[::-1]
            if bounds:
                bounds = int(bounds)
                fill = cmax if cmax is not None else image.max()
                image[:bounds, :] = fill
                image[-bounds:, :] = fill
                image[:, :bounds] = fill
                image[:, -bounds:] = fill

            grid = np.meshgrid(
                np.linspace(0, shape[0], image.shape[0]),
                np.linspace(0, shape[1], image.shape[1])
            )
            if axis == 0:
                x, y, z = loc * np.ones_like(image), grid[0].T + zoom[1].start, grid[1].T + zoom[2].start
            elif axis == 1:
                y, x, z = loc * np.ones_like(image), grid[0].T + zoom[0].start, grid[1].T + zoom[2].start
            else:
                z, x, y = loc * np.ones_like(image), grid[0].T + zoom[0].start, grid[1].T + zoom[1].start

            fig.add_surface(x=x, y=y, z=z, surfacecolor=np.flipud(image), cmin=cmin, cmax=cmax,
                            showscale=False, colorscale=colorscale)
    # Update scene with title, labels and axes
    fig.update_layout(
        {
            'width': kwargs['width'],
            'height': kwargs['height'],
            'scene': {
                'xaxis': {
                    'title': axis_labels[0] if show_axes else '',
                    'showticklabels': show_axes,
                    'range': [zoom[0].stop + margin[0], zoom[0].start - margin[0]]
                },
                'yaxis': {
                    'title': axis_labels[1] if show_axes else '',
                    'showticklabels': show_axes,
                    'range': [zoom[1].start - margin[1], zoom[1].stop + margin[1]]
                },
                'zaxis': {
                    'title': axis_labels[2] if show_axes else '',
                    'showticklabels': show_axes,
                    'range': [zoom[2].stop + margin[2], zoom[2].start - margin[2]]
                },
                'camera': camera or {'eye': {"x": 1.25, "y": 1.5, "z": 1.5}},
            }
        }
    )
    if show:
        fig.show()

    if isinstance(savepath, str):
        ext = os.path.splitext(savepath)[1][1:]
        if ext == 'html':
            fig.write_html(savepath)
        elif ext in ['png', 'jpg', 'jpeg', 'pdf']:
            fig.write_image(savepath, format=ext)
