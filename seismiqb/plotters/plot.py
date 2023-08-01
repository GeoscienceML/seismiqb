""" Plotter with redefined defaults. """
import batchflow



class plot(batchflow.plotter.plot):
    """ Wrapper over original `plot` with custom defaults. """
    COMMON_DEFAULTS = {
        **batchflow.plotter.plot.COMMON_DEFAULTS,
        'suptitle_size': 30,
    }

    IMAGE_DEFAULTS = {
        **batchflow.plotter.plot.IMAGE_DEFAULTS,
        'labeltop': True,
        'labelright': True,
        'xlabel_size': 22,
        'ylabel_size': 22,
        'transpose': (1, 0, 2)
    }
