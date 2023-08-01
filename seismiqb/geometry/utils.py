""" Useful functions, related to geometry. """


def time_to_sample(time, geometry):
    """ Convert time (in ms) into geometry sample value. """
    return round((time - geometry.delay) / geometry.sample_interval)

def sample_to_time(sample, geometry):
    """ Convert geometry sample value into time. """
    return round(sample * geometry.sample_interval + geometry.delay)
