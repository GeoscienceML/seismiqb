""" Interactive field viewer. """
from ipywidgets import widgets
from IPython.display import display

import matplotlib.pyplot as plt



class FieldViewer:
    """ Interactive viewer of a field.

    Creates a figure with two axis -- base map and slice map, as well as some additional controls.
    On the base map we show a top-view image of a field with requested attribute from a geometry / one of the labels.
    On the slice map we show one slice from a cube along desired axis.

    Requires the `ipympl` library and the `%matplotlib widget` magic to work.
    """
    out = widgets.Output()

    def __init__(self, field, figsize=(8, 8)):
        # ax[0] is a `base_map` / `base_ax`
        # ax[1] is a `slice_img` / `slice_ax`

        # Attributes
        self.field = field

        # Initial state
        location = self.field.shape[0] // 2
        self.state_base = {'attribute': 'geometry/snr'}
        self._state_base = {} # previous `state_base`

        self.state_slice = {'location': location, 'axis': 0}
        self._state_slice = {} # previous `state_slice`

        # Make widgets
        self.attribute_dropdown = widgets.Dropdown(options=self.field.available_attributes,
                                                   value=self.field.available_attributes[2],
                                                   description='Base map',
                                                   layout=widgets.Layout(max_width='350px'))
        self.location_text = widgets.IntText(value=location, description='INLINE',
                                             layout=widgets.Layout(max_width='150px'))
        self.location_slider = widgets.IntSlider(value=location, max=self.field.shape[0] - 1, description='INLINE',
                                                 continuous_update=False, layout=widgets.Layout(min_width='500px'))
        self.axis_button = widgets.Button(description='swap axis', layout=widgets.Layout(min_width='200px'))

        self.hbox = widgets.HBox([self.attribute_dropdown, self.location_text, self.location_slider, self.axis_button],
                                 layout=widgets.Layout(min_width='1000px'))

        # Make figure
        with widgets.Output():
            plt.ioff()
            self.fig, self.ax = plt.subplots(1, 2, figsize=figsize, tight_layout=True)
            self.base_ax, self.slice_ax = self.ax
            self.vbox = widgets.VBox([self.hbox,
                                      self.fig.canvas])

        # Setup widgets
        self.attribute_dropdown.observe(self.attribute_dropdown_update, names='value')
        self.location_text.observe(self.location_text_update, names='value')
        self.location_slider.observe(self.location_slider_update, names='value')
        self.axis_button.on_click(self.axis_button_onclick)

        self.fig.canvas.mpl_connect('button_press_event', self.fig_onclick)
        self.fig.canvas.header_visible = False

        # Initialize plots
        with widgets.Output():
            self.draw_base(force=True)
            self.draw_slice(force=True)

        display(self.vbox)


    # Draw methods
    def draw_base(self, force=False):
        """ Draw base top-view map on the left axis. """
        # Cache
        if force is False and self.state_base == self._state_base:
            return

        attribute = self.state_base['attribute']

        # Remove previous image, draw new one. TODO: change to `set_data`
        if self.base_ax.get_images():
            self.base_ax.get_images()[0].remove()
            self.base_ax.created_colorbar.ax.remove()
        self.field.show(attributes=attribute, ax=self.base_ax, labelright=False, colorbar_fraction=1.0)

        # Update previous state
        self._state_base = {**self.state_base}

    def draw_slice(self, force=False):
        """ Draw slice from a field on the right axis. """
        # Cache
        if force is False and self.state_slice == self._state_slice:
            return

        location, axis = self.state_slice['location'], self.state_slice['axis']

        # Remove all previous lines
        lines = [children for children in self.base_ax.get_children()
                if hasattr(children, 'created_by_draw') or hasattr(children, 'created_by_zoom')]
        for line in lines:
            line.remove()

        # Add a line; mark it with attribute
        if axis == 0:
            line = self.base_ax.vlines(location, 0, self.field.shape[1], color='r', linewidth=3)
        elif axis == 1:
            line = self.base_ax.hlines(location, 0, self.field.shape[0], color='r', linewidth=3)
        line.created_by_draw = True

        # Remove previous image, draw new one. TODO: change to `set_data`
        if self.slice_ax.get_images():
            self.slice_ax.clear()
            self.slice_ax.created_colorbar.ax.remove()
        self.field.show_slide(location, axis=axis, ax=self.slice_ax)

        # Update previous state
        self._state_slice = {**self.state_slice}

    def zoom_line(self, point):
        """ Highlight selected (on the right axis) region on the top-view map (left axis). """
        lines = [children for children in self.base_ax.get_children()
                if hasattr(children, 'created_by_zoom')]
        for line in lines:
            line.remove()

        location, axis = self.state_slice['location'], self.state_slice['axis']
        if axis == 0:
            start = max(point - 100, 0)
            stop  = min(point + 100, self.field.shape[1] - 1)
            line = self.base_ax.vlines(location, start, stop, color='w', linewidth=5)
        elif axis == 1:
            start = max(point - 100, 0)
            stop  = min(point + 100, self.field.shape[0] - 1)
            line = self.base_ax.hlines(location, start, stop, color='w', linewidth=5)
        line.created_by_zoom = True

        # For some reason, we need to manually update the figure (not a full redraw though)
        self.fig.canvas.draw_idle()


    # State reactions
    def refresh(self, force=False):
        """ Re-draw both base and slice axis, if needed, and update state of widgets. """
        self.draw_base(force=force)
        self.draw_slice(force=force)

        self.location_text.value = self.state_slice['location']
        self.location_slider.value = self.state_slice['location']

    @out.capture()
    def change_axis(self):
        """ Swap axis and update state of widgets. """
        self.state_slice['axis'] = 1 - self.state_slice['axis']
        self.state_slice['location'] = self.field.shape[self.state_slice['axis']] // 2
        self.refresh()

        if self.state_slice['axis'] == 0:
            self.location_text.description = 'INLINE'
            self.location_slider.description = 'INLINE'
            self.location_slider.max = self.field.shape[0] - 1
            self.axis_button.description = 'change axis to crossline'

        elif self.state_slice['axis'] == 1:
            self.location_text.description = 'CROSSLINE'
            self.location_slider.description = 'CROSSLINE'
            self.location_slider.max = self.field.shape[1] - 1
            self.axis_button.description = 'change axis to inline'


    # Event reactions
    @out.capture()
    def attribute_dropdown_update(self, change):
        """ Select base map attribute by a dropdown. """
        self.state_base['attribute'] = change['new']
        self.refresh()

    @out.capture()
    def location_text_update(self, change):
        """ Change location by entering a number. """
        self.state_slice['location'] = change['new']
        self.refresh()

    @out.capture()
    def location_slider_update(self, change):
        """ Change location by moving a slider. """
        self.state_slice['location'] = change['new']
        self.refresh()

    @out.capture()
    def axis_button_onclick(self, button):
        """ Change axis button. """
        self.change_axis()

    @out.capture()
    def fig_onclick(self, event):
        """ If clicked on the left axis, then change location.
        If clicked on the right axis, then select zoom region.
        """
        if event.xdata is not None and event.ydata is not None:
            point = int(event.xdata + 0.5), int(event.ydata + 0.5)

            if event.inaxes == self.base_ax:
                self.state_slice['location'] = point[self.state_slice['axis']]
                self.refresh()

            elif event.inaxes == self.slice_ax:
                self.zoom_line(point[0])
