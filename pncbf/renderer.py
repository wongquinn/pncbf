"""
Real-time animation using matplotlib in notebooks. Use with %matplotlib inline.
"""

import matplotlib.pyplot as plt


class NotebookRenderer:
    def __init__(
        self,
        title=None,
        xlabel=None,
        ylabel=None,
        xlim=[-1.5, 1.5],
        ylim=None,
        hide_axis=False,
        enable_grid=False,
        enable_legend=False,
    ):
        if ylim is None:
            ylim = xlim

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.hide_axis = hide_axis
        self.enable_grid = enable_grid
        self.enable_legend = enable_legend

        from IPython.display import clear_output

        self.clear_output = clear_output

        self.plot_dict = {}
        self.patch_dict = {}

    def set_data(self, id, x, y, fmt="o", alpha=1, label=None):
        self.plot_dict[id] = (x, y, fmt, alpha, label)

    def set_patch(self, id, artist):
        self.patch_dict[id] = artist

    def draw(self):
        self.clear_output(wait=True)

        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.gca().set_aspect("equal", adjustable="box")

        for id in self.plot_dict:
            x, y, fmt, alpha, label = self.plot_dict[id]
            plt.plot(x, y, fmt, alpha=alpha, label=label)

        for id in self.patch_dict:
            artist = self.patch_dict[id]
            plt.gca().add_patch(artist)

        if self.enable_legend:
            plt.legend()

        plt.show()
