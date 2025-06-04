import numpy as np
import matplotlib.pyplot as plt


def plot_interactive_3d(data3d: np.ndarray):
    """Plot slices through 3D data with scroll wheel controls."""
    fig, ax = plt.subplots()
    xy_tracker = XYSliceIndexTracker(ax, data3d)
    fig.canvas.mpl_connect("scroll_event", xy_tracker.on_scroll)

    fig, ax = plt.subplots()
    xz_tracker = XZSliceIndexTracker(ax, data3d)
    fig.canvas.mpl_connect("scroll_event", xz_tracker.on_scroll)

    fig, ax = plt.subplots()
    yz_tracker = YZSliceIndexTracker(ax, data3d)
    fig.canvas.mpl_connect("scroll_event", yz_tracker.on_scroll)
    plt.show()


# See https://matplotlib.org/stable/gallery/event_handling/image_slices_viewer.html
# for how the slicing plot works (use your scroll wheel!!)
class XYSliceIndexTracker:
    def __init__(self, ax, data3d):
        self.index = 0
        self.data3d = data3d
        self.ax = ax
        self.im = ax.imshow(self.data3d[:, :, self.index], vmin=0, vmax=np.max(data3d))
        self.update()

    def on_scroll(self, event):
        increment = 1 if event.button == "up" else -1
        max_index = self.data3d.shape[2] - 1
        self.index = np.clip(self.index + increment, 0, max_index)
        self.update()

    def update(self):
        self.im.set_data(self.data3d[:, :, self.index])
        self.ax.set_title(f"XY Slice (scroll wheel)\nIndex {self.index}")
        self.im.axes.figure.canvas.draw()


class XZSliceIndexTracker:
    def __init__(self, ax, data3d):
        self.index = 0
        self.data3d = data3d
        self.ax = ax
        self.im = ax.imshow(self.data3d[:, self.index, :], vmin=0, vmax=np.max(data3d))
        self.update()

    def on_scroll(self, event):
        increment = 1 if event.button == "up" else -1
        max_index = self.data3d.shape[1] - 1
        self.index = np.clip(self.index + increment, 0, max_index)
        self.update()

    def update(self):
        self.im.set_data(self.data3d[:, self.index, :])
        self.ax.set_title(f"XZ Slice (scroll wheel)\nIndex {self.index}")
        self.im.axes.figure.canvas.draw()


class YZSliceIndexTracker:
    def __init__(self, ax, data3d):
        self.index = 0
        self.data3d = data3d
        self.ax = ax
        self.im = ax.imshow(self.data3d[self.index, :, :], vmin=0, vmax=np.max(data3d))
        self.update()

    def on_scroll(self, event):
        increment = 1 if event.button == "up" else -1
        max_index = self.data3d.shape[0] - 1
        self.index = np.clip(self.index + increment, 0, max_index)
        self.update()

    def update(self):
        self.im.set_data(self.data3d[self.index, :, :])
        self.ax.set_title(f"YZ Slice (scroll wheel)\nIndex {self.index}")
        self.im.axes.figure.canvas.draw()
