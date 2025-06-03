import os.path
import numpy as np
import matplotlib.pyplot as plt

from stlmeshing.stl import STL
from stlmeshing.points_inside import points_inside

test_stl = STL(os.path.join(os.path.dirname(__file__), "stl_mesh_test.stl"))

print(f"stl_mesh_test surface area: {test_stl.surface_area():.2f}")
print(f"stl_mesh_test volume: {test_stl.volume():.4e}")

((x_min, x_max), (y_min, y_max), (z_min, z_max)) = test_stl.bounding_box()
print("Bounding box:")
print(f"\tx: {x_min} - {x_max}")
print(f"\ty: {y_min} - {y_max}")
print(f"\tz: {z_min} - {z_max}")

# Create a test mesh based on the bounding box
point_spacing = 1.7
border_size = 2
npoints_x = (
    int(np.rint(((x_max - x_min) / point_spacing) + (2 * border_size)).item()) + 1
)
npoints_y = (
    int(np.rint(((y_max - y_min) / point_spacing) + (2 * border_size)).item()) + 1
)
npoints_z = (
    int(np.rint(((z_max - z_min) / point_spacing) + (2 * border_size)).item()) + 1
)

x_points = (x_min - (border_size * point_spacing)) + (
    np.arange(npoints_x) * point_spacing
)
y_points = (y_min - (border_size * point_spacing)) + (
    np.arange(npoints_y) * point_spacing
)
z_points = (z_min - (border_size * point_spacing)) + (
    np.arange(npoints_z) * point_spacing
)

mesh_x, mesh_y, mesh_z = np.meshgrid(x_points, y_points, z_points, indexing="ij")

print("Computing which points are inside the mesh...")
inside = points_inside(test_stl, mesh_x, mesh_y, mesh_z)
print("... Done computing which points are inside the mesh")


# See https://matplotlib.org/stable/gallery/event_handling/image_slices_viewer.html
# for how the slicing plot works (use your scroll wheel!!)
class XYSliceIndexTracker:
    def __init__(self, ax, X):
        self.index = 0
        self.X = X
        self.ax = ax
        self.im = ax.imshow(self.X[:, :, self.index], vmin=0, vmax=np.max(X))
        self.update()

    def on_scroll(self, event):
        increment = 1 if event.button == "up" else -1
        max_index = self.X.shape[2] - 1
        self.index = np.clip(self.index + increment, 0, max_index)
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.index])
        self.ax.set_title(f"XY Slice\nIndex {self.index}")
        self.im.axes.figure.canvas.draw()


class XZSliceIndexTracker:
    def __init__(self, ax, X):
        self.index = 0
        self.X = X
        self.ax = ax
        self.im = ax.imshow(
            self.X[:, self.index, :],
            vmin=0,
            vmax=np.max(X),
        )
        self.update()

    def on_scroll(self, event):
        increment = 1 if event.button == "up" else -1
        max_index = self.X.shape[1] - 1
        self.index = np.clip(self.index + increment, 0, max_index)
        self.update()

    def update(self):
        self.im.set_data(self.X[:, self.index, :])
        self.ax.set_title(f"XZ Slice\nIndex {self.index}")
        self.im.axes.figure.canvas.draw()


class YZSliceIndexTracker:
    def __init__(self, ax, X):
        self.index = 0
        self.X = X
        self.ax = ax
        self.im = ax.imshow(self.X[self.index, :, :], vmin=0, vmax=np.max(X))
        self.update()

    def on_scroll(self, event):
        increment = 1 if event.button == "up" else -1
        max_index = self.X.shape[0] - 1
        self.index = np.clip(self.index + increment, 0, max_index)
        self.update()

    def update(self):
        self.im.set_data(self.X[self.index, :, :])
        self.ax.set_title(f"YZ Slice\nIndex {self.index}")
        self.im.axes.figure.canvas.draw()


# Simple slices through the volume for initial testing
fig, ax = plt.subplots()
xy_tracker = XYSliceIndexTracker(ax, inside)
fig.canvas.mpl_connect("scroll_event", xy_tracker.on_scroll)

fig, ax = plt.subplots()
xz_tracker = XZSliceIndexTracker(ax, inside)
fig.canvas.mpl_connect("scroll_event", xz_tracker.on_scroll)

fig, ax = plt.subplots()
yz_tracker = YZSliceIndexTracker(ax, inside)
fig.canvas.mpl_connect("scroll_event", yz_tracker.on_scroll)

plt.show()
