import os.path
from stlmeshing.stl import Stl

test_stl = Stl(os.path.join(os.path.dirname(__file__), "stl_mesh_test.stl"))

print(f"stl_mesh_test surface area: {test_stl.surface_area():.2f}")
