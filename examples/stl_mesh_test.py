import os.path
from stlmeshing.stl import STL

test_stl = STL(os.path.join(os.path.dirname(__file__), "stl_mesh_test.stl"))

print(f"stl_mesh_test surface area: {test_stl.surface_area():.2f}")
print(f"stl_mesh_test volume: {test_stl.volume():.4e}")

bbox = test_stl.bounding_box()
print("Bounding box:")
print(f"\tx: {bbox[0][0]} - {bbox[0][1]}")
print(f"\ty: {bbox[1][0]} - {bbox[1][1]}")
print(f"\tz: {bbox[2][0]} - {bbox[2][1]}")
