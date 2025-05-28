import struct
import numpy as np
import tqdm


class Stl:
    # Constants relating the STL file structure
    HEADER_SIZE = 80  # Number of bytes in the header
    NUM_TRI_SIZE = 4  # Number of bytes for the number of triangles
    TRIANGLE_SIZE = 50  # Number of bytes for each triangle
    NUM_TRI_FORMAT = "<I"
    TRIANGLE_FORMAT = "<12fh"

    def __init__(self, filename=None):
        """Create an STL object, and optionally initialize it from an STL file."""
        # Number of triangles in the surface
        self.n_triangles = 0

        # The triangles is an array of shape (n_triangles, 3, 3). Each triangle
        # is defined by 3 vertices, with each vertex having (x, y, z) coordinates.
        #     Triangle definition : | V1_x, V1_y, V1_z |
        #                           | V2_x, V2_y, V2_z |
        #                           | V3_x, V3_y, V3_z |
        self.triangles = np.zeros()

        # The normals is an array of shape (n_triangles, 3) containing a vector
        # normal to each triangle. Last dimension is ordered (x, y, z).
        self.normals = np.zeros()

        if filename is not None:
            self.load_file(filename)

    def load_file(self, filename: str):
        """Load a triangulated surface from the STL file specified.

        STL files contain no units, so the coordinates are assumed to be in the
        desired units. This function uses the STL file definition found on
        Wikipedia here: https://en.wikipedia.org/wiki/STL_(file_format).

        Args:
            filename (str): Filename of the input STL file (should end with .stl)
        """
        tri_struct = struct.Struct(Stl.TRIANGLE_FORMAT)

        with open(filename, "rb") as stl_file:
            # Read the header and discard
            stl_file.read(Stl.HEADER_SIZE)

            # Read the field containing the number of triangles
            n_tri_bytes = stl_file.read(Stl.NUM_TRI_SIZE)
            n_triangles = struct.unpack(Stl.NUM_TRI_FORMAT, n_tri_bytes)[0]

            # Allocate the output data structures
            self.triangles = np.zeros((n_triangles, 3, 3))
            self.normals = np.zeros((n_triangles, 3))

            for i in tqdm.trange(n_triangles, desc="Reading STL file:"):
                fields = tri_struct.unpack(
                    Stl.TRIANGLE_FORMAT, stl_file.read(Stl.TRIANGLE_SIZE)
                )
                self.normals[i, 0] = fields[0]
                self.normals[i, 1] = fields[1]
                self.normals[i, 2] = fields[2]
                self.triangles[i, 0, 0] = fields[3]
                self.triangles[i, 0, 1] = fields[4]
                self.triangles[i, 0, 2] = fields[5]
                self.triangles[i, 1, 0] = fields[6]
                self.triangles[i, 1, 1] = fields[7]
                self.triangles[i, 1, 2] = fields[8]
                self.triangles[i, 2, 0] = fields[9]
                self.triangles[i, 2, 1] = fields[10]
                self.triangles[i, 2, 2] = fields[11]

    def surface_area(self):
        """Compute the surface area."""
        pass

    def volume(self):
        """ """
        pass
