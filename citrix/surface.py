import nibabel
import nimesh

from nimesh import CoordinateSystem

from . import models, structures

def load(filename):
    '''Loads a cifti surface file'''
    surface = nimesh.io.load(filename)
    str_xml = str(nibabel.load(filename).to_xml())

    structure = None
    if 'CortexLeft' in str_xml:
        structure = structures.CORTEX_LEFT
    elif 'CortexRight' in str_xml:
        structure = structures.CORTEX_RIGHT

    cifti_surface = CiftiMesh(structure, surface.vertices, surface.triangles,
                              surface.coordinate_system, surface.normals)
    return cifti_surface

class CiftiMesh(nimesh.Mesh):
    def __init__(self, structure, vertices, triangles,
                 coordinate_system=CoordinateSystem.UNKNOWN, normals=None):

        if structure not in [structures.CORTEX_LEFT, structures.CORTEX_RIGHT]:
            raise ValueError(('structure should be structures.CORTEX_LEFT or '
                              'structures.CORTEX_RIGHT'))

        super().__init__(vertices, triangles, coordinate_system, normals)
        self._model = models.SURFACE
        self._structure = structure

    @property
    def model(self):
        return self._model

    @property
    def structure(self):
        return self._structure
