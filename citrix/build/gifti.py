import nibabel
import numpy as np

from ..gifti import GiftiFunction, GiftiMesh
from .. import structures

def check_structure(structure):
    if structure not in [structures.CORTEX_LEFT, structures.CORTEX_RIGHT]:
        raise ValueError(('structure should be structures.CORTEX_LEFT or '
                          'structures.CORTEX_RIGHT'))

def common_meta(structure):
    if structure == structures.CORTEX_LEFT:
        structure = 'CortexLeft'
    else:
        structure = 'CortexRight'

    nvpair = nibabel.gifti.GiftiNVPairs('AnatomicalStructurePrimary',
                                        structure)
    meta = nibabel.gifti.GiftiMetaData(nvpair)
    return meta

def function(data, cifti_structure):
    '''Builds a citrix.gifti.GiftiFunction object for a given brain structure
       with the data'''
    check_structure(cifti_structure)

    meta = common_meta(cifti_structure)

    darray = nibabel.gifti.GiftiDataArray(data=data.astype(np.float32),
                                          intent=11, datatype=16)

    return GiftiFunction(darrays=[darray], meta=meta)

def mesh(vertices, triangles, cifti_structure):
    '''Builds a mesh'''

    meta = common_meta(cifti_structure)

    d0 = nibabel.gifti.GiftiDataArray(data=vertices.astype(np.float32),
                                      intent=1008, datatype=16, meta=meta)
    d1 = nibabel.gifti.GiftiDataArray(data=triangles.astype(np.int32),
                                      intent=1009, datatype=8)

    return GiftiMesh(darrays=[d0, d1], meta=meta)
