import nibabel
import numpy as np

from ..gifti import GiftiFunction
from .. import structures

def function(data, structure):
    '''Builds a citrix.gifti.GiftiFunction object for a given brain structure
       with the data'''
    if structure not in [structures.CORTEX_LEFT, structures.CORTEX_RIGHT]:
        raise ValueError(('structure should be structures.CORTEX_LEFT or '
                          'structures.CORTEX_RIGHT'))

    if structure == structures.CORTEX_LEFT:
        structure = 'CortexLeft'
    else:
        structure = 'CortexRight'

    darray = nibabel.gifti.GiftiDataArray(data=data.astype(np.float32),
                                          intent=11, datatype=16)
    nvpair = nibabel.gifti.GiftiNVPairs('AnatomicalStructurePrimary',
                                        structure)
    meta = nibabel.gifti.GiftiMetaData(nvpair)
    gifti_function = nibabel.gifti.GiftiImage(darrays=[darray], meta=meta)

    return GiftiFunction(gifti_function)
