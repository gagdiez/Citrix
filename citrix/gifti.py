from warnings import warn

import numpy as np

import nibabel
import nimesh

from . import models, structures

def load(filename):
    gifti_file_types = {'.surf.gii': GiftiMesh,
                        '.func.gii': GiftiFunction}

    for file_extension, Class in gifti_file_types.items():
        if filename.endswith(file_extension):
            nib = nibabel.load(filename)
            return Class.from_nibabel(nib)

    warn("Citrix doesn't know how to handle this file type")
    return nibabel.load(filename)


class Gifti(nibabel.gifti.GiftiImage):

    @property
    def model_type(self):
        return models.SURFACE

    @property
    def brain_structure(self):
        str_xml = str(self.to_xml())

        structure = None
        if 'CortexLeft' in str_xml:
            structure = structures.CORTEX_LEFT
        elif 'CortexRight' in str_xml:
            structure = structures.CORTEX_RIGHT

        return structure

    def save(self, filename):
        self.to_filename(filename)

    @classmethod
    def from_nibabel(klass, nib):
        return klass(nib.header, nib.extra, nib.file_map, nib.meta,
                      nib.labeltable, nib.darrays, nib.version)

class GiftiFunction(Gifti):

    @property
    def function_data(self):
        return self.darrays[0].data

    def save(self, filename):
        self.to_filename(filename)


class GiftiMesh(Gifti):

    @property
    def vertices(self):
        return self.darrays[0].data

    @property
    def triangles(self):
        return self.darrays[1].data
