import abc
from warnings import warn

import nibabel

from nibabel.cifti2 import Cifti2MatrixIndicesMap as MatrixIndicesMap
from nibabel.cifti2 import Cifti2BrainModel as BrainModel
from nibabel.cifti2 import Cifti2Volume as Volume
from nibabel.cifti2 import Cifti2Label as Label
from nibabel.cifti2 import Cifti2LabelTable as LabelTable
from nibabel.cifti2 import Cifti2NamedMap as NamedMap
from nibabel.cifti2 import Cifti2Parcel as Parcel

def load(filename):
    cifti_file_types = {'.dconn.nii':DenseDenseConnectivity}

    for file_extension, Class in cifti_file_types.items():
        if filename.endswith(file_extension):
            nib = nibabel.load(filename)
            return Class(nib)

    warn("Citrix doesn't know how to handle this file type")
    return nibabel.load(filename)


class Cifti():
    def __init__(self, nib):

        self._nibabel = nib

        for ext in nib.header.extensions:
            if ext.get_code() == 32:
                header = ext.get_content()

        self._header = header

        self._row = header.matrix.get_index_map(0)
        self._column = header.matrix.get_index_map(1)

    @abc.abstractmethod
    def get_data(self):
        """Returns the data of the cifti file"""
        pass

    @property
    def affine(self):
        return self._nibabel.affine

    @property
    def header(self):
        return self._header

    @property
    def row(self):
        return self._row

    @property
    def column(self):
        return self._column


class DenseDenseConnectivity(Cifti):

    def get_data(self):
        return self._nibabel.get_data()[0, 0, 0, 0]
