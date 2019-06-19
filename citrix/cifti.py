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
    cifti_file_types = {'.dconn.nii': DenseDenseConnectivity,
                        '.dtseries.nii': DenseTimeSeries,
                        '.dscalar.nii': DenseScalar,
                        '.dlabel.nii': DenseLabels}

    for file_extension, Class in cifti_file_types.items():
        if filename.endswith(file_extension):
            nib = nibabel.load(filename)
            return Class.from_nibabel(nib)

    warn("Citrix doesn't know how to handle this file type")
    return nibabel.load(filename)


class Cifti(nibabel.Cifti2Image):

    @property
    def row(self):
        return self.header.matrix.get_index_map(0)

    @property
    def column(self):
        return self.header.matrix.get_index_map(1)

    @classmethod
    def from_nibabel(klass, nib):

        if isinstance(nib, nibabel.Cifti2Image):
            cifti_header = nib.header
            nifti_header = nib.nifti_header
        else:
            for ext in nib.header.extensions:
                if ext.get_code() == 32:
                    cifti_header = ext.get_content()
            nifti_header = nib.header

        return klass(nib.dataobj, cifti_header, nifti_header)


class DenseDenseConnectivity(Cifti):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        shape = self.dataobj.shape
        if len(shape) > 2:
            self._dataobj = self.dataobj.reshape(shape[-2:])

class DenseTimeSeries(Cifti):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shape = self.dataobj.shape
        if len(shape) > 1:
            self._dataobj = self.dataobj.reshape([shape[-1]])

class DenseLabels(Cifti):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shape = self.dataobj.shape
        if shape[0] == 1:
            self._dataobj = self.dataobj.reshape([shape[-1]])

class DenseScalar(Cifti):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shape = self.dataobj.shape
        if shape[0] ==  1:
            self._dataobj = self.dataobj.reshape([shape[-1]])
