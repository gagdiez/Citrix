import numpy as np
import nibabel

from .. import indices, models
from ..cifti import DenseLabels

def label_table(keys, labels, colors):
    """Creates a cifti label table

       Parameters
       ----------
       keys: list
           values in the cifti file that will be assigned labels and colors
       labels: list
           string that is assigned to each key
       colors: list
           color that is assigned to each label

       Returns
       -------
       labeltable: nibabel.cifti2.Cifti2LabelTable"""

    if any([not isinstance(l, list) for l in [keys, labels, colors]]):
        raise ValueError("keys, labels and colors should be of type list")

    if len(labels) != len(colors) or len(colors) != len(keys):
        raise ValueError("different number of keys, labels and colors")

    if np.array(colors).shape[1] != 4:
        raise ValueError("Colors have to be in the (R, G, B, A) format")

    cifti_labels = []

    for k, l, (r, g, b, a) in zip(keys, labels, colors):
        cifti_labels.append(nibabel.cifti2.Cifti2Label(k, l, r, g, b, a))

    label_table = nibabel.cifti2.Cifti2LabelTable()
    for cl in cifti_labels: label_table.append(cl)

    return label_table


def dlabel(data, structures, label_table, volume_shape=None, affine=np.eye(4)):
    """Builds a citrix.cifti.d dlabel cifti"""

    mip_labels = nibabel.cifti2.Cifti2MatrixIndicesMap([0], indices.LABELS)
    named_map = nibabel.cifti2.Cifti2NamedMap('labels', None, label_table)
    mip_labels.append(named_map)

    mip_brain_models = nibabel.cifti2.Cifti2MatrixIndicesMap([1], indices.BRAIN_MODELS)
    for s in structures: mip_brain_models .append(s)

    if any([s.model_type == models.VOXEL] for s in structures):
        if volume_shape is None:
            raise ValueError("A structure is of type voxel, "
                             "but no volume dimension was given")
        transform = nibabel.cifti2.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3)
        transform.matrix = affine
        mip_brain_models.volume = nibabel.cifti2.Cifti2Volume(volume_shape,
                                                              transform)

    matrix = nibabel.cifti2.Cifti2Matrix()
    matrix.append(mip_labels)
    matrix.append(mip_brain_models)

    header = nibabel.cifti2.Cifti2Header(matrix)

    return nibabel.cifti2.Cifti2Image(data[None, :], header, None)
