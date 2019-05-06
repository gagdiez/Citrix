import nibabel
import numpy as np
import os

from .. import models, load, save

def check_input(infile, outfile, reference_file, surface_files):

    if not (infile.endswith('.dlabel.nii')
            or infile.endswith('.dlabel.nii.gz')):
        raise ValueError("dlabel file should end with 'dlabel.nii' or 'dlabel.nii.gz'")

    if not (reference_file.endswith('.nii') or reference_file.endswith('.nii.gz')):
        raise ValueError("reference_file should end with '.nii' or '.nii.gz'")

    if not (outfile.endswith('.nii') or outfile.endswith('.nii.gz')):
        raise ValueError("outfile should end with '.nii' or '.nii.gz'")

    if surface_files:
        for sfile in surface_files:
            if not os.path.exists(sfile):
                raise ValueError('The file {} does not exist'.format(sfile))


def dlabel_to_nifti(dlabel_file, outfile,
                    reference_file=None, surface_files=None):
    """Transforms a dlabel file into a nifti file"""
    check_input(dlabel_file, outfile, reference_file, surface_files)
    # load time series
    dlabel = load(dlabel_file)
    labels = dlabel.get_data()

    # load surfaces if present
    if surface_files is not None:
        surfaces = [load(sf) for sf in surface_files]

    # get information about volume and create it
    if dlabel.column.volume is not None:
        volume = dlabel.column.volume
        shape = volume.volume_dimensions
        affine = volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix
    elif reference_file is not None:
        volume = load(reference_file)
        shape = volume.shape
        affine = volume.affine
    else:
        raise ValueError("The dlabel has no volume information, and no "
                         "reference volume was given")

    nifti = np.zeros(shape)

    for bm in dlabel.column.brain_models:
        if bm.model_type == models.SURFACE:
            if surface_files is None:
                raise ValueError(('There are surface models in the dlabel,'
                                  'but no surface was given as input'))
            idx_vertices = np.array(bm.vertex_indices)

            for s in surfaces:
                if s.brain_structure == bm.brain_structure:
                    surf = s

            vertices = surf.vertices[idx_vertices]
            voxels = nibabel.affines.apply_affine(np.linalg.inv(affine),
                                                  vertices)

            off, cnt = bm.index_offset, bm.index_count

            for o, (x, y, z) in enumerate(voxels):
                neighbors = [(x+i, y+j, z+k) for i in range(-1, 1)
                                             for j in range(-1, 1)
                                             for k in range(-1, 1)]

                neighbors = np.round(neighbors).astype(int)

                nifti[tuple(np.transpose(neighbors))] = labels[off+o]
        else:
            voxels = np.array(bm.voxel_indices_ijk).astype(int)
            off, cnt = bm.index_offset, bm.index_count
            nifti[tuple(np.transpose(voxels))] = labels[off:off+cnt]

    save(outfile, nifti, None, affine, version=2)
