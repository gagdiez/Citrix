import nibabel
import numpy as np
import os

from .. import models, load, save

def check_input(infile, outfile, surface_files):

    if not (infile.endswith('.dtseries.nii')
            or infile.endswith('.dtseries.nii.gz')):
        raise ValueError("dtseries file should end with 'dtseries.nii' or 'dtseries.nii.gz'")

    if not (outfile.endswith('.nii') or outfile.endswith('.nii.gz')):
        raise ValueError("outfile should end with '.nii' or '.nii.gz'")

    if surface_files:
        for sfile in surface_files:
            if not os.path.exists(sfile):
                raise ValueError('The file {} does not exist'.format(sfile))


def dtseries_to_nifti(dtseries_file, outfile, surface_files=None):
    """Transforms a dtseries file into a nifti file"""
    check_input(dtseries_file, outfile, surface_files)
    # load time series
    dtseries = load(dtseries_file)
    time_series = dtseries.get_data()

    # load surfaces if present
    if surface_files is not None:
        surfaces = [load(sf) for sf in surface_files]

    # get information about volume and create it
    volume = dtseries.column.volume
    shape = volume.volume_dimensions
    affine = volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix

    nifti = np.zeros(shape)

    for bm in dtseries.column.brain_models:
        if bm.model_type == models.SURFACE:
            if surface_files is None:
                raise ValueError(('There are surface models in the dtseries,'
                                  'but no surface was given as input'))
            idx_vertices = np.array(bm.vertex_indices)

            for s in surfaces:
                if s.brain_structure == bm.brain_structure:
                    surf = s

            vertices = surf.vertices[idx_vertices]
            voxels = nibabel.affines.apply_affine(np.linalg.inv(affine),
                                                  vertices)
            voxels = np.floor(voxels).astype(int)
        else:
            voxels = bm.voxel_indices_ijk

        off, cnt = bm.index_offset, bm.index_count
        nifti[tuple(np.transpose(voxels))] = time_series[off:off+cnt]

    save(outfile, nifti, None, affine, version=2)
