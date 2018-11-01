import nibabel
import nimesh

from . import utils, surface, cifti

def load(filename):
    """Loading function that can handle gifti, nifti and cifti files"""
    if filename.endswith('surf.gii'):
        # CIFTI SURFACE
        return surface.load(filename)
    elif filename.endswith('gii'):
        # GIFTI SURFACE
        return nimesh.io.load(filename)
    elif filename.endswith('nii'):
        if utils.is_cifti(filename):
            return cifti.load(filename)
        else:
            return nibabel.load(filename)
    else:
        return nibabel.load(filename)

def save(filename, data, header=None, affine=None, version=2):
    ''' Simple wrapper around nibabel.save '''
    if version == 1:
        nif_image = nibabel.Nifti1Image(data, affine, header)
    else:
        nif_image = nibabel.Nifti2Image(data, affine, header)
    nibabel.save(nif_image, filename)
