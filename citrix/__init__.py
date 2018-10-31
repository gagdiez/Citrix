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
