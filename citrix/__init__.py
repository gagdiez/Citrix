import utils

def load(filename):
    """Loading function that can handle gifti, nifti and cifti files"""
    if filename.endswith('surf.gii'):
        # CIFTI SURFACE
        return citrix.surface.load(filename)
    elif filename.endswith('gii'):
        # GIFTI SURFACE
        return nimesh.load(filename)
    elif filename.endswith('nii'):
        if utils.is_cifti(filename):
            return citrix.cifti.load(filename)
        else:
            return nibabel.load(filename)
    else:
        return nibabel.load(filename)
