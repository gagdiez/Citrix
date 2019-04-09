import nibabel

from . import utils, cifti, gifti, build

def load(filename):
    """Loading function that can handle gifti, nifti and cifti files"""
    if filename.endswith('gii'):
        return gifti.load(filename)
    elif filename.endswith('nii') or filename.endswith('nii.gz'):
        if utils.is_cifti(filename):
            return cifti.load(filename)
        else:
            return nibabel.load(filename)
    else:
        raise ValueError("We can only load NIFTI (nii) or GIFTI (gii) files")

def save(filename, data, header=None, affine=None, version=2):
    ''' Simple wrapper around nibabel.save for CIFTI/NIFTI files '''
    if version == 1:
        nif_image = nibabel.Nifti1Image(data, affine, header)
    else:
        nif_image = nibabel.Nifti2Image(data, affine, header)
    nibabel.save(nif_image, filename)
