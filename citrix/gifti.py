from warnings import warn

import nibabel

def load(filename):
    '''Loads a cifti surface file'''

    gifti_file_types = {'.func.gii': GiftiFunction}

    for file_extension, Class in gifti_file_types.items():
        if filename.endswith(file_extension):
            nib = nibabel.load(filename)
            return Class(nib)

    warn("Citrix doesn't know how to handle this file type")
    return nibabel.load(filename)

class Gifti(nibabel.gifti.GiftiImage):
    def __init__(self, nib):

        super().__init__(nib.header, nib.extra, nib.file_map, nib.meta,
                         nib.labeltable, nib.darrays, nib.version)

class GiftiFunction(Gifti):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def function_data(self):
        return self.darrays[0].data
