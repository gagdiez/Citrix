import nibabel

def is_cifti(filename):
    """Checks if the file is a cifti file"""
    # check it has a cifti header
    header = nibabel.load(filename).header
    codes = [e.get_code() for e in header.extensions]
    return 32 in codes

def retrieve_direction(cifti, direction):
    """Returns either the row or column of the cifti matrix"""
    if direction not in ["ROW", "COLUMN"]:
        raise ValueError("direction should be ROW or COLUMN")

    if direction == "ROW":
        return cifti.header.row
    return cifti.header.column

def brain_models_from_direction(cifti, direction):
    """Returns the brain models of a given direction"""
    return retrieve_direction(cifti, direction).brain_models

def volume_from_direction(cifti, direction):
    """Returns the volume information of a given direction"""
    return retrieve_direction(cifti, direction).volume
