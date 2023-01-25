# Citrix - Cifti Tricks

Suite of tools to easily manipulate CIFTI files in Python. CIFTI files is a special type of format used in the Human Connectome Project to store neuroimaging data.
Particularly, this package wraps different cifti-related nibabel functions to ease the process of creating different kind of files such as .dconn, .dlabel, etc.

So far the package includes two useful command line tools: ctrx_dlabel_to_nifti and ctrx_dtseries_to_nifti, allowing to transform between surface-based data and
their volumetric counterpart.

## Install
Clone this repository in your computer, then execute:

```bash
pip install . -e
```
