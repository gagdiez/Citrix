import unittest

import citrix
import nibabel

class TestLoad(unittest.TestCase):

    def test_load(self):
        surface = citrix.load('./citrix/test/data/very_inflated.surf.gii')
        self.assertEqual(type(surface), citrix.surface.CiftiMesh)

        nifti = citrix.load('./citrix/test/data/test.nii')
        self.assertEqual(type(nifti), nibabel.Nifti1Image)

        cifti = citrix.load('./citrix/test/data/merge3.dconn.nii')
        self.assertEqual(type(cifti), citrix.cifti.DenseDenseConnectivity)
