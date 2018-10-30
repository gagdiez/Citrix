''' Setup file '''
from os import path
from distutils import extension

from setuptools import setup
import numpy

package_name = 'citrix'
cli_module = package_name + '.cli'
utils_module = package_name + '.utils'

setup(name=package_name,
      version='0.1',
      description='Tools to manipulate cifti files',
      url='http://github.com/gagdiez/citrix',
      author='Gallardo Diez, Guillermo Alejandro',
      author_email='guillermo-gallardo.diez@inria.fr',
      include_package_data=True,
      packages=[package_name, cli_module],#, utils_module],
      scripts=['scripts/ct_cifti_average'],
      zip_safe=False)
