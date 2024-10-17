"""Installation script for the 'isaacgymenvs' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os

root_dir = os.path.dirname(os.path.realpath(__file__))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "gymnasium==0.29.1", 
    "mujoco>=3.0.0",
]

# Installation operation
setup(
    name="gym_dcmm",
    author="Yuanhang Zhang",
    version="0.0.1",
    description="Mujoco Environment for Dexterous Catch with Mobile Manipulation (DCMM).",
    keywords=["robotics", "rl"],
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.8"],
    zip_safe=False,
)

# EOF