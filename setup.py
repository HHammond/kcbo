import os
import sys

import setuptools
from setuptools import setup, find_packages


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="kcbo",
    version="0.0.2",
    author="Henry Hammond",
    author_email="henryhhammond92@gmail.com",
    description="Bayesian data analysis library",
    license="MIT",
    keywords="bayesian data analysis statistics",
    url="https://github.com/HHammond/kcbo",
    include_package_data=True,
    packages=find_packages(),
    long_description=read('README.rst'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics"
    ],
    install_requires=[
        "numpy>=1.8",
        "scipy",
        "pymc",
        "tabulate",
    ],
    package_data={
        "kcbo": [
            "../README.md",
            "../LICENSE",
        ]
    },
)
