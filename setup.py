import numpy
from setuptools import setup
from setuptools import Extension
from sysconfig import get_paths

if __name__ == "__main__":
    setup(
        ext_modules=[
            Extension(
                'puan/npufunc',
                ['puan/ndarray/npufunc.c'],
                include_dirs=[
                    numpy.get_include(),
                    get_paths()['include']
                ]
            ),
        ]
    )