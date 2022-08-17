import numpy
from setuptools import setup
from setuptools import Extension
from sysconfig import get_paths

if __name__ == "__main__":
    setup(
        name="puan",
        version="0.4.0",
        description = "Function tools for combinatorial optimization",
        author = "Our Studio Void AB",
        author_email = "moa@ourstudio.se",
        install_requires=[
            "numpy>=1.22.3",
            "more-itertools",
            "maz>=0.0.1"
        ],
        packages=[
            'puan',
            'puan.logic',
            'puan.logic.sta',
            'puan.logic.plog',
            'puan.misc',
            'puan.ndarray',
            'puan.modules',
            'puan.modules.configurator',
        ],
        url = "https://puan.io",
        long_description = "",
        ext_modules=[
            Extension(
                'puan/npufunc',
                ['puan/ndarray/npufunc.c'],
                include_dirs=[
                    numpy.get_include(),
                    get_paths()['include']
                ]
            ),
            Extension(
                'puan/logic/logicfunc',
                ['puan/logic/plog/logicfunc.c'],
                include_dirs=[
                    get_paths()['include']
                ]
            ),
        ]
    )