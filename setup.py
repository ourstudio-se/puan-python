from setuptools import setup
from setuptools import Extension

if __name__ == "__main__":
    setup(
        name="puan",
        version="0.3.2",
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
        ],
        url = "https://puan.io",
        long_description = "",
        ext_modules=[
            Extension(
                'puan/npufunc',
                ['puan/ndarray/npufunc.c'],
                include_dirs=[
                    "/usr/local/Cellar/python@3.9/3.9.12/Frameworks/Python.framework/Versions/3.9/include/python3.9",
                    "/usr/local/Cellar/numpy/1.22.3_1/lib/python3.9/site-packages/numpy/core/include/",
                ]
            ),
            Extension(
                'puan/logicfunc',
                ['puan/logic/plog/logicfunc.c'],
                include_dirs=[
                    "/usr/local/Cellar/python@3.9/3.9.12/Frameworks/Python.framework/Versions/3.9/include/python3.9",
                    "/usr/local/Cellar/numpy/1.22.3_1/lib/python3.9/site-packages/numpy/core/include/",
                ]
            ),
        ]
    )