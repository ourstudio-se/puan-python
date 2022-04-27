from setuptools import setup
from setuptools import Extension

if __name__ == "__main__":
    setup(
        name="puan",
        version="0.1",
        description = "Function tools for combinatorial optimization",
        author = "Our Studio Void AB",
        author_email = "moa@ourstudio.se",
        install_requires=[
            "numpy",
            "more-itertools",
            "scipy",
            "maz"
        ],
        packages=['puan'],
        url = "https://puan.io",
        long_description = "",
        # ext_modules=[
        #     Extension(
        #         'puancore/ufunc',
        #         ['puancore/uFuncs/optimized_bit_allocation_64.c'],
        #         include_dirs=[
        #             '/opt/homebrew/Frameworks/Python.framework/Headers', 
        #             '/opt/homebrew/Cellar/numpy/1.22.3_1/lib/python3.9/site-packages/numpy/core/include/'
        #         ]
        #     )
        # ]
    )