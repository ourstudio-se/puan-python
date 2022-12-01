
<h1 align="center">
<img src="https://github.com/ourstudio-se/puan-python/blob/main/puan-logo.svg" width="350">
</h1>

<h4 align="center">A combinatorial optimization python package.</h4>

[![Documentation Status](https://readthedocs.org/projects/puan/badge/?version=latest)](https://puan.readthedocs.io/en/latest/?badge=latest)
[![Tested with Hypothesis](https://img.shields.io/badge/hypothesis-tested-brightgreen.svg)](https://hypothesis.readthedocs.io/)
### Install
```
pip install puan
```

### Usage
Given a predefined matrix, you construct a polyhedron as you would with a numpy array. Reduce its rows by following
```python
>>> import puan
>>> polyhedron = puan.ge_polyhedron([
    [ 0,-2, 1, 1],
    [ 0, 1, 1, 1]
])
>>> polyhedron.reduce(*polyhedron.reducable_rows_and_columns())
ge_polytope([[ 0, -2,  1,  1]])
```

