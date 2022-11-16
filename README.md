[![Documentation Status](https://readthedocs.org/projects/puan/badge/?version=latest)](https://puan.readthedocs.io/en/latest/?badge=latest)

![](https://github.com/ourstudio-se/puan-python/blob/main/header.png)

<h3 align="center">A combinatorial optimization python package.</h3>

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

