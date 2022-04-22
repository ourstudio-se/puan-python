# Puan
A package for combinatorial optimization tools.

### Install
```
pip install puan
```

### Usage
```python
>>> import puan
>>> gp = puan.ge_polytope([
    [ 0,-2, 1, 1],
    [ 0, 1, 1, 1]
])
>>> _gp = gp.reduce(*gp.reducable_rows_and_columns())
>>> _gp
ge_polytope([[ 0, -2,  1,  1]])
```