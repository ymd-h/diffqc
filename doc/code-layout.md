# Source Code Layout


## 1. Package Souce
Acutual package source coudes are located under `diffqc` directory.


### 1.1 `diffqc` package (`diffqc/__init__.py` module)
`diffqc/__init__.py` is a top and entry point module. We import
some sub-modules there for convenience.

```{warning}
Optional module like `diffqc/pennylane.py` should not be imported,
otherwise all users must install such optional dependencies.
```


### 1.2 Core Operation Modules

Quantum circuit core operations are implemented at `diffqc/dense.py`
and `diffqc/sparse.py`. Both modules have same functionalities with
deffetent internal implementation.

To reduce duplicated code, common matrix representation of circuit
operators are implemented at `diffqc/_operators.py` and used by
`diffqc/dense.py` and `diffqc/sparse.py`.

We might add new operation module with different implementation, at
that moment we should implement (almost) same functionalities.

### 1.3 High Level API Modules

Builtin algorithms, neural network modules, and utility functions are
implemented at `diffqc/lib.py`, `diffqc/nn.py`, and `diffqc/util.py`,
respectively.

To support multiple core operation modules, these high level API
functions take core operation module as an argument.


### 1.4 PennyLane Plugin

PennyLane plugin is implemented at `diffqc/pennylane.py`.


## 2. Development Source

### 2.1 Test Code

Unit test codes are placed at `test` directory. Tests are implemented
with Python standard `unittest` module.


`.coveragerc` is a configuration file for
[Coverage.py](https://coverage.readthedocs.io/).

### 2.2 CI

CI configurations are located at `.github/workflows/`.

`.github/workflows/diffqc.yaml` specifies unit test, document site
building, package wheel building, and wheel uploading.

`.github/workflows/codeql.yaml` specifies
[CodeQL](https://codeql.github.com/) security scan.


In `Dockerfile-ci`, actual unit test, page build, and package build
are executed.
`.dockerignore` is used to suppress copy unnecessary files.

## 3. Documents

Documents are placed under `doc` directory. Each markdown (`doc/*.md`)
is a single page.

`doc/conf.py` is a configuration file for
[Sphinx](https://www.sphinx-doc.org/en/master/index.html).

API reference is generated from docstring.


## 4. Example

Example files are located under `example` directory. These are also
included from document site.


## 5. Repository Files

`README.md` and `LICENSE` describe repository wide information, and
they are included package. We also utilize `.gitignore` to avoid
tracking unnecessary files and auto generated files.
