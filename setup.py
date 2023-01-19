import os
from setuptools import setup, find_packages


desc = {}
if os.path.exists("README.md"):
    with open("README.md") as f:
        readme = f.read()

    desc = {
        "long_description": readme,
        "long_description_content_type": "text/markdown",
    }

setup(name="diffqc",
      description="Diiferentiable Quantum Simulator",
      **desc,
      author="H. Yamada",
      version="0.0.2",
      url = "https://github.com/ymd-h/diffqc",
      project_urls = {
          "Bug Report & QA": "https://github.com/ymd-h/diffqc/discussions",
      },
      packages=find_packages(),
      classifiers = [
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python :: 3",
          "Topic :: Scientific/Engineering",
      ],
      install_requires=["jax", "jaxlib"],
      extras_require = {
          "pennylane": ["pennylane"],
          "test": ["coverage", "unittest-xml-reporting", "numpy"],
          "example": ["flax", "optax", "scikit-learn", "tqdm"],
      },
      entry_points = {
          "pennylane.plugins": [
              "diffqc.qubit = diffqc.pennylane:diffqcQubitDevice",
          ]
      })
