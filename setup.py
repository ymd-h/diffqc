import os
from setuptools import setup, find_packages


desc = {}
if os.path.exists("README.md"):
    desc = {
        "long_description": open("README.md").read(),
        "long_description_content_type": "text/markdown",
    }

setup(name="diffq",
      description="Diiferentiable Quantum Simulator",
      **desc,
      version="0.0.0",
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
              "diffq.qubit = diffq.pennylane:JaxQubitDevice",
          ]
      })
