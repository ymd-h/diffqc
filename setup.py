from setuptools import setup, find_packages


setup(name="diffq",
      description="Diiferentiable Quantum Simulator",
      version="0.0.0",
      packages=find_packages(),
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
