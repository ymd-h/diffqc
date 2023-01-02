from setuptools import setup, find_packages


setup(name="pennylane-jaxqubit",
      version="0.0.0",
      packages=find_packages(),
      install_requires=["pennylane", "jax", "jaxlib"],
      entry_points = {
          "pennylane.plugins": [
              "jax.qubit = pennylane_jaxqubit:JaxQubitDevice",
          ]
      })
