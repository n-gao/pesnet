from setuptools import find_packages, setup

install_requires = [
    'numpy',
    'scipy',
    'tqdm',
    'jax',
    'flax',
    'jraph',
    'pyscf',
    'optax',
    'sympy',
    'matplotlib',
    'uncertainties',
    'seml',
    'seml_logger @ git+https://github.com/n-gao/seml_logger.git#egg=seml_logger',
    'sacred'
]


setup(name='pesnet',
      version='0.1.0',
      description='Ab-Initio Potential Energy Surfaces via Graph Neural Networks',
      packages=find_packages('.'),
      install_requires=install_requires,
      zip_safe=False)
