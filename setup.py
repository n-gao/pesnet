from setuptools import find_packages, setup

install_requires = [
    'numpy',
    'seml',
    'scipy',
    'tqdm',
    'flax',
    'jraph',
    'pyscf<2.0',
    'h5py==3.1.0',
    'optax==0.0.9',
    'sympy',
    'matplotlib',
    'jaxboard',
    'uncertainties'
]


setup(name='pesnet',
      version='0.1.0',
      description='Potential Energy Surface Network',
      packages=find_packages('.'),
      install_requires=install_requires,
      zip_safe=False)
