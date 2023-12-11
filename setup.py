from setuptools import find_packages, setup

install_requires = [
    'numpy',
    'scipy',
    'tqdm',
    'jax>=0.4, <0.5',
    'flax',
    'jraph',
    'pyscf',
    'optax',
    'sympy',
    'matplotlib',
    'uncertainties',
    'seml>=0.3',
    'seml_logger @ git+https://github.com/n-gao/seml_logger.git@c1c57611af882fe65f61ebd123ba7228b0fec423#egg=seml_logger',
    'aim>=3.0, <4.0',
    'sacred>=0.8.4, <0.9',
    'matplotlib'
]


setup(name='pesnet',
      version='0.1.0',
      description='Ab-Initio Potential Energy Surfaces via Graph Neural Networks',
      packages=find_packages('.'),
      install_requires=install_requires,
      zip_safe=False)
