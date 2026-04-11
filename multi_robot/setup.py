from setuptools import setup, find_packages
from codecs import open
from os import path


from mmd import __version__


ext_modules = []

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))


setup(name='mmd',
    version=__version__,
    description='Projected Coupled Diffusion (PCD) implementation based on MMD',
    author='Hao Luan',
    author_email='haoluan@comp.nus.edu.sg',
    packages=find_packages(where=''),
    install_requires=requires_list,
)
