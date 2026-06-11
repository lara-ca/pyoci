from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyoci',
    version='0.1.0',
    description='Optimal Control Identification',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
      'numpy>=1.20,<2.0',
      'scipy>=1.7,<1.14',
      'matplotlib>=3.5',
      'pyvrft',
      'pysid',
      'ipython'
    ],
    python_requires=">=3.8",
    author='Lara Colognese de Almeida',
    author_email='colognesealmeida.lara@gmail.com',
)